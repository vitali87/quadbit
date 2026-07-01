"""The kernel on REAL open-weight models: full fused FP4 transformer block on actual
Qwen3-8B weights (small, deploys on one RTX PRO 6000), + shape/speed compat for Qwen3-235B-A22B
(big MoE, deploys via expert/tensor/pipeline sharding). Proves the kernel is the go-to for
current open models irrespective of size: every linear (attn q/k/v/o, FFN gate/up/down, MoE
experts) satisfies the tile constraints (out%256, in%256), and a whole decoder block runs
through the fused kernels (fused RMSNorm+quant -> QKV -> attn(bf16) -> o-proj -> fused
add+RMSNorm+quant -> fused SwiGLU FFN) end-to-end vs a bf16 reference block.

Run:  uv run modal run harness/real_model.py
"""

import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64"})
    .pip_install("torch", index_url="https://download.pytorch.org/whl/nightly/cu128", pre=True)
    .pip_install("transformers", "safetensors", "huggingface-hub", "accelerate")
    .add_local_dir((ROOT / "cuda").as_posix(), "/root/cuda")
)
app = modal.App("quadbit-realmodel", image=image)
cache = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)

SMALL = "Qwen/Qwen3-8B"


@app.function(gpu="RTX-PRO-6000", timeout=3600, volumes={"/cache": cache})
def run() -> None:
    so = "/root/sparse_fp4.so"
    c = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
                        "-o", so, "/root/cuda/sparse_fp4_lib.cu", "-lcuda"], capture_output=True, text=True)
    if c.returncode != 0:
        print(c.stderr, flush=True); return
    so2 = "/root/dense_scaled_fast.so"
    c2 = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
                         "-o", so2, "/root/cuda/dense_scaled_fast_lib.cu", "-lcuda"], capture_output=True, text=True)
    if c2.returncode != 0:
        print(c2.stderr, flush=True); return

    import ctypes
    import os

    import torch
    import torch.nn.functional as F
    os.environ["HF_HOME"] = "/cache/hf"

    lib = ctypes.CDLL(so)
    lib.sparse_fp4_mm.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3
    lib.quantize_act_nvfp4.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2
    lib.quantize_act_nvfp4.restype = None
    lib.fused_swiglu_quant.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    lib.fused_swiglu_quant.restype = None
    lib.rmsnorm_quant.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2 + [ctypes.c_float]
    lib.rmsnorm_quant.restype = None
    lib.add_rmsnorm_quant.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 2 + [ctypes.c_float]
    lib.add_rmsnorm_quant.restype = None
    dlib = ctypes.CDLL(so2)
    dlib.dense_scaled_fast_mm.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3
    dlib.quantize_act_mxfp4.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2
    dlib.quantize_act_mxfp4.restype = None
    dev = torch.device("cuda")

    FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
                       dtype=torch.float32, device=dev)
    BND = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], dtype=torch.float32, device=dev)
    _c = torch.arange(128, device=dev)
    _e, _m = (_c >> 3) & 0xf, _c & 7
    UE4M3 = torch.where(_e == 0, _m.float() * 0.001953125,
                        (1.0 + _m.float() / 8.0) * torch.exp2((_e - 7).float()))
    _MID = (UE4M3[:-1] + UE4M3[1:]) / 2

    def quant_fp4(v):
        return torch.bucketize(v.abs(), BND).to(torch.uint8) | ((v < 0).to(torch.uint8) << 3)

    def enc_ue4m3(s):
        return torch.bucketize(s, _MID).to(torch.uint8)

    class QuadbitLinear:  # 2:4-sparse FP4 weight, real ue4m3 scales (see quadbit_linear.py)
        def __init__(self, W):
            out_f, in_f = W.shape
            assert out_f % 256 == 0 and in_f % 256 == 0, f"{out_f}x{in_f} not %256"
            ks = in_f // 128
            Wg = W.float().view(out_f, ks, 16, 4, 2)
            pmag = Wg.abs().sum(-1)
            top2 = pmag.topk(2, dim=-1).indices
            i01, _ = top2.sort(dim=-1)
            i0, i1 = i01[..., 0], i01[..., 1]
            kept = torch.stack([i0, i1], dim=-1)
            keptW = torch.gather(Wg, 3, kept.unsqueeze(-1).expand(-1, -1, -1, -1, 2).long())
            blk = keptW.reshape(out_f, ks, 4, 8, 2)
            scode = enc_ue4m3(blk.abs().amax(dim=(3, 4)) / 6.0)
            sdeq = UE4M3[scode.long()]
            kc = quant_fp4(blk / sdeq[..., None, None])
            Abytes = (kc[..., 0] | (kc[..., 1] << 4)).reshape(out_f, ks * 32).contiguous()
            nib = (i0 | (i1 << 2)).view(out_f, ks, 2, 8).long()
            sh = (torch.arange(8, device=dev) * 4).view(1, 1, 1, 8)
            meta = (nib << sh).sum(-1).to(torch.int32).permute(1, 0, 2).contiguous()
            self.out_f, self.in_f, self.ks = out_f, in_f, ks
            self.Ac = Abytes.to(torch.uint8)
            self.meta = meta
            self.scaleA = scode.to(torch.uint8).permute(1, 0, 2).contiguous()

        def run(self, Bbytes, scaleB, batch):
            C = torch.empty((self.out_f, batch), dtype=torch.bfloat16, device=dev)
            lib.sparse_fp4_mm(self.Ac.data_ptr(), Bbytes.data_ptr(), self.scaleA.data_ptr(),
                              scaleB.data_ptr(), self.meta.data_ptr(), C.data_ptr(),
                              self.out_f, batch, self.in_f)
            return C  # [out, batch]

    def quant_act(x):  # x[batch,in] bf16 -> (Bbytes, scaleB) via fused quantizer
        batch, in_f = x.shape
        Bb = torch.empty((batch, in_f // 2), dtype=torch.uint8, device=dev)
        sB = torch.empty((in_f // 128, batch, 4), dtype=torch.uint8, device=dev)
        lib.quantize_act_nvfp4(x.contiguous().data_ptr(), Bb.data_ptr(), sB.data_ptr(), batch, in_f)
        return Bb, sB

    from transformers import AutoConfig, AutoModelForCausalLM

    print("loading real Qwen3-8B weights ...", flush=True)
    cfg = AutoConfig.from_pretrained(SMALL)
    model = AutoModelForCausalLM.from_pretrained(SMALL, torch_dtype=torch.bfloat16, cache_dir="/cache/hf")
    lyr = model.model.layers[0]
    H, I = cfg.hidden_size, cfg.intermediate_size
    nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    eps = cfg.rms_norm_eps
    print(f"Qwen3-8B: H={H} I={I} heads={nh} kv={nkv} hd={hd} layers={cfg.num_hidden_layers}", flush=True)

    # real weights [out,in]
    Wq = lyr.self_attn.q_proj.weight.data.to(dev)
    Wk = lyr.self_attn.k_proj.weight.data.to(dev)
    Wv = lyr.self_attn.v_proj.weight.data.to(dev)
    Wo = lyr.self_attn.o_proj.weight.data.to(dev)
    Wg = lyr.mlp.gate_proj.weight.data.to(dev)
    Wu = lyr.mlp.up_proj.weight.data.to(dev)
    Wd = lyr.mlp.down_proj.weight.data.to(dev)
    ln1 = lyr.input_layernorm.weight.data.to(dev).bfloat16()
    ln2 = lyr.post_attention_layernorm.weight.data.to(dev).bfloat16()

    qdim, kvdim = nh * hd, nkv * hd
    q_qkv = QuadbitLinear(torch.cat([Wq, Wk, Wv], 0))    # concat QKV: one FP4 GEMM
    q_o = QuadbitLinear(Wo)
    q_gu = QuadbitLinear(torch.cat([Wg, Wu], 0))         # concat gate+up
    q_d = QuadbitLinear(Wd)

    def rmsnorm(x, w):
        v = x.float()
        return (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps) * w.float()).bfloat16()

    def attn(q, k, v, B):  # q[B,qdim] k,v[B,kvdim] -> [B,qdim] bf16 (GQA sdpa, causal)
        q = q.view(B, nh, hd).transpose(0, 1)
        k = k.view(B, nkv, hd).transpose(0, 1)
        v = v.view(B, nkv, hd).transpose(0, 1)
        rep = nh // nkv
        k = k.repeat_interleave(rep, 0); v = v.repeat_interleave(rep, 0)
        o = F.scaled_dot_product_attention(q[None], k[None], v[None], is_causal=True)[0]
        return o.transpose(0, 1).reshape(B, qdim)

    def fp4_block(x):  # x[B,H] bf16 -> [B,H]; fully fused FP4 linears + bf16 attention
        B = x.shape[0]
        # input RMSNorm + quant (fused) -> QKV GEMM
        Bb = torch.empty((B, H // 2), dtype=torch.uint8, device=dev)
        sB = torch.empty((H // 128, B, 4), dtype=torch.uint8, device=dev)
        lib.rmsnorm_quant(x.data_ptr(), ln1.data_ptr(), Bb.data_ptr(), sB.data_ptr(), B, H, eps)
        qkv = q_qkv.run(Bb, sB, B).t()                    # [B, qdim+2kvdim]
        q, k, v = qkv[:, :qdim], qkv[:, qdim:qdim + kvdim], qkv[:, qdim + kvdim:]
        a = attn(q.contiguous(), k.contiguous(), v.contiguous(), B)
        ob, os = quant_act(a)
        o = q_o.run(ob, os, B).t()                        # [B,H]
        # residual add + post RMSNorm + quant (fused) -> gate/up GEMM
        Hb = torch.empty((B, H // 2), dtype=torch.uint8, device=dev)
        Hs = torch.empty((H // 128, B, 4), dtype=torch.uint8, device=dev)
        resid = torch.empty((B, H), dtype=torch.bfloat16, device=dev)
        lib.add_rmsnorm_quant(o.data_ptr(), x.data_ptr(), ln2.data_ptr(), resid.data_ptr(),
                              Hb.data_ptr(), Hs.data_ptr(), B, H, eps)
        gu = q_gu.run(Hb, Hs, B)                           # [2I, B]
        g_ptr, u_ptr = gu.data_ptr(), gu.data_ptr() + I * B * 2
        Db = torch.empty((B, I // 2), dtype=torch.uint8, device=dev)
        Ds = torch.empty((I // 128, B, 4), dtype=torch.uint8, device=dev)
        lib.fused_swiglu_quant(g_ptr, u_ptr, Db.data_ptr(), Ds.data_ptr(), B, I)
        ffn = q_d.run(Db, Ds, B).t()                       # [B,H]
        return resid + ffn

    Wqb, Wkb, Wvb, Wob = Wq.bfloat16(), Wk.bfloat16(), Wv.bfloat16(), Wo.bfloat16()
    Wgb, Wub, Wdb = Wg.bfloat16(), Wu.bfloat16(), Wd.bfloat16()

    def bf16_block(x):  # identical structure, real bf16 tensor-core matmuls (fair baseline)
        B = x.shape[0]
        h = rmsnorm(x, ln1)
        q, k, v = h @ Wqb.t(), h @ Wkb.t(), h @ Wvb.t()
        a = attn(q, k, v, B)
        o = a @ Wob.t()
        x2 = x + o
        h2 = rmsnorm(x2, ln2)
        ffn = ((F.silu((h2 @ Wgb.t()).float()) * (h2 @ Wub.t()).float()).bfloat16()) @ Wdb.t()
        return x2 + ffn

    def nvfp4_dq(W, blk=16):  # dense NVFP4 quant->dequant (no prune) -- deployable dense-FP4 sim
        out, inn = W.shape
        Wb = W.float().view(out, inn // blk, blk)
        sdeq = UE4M3[enc_ue4m3(Wb.abs().amax(-1, keepdim=True) / 6.0).long()]
        return (FP4[quant_fp4(Wb / sdeq).long()] * sdeq).view(out, inn).bfloat16()

    def act_dq(a, blk=16):
        B, inn = a.shape
        ab = a.float().view(B, inn // blk, blk)
        sdeq = UE4M3[enc_ue4m3(ab.abs().amax(-1, keepdim=True) / 6.0).long()]
        return (FP4[quant_fp4(ab / sdeq).long()] * sdeq).view(B, inn).bfloat16()

    Wq4, Wk4, Wv4, Wo4 = nvfp4_dq(Wq), nvfp4_dq(Wk), nvfp4_dq(Wv), nvfp4_dq(Wo)
    Wg4, Wu4, Wd4 = nvfp4_dq(Wg), nvfp4_dq(Wu), nvfp4_dq(Wd)

    def dense_fp4_block(x):  # DENSE FP4 (no prune) W4A4, math in bf16 -- the deployable accuracy path
        B = x.shape[0]
        h = act_dq(rmsnorm(x, ln1))
        q, k, v = h @ Wq4.t(), h @ Wk4.t(), h @ Wv4.t()
        a = attn(q, k, v, B)
        o = act_dq(a) @ Wo4.t()
        x2 = x + o
        h2 = act_dq(rmsnorm(x2, ln2))
        ffn = act_dq((F.silu((h2 @ Wg4.t()).float()) * (h2 @ Wu4.t()).float()).bfloat16()) @ Wd4.t()
        return x2 + ffn

    def tms(fn, it=20):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(it):
            fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / it

    print("\n# Full fused FP4 decoder block on REAL Qwen3-8B weights vs bf16 (tensor-core) block", flush=True)
    for B in (512, 2048):
        x = torch.randn(B, H, device=dev).bfloat16() * 0.1
        yq, yb, yd = fp4_block(x), bf16_block(x), dense_fp4_block(x)
        rel_s = ((yq.float() - yb.float()).norm() / yb.float().norm()).item()
        rel_d = ((yd.float() - yb.float()).norm() / yb.float().norm()).item()
        msq, msb = tms(lambda: fp4_block(x)), tms(lambda: bf16_block(x))
        print(f"  tokens={B}: sparse-FP4-fused {msq:.3f}ms  bf16 {msb:.3f}ms -> {msb/msq:.2f}x | "
              f"block rel: sparse {rel_s:.3f} (2:4 needs recovery) / dense-FP4 {rel_d:.3f} (deployable, no train)",
              flush=True)

    # BIG frontier open-weight models (July 2026), deployed via expert/tensor/pipeline sharding.
    # Real configs (from each repo's config.json). Each shard's linears run on one GPU = the kernel.
    # H=hidden, I=moe_intermediate(per expert), nh/nkv/hd = attn heads / kv heads / head_dim.
    print("\n# Verified 2026-frontier BIG models: every real linear shape through the kernel", flush=True)
    models = [  # (name, H, I, nh, nkv, hd)
        ("Qwen3.5-397B (Qwen3Moe)", 4096, 1536, 64, 4, 128),
        ("GLM-5.2 (Glm4Moe)",       5120, 1536, 96, 8, 128),
        ("MiniMax-M3",              6144, 12288, 64, 4, 128),
        ("DeepSeek-V3/R1 (671B)",   7168, 2048, 128, 128, 128),
    ]
    B = 256
    for name, H2, I2, nh2, nkv2, hd2 in models:
        qkv = nh2 * hd2 + 2 * nkv2 * hd2
        shapes = [("qkv", qkv, H2), ("o", H2, nh2 * hd2), ("expert gate+up", 2 * I2, H2), ("expert down", H2, I2)]
        allok = all(o % 256 == 0 and i % 256 == 0 for _, o, i in shapes)
        parts = []
        for sn, out_f, in_f in shapes:
            if out_f % 256 or in_f % 256:
                parts.append(f"{sn}={out_f}x{in_f}:TILE-FAIL"); continue
            ql = QuadbitLinear(torch.randn(out_f, in_f, device=dev) * 0.02)
            Bb, sB = quant_act(torch.randn(B, in_f, device=dev).bfloat16())
            ms = tms(lambda: ql.run(Bb, sB, B))
            parts.append(f"{sn} {ms*1e3:.0f}us")
        print(f"  {name:26s} {'ALL-TILE-OK' if allok else 'HAS-FAIL'} | " + "  ".join(parts), flush=True)

    # DEPLOYABLE dense real-scale FP4 THROUGH THE KERNEL (MXFP4, no training) on REAL Qwen3-8B linears
    print("\n# Deployable dense FP4 (real ue8m0 scales) THROUGH THE KERNEL on real Qwen3-8B weights", flush=True)

    def mxfp4_pack(W):  # W[out,in] -> Abytes[out,in/2], SFA[out,in/32] ue8m0 (matches kernel layout)
        out, inn = W.shape
        Wb = W.float().view(out, inn // 32, 32)
        _, e = torch.frexp(Wb.abs().amax(-1) / 6.0)          # amax/6 = m*2^e -> scale 2^e
        code = (e + 127).clamp(0, 255)
        scale = torch.ldexp(torch.ones_like(code, dtype=torch.float32), code - 127)
        q = quant_fp4(Wb / scale[..., None]).view(out, inn)
        Ab = (q[:, 0::2] | (q[:, 1::2] << 4)).to(torch.uint8).contiguous()
        return Ab, code.to(torch.uint8).contiguous()

    class DenseMX:
        def __init__(self, W):
            self.out, self.inn = W.shape
            self.Ab, self.SFA = mxfp4_pack(W.to(dev))

        def run(self, Bb, SFB, batch):
            C = torch.empty((self.out, batch), dtype=torch.bfloat16, device=dev)
            dlib.dense_scaled_fast_mm(self.Ab.data_ptr(), Bb.data_ptr(), self.SFA.data_ptr(),
                                      SFB.data_ptr(), C.data_ptr(), self.out, batch, self.inn)
            return C

    def quant_act_mx(a):
        B, inn = a.shape
        Bb = torch.empty((B, inn // 2), dtype=torch.uint8, device=dev)
        SFB = torch.empty((B, inn // 32), dtype=torch.uint8, device=dev)
        dlib.quantize_act_mxfp4(a.contiguous().data_ptr(), Bb.data_ptr(), SFB.data_ptr(), B, inn)
        return Bb, SFB

    B = 2048
    for nm, W in [("q_proj", Wq), ("o_proj", Wo), ("gate_proj", Wg), ("down_proj", Wd)]:
        d = DenseMX(W)
        x = (torch.randn(B, d.inn, device=dev) * 0.1).bfloat16()
        Bb, SFB = quant_act_mx(x)
        y = d.run(Bb, SFB, B).t()                              # [B, out] through kernel
        ref = (x.float() @ W.to(dev).t().float())              # true bf16-input matmul
        rel = ((y.float() - ref).norm() / ref.norm()).item()
        ms = tms(lambda: d.run(Bb, SFB, B))
        msb = tms(lambda: (x @ W.to(dev).bfloat16().t()))
        print(f"  {nm:10s} out={d.out:5d} in={d.inn:5d}: dense-FP4-KERNEL rel {rel:.3f} "
              f"(no train)  {ms*1e3:.0f}us  bf16 {msb*1e3:.0f}us  {msb/ms:.2f}x", flush=True)


@app.local_entrypoint()
def main() -> None:
    run.remote()
