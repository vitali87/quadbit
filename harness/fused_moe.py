"""Fused-MoE kernel iteration harness (offline, no vLLM serve loop).

The gpt-oss sparse-MoE throughput gap vs stock Marlin is NOT the GEMM math (our 2:4 W4A4 mma has
2-4x Marlin's W4A16 throughput) -- it is the ~40ms of SEPARATE memory-bound torch passes around the
two seg-GEMMs (bf16 intermediate materialization, gather, quant, swiglu 3-pass, combine, index_add).
This harness is the tight loop for building + validating FUSED kernels that collapse that plumbing.

Each fused increment is validated two ways: (a) cos vs the CURRENT unfused seg path must be ~1.0 (the
fusion is value-preserving), and (b) CUDA-event timing must drop. Real gpt-oss experts (layer 11),
prefill-scale routing. Helpers duplicated from gptoss_prep.seg_validate (file convention is duplication)."""

import ctypes
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
MODEL = "openai/gpt-oss-20b"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64", "HF_HOME": "/cache"})
    .pip_install("torch", index_url="https://download.pytorch.org/whl/nightly/cu128", pre=True)
    .pip_install("safetensors", "huggingface_hub", "numpy")
    .add_local_dir((ROOT / "cuda").as_posix(), "/root/cuda")
)
app = modal.App("quadbit-fused-moe", image=image)
vol = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)


@app.function(gpu="RTX-PRO-6000", timeout=3600, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(layer: int = 11, n_experts: int = 32, tokens: int = 4096, topk: int = 4, iters: int = 30) -> None:
    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    so = "/root/sparse_fp4.so"
    c = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
                        "-o", so, "/root/cuda/sparse_fp4_lib.cu", "-lcuda"], capture_output=True, text=True)
    if c.returncode != 0:
        print(c.stderr, flush=True); raise SystemExit(1)
    lib = ctypes.CDLL(so)
    lib.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    lib.sparse_moe_mm_2lvl.argtypes = ([ctypes.c_void_p] * 6 + [ctypes.c_int] * 4
                                       + [ctypes.c_void_p] * 3 + [ctypes.c_int] + [ctypes.c_void_p])
    lib.sparse_moe_mm_2lvl_scatter.argtypes = ([ctypes.c_void_p] * 5 + [ctypes.c_int] * 4
                                               + [ctypes.c_void_p] * 7 + [ctypes.c_int] + [ctypes.c_void_p])
    lib.sparse_moe_gu_swiglu.argtypes = ([ctypes.c_void_p] * 5 + [ctypes.c_int] * 4
                                         + [ctypes.c_void_p] * 6 + [ctypes.c_int] + [ctypes.c_void_p])
    lib.qb_init_moe_attrs()
    lib.qb_init_gusw_attrs()
    dev = torch.device("cuda")
    _BN = 128

    FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
    BND = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], device=dev)
    _cc = torch.arange(128, device=dev); _e, _m = (_cc >> 3) & 0xf, _cc & 7
    UE4M3 = torch.where(_e == 0, _m.float() * 0.001953125, (1.0 + _m.float() / 8.0) * torch.exp2((_e - 7).float()))

    def q_fp4(v):
        return torch.bucketize(v.abs(), BND) | ((v < 0).long() << 3)

    def enc(s):
        mant_f, e = torch.frexp(s.clamp_min(1e-30)); mm = 2.0 * mant_f
        biased = (e - 1) + 7; mant = torch.round((mm - 1.0) * 8.0).long(); carry = mant == 8
        mant = torch.where(carry, torch.zeros_like(mant), mant); biased = torch.where(carry, biased + 1, biased)
        code = (biased.long() << 3) | mant
        code = torch.where(biased < 1, torch.ones_like(code), code)
        code = torch.where(biased > 15, torch.full_like(code, 0x7f), code)
        code = torch.where(s >= 480.0, torch.full_like(code, 0x7f), code)
        return torch.where(s > 0, code, torch.zeros_like(code))

    def pack(w):  # bf16 [out,in] -> (ac,meta,scale,ga); magnitude 2:4, seg_validate verbatim
        out_f, in_f = w.shape; ks = in_f // 128
        wg = w.float().view(out_f, ks, 16, 4, 2)
        i01, _ = wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
        kept = torch.gather(wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
        ga = (kept.abs().amax(dim=(1, 2, 3, 4), keepdim=True) / 2688.0).clamp_min(1e-30).reshape(out_f, 1, 1)
        blk = kept.reshape(out_f, ks, 4, 8, 2)
        scode = enc((blk.abs().amax(dim=(3, 4)) / 6.0) / ga)
        sdeq = UE4M3[scode] * ga
        kc = q_fp4(blk / sdeq.clamp_min(1e-30)[..., None, None])
        ac = (kc[..., 0] | (kc[..., 1] << 4)).reshape(out_f, ks * 32).to(torch.uint8)
        nib = (i01[..., 0] | (i01[..., 1] << 2)).view(out_f, ks, 2, 8)
        sh = (torch.arange(8, device=dev) * 4).view(1, 1, 1, 8)
        meta = (nib << sh).sum(-1).to(torch.int32).permute(1, 0, 2).contiguous()
        return (ac.contiguous(), meta, scode.to(torch.uint8).permute(1, 0, 2).contiguous(),
                ga.reshape(out_f).float().contiguous())

    def stack(packs):
        return (torch.cat([p[0] for p in packs], 0).contiguous(),
                torch.cat([p[1] for p in packs], 1).contiguous(),
                torch.cat([p[2] for p in packs], 1).contiguous(),
                torch.cat([p[3] for p in packs], 0).contiguous())

    def quant_act(x, in_f):  # bf16 [r,in] -> (bb,sb,gb) two-level nvfp4 (seg_gemm's quant step)
        r = x.shape[0]; ks = in_f // 128
        xb = x.to(torch.bfloat16).contiguous()
        bb = torch.empty((r, in_f // 2), dtype=torch.uint8, device=dev)
        sb = torch.empty((ks, r, 4), dtype=torch.uint8, device=dev)
        gb = torch.empty((r,), dtype=torch.float32, device=dev)
        lib.quantize_act_nvfp4_2lvl(xb.data_ptr(), bb.data_ptr(), sb.data_ptr(), gb.data_ptr(), r, in_f)
        return bb, sb, gb

    def seg_gemm(x, w, mpe, in_f, eblk):  # quant + GEMM (outT=1, token-major) -> [r, mpe]
        r = x.shape[0]; ac, meta, scale_a, ga = w
        bb, sb, gb = quant_act(x, in_f)
        cc = torch.empty((r, mpe), dtype=torch.bfloat16, device=dev)
        lib.sparse_moe_mm_2lvl(ac.data_ptr(), bb.data_ptr(), scale_a.data_ptr(), sb.data_ptr(),
                               meta.data_ptr(), cc.data_ptr(), ac.shape[0], mpe, r, in_f,
                               ga.data_ptr(), gb.data_ptr(), eblk.data_ptr(), 1, 0)
        return cc

    def down_scatter(hp, w, mpe, in_f, eblk, sc_tok, sc_w, bias, out):  # FUSED down: quant + scatter-GEMM
        r = hp.shape[0]; ac, meta, scale_a, ga = w
        bb, sb, gb = quant_act(hp, in_f)
        lib.sparse_moe_mm_2lvl_scatter(ac.data_ptr(), bb.data_ptr(), scale_a.data_ptr(), sb.data_ptr(),
                                       meta.data_ptr(), ac.shape[0], mpe, r, in_f, ga.data_ptr(), gb.data_ptr(),
                                       eblk.data_ptr(), sc_tok.data_ptr(), sc_w.data_ptr(), bias.data_ptr(),
                                       out.data_ptr(), bias.shape[1], 0)

    def build_routing(assign, e):
        order = torch.argsort(assign, stable=True)
        counts = torch.bincount(assign, minlength=e)
        padc = (counts + _BN - 1) // _BN * _BN
        r_pad = int(padc.sum().item())
        src = torch.full((r_pad,), -1, dtype=torch.long, device=dev)
        eblk = torch.zeros(r_pad // _BN, dtype=torch.int32, device=dev)
        off = 0; oi = 0
        for ex in range(e):
            ce = int(counts[ex].item()); pe = int(padc[ex].item())
            if pe == 0:
                continue
            src[off:off + ce] = order[oi:oi + ce]; oi += ce
            eblk[off // _BN:(off + pe) // _BN] = ex; off += pe
        return src, eblk, r_pad

    def mxfp4_dequant(blocks_u8, scales_u8):
        b = blocks_u8.to(dev).to(torch.int32) & 0xFF; lead = b.shape[:-1]
        codes = torch.stack([b & 0xF, (b >> 4) & 0xF], dim=-1).reshape(*lead[:-1], lead[-1] * 32).long()
        vals = FP4[codes]; nb = scales_u8.shape[-1]
        mult = torch.exp2(scales_u8.to(dev).float() - 127.0)
        return (vals.reshape(*vals.shape[:-1], nb, 32) * mult[..., None]).reshape(vals.shape).to(torch.bfloat16)

    def pad_w(W):
        out_f, in_f = W.shape
        outp = ((out_f + 255) // 256) * 256; inp = ((in_f + 127) // 128) * 128
        Wp = W.new_zeros(outp, inp); Wp[:out_f, :in_f] = W
        return Wp

    def swiglu(gate, up):
        gate = gate.clamp(max=7.0); up = up.clamp(-7.0, 7.0)
        return (gate * torch.sigmoid(1.702 * gate)) * (up + 1.0)

    idx_path = hf_hub_download(MODEL, "model.safetensors.index.json")
    weight_map = __import__("json").load(open(idx_path))["weight_map"]

    def get(name):
        p = hf_hub_download(MODEL, weight_map[name])
        with safe_open(p, framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    print(f"# FUSED-MoE harness ({MODEL}) layer {layer} E={n_experts} T={tokens} topk={topk}", flush=True)
    guB, guS = get(f"model.layers.{layer}.mlp.experts.gate_up_proj_blocks"), get(f"model.layers.{layer}.mlp.experts.gate_up_proj_scales")
    guBias = get(f"model.layers.{layer}.mlp.experts.gate_up_proj_bias").to(dev).float()
    dnB, dnS = get(f"model.layers.{layer}.mlp.experts.down_proj_blocks"), get(f"model.layers.{layer}.mlp.experts.down_proj_scales")
    dnBias = get(f"model.layers.{layer}.mlp.experts.down_proj_bias").to(dev).float()

    gW, uW, dW, gB_, uB_, dB_, gp, up_, dp = [], [], [], [], [], [], [], [], []
    gfu, gfu_bias = [], []   # native pairwise-interleaved gate_up (no de-interleave) for fused GEMM1
    for e in range(n_experts):
        gu = mxfp4_dequant(guB[e], guS[e]).float(); dn = mxfp4_dequant(dnB[e], dnS[e]).float()
        g, u = gu[0::2, :], gu[1::2, :]
        gW.append(g); uW.append(u); dW.append(dn)
        gB_.append(guBias[e][0::2]); uB_.append(guBias[e][1::2]); dB_.append(dnBias[e])
        gp.append(pack(pad_w(g))); up_.append(pack(pad_w(u))); dp.append(pack(pad_w(dn)))
        gfu.append(pack(pad_w(gu)))                                 # [2I,H]->pad[mpe_gu,Hp]
        gfu_bias.append(F.pad(guBias[e], (0, pad_w(gu).shape[0] - gu.shape[0])))
    I, H = gW[0].shape[0], gW[0].shape[1]
    Ip, Hp = pad_w(dW[0]).shape[1], pad_w(gW[0]).shape[1]
    mpeI, mpeH = pad_w(gW[0]).shape[0], pad_w(dW[0]).shape[0]
    mpe_gu = pad_w(mxfp4_dequant(guB[0], guS[0]).float()).shape[0]  # 5888
    Ih = mpe_gu // 2                                                # h-dim count = down in_f = Ip
    gS, uS_, dS = stack(gp), stack(up_), stack(dp)
    guS_fused = stack(gfu)
    guBias_fused = torch.stack(gfu_bias).to(torch.bfloat16).contiguous()  # [E, mpe_gu]
    gBias_s = torch.stack(gB_); uBias_s = torch.stack(uB_)
    dBias_s = torch.stack(dB_).to(torch.bfloat16).contiguous()  # [E,H] for the fused epilogue

    torch.manual_seed(layer)
    x = torch.randn(tokens, H, device=dev, dtype=torch.bfloat16) * 0.1
    topk_ids = torch.stack([torch.randperm(n_experts, device=dev)[:topk] for _ in range(tokens)])
    topk_w = torch.rand(tokens, topk, device=dev); topk_w = topk_w / topk_w.sum(1, keepdim=True)

    assign = topk_ids.reshape(-1).to(torch.long)
    tok_of = torch.arange(tokens, device=dev).repeat_interleave(topk)
    w_of = topk_w.reshape(-1).float()
    src, eblk, r_pad = build_routing(assign, n_experts)
    valid = src >= 0; srcc = src.clamp_min(0)
    row_exp = assign[srcc]
    xs = x[tok_of[srcc]].to(torch.bfloat16) * valid[:, None]
    xsp = torch.zeros(r_pad, Hp, device=dev, dtype=torch.bfloat16); xsp[:, :H] = xs
    # gate/up + swiglu shared by both paths (increment 2 fuses only the DOWN side)
    g = seg_gemm(xsp, gS, mpeI, Hp, eblk)[:, :I].float() + gBias_s[row_exp]
    u = seg_gemm(xsp, uS_, mpeI, Hp, eblk)[:, :I].float() + uBias_s[row_exp]
    h = swiglu(g, u)
    hp = torch.zeros(r_pad, Ip, device=dev, dtype=torch.bfloat16); hp[:, :I] = h.to(torch.bfloat16)
    sc_tok = torch.where(valid, tok_of[srcc], torch.full_like(tok_of[srcc], -1)).to(torch.int32).contiguous()
    sc_w = (w_of[srcc] * valid.float()).contiguous()

    # ---- UNFUSED down (current path): seg-GEMM -> +bias -> *w -> index_add ----
    def unfused_down():
        d = seg_gemm(hp, dS, mpeH, Ip, eblk)[:, :H].float() + torch.stack(dB_)[row_exp]
        out = torch.zeros(tokens, H, device=dev)
        out.index_add_(0, sc_tok.long().clamp_min(0), d * (sc_w * valid.float())[:, None])
        return out

    # ---- FUSED down (increment 2): quant -> scatter-GEMM epilogue ----
    def fused_down():
        out = torch.zeros(tokens, H, device=dev)
        down_scatter(hp, dS, mpeH, Ip, eblk, sc_tok, sc_w, dBias_s, out)
        return out

    ref = unfused_down()
    fus = fused_down()
    cos = F.cosine_similarity(ref.flatten(), fus.flatten(), dim=0).item()
    rel = (ref - fus).abs().max().item() / (ref.abs().max().item() + 1e-9)
    print(f"# CORRECTNESS fused-down vs unfused-down: cos={cos:.6f} max_rel_err={rel:.3e} "
          f"{'OK' if cos > 0.9999 else 'MISMATCH'}", flush=True)

    def bench(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(iters):
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        return ts[len(ts) // 2]

    t_un = bench(unfused_down)
    t_fu = bench(fused_down)
    print(f"# TIMING down-side  unfused={t_un:.3f}ms  fused={t_fu:.3f}ms  "
          f"speedup={t_un / t_fu:.2f}x  (r_pad={r_pad} rows)", flush=True)

    # ================= INCREMENT 3: fused GEMM1 (gate_up -> swiglu -> MXFP4 quant) =================
    def deq_h(Hw, sH):  # single-level FP4 h -> float [r_pad, Ih] (gB=1)
        r = Hw.shape[0]; b = Hw.to(torch.long)
        vals = FP4[torch.stack([b & 0xF, (b >> 4) & 0xF], dim=-1).reshape(r, Ih)]
        nblk = Ih // 32; bl = torch.arange(nblk, device=dev)
        scale = UE4M3[sH[bl // 4, :, bl % 4].long()].transpose(0, 1)  # [r, nblk]
        return (vals.reshape(r, nblk, 32) * scale[..., None]).reshape(r, Ih)

    def gemm1_fused():
        bb, sb, gb = quant_act(xsp, Hp)
        Hw = torch.empty((r_pad, Ih // 2), dtype=torch.uint8, device=dev)
        sH = torch.empty((Ih // 128, r_pad, 4), dtype=torch.uint8, device=dev)
        ac, meta, sA, ga = guS_fused
        lib.sparse_moe_gu_swiglu(ac.data_ptr(), bb.data_ptr(), sA.data_ptr(), sb.data_ptr(), meta.data_ptr(),
                                 ac.shape[0], mpe_gu, r_pad, Hp, ga.data_ptr(), gb.data_ptr(), eblk.data_ptr(),
                                 guBias_fused.data_ptr(), Hw.data_ptr(), sH.data_ptr(), Ih, 0)
        return Hw, sH

    def gemm1_unfused():  # current path: 2 seg-GEMMs + torch swiglu + quant
        gg = seg_gemm(xsp, gS, mpeI, Hp, eblk)[:, :I].float() + gBias_s[row_exp]
        uu = seg_gemm(xsp, uS_, mpeI, Hp, eblk)[:, :I].float() + uBias_s[row_exp]
        h_ = swiglu(gg, uu)
        hp_ = torch.zeros(r_pad, Ip, device=dev, dtype=torch.bfloat16); hp_[:, :I] = h_.to(torch.bfloat16)
        return quant_act(hp_, Ip)

    Hw, sH = gemm1_fused()
    h_fused = deq_h(Hw, sH)[:, :I]
    c_h = F.cosine_similarity(h_fused.flatten(), h.flatten(), dim=0).item()  # vs reference swiglu h
    print(f"# CORRECTNESS fused-GEMM1 h vs reference swiglu-h: cos={c_h:.5f} "
          f"{'OK' if c_h > 0.98 else 'LAYOUT-BUG' if c_h < 0.5 else 'CHECK'}  (single-level FP4 quant tax)", flush=True)

    def fused_full():
        Hw, sH = gemm1_fused()
        out = torch.zeros(tokens, H, device=dev)
        ac, meta, sA, ga = dS
        lib.sparse_moe_mm_2lvl_scatter(ac.data_ptr(), Hw.data_ptr(), sA.data_ptr(), sH.data_ptr(), meta.data_ptr(),
                                       ac.shape[0], mpeH, r_pad, Ip, ga.data_ptr(), 0, eblk.data_ptr(),
                                       sc_tok.data_ptr(), sc_w.data_ptr(), dBias_s.data_ptr(), out.data_ptr(), H, 0)
        return out

    ff = fused_full()
    c_full = F.cosine_similarity(ff.flatten(), ref.flatten(), dim=0).item()  # end-to-end vs 2-level unfused
    print(f"# CORRECTNESS full-fused MoE vs unfused MoE (end-to-end): cos={c_full:.5f} "
          f"{'OK' if c_full > 0.99 else 'CHECK'}", flush=True)

    t_g1un = bench(gemm1_unfused)
    t_g1fu = bench(gemm1_fused)
    print(f"# TIMING GEMM1 (gate_up+swiglu+quant)  unfused={t_g1un:.3f}ms  fused={t_g1fu:.3f}ms  "
          f"speedup={t_g1un / t_g1fu:.2f}x", flush=True)
    # whole-MoE: unfused = gate_up-unfused + down-unfused ; fused = gemm1_fused + down-scatter
    t_moe_un = bench(lambda: (gemm1_unfused(), unfused_down()))
    t_moe_fu = bench(fused_full)
    print(f"# TIMING whole-MoE  unfused={t_moe_un:.3f}ms  fused={t_moe_fu:.3f}ms  "
          f"speedup={t_moe_un / t_moe_fu:.2f}x  (T={tokens} topk={topk}, r_pad={r_pad})", flush=True)
    print("# fused_moe done", flush=True)
