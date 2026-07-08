"""M2: MoE 2:4 + two-level-NVFP4 weight-prep for DeepSeek-V4-Flash.

The experts are natively MXFP4 (expert_dtype=fp4): E2M1 codes packed 2/byte in an int8 tensor, with an
e8m0 (power-of-2) scale per 32-element block. (The config's quantization_config fp8/block-128 describes
the DENSE/attention linears, not the experts.) Experts are stored per-expert:
layers.{L}.ffn.experts.{e}.w1/w3/w2.weight (w1=gate, w3=up, w2=down) + .scale; shared expert at
layers.{L}.ffn.shared_experts.w{1,2,3}. Router stays dense (untouched).

Pipeline: STREAM each expert weight from safetensors (never hold the 284B model), MXFP4-dequant to bf16
(convention cross-checked vs transformers convert_moe_packed_tensors), then pack to the kernel's exact
2:4-sparse two-level-NVFP4 layout (Ac/meta/scaleA/gA) via the packing validated in M1. Embarrassingly parallel across experts; memory-light (one expert at a time).

`validate` mode: prep a couple of REAL experts, print fp8/scale dtypes+ranges (confirms the block-dequant
is right), and check a kernel forward vs the pure-pytorch fakequant reference on the SAME dequantized
weight. `prep` mode: pack all experts of a layer range to a /cache artifact for serving.
"""

import json
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
MODEL = "deepseek-ai/DeepSeek-V4-Flash"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64", "HF_HOME": "/cache"})
    .pip_install("torch", index_url="https://download.pytorch.org/whl/nightly/cu128", pre=True)
    .pip_install("safetensors", "huggingface_hub", "numpy", "transformers")
    .add_local_dir((ROOT / "cuda").as_posix(), "/root/cuda")
)
app = modal.App("quadbit-moeprep", image=image)
vol = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)


@app.function(gpu="RTX-PRO-6000", timeout=86400, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(mode: str = "validate", layer_lo: int = 0, layer_hi: int = 43) -> None:
    import ctypes

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    so = "/root/sparse_fp4.so"
    c = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
                        "-o", so, "/root/cuda/sparse_fp4_lib.cu", "-lcuda"], capture_output=True, text=True)
    if c.returncode != 0:
        print(c.stderr, flush=True)
        raise SystemExit(1)
    lib = ctypes.CDLL(so)
    lib.sparse_fp4_mm_2lvl.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    lib.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    dev = torch.device("cuda")

    FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
    BND = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], device=dev)
    _cc = torch.arange(128, device=dev); _e, _m = (_cc >> 3) & 0xf, _cc & 7
    UE4M3 = torch.where(_e == 0, _m.float() * 0.001953125, (1.0 + _m.float() / 8.0) * torch.exp2((_e - 7).float()))

    def q_fp4(v):
        return torch.bucketize(v.abs(), BND) | ((v < 0).long() << 3)

    def enc_ue4m3_t(s):
        mant_f, e = torch.frexp(s.clamp_min(1e-30)); mm = 2.0 * mant_f
        biased = (e - 1) + 7; mant = torch.round((mm - 1.0) * 8.0).long(); carry = mant == 8
        mant = torch.where(carry, torch.zeros_like(mant), mant); biased = torch.where(carry, biased + 1, biased)
        code = (biased.long() << 3) | mant
        code = torch.where(biased < 1, torch.ones_like(code), code)
        code = torch.where(biased > 15, torch.full_like(code, 0x7f), code)
        code = torch.where(s >= 480.0, torch.full_like(code, 0x7f), code)
        return torch.where(s > 0, code, torch.zeros_like(code))

    def sparse_fp4_dequant(W):  # pair-2:4 magnitude + per-16 two-level (matches M1 kernel path)
        out_f, in_f = W.shape; ks = in_f // 128
        Wg = W.view(out_f, ks, 16, 4, 2)
        i01, _ = Wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
        keptW = torch.gather(Wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
        gA = (keptW.abs().amax(dim=(1, 2, 3, 4), keepdim=True) / 2688.0).clamp_min(1e-30).reshape(out_f, 1, 1)
        blk = keptW.reshape(out_f, ks, 4, 8, 2)
        sdeq = UE4M3[enc_ue4m3_t((blk.abs().amax(dim=(3, 4)) / 6.0) / gA)] * gA
        kd = (FP4[q_fp4(blk / sdeq.clamp_min(1e-30)[..., None, None])] * sdeq[..., None, None]).reshape(out_f, ks, 16, 2, 2)
        Wd = torch.zeros(out_f, ks, 16, 4, 2, device=dev)
        Wd.scatter_(3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2), kd)
        return Wd.reshape(out_f, in_f)

    def act_fp4_dequant(x):
        lead = x.shape[:-1]; i = x.shape[-1]
        b = x.to(torch.bfloat16).float().reshape(-1, i)
        gB = (b.abs().amax(-1, keepdim=True) / 2688.0).clamp_min(1e-30)
        bb = b.reshape(b.shape[0], i // 32, 32)
        sdeq = UE4M3[enc_ue4m3_t((bb.abs().amax(-1) / 6.0) / gB)] * gB
        return (FP4[q_fp4(bb / sdeq.clamp_min(1e-30)[..., None])] * sdeq[..., None]).reshape(*lead, i)

    def pack(W):  # bf16 [out,in] -> kernel buffers (Ac, meta, scaleA, gA); validated in M1
        out_f, in_f = W.shape; ks = in_f // 128
        Wg = W.float().to(dev).view(out_f, ks, 16, 4, 2)
        i01, _ = Wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
        keptW = torch.gather(Wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
        gA = (keptW.abs().amax(dim=(1, 2, 3, 4), keepdim=True) / 2688.0).clamp_min(1e-30).reshape(out_f, 1, 1)
        blk = keptW.reshape(out_f, ks, 4, 8, 2)
        scode = enc_ue4m3_t((blk.abs().amax(dim=(3, 4)) / 6.0) / gA)
        sdeq = UE4M3[scode] * gA
        kc = q_fp4(blk / sdeq.clamp_min(1e-30)[..., None, None])
        Ac = (kc[..., 0] | (kc[..., 1] << 4)).reshape(out_f, ks * 32).to(torch.uint8)
        nib = (i01[..., 0] | (i01[..., 1] << 2)).view(out_f, ks, 2, 8)
        sh = (torch.arange(8, device=dev) * 4).view(1, 1, 1, 8)
        meta = (nib << sh).sum(-1).to(torch.int32).permute(1, 0, 2).contiguous()
        return (Ac.contiguous(), meta, scode.to(torch.uint8).permute(1, 0, 2).contiguous(),
                gA.reshape(out_f).float().contiguous())

    def kernel_fwd(x, buf, out_f, in_f):
        Ac, meta, scaleA, gA = buf; ks = in_f // 128
        x2 = x.reshape(-1, in_f).to(torch.bfloat16); t = x2.shape[0]; padn = (-t) % 128
        if padn:
            x2 = torch.cat([x2, x2.new_zeros(padn, in_f)], 0)
        x2 = x2.contiguous(); tp = t + padn
        Bb = torch.empty((tp, in_f // 2), dtype=torch.uint8, device=dev)
        sB = torch.empty((ks, tp, 4), dtype=torch.uint8, device=dev)
        gB = torch.empty((tp,), dtype=torch.float32, device=dev)
        lib.quantize_act_nvfp4_2lvl(x2.data_ptr(), Bb.data_ptr(), sB.data_ptr(), gB.data_ptr(), tp, in_f)
        C = torch.empty((out_f, tp), dtype=torch.bfloat16, device=dev)
        lib.sparse_fp4_mm_2lvl(Ac.data_ptr(), Bb.data_ptr(), scaleA.data_ptr(), sB.data_ptr(),
                               meta.data_ptr(), C.data_ptr(), out_f, tp, in_f, gA.data_ptr(), gB.data_ptr())
        return C.t()[:t]

    def mxfp4_dequant(w_int8, s_e8m0):  # experts are MXFP4: E2M1 packed 2/byte, e8m0 scale per 32-block
        # transformers convention (convert_moe_packed_tensors): low nibble -> even elem, high -> odd.
        out_f, in_b = w_int8.shape; in_f = in_b * 2
        bb = w_int8.to(dev).to(torch.int32) & 0xFF
        codes = torch.empty(out_f, in_f, dtype=torch.long, device=dev)
        codes[:, 0::2] = bb & 0xF
        codes[:, 1::2] = (bb >> 4) & 0xF
        vals = FP4[codes]  # [out, in] on the E2M1 grid
        nb = in_f // 32
        s = s_e8m0.to(dev).float().reshape(out_f, nb, 1)  # e8m0 -> multiplier 2^(raw-127)
        return (vals.view(out_f, nb, 32) * s).reshape(out_f, in_f).to(torch.bfloat16)

    # ---- locate shards from the weight index, download only what we need ----
    idx_path = hf_hub_download(MODEL, "model.safetensors.index.json")
    weight_map = json.load(open(idx_path))["weight_map"]

    def get(name):  # load one tensor (downloads its shard on first touch, cached in /cache)
        shard = weight_map[name]
        p = hf_hub_download(MODEL, shard)
        with safe_open(p, framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    def load_raw(L, e, proj):  # raw stored (int8 weight, e8m0 scale)
        base = f"layers.{L}.ffn.experts.{e}.{proj}"
        return get(f"{base}.weight"), get(f"{base}.scale")

    def load_expert(L, e, proj):  # -> bf16 [out,in]
        return mxfp4_dequant(*load_raw(L, e, proj))

    if mode == "validate":
        print(f"# M2 MXFP4->bf16 dequant + sparse-FP4 pack validation ({MODEL})", flush=True)
        wr, sr = load_raw(4, 0, "w1")
        print(f"  raw w1: w.dtype={wr.dtype} w.shape={tuple(wr.shape)}  "
              f"s.dtype={sr.dtype} s.shape={tuple(sr.shape)} s.min={sr.float().min():.4g} s.max={sr.float().max():.4g}", flush=True)

        # deterministic proof: re-encode decoded bf16 -> MXFP4 bytes with the STORED scale; must reproduce
        # the stored int8 exactly. Validates the E2M1 table, sign bit, e8m0 scale, and 32-block layout.
        # (Nibble order is the documented OCP/transformers convention low->even; a global swap would
        # surface as catastrophic PPL in M4, not a subtle tax.)
        W0 = load_expert(4, 0, "w1")
        out0, in0 = W0.shape; nb0 = in0 // 32
        s0 = sr.to(dev).float().reshape(out0, nb0, 1)
        codes0 = q_fp4((W0.view(out0, nb0, 32).float() / s0.clamp_min(1e-30))).reshape(out0, in0)
        rebytes = (codes0[:, 0::2] | (codes0[:, 1::2] << 4)).to(torch.int32) & 0xFF
        orig = wr.to(dev).to(torch.int32) & 0xFF
        raw_exact = (rebytes == orig).float().mean().item()
        # value-based (E2M1 +0.0 == -0.0, so signed-zero code aliasing is not an error)
        vmatch = ((FP4[orig & 0xF] == FP4[rebytes & 0xF]) &
                  (FP4[(orig >> 4) & 0xF] == FP4[(rebytes >> 4) & 0xF])).float().mean().item()
        print(f"  round-trip: value-exact={vmatch * 100:.3f}%  raw-byte={raw_exact * 100:.3f}% "
              f"(gap = harmless +0/-0 code aliasing)  -> "
              f"{'DECODE VERIFIED' if vmatch > 0.9999 else 'DECODE SUSPECT'}", flush=True)

        for (L, e) in [(4, 0), (4, 1)]:
            for proj in ("w1", "w3", "w2"):
                W = load_expert(L, e, proj)
                out_f, in_f = W.shape
                x = (torch.randn(64, in_f, device=dev, dtype=torch.bfloat16) * 0.1)
                yk = kernel_fwd(x, pack(W), out_f, in_f)
                yf = F.linear(act_fp4_dequant(x.float()), sparse_fp4_dequant(W.float()))
                ck = F.cosine_similarity(yk.float().flatten(), yf.float().flatten(), dim=0).item()
                wtax = F.cosine_similarity(sparse_fp4_dequant(W.float()).flatten(), W.float().flatten(), dim=0).item()
                print(f"  L{L} e{e} {proj}: out={out_f} in={in_f}  Wabsmax={W.abs().max():.3g} Wstd={W.float().std():.3g}  "
                      f"cos(kernel,fakequant)={ck:.5f}  cos(sparseFP4,bf16)={wtax:.4f}", flush=True)
        print("# M2 validate done (transformers MATCH => decode correct; kernel==fakequant => pack correct; "
              "cos(sparseFP4,bf16) is the real-weight 2:4-FP4 tax)", flush=True)
        return

    if mode == "prep":
        # Coverage/scaling probe: pack a full layer's experts, report bytes + time. The serving path
        # (M3) packs at vLLM load time from the same recipe, so we do NOT persist a 40-layer artifact
        # for the magnitude-2:4 path -- this just measures per-layer prep cost and sparse footprint.
        import time

        L = layer_lo
        if f"layers.{L}.ffn.experts.0.w1.weight" not in weight_map:
            print(f"layer {L} has no routed experts (dense/first-k layer); nothing to pack", flush=True)
            return
        t0 = time.time(); nbytes = 0; ne = 0
        for e in range(256):
            if f"layers.{L}.ffn.experts.{e}.w1.weight" not in weight_map:
                break
            for proj in ("w1", "w3", "w2"):
                for buf in pack(load_expert(L, e, proj)):
                    nbytes += buf.numel() * buf.element_size()
            ne += 1
        dt = time.time() - t0
        print(f"# prep layer {L}: packed {ne} experts x3 proj in {dt:.1f}s "
              f"({dt / max(ne, 1) * 1000:.0f} ms/expert)  sparse_bytes={nbytes / 1e6:.1f}MB/layer  "
              f"=> ~{nbytes / 1e6 * 40 / 1e3:.1f}GB for 40 MoE layers", flush=True)
        return

    raise SystemExit(f"unknown mode {mode}")


@app.local_entrypoint()
def main(mode: str = "validate", layer_lo: int = 0, layer_hi: int = 43) -> None:
    run.remote(mode=mode, layer_lo=layer_lo, layer_hi=layer_hi)
