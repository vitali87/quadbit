"""Verify the TWO-LEVEL sparse FP4 kernel: does the per-row/col fp32 global rescale in the epilogue
close the sparse deploy gap (single-level through-kernel 10.97 vs two-level fake-quant 8.30)?

The mma applies the per-16-kept (A) / per-32 (B) ue4m3 LOCAL scales; the new epilogue multiplies the
accumulator by the per-weight-row gA and per-token gB fp32 globals -- exactly the two-level move that
took dense NVFP4 from block-rel 0.38 to 0.097. This checks the ported kernel against its own two-level
dequant (RED: arithmetic/layout must be near-exact) and against the two-level sparse fake-quant the
recovery trains on (the deploy-gap number), and contrasts it with the single-level kernel vs a bf16
2:4 reference so the improvement is visible in relative-error terms.

Run:  uv run modal run harness/verify_sparse_2lvl.py
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
    .add_local_dir((ROOT / "cuda").as_posix(), "/root/cuda")
)
app = modal.App("quadbit-verify-sp2lvl", image=image)


@app.function(gpu="RTX-PRO-6000", timeout=1200)
def run() -> None:
    import ctypes

    import torch
    import torch.nn.functional as F

    so = "/root/sparse_fp4.so"
    c = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
                        "-o", so, "/root/cuda/sparse_fp4_lib.cu", "-lcuda"], capture_output=True, text=True)
    if c.returncode != 0:
        print(c.stderr, flush=True); return
    lib = ctypes.CDLL(so)
    lib.sparse_fp4_mm.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3
    lib.sparse_fp4_mm_2lvl.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    lib.sparse_fp4_mm_2lvl_t.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    lib.quantize_act_nvfp4.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2
    lib.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    dev = torch.device("cuda")

    FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
                       dtype=torch.float32, device=dev)
    BND = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], dtype=torch.float32, device=dev)
    _cc = torch.arange(128, device=dev)
    _e, _m = (_cc >> 3) & 0xf, _cc & 7
    UE4M3 = torch.where(_e == 0, _m.float() * 0.001953125,
                        (1.0 + _m.float() / 8.0) * torch.exp2((_e - 7).float()))

    def q_fp4(v):
        return torch.bucketize(v.abs(), BND) | ((v < 0).long() << 3)

    def enc_ue4m3_t(s):  # torch replica of the kernel's enc_ue4m3 (no denormals, min-normal clamp)
        mant_f, e = torch.frexp(s.clamp_min(1e-30))
        mm = 2.0 * mant_f
        biased = (e - 1) + 7
        mant = torch.round((mm - 1.0) * 8.0).long()
        carry = mant == 8
        mant = torch.where(carry, torch.zeros_like(mant), mant)
        biased = torch.where(carry, biased + 1, biased)
        code = (biased.long() << 3) | mant
        code = torch.where(biased < 1, torch.ones_like(code), code)
        code = torch.where(biased > 15, torch.full_like(code, 0x7f), code)
        code = torch.where(s >= 480.0, torch.full_like(code, 0x7f), code)
        return torch.where(s > 0, code, torch.zeros_like(code))

    def pair24(W):  # pair-granular 2:4 mask indices: (out, ks, 16, 2) kept-pair indices
        out_f, in_f = W.shape; ks = in_f // 128
        Wg = W.view(out_f, ks, 16, 4, 2)
        i01, _ = Wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
        keptW = torch.gather(Wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
        return Wg, i01, keptW, ks

    def sparse_fp4_dequant_2lvl(W):  # TWO-LEVEL weight fake-quant (== finetune_pair.sparse_fp4_dequant)
        out_f, in_f = W.shape
        Wg, i01, keptW, ks = pair24(W)
        gA = (keptW.abs().amax(dim=(1, 2, 3, 4), keepdim=True) / 2688.0).clamp_min(1e-30).reshape(out_f, 1, 1)
        blk = keptW.reshape(out_f, ks, 4, 8, 2)
        sdeq = UE4M3[enc_ue4m3_t((blk.abs().amax(dim=(3, 4)) / 6.0) / gA)] * gA
        kd = (FP4[q_fp4(blk / sdeq[..., None, None])] * sdeq[..., None, None]).reshape(out_f, ks, 16, 2, 2)
        Wd = torch.zeros(out_f, ks, 16, 4, 2, device=dev)
        Wd.scatter_(3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2), kd)
        return Wd.reshape(out_f, in_f)

    def masked_bf16(W):  # 2:4-pruned W, NO fp4 (the full-precision-kept reference for the sparse pattern)
        out_f, in_f = W.shape
        Wg, i01, keptW, ks = pair24(W)
        Wd = torch.zeros(out_f, ks, 16, 4, 2, device=dev)
        Wd.scatter_(3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2), keptW.reshape(out_f, ks, 16, 2, 2))
        return Wd.reshape(out_f, in_f).to(torch.bfloat16).float()

    def act_2lvl_deq_p32(x):  # TWO-LEVEL per-32 activation fake-quant (matches the sparse mma B-side)
        lead = x.shape[:-1]; i = x.shape[-1]
        b = x.to(torch.bfloat16).float().reshape(-1, i)
        gB = (b.abs().amax(-1, keepdim=True) / 2688.0).clamp_min(1e-30)
        bb = b.reshape(b.shape[0], i // 32, 32)
        sdeq = UE4M3[enc_ue4m3_t((bb.abs().amax(-1) / 6.0) / gB)] * gB
        return (FP4[q_fp4(bb / sdeq[..., None])] * sdeq[..., None]).reshape(*lead, i)

    def pack_weight_2lvl(W):  # kernel packing: LOCAL ue4m3 codes + fp32 per-row global gA
        out_f, in_f = W.shape
        Wg, i01, keptW, ks = pair24(W)
        gA = (keptW.abs().amax(dim=(1, 2, 3, 4), keepdim=True) / 2688.0).clamp_min(1e-30).reshape(out_f, 1, 1)
        blk = keptW.reshape(out_f, ks, 4, 8, 2)
        scode = enc_ue4m3_t((blk.abs().amax(dim=(3, 4)) / 6.0) / gA)   # LOCAL codes (relative to gA)
        sdeq = UE4M3[scode] * gA                                       # full dequant scale for coding
        kc = q_fp4(blk / sdeq[..., None, None])
        Ac = (kc[..., 0] | (kc[..., 1] << 4)).reshape(out_f, ks * 32).to(torch.uint8)
        nib = (i01[..., 0] | (i01[..., 1] << 2)).view(out_f, ks, 2, 8)
        sh = (torch.arange(8, device=dev) * 4).view(1, 1, 1, 8)
        meta = (nib << sh).sum(-1).to(torch.int32).permute(1, 0, 2).contiguous()
        scaleA = scode.to(torch.uint8).permute(1, 0, 2).contiguous()
        gAf = gA.reshape(out_f).float().contiguous()
        return Ac.contiguous(), meta, scaleA, gAf, ks

    def kernel_mm_2lvl(W, x):  # QuadbitLinear.forward on the TWO-LEVEL path
        out_f, in_f = W.shape
        Ac, meta, scaleA, gA, ks = pack_weight_2lvl(W)
        x2 = x.reshape(-1, in_f).to(torch.bfloat16)
        t = x2.shape[0]; pad = (-t) % 128
        if pad:
            x2 = torch.cat([x2, x2.new_zeros(pad, in_f)], 0)
        x2 = x2.contiguous(); tp = t + pad
        Bb = torch.empty((tp, in_f // 2), dtype=torch.uint8, device=dev)
        sB = torch.empty((ks, tp, 4), dtype=torch.uint8, device=dev)
        gB = torch.empty((tp,), dtype=torch.float32, device=dev)
        lib.quantize_act_nvfp4_2lvl(x2.data_ptr(), Bb.data_ptr(), sB.data_ptr(), gB.data_ptr(), tp, in_f)
        C = torch.empty((out_f, tp), dtype=torch.bfloat16, device=dev)
        lib.sparse_fp4_mm_2lvl(Ac.data_ptr(), Bb.data_ptr(), scaleA.data_ptr(),
                               sB.data_ptr(), meta.data_ptr(), C.data_ptr(), out_f, tp, in_f,
                               gA.data_ptr(), gB.data_ptr())
        # zero-copy transposed entry: C is [tp, out_f] contiguous; return C[:t] directly (no .t())
        Ct = torch.empty((tp, out_f), dtype=torch.bfloat16, device=dev)
        lib.sparse_fp4_mm_2lvl_t(Ac.data_ptr(), Bb.data_ptr(), scaleA.data_ptr(),
                                 sB.data_ptr(), meta.data_ptr(), Ct.data_ptr(), out_f, tp, in_f,
                                 gA.data_ptr(), gB.data_ptr())
        Ct = Ct[:t]
        assert Ct.is_contiguous() and Ct.stride() == (out_f, 1) and Ct.storage_offset() == 0
        return C.t()[:t].float(), Bb[:t], sB[:, :t], gB[:t], Ct.float()

    def deq_from_kernel_act_2lvl(Bb, sB, gB, in_f):  # dequant EXACTLY what mma+epilogue see for acts
        t = Bb.shape[0]; ks = in_f // 128
        codes = torch.empty(t, in_f, dtype=torch.long, device=dev)
        codes[:, 0::2] = (Bb & 0xf).long()
        codes[:, 1::2] = (Bb >> 4).long()
        vals = FP4[codes]
        sc = sB.permute(1, 0, 2).reshape(t, ks * 4).long()          # (t, in_f/32) LOCAL codes
        local = UE4M3[sc].repeat_interleave(32, dim=1)              # (t, in_f)
        return vals * local * gB[:, None]                          # * per-token global

    _MID = (UE4M3[:-1] + UE4M3[1:]) / 2

    def kernel_mm_1lvl(W, x):  # OLD single-level path (no global), for contrast
        out_f, in_f = W.shape; ks = in_f // 128
        Wg, i01, keptW, _ = pair24(W.float())
        blk = keptW.reshape(out_f, ks, 4, 8, 2)
        scode = torch.bucketize(blk.abs().amax(dim=(3, 4)) / 6.0, _MID)
        kc = q_fp4(blk / UE4M3[scode][..., None, None])
        Ac = (kc[..., 0] | (kc[..., 1] << 4)).reshape(out_f, ks * 32).to(torch.uint8).contiguous()
        nib = (i01[..., 0] | (i01[..., 1] << 2)).view(out_f, ks, 2, 8)
        sh = (torch.arange(8, device=dev) * 4).view(1, 1, 1, 8)
        meta = (nib << sh).sum(-1).to(torch.int32).permute(1, 0, 2).contiguous()
        scaleA = scode.to(torch.uint8).permute(1, 0, 2).contiguous()
        x2 = x.reshape(-1, in_f).to(torch.bfloat16)
        t = x2.shape[0]; pad = (-t) % 128
        if pad:
            x2 = torch.cat([x2, x2.new_zeros(pad, in_f)], 0)
        x2 = x2.contiguous(); tp = t + pad
        Bb = torch.empty((tp, in_f // 2), dtype=torch.uint8, device=dev)
        sB = torch.empty((ks, tp, 4), dtype=torch.uint8, device=dev)
        lib.quantize_act_nvfp4(x2.data_ptr(), Bb.data_ptr(), sB.data_ptr(), tp, in_f)
        C = torch.empty((out_f, tp), dtype=torch.bfloat16, device=dev)
        lib.sparse_fp4_mm(Ac.data_ptr(), Bb.data_ptr(), scaleA.data_ptr(), sB.data_ptr(),
                          meta.data_ptr(), C.data_ptr(), out_f, tp, in_f)
        return C.t()[:t].float()

    def rel(a, b):
        return (a - b).norm().item() / b.norm().item()

    torch.manual_seed(0)
    print("shape              RED wpath   kernel-vs-2lvl-STE   1lvl-vs-bf16   2lvl-vs-bf16", flush=True)
    worst_red = worst_track = 0.0
    for (out_f, in_f, toks) in [(512, 512, 128), (2048, 2048, 256), (5632, 2048, 128), (4096, 4096, 512)]:
        W = torch.randn(out_f, in_f, device=dev) * 0.02
        x = torch.randn(toks, in_f, device=dev)

        C2, Bb, sB, gB, C2t = kernel_mm_2lvl(W, x)
        tmax = (C2 - C2t).abs().max().item()      # zero-copy transposed path must be BITWISE-equal to .t()
        assert tmax == 0.0, f"zero-copy mismatch {out_f}x{in_f}: max|diff| {tmax}"
        Wd2 = sparse_fp4_dequant_2lvl(W.float())
        act_kernel = deq_from_kernel_act_2lvl(Bb, sB, gB, in_f)
        red = rel(C2, F.linear(act_kernel, Wd2))                    # kernel vs its OWN two-level dequant
        track = rel(C2, F.linear(act_2lvl_deq_p32(x.float()), Wd2))  # kernel vs two-level sparse fake-quant

        # single-level kernel (old path) for contrast, both vs the bf16 2:4 reference
        Wmask = masked_bf16(W)
        ref_bf16 = F.linear(x.to(torch.bfloat16).float(), Wmask)
        C1 = kernel_mm_1lvl(W, x)
        d1 = rel(C1, ref_bf16); d2 = rel(C2, ref_bf16)
        worst_red = max(worst_red, red); worst_track = max(worst_track, track)
        print(f"{out_f}x{in_f} t{toks:<4}   {red:.4f}      {track:.4f}              {d1:.4f}         {d2:.4f}", flush=True)

    print(f"\nRED (kernel vs own two-level dequant, must be < 0.02): worst {worst_red:.4f}  "
          f"-> {'PASS' if worst_red < 0.02 else 'FAIL'}", flush=True)
    print(f"TRACK (kernel vs two-level sparse fake-quant): worst {worst_track:.4f}  "
          f"-> {'deploy gap closed' if worst_track < 0.03 else 'residual'}", flush=True)
    print("1lvl-vs-bf16 >> 2lvl-vs-bf16 shows the global rescale recovering the single-level loss.", flush=True)
    print("ZERO-COPY: transposed [N,M] contiguous output is bitwise-equal to .t() on all shapes -> PASS", flush=True)


@app.local_entrypoint()
def main() -> None:
    run.remote()
