"""Localize the STE-vs-kernel PPL gap (fake-quant eval 9.15 vs real-kernel eval 10.03).

The finetune QAT optimizes against a torch STE forward (sparse_fp4_dequant + act_fp4_dequant in
fp32); deployment runs the CUDA sparse_fp4_mm. If those two functions disagree, QAT is training
against a different function than ships. Earlier kernel validation used UNIFORM weights (0x22
everywhere -> every ue4m3 scale identical -> a scale/layout permutation bug is invisible). This
probe drives the real kernel with real NON-UNIFORM weights + activations and splits the error:

  weight-path : compare kernel C against a reference that dequants the SAME activation the kernel
                quantized (call quant_act, read its bytes+scales back, dequant in torch). If this
                is ~0, the weight/scale/meta layout the mma consumes is correct.
  act-quant   : compare that kernel-matched-activation reference against the fp32 STE act dequant.
                This is the part QAT could close by matching its STE to the kernel's quantizer.

Run:  uv run modal run harness/probe_ste_kernel.py
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
app = modal.App("quadbit-probe-ste", image=image)


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
    lib.quantize_act_nvfp4.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2
    dev = torch.device("cuda")

    FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
                       dtype=torch.float32, device=dev)
    BND = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], dtype=torch.float32, device=dev)
    _cc = torch.arange(128, device=dev)
    _e, _m = (_cc >> 3) & 0xf, _cc & 7
    UE4M3 = torch.where(_e == 0, _m.float() * 0.001953125,
                        (1.0 + _m.float() / 8.0) * torch.exp2((_e - 7).float()))
    _MID = (UE4M3[:-1] + UE4M3[1:]) / 2

    def q_fp4(v):
        return torch.bucketize(v.abs(), BND) | ((v < 0).long() << 3)

    def enc(s):
        return torch.bucketize(s, _MID)

    def sparse_fp4_dequant(W):  # identical to finetune_pair / accuracy_sparse
        out_f, in_f = W.shape; ks = in_f // 128
        Wg = W.view(out_f, ks, 16, 4, 2)
        i01, _ = Wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
        keptW = torch.gather(Wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
        blk = keptW.reshape(out_f, ks, 4, 8, 2)
        sdeq = UE4M3[enc(blk.abs().amax(dim=(3, 4)) / 6.0)]
        kd = (FP4[q_fp4(blk / sdeq[..., None, None])] * sdeq[..., None, None]).reshape(out_f, ks, 16, 2, 2)
        Wd = torch.zeros(out_f, ks, 16, 4, 2, device=dev)
        Wd.scatter_(3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2), kd)
        return Wd.reshape(out_f, in_f)

    def act_fp4_dequant(x):  # fp32 STE activation fake-quant (what QAT optimizes against today)
        lead = x.shape[:-1]; i = x.shape[-1]
        b = x.reshape(-1, i // 32, 32)
        s = UE4M3[enc(b.abs().amax(-1) / 6.0)]
        return (FP4[q_fp4(b / s[..., None])] * s[..., None]).reshape(*lead, i)

    def enc_ue4m3_t(s):  # torch replica of the kernel's enc_ue4m3 (no denormals, min-normal clamp)
        mant_f, e = torch.frexp(s.clamp_min(1e-30))    # s = mant_f * 2^e, mant_f in [0.5,1)
        mm = 2.0 * mant_f                              # [1,2)
        biased = (e - 1) + 7
        mant = torch.round((mm - 1.0) * 8.0).long()    # 0..8
        carry = mant == 8
        mant = torch.where(carry, torch.zeros_like(mant), mant)
        biased = torch.where(carry, biased + 1, biased)
        code = (biased.long() << 3) | mant
        code = torch.where(biased < 1, torch.ones_like(code), code)
        code = torch.where(biased > 15, torch.full_like(code, 0x7f), code)
        code = torch.where(s >= 480.0, torch.full_like(code, 0x7f), code)
        return torch.where(s > 0, code, torch.zeros_like(code))

    def act_fp4_dequant_km(x):  # kernel-MATCHED STE: bf16 pre-round + no-denormal ue4m3 encode
        lead = x.shape[:-1]; i = x.shape[-1]
        xb = x.to(torch.bfloat16).float()              # kernel reads activations as bf16
        b = xb.reshape(-1, i // 32, 32)
        s = UE4M3[enc_ue4m3_t(b.abs().amax(-1) / 6.0)]
        return (FP4[q_fp4(b / s[..., None])] * s[..., None]).reshape(*lead, i)

    def pack_weight(W):  # QuadbitLinear packing, verbatim
        out_f, in_f = W.shape; ks = in_f // 128
        Wg = W.float().view(out_f, ks, 16, 4, 2)
        i01, _ = Wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
        keptW = torch.gather(Wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
        blk = keptW.reshape(out_f, ks, 4, 8, 2)
        scode = enc(blk.abs().amax(dim=(3, 4)) / 6.0)
        sdeq = UE4M3[scode]
        kc = q_fp4(blk / sdeq[..., None, None])
        Ac = (kc[..., 0] | (kc[..., 1] << 4)).reshape(out_f, ks * 32).to(torch.uint8)
        nib = (i01[..., 0] | (i01[..., 1] << 2)).view(out_f, ks, 2, 8)
        sh = (torch.arange(8, device=dev) * 4).view(1, 1, 1, 8)
        meta = (nib << sh).sum(-1).to(torch.int32).permute(1, 0, 2).contiguous()
        scaleA = scode.to(torch.uint8).permute(1, 0, 2).contiguous()
        return Ac.contiguous(), meta, scaleA, ks

    def kernel_mm(W, x):  # QuadbitLinear.forward, verbatim
        out_f, in_f = W.shape
        Ac, meta, scaleA, ks = pack_weight(W)
        x2 = x.reshape(-1, in_f).to(torch.bfloat16)
        t = x2.shape[0]; pad = (-t) % 128
        if pad:
            x2 = torch.cat([x2, x2.new_zeros(pad, in_f)], 0)
        x2 = x2.contiguous(); tp = t + pad
        Bb = torch.empty((tp, in_f // 2), dtype=torch.uint8, device=dev)
        sB = torch.empty((ks, tp, 4), dtype=torch.uint8, device=dev)
        lib.quantize_act_nvfp4(x2.data_ptr(), Bb.data_ptr(), sB.data_ptr(), tp, in_f)
        C = torch.empty((out_f, tp), dtype=torch.bfloat16, device=dev)
        lib.sparse_fp4_mm(Ac.data_ptr(), Bb.data_ptr(), scaleA.data_ptr(),
                          sB.data_ptr(), meta.data_ptr(), C.data_ptr(), out_f, tp, in_f)
        return C.t()[:t].float(), Bb[:t], sB[:, :t], tp

    def deq_from_kernel_act(Bb, sB, in_f):  # dequant EXACTLY what the mma reads for activations
        t = Bb.shape[0]; ks = in_f // 128
        codes = torch.empty(t, in_f, dtype=torch.long, device=dev)
        codes[:, 0::2] = (Bb & 0xf).long()
        codes[:, 1::2] = (Bb >> 4).long()
        vals = FP4[codes]
        sc = sB.permute(1, 0, 2).reshape(t, ks * 4).long()          # (t, in_f/32)
        scale = UE4M3[sc].repeat_interleave(32, dim=1)              # (t, in_f)
        return vals * scale

    torch.manual_seed(0)
    for (out_f, in_f, toks) in [(512, 512, 128), (2048, 2048, 256), (5632, 2048, 128)]:
        W = torch.randn(out_f, in_f, device=dev) * 0.02          # realistic weight scale
        x = torch.randn(toks, in_f, device=dev)                  # realistic activation scale

        C, Bb, sB, _ = kernel_mm(W, x)
        Wd = sparse_fp4_dequant(W.float())
        act_kernel = deq_from_kernel_act(Bb, sB, in_f)           # what the mma actually sees
        ref_weightpath = F.linear(act_kernel, Wd)                # kernel-matched act, torch matmul
        ref_full_ste = F.linear(act_fp4_dequant(x.float()), Wd)  # what QAT optimizes against

        def rel(a, b):
            return (a - b).norm().item() / b.norm().item()

        ref_km_ste = F.linear(act_fp4_dequant_km(x.float()), Wd)  # NEW kernel-matched STE
        wp = rel(C, ref_weightpath)      # kernel arithmetic/layout error (should be ~0)
        aq = rel(ref_weightpath, ref_full_ste)  # act-quant mismatch (kernel quant vs fp32 STE)
        tot = rel(C, ref_full_ste)       # total kernel-vs-STE gap (today)
        km = rel(C, ref_km_ste)          # kernel vs NEW matched STE (target: ~weight-path)
        print(f"[{out_f}x{in_f} t{toks}]  weight-path {wp:.4f}  act-quant {aq:.4f}  "
              f"TOTAL(kernel vs fp32-STE) {tot:.4f}  kernel vs MATCHED-STE {km:.4f}", flush=True)

    # localize the replica error: compare torch quantizers directly against the kernel's activation
    W = torch.randn(512, 512, device=dev) * 0.02
    x = torch.randn(128, 512, device=dev)
    _, Bb, sB, _ = kernel_mm(W, x)
    act_kernel = deq_from_kernel_act(Bb, sB, 512)
    r_fp32 = ((act_fp4_dequant(x.float()) - act_kernel).norm() / act_kernel.norm()).item()
    r_km = ((act_fp4_dequant_km(x.float()) - act_kernel).norm() / act_kernel.norm()).item()
    # compare scale codes: kernel sB (ks, t, 4) -> (t, ks*4); my replica per 32-block
    sc_kernel = sB.permute(1, 0, 2).reshape(128, 16).long()
    xb = x.to(torch.bfloat16).float().reshape(128, 16, 32)
    sc_km = enc_ue4m3_t(xb.abs().amax(-1) / 6.0)
    mism = (sc_kernel != sc_km).float().mean().item()
    print(f"\ndirect act rel: fp32-STE {r_fp32:.4f}  matched-STE {r_km:.4f}  "
          f"scale-code mismatch frac {mism:.4f}", flush=True)
    bad = (sc_kernel != sc_km).nonzero()[:5]
    for r, cbl in bad.tolist():
        print(f"  block[{r},{cbl}] kernel_code={sc_kernel[r, cbl].item()} "
              f"km_code={sc_km[r, cbl].item()} amax/6={(xb[r, cbl].abs().amax() / 6).item():.5f}", flush=True)

    # RED check: weight path must be near-exact; if it isn't, there's a scale/meta layout bug
    W = torch.randn(512, 512, device=dev) * 0.02
    x = torch.randn(128, 512, device=dev)
    C, Bb, sB, _ = kernel_mm(W, x)
    wp = ((C - F.linear(deq_from_kernel_act(Bb, sB, 512), sparse_fp4_dequant(W.float()))).norm()
          / F.linear(deq_from_kernel_act(Bb, sB, 512), sparse_fp4_dequant(W.float())).norm()).item()
    assert wp < 0.02, f"weight-path layout bug: kernel disagrees with its own dequant by {wp:.4f}"
    print(f"OK weight-path rel {wp:.4f} < 0.02", flush=True)


@app.local_entrypoint()
def main() -> None:
    run.remote()
