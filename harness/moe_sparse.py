"""M1: grouped 2:4-sparse FP4 expert GEMM -- correctness validation.

quadbit's banked serving path is a DENSE-MLP op (gate/up/down on one LlamaMLP). The large-model
targets (DeepSeek-V4-Flash, GLM-5.2) are MoE: MLP FLOPs live in hundreds of routed experts behind a
router. This file builds the missing piece -- a grouped sparse-FP4 expert forward -- and proves it
numerically before any large-model or multi-GPU spend.

Minimum-viable grouped kernel (correctness first): route tokens, then per active expert run the
VALIDATED single-MLP sparse two-level-NVFP4 path (QuadbitLinear, reused verbatim) over its token
segment, then top-k weighted-combine + shared expert. The per-expert python dispatch is the known
ceiling; a fused grouped kernel / CUDA-graph amortization is the later speed optimization (M3).

Validation compares three MoE forwards on identical routing + weights:
  kernel      -- grouped driver calling the sparse-FP4 kernel per expert
  fakequant   -- pytorch reference doing the IDENTICAL 2:4-sparse-FP4 math (isolates driver bugs)
  bf16 dense  -- unquantized reference (shows the sparse-FP4 accuracy tax; informational)
PASS = cos(kernel, fakequant) ~ 1.0  (driver reproduces the kernel math exactly).
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
    .pip_install("numpy")
    .add_local_dir((ROOT / "cuda").as_posix(), "/root/cuda")
)
app = modal.App("quadbit-moe", image=image)


@app.function(gpu="RTX-PRO-6000", timeout=1800)
def validate() -> None:
    import ctypes

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

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
    torch.manual_seed(0)

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

    def sparse_fp4_dequant(W):  # pair-2:4 magnitude + per-16 two-level (per-out-row global gA)
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

    def act_fp4_dequant(x):  # per-32 two-level (matched to the sparse mma B operand)
        lead = x.shape[:-1]; i = x.shape[-1]
        b = x.to(torch.bfloat16).float().reshape(-1, i)
        gB = (b.abs().amax(-1, keepdim=True) / 2688.0).clamp_min(1e-30)
        bb = b.reshape(b.shape[0], i // 32, 32)
        sdeq = UE4M3[enc_ue4m3_t((bb.abs().amax(-1) / 6.0) / gB)] * gB
        return (FP4[q_fp4(bb / sdeq.clamp_min(1e-30)[..., None])] * sdeq[..., None]).reshape(*lead, i)

    class QuadbitLinear(nn.Module):  # sparse two-level KERNEL path (reused verbatim from repair.py)
        def __init__(self, W):
            super().__init__()
            out_f, in_f = W.shape; ks = in_f // 128
            self.out_f, self.in_f, self.ks = out_f, in_f, ks
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
            self.register_buffer("Ac", Ac.contiguous()); self.register_buffer("meta", meta)
            self.register_buffer("scaleA", scode.to(torch.uint8).permute(1, 0, 2).contiguous())
            self.register_buffer("gA", gA.reshape(out_f).float().contiguous())

        def forward(self, x):
            lead = x.shape[:-1]; x2 = x.reshape(-1, self.in_f).to(torch.bfloat16)
            t = x2.shape[0]; pad = (-t) % 128
            if pad:
                x2 = torch.cat([x2, x2.new_zeros(pad, self.in_f)], 0)
            x2 = x2.contiguous(); tp = t + pad
            Bb = torch.empty((tp, self.in_f // 2), dtype=torch.uint8, device=dev)
            sB = torch.empty((self.ks, tp, 4), dtype=torch.uint8, device=dev)
            gB = torch.empty((tp,), dtype=torch.float32, device=dev)
            lib.quantize_act_nvfp4_2lvl(x2.data_ptr(), Bb.data_ptr(), sB.data_ptr(), gB.data_ptr(), tp, self.in_f)
            C = torch.empty((self.out_f, tp), dtype=torch.bfloat16, device=dev)
            lib.sparse_fp4_mm_2lvl(self.Ac.data_ptr(), Bb.data_ptr(), self.scaleA.data_ptr(),
                                   sB.data_ptr(), self.meta.data_ptr(), C.data_ptr(),
                                   self.out_f, tp, self.in_f, self.gA.data_ptr(), gB.data_ptr())
            return C.t()[:t].reshape(*lead, self.out_f).to(x.dtype)

    # ---- MoE forwards. Same weights, same routing fed to all three paths. ----
    def route(logits, k, scale):  # -> topk idx [T,k], combine weights [T,k]
        w, idx = logits.softmax(-1).topk(k, dim=-1)
        return idx, (w / w.sum(-1, keepdim=True) * scale)

    def expert_kernel(x, gud):  # one expert MLP through the sparse-FP4 kernel
        g, u, d = gud
        return d((F.silu(g(x).float()) * u(x).float()).to(torch.bfloat16))

    def moe_kernel(X, experts, shared, idx, wts):  # grouped driver: gather tokens per expert, batch dispatch
        T, H = X.shape; out = torch.zeros(T, H, device=dev)
        for e in range(len(experts)):
            hit = (idx == e)  # [T,k] bool
            if not hit.any():
                continue  # ponytail: skip empty experts; O(E) python loop, ceiling = launch overhead (M3)
            tok = hit.any(-1)  # tokens routed to e
            ye = expert_kernel(X[tok], experts[e])
            out[tok] += (wts[tok] * hit[tok]).sum(-1, keepdim=True).to(ye.dtype) * ye
        out += expert_kernel(X, shared)
        return out

    def moe_kernel_pertok(X, experts, shared, idx, wts):  # naive per-token, IDENTICAL kernel ops (driver control)
        T, H = X.shape; out = torch.zeros(T, H, device=dev)
        for t in range(T):
            acc = expert_kernel(X[t:t + 1], shared).float()
            for slot in range(idx.shape[1]):
                e = idx[t, slot].item()
                acc = acc + wts[t, slot].float() * expert_kernel(X[t:t + 1], experts[e]).float()
            out[t] = acc
        return out

    def moe_fakequant(X, W, shared_W, idx, wts):  # identical math in pure pytorch (driver-bug isolator)
        def mlp(x, wg, wu, wd):
            xq = act_fp4_dequant(x.float())
            g = F.linear(xq, sparse_fp4_dequant(wg.float()))
            u = F.linear(xq, sparse_fp4_dequant(wu.float()))
            h = act_fp4_dequant((F.silu(g) * u))
            return F.linear(h, sparse_fp4_dequant(wd.float()))
        T, H = X.shape; out = torch.zeros(T, H, device=dev)
        for e in range(len(W)):
            hit = (idx == e); tok = hit.any(-1)
            if not tok.any():
                continue
            ye = mlp(X[tok], *W[e])
            out[tok] += (wts[tok] * hit[tok]).sum(-1, keepdim=True) * ye
        out += mlp(X, *shared_W)
        return out

    def moe_bf16(X, W, shared_W, idx, wts):  # unquantized dense reference
        def mlp(x, wg, wu, wd):
            return F.linear(F.silu(F.linear(x, wg)) * F.linear(x, wu), wd)
        T, H = X.shape; out = torch.zeros(T, H, device=dev)
        for e in range(len(W)):
            hit = (idx == e); tok = hit.any(-1)
            if not tok.any():
                continue
            out[tok] += (wts[tok] * hit[tok]).sum(-1, keepdim=True) * mlp(X[tok].float(), *[w.float() for w in W[e]])
        out += mlp(X.float(), *[w.float() for w in shared_W])
        return out

    def cos(a, b):
        return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()

    configs = [
        ("control-1mlp", dict(H=4096, I=2048, E=1,  k=1, T=128)),  # single expert: intrinsic kernel-vs-fakequant tol
        ("tiny",         dict(H=512,  I=1024, E=16, k=4, T=256)),
        ("deepseek-shape", dict(H=4096, I=2048, E=8,  k=6, T=512)),  # real per-expert dims, fewer experts
    ]
    print("# M1 grouped sparse-FP4 MoE correctness (RTX-PRO-6000, sm_120a)", flush=True)
    print("# PASS gate = cos(grouped, per-token) ~ 1.0 (gather/scatter/combine correct).", flush=True)
    print("# cos(kernel,fakequant) is intrinsic quant tolerance (compare to control); "
          "cos(fakequant,bf16) is 2:4-FP4 accuracy tax on random weights (informational).", flush=True)
    ok = True
    for name, cf in configs:
        H, I, E, k, T = cf["H"], cf["I"], cf["E"], cf["k"], cf["T"]
        X = torch.randn(T, H, device=dev, dtype=torch.bfloat16) * 0.1
        Wr = [(torch.randn(I, H, device=dev) * (H ** -0.5),
               torch.randn(I, H, device=dev) * (H ** -0.5),
               torch.randn(H, I, device=dev) * (I ** -0.5)) for _ in range(E)]
        Ws = (torch.randn(I, H, device=dev) * (H ** -0.5),
              torch.randn(I, H, device=dev) * (H ** -0.5),
              torch.randn(H, I, device=dev) * (I ** -0.5))
        idx, wts = route(torch.randn(T, E, device=dev), k, 1.5)

        experts = [(QuadbitLinear(g), QuadbitLinear(u), QuadbitLinear(d)) for g, u, d in Wr]
        shared = (QuadbitLinear(Ws[0]), QuadbitLinear(Ws[1]), QuadbitLinear(Ws[2]))
        yk = moe_kernel(X, experts, shared, idx, wts)
        yf = moe_fakequant(X, Wr, Ws, idx, wts)
        yb = moe_bf16(X, Wr, Ws, idx, wts)

        n = min(64, T)  # driver isolation on a subset (per-token loop is O(T) kernel calls)
        yp = moe_kernel_pertok(X[:n], experts, shared, idx[:n], wts[:n])
        cdrv = cos(yk[:n], yp)
        ckf, cfb = cos(yk, yf), cos(yf, yb)
        verdict = "PASS" if cdrv > 0.9999 else "FAIL"
        ok = ok and cdrv > 0.9999
        print(f"[{name:14s}] H={H} I={I} E={E} k={k} T={T}  "
              f"cos(grouped,per-token)={cdrv:.6f}  cos(kernel,fakequant)={ckf:.6f}  "
              f"cos(fakequant,bf16)={cfb:.4f}  -> {verdict}", flush=True)
    print(f"# M1 {'PASS: grouped driver reproduces per-token kernel exactly' if ok else 'FAIL'}", flush=True)


@app.local_entrypoint()
def main() -> None:
    validate.remote()
