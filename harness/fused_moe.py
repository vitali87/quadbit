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


@app.function(gpu="RTX-PRO-6000", timeout=1800)
def profile_sass(kernels: str = "matmul_sp_moe_gu_swiglu,matmul_sp_moe") -> None:
    """Deterministic kernel resource + SASS profile (ncu is blocked on Modal, see quadbit-perf-bottleneck).
    nvcc --ptxas-options=-v gives regs/spills/smem per kernel (the occupancy signal); cuobjdump -sass gives
    the opcode histogram (OMMA% = tensor-core utilization; if OMMA is a tiny fraction, we're issue/latency
    bound on address+scale arithmetic, the known wall). No clock-noise: these are compile-time facts."""
    import re
    import subprocess
    from collections import Counter

    cub = "/root/k.cubin"
    c = subprocess.run(["nvcc", "-arch=sm_120a", "-cubin", "--ptxas-options=-v", "-O3",
                        "-o", cub, "/root/cuda/sparse_fp4_lib.cu", "-lcuda"], capture_output=True, text=True)
    print("# ===== ptxas -v (registers / spills / smem per kernel) =====", flush=True)
    for ln in c.stderr.splitlines():
        if "Function properties" in ln or "registers" in ln or "spill" in ln or "bytes smem" in ln or "Compiling entry" in ln:
            print("  " + ln.strip(), flush=True)
    if c.returncode != 0:
        print(c.stderr[-2000:], flush=True); raise SystemExit(1)
    full = subprocess.run(["cuobjdump", "-sass", cub], capture_output=True, text=True).stdout
    knames = kernels.split(",")
    blocks = {}
    cur = None
    for ln in full.splitlines():
        if "Function :" in ln or ".text." in ln:
            cur = next((k for k in knames if k in ln), None)
            if cur:
                blocks.setdefault(cur, Counter())
        m = re.search(r"/\*[0-9a-f]+\*/\s+@?!?P?\d?\s*([A-Z][A-Z0-9_]+)", ln)
        if m and cur:
            blocks[cur][m.group(1)] += 1
    for kname in kernels.split(","):
        ops = blocks.get(kname)
        if not ops:
            print(f"# {kname}: (not found; have {list(blocks)[:6]})", flush=True); continue
        tot = sum(ops.values())
        omma = sum(v for k, v in ops.items() if "MMA" in k)
        ldsm = sum(v for k, v in ops.items() if k.startswith("LDSM"))
        ldg = sum(v for k, v in ops.items() if k in ("LDG", "LD", "UTMALDG"))
        print(f"# {kname}: {tot} SASS insts | MMA {omma} ({100*omma/tot:.1f}%) "
              f"LDSM {ldsm} LDG/TMA {ldg} | top: "
              + " ".join(f"{k}={v}" for k, v in ops.most_common(10)), flush=True)
    print("# profile_sass done", flush=True)


@app.function(gpu="RTX-PRO-6000", timeout=1800)
def bench_shapes(iters: int = 20) -> None:
    """Where does the small-expert penalty flip? Times the FUSED GEMM1 (sparse_moe_gu_swiglu) at gpt-oss's
    tiny experts vs frontier large-expert shapes (GLM-5.2 / DeepSeek-V4 class), reporting achieved TFLOP/s
    (counting the FULL 2I*H macs, so directly comparable to Marlin's dense-bf16 math). Marlin W4A16 dequants
    to bf16 -> bounded by the ~500 TFLOP/s bf16 tensor peak on SM120. If our sparse-FP4 kernel CLEARS 500
    at large-expert shapes, we provably beat Marlin's math there; gpt-oss's ~1016-tok/expert 2944x3072 is the
    worst case. Synthetic random packed weights (efficiency is shape-, not value-, dependent)."""
    import torch
    dev = torch.device("cuda")
    so = "/root/sparse_fp4.so"
    c = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
                        "-o", so, "/root/cuda/sparse_fp4_lib.cu", "-lcuda"], capture_output=True, text=True)
    if c.returncode != 0:
        print(c.stderr, flush=True); raise SystemExit(1)
    lib = ctypes.CDLL(so)
    lib.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    lib.sparse_moe_gu_swiglu.argtypes = ([ctypes.c_void_p] * 5 + [ctypes.c_int] * 4
                                         + [ctypes.c_void_p] * 6 + [ctypes.c_int] * 2 + [ctypes.c_void_p])
    lib.qb_init_gusw_attrs()
    FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
    BND = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], device=dev)
    cc = torch.arange(128, device=dev); e_, m_ = (cc >> 3) & 0xf, cc & 7
    UE4M3 = torch.where(e_ == 0, m_.float() * 0.001953125, (1.0 + m_.float() / 8.0) * torch.exp2((e_ - 7).float()))

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

    def pack(w):
        out_f, in_f = w.shape; ks = in_f // 128
        wg = w.float().view(out_f, ks, 16, 4, 2)
        i01, _ = wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
        kept = torch.gather(wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
        ga = (kept.abs().amax(dim=(1, 2, 3, 4), keepdim=True) / 2688.0).clamp_min(1e-30).reshape(out_f, 1, 1)
        blk = kept.reshape(out_f, ks, 4, 8, 2)
        scode = enc((blk.abs().amax(dim=(3, 4)) / 6.0) / ga); sdeq = UE4M3[scode] * ga
        kc = q_fp4(blk / sdeq.clamp_min(1e-30)[..., None, None])
        ac = (kc[..., 0] | (kc[..., 1] << 4)).reshape(out_f, ks * 32).to(torch.uint8)
        nib = (i01[..., 0] | (i01[..., 1] << 2)).view(out_f, ks, 2, 8)
        sh = (torch.arange(8, device=dev) * 4).view(1, 1, 1, 8)
        meta = (nib << sh).sum(-1).to(torch.int32).permute(1, 0, 2).contiguous()
        return (ac.contiguous(), meta, scode.to(torch.uint8).permute(1, 0, 2).contiguous(),
                ga.reshape(out_f).float().contiguous())

    def stack(ps):
        return tuple(torch.cat([p[i] for p in ps], 0 if i in (0, 3) else 1).contiguous() for i in range(4))

    def bench(fn):
        for _ in range(4):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(iters):
            s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
        ts.sort(); return ts[len(ts) // 2]

    # (label, H hidden, I intermediate, E experts, tok/expert) — pad to kernel alignment inside
    shapes = [
        ("gpt-oss-20b   ", 3072, 2944, 32, 1016),
        ("gpt-oss @4k/e ", 3072, 2944, 32, 4096),
        ("GLM-class     ", 5120, 1536, 128, 2048),
        ("DeepSeek-class", 7168, 2048, 64, 2048),
        ("DeepSeek @8k/e", 7168, 2048, 64, 8192),
    ]
    print("# FUSED GEMM1 (sparse_moe_gu_swiglu) efficiency vs expert shape — Marlin bf16 ceiling ~500 TFLOP/s", flush=True)
    for label, H, I, E, tpe in shapes:
        Hp = ((H + 127) // 128) * 128
        mpe_gu = ((2 * I + 255) // 256) * 256; ih = mpe_gu // 2
        torch.manual_seed(0)
        packs = [pack(torch.randn(mpe_gu, Hp, device=dev) * 0.05) for _ in range(E)]
        gu = stack(packs)
        gubias = torch.zeros(E, mpe_gu, device=dev, dtype=torch.bfloat16)
        tok = E * tpe
        assign = torch.randint(0, E, (tok,), device=dev)
        order = torch.argsort(assign); counts = torch.bincount(assign, minlength=E)
        padc = ((counts + 127) // 128) * 128; r_pad = int(padc.sum().item())
        poff = torch.cumsum(padc, 0) - padc; coff = torch.cumsum(counts, 0) - counts
        se = assign[order]; within = torch.arange(tok, device=dev) - coff[se]
        src = torch.full((r_pad,), 0, dtype=torch.long, device=dev)
        src[poff[se] + within] = order
        ends = torch.cumsum(padc, 0)
        eblk = torch.bucketize(torch.arange(r_pad // 128, device=dev) * 128, ends, right=True).clamp_max_(E - 1).to(torch.int32)
        x = torch.randn(tok, Hp, device=dev, dtype=torch.bfloat16) * 0.1
        ks = Hp // 128
        bb = torch.empty((tok, Hp // 2), dtype=torch.uint8, device=dev)
        sb = torch.empty((ks, tok, 4), dtype=torch.uint8, device=dev)
        gb = torch.empty((tok,), dtype=torch.float32, device=dev)
        lib.quantize_act_nvfp4_2lvl(x.data_ptr(), bb.data_ptr(), sb.data_ptr(), gb.data_ptr(), tok, Hp)
        idx = src
        bb_g = bb[idx].contiguous(); sb_g = sb[:, idx, :].contiguous(); gb_g = gb[idx].contiguous()
        hw = torch.empty((r_pad, ih // 2), dtype=torch.uint8, device=dev)
        sh = torch.empty((ih // 128, r_pad, 4), dtype=torch.uint8, device=dev)
        ac, meta, sA, ga = gu

        def g1():
            lib.sparse_moe_gu_swiglu(ac.data_ptr(), bb_g.data_ptr(), sA.data_ptr(), sb_g.data_ptr(),
                                     meta.data_ptr(), ac.shape[0], mpe_gu, r_pad, Hp, ga.data_ptr(),
                                     gb_g.data_ptr(), eblk.data_ptr(), gubias.data_ptr(), hw.data_ptr(),
                                     sh.data_ptr(), ih, 0, 0)
        ms = bench(g1)
        flop = 2.0 * r_pad * mpe_gu * Hp   # full 2I*H macs x2
        tflops = flop / 1e12 / (ms / 1e3)
        beat = "BEATS bf16" if tflops > 500 else "below bf16"
        print(f"# {label} H={H} I={I} E={E} tok/e={tpe:5d} r_pad={r_pad:7d}: {ms:6.2f}ms  "
              f"{tflops:6.0f} TFLOP/s  ({beat} ~500)", flush=True)
    print("# bench_shapes done", flush=True)


@app.function(gpu="RTX-PRO-6000", timeout=1800)
def bench_moe_vs_bf16(iters: int = 20) -> None:
    """PROVE the large-expert win: our FULL fused W4A4 monolith MoE (route+quant+gather+gu_swiglu+seg_bw+
    combine) vs a Marlin-class W4A16 baseline (bf16 grouped GEMM, NO activation-quant -- the honest Marlin
    compute) across gpt-oss-small -> GLM/DeepSeek-large expert dims, SAME invocation (no clock noise). Shows
    the crossover: Marlin's W4A16 wins gpt-oss's tiny experts; our sparse-FP4 monolith wins the frontier
    large-expert shapes where the 2:4 mma advantage dominates. Both paths share routing + inverse combine."""
    import torch
    dev = torch.device("cuda")
    so = "/root/sparse_fp4.so"
    c = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
                        "-o", so, "/root/cuda/sparse_fp4_lib.cu", "-lcuda"], capture_output=True, text=True)
    if c.returncode != 0:
        print(c.stderr, flush=True); raise SystemExit(1)
    lib = ctypes.CDLL(so)
    lib.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    lib.sparse_moe_gu_swiglu.argtypes = ([ctypes.c_void_p] * 5 + [ctypes.c_int] * 4 + [ctypes.c_void_p] * 6 + [ctypes.c_int] * 2 + [ctypes.c_void_p])
    lib.sparse_moe_mm_2lvl_bw.argtypes = ([ctypes.c_void_p] * 6 + [ctypes.c_int] * 4 + [ctypes.c_void_p] * 5 + [ctypes.c_int] + [ctypes.c_void_p])
    lib.moe_combine_topk.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    lib.moe_route.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
    lib.qb_init_moe_attrs(); lib.qb_init_gusw_attrs()
    FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
    BND = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], device=dev)
    cc = torch.arange(128, device=dev); e_, m_ = (cc >> 3) & 0xf, cc & 7
    UE4M3 = torch.where(e_ == 0, m_.float() * 0.001953125, (1.0 + m_.float() / 8.0) * torch.exp2((e_ - 7).float()))

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

    def pack(w):
        out_f, in_f = w.shape; ks = in_f // 128
        wg = w.float().view(out_f, ks, 16, 4, 2)
        i01, _ = wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
        kept = torch.gather(wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
        ga = (kept.abs().amax(dim=(1, 2, 3, 4), keepdim=True) / 2688.0).clamp_min(1e-30).reshape(out_f, 1, 1)
        blk = kept.reshape(out_f, ks, 4, 8, 2)
        scode = enc((blk.abs().amax(dim=(3, 4)) / 6.0) / ga); sdeq = UE4M3[scode] * ga
        kc = q_fp4(blk / sdeq.clamp_min(1e-30)[..., None, None])
        ac = (kc[..., 0] | (kc[..., 1] << 4)).reshape(out_f, ks * 32).to(torch.uint8)
        nib = (i01[..., 0] | (i01[..., 1] << 2)).view(out_f, ks, 2, 8)
        sh = (torch.arange(8, device=dev) * 4).view(1, 1, 1, 8)
        meta = (nib << sh).sum(-1).to(torch.int32).permute(1, 0, 2).contiguous()
        return (ac.contiguous(), meta, scode.to(torch.uint8).permute(1, 0, 2).contiguous(), ga.reshape(out_f).float().contiguous())

    def stack(ps):
        return tuple(torch.cat([p[i] for p in ps], 0 if i in (0, 3) else 1).contiguous() for i in range(4))

    def bench(fn):
        for _ in range(4):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); ts = []
        for _ in range(iters):
            s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
        ts.sort(); return ts[len(ts) // 2]

    shapes = [("gpt-oss-20b   ", 3072, 2944, 32, 1016), ("GLM-5.2-class ", 5120, 1536, 128, 2048),
              ("DeepSeek-V4   ", 7168, 2048, 64, 2048), ("DeepSeek @8k/e", 7168, 2048, 64, 8192)]
    print("# PROVE large-expert win: fused W4A4 monolith MoE vs Marlin-class W4A16 bf16 grouped-GEMM (same invocation)", flush=True)
    for label, H, I, E, tpe in shapes:
        Hp = ((H + 127) // 128) * 128; mpe_gu = ((2 * I + 255) // 256) * 256; Ih = mpe_gu // 2
        Ip = ((I + 127) // 128) * 128; mpeH = ((H + 255) // 256) * 256
        torch.manual_seed(1)
        guS = stack([pack(torch.randn(mpe_gu, Hp, device=dev) * 0.05) for _ in range(E)])
        dS = stack([pack(torch.randn(mpeH, Ip, device=dev) * 0.05) for _ in range(E)])
        gu_bf = (torch.randn(E, Hp, mpe_gu, device=dev) * 0.05).to(torch.bfloat16)   # [E,K,N] for grouped_mm
        dn_bf = (torch.randn(E, Ip, mpeH, device=dev) * 0.05).to(torch.bfloat16)
        gubias = torch.zeros(E, mpe_gu, device=dev, dtype=torch.bfloat16)
        tok = E * tpe; topk = 1
        assign = torch.randint(0, E, (tok,), device=dev)
        counts = torch.bincount(assign, minlength=E); padc = ((counts + 127) // 128) * 128
        poff = (torch.cumsum(padc, 0) - padc).to(torch.int64); r_pad = int(padc.sum().item())
        offs = torch.cumsum(padc, 0).to(torch.int32)
        src = torch.full((r_pad,), -1, dtype=torch.int32, device=dev); counter = torch.empty(E, dtype=torch.int32, device=dev)
        lib.moe_route(assign.to(torch.int32).data_ptr(), poff.data_ptr(), counter.data_ptr(), src.data_ptr(), tok, E, 0)
        ends = torch.cumsum(padc, 0)
        eblk = torch.bucketize(torch.arange(r_pad // 128, device=dev) * 128, ends, right=True).clamp_max_(E - 1).to(torch.int32)
        valid = src >= 0; srcc = src.clamp_min(0).long()
        x = torch.randn(tok, Hp, device=dev, dtype=torch.bfloat16) * 0.1
        inv = torch.full((tok * topk + 1,), -1, dtype=torch.int32, device=dev)
        inv.scatter_(0, torch.where(valid, src, torch.full_like(src, tok * topk)).long(), torch.arange(r_pad, dtype=torch.int32, device=dev))
        inv = inv[:tok * topk].contiguous()
        acg, mg, sAg, gag = guS; acd, md, sAd, gad = dS

        def fused_moe():
            bb = torch.empty((tok, Hp // 2), dtype=torch.uint8, device=dev); sb = torch.empty((Hp // 128, tok, 4), dtype=torch.uint8, device=dev); gb = torch.empty((tok,), dtype=torch.float32, device=dev)
            lib.quantize_act_nvfp4_2lvl(x.data_ptr(), bb.data_ptr(), sb.data_ptr(), gb.data_ptr(), tok, Hp)
            bbg = bb[srcc].contiguous(); sbg = sb[:, srcc, :].contiguous(); gbg = (gb[srcc] * valid.float()).contiguous()
            hw = torch.empty((r_pad, Ih // 2), dtype=torch.uint8, device=dev); sh = torch.empty((Ih // 128, r_pad, 4), dtype=torch.uint8, device=dev)
            lib.sparse_moe_gu_swiglu(acg.data_ptr(), bbg.data_ptr(), sAg.data_ptr(), sbg.data_ptr(), mg.data_ptr(), acg.shape[0], mpe_gu, r_pad, Hp, gag.data_ptr(), gbg.data_ptr(), eblk.data_ptr(), gubias.data_ptr(), hw.data_ptr(), sh.data_ptr(), Ih, 0, 0)
            scw = valid.float().contiguous(); dbuf = torch.empty((r_pad, mpeH), dtype=torch.bfloat16, device=dev)
            lib.sparse_moe_mm_2lvl_bw(acd.data_ptr(), hw.data_ptr(), sAd.data_ptr(), sh.data_ptr(), md.data_ptr(), dbuf.data_ptr(), acd.shape[0], mpeH, r_pad, Ip, gad.data_ptr(), 0, eblk.data_ptr(), scw.data_ptr(), 0, H, 0)
            y = torch.empty(tok, H, dtype=torch.bfloat16, device=dev)
            lib.moe_combine_topk(dbuf.data_ptr(), inv.data_ptr(), y.data_ptr(), tok, H, topk, mpeH, 0)
            return y

        def bf16_moe():  # Marlin-class W4A16 compute: bf16 grouped GEMM, NO act-quant
            xs = x[srcc] * valid[:, None]                                  # gather bf16 (no quant)
            gu = torch._grouped_mm(xs, gu_bf, offs=offs)                   # [r_pad, mpe_gu]
            g, u = gu[:, 0::2], gu[:, 1::2]
            h = (g * torch.sigmoid(1.702 * g)) * (u + 1.0)                 # swiglu
            hp = torch.zeros(r_pad, Ip, device=dev, dtype=torch.bfloat16); hp[:, :Ih] = h.to(torch.bfloat16)
            d = torch._grouped_mm(hp, dn_bf, offs=offs)[:, :H]             # [r_pad, H]
            y = torch.empty(tok, H, dtype=torch.bfloat16, device=dev)
            lib.moe_combine_topk(d.contiguous().data_ptr(), inv.data_ptr(), y.data_ptr(), tok, H, topk, H, 0)
            return y

        try:
            tf = bench(fused_moe); tb = bench(bf16_moe)
            win = "quadbit WINS" if tf < tb else "Marlin wins"
            print(f"# {label} H={H} I={I} E={E} tok/e={tpe:5d}: fused-W4A4 {tf:6.2f}ms | bf16-W4A16 {tb:6.2f}ms | "
                  f"{tb / tf:.2f}x  ({win})", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"# {label}: skipped ({type(ex).__name__}: {ex})", flush=True)
    print("# bench_moe_vs_bf16 done", flush=True)


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
                                         + [ctypes.c_void_p] * 6 + [ctypes.c_int] * 2 + [ctypes.c_void_p])
    lib.sparse_moe_mm_2lvl_bw.argtypes = ([ctypes.c_void_p] * 6 + [ctypes.c_int] * 4
                                          + [ctypes.c_void_p] * 5 + [ctypes.c_int] + [ctypes.c_void_p])
    lib.moe_combine_topk.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    lib.moe_route.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
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
                                 guBias_fused.data_ptr(), Hw.data_ptr(), sH.data_ptr(), Ih, 0, 0)
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

    # bw-down correctness: fused bias+weight (coalesced, outT=3) + index_add vs the reference `ref`
    Hw2, sH2 = gemm1_fused()
    acd_, metad_, sAd_, gad_ = dS
    sc_w_chk = (w_of[srcc] * valid.float()).contiguous()
    dbuf_bw = torch.empty((r_pad, mpeH), dtype=torch.bfloat16, device=dev)
    lib.sparse_moe_mm_2lvl_bw(acd_.data_ptr(), Hw2.data_ptr(), sAd_.data_ptr(), sH2.data_ptr(), metad_.data_ptr(),
                              dbuf_bw.data_ptr(), acd_.shape[0], mpeH, r_pad, Ip, gad_.data_ptr(), 0, eblk.data_ptr(),
                              sc_w_chk.data_ptr(), dBias_s.data_ptr(), H, 0)
    y_bw = torch.zeros(tokens, H, dtype=torch.bfloat16, device=dev)
    y_bw.index_add_(0, tok_of[srcc], dbuf_bw[:, :H])
    c_bw = F.cosine_similarity(y_bw.flatten().float(), ref.flatten().float(), dim=0).item()
    print(f"# CORRECTNESS bw-down (coalesced bias+weight + index_add) vs unfused ref: cos={c_bw:.5f} "
          f"{'OK' if c_bw > 0.99 else 'CHECK'}", flush=True)

    # combine kernel correctness: inverse-index top-k combine must MATCH index_add exactly (same bf16 sum)
    inv_chk = torch.full((tokens * topk,), -1, dtype=torch.int32, device=dev)
    inv_chk[src[valid].long()] = torch.arange(r_pad, dtype=torch.int32, device=dev)[valid]
    y_comb = torch.empty(tokens, H, dtype=torch.bfloat16, device=dev)
    lib.moe_combine_topk(dbuf_bw.data_ptr(), inv_chk.data_ptr(), y_comb.data_ptr(), tokens, H, topk, mpeH, 0)
    c_comb = F.cosine_similarity(y_comb.flatten().float(), y_bw.flatten().float(), dim=0).item()
    print(f"# CORRECTNESS moe_combine vs index_add: cos={c_comb:.6f} "
          f"{'OK' if c_comb > 0.999 else 'MISMATCH'}", flush=True)

    t_g1un = bench(gemm1_unfused)
    t_g1fu = bench(gemm1_fused)
    print(f"# TIMING GEMM1 (gate_up+swiglu+quant)  unfused={t_g1un:.3f}ms  fused={t_g1fu:.3f}ms  "
          f"speedup={t_g1un / t_g1fu:.2f}x", flush=True)
    # whole-MoE: unfused = gate_up-unfused + down-unfused ; fused = gemm1_fused + down-scatter
    t_moe_un = bench(lambda: (gemm1_unfused(), unfused_down()))
    t_moe_fu = bench(fused_full)
    print(f"# TIMING whole-MoE  unfused={t_moe_un:.3f}ms  fused={t_moe_fu:.3f}ms  "
          f"speedup={t_moe_un / t_moe_fu:.2f}x  (T={tokens} topk={topk}, r_pad={r_pad})", flush=True)

    # ---- gather cost (the shared wall): serve does bbx[idx] + sbx[:,idx,:] + gbx[idx] per layer ----
    bbx, sbx, gbx = quant_act(x.to(torch.bfloat16), H)   # per-token quant (t rows), once
    idx = tok_of[srcc]

    def gather_only():
        _bb = bbx[idx].contiguous()
        _sb = sbx[:, idx, :].contiguous()
        _gb = (gbx[idx] * valid.float()).contiguous()
        return _bb, _sb, _gb

    t_gather = bench(gather_only)
    print(f"# TIMING gather (bbx[idx]+sbx[:,idx,:]+gbx[idx]) = {t_gather:.3f}ms  "
          f"({100 * t_gather / t_moe_fu:.0f}% of fused whole-MoE) -- the shared fast_gu/fused wall", flush=True)

    # ---- TRUE serve per-phase split via CUDA EVENTS (GPU timeline, NO cpu-sync inflation that falsely
    # blamed routing). Replicates the plugin fast_fused path exactly: route -> quant x -> gather -> GEMM1
    # -> down(seg+bias) -> index_add. This decides which fusion actually pays before building it. ----
    _rmax = (((tokens * topk + n_experts * 128) + 127) // 128) * 128
    dB_stack = torch.stack(dB_)  # [E,H] float, for the bias add

    def fast_route(assign):
        order = torch.argsort(assign)
        counts = torch.bincount(assign, minlength=n_experts)
        padc = ((counts + 127) // 128) * 128
        poff = torch.cumsum(padc, 0) - padc; coff = torch.cumsum(counts, 0) - counts
        se = assign[order]; within = torch.arange(order.numel(), device=dev) - coff[se]
        src2 = torch.full((_rmax,), -1, dtype=torch.long, device=dev)
        src2[poff[se] + within] = order
        ends = torch.cumsum(padc, 0)
        eblk2 = torch.bucketize(torch.arange(_rmax // 128, device=dev) * 128, ends, right=True).clamp_max_(n_experts - 1).to(torch.int32)
        return src2, eblk2

    acg, metag, sAg, gag = guS_fused
    acd, metad, sAd, gad = dS
    names = ["route", "quant", "gather", "gemm1", "down", "scatter"]
    acc_ms = [0.0] * 6
    NREP = 20
    for _ in range(NREP + 3):
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(7)]
        ev[0].record()
        assign2 = topk_ids.reshape(-1).to(torch.long)
        tok2 = torch.arange(tokens, device=dev).repeat_interleave(topk)
        w2 = topk_w.reshape(-1).float()
        src2, eblk2 = fast_route(assign2)
        valid2 = src2 >= 0; srcc2 = src2.clamp_min(0); row_exp2 = assign2[srcc2]
        ev[1].record()
        bbx2, sbx2, gbx2 = quant_act(x, H)
        ev[2].record()
        idx2 = tok2[srcc2]
        bb_g = bbx2[idx2].contiguous(); sb_g = sbx2[:, idx2, :].contiguous()
        gb_g = (gbx2[idx2] * valid2.float()).contiguous()
        ev[3].record()
        Hw = torch.empty((_rmax, Ih // 2), dtype=torch.uint8, device=dev)
        sH = torch.empty((Ih // 128, _rmax, 4), dtype=torch.uint8, device=dev)
        lib.sparse_moe_gu_swiglu(acg.data_ptr(), bb_g.data_ptr(), sAg.data_ptr(), sb_g.data_ptr(), metag.data_ptr(),
                                 acg.shape[0], mpe_gu, _rmax, Hp, gag.data_ptr(), gb_g.data_ptr(), eblk2.data_ptr(),
                                 guBias_fused.data_ptr(), Hw.data_ptr(), sH.data_ptr(), Ih, 0, 0)
        ev[4].record()
        sc_w2 = (w2[srcc2] * valid2.float()).contiguous()
        dbuf = torch.empty((_rmax, mpeH), dtype=torch.bfloat16, device=dev)
        lib.sparse_moe_mm_2lvl_bw(acd.data_ptr(), Hw.data_ptr(), sAd.data_ptr(), sH.data_ptr(), metad.data_ptr(),
                                  dbuf.data_ptr(), acd.shape[0], mpeH, _rmax, Ip, gad.data_ptr(), 0, eblk2.data_ptr(),
                                  sc_w2.data_ptr(), dBias_s.data_ptr(), H, 0)  # down-GEMM + fused bias+weight, coalesced
        ev[5].record()
        rr2 = torch.arange(_rmax, dtype=torch.int32, device=dev)
        src_idx2 = torch.where(valid2, src2, torch.full_like(src2, tokens * topk))
        inverse2 = torch.full((tokens * topk + 1,), -1, dtype=torch.int32, device=dev)
        inverse2.scatter_(0, src_idx2, rr2)
        y = torch.empty(tokens, H, dtype=torch.bfloat16, device=dev)
        lib.moe_combine_topk(dbuf.data_ptr(), inverse2[:tokens * topk].data_ptr(), y.data_ptr(), tokens, H, topk, mpeH, 0)
        ev[6].record()
        torch.cuda.synchronize()
        for i in range(6):
            acc_ms[i] += ev[i].elapsed_time(ev[i + 1])
    tot = sum(acc_ms)
    print("# ===== TRUE serve per-phase (CUDA events, per-layer, median-ish over reps) =====", flush=True)
    for nm, a in zip(names, acc_ms, strict=False):
        print(f"#   {nm:8s} {a / (NREP + 3):6.3f}ms  ({100 * a / tot:4.1f}%)", flush=True)
    print(f"#   TOTAL    {tot / (NREP + 3):6.3f}ms/layer   (kernels gemm1+down vs plumbing route+quant+gather+scatter)", flush=True)

    # ---- MONOLITH increment: fused routing kernel (counting sort) vs torch _fast_route ----
    def route_kernel(assign_i):
        counts = torch.bincount(assign_i, minlength=n_experts)
        padc = ((counts + 127) // 128) * 128
        poff = (torch.cumsum(padc, 0) - padc).to(torch.int64)
        srck = torch.full((_rmax,), -1, dtype=torch.int32, device=dev)
        counter = torch.empty(n_experts, dtype=torch.int32, device=dev)
        lib.moe_route(assign_i.data_ptr(), poff.data_ptr(), counter.data_ptr(), srck.data_ptr(),
                      assign_i.numel(), n_experts, 0)
        ends = torch.cumsum(padc, 0)
        eblkk = torch.bucketize(torch.arange(_rmax // 128, device=dev) * 128, ends, right=True).clamp_max_(n_experts - 1).to(torch.int32)
        return srck, eblkk

    assign_i = topk_ids.reshape(-1).to(torch.int32)
    srck, eblkk = route_kernel(assign_i)
    validk = srck >= 0
    placed_exp = assign_i[srck.clamp_min(0).long()]
    block_exp = eblkk[torch.arange(_rmax, device=dev) // 128]
    ok_route = bool((placed_exp[validk] == block_exp[validk]).all().item())
    cnt_match = int(validk.sum().item()) == tokens * topk
    print(f"# CORRECTNESS moe_route grouping: {'OK' if ok_route and cnt_match else 'MISMATCH'} "
          f"(placed {int(validk.sum().item())}/{tokens * topk})", flush=True)
    t_tr = bench(lambda: fast_route(topk_ids.reshape(-1).to(torch.long)))
    t_kr = bench(lambda: route_kernel(topk_ids.reshape(-1).to(torch.int32)))
    print(f"# TIMING routing  torch _fast_route={t_tr:.3f}ms  kernel counting-sort={t_kr:.3f}ms  "
          f"speedup={t_tr / t_kr:.2f}x", flush=True)
    print("# fused_moe done", flush=True)
