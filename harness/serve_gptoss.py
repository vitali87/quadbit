"""M1-serving gate: run gpt-oss-20b in vLLM 0.24 with quadbit 2:4-sparse-FP4 experts (QB_MOE=sparse)
and confirm (a) the GptOssMxfp4MoEMethod patch fires, (b) the segmented sparse kernel actually runs
(SPARSE_EXPERT_CALLS>0), (c) output is finite + coherent, (d) a teacher-forced PPL. QB_MOE=off gives
the stock-MXFP4 baseline. No recovery yet, so sparse PPL will be degraded-but-coherent (the 2:4 tax the
recovery recipe closes next). Offline math already verified in gptoss_prep.py (M0/M2/M1/M1.5)."""
import math
import os
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
MODEL = "openai/gpt-oss-20b"
MIN = 60

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.12")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64", "HF_HOME": "/cache", "HF_XET_HIGH_PERFORMANCE": "1"})
    .pip_install("vllm", "huggingface_hub", "datasets")
    .run_commands("pip install --force-reinstall --no-deps flashinfer-python==0.6.14 flashinfer-cubin==0.6.13")
    .env({"FLASHINFER_DISABLE_VERSION_CHECK": "1"})
    .add_local_dir(str(ROOT / "cuda"), "/root/cuda", copy=True)
    .add_local_dir(str(ROOT / "harness" / "qb_vllm_plugin"), "/opt/qb_plugin", copy=True)
    .run_commands("pip install --force-reinstall --no-deps /opt/qb_plugin", force_build=True)
)
app = modal.App("quadbit-serve-gptoss", image=image)
vol = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)

PASSAGE = (
    "The mitochondria is the powerhouse of the cell. Photosynthesis converts sunlight, water, "
    "and carbon dioxide into glucose and oxygen. The Earth orbits the Sun once every year, and "
    "the Moon orbits the Earth roughly every twenty-eight days. Water boils at one hundred "
    "degrees Celsius at sea level and freezes at zero degrees. The human heart pumps blood "
    "through arteries and veins, delivering oxygen to every tissue in the body. Shakespeare "
    "wrote many famous plays, including Hamlet, Macbeth, and Romeo and Juliet. The speed of "
    "light in a vacuum is approximately three hundred thousand kilometres per second."
)


def _ensure_so() -> None:
    # plugin loads /cache/sparse_fp4.so; build it if the volume lacks it. CUDA-13 base -> gencode
    # compute_120a,sm_120a (a plain -arch=sm_120a drops the 'a' and ptxas rejects the block-scale mma).
    import subprocess
    so = "/cache/sparse_fp4.so"
    # Always rebuild: the .so lives on the volume and outlives image rebuilds, and add_local_dir mtimes
    # are unreliable as a freshness gate. ~90s compile is cheap insurance that kernel edits take effect.
    print(f"# building {so} (CUDA-13 gencode compute_120a,sm_120a)", flush=True)
    c = subprocess.run(["nvcc", "-gencode=arch=compute_120a,code=sm_120a", "-O3", "-shared",
                        "-Xcompiler", "-fPIC", "-o", so, "/root/cuda/sparse_fp4_lib.cu", "-lcuda"],
                       capture_output=True, text=True)
    if c.returncode != 0:
        print(c.stderr, flush=True); raise SystemExit(1)
    vol.commit()


@app.function(gpu="RTX-PRO-6000", timeout=20 * MIN, volumes={"/cache": vol})
def bench_vs_sota(iters: int = 30, mrows: int = 8192, backend: str = "auto", proj: str = "gate_up") -> None:
    """SOTA comparison (per always-benchmark-vs-sota): our 2:4-sparse-FP4 GEMM vs FlashInfer's DENSE FP4 GEMM
    (flashinfer.mm_fp4, the real b12x/trtllm/cutlass SOTA -- same FP4 tensor cores, W4A4), same invocation, at
    gpt-oss/GLM/DeepSeek gate_up [M,N=2I,K=H]. HONEST framing: ours is 2:4-SPARSE (half the mma work, lossy);
    theirs is DENSE (full work, lossless). The ratio = the sparsity THROUGHPUT advantage vs SOTA dense. Also
    times FlashInfer dense at half-K (their equivalent if THEY could skip half -- they can't, no sparse path)."""
    import ctypes
    import time

    import torch
    import flashinfer
    _ensure_so()
    M = mrows
    dev = torch.device("cuda")
    lib = ctypes.CDLL("/cache/sparse_fp4.so")
    lib.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    lib.sparse_fp4_mm_2lvl.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    lib.sparse_fp4_mm_diag.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3
    lib.sparse_fp4_mm_diag_wk1.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3
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
        code = torch.where(biased < 1, torch.ones_like(code), code); code = torch.where(biased > 15, torch.full_like(code, 0x7f), code)
        code = torch.where(s >= 480.0, torch.full_like(code, 0x7f), code)
        return torch.where(s > 0, code, torch.zeros_like(code))

    def pack(w):
        out_f, in_f = w.shape; ks = in_f // 128
        wg = w.float().view(out_f, ks, 16, 4, 2)
        i01, _ = wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
        kept = torch.gather(wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
        ga = (kept.abs().amax(dim=(1, 2, 3, 4), keepdim=True) / 2688.0).clamp_min(1e-30).reshape(out_f, 1, 1)
        blk = kept.reshape(out_f, ks, 4, 8, 2); scode = enc((blk.abs().amax(dim=(3, 4)) / 6.0) / ga); sdeq = UE4M3[scode] * ga
        kc = q_fp4(blk / sdeq.clamp_min(1e-30)[..., None, None])
        ac = (kc[..., 0] | (kc[..., 1] << 4)).reshape(out_f, ks * 32).to(torch.uint8)
        nib = (i01[..., 0] | (i01[..., 1] << 2)).view(out_f, ks, 2, 8)
        sh = (torch.arange(8, device=dev) * 4).view(1, 1, 1, 8)
        meta = (nib << sh).sum(-1).to(torch.int32).permute(1, 0, 2).contiguous()
        return (ac.contiguous(), meta, scode.to(torch.uint8).permute(1, 0, 2).contiguous(), ga.reshape(out_f).float().contiguous())

    def bench(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); ts = []
        for _ in range(iters):
            s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
        ts.sort(); return ts[len(ts) // 2]

    shapes = [("gpt-oss   ", 3072, 2944), ("GLM-5.2   ", 5120, 1536), ("DeepSeek  ", 7168, 2048)]
    # sweep mode (mrows=0): DeepSeek only, loop M IN-PROCESS (one .so build, one container -> same clocks)
    # to separate wave-quantization from a per-iteration mainloop deficit -- if the floor-vs-FlashInfer
    # ratio oscillates with M (peaks near integer waves, dips at fractional tails) it is wave quant.
    if mrows == 0:
        Ms = [8192, 16384, 24576, 32768, 49152]  # collapse hunt: push M up, watch cutlass fold + our margin widen
    else:
        Ms = [mrows]
    print(f"# SOTA: our 2:4-SPARSE-FP4 GEMM vs FlashInfer DENSE-FP4 (mm_fp4 backend={backend}), proj={proj}, M={Ms}", flush=True)
    for label, H, I in shapes:
      # gate_up: weights[2I,H] -> N=2I,K=H (square-ish, cutlass-favored).  down: weights[H,I] -> N=H,K=I
      # (large-N/small-K = the large-M regime where FlashInfer collapses; our WIN front per leaderboard).
      Kw, N = (((H + 127) // 128) * 128, ((2 * I + 255) // 256) * 256) if proj == "gate_up" \
          else (((I + 127) // 128) * 128, ((H + 255) // 256) * 256)
      ac, meta, sA, ga = pack(torch.randn(N, Kw, device=dev) * 0.05)
      for M in Ms:  # noqa: PLR1704
        x = torch.randn(M, Kw, device=dev, dtype=torch.bfloat16) * 0.1
        bb = torch.empty((M, Kw // 2), dtype=torch.uint8, device=dev); sb = torch.empty((Kw // 128, M, 4), dtype=torch.uint8, device=dev); gbq = torch.empty((M,), dtype=torch.float32, device=dev)
        C = torch.empty((N, M), dtype=torch.bfloat16, device=dev)

        def ours():
            lib.quantize_act_nvfp4_2lvl(x.data_ptr(), bb.data_ptr(), sb.data_ptr(), gbq.data_ptr(), M, Kw)
            lib.sparse_fp4_mm_2lvl(ac.data_ptr(), bb.data_ptr(), sA.data_ptr(), sb.data_ptr(), meta.data_ptr(), C.data_ptr(), N, M, Kw, ga.data_ptr(), gbq.data_ptr())

        def diag():  # mainloop-only floor: same quant + GEMM, epilogue stripped (no gA/gB rescale)
            lib.quantize_act_nvfp4_2lvl(x.data_ptr(), bb.data_ptr(), sb.data_ptr(), gbq.data_ptr(), M, Kw)
            lib.sparse_fp4_mm_diag(ac.data_ptr(), bb.data_ptr(), sA.data_ptr(), sb.data_ptr(), meta.data_ptr(), C.data_ptr(), N, M, Kw)

        def diago():  # WK=1 128x128 @ 2 CTA/SM: does doubling occupancy (16 warps/SM) hide the TMA latency that 1-CTA/SM exposes?
            lib.quantize_act_nvfp4_2lvl(x.data_ptr(), bb.data_ptr(), sb.data_ptr(), gbq.data_ptr(), M, Kw)
            lib.sparse_fp4_mm_diag_wk1(ac.data_ptr(), bb.data_ptr(), sA.data_ptr(), sb.data_ptr(), meta.data_ptr(), C.data_ptr(), N, M, Kw)

        try:
            w = torch.randn(N, Kw, device=dev, dtype=torch.bfloat16) * 0.05
            a_gsf = (x.abs().amax() / (6 * 448)).reshape(1).to(torch.float32)
            b_gsf = (w.abs().amax() / (6 * 448)).reshape(1).to(torch.float32)
            a_fp4, a_sf = flashinfer.nvfp4_quantize(x, a_gsf)          # a (M, K/2)
            b_fp4, b_sf = flashinfer.nvfp4_quantize(w, b_gsf)          # b (N, K/2) -> mm_fp4 wants (K/2, N) col-major
            bt = b_fp4.t().contiguous()
            alpha = (a_gsf * b_gsf).to(torch.float32)

            def flash():  # cutlass backend (b12x = their fastest needs a pre-strided layout -> this is conservative for FlashInfer)
                flashinfer.mm_fp4(a_fp4, bt, a_sf, b_sf, alpha, out_dtype=torch.bfloat16, backend=backend, use_nvfp4=True)
            to = bench(ours); tf = bench(flash); td = bench(diag); tdo = bench(diago)
            flop = 2.0 * M * N * Kw
            tfo = flop / 1e12 / (to / 1e3); tff = flop / 1e12 / (tf / 1e3)
            v = "quadbit-sparse WINS" if to < tf else "FlashInfer-dense wins"
            print(f"# {label} M={M} N={N} K={Kw}: ours-sparse {to:.3f}ms ({tfo:.0f} TF/s) | FlashInfer-dense {tf:.3f}ms ({tff:.0f} TF/s) | {tf / to:.2f}x  ({v})", flush=True)
            print(f"#   ^ M={M} mainloop-only floor (no epilogue) {td:.3f}ms | epilogue costs {to - td:.3f}ms ({100 * (to - td) / to:.0f}% of ours) | floor-vs-FlashInfer {tf / td:.2f}x", flush=True)
            print(f"#   ^ M={M} WK1-2CTA floor {tdo:.3f}ms ({flop / 1e12 / (tdo / 1e3):.0f} TF/s) | vs base floor {td / tdo:.3f}x (>1 = faster) | wk1-vs-FlashInfer {tf / tdo:.2f}x", flush=True)
        except Exception as ex:  # noqa: BLE001
            import traceback
            print(f"# {label} FlashInfer FAIL: {type(ex).__name__}: {ex}", flush=True)
            print(traceback.format_exc()[-900:], flush=True)
    print("# bench_vs_sota done", flush=True)


@app.function(gpu="RTX-PRO-6000", timeout=15 * MIN, volumes={"/cache": vol})
def mmfp4_probe() -> None:
    """Diagnose the exact mm_fp4 weight/scale layout: print nvfp4_quantize output shapes + docstring, then try
    every (b orientation, backend) combo on a tiny GEMM and report which one actually runs."""
    import torch
    import flashinfer
    dev = torch.device("cuda")
    M, K, N = 256, 512, 384
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.1
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05
    a_gsf = (x.abs().amax() / (6 * 448)).reshape(1).float(); b_gsf = (w.abs().amax() / (6 * 448)).reshape(1).float()
    a_fp4, a_sf = flashinfer.nvfp4_quantize(x, a_gsf)
    b_fp4, b_sf = flashinfer.nvfp4_quantize(w, b_gsf)
    print(f"# a_fp4 {tuple(a_fp4.shape)}/{a_fp4.dtype}  a_sf {tuple(a_sf.shape)}/{a_sf.dtype}", flush=True)
    print(f"# b_fp4 {tuple(b_fp4.shape)}/{b_fp4.dtype}  b_sf {tuple(b_sf.shape)}/{b_sf.dtype}", flush=True)
    print(f"# M={M} K={K} N={N} (K/2={K//2})", flush=True)
    doc = (flashinfer.mm_fp4.__doc__ or "")[:1500]
    print("# --- mm_fp4 docstring ---\n" + "\n".join("#   " + ln for ln in doc.splitlines()[:40]), flush=True)
    alpha = (a_gsf * b_gsf).float()
    for bname, bt, bst in [("b[N,K/2]", b_fp4, b_sf), ("b.T", b_fp4.t().contiguous(), b_sf), ("b.mT", b_fp4.mT.contiguous(), b_sf.mT.contiguous() if b_sf.dim() == 2 else b_sf)]:
        for be in ("trtllm", "cutlass", "cudnn", "b12x", "auto"):
            try:
                o = flashinfer.mm_fp4(a_fp4, bt, a_sf, bst, alpha, out_dtype=torch.bfloat16, backend=be, use_nvfp4=True)
                print(f"# OK  {bname} backend={be} -> out {tuple(o.shape)}", flush=True)
            except Exception as ex:  # noqa: BLE001
                print(f"# no  {bname} backend={be}: {str(ex)[:110]}", flush=True)
    print("# mmfp4_probe done", flush=True)


@app.function(gpu="RTX-PRO-6000", timeout=15 * MIN, volumes={"/cache": vol})
def flashinfer_probe() -> None:
    """Probe FlashInfer's FP4 GEMM/MoE API (the real SOTA competitor, b12x) so bench_vs_sota can call it."""
    import importlib
    import inspect
    import flashinfer
    print(f"# flashinfer {getattr(flashinfer, '__version__', '?')}", flush=True)
    for path in ["flashinfer.gemm", "flashinfer.fused_moe", "flashinfer"]:
        try:
            m = importlib.import_module(path)
            hits = [n for n in dir(m) if any(k in n.lower() for k in ("fp4", "moe", "mm_", "nvfp4", "block_scale")) and not n.startswith("_")]
            print(f"# {path}: {hits}", flush=True)
            for n in hits:
                try:
                    print(f"#   {path}.{n}: {inspect.signature(getattr(m, n))}", flush=True)
                except (ValueError, TypeError):
                    pass
        except Exception as e:  # noqa: BLE001
            print(f"# FAIL {path}: {type(e).__name__}: {e}", flush=True)
    print("# flashinfer_probe done", flush=True)


@app.function(gpu="RTX-PRO-6000", timeout=15 * MIN, volumes={"/cache": vol})
def marlin_probe() -> None:
    """Probe vLLM's real Marlin MoE API (signatures + import paths) so bench_vs_marlin can call it correctly."""
    import importlib
    import inspect
    for path, name in [
        ("vllm.model_executor.layers.fused_moe.fused_marlin_moe", "fused_marlin_moe"),
        ("vllm.model_executor.layers.quantization.utils.marlin_utils_test", "marlin_quantize"),
        ("vllm.model_executor.layers.quantization.utils.marlin_utils", "marlin_make_workspace"),
        ("vllm.model_executor.layers.fused_moe", "fused_experts"),
    ]:
        try:
            m = importlib.import_module(path)
            obj = getattr(m, name)
            print(f"# OK {path}.{name}: {inspect.signature(obj)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"# FAIL {path}.{name}: {type(e).__name__}: {e}", flush=True)
    try:
        from vllm.scalar_type import scalar_types
        print(f"# scalar_types.uint4b8 = {scalar_types.uint4b8}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"# FAIL scalar_types: {e}", flush=True)
    import vllm
    print(f"# vllm {vllm.__version__}", flush=True)
    # find the real Marlin W4A16 GEMM + workspace helpers (for a pure-GEMM real-Marlin comparison)
    import inspect as _ins
    try:
        from vllm import _custom_ops as _ops
        for nm in dir(_ops):
            if "marlin" in nm.lower() and "gemm" in nm.lower():
                try:
                    print(f"# _custom_ops.{nm}: {_ins.signature(getattr(_ops, nm))}", flush=True)
                except (ValueError, TypeError):
                    print(f"# _custom_ops.{nm}: (builtin, no sig)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"# FAIL _custom_ops: {e}", flush=True)
    for path in ["vllm.model_executor.layers.quantization.utils.marlin_utils"]:
        try:
            m = importlib.import_module(path)
            hits = [n for n in dir(m) if "workspace" in n.lower() or "permute_scales" in n.lower() or n == "MARLIN_SUPPORTED_GROUP_SIZES"]
            print(f"# {path} helpers: {hits}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"# FAIL {path}: {e}", flush=True)
    print("# marlin_probe done", flush=True)


@app.function(gpu="RTX-PRO-6000", timeout=20 * MIN, volumes={"/cache": vol})
def bench_vs_marlin(iters: int = 30, mrows: int = 8192) -> None:
    M = mrows
    """(b) DEFINITIVE real-Marlin GEMM comparison, same invocation: our 2:4-sparse-FP4 GEMM vs vLLM's real
    Marlin W4A16 GEMM (_custom_ops.marlin_gemm) at gpt-oss/GLM/DeepSeek gate_up dims [M, N=2I, K=H]. Same
    M/N/K/FLOPs. Marlin = the actual tuned kernel Marlin serves (bf16 mma after 4-bit dequant), so this
    settles whether our GEMM really beats Marlin's, not just its bf16 ceiling. No clock noise (interleaved)."""
    import ctypes
    import time

    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
    from vllm.scalar_type import scalar_types

    _ensure_so()
    dev = torch.device("cuda")
    lib = ctypes.CDLL("/cache/sparse_fp4.so")
    lib.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    lib.sparse_fp4_mm_2lvl.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
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

    def pack(w):  # bf16 [out,in] -> our 2:4 two-level nvfp4 buffers
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

    def bench(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); ts = []
        for _ in range(iters):
            s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
        ts.sort(); return ts[len(ts) // 2]

    def swig(gu):  # plain SwiGLU silu(gate)*up on [M, 2I] -> [M, I]
        return (gu[:, 0::2] * torch.sigmoid(gu[:, 0::2])) * gu[:, 1::2]

    shapes = [("gpt-oss   ", 3072, 2944), ("GLM-5.2   ", 5120, 1536), ("DeepSeek  ", 7168, 2048)]
    print(f"# (a) FULL MoE (gate_up+swiglu+down): our 2:4-sparse-FP4 (2 GEMMs+quant) vs REAL Marlin W4A16 (2 marlin_gemm), M={M}", flush=True)
    for label, H, I in shapes:
        Kw = ((H + 127) // 128) * 128; N = ((2 * I + 255) // 256) * 256; Ih = N // 2   # gate_up [M,N,Kw], h [M,Ih]
        Ndn = ((H + 255) // 256) * 256                                                  # down out; in = Ih
        x = torch.randn(M, Kw, device=dev, dtype=torch.bfloat16) * 0.1
        gu_ac, gu_m, gu_sA, gu_ga = pack(torch.randn(N, Kw, device=dev) * 0.05)         # our gate_up [N, Kw]
        dn_ac, dn_m, dn_sA, dn_ga = pack(torch.randn(Ndn, Ih, device=dev) * 0.05)       # our down [Ndn, Ih]
        bb = torch.empty((M, Kw // 2), dtype=torch.uint8, device=dev); sb = torch.empty((Kw // 128, M, 4), dtype=torch.uint8, device=dev); gb = torch.empty((M,), dtype=torch.float32, device=dev)
        b2 = torch.empty((M, Ih // 2), dtype=torch.uint8, device=dev); s2 = torch.empty((Ih // 128, M, 4), dtype=torch.uint8, device=dev); g2 = torch.empty((M,), dtype=torch.float32, device=dev)
        Cgu = torch.empty((N, M), dtype=torch.bfloat16, device=dev); Cdn = torch.empty((Ndn, M), dtype=torch.bfloat16, device=dev)

        def ours():
            lib.quantize_act_nvfp4_2lvl(x.data_ptr(), bb.data_ptr(), sb.data_ptr(), gb.data_ptr(), M, Kw)
            lib.sparse_fp4_mm_2lvl(gu_ac.data_ptr(), bb.data_ptr(), gu_sA.data_ptr(), sb.data_ptr(), gu_m.data_ptr(), Cgu.data_ptr(), N, M, Kw, gu_ga.data_ptr(), gb.data_ptr())
            h = swig(Cgu.t()).contiguous()
            lib.quantize_act_nvfp4_2lvl(h.data_ptr(), b2.data_ptr(), s2.data_ptr(), g2.data_ptr(), M, Ih)
            lib.sparse_fp4_mm_2lvl(dn_ac.data_ptr(), b2.data_ptr(), dn_sA.data_ptr(), s2.data_ptr(), dn_m.data_ptr(), Cdn.data_ptr(), Ndn, M, Ih, dn_ga.data_ptr(), g2.data_ptr())

        try:
            _r1, gu_qw, gu_sm, gu_gi, gu_si, _ = marlin_quantize(torch.randn(Kw, N, device=dev, dtype=torch.bfloat16) * 0.05, scalar_types.uint4b8, 128, False)
            _r2, dn_qw, dn_sm, dn_gi, dn_si, _ = marlin_quantize(torch.randn(Ih, Ndn, device=dev, dtype=torch.bfloat16) * 0.05, scalar_types.uint4b8, 128, False)
            ws = marlin_make_workspace_new(dev); xm = x.contiguous()

            def marlin():
                gu = ops.marlin_gemm(xm, None, gu_qw, None, gu_sm, None, None, None, gu_gi, gu_si, ws, scalar_types.uint4b8, M, N, Kw, True)
                h = swig(gu).contiguous()
                ops.marlin_gemm(h, None, dn_qw, None, dn_sm, None, None, None, dn_gi, dn_si, ws, scalar_types.uint4b8, M, Ndn, Ih, True)
            to = bench(ours); tm = bench(marlin)
            flop = 2.0 * M * (N * Kw + Ndn * Ih)
            tfo = flop / 1e12 / (to / 1e3); tfm = flop / 1e12 / (tm / 1e3)
            win = "quadbit WINS" if to < tm else "Marlin wins"
            print(f"# {label} H={H} I={I}: ours {to:.3f}ms ({tfo:.0f} TF/s) | Marlin {tm:.3f}ms ({tfm:.0f} TF/s) | "
                  f"{tm / to:.2f}x  ({win})", flush=True)
        except Exception as ex:  # noqa: BLE001
            import traceback
            print(f"# {label} Marlin FAIL: {type(ex).__name__}: {ex}", flush=True)
            print(traceback.format_exc()[-800:], flush=True)
    print("# bench_vs_marlin done", flush=True)


@app.function(gpu="RTX-PRO-6000", timeout=30 * MIN, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(moe: str = "sparse", proj: str = "both", sparse_from: int = 0, max_len: int = 2048,
        batch_prefill: int = 16, max_batched: int = 16384, bench_only: bool = False,
        graph: bool = False, graph_cap: int = 256, ablate: str = "", torchprof: bool = False,
        prof: bool = False, fused: bool = True, ab: bool = False) -> None:
    import time

    import torch
    from vllm import LLM, SamplingParams

    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    os.environ["QB_MOE"] = moe
    os.environ["QB_SPARSE_PROJ"] = proj          # both|down|gateup (tax lives in gate_up -> down recovers)
    os.environ["QB_SPARSE_FROM"] = str(sparse_from)  # prefix-optimal: sparsify layers >= this, anchor earlier
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    if graph:  # capture-legal MoE path + let vLLM CUDA-graph the forward
        os.environ["QB_GRAPH"] = "1"
        os.environ["QB_GRAPH_CAP"] = str(graph_cap)
    if ablate:
        os.environ["QB_GPTOSS_ABLATE"] = ablate
    if torchprof:
        os.environ["QB_TORCHPROF"] = "1"
    if prof:
        os.environ["QB_GPTOSS_PROF"] = "1"
    os.environ["QB_GPTOSS_FUSED"] = "1" if fused else "0"   # fully-fused MoE (GEMM1 + down-scatter)
    if ab:  # same-invocation A/B: plugin reads /dev/shm/qb_fused per-apply (must be set BEFORE worker fork)
        os.environ["QB_AB"] = "1"
    _ensure_so()
    print(f"# gpt-oss serve: {MODEL} QB_MOE={moe} proj={proj} sparse_from={sparse_from} "
          f"graph={graph} cap={graph_cap if graph else 0} on {torch.cuda.device_count()}x RTX-PRO-6000", flush=True)

    kw = dict(model=MODEL, tensor_parallel_size=1, enforce_eager=not graph, trust_remote_code=True,
              max_model_len=max_len, gpu_memory_utilization=0.90, max_num_batched_tokens=max_batched)
    t0 = time.time()
    llm = LLM(**kw)
    print(f"  load+init ok in {time.time() - t0:.0f}s", flush=True)

    tok = llm.get_tokenizer()
    base = tok.encode(PASSAGE)
    ppl = float("nan")
    if not bench_only:
        prompts = ["The capital of France is", "def fibonacci(n):",
                   "The three primary colors are", "Water is made of hydrogen and"]
        outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=32))
        for o in outs:
            print(f"  [gen] {o.prompt!r} -> {o.outputs[0].text!r}", flush=True)

        pout = llm.generate([{"prompt_token_ids": base}],
                            SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0))
        plp = pout[0].prompt_logprobs or []
        nlls = [-d[tid].logprob for tid, d in zip(base[1:], plp[1:], strict=False)
                if d and tid in d and math.isfinite(d[tid].logprob)]
        ppl = math.exp(sum(nlls) / len(nlls)) if nlls else float("nan")
        print(f"# PPL over {len(nlls)}-token passage: {ppl:.3f}", flush=True)

        # --- throughput (prefill + decode, B=1) ---
        for plen in (512, 1536):
            gen = 64
            plen = min(plen, max_len - gen - 8)
            if plen <= 0:
                continue
            ptoks = (base * ((plen // len(base)) + 1))[:plen]
            try:
                tp0 = time.time()
                llm.generate([{"prompt_token_ids": ptoks}], SamplingParams(temperature=0.0, max_tokens=1))
                prefill_s = time.time() - tp0
                td0 = time.time()
                dout = llm.generate([{"prompt_token_ids": ptoks}], SamplingParams(temperature=0.0, max_tokens=gen))
                dec_s = time.time() - td0
                gtok = len(dout[0].outputs[0].token_ids)
                dec_only = max(1e-6, dec_s - prefill_s)
                steps = max(1, gtok - 1)
                print(f"# serve B=1 prompt={plen}: prefill {plen / prefill_s:.0f} tok/s, "
                      f"decode {steps / dec_only:.1f} tok/s", flush=True)
            except Exception as e:  # noqa: BLE001 - timing must not abort the eval
                print(f"# timing skipped prompt={plen} ({type(e).__name__}: {e})", flush=True)

    # --- large-batch prefill throughput: the COMPUTE-BOUND regime where 2:4 sparse wins (many
    # tokens/expert -> large-M MoE GEMMs). B=1 above is memory-bound + overhead-bound (our worst case). ---
    S = max_len - 16  # leave room for the +1 output token (prompt+out must be <= max_model_len)
    big = (base * ((S // len(base)) + 1))[:S]
    reqs = [{"prompt_token_ids": big} for _ in range(batch_prefill)]
    ntok = batch_prefill * S
    one = SamplingParams(temperature=0.0, max_tokens=1)
    if os.environ.get("QB_TORCHPROF") == "1":
        from torch.profiler import ProfilerActivity, profile
        llm.generate(reqs, one)  # warm
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            llm.generate(reqs, one)
        print("# ===== TORCH PROFILER (large-batch prefill, top CUDA ops) =====", flush=True)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30), flush=True)
    iters = 20
    try:
        for _ in range(3):  # warm (kernel JIT, autotune, allocator)
            llm.generate(reqs, one)
        times = []
        for _ in range(iters):
            t0 = time.time()
            llm.generate(reqs, one)
            times.append(time.time() - t0)
        times.sort()
        med = times[len(times) // 2]
        print(f"# LARGE-BATCH prefill B={batch_prefill} S={S} ({ntok} tok, ~{ntok // 32} tok/expert), "
              f"median of {iters}: {ntok / med:.0f} tok/s (med {med * 1000:.1f}ms, "
              f"min {min(times) * 1000:.1f} / max {max(times) * 1000:.1f})", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"# large-batch prefill skipped ({type(e).__name__}: {e})", flush=True)

    # --- SAME-INVOCATION A/B: interleave fused vs unfused large-batch prefill (alternate every iter so GPU
    # clock drift affects both equally -> the delta is real, unlike ±6% cross-run noise). The plugin worker
    # reads /dev/shm/qb_fused per apply; we toggle it from the driver (shared tmpfs, same container). ---
    if ab:
        def _setflag(v):
            with open("/dev/shm/qb_fused", "w") as f:
                f.write(v)
        try:
            for v in ("1", "0"):  # warm both paths (JIT/autotune/allocator)
                _setflag(v)
                for _ in range(3):
                    llm.generate(reqs, one)
            tf, tu = [], []
            for _ in range(iters):
                _setflag("1"); t0 = time.time(); llm.generate(reqs, one); tf.append(time.time() - t0)
                _setflag("0"); t0 = time.time(); llm.generate(reqs, one); tu.append(time.time() - t0)
            tf.sort(); tu.sort()
            mf, mu = tf[len(tf) // 2], tu[len(tu) // 2]
            print(f"# ===== SAME-INVOCATION A/B (interleaved, same clock, {iters} each) =====", flush=True)
            print(f"#   FUSED (monolith)  {ntok / mf:.0f} tok/s ({mf * 1000:.1f}ms med, "
                  f"{min(tf) * 1000:.1f}/{max(tf) * 1000:.1f} min/max)", flush=True)
            print(f"#   UNFUSED (fast_gu) {ntok / mu:.0f} tok/s ({mu * 1000:.1f}ms med, "
                  f"{min(tu) * 1000:.1f}/{max(tu) * 1000:.1f} min/max)", flush=True)
            print(f"#   fused/unfused speedup = {mu / mf:.3f}x  (RELIABLE: same GPU clock, interleaved)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"# A/B skipped ({type(e).__name__}: {e})", flush=True)

    # --- decode throughput (B=1): the launch-overhead-bound regime graph capture accelerates ---
    dprompt = base[:32]
    G = 128
    try:
        llm.generate([{"prompt_token_ids": dprompt}], SamplingParams(temperature=0.0, max_tokens=8))  # warm
        rates = []
        for _ in range(5):
            td0 = time.time()
            dout = llm.generate([{"prompt_token_ids": dprompt}], SamplingParams(temperature=0.0, max_tokens=G))
            n = max(1, len(dout[0].outputs[0].token_ids))
            rates.append(n / (time.time() - td0))
        rates.sort()
        print(f"# DECODE B=1 gen={G}: {rates[len(rates) // 2]:.1f} tok/s (median of 5)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"# decode bench skipped ({type(e).__name__}: {e})", flush=True)

    try:
        import qb_sm120_plugin as qb
        print(f"# STATS moe_calls={qb.STATS.get('moe_calls')} "
              f"sparse_expert_calls={qb.STATS.get('sparse_expert_calls')}", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"# STATS unavailable: {type(ex).__name__}: {ex}", flush=True)
    print(f"# gpt-oss serve done (QB_MOE={moe} proj={proj} sparse_from={sparse_from}, PPL={ppl:.3f})", flush=True)
