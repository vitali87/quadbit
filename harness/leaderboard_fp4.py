"""P0: SM120 FP4 DENSE backend leaderboard -- quadbit vs the real competitor set.

Not just CUTLASS-79b. Benchmarks the deployed quadbit two-level NVFP4 dense kernel
(dense_nvfp4_mm) head-to-head against FlashInfer's mm_fp4 across every backend that
claims SM120 support (auto / b12x / cutlass / cudnn) on the SAME RTX PRO 6000, same
CUDA build, same shape suite, same correctness gate. trtllm and cute-dsl are excluded
by FlashInfer itself (trtllm skips SM110/120/121; cute-dsl is SM100/103 only).

CUDA 13 image on purpose: FlashInfer's fastest SM120 NVFP4 path, backend="b12x", requires
CUDA 13+ (flashinfer test gate). cudnn needs cuDNN >= 9.14 or it raises a version error --
if it does, that IS the row (captured as an artifact, not hidden).

Correctness gate (identical for every backend, quadbit included): each backend quantizes
the SAME bf16 A,B with its own native quantizer and its output is scored against the fp32
reference A @ B^T -- cosine sim, max rel error, mean abs error. cos > 0.97 = PASS, else the
throughput is voided (a fast-but-wrong kernel -- e.g. FlashInfer CUTLASS silently returning
zeros on SM120, issue #2577 -- does not get a number).

GEMM-only timing (activation quantization NOT included) is the apples-to-apples column, since
FlashInfer takes pre-quantized operands too; quadbit's full (quant+kernel) time is also shown.

Run:  uv run modal run harness/leaderboard_fp4.py
"""

import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent

# CUDA 13 devel: b12x needs CUDA 13+. flashinfer JIT-compiles the SM120 b12x/cutlass kernels
# in-container. NOTE: flashinfer pins torch cu128 (2.9.1), so the effective toolkit for its
# kernels is CUDA 13 ptxas + torch-bundled cu12 runtime -- fine for flashinfer, which JITs.
fi_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.12")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64"})
    .pip_install("torch", index_url="https://download.pytorch.org/whl/nightly/cu130", pre=True)
    .pip_install("flashinfer-python==0.6.13", "flashinfer-cubin==0.6.13")
    .add_local_dir((ROOT / "cuda").as_posix(), "/root/cuda")
)

# quadbit's sm_120a block-scale mma (kind::mxf4nvf4 / block_scale / scale_vec::4X) is REJECTED
# by ptxas 13.0 ("not supported on .target sm_120") -- it only assembles under CUDA <= 12.8.
# So quadbit runs in its own 12.8.1 container; b12x + quadbit cannot coexist (a finding itself).
qb_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64"})
    .pip_install("torch", index_url="https://download.pytorch.org/whl/nightly/cu128", pre=True)
    .add_local_dir((ROOT / "cuda").as_posix(), "/root/cuda")
)
app = modal.App("quadbit-leaderboard", image=fi_image)

# (tag, M, N, K).  N = out-features, K = in-features, M = tokens.  flop = 2*M*N*K.
# Llama-3-8B: hidden 4096, FFN intermediate 14336.
_LLAMA = (("attn", 4096, 4096), ("ffn_up", 14336, 4096), ("ffn_dn", 4096, 14336))  # (N,K)


def _shapes() -> list[tuple[str, int, int, int]]:
    sh: list[tuple[str, int, int, int]] = []
    for sz in (2048, 4096, 8192, 16384):  # square
        sh.append((f"sq{sz}", sz, sz, sz))
    for name, n, k in _LLAMA:  # prefill (M=4096) rectangular
        sh.append((f"prefill_{name}", 4096, n, k))
    for m in (1, 8, 16, 32, 64, 128):  # decode / small-M
        for name, n, k in _LLAMA:
            sh.append((f"dec{m}_{name}", m, n, k))
    for b in (1, 8, 32, 64):  # real serving M = B*S, S=2048
        for name, n, k in _LLAMA:
            sh.append((f"serve_b{b}_{name}", b * 2048, n, k))
    return sh


HDR = (f"{'shape':>18} {'M':>6} {'N':>6} {'K':>6} | {'backend':>10} {'gate':>5} "
       f"{'ms':>9} {'TF/s':>7} {'GB/s':>7} {'cos':>7} {'maxrel':>9} {'mae':>9}")


def _bytes_moved(M, N, K):  # est: fp4 operands + ue4m3 scales + bf16 out
    return (M * K + N * K) / 2 + (M + N) * K / 16 + M * N * 2


def _helpers(torch):
    """(tms, score, emit) closures -- identical timing/scoring/formatting for every backend."""
    def tms(fn, it=30):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(it):
            fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / it

    def score(y, ref):  # y,ref: [M,N] fp32 on device
        import torch.nn.functional as F
        cos = F.cosine_similarity(y.reshape(-1), ref.reshape(-1), dim=0).item()
        maxrel = ((y - ref).abs() / (ref.abs() + 1.0)).max().item()
        mae = (y - ref).abs().mean().item()
        return cos, maxrel, mae

    def emit(tag, M, N, K, backend, gate, ms, cos, maxrel, mae):
        tf = (2.0 * M * N * K / (ms / 1e3) / 1e12) if (ms and gate == "PASS") else float("nan")
        gbs = (_bytes_moved(M, N, K) / (ms / 1e3) / 1e9) if (ms and gate == "PASS") else float("nan")
        print(f"{tag:>18} {M:>6} {N:>6} {K:>6} | {backend:>10} {gate:>5} "
              f"{(ms if ms else float('nan')):>9.4f} {tf:>7.0f} {gbs:>7.0f} "
              f"{cos:>7.4f} {maxrel:>9.4f} {mae:>9.5f}", flush=True)
    return tms, score, emit


def _env_header(torch, commit, extra=""):
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    cap = torch.cuda.get_device_capability(0)
    print(f"commit {commit} | {smi} | sm_{cap[0]}{cap[1]} | torch {torch.__version__} "
          f"cuda {torch.version.cuda} | cudnn {torch.backends.cudnn.version()} | {extra} | {nvcc}", flush=True)


@app.function(gpu="RTX-PRO-6000", timeout=5400, image=fi_image)
def run_flashinfer(commit: str = "?") -> None:
    import torch

    dev = torch.device("cuda")
    try:
        import flashinfer
        fi_ver = getattr(flashinfer, "__version__", "?")
    except Exception as e:  # noqa: BLE001
        fi_ver = f"IMPORT FAILED: {e}"
    print("=== SM120 FP4 dense leaderboard :: FlashInfer mm_fp4 ===", flush=True)
    _env_header(torch, commit, f"flashinfer {fi_ver}")
    tms, score, emit = _helpers(torch)

    from flashinfer import SfLayout, mm_fp4, nvfp4_quantize
    BACKENDS = ["auto", "b12x", "cutlass", "cudnn"]  # trtllm/cute-dsl excluded by flashinfer on SM120

    print(HDR, flush=True)
    print("-" * len(HDR), flush=True)
    for tag, M, N, K in _shapes():
        torch.manual_seed(0)
        A = torch.randn(M, K, device=dev, dtype=torch.bfloat16)   # activations [tokens, in]
        B = torch.randn(N, K, device=dev, dtype=torch.bfloat16)   # weight [out, in]
        ref = A.float() @ B.float().T                             # [M, N] fp32 reference

        gsa = (448 * 6) / A.float().abs().nan_to_num().max()
        gsb = (448 * 6) / B.float().abs().nan_to_num().max()
        try:
            a_fp4, a_s = nvfp4_quantize(A, gsa, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
            b_fp4, b_s = nvfp4_quantize(B, gsb, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
            alpha = 1.0 / (gsa * gsb)
        except Exception as e:  # noqa: BLE001
            for bk in BACKENDS:
                print(f"{tag:>18} {M:>6} {N:>6} {K:>6} | {bk:>10}  N/A  quantize failed: "
                      f"{str(e)[:60]}", flush=True)
            continue
        for bk in BACKENDS:
            out = torch.empty((M, N), device=dev, dtype=torch.bfloat16)

            def _f():
                mm_fp4(a_fp4, b_fp4.T, a_s, b_s.T, alpha, torch.bfloat16, out,
                       block_size=16, use_8x4_sf_layout=False, backend=bk, use_nvfp4=True)
            try:
                _f()
                cos, maxrel, mae = score(out.float(), ref)
                gate = "PASS" if cos > 0.97 else "FAIL"
                emit(tag, M, N, K, bk, gate, tms(_f), cos, maxrel, mae)
            except Exception as e:  # noqa: BLE001  -- broken backend IS the result
                print(f"{tag:>18} {M:>6} {N:>6} {K:>6} | {bk:>10}  N/A  {str(e)[:80]}", flush=True)


@app.function(gpu="RTX-PRO-6000", timeout=5400, image=qb_image)
def run_quadbit(commit: str = "?") -> None:
    import ctypes

    import torch

    dev = torch.device("cuda")
    print("=== SM120 FP4 dense leaderboard :: quadbit deployed two-level NVFP4 kernel ===", flush=True)
    _env_header(torch, commit, "quadbit dense_nvfp4_fast_lib")
    tms, score, emit = _helpers(torch)

    so = "/root/dn.so"
    c = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
                        "-o", so, "/root/cuda/dense_nvfp4_fast_lib.cu", "-lcuda"],
                       capture_output=True, text=True)
    if c.returncode != 0:
        print(f"!! quadbit nvcc FAILED under CUDA {torch.version.cuda}:\n{c.stderr[-2000:]}", flush=True)
        return
    qb = ctypes.CDLL(so)
    qb.dense_nvfp4_mm.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    qb.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    qb.quantize_act_nvfp4_2lvl.restype = None

    # two-level pack (weight side), mirrors dense_e2e.DenseNVLinear exactly
    FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
    BND = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], device=dev)
    _c = torch.arange(128, device=dev); _e, _m = (_c >> 3) & 0xf, _c & 7
    UE4M3 = torch.where(_e == 0, _m.float() * 0.001953125, (1.0 + _m.float() / 8.0) * torch.exp2((_e - 7).float()))

    def quant_fp4(v):
        return torch.bucketize(v.abs(), BND) | ((v < 0).long() << 3)

    def enc_ue4m3(s):
        mf, e = torch.frexp(s.clamp_min(1e-30)); mant = torch.round((2.0 * mf - 1.0) * 8.0).long()
        biased = (e - 1) + 7; carry = mant == 8
        mant = torch.where(carry, torch.zeros_like(mant), mant); biased = torch.where(carry, biased + 1, biased)
        code = (biased << 3) | mant
        code = torch.where(biased < 1, torch.ones_like(code), code)
        return torch.where((biased > 15) | (s >= 480.0), torch.full_like(code, 0x7f),
                           torch.where(s > 0, code, torch.zeros_like(code)))

    def nvfp4_pack(W):  # W[out,in] -> Ab[out,in/2], SFA[step,out,8] ue4m3, gA[out] fp32
        out, inn = W.shape; Wf = W.float()
        gA = (Wf.abs().amax(-1) / 2688.0).clamp(min=1e-30)
        Wb = Wf.view(out, inn // 16, 16)
        scode = enc_ue4m3((Wb.abs().amax(-1) / 6.0) / gA[:, None])
        sdeq = UE4M3[scode.long()] * gA[:, None]
        q = quant_fp4(Wb / sdeq[..., None]).view(out, inn)
        Ab = (q[:, 0::2] | (q[:, 1::2] << 4)).to(torch.uint8).contiguous()
        SFA = scode.view(out, inn // 128, 8).permute(1, 0, 2).contiguous().to(torch.uint8)
        return Ab, SFA, gA.contiguous()

    print(HDR, flush=True)
    print("-" * len(HDR), flush=True)
    for tag, M, N, K in _shapes():
        # quadbit tile constraints: M(tokens)%256, N(out)%256, K(in)%256
        if not (M % 256 == 0 and N % 256 == 0 and K % 256 == 0):
            emit(tag, M, N, K, "quadbit", "SKIP", None, float("nan"), float("nan"), float("nan"))
            continue
        torch.manual_seed(0)
        A = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        B = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
        ref = A.float() @ B.float().T

        Ab, SFA, gA = nvfp4_pack(B)
        Bb = torch.empty((M, K // 2), dtype=torch.uint8, device=dev)
        SFB = torch.empty((K // 128, M, 8), dtype=torch.uint8, device=dev)
        gB = torch.empty((M,), dtype=torch.float32, device=dev)
        qb.quantize_act_nvfp4_2lvl(A.contiguous().data_ptr(), Bb.data_ptr(), SFB.data_ptr(),
                                   gB.data_ptr(), M, K)
        C = torch.empty((N, M), dtype=torch.bfloat16, device=dev)

        def _k():
            qb.dense_nvfp4_mm(Ab.data_ptr(), Bb.data_ptr(), SFA.data_ptr(), SFB.data_ptr(),
                              C.data_ptr(), N, M, K, gA.data_ptr(), gB.data_ptr())
        _k()
        cos, maxrel, mae = score(C.t().float(), ref)
        gate = "PASS" if cos > 0.97 else "FAIL"
        emit(tag, M, N, K, "quadbit", gate, tms(_k), cos, maxrel, mae)


@app.local_entrypoint()
def main(mode: str = "both") -> None:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    calls = []
    if mode in ("both", "flashinfer"):
        c = run_flashinfer.spawn(commit); print(f"SPAWN_ID flashinfer {c.object_id}", flush=True); calls.append(c)
    if mode in ("both", "quadbit"):
        c = run_quadbit.spawn(commit); print(f"SPAWN_ID quadbit {c.object_id}", flush=True); calls.append(c)
    for c in calls:
        c.get()
