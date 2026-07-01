"""#2 -- head-to-head across REAL LLM GEMM shapes, not just square: our dense FP4 vs CUTLASS
79b (dense NVFP4) and our 2:4-sparse FP4 vs CUTLASS 80b (sparse NVFP4), same card, correctness-
gated on each CUTLASS example's own reference check. Square (#1) said we win 4-8K / lose 16K;
this asks whether that holds on the rectangular prefill shapes real models actually run.

Reuses the cached CUTLASS build (quadbit-cutlass-cache: build/ = 79b, build80/ = 80b). 12.8.1
so our kernels assemble; the cached sm_120a cubins run under any 12.x runtime.

Run:  uv run modal run harness/cutlass_shapes.py
"""

import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
CUTLASS_REPO = "https://github.com/NVIDIA/cutlass"
DENSE_EX = "79b_blackwell_geforce_nvfp4_nvfp4_gemm"
SPARSE_EX = "80b_blackwell_geforce_nvfp4_nvfp4_sparse_gemm"
DENSE_EXE = f"/cache/cutlass/build/examples/79_blackwell_geforce_gemm/{DENSE_EX}"
SPARSE_EXE = f"/cache/cutlass/build80/examples/80_blackwell_geforce_sparse_gemm/{SPARSE_EX}"
# real Llama-3-8B prefill GEMM shapes (M=tokens): (name, M, N, K)
SHAPES = [
    ("attn qkv/o  4096^3", 4096, 4096, 4096),
    ("ffn up   N=14336", 4096, 14336, 4096),
    ("ffn down K=14336", 4096, 4096, 14336),
]

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "cmake", "build-essential", "ninja-build")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64"})
    .pip_install("torch", index_url="https://download.pytorch.org/whl/nightly/cu128", pre=True)
    .add_local_dir((ROOT / "cuda").as_posix(), "/root/cuda")
)
cache = modal.Volume.from_name("quadbit-cutlass-cache", create_if_missing=True)
app = modal.App("quadbit-cutlass-shapes", image=image)


def _sh(cmd: list[str], cwd: str) -> int:
    print(f"$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    print((r.stdout + r.stderr)[-8000:], flush=True)
    return r.returncode


def _build(exdir: str, target: str, builddir: str) -> None:
    import os
    if not os.path.exists(f"{builddir}/build.ninja"):
        os.makedirs(builddir, exist_ok=True)
        _sh(["cmake", "-G", "Ninja", "..", "-DCUTLASS_NVCC_ARCHS=120a",
             "-DCUTLASS_ENABLE_TESTS=OFF", "-DCUTLASS_ENABLE_LIBRARY=OFF",
             "-DCUTLASS_ENABLE_PROFILER=OFF", "-DCMAKE_BUILD_TYPE=Release"], builddir)
    _sh(["cmake", "--build", ".", "--target", target, "-j", "8"], builddir)
    cache.commit()


@app.function(gpu="RTX-PRO-6000", timeout=3600, volumes={"/cache": cache})
def run() -> None:
    import ctypes
    import os
    import re

    import torch

    if not os.path.exists("/cache/cutlass"):
        _sh(["git", "clone", "--depth", "1", CUTLASS_REPO, "/cache/cutlass"], "/cache")
    if not os.path.exists(DENSE_EXE):
        _build("79_blackwell_geforce_gemm", DENSE_EX, "/cache/cutlass/build")
    if not os.path.exists(SPARSE_EXE):
        _build("80_blackwell_geforce_sparse_gemm", SPARSE_EX, "/cache/cutlass/build80")

    libs = {}
    for name, src in (("de", "dense_fp4_lib"), ("sp", "sparse_fp4_lib")):
        so = f"/root/{name}.so"
        c = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
                            "-o", so, f"/root/cuda/{src}.cu", "-lcuda"], capture_output=True, text=True)
        if c.returncode != 0:
            print(c.stderr, flush=True); return
        libs[name] = ctypes.CDLL(so)
    libs["de"].dense_fp4_mm.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3
    libs["sp"].sparse_fp4_mm.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3

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

    def cutlass_tf(exe, M, N, K, flop):
        r = subprocess.run([exe, f"--m={M}", f"--n={N}", f"--k={K}", "--iterations=100"],
                           capture_output=True, text=True)
        out = r.stdout + r.stderr
        ok = bool(re.search(r"Disposition:\s*Passed", out))
        mm = re.search(r"([0-9.]+)\s*ms", out)
        return (flop / (float(mm.group(1)) / 1e3) / 1e12 if (ok and mm) else float("nan")), ok

    def ours_dense(M, N, K):
        A = torch.full((M, K // 2), 0x22, dtype=torch.uint8, device="cuda")
        B = torch.full((N, K // 2), 0x22, dtype=torch.uint8, device="cuda")
        C = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
        return tms(lambda: libs["de"].dense_fp4_mm(A.data_ptr(), B.data_ptr(), C.data_ptr(), M, N, K))

    def ours_sparse(M, N, K):
        ks = K // 128
        Ac = torch.full((M, K // 4), 0x22, dtype=torch.uint8, device="cuda")
        Bs = torch.full((N, K // 2), 0x22, dtype=torch.uint8, device="cuda")
        sA = torch.full((ks, M, 4), 0x38, dtype=torch.uint8, device="cuda")
        sB = torch.full((ks, N, 4), 0x38, dtype=torch.uint8, device="cuda")
        mt = torch.full((ks, M, 2), 0x44444444, dtype=torch.int32, device="cuda")
        C = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
        return tms(lambda: libs["sp"].sparse_fp4_mm(Ac.data_ptr(), Bs.data_ptr(), sA.data_ptr(),
                                                    sB.data_ptr(), mt.data_ptr(), C.data_ptr(), M, N, K))

    print(f"\n{'shape':<20} {'79b dense':>12} {'ours dense':>13} {'':>6} | "
          f"{'80b sparse':>12} {'ours sparse':>13} {'':>6}", flush=True)
    print("-" * 92, flush=True)
    for name, M, N, K in SHAPES:
        flop = 2.0 * M * N * K
        c79, ok79 = cutlass_tf(DENSE_EXE, M, N, K, flop)
        c80, ok80 = cutlass_tf(SPARSE_EXE, M, N, K, flop)
        od = flop / (ours_dense(M, N, K) / 1e3) / 1e12
        os_ = flop / (ours_sparse(M, N, K) / 1e3) / 1e12
        rd = od / c79 if c79 == c79 else float("nan")
        rs = os_ / c80 if c80 == c80 else float("nan")
        d79 = "PASS" if ok79 else "FAIL"; d80 = "PASS" if ok80 else "FAIL"
        print(f"{name:<20} [{d79}]{c79:>7.0f}TF {od:>10.0f}TF {rd:>5.2f}x | "
              f"[{d80}]{c80:>7.0f}TF {os_:>10.0f}TF {rs:>5.2f}x", flush=True)


@app.local_entrypoint()
def main() -> None:
    run.remote()
