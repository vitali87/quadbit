"""THE gating head-to-head: our 2:4-sparse FP4 kernel vs CUTLASS's own sparse NVFP4
SM120 example (80b), same card, same container, same effective-FLOP convention.

Until this runs, "fastest sparse FP4" is a roofline claim, not a win. This settles it
on a number. One container builds CUTLASS 80b AND compiles our sparse_fp4_lib.cu, then
times both on identical M=N=K shapes with the same cudaEvent method and the same
effective-FLOP count (2*M*N*K, what NVIDIA reports for 2:4).

CUTLASS is cloned/built once into a Volume (build80/ dir, separate from 79b) and cached.

Run:  uv run modal run harness/cutlass_sparse.py
"""

import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
CUTLASS_REPO = "https://github.com/NVIDIA/cutlass"
EX_DIR = "80_blackwell_geforce_sparse_gemm"
EXAMPLE = "80b_blackwell_geforce_nvfp4_nvfp4_sparse_gemm"
BUILD = "/cache/cutlass/build80"
EXE = f"{BUILD}/examples/{EX_DIR}/{EXAMPLE}"
SIZES = [4096, 8192, 16384]

# 12.8.1: the toolchain where OUR sparse_fp4_lib assembles (12.9.1 ptxas rejects the
# block-scaled sparse mma on sm_120). The cached 80b cubin (built under 12.9.1) runs fine
# here -- sm_120a cubins are driver-compatible across 12.x runtimes.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "cmake", "build-essential", "ninja-build")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64"})
    .pip_install("torch", index_url="https://download.pytorch.org/whl/nightly/cu128", pre=True)
    .add_local_dir((ROOT / "cuda").as_posix(), "/root/cuda")
)

cache = modal.Volume.from_name("quadbit-cutlass-cache", create_if_missing=True)
app = modal.App("quadbit-cutlass-sparse", image=image)


def _run(cmd: list[str], cwd: str) -> int:
    print(f"$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    out = r.stdout + r.stderr
    print(out[-12000:] if len(out) > 12000 else out, flush=True)
    return r.returncode


@app.function(gpu="RTX-PRO-6000", timeout=3600, volumes={"/cache": cache})
def run() -> None:
    import ctypes
    import os
    import re

    import torch

    _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], "/")

    if not os.path.exists("/cache/cutlass"):
        _run(["git", "clone", "--depth", "1", CUTLASS_REPO, "/cache/cutlass"], "/cache")

    if not os.path.exists(EXE):
        os.makedirs(BUILD, exist_ok=True)
        if not os.path.exists(f"{BUILD}/build.ninja"):
            cfg = _run(["cmake", "-G", "Ninja", "..",
                        "-DCUTLASS_NVCC_ARCHS=120a", "-DCUTLASS_ENABLE_TESTS=OFF",
                        "-DCUTLASS_ENABLE_LIBRARY=OFF", "-DCUTLASS_ENABLE_PROFILER=OFF",
                        "-DCMAKE_BUILD_TYPE=Release"], BUILD)
            if cfg != 0:
                print(">>> cmake configure failed", flush=True); return
        if _run(["cmake", "--build", ".", "--target", EXAMPLE, "-j", "8"], BUILD) != 0:
            print(">>> build failed", flush=True); return
        cache.commit()

    # our sparse kernel, same container
    so = "/root/sp.so"
    c = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
                        "-o", so, "/root/cuda/sparse_fp4_lib.cu", "-lcuda"], capture_output=True, text=True)
    if c.returncode != 0:
        print(c.stderr, flush=True); return
    sp = ctypes.CDLL(so)
    sp.sparse_fp4_mm.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3

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

    print(f"\n{'M=N=K':>7} | {'CUTLASS 80b sparse':>26} | {'quadbit sparse (ours)':>26} | {'speedup':>8}", flush=True)
    print("-" * 78, flush=True)
    for sz in SIZES:
        M = N = K = sz
        flop = 2.0 * M * N * K  # effective-FLOP (dense-equivalent), same as our bench

        # CUTLASS 80b: run WITH its built-in reference verification (the #3096 gate -- SM120
        # block-scaled path has documented correctness failures; a TFLOP/s from a kernel that
        # emits garbage is void). Capture the Passed/Failed disposition, then parse runtime.
        r = subprocess.run([EXE, f"--m={sz}", f"--n={sz}", f"--k={sz}", "--iterations=100"],
                           capture_output=True, text=True)
        cut = r.stdout + r.stderr
        disp = ("PASS" if re.search(r"Disposition:\s*Passed", cut)
                else "FAIL" if re.search(r"Disposition:\s*Failed", cut) else "NO-VERIFY")
        mm = re.search(r"([0-9.]+)\s*ms", cut)
        gf = re.search(r"([0-9.]+)\s*GFLOP", cut)
        cut_ms = float(mm.group(1)) if mm else float("nan")
        cut_tf = flop / (cut_ms / 1e3) / 1e12 if mm else float("nan")
        if disp != "PASS":  # void the perf number; correctness is the gate
            cut_tf = float("nan")

        # ours: same cudaEvent timing, dummy packed operands (perf is data-independent)
        ks = K // 128
        Ac = torch.full((M, K // 4), 0x22, dtype=torch.uint8, device="cuda")
        Bs = torch.full((N, K // 2), 0x22, dtype=torch.uint8, device="cuda")
        sA = torch.full((ks, M, 4), 0x38, dtype=torch.uint8, device="cuda")
        sB = torch.full((ks, N, 4), 0x38, dtype=torch.uint8, device="cuda")
        mt = torch.full((ks, M, 2), 0x44444444, dtype=torch.int32, device="cuda")
        Cs = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
        our_ms = tms(lambda: sp.sparse_fp4_mm(Ac.data_ptr(), Bs.data_ptr(), sA.data_ptr(),
                                              sB.data_ptr(), mt.data_ptr(), Cs.data_ptr(), M, N, K))
        our_tf = flop / (our_ms / 1e3) / 1e12
        sp_ratio = our_tf / cut_tf if cut_tf == cut_tf else float("nan")
        print(f"{sz:>7} | [{disp:^8}] {cut_ms:7.3f}ms {cut_tf:7.0f}TF/s | {our_ms:7.3f}ms {our_tf:7.0f}TF/s | "
              f"{sp_ratio:6.2f}x", flush=True)
        if gf:
            print(f"        (CUTLASS self-reported: {gf.group(1)} GFLOP/s -- for FLOP-convention cross-check)",
                  flush=True)


@app.local_entrypoint()
def main() -> None:
    call = run.spawn()
    print(f"SPAWN_ID {call.object_id}", flush=True)
    call.get()
