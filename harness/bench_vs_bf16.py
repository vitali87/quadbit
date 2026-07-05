"""Head-to-head on RTX PRO 6000: cuBLAS bf16 (what production actually runs) vs our dense FP4
vs our 2:4-sparse FP4 -- same shapes, same hardware, one run, real numbers. The honest
"are we winning" question isn't FP4-vs-CUTLASS-FP4 (a wash); it's FP4-vs-the-bf16-everyone-ships.

Run:  uv run modal run harness/bench_vs_bf16.py
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
app = modal.App("quadbit-bench", image=image)


@app.function(gpu="RTX-PRO-6000", timeout=1800)
def run() -> None:
    import ctypes

    import torch

    # DEPLOYED two-level kernels (match the shipped W4A4 path + the accuracy recipe), not the old
    # single-level binaries. Raw-kernel cudaEvent protocol (standalone, no torch dispatch).
    outs = {}
    for name, src in (("sp", "sparse_fp4_lib"), ("de", "dense_nvfp4_fast_lib")):
        so = f"/root/{name}.so"
        c = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
                            "-o", so, f"/root/cuda/{src}.cu", "-lcuda"], capture_output=True, text=True)
        if c.returncode != 0:
            print(c.stderr, flush=True); return
        outs[name] = ctypes.CDLL(so)
    sp, de = outs["sp"], outs["de"]
    sp.sparse_fp4_mm_2lvl.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    de.dense_nvfp4_mm.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2

    print(f"{torch.cuda.get_device_name(0)}\n", flush=True)

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

    print(f"{'M=N=K':>7} | {'cuBLAS bf16':>12} | {'dense 2lvl FP4 (ours)':>24} | {'2:4-sparse 2lvl FP4 (ours)':>28}", flush=True)
    print("-" * 80, flush=True)
    for sz in (4096, 8192, 16384):
        M = N = K = sz
        flop = 2.0 * M * N * K
        Ab, Bb = torch.randn(M, K, device="cuda", dtype=torch.bfloat16), torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        ms_bf16 = tms(lambda: torch.matmul(Ab, Bb.t()))

        # deployed dense two-level: per-16 ue4m3 local (SFA/SFB, 8 per 128-K) + per-row/col fp32 global
        ks = K // 128
        Ad = torch.full((M, K // 2), 0x22, dtype=torch.uint8, device="cuda")
        Bd = torch.full((N, K // 2), 0x22, dtype=torch.uint8, device="cuda")
        SFA = torch.full((ks, M, 8), 0x38, dtype=torch.uint8, device="cuda")
        SFB = torch.full((ks, N, 8), 0x38, dtype=torch.uint8, device="cuda")
        gAd = torch.ones(M, dtype=torch.float32, device="cuda")
        gBd = torch.ones(N, dtype=torch.float32, device="cuda")
        Cd = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
        ms_de = tms(lambda: de.dense_nvfp4_mm(Ad.data_ptr(), Bd.data_ptr(), SFA.data_ptr(), SFB.data_ptr(),
                                              Cd.data_ptr(), M, N, K, gAd.data_ptr(), gBd.data_ptr()))

        # deployed sparse two-level: 2:4 A (K/4 bytes) + metadata + per-32/per-16-kept local + fp32 global
        Ac = torch.full((M, K // 4), 0x22, dtype=torch.uint8, device="cuda")
        Bs = torch.full((N, K // 2), 0x22, dtype=torch.uint8, device="cuda")
        sA = torch.full((ks, M, 4), 0x38, dtype=torch.uint8, device="cuda")
        sB = torch.full((ks, N, 4), 0x38, dtype=torch.uint8, device="cuda")
        mt = torch.full((ks, M, 2), 0x44444444, dtype=torch.int32, device="cuda")
        gAs = torch.ones(M, dtype=torch.float32, device="cuda")
        gBs = torch.ones(N, dtype=torch.float32, device="cuda")
        Cs = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
        ms_sp = tms(lambda: sp.sparse_fp4_mm_2lvl(Ac.data_ptr(), Bs.data_ptr(), sA.data_ptr(),
                                                  sB.data_ptr(), mt.data_ptr(), Cs.data_ptr(), M, N, K,
                                                  gAs.data_ptr(), gBs.data_ptr()))

        def line(ms):
            return f"{ms:.3f}ms {flop/(ms/1e3)/1e12:6.0f}TF/s"
        print(f"{sz:>7} | {line(ms_bf16):>12} | {line(ms_de)} ({ms_bf16/ms_de:.2f}x) | "
              f"{line(ms_sp)} ({ms_bf16/ms_sp:.2f}x)", flush=True)


@app.local_entrypoint()
def main() -> None:
    call = run.spawn()
    print(f"SPAWN_ID {call.object_id}", flush=True)
    call.get()
