"""Honest 'why isn't 4-bit 4x' bench: FP4 vs bf16 across regimes on RTX PRO 6000.
Compute-bound speedup = FP4:bf16 FLOP-rate ratio (silicon, ~3.7x here), NOT bit-width.
Bit-width -> 4x is a MEMORY argument, realized only in memory-bound decode. Sparse 2:4 doubles
the FP4 compute advantage. Uses fixed packed buffers + scales (timing only; accuracy tested
elsewhere). Columns: bf16 (cuBLAS) | unit-scale FP4 (compute ceiling) | real-scale MXFP4 (deployable)
| 2:4-sparse FP4.

Run:  uv run modal run harness/bench_bitwidth.py
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
app = modal.App("quadbit-bitwidth", image=image)

# (label, M, N, K, regime)
SHAPES = [
    ("square 8192  (compute-bound)", 8192, 8192, 8192),
    ("square 16384 (compute-bound)", 16384, 16384, 16384),
    ("prefill FFN  4096x14336x4096", 4096, 14336, 4096),
    ("decode  FFN  128x14336x4096 ", 128, 14336, 4096),
]


@app.function(gpu="RTX-PRO-6000", timeout=1800)
def run() -> None:
    import ctypes

    import torch

    libs = {}
    for nm, src in (("u", "dense_fp4_lib"), ("s", "dense_scaled_fast_lib"), ("sp", "sparse_fp4_lib")):
        so = f"/root/{nm}.so"
        c = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
                            "-o", so, f"/root/cuda/{src}.cu", "-lcuda"], capture_output=True, text=True)
        if c.returncode != 0:
            print(c.stderr, flush=True); return
        libs[nm] = ctypes.CDLL(so)
    u, s, sp = libs["u"], libs["s"], libs["sp"]
    u.dense_fp4_mm.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3
    s.dense_scaled_fast_mm.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3
    sp.sparse_fp4_mm.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3
    print(torch.cuda.get_device_name(0), flush=True)

    def tms(fn, it=20):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(it):
            fn()
        b.record(); torch.cuda.synchronize()
        return a.elapsed_time(b) / it

    hdr = f"{'shape':>30} | {'bf16':>10} | {'unit-FP4':>13} | {'real-scale FP4':>15} | {'2:4-sparse':>13}"
    print(hdr + "\n" + "-" * len(hdr), flush=True)
    for label, M, N, K in SHAPES:
        flop = 2.0 * M * N * K
        Ab, Bb = torch.randn(M, K, device="cuda", dtype=torch.bfloat16), torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        tb = tms(lambda: torch.matmul(Ab, Bb.t()))

        Ad = torch.full((M, K // 2), 0x22, dtype=torch.uint8, device="cuda")
        Bd = torch.full((N, K // 2), 0x22, dtype=torch.uint8, device="cuda")
        Cd = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
        tu = tms(lambda: u.dense_fp4_mm(Ad.data_ptr(), Bd.data_ptr(), Cd.data_ptr(), M, N, K))

        # real-scale MXFP4: step-major SFA[K/128][M][4], SFB[K/128][N][4], fixed ue8m0=127 (scale 1)
        SFA = torch.full((K // 128, M, 4), 127, dtype=torch.uint8, device="cuda")
        SFB = torch.full((K // 128, N, 4), 127, dtype=torch.uint8, device="cuda")
        tr = tms(lambda: s.dense_scaled_fast_mm(Ad.data_ptr(), Bd.data_ptr(), SFA.data_ptr(),
                                                SFB.data_ptr(), Cd.data_ptr(), M, N, K))

        sp_str = "  n/a (M<256)"
        if M % 256 == 0:
            ks = K // 128
            Ac = torch.full((M, K // 4), 0x22, dtype=torch.uint8, device="cuda")
            Bs = torch.full((N, K // 2), 0x22, dtype=torch.uint8, device="cuda")
            sA = torch.full((ks, M, 4), 0x38, dtype=torch.uint8, device="cuda")
            sB = torch.full((ks, N, 4), 0x38, dtype=torch.uint8, device="cuda")
            mt = torch.full((ks, M, 2), 0x44444444, dtype=torch.int32, device="cuda")
            Cs = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
            tsp = tms(lambda: sp.sparse_fp4_mm(Ac.data_ptr(), Bs.data_ptr(), sA.data_ptr(),
                                               sB.data_ptr(), mt.data_ptr(), Cs.data_ptr(), M, N, K))
            sp_str = f"{tb/tsp:5.2f}x"
        print(f"{label:>30} | {tb*1e3:7.0f}us | {tb/tu:5.2f}x {flop/(tu/1e3)/1e12:5.0f}TF | "
              f"{tb/tr:5.2f}x | {sp_str:>13}", flush=True)


@app.local_entrypoint()
def main() -> None:
    run.remote()
