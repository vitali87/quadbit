"""PyTorch binding for the sparse FP4 kernel: compile cuda/sparse_fp4_lib.cu to a .so
(nvcc -arch=sm_120a, the bleeding-edge arch torch's own build won't target), then call it
from PyTorch via ctypes on torch CUDA tensors' data_ptr(). Proves the kernel is usable as
a torch op. Builds all-ones packed tensors (output must equal Klog/2) for a clean
correctness check, then times it.

Run:  uv run modal run harness/run_torch.py
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

app = modal.App("quadbit-torch", image=image)


@app.function(gpu="RTX-PRO-6000", timeout=1800)
def run() -> None:
    print(subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout, flush=True)
    so = "/root/sparse_fp4.so"
    c = subprocess.run(
        ["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
         "-o", so, "/root/cuda/sparse_fp4_lib.cu", "-lcuda"],
        capture_output=True, text=True,
    )
    print(c.stdout + c.stderr, flush=True)
    if c.returncode != 0:
        print(">>> nvcc failed", flush=True)
        return

    import ctypes
    import torch

    print(f"torch {torch.__version__}, device {torch.cuda.get_device_name(0)}", flush=True)
    lib = ctypes.CDLL(so)
    lib.sparse_fp4_mm.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3
    lib.sparse_fp4_mm.restype = ctypes.c_int

    def sparse_fp4_mm(Ac, B, scaleA, scaleB, meta, M, N, K):
        """torch op: packed 2:4-sparse FP4 A @ B -> bf16 [M,N]."""
        C = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
        rc = lib.sparse_fp4_mm(Ac.data_ptr(), B.data_ptr(), scaleA.data_ptr(),
                               scaleB.data_ptr(), meta.data_ptr(), C.data_ptr(), M, N, K)
        if rc != 0:
            raise RuntimeError(f"kernel returned cuda error {rc}")
        return C

    for sz in (2048, 4096, 8192):
        M = N = K = sz
        ksteps = K // 128
        # all-ones packed inputs (fp4 1.0=0x22, ue4m3 unit=0x38, meta pairs{0,1}=0x44444444)
        Ac = torch.full((M, K // 4), 0x22, dtype=torch.uint8, device="cuda")
        B = torch.full((N, K // 2), 0x22, dtype=torch.uint8, device="cuda")
        scaleA = torch.full((ksteps, M, 4), 0x38, dtype=torch.uint8, device="cuda")
        scaleB = torch.full((ksteps, N, 4), 0x38, dtype=torch.uint8, device="cuda")
        meta = torch.full((ksteps, M, 2), 0x44444444, dtype=torch.int32, device="cuda")

        C = sparse_fp4_mm(Ac, B, scaleA, scaleB, meta, M, N, K)
        torch.cuda.synchronize()
        exp = float(K // 2)
        ok = bool((C.float() == exp).all().item())

        it = 20
        for _ in range(3):
            sparse_fp4_mm(Ac, B, scaleA, scaleB, meta, M, N, K)
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(it):
            sparse_fp4_mm(Ac, B, scaleA, scaleB, meta, M, N, K)
        en.record(); torch.cuda.synchronize()
        ms = st.elapsed_time(en) / it
        gf = 2.0 * M * N * K / (ms / 1e3) / 1e9
        print(f"torch sparse_fp4_mm {M}x{N}x{K}: {ms:.3f} ms  {gf:.1f} GFLOP/s  "
              f"{'PASS' if ok else 'FAIL'} (out[0,0]={C[0,0].item():.0f} exp {exp:.0f})", flush=True)


@app.local_entrypoint()
def main() -> None:
    run.remote()
