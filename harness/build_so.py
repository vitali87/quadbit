"""Staged .so build: compile the sparse-FP4 kernel under CUDA 12.8 (ptxas that assembles the mma.sp
mxf4nvf4) and write it to the shared /cache volume, so the CUDA-13 vLLM serve image can ctypes-load it
without a second toolchain. Run once; serve_dsv4.py's quadbit mode loads /cache/sparse_fp4.so.
"""

import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64"})
    .add_local_dir((ROOT / "cuda").as_posix(), "/root/cuda")
)
app = modal.App("quadbit-buildso", image=image)
vol = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)


@app.function(gpu="RTX-PRO-6000", timeout=1200, volumes={"/cache": vol})
def build() -> None:
    out = "/cache/sparse_fp4.so"
    c = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
                        "-o", out, "/root/cuda/sparse_fp4_lib.cu", "-lcuda"], capture_output=True, text=True)
    if c.returncode != 0:
        print(c.stderr, flush=True)
        raise SystemExit(1)
    vol.commit()
    import os
    print(f"# built {out} ({os.path.getsize(out)} bytes) under CUDA 12.8 ptxas -> committed to volume", flush=True)


@app.local_entrypoint()
def main() -> None:
    build.remote()
