"""Compile and run a raw CUDA/PTX (.cu) file on a Modal SM120 GPU.

The CubeCL path tops out at ~616k GFLOP/s (matmul_fp4_ldm2); its SASS shows OMMA
is only ~5.6% of the stream, i.e. the codegen wraps each mma.sync in more overhead
than CUTLASS's hand-PTX mainloop. This harness is the raw-PTX track: hand-written
.cu files in cuda/, compiled with nvcc -arch=sm_120a directly (no Rust/CubeCL), to
control the instruction stream tightly. Self-contained .cu (kernel + main).

Run:  uv run modal run harness/run_cuda.py --cu hello_fp4
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

app = modal.App("quadbit-cuda", image=image)


def _run(cmd: list[str], cwd: str) -> int:
    print(f"$ {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    return proc.returncode


@app.function(gpu="RTX-PRO-6000", timeout=1200)
def build_and_run(cu: str, arch: str) -> None:
    _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], "/")
    _run(["nvcc", "--version"], "/")
    code = _run(
        ["nvcc", f"-arch={arch}", "-O3", "-o", f"/root/{cu}", f"/root/cuda/{cu}.cu", "-lcuda"],
        "/root",
    )
    if code != 0:
        print(f">>> nvcc failed (exit {code})", flush=True)
        return
    code = _run([f"/root/{cu}"], "/root")
    print(f">>> exit {code}", flush=True)


@app.local_entrypoint()
def main(cu: str = "hello_fp4", arch: str = "sm_120a") -> None:
    build_and_run.remote(cu, arch)
