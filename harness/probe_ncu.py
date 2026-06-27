"""Probe whether Nsight Compute (ncu) hardware-counter profiling works on Modal.

Modal runs GPU containers on gVisor + nvproxy, which whitelists only the
compute/utility/graphics/video driver capabilities. The perfmon/profiling
ioctl path that `ncu` needs is likely absent, so we expect ERR_NVGPUCTRPERM.
This script settles the question empirically on the cheapest GPU (T4): the
result is identical across all Modal GPUs, since the runtime is the same.

Run:  uv run modal run harness/probe_ncu.py
"""

import subprocess

import modal

CUDA_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu22.04"

image = modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")

app = modal.App("quadbit-ncu-probe", image=image)

KERNEL_SRC = r"""
#include <cstdio>
__global__ void add_one(float* a) { a[threadIdx.x] += 1.0f; }
int main() {
    float* d;
    cudaMalloc(&d, 256 * sizeof(float));
    add_one<<<1, 256>>>(d);
    cudaDeviceSynchronize();
    cudaFree(d);
    printf("kernel ran\n");
    return 0;
}
"""


def _run(cmd: list[str]) -> tuple[int, str]:
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)


@app.function(gpu="T4", timeout=600)
def probe() -> None:
    print("=== environment ===")
    for cmd in (["nvidia-smi"], ["nvcc", "--version"], ["ncu", "--version"], ["nsys", "--version"]):
        code, out = _run(cmd)
        print(f"$ {' '.join(cmd)}  (exit {code})\n{out.strip()[:600]}\n")

    print("=== compile a trivial kernel ===")
    with open("k.cu", "w") as f:
        f.write(KERNEL_SRC)
    code, out = _run(["nvcc", "k.cu", "-o", "k"])
    print(f"nvcc exit {code}\n{out.strip()}")
    if code != 0:
        print(">>> RESULT: nvcc failed; cannot continue")
        return

    print("\n=== THE DECISIVE TEST: ncu hardware counters ===")
    code, out = _run(["ncu", "--set", "basic", "./k"])
    print(f"ncu exit {code}\n{out.strip()[:3000]}")
    ncu_blocked = code != 0 or "ERR_NVGPUCTRPERM" in out or "not supported" in out.lower()

    print("\n=== nsys trace (expected to work) ===")
    code, out = _run(["nsys", "profile", "-o", "trace", "--force-overwrite", "true", "./k"])
    print(f"nsys exit {code}\n{out.strip()[:1500]}")
    nsys_ok = code == 0

    print("\n========================= VERDICT =========================")
    print(f"ncu (hardware counters): {'BLOCKED' if ncu_blocked else 'WORKS'}")
    print(f"nsys (trace profiling):  {'WORKS' if nsys_ok else 'BLOCKED'}")
    if ncu_blocked:
        print("-> Split workflow: Modal for dev/run/time/autotune; bare-metal box for ncu.")
    else:
        print("-> ncu works on Modal; full kernel dev + profiling stays on Modal.")
