"""Probe whether ptxas accepts setmaxnreg (warpgroup register reallocation) on
sm_120a. This is the primitive that makes warp specialization worthwhile: it lets
consumer warps claim registers freed by producer warps. CubeCL exposes no such
intrinsic, so warp specialization could only help if (a) sm_120a supports the
instruction and (b) we inject it via custom PTX. setmaxnreg is a Hopper (sm_90a)
feature; consumer Blackwell sm_120a lacks several datacenter features, so this
decides whether the register-reallocation path exists at all.

Run:  uv run modal run harness/probe_setmaxnreg.py
"""

import subprocess
from pathlib import Path

import modal

image = modal.Image.from_registry(
    "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
).env(
    {
        "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
    }
)

app = modal.App("quadbit-probe-setmaxnreg", image=image)


def _kernel(op: str, n: int) -> str:
    return f"""
extern "C" __global__ void probe() {{
  asm volatile("setmaxnreg.{op}.sync.aligned.u32 {n};");
}}
"""


@app.function(timeout=600)
def probe() -> None:
    for op, n in [("inc", 240), ("dec", 96), ("inc", 256), ("dec", 24)]:
        src = _kernel(op, n)
        p = Path("/root/v.cu")
        p.write_text(src)
        for arch in ("sm_120a", "sm_90a"):
            r = subprocess.run(
                ["nvcc", f"-arch={arch}", "-cubin", "-o", "/root/v.cubin", str(p)],
                capture_output=True,
                text=True,
            )
            label = f"{arch:8} setmaxnreg.{op} {n}"
            if r.returncode == 0:
                print(f"PASS  {label}", flush=True)
            else:
                err = (r.stderr or r.stdout).strip().splitlines()
                msg = next((ln for ln in err if "error" in ln.lower()), err[-1] if err else "?")
                print(f"FAIL  {label}  :: {msg.strip()[:150]}", flush=True)


@app.local_entrypoint()
def main() -> None:
    probe.remote()
