"""Probe which FP4 ldmatrix PTX variants ptxas accepts on sm_120a.

CubeCL 0.10/0.11-pre emit `ldmatrix.sync.aligned.m8n16.x{n}{.trans}.shared::cta.b8`
for packed e2m1 (FP4) operands, which ptxas rejects two ways: a plain `.b8` is the
wrong type for m8n16 ("unexpected number of instruction types"), and `.trans` is not
allowed on m8n16. The Blackwell-correct form is a format-converting load with a
`.dst_fmt.src_fmt` pair (`.b8x16.b4x16_p64`), using m8n16 for the non-transposed
operand and m16n16 for the transposed one.

This compiles each candidate variant as its own tiny .cu with nvcc and reports which
assemble, so we know exactly what string to make CubeCL emit before patching it.

Run:  uv run modal run harness/probe_ldmatrix.py
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

app = modal.App("quadbit-probe-ldmatrix", image=image)


def _kernel(shape: str, num: int, trans: bool, regs: int) -> str:
    trans_mod = ".trans" if trans else ""
    dst = ", ".join(f"%{i}" for i in range(regs))
    outs = ", ".join(f'"=r"(r[{i}])' for i in range(regs))
    return f"""
#include <cstdint>
extern "C" __global__ void probe(const uint8_t* __restrict__ src, uint32_t* __restrict__ out) {{
  __shared__ uint8_t smem[2048];
  unsigned t = threadIdx.x;
  smem[t * 4 + 0] = src[t * 4 + 0];
  smem[t * 4 + 1] = src[t * 4 + 1];
  smem[t * 4 + 2] = src[t * 4 + 2];
  smem[t * 4 + 3] = src[t * 4 + 3];
  __syncwarp();
  uint32_t r[{regs}] = {{0}};
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem[t * 8]));
  asm volatile(
    "ldmatrix.sync.aligned.{shape}.x{num}{trans_mod}.shared::cta.b8x16.b4x16_p64 {{{dst}}}, [%{regs}];"
    : {outs}
    : "r"(addr));
  for (int i = 0; i < {regs}; ++i) out[t * {regs} + i] = r[i];
}}
"""


@app.function(timeout=600)
def probe() -> None:
    base = "/root"
    Path(f"{base}/v.cu")  # ensure dir exists implicitly
    variants = [
        ("m8n16", 1, False, 1),
        ("m8n16", 2, False, 2),
        ("m8n16", 4, False, 4),
        ("m16n16", 1, True, 1),
        ("m16n16", 2, True, 2),
        ("m16n16", 4, True, 4),
        # also try the register count = 2x the num (b8x16 dst is 128-bit = 4 regs per mat)
        ("m8n16", 1, False, 2),
        ("m8n16", 2, False, 4),
        ("m16n16", 1, True, 2),
        ("m16n16", 2, True, 4),
    ]
    for shape, num, trans, regs in variants:
        src = _kernel(shape, num, trans, regs)
        p = Path(f"{base}/v.cu")
        p.write_text(src)
        r = subprocess.run(
            ["nvcc", "-arch=sm_120a", "-cubin", "-o", f"{base}/v.cubin", str(p)],
            capture_output=True,
            text=True,
        )
        tag = "trans" if trans else "     "
        label = f"{shape:8} x{num} {tag} regs={regs}"
        if r.returncode == 0:
            print(f"PASS  {label}", flush=True)
        else:
            err = (r.stderr or r.stdout).strip().splitlines()
            msg = next((ln for ln in err if "error" in ln.lower() or "ptxas" in ln.lower()), err[-1] if err else "?")
            print(f"FAIL  {label}  :: {msg.strip()[:160]}", flush=True)


@app.local_entrypoint()
def main() -> None:
    probe.remote()
