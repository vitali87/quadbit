"""Reference benchmark: NVIDIA CUTLASS block-scaled FP4 GEMM on SM120.

Builds and runs CUTLASS example 79b (nv_float4 x nv_float4 -> f32, ArchTag Sm120),
the apples-to-apples reference for our `matmul_fp4_fed` kernel: same FP4 mma.sync
path (no tcgen05 on sm_120), same 2*M*N*K flop count, f32 accumulate, with a
built-in reference verification. Tells us the achievable FP4 ceiling on this card
and how much room our ~505k GFLOP/s kernel leaves on the table.

CUTLASS is cloned and built inside the container (one example target only,
library/profiler/tests off) and cached in a Volume so re-runs skip the rebuild.

Run:  uv run modal run harness/cutlass_fp4.py
"""

import subprocess

import modal

CUTLASS_REPO = "https://github.com/NVIDIA/cutlass"
EXAMPLE = "79b_blackwell_geforce_nvfp4_nvfp4_gemm"
EXE = f"/cache/cutlass/build/examples/79_blackwell_geforce_gemm/{EXAMPLE}"
SIZES = [2048, 4096, 8192]

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "cmake", "build-essential", "ninja-build")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64"})
)

cache = modal.Volume.from_name("quadbit-cutlass-cache", create_if_missing=True)
app = modal.App("quadbit-cutlass", image=image)


def _run(cmd: list[str], cwd: str) -> int:
    print(f"$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    out = r.stdout + r.stderr
    print(out[-12000:] if len(out) > 12000 else out, flush=True)
    return r.returncode


@app.function(gpu="RTX-PRO-6000", timeout=3600, volumes={"/cache": cache})
def build_and_run() -> None:
    import os

    _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], "/")
    _run(["nvcc", "--version"], "/")

    if not os.path.exists("/cache/cutlass"):
        _run(["git", "clone", "--depth", "1", CUTLASS_REPO, "/cache/cutlass"], "/cache")

    if not os.path.exists(EXE):
        # clean rebuild: a cached build dir may hold a stale cmake cache from a
        # different CUDA toolchain
        _run(["rm", "-rf", "/cache/cutlass/build"], "/cache")
        os.makedirs("/cache/cutlass/build", exist_ok=True)
        cfg = _run(
            [
                "cmake", "-G", "Ninja", "..",
                "-DCUTLASS_NVCC_ARCHS=120a",
                "-DCUTLASS_ENABLE_TESTS=OFF",
                "-DCUTLASS_ENABLE_LIBRARY=OFF",
                "-DCUTLASS_ENABLE_PROFILER=OFF",
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            "/cache/cutlass/build",
        )
        if cfg != 0:
            print(">>> cmake configure failed", flush=True)
            return
        b = _run(["cmake", "--build", ".", "--target", EXAMPLE, "-j", "8"], "/cache/cutlass/build")
        if b != 0:
            print(">>> build failed", flush=True)
            return
    cache.commit()

    for sz in SIZES:
        print(f"\n===== CUTLASS {EXAMPLE}  {sz}x{sz}x{sz} =====", flush=True)
        _run([EXE, f"--m={sz}", f"--n={sz}", f"--k={sz}", "--iterations=20"], "/")


@app.function(gpu="RTX-PRO-6000", timeout=1200, volumes={"/cache": cache})
def dissect() -> None:
    """SASS/resource dissection of the cached 79b binary: which staging path
    (TMA / cp.async / ldmatrix), pipeline depth (shared mem), warp specialization,
    and the OMMA-to-overhead instruction ratio that we are losing to."""
    import re
    import subprocess

    if not __import__("os").path.exists(EXE):
        print(">>> binary not built; run the bench entrypoint first", flush=True)
        return

    print("=== cuobjdump -res-usage (registers / shared mem per kernel) ===", flush=True)
    ru = subprocess.run(["cuobjdump", "-res-usage", EXE], capture_output=True, text=True)
    print(ru.stdout[-6000:], flush=True)

    sass = subprocess.run(["cuobjdump", "-sass", EXE], capture_output=True, text=True).stdout
    # split into per-function blocks; keep the one(s) with OMMA (the GEMM mainloop)
    blocks = re.split(r"\n\s*Function : ", sass)
    key = ["OMMA", "LDSM", "LDGSTS", "UTMALDG", "UBLKCP", "BAR", "DEPBAR", "ELECT",
           "BMOV", "WARPSYNC", "LDS", "STS", "LDG", "STG"]
    for blk in blocks:
        if "OMMA" not in blk:
            continue
        name = blk.splitlines()[0][:90]
        ops: dict[str, int] = {}
        for line in blk.splitlines():
            m = re.search(r"/\*[0-9a-f]+\*/\s+@?!?P?\d?\s*([A-Z][A-Z0-9_.]+)", line)
            if m:
                op = m.group(1).split(".")[0]
                ops[op] = ops.get(op, 0) + 1
        total = sum(ops.values())
        print(f"\n=== GEMM function: {name} ===  (total SASS ~{total})", flush=True)
        for k in key:
            c = sum(v for o, v in ops.items() if o == k)
            if c:
                print(f"  {k:10} {c}", flush=True)
        top = sorted(ops.items(), key=lambda kv: -kv[1])[:15]
        print("  top opcodes: " + ", ".join(f"{o}:{c}" for o, c in top), flush=True)


@app.local_entrypoint()
def main(mode: str = "bench") -> None:
    if mode == "dissect":
        dissect.remote()
    else:
        build_and_run.remote()
