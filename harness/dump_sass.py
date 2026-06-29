"""Static profiling substitute for ncu (which is blocked on Modal's gVisor stack).

Runs a CubeCL Rust binary with compilation logging on to capture the generated
CUDA source, then recompiles that source offline with `nvcc --ptxas-options=-v`
to recover the numbers ncu would otherwise give us: registers/thread, shared
memory, and spill stores/loads (the occupancy story). Then disassembles the
cubin and histograms the SASS opcodes (MMA / LDS / STS / LDG / BAR) to confirm
the tensor-core instruction stream looks the way the kernel intends.

Run:  uv run modal run harness/dump_sass.py --bin matmul_fp4_fed
"""

import re
import subprocess
from collections import Counter
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("curl", "build-essential", "pkg-config")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        "| sh -s -- -y --default-toolchain stable --profile minimal"
    )
    .env(
        {
            "PATH": "/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:"
            "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
        }
    )
    .add_local_file((ROOT / "Cargo.toml").as_posix(), "/root/quadbit/Cargo.toml")
    .add_local_dir((ROOT / "src").as_posix(), "/root/quadbit/src")
)

cache = modal.Volume.from_name("quadbit-rust-cache", create_if_missing=True)
app = modal.App("quadbit-dump-sass", image=image)


def _run(cmd: list[str], cwd: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)


@app.function(gpu="RTX-PRO-6000", timeout=1800, volumes={"/cache": cache})
def dump(bin_name: str, arch: str) -> None:
    base_env = {
        "PATH": "/root/.cargo/bin:/usr/local/cuda/bin:/usr/bin:/bin",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
        "CARGO_HOME": "/cache/cargo",
        "CARGO_TARGET_DIR": "/cache/target",
    }
    subprocess.run(["cp", "-r", "/root/quadbit", "/root/build"], check=True)

    # run with compilation logging so the generated CUDA source hits stdout
    run_env = {**base_env, "CUBECL_DEBUG_LOG": "stdout", "CUBECL_DEBUG_OPTION": "debug-full"}
    r = _run(
        ["cargo", "run", "--release", "--features", "cuda", "--bin", bin_name],
        cwd="/root/build",
        env=run_env,
    )
    out = r.stdout + r.stderr

    blocks = re.findall(
        r"\[START_KERNEL_COMPILATION\].*?```[a-z]*\n(.*?)```\s*\[END_KERNEL_COMPILATION\]",
        out,
        re.DOTALL,
    )
    if not blocks:
        print(">>> no kernel source captured; raw tail follows:")
        print(out[-4000:])
        return
    # the matmul kernel is the largest captured source
    source = max(blocks, key=len)
    Path("/root/kernel.cu").write_text(source)
    print(f">>> captured {len(blocks)} kernel(s); largest source = {len(source)} bytes")

    # recompile offline to recover registers / smem / spills (ncu-equivalent numbers)
    print(f"\n=== ptxas -v (arch {arch}) ===", flush=True)
    c = _run(
        ["nvcc", f"-arch={arch}", "-cubin", "-o", "/root/k.cubin", "/root/kernel.cu",
         "--ptxas-options=-v"],
        cwd="/root",
        env=base_env,
    )
    print((c.stdout + c.stderr).strip()[:4000])
    if c.returncode != 0:
        print(">>> nvcc failed; first source lines for diagnosis:")
        print("\n".join(source.splitlines()[:40]))
        return

    # disassemble and histogram the SASS opcode stream
    print("\n=== SASS opcode histogram ===", flush=True)
    d = _run(["cuobjdump", "-sass", "/root/k.cubin"], cwd="/root", env=base_env)
    sass = d.stdout + d.stderr
    ops: Counter[str] = Counter()
    for line in sass.splitlines():
        m = re.search(r"/\*[0-9a-f]+\*/\s+@?!?P?\d?\s*([A-Z][A-Z0-9_.]+)", line)
        if m:
            ops[m.group(1).split(".")[0]] += 1
    total = sum(ops.values())
    print(f"total SASS instructions: {total}")
    for op, n in ops.most_common(25):
        tag = "  <-- tensor MMA" if "MMA" in op else ""
        print(f"  {op:12} {n:6}{tag}")
    interesting = {k: v for k, v in ops.items() if any(x in k for x in ("MMA", "LDS", "STS", "LDG", "STG", "BAR"))}
    print(f"\nkey ops: {interesting}")


@app.local_entrypoint()
def main(bin: str = "matmul_fp4_fed", arch: str = "sm_120a") -> None:
    dump.remote(bin, arch)
