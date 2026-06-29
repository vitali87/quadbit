"""Build and run a CubeCL (Rust) binary on a Modal GPU.

Image = CUDA 12.8 devel + rustup stable. Source (Cargo.toml + src/) is mounted
read-only; it is copied to a writable dir and built there. A Modal Volume caches
the cargo registry and the target/ dir so re-runs only recompile what changed.

Run:  uv run modal run harness/run_rust.py                  # hello on RTX-PRO-6000
      uv run modal run harness/run_rust.py --bin hello --gpu B200
"""

import subprocess
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
    .add_local_dir((ROOT / "vendor").as_posix(), "/root/quadbit/vendor")
)

cache = modal.Volume.from_name("quadbit-rust-cache", create_if_missing=True)

app = modal.App("quadbit-rust", image=image)


def _run(cmd: list[str], cwd: str = "/root/build") -> int:
    print(f"$ {' '.join(cmd)}", flush=True)
    env = {
        "PATH": "/root/.cargo/bin:/usr/local/cuda/bin:/usr/bin:/bin",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
        "CARGO_HOME": "/cache/cargo",
        "CARGO_TARGET_DIR": "/cache/target",
    }
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    out = r.stdout + r.stderr
    if len(out) > 16000:
        print(out[:8000] + "\n...[trimmed]...\n" + out[-8000:], flush=True)
    else:
        print(out, flush=True)
    return r.returncode


@app.function(gpu="RTX-PRO-6000", timeout=1800, volumes={"/cache": cache})
def build_and_run(bin_name: str) -> None:
    _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], cwd="/root")
    _run(["rustc", "--version"], cwd="/root")
    _run(["cp", "-r", "/root/quadbit", "/root/build"], cwd="/root")
    code = _run(["cargo", "run", "--release", "--features", "cuda", "--bin", bin_name])
    print(f">>> exit {code}", flush=True)


@app.local_entrypoint()
def main(bin: str = "hello", gpu: str = "RTX-PRO-6000") -> None:
    build_and_run.with_options(gpu=gpu).remote(bin)
