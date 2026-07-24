"""Feasibility probe: what GPU counts does Modal grant, and what is the interconnect?

Decides the campaign-wide stop-gate "no >=4 GPU run is possible". Prints device count + topology.
RTX-PRO-6000 has no NVLink -> PCIe P2P, which forces expert-parallel over tensor-parallel and sets
the per-layer all-reduce floor that dominates decode (docs/c4/floor_decomposition.md). The B200
probe tests the counterfactual: SM100 datacenter Blackwell with NVLink, where that floor collapses.
"""

import subprocess

import modal

image = modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12").pip_install(
    "torch", index_url="https://download.pytorch.org/whl/nightly/cu128", pre=True
)
app = modal.App("quadbit-gpuprobe", image=image)

MODES = {"rtx", "b200"}


def _probe() -> None:
    """Dump device count, per-GPU name/memory/arch, the P2P matrix, and nvidia-smi topology."""
    import torch

    n = torch.cuda.device_count()
    print(f"device_count={n}", flush=True)
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"  gpu{i}: {p.name} mem={p.total_memory / 1e9:.1f}GB "
              f"sm_{p.major}{p.minor}", flush=True)
    # P2P reachability decides whether a one-shot custom all-reduce is even legal (C4).
    print("=== p2p access matrix ===", flush=True)
    for i in range(n):
        row = ["1" if i == j or torch.cuda.can_device_access_peer(i, j) else "0" for j in range(n)]
        print(f"  gpu{i}: {' '.join(row)}", flush=True)
    print("=== nvidia-smi topo ===", flush=True)
    # Topology is diagnostic only, and nvidia-smi topo is not always permitted in a container, so a
    # failure here must not lose the device/P2P dump above.
    try:
        r = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=60)
        print(r.stdout or f"(nvidia-smi topo rc={r.returncode}: {r.stderr.strip()})", flush=True)
    except (OSError, subprocess.SubprocessError) as ex:
        print(f"(nvidia-smi topo unavailable: {type(ex).__name__}: {ex})", flush=True)


@app.function(gpu="RTX-PRO-6000:4", timeout=600)
def probe4() -> None:
    """Probe 4x RTX PRO 6000 (SM120): expect no NVLink, i.e. PCIe P2P."""
    _probe()


@app.function(gpu="B200:4", timeout=600)
def probe_b200() -> None:
    """Probe 4x B200 (SM100): the NVLink counterfactual to probe4."""
    _probe()


@app.local_entrypoint()
def main(mode: str = "rtx") -> None:
    """Dispatch to a probe. Unknown modes raise rather than silently booking the wrong hardware."""
    if mode not in MODES:
        raise SystemExit(f"unknown --mode {mode!r}; expected one of {sorted(MODES)}")
    (probe_b200 if mode == "b200" else probe4).remote()
