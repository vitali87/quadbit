"""Feasibility probe: does Modal grant RTX-PRO-6000 at count>=4, and what is the interconnect?

Decides the campaign-wide stop-gate "no >=4 GPU run is possible". Prints device count + topology
(no NVLink expected on RTX-PRO-6000 -> PCIe P2P, which forces expert-parallel over tensor-parallel).
"""

import subprocess

import modal

image = modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12").pip_install(
    "torch", index_url="https://download.pytorch.org/whl/nightly/cu128", pre=True
)
app = modal.App("quadbit-gpuprobe", image=image)


@app.function(gpu="RTX-PRO-6000:4", timeout=600)
def probe4() -> None:
    import torch

    print(f"device_count={torch.cuda.device_count()}", flush=True)
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  gpu{i}: {p.name} mem={p.total_memory / 1e9:.1f}GB", flush=True)
    print("=== nvidia-smi topo ===", flush=True)
    print(subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True).stdout, flush=True)


@app.local_entrypoint()
def main() -> None:
    probe4.remote()
