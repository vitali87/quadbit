"""#2c: make QuadbitLinear CONSUME modelopt's calibrated NVFP4 checkpoint (buy, not build) and
run it through our kernel, so dense accuracy is the reference +0.7 without us reimplementing
calibration. Step 1 (this file, mode=inspect): dump the on-disk modelopt NVFP4 format -- which
scale tensors it stores (weight_packed FP4, weight_scale per-16, weight_scale_2 global,
input_scale), their shapes/dtypes -- so we can map them to our kernel's per-16 two-level layout.

Run:  uv run modal run harness/modelopt_ingest.py --mode inspect
"""

import modal

MODEL = "nvidia/Llama-3.1-8B-Instruct-NVFP4"

image = (
    modal.Image.from_registry("python:3.12-slim")
    .pip_install("huggingface_hub", "safetensors", "numpy")
)
app = modal.App("quadbit-modelopt-ingest", image=image)
vol = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)


@app.function(timeout=600, volumes={"/cache": vol}, secrets=[modal.Secret.from_name("huggingface")],
              image=image.env({"HF_HOME": "/cache"}))
def inspect() -> None:
    import json

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    # config: quant algo, group size, which layers excluded (kept higher precision)
    cfg = json.load(open(hf_hub_download(MODEL, "config.json")))
    qc = cfg.get("quantization_config", {})
    print("quantization_config:", json.dumps(qc, indent=2)[:2000], flush=True)

    idx_path = hf_hub_download(MODEL, "model.safetensors.index.json")
    wmap = json.load(open(idx_path))["weight_map"]
    # every distinct tensor suffix for layer 0 (weight, scales, etc.) + which shards
    l0 = sorted(k for k in wmap if k.startswith("model.layers.0."))
    shard = wmap[l0[0]]
    with safe_open(hf_hub_download(MODEL, shard), framework="numpy") as f:  # metadata only, no torch
        print(f"\nlayer-0 tensors ({shard}):", flush=True)
        for k in l0:
            t = f.get_slice(k)
            print(f"  {k:<58} {str(t.get_dtype()):<12} {list(t.get_shape())}", flush=True)
    # names present anywhere (to catch excluded/kept-precision layers)
    suffixes = sorted({k.split(".", 3)[-1] for k in wmap if ".layers." in k})
    print("\ndistinct per-layer suffixes:", suffixes, flush=True)
    non_layer = sorted(k for k in wmap if ".layers." not in k)
    print("non-layer tensors (lm_head/embed/norm - check if quantized):", non_layer, flush=True)


@app.local_entrypoint()
def main(mode: str = "inspect") -> None:
    inspect.remote()
