"""#3 smoke test: does vLLM run NATIVE NVFP4 (W4A4, real FP4 tensor core) on SM120, or fall
back to Marlin (W4A16, dequant->FP16 GEMM)? Uses Modal's recommended vLLM recipe (their
vllm_inference example: cuda 12.9.0-devel + vllm==0.21.0, HF/vLLM cache volumes, enforce_eager)
but pinned to RTX PRO 6000 (SM120, our target -- their example runs H200) and NVIDIA's official
dense NVFP4 checkpoint (nvidia/Llama-3.1-8B-Instruct-NVFP4).

The verdict is in the init log: a "does not have native support for FP4" warning or a
'marlin' quant method = W4A16 fallback (not true FP4); a modelopt/nvfp4 cutlass method = native
W4A4. This decides whether #3 can be a true FP4-vs-FP4 head-to-head or is our-FP4 vs their-W4A16.

Run:  uv run modal run harness/vllm_nvfp4.py
"""

import modal

MODEL = "nvidia/Llama-3.1-8B-Instruct-NVFP4"  # official dense NVFP4 (linear W+A 4-bit, attn bf16)
MINUTES = 60

# Modal's recommended vLLM image recipe (modal-examples vllm_inference.py), vllm pinned per their example.
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "huggingface_hub[hf_transfer]")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)
app = modal.App("quadbit-vllm-nvfp4", image=vllm_image)


@app.function(gpu="RTX-PRO-6000", timeout=30 * MINUTES,
              volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
              secrets=[modal.Secret.from_name("huggingface")])
def smoke() -> None:
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    print(f"vllm {vllm.__version__}  |  {torch.cuda.get_device_name(0)}  "
          f"sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}", flush=True)

    # dense NVFP4 loads directly on Blackwell; let vLLM auto-detect the quant method from the
    # checkpoint config so we observe which backend it actually picks (native cutlass vs marlin).
    llm = LLM(model=MODEL, enforce_eager=True, max_model_len=2048, gpu_memory_utilization=0.85)

    # introspect the linear method actually bound to a quantized layer
    try:
        mc = llm.llm_engine.model_config
        print(f"model_config.quantization = {mc.quantization}", flush=True)
    except Exception as e:
        print(f"(quant introspection failed: {e})", flush=True)

    out = llm.generate(["The capital of France is"], SamplingParams(temperature=0, max_tokens=16))
    print(f"GEN: {out[0].outputs[0].text!r}", flush=True)
    print("SMOKE_OK", flush=True)


@app.local_entrypoint()
def main() -> None:
    smoke.remote()
