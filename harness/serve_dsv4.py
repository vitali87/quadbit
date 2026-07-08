"""M3.6 substrate + M4 baseline: serve DeepSeek-V4-Flash NVFP4 in vLLM on RTX-PRO-6000 (sm_120).

DeepSeek-V4-Flash NVFP4 (~142GB) exceeds one 102GB GPU, so >=2 GPUs are required just to LOAD it
(expert/tensor parallel). This harness first proves the native stack loads + generates + graph-captures
(the M4 dense-NVFP4 baseline row and the M3.6 integration substrate). The quadbit sparse MoE monkeypatch
(M3.6) is layered on top once this passes.

Reports: load success, backend/quant method actually selected (fallback detection), a sanity generation,
prefill/decode timing, per-GPU memory, graph-capture (enforce_eager=False) status.
"""

import time
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
MODEL = "nvidia/DeepSeek-V4-Flash-NVFP4"
MIN = 60

# Let vLLM resolve its own torch + flashinfer (pinning them conflicts). vLLM >=0.20 ships the SM120
# Blackwell FP4 kernels and the deepseek_v4 model. CUDA 13 base for the runtime libs it expects.
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.12")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64", "HF_HOME": "/cache",
          "HF_XET_HIGH_PERFORMANCE": "1"})
    .pip_install("vllm", "huggingface_hub")
)
app = modal.App("quadbit-serve", image=image)
vol = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)


@app.function(gpu="RTX-PRO-6000:2", timeout=60 * MIN, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def baseline(tp: int = 2, eager: bool = False, max_len: int = 4096) -> None:
    import torch
    from vllm import LLM, SamplingParams

    print(f"# M4 baseline: {MODEL} tp={tp} eager={eager} on {torch.cuda.device_count()}x RTX-PRO-6000", flush=True)
    # DeepSeek-V4 config uses rope_scaling {type: yarn}; newer configs want rope_type -> patch via hf_overrides.
    rope = {"rope_type": "yarn", "factor": 16, "original_max_position_embeddings": 65536,
            "beta_fast": 32, "beta_slow": 1}
    t0 = time.time()
    kw = dict(model=MODEL, tensor_parallel_size=tp, enforce_eager=eager, trust_remote_code=True,
              max_model_len=max_len, gpu_memory_utilization=0.9, kv_cache_dtype="fp8",
              hf_overrides={"rope_scaling": rope})
    try:
        llm = LLM(tokenizer_mode="deepseek_v4", **kw)
    except Exception as ex:  # noqa: BLE001 -- deepseek_v4 tokenizer mode may not exist in this build
        print(f"  (deepseek_v4 tokenizer_mode rejected: {type(ex).__name__}; retrying default) ", flush=True)
        llm = LLM(**kw)
    print(f"  load ok in {time.time() - t0:.0f}s", flush=True)

    # which MoE / quant method actually got selected (fallback detection)
    try:
        eng = llm.llm_engine.model_executor.driver_worker.model_runner.model
        seen = set()
        for n, m in eng.named_modules():
            qm = type(getattr(m, "quant_method", None)).__name__
            if "moe" in type(m).__name__.lower() or "expert" in n.lower():
                seen.add(f"{type(m).__name__}:{qm}")
        print(f"  MoE modules/methods: {sorted(seen)[:8]}", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"  (module introspection failed: {type(ex).__name__}: {ex})", flush=True)

    prompts = ["The capital of France is", "def fibonacci(n):"]
    sp = SamplingParams(temperature=0.0, max_tokens=32)
    t1 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t1
    for o in outs:
        print(f"  [gen] {o.prompt!r} -> {o.outputs[0].text!r}", flush=True)
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"  generated {ntok} tok in {dt:.2f}s ({ntok / dt:.1f} tok/s)", flush=True)

    import subprocess
    mem = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    print(f"  per-GPU mem used (MB): {mem.split(chr(10))}", flush=True)
    print(f"# baseline {'PASS' if ntok > 0 else 'FAIL'} (graph_capture={'eager-OFF' if not eager else 'eager'})", flush=True)


@app.local_entrypoint()
def main(tp: int = 2, eager: bool = False, max_len: int = 4096) -> None:
    baseline.remote(tp=tp, eager=eager, max_len=max_len)
