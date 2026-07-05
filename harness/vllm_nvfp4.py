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
BASE = "meta-llama/Llama-3.1-8B-Instruct"     # the bf16 source; shares the tokenizer with MODEL
MINUTES = 60

# Modal's recommended vLLM image recipe (modal-examples vllm_inference.py), vllm pinned per their example.
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "huggingface_hub", "pyarrow")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
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


@app.function(gpu="RTX-PRO-6000", timeout=30 * MINUTES,
              volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
              secrets=[modal.Secret.from_name("huggingface")])
def ppl(model: str = MODEL) -> None:
    # Reference PPL of the official NVFP4 checkpoint through vLLM's native W4A4 kernel, on the
    # EXACT same wikitext-2 windows quadbit's recovery_worth uses (16x2048, HF-tokenized). bf16 KV
    # (kv_cache_dtype=auto) so only the linear W4A4 quant differs from a bf16 run -- matches quadbit's
    # bf16-attention fake-quant. If quadbit's W4A4 PPL matches this, our whole quant path is validated.
    import math

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(BASE)
    ids = tok("\n\n".join(pq.read_table(hf_hub_download(
        "Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
        repo_type="dataset")).column("text").to_pylist())).input_ids
    seq, windows = 2048, 16
    wins = [ids[i:i + seq] for i in range(0, min(len(ids), windows * seq) - seq, seq)]

    llm = LLM(model=model, enforce_eager=True, max_model_len=seq + 16, kv_cache_dtype="auto",
              gpu_memory_utilization=0.85)  # +16: prompt (seq) + the mandatory >=1 output token
    outs = llm.generate([{"prompt_token_ids": w} for w in wins],
                        SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1))
    nll, n = 0.0, 0
    for w, o in zip(wins, outs):
        plp = o.prompt_logprobs  # plp[0] is None (no context for first token)
        nll += -sum(plp[j][w[j]].logprob for j in range(1, seq)); n += seq - 1
    print(f"RESULT vllm_ppl={math.exp(nll / n):.4f}  (model={model}, bf16 KV, {len(wins)}x{seq})", flush=True)


@app.function(gpu="RTX-PRO-6000", timeout=30 * MINUTES,
              volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
              secrets=[modal.Secret.from_name("huggingface")])
def bench() -> None:
    # Throughput + memory for gap #3 (dense end-to-end): batched decode tok/s and the weights
    # footprint of vLLM native NVFP4 on SM120. Fair-comparison caveat printed: vLLM pre-allocates a
    # KV pool to gpu_memory_utilization, so "weights GiB" (its own load log) is the comparable
    # memory number, not total reserved. Paired with quadbit's full-forward bench (real_model).
    import time

    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(model=MODEL, enforce_eager=True, max_model_len=4096, gpu_memory_utilization=0.9)
    B, GEN = 64, 256
    prompts = [f"Question {i}: explain in detail how a transformer language model works." for i in range(B)]
    sp = SamplingParams(temperature=0, max_tokens=GEN, ignore_eos=True)
    llm.generate(prompts[:8], sp)                     # warmup
    torch.cuda.synchronize(); t = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"RESULT vllm_nvfp4 bench: {ntok} tok / {dt:.2f}s = {ntok / dt:.0f} tok/s "
          f"(batch {B}, gen {GEN}); peak_reserved {torch.cuda.max_memory_reserved() / 1e9:.1f}GB "
          f"(incl KV pool). Weights GiB in the 'Model loading took' log line above.", flush=True)


@app.function(gpu="RTX-PRO-6000", timeout=30 * MINUTES,
              volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
              secrets=[modal.Secret.from_name("huggingface")])
def serve(model: str = MODEL, eager: bool = False) -> None:
    # Serving-table row: separates PREFILL from DECODE and sweeps concurrency 1/8/32/64, plus memory.
    # Same engine for the NVFP4 and bf16 rows (pass model) so the quant path is the only variable.
    # prefill tok/s: B prompts of S=2048 tokens, 1 output (prefill-dominated). decode tok/s @ B:
    # B short prompts, GEN=128 ignore_eos (generation-dominated) -> throughput at concurrency B.
    # eager=False -> CUDA graphs captured (representative serving decode); eager=True -> no graphs.
    # Memory via nvidia-smi (vLLM runs the model in an EngineCore SUBPROCESS -> in-process torch reads 0).
    import subprocess
    import time

    import torch
    from vllm import LLM, SamplingParams

    def gpu_mib():  # total device memory in use (single vLLM process) -> real footprint incl KV pool
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.strip().splitlines()
        return int(out[0])

    S, GEN = 2048, 128
    llm = LLM(model=model, enforce_eager=eager, max_model_len=S + GEN + 16,
              kv_cache_dtype="auto", gpu_memory_utilization=0.9)
    mode = "eager" if eager else "cudagraph"
    print(f"[{model}] {torch.cuda.get_device_name(0)} mode={mode}", flush=True)
    mem_load = gpu_mib()

    # DISTINCT prompts per request: identical prompts + prefix caching collapse to one real prefill
    # and inflate prefill tok/s. Distinct S-token id sequences (valid vocab range) measure real work.
    def distinct_ids(i):
        return [((j * 131 + i * 7919) % 128000) + 1 for j in range(S)]

    def distinct_short(i):
        return f"Explain concept {i}: how does mechanism {i * 7 + 3} operate in practice, step by step?"

    def tps(prompts, sp):
        torch.cuda.synchronize(); t = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t
        return outs, dt

    llm.generate([distinct_short(0)], SamplingParams(temperature=0, max_tokens=8), use_tqdm=False)  # warmup
    mem_warm = gpu_mib(); mem_peak = mem_warm
    print(f"{'B':>4} | {'prefill tok/s':>14} | {'decode tok/s':>13}", flush=True)
    for B in (1, 8, 32, 64):
        pf_sp = SamplingParams(temperature=0, max_tokens=1)
        _, pf_dt = tps([{"prompt_token_ids": distinct_ids(i)} for i in range(B)], pf_sp)
        prefill_tps = B * S / pf_dt
        dc_sp = SamplingParams(temperature=0, max_tokens=GEN, ignore_eos=True)
        outs, dc_dt = tps([distinct_short(i) for i in range(B)], dc_sp)
        gen = sum(len(o.outputs[0].token_ids) for o in outs)
        mem_peak = max(mem_peak, gpu_mib())
        print(f"{B:>4} | {prefill_tps:>14.0f} | {gen / dc_dt:>13.0f}", flush=True)
    print(f"RESULT [{model}] mode={mode}: device MiB post-load {mem_load}, post-warmup {mem_warm}, "
          f"peak {mem_peak} (nvidia-smi, single vLLM proc; total incl KV pool @ util 0.9). "
          f"Model-weight GiB = 'Model loading took' log line. KV dtype auto(bf16).", flush=True)


@app.local_entrypoint()
def main(mode: str = "smoke", model: str = MODEL, eager: bool = False) -> None:
    # spawn (not remote): compute survives local-client disconnect in detached runs. See memory.
    fn = {"ppl": ppl, "bench": bench, "serve": serve}.get(mode, smoke)
    if mode == "serve":
        call = fn.spawn(model, eager)
    elif mode == "ppl":
        call = fn.spawn(model)
    else:
        call = fn.spawn()
    print(f"SPAWN_ID {call.object_id}", flush=True)
    call.get()
