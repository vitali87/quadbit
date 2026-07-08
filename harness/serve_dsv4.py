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
    # SM120 dense/attention unblock plugin: registered as a vllm.general_plugins entry point so the
    # monkeypatch runs in every spawned worker (an imperative patch in the driver does not survive).
    .add_local_dir(str(ROOT / "harness" / "qb_vllm_plugin"), "/opt/qb_plugin", copy=True)
    .run_commands("pip install /opt/qb_plugin")
)
app = modal.App("quadbit-serve", image=image)
vol = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)


@app.function(gpu="RTX-PRO-6000:2", timeout=60 * MIN, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def unblock(tp: int = 2, eager: bool = True, max_len: int = 2048, dense: str = "bf16") -> None:
    """WS0/WS1 end-to-end unblock: SM120-safe dense/attention (dense='bf16' or 'nvfp4') + native
    NVFP4 MoE. The replacement is installed by the qb_sm120 vLLM plugin (survives worker spawn); we
    only select it via QB_DENSE here. eager=True first (correctness), then flip for graph-capture."""
    import os

    import torch
    from vllm import LLM, SamplingParams

    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    os.environ["QB_DENSE"] = dense  # read by the qb_sm120 plugin in every spawned worker
    print(f"# WS0/1 unblock: dense={dense} tp={tp} eager={eager} on "
          f"{torch.cuda.device_count()}x RTX-PRO-6000", flush=True)

    rope = {"rope_type": "yarn", "factor": 16, "original_max_position_embeddings": 65536,
            "beta_fast": 32, "beta_slow": 1}
    kw = dict(model=MODEL, tensor_parallel_size=tp, enforce_eager=eager, trust_remote_code=True,
              max_model_len=max_len, gpu_memory_utilization=0.9, kv_cache_dtype="fp8",
              hf_overrides={"rope_scaling": rope})
    t0 = time.time()
    try:
        llm = LLM(tokenizer_mode="deepseek_v4", **kw)
    except Exception as ex:  # noqa: BLE001
        print(f"  (deepseek_v4 tokenizer_mode rejected: {type(ex).__name__}; retrying default)", flush=True)
        llm = LLM(**kw)
    print(f"  load+init forward ok in {time.time() - t0:.0f}s (the SM120 wall is cleared)", flush=True)

    prompts = ["The capital of France is", "def fibonacci(n):", "The three primary colors are"]
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
    ok = ntok > 0 and all(o.outputs[0].text.strip() for o in outs)
    print(f"# WS0 unblock {'PASS' if ok else 'FAIL'} "
          f"(bf16 dense fallback + native NVFP4 MoE, graph={'eager' if eager else 'captured'})", flush=True)


@app.function(gpu="RTX-PRO-6000:2", timeout=60 * MIN, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def baseline(tp: int = 2, eager: bool = False, max_len: int = 4096) -> None:
    import torch
    from vllm import LLM, SamplingParams

    import os
    # DeepGEMM's ue8m0 scale-factor transform asserts on SM120 ("Unknown SF transformation"); disable it
    # so vLLM routes FP8/scaled GEMMs to the FlashInfer/CUTLASS path that SM120 supports.
    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
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


@app.function(gpu="RTX-PRO-6000:2", timeout=60 * MIN, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def inspect_moe(tp: int = 2, max_len: int = 2048) -> None:
    # M3.6 recon: dump the exact FusedMoE structure so the sparse injection targets real attrs (model
    # is volume-cached after the first baseline load -> this loads fast).
    import inspect
    import os

    import torch
    from vllm import LLM

    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    rope = {"rope_type": "yarn", "factor": 16, "original_max_position_embeddings": 65536,
            "beta_fast": 32, "beta_slow": 1}
    llm = LLM(model=MODEL, tensor_parallel_size=tp, enforce_eager=True, trust_remote_code=True,
              max_model_len=max_len, gpu_memory_utilization=0.9, kv_cache_dtype="fp8",
              hf_overrides={"rope_scaling": rope}, tokenizer_mode="deepseek_v4")
    m = llm.llm_engine.model_executor.driver_worker.model_runner.model
    shown = 0
    for name, mod in m.named_modules():
        cls = type(mod).__name__
        if "fusedmoe" in cls.lower() or "experts" in cls.lower() or name.endswith("experts"):
            if shown >= 2:
                break
            shown += 1
            print(f"\n=== {name}  ({cls}) ===", flush=True)
            qm = getattr(mod, "quant_method", None)
            print(f"  quant_method: {type(qm).__name__}", flush=True)
            for an in dir(mod):
                if an.startswith("_"):
                    continue
                v = getattr(mod, an, None)
                if isinstance(v, torch.Tensor):
                    print(f"  tensor {an}: shape={tuple(v.shape)} dtype={v.dtype}", flush=True)
                elif isinstance(v, (int, bool)) and an in (
                        "top_k", "num_experts", "global_num_experts", "local_num_experts",
                        "intermediate_size_per_partition", "hidden_size", "renormalize",
                        "use_grouped_topk", "num_expert_group", "topk_group"):
                    print(f"  attr {an} = {v}", flush=True)
            if qm is not None and hasattr(qm, "apply"):
                try:
                    print(f"  apply{inspect.signature(qm.apply)}", flush=True)
                except (ValueError, TypeError):
                    pass
    print("\n# inspect_moe done", flush=True)


@app.function(gpu="RTX-PRO-6000:2", timeout=60 * MIN, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def quadbit(tp: int = 2, eager: bool = False, max_len: int = 2048) -> None:
    # M3.6: inject the segmented sparse-FP4 MoE op into vLLM's FusedMoE layers. Loads the staged .so
    # (built under CUDA 12.8 by build_so.py -> /cache/sparse_fp4.so) since the sparse mma won't assemble
    # under the serve image's CUDA 13 ptxas. Structure-specific wiring (expert-weight dequant + attr
    # names) is finalized against inspect_moe's dump; this run reports SPARSE_EXPERT_CALLS + fallbacks.
    import ctypes
    import os

    import torch
    from vllm import LLM, SamplingParams

    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    so = "/cache/sparse_fp4.so"
    lib = ctypes.CDLL(so)
    lib.sparse_moe_mm_2lvl.argtypes = ([ctypes.c_void_p] * 6 + [ctypes.c_int] * 4 +
                                       [ctypes.c_void_p] * 3 + [ctypes.c_int] + [ctypes.c_void_p])
    lib.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    lib.qb_init_moe_attrs()
    print(f"# M3.6 quadbit sparse-MoE injection: loaded {so}", flush=True)

    rope = {"rope_type": "yarn", "factor": 16, "original_max_position_embeddings": 65536,
            "beta_fast": 32, "beta_slow": 1}
    llm = LLM(model=MODEL, tensor_parallel_size=tp, enforce_eager=eager, trust_remote_code=True,
              max_model_len=max_len, gpu_memory_utilization=0.9, kv_cache_dtype="fp8",
              hf_overrides={"rope_scaling": rope}, tokenizer_mode="deepseek_v4")
    m = llm.llm_engine.model_executor.driver_worker.model_runner.model

    # locate FusedMoE layers (finalized against inspect_moe output)
    moe = [(n, mod) for n, mod in m.named_modules()
           if "fusedmoe" in type(mod).__name__.lower() and hasattr(mod, "quant_method")]
    print(f"  found {len(moe)} FusedMoE layers", flush=True)
    # The sparse injection (dequant experts NVFP4->bf16, 2:4 pack, replace quant_method.apply with a
    # sparse_moe_mm_2lvl path) is NOT yet wired: it needs the expert weight/scale attr names from
    # `inspect_moe`, and on SM120 this model's FP8 attention GEMM has no kernel so LLM() cannot even
    # finish init (see docs/paper.md Section 10). We deliberately do NOT run llm.generate() here -- that
    # would exercise the NATIVE vLLM MoE path and could be mistaken for a sparse-served result. This
    # entry point is ready to complete on a host whose FP8-attention backend supports SM120 (or Hopper).
    _ = SamplingParams  # referenced once wiring lands
    raise NotImplementedError(
        "quadbit sparse-MoE injection not wired: replace each FusedMoE.quant_method.apply with the "
        "segmented sparse_moe_mm_2lvl path using per-expert weights from `inspect` mode. This does NOT "
        "serve sparse yet; running native generate here would misrepresent the result.")


@app.local_entrypoint()
def main(mode: str = "baseline", tp: int = 2, eager: bool = False, max_len: int = 4096) -> None:
    if mode == "inspect":
        inspect_moe.remote(tp=tp, max_len=max_len)
    elif mode == "quadbit":
        quadbit.remote(tp=tp, eager=eager, max_len=max_len)
    elif mode == "unblock":
        call = unblock.spawn(tp=tp, eager=eager, max_len=max_len)
        print(f"SPAWN_ID={call.object_id}", flush=True)
    else:
        baseline.remote(tp=tp, eager=eager, max_len=max_len)
