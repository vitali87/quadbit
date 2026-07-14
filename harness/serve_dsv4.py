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
    .env(
        {
            "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
            "HF_HOME": "/cache",
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
    .pip_install("vllm", "huggingface_hub", "datasets")
    # vLLM 0.24.0's deepseek_v4 sparse-MLA path calls flashinfer's newer
    # trtllm_batch_decode_sparse_mla_dsv4(swa_topk_lens=..., extra_sparse_indices=...) API, but
    # 0.24.0 exact-pins flashinfer 0.6.12 (older sparse_topk_lens/seq_lens sig -> TypeError). Force
    # 0.6.14 with --no-deps so the resolver can't backtrack vLLM to 0.11.0 (which lacks deepseek_v4).
    # The swa_topk_lens/extra_sparse_* API vLLM 0.24.0 calls exists ONLY in flashinfer-python 0.6.14,
    # but flashinfer-cubin stops at 0.6.13. Use python 0.6.14 (for the API) + cubin 0.6.13 (latest
    # precompiled kernels) and bypass the python/cubin version check -- the sparse-MLA kernel JITs at
    # runtime regardless. --no-deps keeps vLLM 0.24.0 pinned (a plain pin backtracks it to 0.11.0).
    .run_commands(
        "pip install --force-reinstall --no-deps flashinfer-python==0.6.14 flashinfer-cubin==0.6.13"
    )
    .env({"FLASHINFER_DISABLE_VERSION_CHECK": "1"})
    # SM120 dense/attention unblock plugin: registered as a vllm.general_plugins entry point so the
    # monkeypatch runs in every spawned worker (an imperative patch in the driver does not survive).
    .add_local_dir(str(ROOT / "harness" / "qb_vllm_plugin"), "/opt/qb_plugin", copy=True)
    # force_build so a plugin edit always redeploys (Modal's layer cache has served stale bytecode
    # here despite --force-reinstall); ~10s cost, buys reliable deploys.
    .run_commands("pip install --force-reinstall --no-deps /opt/qb_plugin", force_build=True)
)
app = modal.App("quadbit-serve", image=image)
vol = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)


@app.function(timeout=10 * MIN)
def inspect_dp() -> None:
    """C5: find the correct vLLM offline data-parallel launch API (LLM(data_parallel_size>1) is rejected)."""
    import os
    import subprocess

    import vllm
    root = os.path.dirname(vllm.__file__)
    # locate the exact guard that raises "single-process usage" and dump surrounding code for a bypass
    r = subprocess.run(["grep", "-rn", "single-process usage", root], capture_output=True, text=True)
    print(f"# guard hits:\n{r.stdout}", flush=True)
    for line in r.stdout.strip().split("\n"):
        if ":" not in line:
            continue
        fp = line.split(":")[0]
        ln = int(line.split(":")[1])
        print(f"# ===== {fp} around {ln} =====", flush=True)
        src = open(fp).read().split("\n")
        for i in range(max(0, ln - 22), min(len(src), ln + 3)):
            print(f"{i+1:4d}: {src[i]}", flush=True)
        break
    # also: does data_parallel_rank / a bypass flag exist on LLM/EngineArgs?
    r2 = subprocess.run(["grep", "-rn", "data_parallel_rank\\|_data_parallel_master\\|dp_rank",
                         f"{root}/entrypoints/llm.py", f"{root}/engine/arg_utils.py"],
                        capture_output=True, text=True)
    print(f"# dp_rank/bypass refs:\n{r2.stdout[:2000]}", flush=True)


@app.function(timeout=10 * MIN)
def inspect_vllm() -> None:
    """C4: dump vLLM's custom_all_reduce guard so the enable-on-4-PCIe patch is exact (the ncclDevKernel
    RING_LL all-reduce is 90.8% of decode; custom AR is disabled by a >2-PCIe policy check)."""
    import inspect

    from vllm.distributed.device_communicators import custom_all_reduce as car
    src = inspect.getsource(car).split("\n")
    print(f"# custom_all_reduce.py ({car.__file__})", flush=True)
    # full __init__ (lines ~55-210) + should_custom_ar (~230-263): the exact guard + nvlink detection
    for i, line in enumerate(src, 1):
        if 55 <= i <= 210 or 228 <= i <= 265 or 25 <= i <= 50:
            print(f"{i:4d}: {line}", flush=True)
    print("# --- also check the tp group all_reduce dispatch ---", flush=True)
    try:
        from vllm.distributed.parallel_state import GroupCoordinator
        gsrc = inspect.getsource(GroupCoordinator.all_reduce)
        for line in gsrc.split("\n"):
            print(f"    {line}", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"  (GroupCoordinator.all_reduce introspect failed: {ex})", flush=True)


@app.function(
    gpu="RTX-PRO-6000:8",
    timeout=90 * MIN,
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def glm_baseline(
    tp: int = 8,
    eager: bool = True,
    max_len: int = 2048,
    dense: str = "nvfp4",
    moe: str = "dense",
    dense_layers: str = "",
    sparse_proj: str = "both",
    route_slot: int = 0,
) -> None:
    """GLM-5.2 transfer load + coherence gate on 8x RTX PRO 6000 (EP). Tests: (a) 8-GPU schedule,
    (b) glm_moe_dsa loads on SM120 under the quadbit plugin, (c) DSA attention runs, (d) coherent
    generation. moe=dense first (NVFP4->bf16 dequant experts, no sparse); moe=sparse + dense_layers/
    sparse_proj/route_slot reuses the DeepSeek-proven structural policies on GLM's expert path."""
    import math
    import os
    import subprocess
    import time

    import torch
    from vllm import LLM, SamplingParams

    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    os.environ["QB_DENSE"] = dense
    os.environ["QB_MOE"] = moe
    os.environ["QB_DENSE_LAYERS"] = dense_layers
    os.environ["QB_SPARSE_PROJ"] = sparse_proj
    os.environ["QB_ROUTE_SLOT"] = str(route_slot)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    print(
        f"# GLM-5.2 baseline: {GLM_MODEL} dense={dense} moe={moe} tp={tp} eager={eager} "
        f"dense_layers=[{dense_layers}] proj={sparse_proj} route_slot={route_slot} on "
        f"{torch.cuda.device_count()}x RTX-PRO-6000",
        flush=True,
    )

    # GLM keeps its own rope/config (1M context); do NOT force DeepSeek's yarn override.
    kw = dict(
        model=GLM_MODEL,
        tensor_parallel_size=tp,
        enforce_eager=eager,
        trust_remote_code=True,
        max_model_len=max_len,
        gpu_memory_utilization=0.92,
        kv_cache_dtype="fp8",
        max_num_batched_tokens=max_len,
        enable_expert_parallel=True,
    )
    t0 = time.time()
    llm = LLM(**kw)
    print(f"  load+init forward ok in {time.time() - t0:.0f}s", flush=True)

    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "The three primary colors are",
        "Water is made of hydrogen and",
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=32)
    outs = llm.generate(prompts, sp)
    for o in outs:
        print(f"  [gen] {o.prompt!r} -> {o.outputs[0].text!r}", flush=True)
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)

    # --- PPL: teacher-forced perplexity over a fixed held-out passage (per-policy quality) ---
    passage = (
        "The mitochondria is the powerhouse of the cell. Photosynthesis converts sunlight, water, "
        "and carbon dioxide into glucose and oxygen. The Earth orbits the Sun once every year, and "
        "the Moon orbits the Earth roughly every twenty-eight days. Water boils at one hundred "
        "degrees Celsius at sea level and freezes at zero degrees. The human heart pumps blood "
        "through arteries and veins, delivering oxygen to every tissue in the body. Shakespeare "
        "wrote many famous plays, including Hamlet, Macbeth, and Romeo and Juliet. The speed of "
        "light in a vacuum is approximately three hundred thousand kilometres per second."
    )
    pids = llm.get_tokenizer().encode(passage)
    pout = llm.generate(
        [{"prompt_token_ids": pids}],
        SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0),
    )
    plp = pout[0].prompt_logprobs or []
    nlls = [
        -d[tid].logprob
        for tid, d in zip(pids[1:], plp[1:], strict=False)
        if d and tid in d and math.isfinite(d[tid].logprob)
    ]
    ppl = math.exp(sum(nlls) / len(nlls)) if nlls else float("nan")
    print(f"# PPL over {len(nlls)}-token held-out passage: {ppl:.3f}", flush=True)

    # --- coarse serving timing: prefill (prompt-heavy, gen 1) and decode (gen 64) at B=1 ---
    # prompt+gen must fit max_len; clamp so a long-prompt row never aborts the eval.
    tok = llm.get_tokenizer()
    base = tok.encode(passage)
    for plen in (512, 1536):
        gen = 64
        plen = min(plen, max_len - gen - 8)
        if plen <= 0:
            continue
        ptoks = (base * ((plen // len(base)) + 1))[:plen]
        one = SamplingParams(temperature=0.0, max_tokens=1)
        try:
            tp0 = time.time()
            llm.generate([{"prompt_token_ids": ptoks}], one)
            prefill_s = time.time() - tp0
            td0 = time.time()
            dout = llm.generate(
                [{"prompt_token_ids": ptoks}], SamplingParams(temperature=0.0, max_tokens=gen)
            )
            dec_s = time.time() - td0
            gtok = len(dout[0].outputs[0].token_ids)
        except Exception as e:  # noqa: BLE001 - timing must never abort the quality eval
            print(f"# serve B=1 prompt={plen}: timing skipped ({type(e).__name__})", flush=True)
            continue
        # dec_s = prefill + gtok decode steps; the gen=1 call above is prefill + 1 step, so
        # (dec_s - prefill_s) isolates the remaining gtok-1 decode steps -> true decode-only rate.
        dec_only = max(1e-6, dec_s - prefill_s)
        steps = max(1, gtok - 1)
        print(
            f"# serve B=1 prompt={plen} gen={gen}: prefill(TTFT~){prefill_s:.2f}s "
            f"decode-only {steps}tok in {dec_only:.2f}s = {steps / dec_only:.2f} tok/s "
            f"TPOT {dec_only / steps:.3f}s (end-to-end {gtok / max(dec_s, 1e-6):.2f} tok/s)",
            flush=True,
        )

    mem = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"  per-GPU mem used (MB): {mem.splitlines()}", flush=True)
    ok = ntok > 0 and all(o.outputs[0].text.strip() for o in outs)
    print(
        f"# GLM baseline {'PASS' if ok else 'FAIL'} dense={dense} moe={moe} "
        f"(glm_moe_dsa on SM120, tp={tp} EP, eager={eager})",
        flush=True,
    )


@app.function(
    gpu="RTX-PRO-6000:2",
    timeout=60 * MIN,
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def unblock(
    tp: int = 2,
    eager: bool = True,
    max_len: int = 2048,
    dense: str = "bf16",
    kv: str = "fp8",
    moe: str = "off",
) -> None:
    """WS0/WS1 end-to-end unblock: SM120-safe dense/attention (dense='bf16' or 'nvfp4') + native
    NVFP4 MoE. The replacement is installed by the qb_sm120 vLLM plugin (survives worker spawn); we
    only select it via QB_DENSE here. eager=True first (correctness), then flip for graph-capture."""
    import os

    import torch
    from vllm import LLM, SamplingParams

    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    os.environ["QB_DENSE"] = dense  # read by the qb_sm120 plugin in every spawned worker
    os.environ["QB_MOE"] = moe  # off|dense|sparse: quadbit sparse-expert injection selector
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # quiet load bars so worker tracebacks survive
    print(
        f"# WS0/1 unblock: dense={dense} tp={tp} eager={eager} on "
        f"{torch.cuda.device_count()}x RTX-PRO-6000",
        flush=True,
    )

    rope = {
        "rope_type": "yarn",
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "beta_fast": 32,
        "beta_slow": 1,
    }
    # bf16 fallback keeps fp8 originals (MLA absorption) + bf16 copies -> dense weights ~3x; push
    # gpu_mem_util high and keep batched tokens modest so KV cache memory stays positive.
    kw = dict(
        model=MODEL,
        tensor_parallel_size=tp,
        enforce_eager=eager,
        trust_remote_code=True,
        max_model_len=max_len,
        gpu_memory_utilization=0.95,
        kv_cache_dtype=kv,
        max_num_batched_tokens=max_len,
        hf_overrides={"rope_scaling": rope},
    )
    t0 = time.time()
    try:
        llm = LLM(tokenizer_mode="deepseek_v4", **kw)
    except Exception as ex:  # noqa: BLE001
        # only fall back when the tokenizer_mode itself is the problem; never mask a forward/init error
        if "tokenizer_mode" not in str(ex) and "deepseek_v4" not in str(ex).lower():
            raise
        print(
            f"  (deepseek_v4 tokenizer_mode rejected: {type(ex).__name__}; retrying default)",
            flush=True,
        )
        llm = LLM(**kw)
    print(
        f"  load+init forward ok in {time.time() - t0:.0f}s (the SM120 wall is cleared)", flush=True
    )

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

    mem = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"  per-GPU mem used (MB): {mem.splitlines()}", flush=True)
    # audit the actual dense path taken (nvfp4 vs per-layer bf16 fallback) for honest labeling
    try:
        import qb_sm120_plugin as _p

        print(f"  qb STATS (driver view): {_p.STATS}", flush=True)
    except Exception:  # noqa: BLE001
        pass
    ok = ntok > 0 and all(o.outputs[0].text.strip() for o in outs)
    label = {
        "bf16": "bf16 dense fallback",
        "nvfp4": "NVFP4 dense (mm_fp4 cutlass)",
        "off": "native SM120 path",
    }.get(dense, dense)
    moe_label = {
        "off": "native NVFP4 MoE",
        "dense": "NVFP4->bf16 dequant experts (dense)",
        "sparse": "quadbit 2:4 sparse-FP4 experts",
    }.get(moe, moe)
    print(
        f"# unblock {'PASS' if ok else 'FAIL'} dense={dense} moe={moe} "
        f"({label} + {moe_label} + bf16 DSA indexer, "
        f"graph={'eager' if eager else 'captured'})",
        flush=True,
    )


def _graph_gate_body(
    tp: int = 2,
    eager: bool = False,
    force_graph_path: bool = False,
    proj: str = "both",
    route_slot: int = 0,
    dense_layers: str = "",
    cap: int = 512,
    max_seqs: int = 8,
    max_len: int = 2048,
    gpu_mem: float = 0.9,
    glm: bool = False,
    dense_anchor_backend: str = "dequant",
    baseline: str = "",
    dp: int = 1,
) -> None:
    """P4 M4 graph-capture gate on DeepSeek-V4-Flash sparse-FP4 (2 GPU, EP). Three configs:
      A eager=True  force_graph_path=False -> QB_GRAPH=0, enforce_eager=True   (frozen Campaign-B path)
      B eager=True  force_graph_path=True  -> QB_GRAPH=1, enforce_eager=True   (graph-safe path, eager exec)
      C eager=False                        -> QB_GRAPH=1, enforce_eager=False  (graph-safe path, CAPTURED)
    B vs A isolates the code-path change; C vs B isolates capture; C vs A is the end-to-end M4 claim.
    Prints greedy token_ids (exact diff) + teacher-forced PPL. cap is the fixed per-local-expert row
    capacity; max_seqs bounds the decode batch so cap can't overflow (a drop would break quality)."""
    import math
    import os

    import torch
    from vllm import LLM, SamplingParams

    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    gp = force_graph_path or (not eager)
    os.environ["QB_DENSE"] = "nvfp4"
    # C2 SOTA board: baseline="dense_nvfp4" -> QB_MOE=off, which makes patched_moe_pw return early so
    # vLLM's native FlashInfer-CUTLASS NVFP4 fused MoE runs unchanged (the production dense NVFP4 path),
    # with attention/DSA still SM120-unblocked. Same passage/decode-formula/graph mode as the sparse
    # rows -> an apples-to-apples same-harness dense baseline for the SOTA table. "" keeps the sparse path.
    os.environ["QB_MOE"] = "off" if baseline == "dense_nvfp4" else "sparse"
    os.environ["QB_SPARSE_PROJ"] = proj
    os.environ["QB_ROUTE_SLOT"] = str(route_slot)
    os.environ["QB_DENSE_LAYERS"] = dense_layers
    os.environ["QB_GRAPH"] = "1" if gp else "0"
    os.environ["QB_GRAPH_CAP"] = str(cap)
    # C1: dense-anchor projection backend. "dequant" = the frozen range(E) dequant-to-bf16 loop
    # (_dense_seg_gs); "native_nvfp4" = flashinfer group_gemm_nvfp4_nt_groupwise (fused NVFP4 grouped).
    os.environ["QB_DENSE_BACKEND"] = dense_anchor_backend
    cfg = "C-captured" if (gp and not eager) else ("B-graphpath-eager" if gp else "A-frozen-eager")
    pol = (f"BASELINE-dense-nvfp4 (QB_MOE=off, native FlashInfer-CUTLASS fused MoE)"
           if baseline == "dense_nvfp4"
           else f"proj={proj} route_slot={route_slot} dense_layers=[{dense_layers}]")
    print(f"# M4 graph_gate cfg={cfg} {pol} cap={cap} max_seqs={max_seqs} "
          f"QB_GRAPH={os.environ['QB_GRAPH']} enforce_eager={eager} tp={tp} "
          f"model={'GLM' if glm else 'DeepSeek'} dense_backend={dense_anchor_backend}", flush=True)

    t0 = time.time()
    if glm:
        # GLM-5.2 keeps its own rope/config (1M ctx); no DeepSeek yarn override or tokenizer_mode.
        kw = dict(model=GLM_MODEL, tensor_parallel_size=tp, enforce_eager=eager, trust_remote_code=True,
                  max_model_len=max_len, gpu_memory_utilization=gpu_mem, kv_cache_dtype="fp8",
                  max_num_batched_tokens=max(2048, max_len), max_num_seqs=max_seqs, enable_expert_parallel=True)
        llm = LLM(**kw)
    else:
        rope = {"rope_type": "yarn", "factor": 16, "original_max_position_embeddings": 65536,
                "beta_fast": 32, "beta_slow": 1}
        kw = dict(model=MODEL, tensor_parallel_size=tp, enforce_eager=eager, trust_remote_code=True,
                  max_model_len=max_len, gpu_memory_utilization=gpu_mem, kv_cache_dtype="fp8",
                  max_num_batched_tokens=max(2048, max_len), max_num_seqs=max_seqs,
                  enable_expert_parallel=True, hf_overrides={"rope_scaling": rope})
        if dp > 1:
            # C5: DP attention + EP MoE. tp=1 so attention runs replicated per DP rank (NO per-layer TP
            # all-reduce, the 94.5% floor); experts EP-sharded across all dp ranks (MoE all-to-all only).
            kw["data_parallel_size"] = dp
            print(f"  C5 DP-attention: tp={tp} data_parallel_size={dp} (removes the attention TP all-reduce)",
                  flush=True)
        try:
            llm = LLM(tokenizer_mode="deepseek_v4", **kw)
        except Exception as ex:  # noqa: BLE001
            print(f"  (deepseek_v4 tokenizer_mode rejected: {type(ex).__name__}; default) ", flush=True)
            llm = LLM(**kw)
    print(f"  load+capture ok in {time.time() - t0:.0f}s (captured => graph capture SUCCEEDED)", flush=True)

    prompts = ["The capital of France is", "def fibonacci(n):", "The three primary colors are",
               "Water is made of hydrogen and"]
    sp = SamplingParams(temperature=0.0, max_tokens=24)
    outs = llm.generate(prompts, sp)
    for o in outs:
        print(f"  [gen] {o.prompt!r} -> {o.outputs[0].text!r}", flush=True)
        print(f"        ids={list(o.outputs[0].token_ids)}", flush=True)
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)

    passage = (
        "The mitochondria is the powerhouse of the cell. Photosynthesis converts sunlight, water, "
        "and carbon dioxide into glucose and oxygen. The Earth orbits the Sun once every year, and "
        "the Moon orbits the Earth roughly every twenty-eight days. Water boils at one hundred "
        "degrees Celsius at sea level and freezes at zero degrees. The human heart pumps blood "
        "through arteries and veins, delivering oxygen to every tissue in the body."
    )
    pids = llm.get_tokenizer().encode(passage)
    pout = llm.generate([{"prompt_token_ids": pids}],
                        SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0))
    plp = pout[0].prompt_logprobs or []
    nlls = [-d[tid].logprob for tid, d in zip(pids[1:], plp[1:], strict=False)
            if d and tid in d and math.isfinite(d[tid].logprob)]
    ppl = math.exp(sum(nlls) / len(nlls)) if nlls else float("nan")
    print(f"# PPL over {len(nlls)}-token passage: {ppl:.4f}", flush=True)

    # Decode tok/s (C1 speed metric): two-run TTFT-subtracted -- time a 64-token and a 1-token
    # generation from the same prompt; decode_tps = 63 / (wall64 - wall1) removes the shared prefill.
    tp = "The history of the Roman empire spans many centuries and"
    tids = llm.get_tokenizer().encode(tp)
    def _wall(n):
        torch.cuda.synchronize()
        t = time.time()
        llm.generate([{"prompt_token_ids": tids}], SamplingParams(temperature=0.0, max_tokens=n))
        torch.cuda.synchronize()
        return time.time() - t
    _wall(4)  # warm
    w1, w64 = _wall(1), _wall(64)
    dtps = 63.0 / (w64 - w1) if w64 > w1 else float("nan")
    print(f"# decode tok/s: {dtps:.3f} (wall1={w1:.3f}s wall64={w64:.3f}s)", flush=True)
    print(f"# graph_gate {cfg} {'PASS' if ntok > 0 else 'FAIL'} (ntok={ntok}, ppl={ppl:.4f}, "
          f"decode_tps={dtps:.3f})", flush=True)


@app.function(
    gpu="RTX-PRO-6000:2",
    timeout=75 * MIN,
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def graph_gate(
    tp: int = 2,
    eager: bool = False,
    force_graph_path: bool = False,
    proj: str = "both",
    route_slot: int = 0,
    dense_layers: str = "",
    cap: int = 512,
    max_seqs: int = 8,
    max_len: int = 2048,
    gpu_mem: float = 0.9,
) -> None:
    """2-GPU P4 M4 graph-capture gate (down49 / gateup49). See _graph_gate_body for config A/B/C."""
    _graph_gate_body(tp, eager, force_graph_path, proj, route_slot, dense_layers,
                     cap, max_seqs, max_len, gpu_mem)


@app.function(
    gpu="RTX-PRO-6000:4",
    timeout=90 * MIN,
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def graph_gate4(
    tp: int = 4,
    eager: bool = False,
    force_graph_path: bool = False,
    proj: str = "both",
    route_slot: int = 2,
    dense_layers: str = "",
    cap: int = 512,
    max_seqs: int = 8,
    max_len: int = 2048,
    gpu_mem: float = 0.9,
    dense_anchor_backend: str = "dequant",
    baseline: str = "",
    c3_skip: str = "",
    compact: bool = False,
    a_dense: int = 0,
    a_sparse: int = 0,
    nccl_algo: str = "",
    nccl_proto: str = "",
    nccl_nchannels: int = 0,
    force_custom_ar: bool = False,
) -> None:
    """4-GPU P4 M4 graph-capture gate for route-slot D2 (dual residency: raw NVFP4 dense slots +
    packed sparse codes need 4-way EP). Defaults tp=4, route_slot=2. See _graph_gate_body for A/B/C.
    C1: dense_anchor_backend=native_nvfp4 routes the dense group through group_gemm_nvfp4.
    C2: baseline=dense_nvfp4 runs vLLM's native NVFP4 fused MoE (QB_MOE=off) through the same board.
    C3 Task 1A: c3_skip in {moe,dense,sparse} no-ops that component under CAPTURE for differential decode
    attribution (PPL is meaningless for a skip variant — read only decode tok/s)."""
    import os
    if c3_skip and c3_skip.lower() not in ("moe", "dense", "sparse"):
        raise ValueError(f"c3_skip must be one of moe/dense/sparse (or empty), got {c3_skip!r}: "
                         "the plugin only reads QB_C3_SKIP_{MOE,DENSE,SPARSE}, so a typo would "
                         "silently record attribution for the unskipped baseline.")
    # Clear every C3 flag first so a warm Modal container never inherits a prior invocation's state
    # (a stale QB_C3_SKIP_*/QB_COMPACT_DECODE/QB_A_* would silently run the wrong benchmark variant).
    for _k in ("QB_C3_SKIP_MOE", "QB_C3_SKIP_DENSE", "QB_C3_SKIP_SPARSE",
               "QB_COMPACT_DECODE", "QB_A_DENSE", "QB_A_SPARSE"):
        os.environ.pop(_k, None)
    if c3_skip:
        os.environ[f"QB_C3_SKIP_{c3_skip.upper()}"] = "1"
    if compact:
        os.environ["QB_COMPACT_DECODE"] = "1"
    if a_dense:
        os.environ["QB_A_DENSE"] = str(a_dense)
    if a_sparse:
        os.environ["QB_A_SPARSE"] = str(a_sparse)
    # C4: NCCL collective tuning. Decode is 90.8% RING_LL all-reduce over PCIe (no NVLink), latency-bound at
    # batch=1. NCCL_ALGO=Tree has fewer latency hops than Ring for tiny payloads at 4 ranks. Must be set
    # before LLM() (NCCL reads these at communicator init); worker subprocs inherit this env.
    for _k in ("NCCL_ALGO", "NCCL_PROTO", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS", "NCCL_NCHANNELS"):
        os.environ.pop(_k, None)
    if nccl_algo:
        os.environ["NCCL_ALGO"] = nccl_algo
    if nccl_proto:
        os.environ["NCCL_PROTO"] = nccl_proto
    if nccl_nchannels:
        os.environ["NCCL_MIN_NCHANNELS"] = str(nccl_nchannels)
        os.environ["NCCL_MAX_NCHANNELS"] = str(nccl_nchannels)
    if nccl_algo or nccl_proto or nccl_nchannels:
        print(f"# C4 NCCL tuning: ALGO={nccl_algo or '(auto)'} PROTO={nccl_proto or '(auto)'} "
              f"NCHANNELS={nccl_nchannels or '(auto)'}", flush=True)
    os.environ.pop("QB_FORCE_CUSTOM_AR", None)
    if force_custom_ar:
        os.environ["QB_FORCE_CUSTOM_AR"] = "1"
        print("# C4: QB_FORCE_CUSTOM_AR=1 (enable vLLM one-shot custom all-reduce on 4 PCIe GPUs)", flush=True)
    _graph_gate_body(tp, eager, force_graph_path, proj, route_slot, dense_layers,
                     cap, max_seqs, max_len, gpu_mem, dense_anchor_backend=dense_anchor_backend,
                     baseline=baseline)


@app.function(
    gpu="RTX-PRO-6000:2",
    timeout=90 * MIN,
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def graph_gate2(
    proj: str = "both",
    route_slot: int = 2,
    dense_layers: str = "",
    cap: int = 128,
    max_seqs: int = 2,
    max_len: int = 2048,
    gpu_mem: float = 0.92,
    dense_anchor_backend: str = "native_nvfp4",
    baseline: str = "dense_nvfp4",
    nccl_algo: str = "",
    nccl_proto: str = "",
    force_custom_ar: bool = False,
) -> None:
    """C5 collective-floor lever: TP=2 variant of the graph_gate board. At world_size=2 the per-layer TP
    all-reduce is a single peer exchange (custom AR is natively enabled by vLLM for world_size==2), so decode
    collective latency should drop vs the 4-GPU ring/one-shot. Same _graph_gate_body, tp=2, captured.
    force_custom_ar still available (verifies the 2x2 P2P matrix, bypasses the flaky probe). Memory is tight
    at TP=2 (~82 GiB/GPU weights of 96), so gpu_mem default 0.92 and short max_len."""
    import os
    for _k in ("NCCL_ALGO", "NCCL_PROTO", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS", "NCCL_NCHANNELS"):
        os.environ.pop(_k, None)
    if nccl_algo:
        os.environ["NCCL_ALGO"] = nccl_algo
    if nccl_proto:
        os.environ["NCCL_PROTO"] = nccl_proto
    os.environ.pop("QB_FORCE_CUSTOM_AR", None)
    if force_custom_ar:
        os.environ["QB_FORCE_CUSTOM_AR"] = "1"
        print("# C5: QB_FORCE_CUSTOM_AR=1 (TP=2, custom AR native at world_size==2)", flush=True)
    print(f"# C5 TP=2 board: baseline={baseline} route_slot={route_slot} cap={cap} gpu_mem={gpu_mem}",
          flush=True)
    _graph_gate_body(2, False, False, proj, route_slot, dense_layers, cap, max_seqs, max_len, gpu_mem,
                     dense_anchor_backend=dense_anchor_backend, baseline=baseline)


@app.function(
    gpu="RTX-PRO-6000:4",
    timeout=90 * MIN,
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def graph_gate_dp(
    dp: int = 4,
    cap: int = 128,
    max_seqs: int = 2,
    max_len: int = 2048,
    gpu_mem: float = 0.9,
    baseline: str = "dense_nvfp4",
    eager: bool = False,
) -> None:
    """C5 reduce-count lever: DP attention + EP MoE (tp=1, data_parallel_size=dp). Attention runs replicated
    per DP rank so there is NO per-layer TP all-reduce (the 94.5% decode floor); the MoE stays EP (already
    all-to-all today, not the dominant cost), so DP removes the ~43 attention all-reduces without adding MoE
    collectives. Spawns dp processes (vLLM offline DP needs per-proc VLLM_DP_* env). Rank 0 reports."""
    import subprocess
    import sys

    # Launch each DP rank as a FRESH python interpreter (subprocess), not multiprocessing.spawn: spawn
    # re-imports this Modal-decorated __main__ in the child and chokes on the @app.function decorators.
    # A fresh interpreter importing the installed qb_dp_worker module avoids both re-import and pickling.
    print(f"# C5 DP board: dp={dp} tp=1 baseline={baseline} eager={eager} (subprocess DP attention)",
          flush=True)
    port = 29591
    procs = []
    for r in range(dp):
        code = (f"import qb_dp_worker; qb_dp_worker.dp_worker({r},{dp},{port},{MODEL!r},{cap},"
                f"{max_seqs},{max_len},{gpu_mem},{baseline!r},{eager})")
        procs.append(subprocess.Popen([sys.executable, "-c", code]))
    # Poll all ranks: if any exits non-zero, the survivors would hang on that rank's missing collective, so
    # terminate them and fail loudly (a DP init error / OOM must not be reported as a successful run).
    codes: list = [None] * dp
    while any(c is None for c in codes):
        for i, p in enumerate(procs):
            if codes[i] is None and (rc := p.poll()) is not None:
                codes[i] = rc
        if any(c is not None and c != 0 for c in codes):
            for p in procs:
                if p.poll() is None:
                    p.terminate()
            for p in procs:
                try:
                    p.wait(timeout=15)
                except Exception:  # noqa: BLE001
                    p.kill()
            for i, p in enumerate(procs):
                codes[i] = p.poll()
            break
        time.sleep(2)
    print(f"# C5 DP board DONE (exit codes: {codes})", flush=True)
    if any(c != 0 for c in codes):
        raise RuntimeError(f"C5 DP workers failed (exit codes {codes}); see log. Offline LLM likely "
                           "rejected data_parallel_size>1 (needs vllm serve/AsyncLLM).")


@app.function(
    gpu="RTX-PRO-6000:8",
    timeout=120 * MIN,
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def glm_graph_gate(
    tp: int = 8,
    eager: bool = False,
    force_graph_path: bool = False,
    proj: str = "both",
    route_slot: int = 2,
    dense_layers: str = "",
    cap: int = 512,
    max_seqs: int = 8,
    max_len: int = 2048,
    gpu_mem: float = 0.92,
    dense_anchor_backend: str = "dequant",
    baseline: str = "",
) -> None:
    """8-GPU P4 M4 graph-capture gate on GLM-5.2 route-slot D2 (directive #4). GLM's EP MoE capture was
    previously blocked by the plugin's torch.unique().tolist() host-sync; the QB_GRAPH graph-safe path
    (route_fixed_cap) removes it. Config A/B/C as in _graph_gate_body; defaults tp=8, route_slot=2 (D2).
    C1: dense_anchor_backend=native_nvfp4 routes the dense group through group_gemm_nvfp4.
    C2: baseline=dense_nvfp4 runs vLLM's native NVFP4 fused MoE (QB_MOE=off) through the same board."""
    _graph_gate_body(tp, eager, force_graph_path, proj, route_slot, dense_layers,
                     cap, max_seqs, max_len, gpu_mem, glm=True,
                     dense_anchor_backend=dense_anchor_backend, baseline=baseline)


@app.function(
    gpu="RTX-PRO-6000:4",
    timeout=90 * MIN,
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def c3_profile(
    tp: int = 4,
    route_slot: int = 2,
    dense_layers: str = "",
    cap: int = 128,
    max_seqs: int = 2,
    max_len: int = 2048,
    gpu_mem: float = 0.9,
    ntok: int = 32,
) -> None:
    """C3 Task 1: profile the DeepSeek-D2 native decode path (graph-safe EAGER, config B). vLLM V1 runs the
    model in worker subprocesses, so driver-side torch.profiler sees nothing; instead the plugin does
    worker-side CUDA-event timing (QB_PROFILE) and logs [qb_prof] MoE-component cumulative ms
    (dense_anchor / sparse_group / sparse_seg(matmul_sp) / sparse_quant) plus [qb_profile] padded-vs-real
    routing waste per layer. This function drives a long decode so those cumulative ratios ≈ decode ratios,
    and reports decode-only tok/s (two-run TTFT-subtracted) so the MoE cost can be sized against the step."""
    import os

    import torch
    from vllm import LLM, SamplingParams

    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    os.environ["QB_DENSE"] = "nvfp4"
    os.environ["QB_MOE"] = "sparse"
    os.environ["QB_SPARSE_PROJ"] = "both"
    os.environ["QB_ROUTE_SLOT"] = str(route_slot)
    os.environ["QB_DENSE_LAYERS"] = dense_layers
    os.environ["QB_GRAPH"] = "1"            # graph-safe code path...
    os.environ["QB_GRAPH_CAP"] = str(cap)
    os.environ["QB_DENSE_BACKEND"] = "native_nvfp4"
    os.environ["QB_PROFILE"] = "1"          # routing-waste logging
    print(f"# C3 profile: D2 native graph-safe EAGER route_slot={route_slot} cap={cap} "
          f"dense_layers=[{dense_layers}] ntok={ntok} tp={tp}", flush=True)

    rope = {"rope_type": "yarn", "factor": 16, "original_max_position_embeddings": 65536,
            "beta_fast": 32, "beta_slow": 1}
    kw = dict(model=MODEL, tensor_parallel_size=tp, enforce_eager=True, trust_remote_code=True,
              max_model_len=max_len, gpu_memory_utilization=gpu_mem, kv_cache_dtype="fp8",
              max_num_batched_tokens=max(2048, max_len), max_num_seqs=max_seqs,
              enable_expert_parallel=True, hf_overrides={"rope_scaling": rope})
    t0 = time.time()
    try:
        llm = LLM(tokenizer_mode="deepseek_v4", **kw)
    except Exception:  # noqa: BLE001
        llm = LLM(**kw)
    print(f"  load ok in {time.time() - t0:.0f}s", flush=True)

    tp_prompt = "The history of the Roman empire spans many centuries and"
    tids = llm.get_tokenizer().encode(tp_prompt)
    # warm (JITs DSA kernels, fills caches; also lets QB_PROFILE log per-layer waste on first decode steps)
    llm.generate([{"prompt_token_ids": tids}], SamplingParams(temperature=0.0, max_tokens=8))
    torch.cuda.synchronize()

    # decode-only tok/s (same two-run TTFT-subtracted formula as _graph_gate_body) to size the step budget
    def _wall(n):
        torch.cuda.synchronize()
        t = time.time()
        llm.generate([{"prompt_token_ids": tids}], SamplingParams(temperature=0.0, max_tokens=n))
        torch.cuda.synchronize()
        return time.time() - t
    w1, w64 = _wall(1), _wall(64)
    dtps = 63.0 / (w64 - w1) if w64 > w1 else float("nan")
    step_ms = 1000.0 * (w64 - w1) / 63.0
    # the long (64-tok) decode above dominates the plugin's cumulative [qb_prof]/[qb_profile] logs, so their
    # ratios ≈ decode ratios. Final flush happens on the worker side as pairs accumulate.
    print(f"# C3 decode: {dtps:.3f} tok/s  step={step_ms:.1f} ms/token (wall1={w1:.3f}s wall64={w64:.3f}s)",
          flush=True)
    print("# C3 profile DONE — see worker [qb_prof] (MoE component ms) + [qb_profile] (routing waste) lines",
          flush=True)


@app.function(
    gpu="RTX-PRO-6000:4",
    timeout=90 * MIN,
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def floor_profile(
    tp: int = 4,
    max_seqs: int = 2,
    max_len: int = 2048,
    gpu_mem: float = 0.9,
    baseline: str = "dense_nvfp4",
    force_custom_ar: bool = False,
) -> None:
    """C4 gating: the decode step is 94.5% non-MoE floor (19.6 ms) and 5.5% MoE apply (1.13 ms), so the
    only decode headroom worth chasing is the floor. Profile the DENSE baseline (QB_MOE=off, the 48.248
    tok/s SOTA config) with vLLM's worker-side profiler (captures NCCL collectives + attention/DSA + GEMM
    + norm kernels), then sum GPU-kernel time by category to localize the floor. Eager so kernel boundaries
    are clean; GPU kernel durations are the same eager vs captured, only launch gaps differ, so the category
    breakdown is representative. Prime suspect: 4-GPU EP all-to-all over PCIe (no NVLink), 43 layers x 2."""
    import glob
    import gzip
    import json
    import os

    import torch
    from vllm import LLM, SamplingParams

    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    os.environ["QB_DENSE"] = "nvfp4"
    os.environ["QB_MOE"] = "off" if baseline == "dense_nvfp4" else "sparse"
    os.environ.pop("QB_FORCE_CUSTOM_AR", None)
    if force_custom_ar:
        os.environ["QB_FORCE_CUSTOM_AR"] = "1"
    prof_dir = "/cache/floorprof"
    os.environ["VLLM_TORCH_PROFILER_DIR"] = prof_dir
    os.makedirs(prof_dir, exist_ok=True)
    for f in glob.glob(f"{prof_dir}/*.json*"):
        os.remove(f)
    print(f"# floor_profile: baseline={baseline} QB_MOE={os.environ['QB_MOE']} tp={tp} -> {prof_dir}",
          flush=True)

    rope = {"rope_type": "yarn", "factor": 16, "original_max_position_embeddings": 65536,
            "beta_fast": 32, "beta_slow": 1}
    kw = dict(model=MODEL, tensor_parallel_size=tp, enforce_eager=True, trust_remote_code=True,
              max_model_len=max_len, gpu_memory_utilization=gpu_mem, kv_cache_dtype="fp8",
              max_num_batched_tokens=max(2048, max_len), max_num_seqs=max_seqs,
              enable_expert_parallel=True, hf_overrides={"rope_scaling": rope},
              profiler_config={"profiler": "torch", "torch_profiler_dir": prof_dir})
    try:
        llm = LLM(tokenizer_mode="deepseek_v4", **kw)
    except Exception as ex:  # noqa: BLE001
        print(f"  (deepseek_v4 mode / profiler_config rejected: {type(ex).__name__}: {ex}; retry)", flush=True)
        try:
            llm = LLM(**kw)
        except Exception as ex2:  # noqa: BLE001
            print(f"  (profiler_config rejected: {type(ex2).__name__}: {ex2}; retry w/o it)", flush=True)
            kw.pop("profiler_config", None)
            llm = LLM(**kw)

    tids = llm.get_tokenizer().encode("The history of the Roman empire spans many centuries and")
    llm.generate([{"prompt_token_ids": tids}], SamplingParams(temperature=0.0, max_tokens=8))  # warm
    torch.cuda.synchronize()
    llm.start_profile()
    llm.generate([{"prompt_token_ids": tids}], SamplingParams(temperature=0.0, max_tokens=16))
    llm.stop_profile()
    torch.cuda.synchronize()
    vol.commit()

    # categorize GPU-kernel time from the rank-0 trace (each rank ~symmetric; collectives show on all)
    cats = [
        ("collective(EP a2a / allreduce)", ("nccl", "alltoall", "all_to_all", "allreduce", "all_reduce",
                                            "reduce_scatter", "reducescatter", "sendrecv", "_reduce", "gather")),
        ("attention+DSA", ("flash", "fmha", "attn", "mla", "sparse_mla", "dsv3", "mqa", "sm120_decode",
                           "paged", "rope")),
        ("gemm/moe", ("gemm", "cutlass", "nvfp4", "matmul", "moe", "quant", "fp4", "fp8", "wgmma")),
        ("norm/elementwise", ("norm", "rms", "silu", "add", "mul", "cast", "copy", "elementwise", "act")),
    ]
    def bucket(name):
        n = name.lower()
        for label, keys in cats:
            if any(k in n for k in keys):
                return label
        return "other"

    traces = sorted(glob.glob(f"{prof_dir}/*.json*"))
    print(f"# traces: {[os.path.basename(t) for t in traces]}", flush=True)
    if not traces:
        print("# NO TRACE WRITTEN — profiler dir empty", flush=True)
        return
    t = traces[0]
    raw = gzip.open(t, "rt").read() if t.endswith(".gz") else open(t).read()
    data = json.loads(raw)
    evs = data.get("traceEvents", data) if isinstance(data, dict) else data
    by_cat: dict = {}
    by_name: dict = {}
    by_name_cnt: dict = {}
    total = 0.0
    for e in evs:
        if not isinstance(e, dict) or e.get("ph") != "X":
            continue
        cat = str(e.get("cat", ""))
        if cat not in ("kernel", "gpu_op", "Kernel"):   # GPU kernels only
            continue
        dur = float(e.get("dur", 0.0))
        nm = str(e.get("name", ""))
        b = bucket(nm)
        by_cat[b] = by_cat.get(b, 0.0) + dur
        by_name[nm] = by_name.get(nm, 0.0) + dur
        by_name_cnt[nm] = by_name_cnt.get(nm, 0) + 1
        total += dur
    if total <= 0:
        print("# no GPU-kernel events found (cat filter); dumping distinct cats seen:", flush=True)
        seen = {}
        for e in evs:
            if isinstance(e, dict) and e.get("ph") == "X":
                seen[str(e.get("cat", ""))] = seen.get(str(e.get("cat", "")), 0) + 1
        print(f"#   {seen}", flush=True)
        return

    # count key kernels -> all-reduce COUNT per layer / per decode token. The DSA decode kernel fires once
    # per layer per decode forward, so ar_per_layer = n_customAR / n_dsa (both scale with forwards*layers),
    # independent of how many decode steps the trace captured. n_layers from config gives per-token count.
    def cnt(*subs):
        return sum(c for k, c in by_name_cnt.items() if any(s in k.lower() for s in subs))
    n_customar = cnt("cross_device_reduce")
    n_ring_ar = cnt("allreduce_") + cnt("allreduce ")  # ncclDevKernel_AllReduce_*
    n_dsa = cnt("sparse_mla_decode", "sparse_mla")
    # n_layers from the model config. The model is already on the /cache volume (HF_HOME), so
    # from_pretrained reads the local cached config.json (no network); local_files_only makes that explicit.
    n_layers = 0
    try:
        from transformers import AutoConfig
        n_layers = int(getattr(AutoConfig.from_pretrained(MODEL, trust_remote_code=True,
                                                          local_files_only=True),
                               "num_hidden_layers", 0))
    except Exception:  # noqa: BLE001
        pass
    fwd = (n_dsa / n_layers) if (n_dsa and n_layers) else 0.0   # decode forward passes in the trace
    ar_active = n_customar or n_ring_ar
    ar_per_layer = (ar_active / n_dsa) if n_dsa else 0.0
    ar_per_tok = (ar_active / fwd) if fwd else 0.0
    step_ms = (total / fwd) if fwd else 0.0                     # GPU-busy ms per decode forward

    print(f"# === POST-C4 ROOFLINE (rank0 GPU-kernel time; n_layers={n_layers}, "
          f"decode_forwards~{fwd:.1f}, per-token GPU-busy~{step_ms/1000:.2f} ms) ===", flush=True)
    print(f"# all-reduce: custom_1stage={n_customar} ring={n_ring_ar} | DSA-decode={n_dsa} | "
          f"per-layer={ar_per_layer:.2f} per-token={ar_per_tok:.1f}", flush=True)
    print(f"# --- category (of {total/1000:.2f} ms GPU-busy total) ---", flush=True)
    for label, us in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        pt = (us / fwd / 1000) if fwd else 0.0
        print(f"#   {label:34s} {us/1000:8.2f} ms  {100*us/total:5.1f}%  ({pt:.2f} ms/tok)", flush=True)
    print("# --- top 15 kernels (time, %, count, count/tok) ---", flush=True)
    for nm, us in sorted(by_name.items(), key=lambda kv: -kv[1])[:15]:
        c = by_name_cnt.get(nm, 0)
        cpt = (c / fwd) if fwd else 0.0
        print(f"#   {us/1000:8.2f} ms  {100*us/total:4.1f}%  n={c:5d} ({cpt:4.1f}/tok)  {nm[:78]}", flush=True)
    print("# floor_profile DONE", flush=True)


@app.function(
    gpu="RTX-PRO-6000:2",
    timeout=60 * MIN,
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def baseline(tp: int = 2, eager: bool = False, max_len: int = 4096) -> None:
    import torch
    from vllm import LLM, SamplingParams

    import os

    # DeepGEMM's ue8m0 scale-factor transform asserts on SM120 ("Unknown SF transformation"); disable it
    # so vLLM routes FP8/scaled GEMMs to the FlashInfer/CUTLASS path that SM120 supports.
    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    print(
        f"# M4 baseline: {MODEL} tp={tp} eager={eager} on {torch.cuda.device_count()}x RTX-PRO-6000",
        flush=True,
    )
    # DeepSeek-V4 config uses rope_scaling {type: yarn}; newer configs want rope_type -> patch via hf_overrides.
    rope = {
        "rope_type": "yarn",
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "beta_fast": 32,
        "beta_slow": 1,
    }
    t0 = time.time()
    kw = dict(
        model=MODEL,
        tensor_parallel_size=tp,
        enforce_eager=eager,
        trust_remote_code=True,
        max_model_len=max_len,
        gpu_memory_utilization=0.9,
        kv_cache_dtype="fp8",
        hf_overrides={"rope_scaling": rope},
    )
    try:
        llm = LLM(tokenizer_mode="deepseek_v4", **kw)
    except Exception as ex:  # noqa: BLE001 -- deepseek_v4 tokenizer mode may not exist in this build
        print(
            f"  (deepseek_v4 tokenizer_mode rejected: {type(ex).__name__}; retrying default) ",
            flush=True,
        )
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

    mem = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"  per-GPU mem used (MB): {mem.splitlines()}", flush=True)
    print(
        f"# baseline {'PASS' if ntok > 0 else 'FAIL'} (graph_capture={'eager-OFF' if not eager else 'eager'})",
        flush=True,
    )


@app.function(
    gpu="RTX-PRO-6000:2",
    timeout=60 * MIN,
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def inspect_moe(tp: int = 2, max_len: int = 2048) -> None:
    # M3.6 recon: dump the exact FusedMoE structure so the sparse injection targets real attrs (model
    # is volume-cached after the first baseline load -> this loads fast).
    import inspect
    import os

    import torch
    from vllm import LLM

    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    os.environ.setdefault("QB_DENSE", "bf16")  # qb_sm120 plugin (bf16 dense) makes SM120 init work
    rope = {
        "rope_type": "yarn",
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "beta_fast": 32,
        "beta_slow": 1,
    }
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=tp,
        enforce_eager=True,
        trust_remote_code=True,
        max_model_len=max_len,
        gpu_memory_utilization=0.95,
        kv_cache_dtype="fp8",
        max_num_batched_tokens=max_len,
        hf_overrides={"rope_scaling": rope},
        tokenizer_mode="deepseek_v4",
    )
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
                    "top_k",
                    "num_experts",
                    "global_num_experts",
                    "local_num_experts",
                    "intermediate_size_per_partition",
                    "hidden_size",
                    "renormalize",
                    "use_grouped_topk",
                    "num_expert_group",
                    "topk_group",
                ):
                    print(f"  attr {an} = {v}", flush=True)
            if qm is not None and hasattr(qm, "apply"):
                try:
                    print(f"  apply{inspect.signature(qm.apply)}", flush=True)
                except (ValueError, TypeError):
                    pass
    print("\n# inspect_moe done", flush=True)


@app.function(gpu="RTX-PRO-6000", timeout=20 * MIN, volumes={"/cache": vol})
def test_so() -> None:
    # Step A de-risk: does the staged sm_120 quadbit kernel .so (built under CUDA 12.8) ctypes-load
    # and RUN on the CUDA-13 serve image / sm_120? Port moe_layer.py's pack/quant_act/seg_gemm and
    # run one synthetic segmented MoE matmul vs a bf16 reference. Expect finite output + the ~0.88
    # 2:4-FP4 accuracy tax (NOT ~1.0 -- sparse is lossy by design). Crash/garbage => must rebuild .so.
    import ctypes

    import torch
    import torch.nn.functional as F

    bn = 128
    dev = torch.device("cuda")
    print(
        f"# test_so: torch {torch.__version__} cuda {torch.version.cuda} "
        f"cap {torch.cuda.get_device_capability()}",
        flush=True,
    )
    so = None
    for cand in ("/cache/sparse_fp4_sm120.so", "/cache/sparse_fp4.so"):
        try:
            lib = ctypes.CDLL(cand)
            so = cand
            break
        except Exception as ex:  # noqa: BLE001
            print(f"  CDLL {cand} failed: {type(ex).__name__}: {ex}", flush=True)
    if so is None:
        print("# test_so FAIL: no loadable .so", flush=True)
        return
    print(f"  loaded {so}", flush=True)
    lib.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    lib.sparse_moe_mm_2lvl.argtypes = (
        [ctypes.c_void_p] * 6
        + [ctypes.c_int] * 4
        + [ctypes.c_void_p] * 3
        + [ctypes.c_int]
        + [ctypes.c_void_p]
    )
    lib.qb_init_moe_attrs()
    torch.manual_seed(0)

    fp4 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device=dev)
    bnd = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=dev)
    cc = torch.arange(128, device=dev)
    e_, m_ = (cc >> 3) & 0xF, cc & 7
    ue4m3 = torch.where(
        e_ == 0, m_.float() * 0.001953125, (1.0 + m_.float() / 8.0) * torch.exp2((e_ - 7).float())
    )

    def q_fp4(v):
        return torch.bucketize(v.abs(), bnd) | ((v < 0).long() << 3)

    def enc(s):
        mant_f, e = torch.frexp(s.clamp_min(1e-30))
        mm = 2.0 * mant_f
        biased = (e - 1) + 7
        mant = torch.round((mm - 1.0) * 8.0).long()
        carry = mant == 8
        mant = torch.where(carry, torch.zeros_like(mant), mant)
        biased = torch.where(carry, biased + 1, biased)
        code = (biased.long() << 3) | mant
        code = torch.where(biased < 1, torch.ones_like(code), code)
        code = torch.where(biased > 15, torch.full_like(code, 0x7F), code)
        code = torch.where(s >= 480.0, torch.full_like(code, 0x7F), code)
        return torch.where(s > 0, code, torch.zeros_like(code))

    def pack(w):
        out_f, in_f = w.shape
        ks = in_f // 128
        wg = w.float().to(dev).view(out_f, ks, 16, 4, 2)
        i01, _ = wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
        kept = torch.gather(wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
        ga = (
            (kept.abs().amax(dim=(1, 2, 3, 4), keepdim=True) / 2688.0)
            .clamp_min(1e-30)
            .reshape(out_f, 1, 1)
        )
        blk = kept.reshape(out_f, ks, 4, 8, 2)
        scode = enc((blk.abs().amax(dim=(3, 4)) / 6.0) / ga)
        sdeq = ue4m3[scode] * ga
        kc = q_fp4(blk / sdeq.clamp_min(1e-30)[..., None, None])
        ac = (kc[..., 0] | (kc[..., 1] << 4)).reshape(out_f, ks * 32).to(torch.uint8)
        nib = (i01[..., 0] | (i01[..., 1] << 2)).view(out_f, ks, 2, 8)
        sh = (torch.arange(8, device=dev) * 4).view(1, 1, 1, 8)
        meta = (nib << sh).sum(-1).to(torch.int32).permute(1, 0, 2).contiguous()
        return (
            ac.contiguous(),
            meta,
            scode.to(torch.uint8).permute(1, 0, 2).contiguous(),
            ga.reshape(out_f).float().contiguous(),
        )

    def quant_act(x):
        r, in_f = x.shape
        ks = in_f // 128
        x = x.to(torch.bfloat16).contiguous()
        bb = torch.empty((r, in_f // 2), dtype=torch.uint8, device=dev)
        sb = torch.empty((ks, r, 4), dtype=torch.uint8, device=dev)
        gb = torch.empty((r,), dtype=torch.float32, device=dev)
        lib.quantize_act_nvfp4_2lvl(
            x.data_ptr(), bb.data_ptr(), sb.data_ptr(), gb.data_ptr(), r, in_f
        )
        return bb, sb, gb

    # single-expert segmented call (eblk all-zero -> one expert, Mpe=M) as the minimal kernel exercise
    m_out, k_in, rows = 512, 1024, 256
    w = (torch.randn(m_out, k_in, device=dev) * (k_in**-0.5)).to(torch.bfloat16)
    x = (torch.randn(rows, k_in, device=dev) * (k_in**-0.5) * 4).to(torch.bfloat16)
    ac, meta, scale_a, ga = pack(w)
    bb, sb, gb = quant_act(x)
    c = torch.empty((rows, m_out), dtype=torch.bfloat16, device=dev)
    eblk = torch.zeros(rows // bn, dtype=torch.int32, device=dev)
    lib.sparse_moe_mm_2lvl(
        ac.data_ptr(),
        bb.data_ptr(),
        scale_a.data_ptr(),
        sb.data_ptr(),
        meta.data_ptr(),
        c.data_ptr(),
        ac.shape[0],
        m_out,
        rows,
        k_in,
        ga.data_ptr(),
        gb.data_ptr(),
        eblk.data_ptr(),
        1,
        0,
    )
    torch.cuda.synchronize()
    ref = F.linear(x.float(), w.float())
    nonfin = int((~torch.isfinite(c)).sum().item())
    cos = F.cosine_similarity(c.float().flatten(), ref.flatten(), dim=0).item()
    print(
        f"  seg_gemm ran: out={tuple(c.shape)} nonfin={nonfin} cos(seg,dense-bf16)={cos:.4f}",
        flush=True,
    )
    ok = nonfin == 0 and cos > 0.8
    print(
        f"# test_so {'PASS' if ok else 'FAIL'} (staged sm_120 .so runs on CUDA-13 serve image)",
        flush=True,
    )


@app.function(
    gpu="RTX-PRO-6000:2",
    timeout=60 * MIN,
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
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
    lib.sparse_moe_mm_2lvl.argtypes = (
        [ctypes.c_void_p] * 6
        + [ctypes.c_int] * 4
        + [ctypes.c_void_p] * 3
        + [ctypes.c_int]
        + [ctypes.c_void_p]
    )
    lib.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    lib.qb_init_moe_attrs()
    print(f"# M3.6 quadbit sparse-MoE injection: loaded {so}", flush=True)

    rope = {
        "rope_type": "yarn",
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "beta_fast": 32,
        "beta_slow": 1,
    }
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=tp,
        enforce_eager=eager,
        trust_remote_code=True,
        max_model_len=max_len,
        gpu_memory_utilization=0.9,
        kv_cache_dtype="fp8",
        hf_overrides={"rope_scaling": rope},
        tokenizer_mode="deepseek_v4",
    )
    m = llm.llm_engine.model_executor.driver_worker.model_runner.model

    # locate FusedMoE layers (finalized against inspect_moe output)
    moe = [
        (n, mod)
        for n, mod in m.named_modules()
        if "fusedmoe" in type(mod).__name__.lower() and hasattr(mod, "quant_method")
    ]
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
        "serve sparse yet; running native generate here would misrepresent the result."
    )


@app.function(image=image)
def versions() -> None:
    # CPU-only: pin down the vLLM<->FlashInfer skew behind the swa_topk_lens TypeError.
    import inspect
    from importlib.metadata import version

    for pkg in ["vllm", "flashinfer-python", "flashinfer", "torch"]:
        try:
            print(f"{pkg} == {version(pkg)}", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"{pkg}: {type(ex).__name__}", flush=True)
    try:
        import flashinfer

        print(f"flashinfer.__version__ = {getattr(flashinfer, '__version__', '?')}", flush=True)
        fn = getattr(flashinfer, "trtllm_batch_decode_sparse_mla_dsv4", None)
        if fn is None:
            import flashinfer.decode as fd

            fn = getattr(fd, "trtllm_batch_decode_sparse_mla_dsv4", None)
        print(
            f"trtllm_batch_decode_sparse_mla_dsv4 sig: "
            f"{inspect.signature(fn) if fn else 'NOT FOUND'}",
            flush=True,
        )
    except Exception as ex:  # noqa: BLE001
        print(f"flashinfer introspection failed: {type(ex).__name__}: {ex}", flush=True)


GLM_MODEL = "nvidia/GLM-5.2-NVFP4"


@app.function(
    image=image,
    timeout=30 * MIN,
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def glm_inspect() -> None:
    # CPU-only GLM-5.2 transfer feasibility probe: (1) does the installed vLLM register the
    # glm_moe_dsa model class, (2) exact NVFP4 checkpoint size + per-module-dtype breakdown from the
    # safetensors index (no model load), (3) 2/4/8x RTX-PRO-6000 (95 GiB) memory fit. No GPU needed.
    import json
    from collections import defaultdict
    from importlib.metadata import version

    from huggingface_hub import hf_hub_download

    print(f"# vllm == {version('vllm')}", flush=True)
    try:
        from vllm import ModelRegistry

        arches = sorted(ModelRegistry.get_supported_archs())
        glm = [a for a in arches if "glm" in a.lower() or "dsa" in a.lower()]
        print(f"# vLLM registry: glm/dsa arches = {glm}", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"# vLLM registry check failed: {type(ex).__name__}: {ex}", flush=True)

    cfg_path = hf_hub_download(GLM_MODEL, "config.json")
    cfg = json.loads(open(cfg_path).read())
    arch = cfg.get("architectures", ["?"])
    print(
        f"# config: model_type={cfg.get('model_type')} architectures={arch} "
        f"layers={cfg.get('num_hidden_layers')} hidden={cfg.get('hidden_size')} "
        f"experts={cfg.get('n_routed_experts')} topk={cfg.get('num_experts_per_tok')} "
        f"moe_int={cfg.get('moe_intermediate_size')} shared={cfg.get('n_shared_experts')} "
        f"first_k_dense={cfg.get('first_k_dense_replace')}",
        flush=True,
    )

    idx_path = hf_hub_download(GLM_MODEL, "model.safetensors.index.json")
    idx = json.loads(open(idx_path).read())
    total = idx.get("metadata", {}).get("total_size", 0)
    wmap = idx.get("weight_map", {})
    shards = sorted(set(wmap.values()))
    # bucket parameter names by role to see what is NVFP4 (uint8 experts) vs kept-precision
    buckets = defaultdict(int)
    for name in wmap:
        if ".experts." in name and "shared" not in name:
            key = "routed_experts"
        elif "shared_expert" in name:
            key = "shared_expert"
        elif any(a in name for a in ("self_attn", "attention", "q_proj", "kv", "o_proj")):
            key = "attention"
        else:
            key = "other"
        buckets[key] += 1
    gib = total / (1024**3)
    print(
        f"# checkpoint total_size = {total} bytes = {gib:.1f} GiB across {len(shards)} shards",
        flush=True,
    )
    print(f"# tensor-name buckets (count of entries): {dict(buckets)}", flush=True)
    usable = 94.97
    for ng in (2, 4, 8):
        cap = ng * usable
        fits = "FITS (weights only)" if gib < cap * 0.92 else "DOES NOT FIT"
        print(
            f"#   {ng}x RTX-PRO-6000 = {cap:.0f} GiB -> {fits} "
            f"(weights {gib:.0f} GiB, leaves {cap - gib:.0f} GiB for KV+act)",
            flush=True,
        )


@app.function(image=image)
def dumpsrc() -> None:
    # CPU-only: read the exact source of the deepseek_v4 o_proj DeepGEMM path so the SM120-safe
    # replacement is faithful (deep_gemm_fp8_o_proj bypasses Fp8LinearMethod -> our linear patch
    # never sees it; the DeepGEMM fp8_einsum asserts t.dim()==N on sm_120).
    import vllm

    base = Path(vllm.__file__).parent

    def show_full(label, p):
        print(f"\n===== {label}  ({p}) =====", flush=True)
        if not p.exists():
            print(f"  MISSING {p}", flush=True)
            return
        text = p.read_text().splitlines()
        print("\n".join(f"{i + 1:4} {ln}" for i, ln in enumerate(text)), flush=True)

    def show_defs(label, p, keys):
        print(f"\n===== {label}  ({p}) =====", flush=True)
        if not p.exists():
            print(f"  MISSING {p}", flush=True)
            return
        text = p.read_text().splitlines()
        for key in keys:
            hits = [i for i, ln in enumerate(text) if key in ln]
            for h in hits[:4]:
                lo, hi = max(0, h - 1), min(len(text), h + 45)
                print(f"  --- '{key}' @ line {h + 1} ---", flush=True)
                print("\n".join(f"{i + 1:4} {text[i]}" for i in range(lo, hi)), flush=True)

    show_defs(
        "deep_gemm.py mqa-logits",
        base / "utils/deep_gemm.py",
        [
            "def get_paged_mqa_logits_metadata",
            "def fp8_paged_mqa_logits",
            "def _lazy_init",
            "_get_paged_mqa_logits_metadata_impl =",
            "_fp8_paged_mqa_logits_impl =",
            "def _missing",
        ],
    )
    show_defs(
        "indexer.py backend",
        base / "v1/attention/backends/mla/indexer.py",
        [
            "get_paged_mqa_logits_metadata",
            "fp8_paged_mqa_logits",
            "def build",
            "scheduler_metadata_buffer",
        ],
    )
    show_defs(
        "flashinfer_sparse_mla_warmup.py",
        base / "model_executor/warmup/flashinfer_sparse_mla_warmup.py",
        ["def deepseek_v4_sparse_mla_attention_warmup"],
    )

    import subprocess

    def show_range(label, p, lo, hi):
        print(f"\n===== {label} ({p}) [{lo}-{hi}] =====", flush=True)
        text = p.read_text().splitlines()
        print(
            "\n".join(f"{i + 1:4} {text[i]}" for i in range(lo - 1, min(hi, len(text)))), flush=True
        )

    def show_range2(label, p, lo, hi):
        print(f"\n===== {label} [{lo}-{hi}] =====", flush=True)
        text = p.read_text().splitlines()
        print(
            "\n".join(f"{i + 1:4} {text[i]}" for i in range(lo - 1, min(hi, len(text)))), flush=True
        )

    import subprocess as _sp

    # SharedExperts.forward wants an `order` positional arg. Dump its signature + how the
    # native modelopt apply / model calls it so our patched_moe_apply passes `order` correctly.
    dv = base / "models/deepseek_v4/nvidia/model.py"
    dtext = dv.read_text().splitlines()
    for key in [
        "class SharedExperts",
        "def _forward_fused_moe",
        "shared_experts(",
        "SharedExperts(",
        "order",
    ]:
        hits = [i for i, ln in enumerate(dtext) if key in ln]
        for h in hits[:6]:
            lo, hi = max(0, h - 1), min(len(dtext), h + 20)
            print(f"\n--- model.py '{key}' @ {h + 1} ---", flush=True)
            print("\n".join(f"{i + 1:4} {dtext[i]}" for i in range(lo, hi)), flush=True)
    # find the SharedExperts class def + its forward(order) signature, tree-wide
    r = _sp.run(["grep", "-rn", r"class SharedExperts", str(base)], capture_output=True, text=True)
    print(f"\n### class SharedExperts defs ###\n{r.stdout}", flush=True)
    for ln in r.stdout.strip().splitlines():
        fp = ln.split(":")[0]
        if not fp.endswith(".py"):
            continue
        stext = Path(fp).read_text().splitlines()
        cs = [i for i, l in enumerate(stext) if "class SharedExperts" in l]
        for c in cs:
            print(f"\n--- {fp} SharedExperts [{c + 1}] ---", flush=True)
            print(
                "\n".join(f"{i + 1:4} {stext[i]}" for i in range(c, min(len(stext), c + 55))),
                flush=True,
            )
    # full shared_experts.py (forward + retrieval) and moe_runner apply/_maybe_apply 500-585
    se = base / "model_executor/layers/fused_moe/runner/shared_experts.py"
    stext = se.read_text().splitlines()
    print(f"\n### shared_experts.py [90-{len(stext)}] ###", flush=True)
    print("\n".join(f"{i + 1:4} {stext[i]}" for i in range(89, len(stext))), flush=True)
    mr = base / "model_executor/layers/fused_moe/runner/moe_runner.py"
    mt = mr.read_text().splitlines()
    print(f"\n### moe_runner.py [520-590] ###", flush=True)
    print("\n".join(f"{i + 1:4} {mt[i]}" for i in range(519, min(len(mt), 590))), flush=True)
    return  # SharedExperts recon only this run

    show_range2(
        "flashinfer_sparse.py _forward_prefill call",
        base / "models/deepseek_v4/nvidia/flashinfer_sparse.py",
        820,
        895,
    )
    show_defs(
        "vllm flashinfer.py wrapper",
        base / "utils/flashinfer.py",
        ["trtllm_batch_decode_sparse_mla_dsv4", "def has_flashinfer_sparse_mla_sm120"],
    )
    import subprocess

    r = subprocess.run(
        ["pip", "index", "versions", "flashinfer-python"], capture_output=True, text=True
    )
    print(f"\n### pip index versions flashinfer-python:\n{r.stdout}\n{r.stderr}", flush=True)
    r2 = subprocess.run(
        [
            "grep",
            "-rn",
            "flashinfer",
            str(base.parent / "vllm-0.24.0.dist-info")
            if (base.parent / "vllm-0.24.0.dist-info").exists()
            else str(base),
        ],
        capture_output=True,
        text=True,
    )
    print(f"\n### vllm flashinfer pin refs:\n{r2.stdout[:1500]}", flush=True)


@app.function(image=image)
def c1_recon() -> None:
    # C1: does a native fused NVFP4 grouped primitive already exist that can express the dense-anchor
    # branch? Two shapes matter: (1) D2 route-slot dense group = a FULL MoE over the top-N slots ->
    # needs ModelOptNvFp4FusedMoE.apply to consume EXTERNAL topk_ids/topk_weights (so we pass top-2);
    # (2) down49/gateup49 single-projection anchor -> needs a STANDALONE grouped NVFP4 GEMM (fused MoE
    # can't emit half a MoE). Dump the apply/process_weights source + the flashinfer fused_moe/gemm
    # API surface so the design note is grounded, not guessed.
    import inspect

    import vllm

    base = Path(vllm.__file__).parent

    def defs(label, p, keys, ctx=55):
        print(f"\n===== {label}  ({p}) =====", flush=True)
        if not p.exists():
            print(f"  MISSING {p}", flush=True)
            return
        text = p.read_text().splitlines()
        for key in keys:
            for h in [i for i, ln in enumerate(text) if key in ln][:3]:
                lo, hi = max(0, h - 1), min(len(text), h + ctx)
                print(f"  --- '{key}' @ {h + 1} ---", flush=True)
                print("\n".join(f"{i + 1:4} {text[i]}" for i in range(lo, hi)), flush=True)

    mo = base / "model_executor/layers/quantization/modelopt.py"
    defs("modelopt ModelOptNvFp4FusedMoE", mo,
         ["class ModelOptNvFp4FusedMoE",
          "def process_weights_after_loading",
          "def apply",
          "def select_gemm_impl",
          "flashinfer",
          "cutlass_fused_moe",
          "trtllm"], ctx=70)

    # flashinfer API surface: which grouped/MoE/gemm primitives exist + signatures.
    print("\n===== flashinfer API surface =====", flush=True)
    try:
        import flashinfer
        print("flashinfer", getattr(flashinfer, "__version__", "?"), flush=True)
        for modname in ["fused_moe", "gemm"]:
            try:
                mod = __import__(f"flashinfer.{modname}", fromlist=["x"])
            except Exception as ex:  # noqa: BLE001
                print(f"  flashinfer.{modname}: import FAIL {type(ex).__name__}: {ex}", flush=True)
                continue
            names = [n for n in dir(mod) if not n.startswith("_")]
            print(f"\n  --- flashinfer.{modname} public names ---\n  {names}", flush=True)
            for n in names:
                obj = getattr(mod, n)
                if callable(obj) and any(k in n.lower() for k in
                                         ("moe", "group", "gemm", "fp4", "mxfp4", "nvfp4")):
                    try:
                        print(f"    {n}{inspect.signature(obj)}", flush=True)
                    except (ValueError, TypeError):
                        print(f"    {n}(...)", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"  flashinfer import FAIL {type(ex).__name__}: {ex}", flush=True)

    # vllm's own cutlass_moe / fp4 grouped custom ops (another possible standalone-GEMM source).
    import subprocess
    r = subprocess.run(["grep", "-rln", "-e", "cutlass_moe_fp4", "-e", "group_gemm",
                        "-e", "grouped_gemm", str(base)], capture_output=True, text=True)
    print(f"\n### vllm files mentioning fp4/grouped gemm:\n{r.stdout}", flush=True)


@app.function(image=image)
def c1_recon2() -> None:
    # C1 layer-2: the exact tensor contract for group_gemm_nvfp4_nt_groupwise (drop-in for the slow
    # _dense_seg_gs) + the FlashInfer NVFP4 activation-quant helper (to feed `a`/`a_scale`) + a real
    # usage example from the FlashInfer test suite (scale swizzle is fiddly; copy a working call).
    import inspect

    import flashinfer
    from flashinfer import gemm as figemm

    print(f"flashinfer {flashinfer.__version__}\n", flush=True)
    for fn in ["group_gemm_nvfp4_nt_groupwise", "mm_fp4"]:
        obj = getattr(figemm, fn, None)
        if obj is None:
            print(f"### {fn}: MISSING", flush=True)
            continue
        print(f"\n===== flashinfer.gemm.{fn} SOURCE =====", flush=True)
        try:
            print(inspect.getsource(obj), flush=True)
        except (OSError, TypeError) as ex:
            print(f"  no source: {ex}; doc:\n{obj.__doc__}", flush=True)

    # NVFP4 activation quant helpers exposed at flashinfer top level / in flashinfer.fp4_quantization.
    print("\n===== flashinfer nvfp4 quant helpers =====", flush=True)
    top = [n for n in dir(flashinfer) if any(k in n.lower()
           for k in ("fp4", "nvfp4", "quantize", "scale_factor", "swizzle"))]
    print(f"  top-level: {top}", flush=True)
    for n in top:
        obj = getattr(flashinfer, n)
        if callable(obj):
            try:
                print(f"    {n}{inspect.signature(obj)}", flush=True)
            except (ValueError, TypeError):
                pass
    for modname in ["fp4_quantization", "utils"]:
        try:
            mod = __import__(f"flashinfer.{modname}", fromlist=["x"])
            names = [n for n in dir(mod) if any(k in n.lower()
                     for k in ("fp4", "quantize", "scale", "swizzle", "block_scale"))]
            print(f"  flashinfer.{modname}: {names}", flush=True)
            for n in names:
                obj = getattr(mod, n)
                if callable(obj):
                    try:
                        print(f"    {n}{inspect.signature(obj)}", flush=True)
                    except (ValueError, TypeError):
                        pass
        except Exception as ex:  # noqa: BLE001
            print(f"  flashinfer.{modname}: {type(ex).__name__}: {ex}", flush=True)

    # a real working call: find the test that exercises group_gemm_nvfp4 (scale layout copied from it).
    import subprocess
    fi = Path(flashinfer.__file__).parent
    for root in [fi, fi.parent]:
        r = subprocess.run(["grep", "-rl", "group_gemm_nvfp4", str(root)],
                           capture_output=True, text=True)
        if r.stdout.strip():
            print(f"\n### files using group_gemm_nvfp4 under {root}:\n{r.stdout}", flush=True)
            break
    # also dump the cutlass_fused_moe quant_scales contract (the D2 both-proj alternative).
    from flashinfer import fused_moe as fimoe
    print("\n===== cutlass_fused_moe doc (quant_scales contract) =====", flush=True)
    print((fimoe.cutlass_fused_moe.__doc__ or "no doc")[:3000], flush=True)


@app.function(gpu="RTX-PRO-6000", timeout=30 * MIN, volumes={"/cache": vol})
def c1_dense_anchor() -> None:
    # C1 standalone A/B (validation-ladder A): can flashinfer's native grouped NVFP4 GEMM
    # (group_gemm_nvfp4_nt_groupwise) express the dense-anchor projection that _dense_seg_gs runs as a
    # decode-slow range(E) dequant loop? Same NVFP4 weights + same fixed-cap seg layout, both paths:
    #   A (old) = per-expert dequant-to-bf16 then bf16 matmul (mirror of plugin _dense_seg_gs)
    #   B (native) = one group_gemm_nvfp4 call over the whole seg buffer (acts also NVFP4-quantized)
    # Report cos/relL2/maxabs/nonfinite vs the true-bf16 grouped matmul AND vs each other, eager +
    # graph-replay latency, and capture success. This is the GATE: if B cannot express the branch here
    # (layout / scales / capture / quality), stop and write the failure table before touching serving.
    import statistics

    import torch
    import torch.nn.functional as F
    from flashinfer import SfLayout, nvfp4_quantize
    from flashinfer.gemm import group_gemm_nvfp4_nt_groupwise

    dev = torch.device("cuda")
    torch.manual_seed(0)
    print(f"# torch {torch.__version__} | {torch.cuda.get_device_name(0)} "
          f"cap {torch.cuda.get_device_capability(0)} | flashinfer group_gemm_nvfp4", flush=True)
    SFV = 448.0 * 6.0  # (e4m3 max block-scale) * (e2m1 max element)

    def quantize_group_inputs(a_float, b_float, m_indptr):
        # Verbatim port of flashinfer tests/gemm/test_group_gemm_fp4.py::_quantize_nvfp4_group_inputs:
        # per-group nvfp4_quantize (128x4 layout, no shuffle) + a_scale placed at 128-aligned swizzle
        # offsets. .clamp_min guards the empty-expert (all-zero) segment (max=0 -> inf global scale).
        align = 128
        a_fp4_c, a_sf_c, b_fp4_c, b_sf_c, alpha_c = [], [], [], [], []
        for a_g, b_g in zip(a_float, b_float, strict=True):
            a_gsf = SFV / a_g.float().abs().nan_to_num().max().clamp_min(1e-6)
            b_gsf = SFV / b_g.float().abs().nan_to_num().max().clamp_min(1e-6)
            a_q, a_s = nvfp4_quantize(a_g, a_gsf, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
            b_q, b_s = nvfp4_quantize(b_g, b_gsf, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
            a_fp4_c.append(a_q); a_sf_c.append(a_s); b_fp4_c.append(b_q); b_sf_c.append(b_s)
            alpha_c.append(1.0 / (a_gsf * b_gsf))
        ng = a_float.shape[0]
        sf_k = a_sf_c[0].shape[1]
        mi = m_indptr.cpu().tolist()
        last = ng - 1
        last_off = (mi[last] + last * (align - 1)) // align * align
        total_sf = last_off + a_sf_c[last].shape[0]
        a_sf_pad = torch.zeros((total_sf, sf_k), dtype=a_sf_c[0].dtype, device=a_float.device)
        for i in range(ng):
            off = (mi[i] + i * (align - 1)) // align * align
            a_sf_pad[off:off + a_sf_c[i].shape[0]] = a_sf_c[i]
        return (torch.cat(a_fp4_c, 0), torch.stack(b_fp4_c, 0), a_sf_pad, torch.stack(b_sf_c, 0),
                torch.tensor(alpha_c, dtype=torch.float32, device=a_float.device))

    def stats_ms(fn, it=30):
        for _ in range(5):
            fn()
        torch.cuda.synchronize(); ts = []
        for _ in range(it):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
        return statistics.median(sorted(ts))

    def cos(x, y):
        return F.cosine_similarity(x.float().flatten(), y.float().flatten(), dim=0).item()

    def relL2(x, y):
        return (x.float() - y.float()).norm().item() / y.float().norm().clamp_min(1e-9).item()

    # DeepSeek-V4-Flash MoE shapes: hidden H=4096, moe_intermediate I=2048.
    # gate_up projection: in=H=4096, out=2I=4096. down projection: in=I=2048, out=H=4096.
    H, I = 4096, 2048
    cap = 128
    for tag, (K, N) in [("gate_up", (H, 2 * I)), ("down", (I, H))]:
        for E in [32, 64]:
            M = E * cap
            W = (torch.randn(E, N, K, device=dev, dtype=torch.bfloat16) * (K ** -0.5))
            X = (torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.1)
            for le in range(0, E, 4):  # every 4th expert empty (zero rows) -> empty-seg case
                X[le * cap:(le + 1) * cap] = 0

            # --- quantize weights + activations to NVFP4 per the flashinfer test recipe ---
            m_indptr = (torch.arange(E + 1, device=dev, dtype=torch.int32) * cap)
            a_fp4, b_w, a_sf_arg, b_sf_arg, alpha = quantize_group_inputs(
                X.view(E, cap, K), W, m_indptr)

            if E == 32 and tag == "gate_up":
                print(f"# shapes: b_w {tuple(b_w.shape)} b_sf_arg {tuple(b_sf_arg.shape)} "
                      f"a_fp4 {tuple(a_fp4.shape)} a_sf_arg {tuple(a_sf_arg.shape)} "
                      f"alpha {tuple(alpha.shape)} m_indptr {tuple(m_indptr.shape)}", flush=True)

            # --- reference: true-bf16 grouped matmul (ceiling) + old-path dequant-loop analogue ---
            def ref_bf16():
                out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
                for le in range(E):
                    out[le * cap:(le + 1) * cap] = (
                        X[le * cap:(le + 1) * cap].float() @ W[le].float().t()).to(torch.bfloat16)
                return out
            y_ref = ref_bf16()
            eager_a = stats_ms(ref_bf16)

            # --- native path B: group_gemm_nvfp4 ---
            res = {"tag": tag, "E": E}
            try:
                def native():
                    return group_gemm_nvfp4_nt_groupwise(
                        a_fp4, b_w, a_sf_arg, b_sf_arg, m_indptr, alpha=alpha,
                        out_dtype=torch.bfloat16)
                y_nat = native()
                res["cos_nat_ref"] = cos(y_nat, y_ref)
                res["relL2"] = relL2(y_nat, y_ref)
                res["maxabs"] = (y_nat.float() - y_ref.float()).abs().amax().item()
                res["nf"] = int((~torch.isfinite(y_nat)).sum().item())
                res["eager_b"] = stats_ms(native)
                # capture
                try:
                    g = torch.cuda.CUDAGraph()
                    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(st):
                        for _ in range(3):
                            native()
                    torch.cuda.current_stream().wait_stream(st)
                    with torch.cuda.graph(g):
                        y_cap = native()
                    for _ in range(3):
                        g.replay()
                    torch.cuda.synchronize()
                    res["cos_cap"] = cos(y_cap, y_ref)
                    res["graph_b"] = stats_ms(lambda: g.replay())
                    res["capture"] = "OK"
                except Exception as ex:  # noqa: BLE001
                    res["capture"] = f"FAIL {type(ex).__name__}: {str(ex)[:120]}"
            except Exception as ex:  # noqa: BLE001
                res["native"] = f"FAIL {type(ex).__name__}: {str(ex)[:200]}"
            res["eager_a"] = eager_a
            print(f"# ROW {res}", flush=True)
            del W, X, b_w, a_fp4
            torch.cuda.empty_cache()
    print("# c1_dense_anchor done", flush=True)


@app.function(image=image)
def probe() -> None:
    # CPU-only: confirm which plugin code is actually baked into the image (guards against pip
    # skipping a same-version reinstall). Prints the installed version + the dequant source.
    import inspect
    from importlib.metadata import entry_points, version

    import torch

    import qb_sm120_plugin as p

    print(f"# qb-sm120-plugin version={version('qb-sm120-plugin')} file={p.__file__}", flush=True)
    eps = [e.value for e in entry_points(group="vllm.general_plugins")]
    print(f"# vllm.general_plugins entry points: {eps}", flush=True)
    print(inspect.getsource(p._dequant_block), flush=True)
    # call the ACTUAL installed function on the real failing shape (w[16384,1024], s[128,8])
    for sshape in [(128, 8), (8, 128), (12, 32)]:
        wh = (16384, 1024) if sshape != (12, 32) else (1536, 4096)
        w = torch.randint(-7, 8, wh, dtype=torch.float32)
        s = torch.rand(*sshape) + 0.1
        try:
            out = p._dequant_block(w, s, (128, 128))
            print(f"  CALL p._dequant_block w={wh} s={sshape} -> {tuple(out.shape)} OK", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(
                f"  CALL p._dequant_block w={wh} s={sshape} -> {type(ex).__name__}: {ex}",
                flush=True,
            )


def _sweep_impl(tp: int, runtag: str, matrix: list, instr: bool, tax: bool, max_len: int) -> None:
    """WS3: quadbit 2:4 sparse-FP4 expert performance sweep. Loads DeepSeek-V4-Flash-NVFP4 once with
    QB_MOE=sparse on `tp` GPUs, then times a prompt x gen x batch matrix, emitting figure-ready CSV
    rows + a compute/comm breakdown (from the plugin's per-worker instrumentation flushed to /cache)
    + the in-situ per-layer tax probe. Eager only (graph capture unproven on this SM120 path)."""
    import glob
    import json
    import os
    import statistics as st
    import subprocess
    import time

    import torch
    from vllm import LLM, SamplingParams

    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    os.environ["QB_DENSE"] = "bf16"
    os.environ["QB_MOE"] = "sparse"
    os.environ["QB_RUNTAG"] = runtag
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    if instr:
        os.environ["QB_INSTR"] = "1"
    if tax:
        os.environ["QB_TAXPROBE"] = "1"
    # clear stale per-worker metric files for this runtag
    for f in glob.glob(f"/cache/qb_metrics_{runtag}_dev*.json"):
        os.remove(f)

    ngpu = torch.cuda.device_count()
    print(
        f"# WS3 sweep tp={tp} runtag={runtag} instr={instr} tax={tax} on {ngpu}x RTX-PRO-6000; "
        f"commit=$(git) matrix={matrix}",
        flush=True,
    )

    def smi():
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        ).stdout
        used, free = [], []
        for line in out.strip().splitlines():
            u, fr = line.split(",")
            used.append(int(u))
            free.append(int(fr))
        return used, free

    rope = {
        "rope_type": "yarn",
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "beta_fast": 32,
        "beta_slow": 1,
    }
    kw = dict(
        model=MODEL,
        tensor_parallel_size=tp,
        enforce_eager=True,
        trust_remote_code=True,
        max_model_len=max_len,
        gpu_memory_utilization=0.95,
        kv_cache_dtype="fp8",
        max_num_batched_tokens=max(2048, max_len),
        hf_overrides={"rope_scaling": rope},
        disable_log_stats=False,
    )
    t0 = time.time()
    llm = LLM(tokenizer_mode="deepseek_v4", **kw)
    load_s = time.time() - t0
    load_used, load_free = smi()
    print(f"  loaded in {load_s:.0f}s; per-GPU used MB={load_used} free MB={load_free}", flush=True)

    tok = llm.get_tokenizer()
    base = tok.encode("The quick brown fox jumps over the lazy dog and then keeps running. ")

    def make_prompt(n):
        ids = (base * (n // len(base) + 1))[:n]
        return {"prompt_token_ids": ids}

    # single global warmup so per-config timings exclude first-call JIT/autotune
    llm.generate([make_prompt(128)], SamplingParams(temperature=0.0, max_tokens=8, min_tokens=8))

    rows = []
    peak_used = list(load_used)

    def med(x):
        return st.median(x) if x else 0.0

    for P, G, B in matrix:
        prompts = [make_prompt(P) for _ in range(B)]
        sp = SamplingParams(temperature=0.0, max_tokens=G, min_tokens=G, ignore_eos=True)
        torch.cuda.synchronize()
        t1 = time.time()
        try:
            outs = llm.generate(prompts, sp)
        except Exception as e:
            # engine death (e.g. long-prefill scheduler KeyError) must not lose completed rows
            print(
                f"  [cfg] P={P} G={G} B={B}: FAILED {type(e).__name__}: {str(e)[:160]}", flush=True
            )
            break
        wall = time.time() - t1
        u, _fr = smi()
        peak_used = [max(a, b) for a, b in zip(peak_used, u)]
        # decode time from the monotonic (*_ts) pair only; arrival_time is unix-epoch (different clock),
        # so TTFT is derived from wall - decode instead of mixing clocks.
        decs, tpots = [], []
        gen_tok = 0
        for o in outs:
            gen_tok += len(o.outputs[0].token_ids)
            m = o.metrics
            if m is None:
                continue
            ft = getattr(m, "first_token_time", None) or getattr(m, "first_token_ts", None)
            lt = getattr(m, "last_token_time", None) or getattr(m, "last_token_ts", None)
            fin = getattr(m, "finished_time", None) or getattr(m, "finished_ts", None) or lt
            if fin and ft and G > 1:
                d = fin - ft
                decs.append(d)
                tpots.append(d / (G - 1))

        dec_wall = med(decs)
        ttft = max(0.0, wall - dec_wall)  # B=1: exact; B>1: proxy (requests overlap)
        tpot = med(tpots)
        prefill_tps = (B * P) / ttft if ttft > 0 else 0.0
        dec_tps = (B * (G - 1)) / dec_wall if dec_wall > 0 else 0.0
        tot_tps = gen_tok / wall if wall > 0 else 0.0
        row = {
            "tp": tp,
            "prompt": P,
            "gen": G,
            "batch": B,
            "wall_s": round(wall, 3),
            "ttft_s": round(ttft, 4),
            "tpot_ms": round(tpot * 1000, 3),
            "prefill_tps": round(prefill_tps, 1),
            "decode_tps": round(dec_tps, 1),
            "total_tps": round(tot_tps, 1),
            "req_latency_s": round(wall, 3),
            "gen_tok": gen_tok,
        }
        rows.append(row)
        print(
            f"  [cfg] P={P} G={G} B={B}: wall={wall:.1f}s ttft={ttft:.3f}s "
            f"tpot={tpot * 1000:.1f}ms prefill={prefill_tps:.1f}tok/s decode={dec_tps:.1f}tok/s "
            f"total={tot_tps:.1f}tok/s",
            flush=True,
        )

    print(f"  peak per-GPU used MB={peak_used}", flush=True)

    # aggregate per-worker instrumentation + tax from /cache
    metric_files = sorted(glob.glob(f"/cache/qb_metrics_{runtag}_dev*.json"))
    agg = {
        "expert_ms": 0.0,
        "dense_ms": 0.0,
        "expert_calls": 0,
        "sparse_expert_calls": 0,
        "imbalance_mean": [],
        "imbalance_max": [],
        "tax_cos": [],
    }
    for mf in metric_files:
        with open(mf) as f:
            d = json.load(f)
        agg["expert_ms"] += d["t_ms"].get("expert", 0.0)
        agg["dense_ms"] += d["t_ms"].get("dense", 0.0)
        agg["expert_calls"] += d["counts"].get("expert_calls", 0)
        agg["sparse_expert_calls"] += d["stats"].get("sparse_expert_calls", 0)
        agg["imbalance_mean"].append(d.get("imbalance_mean", 0.0))
        agg["imbalance_max"].append(d.get("imbalance_max", 0.0))
        agg["tax_cos"].extend(d.get("tax_cos", []))
    print(f"\n# --- instrumentation (ranks={len(metric_files)}) ---", flush=True)
    print(
        f"  expert-kernel ms(sum over ranks)={agg['expert_ms']:.1f} "
        f"dense/attn ms={agg['dense_ms']:.1f} expert_calls={agg['expert_calls']} "
        f"SPARSE_EXPERT_CALLS={agg['sparse_expert_calls']}",
        flush=True,
    )
    if agg["imbalance_mean"]:
        print(
            f"  expert imbalance (max/mean tokens-per-expert): "
            f"mean={st.mean(agg['imbalance_mean']):.3f} max={max(agg['imbalance_max']):.3f}",
            flush=True,
        )
    cosv = sorted(c["cos"] for c in agg["tax_cos"])
    if cosv:

        def pct(p):
            return cosv[min(len(cosv) - 1, int(p * len(cosv)))]

        worst = min(agg["tax_cos"], key=lambda c: c["cos"])
        print(f"\n# --- in-situ per-layer tax cos(sparse,dense) (n={len(cosv)}) ---", flush=True)
        print(
            f"  median={st.median(cosv):.4f} p10={pct(0.1):.4f} p90={pct(0.9):.4f} "
            f"min={cosv[0]:.4f} max={cosv[-1]:.4f}",
            flush=True,
        )
        print(
            f"  worst: layer={worst['layer']} expert={worst['expert']} cos={worst['cos']:.4f}",
            flush=True,
        )

    # figure-ready CSV to /cache
    csv_path = f"/cache/qb_sweep_{runtag}.csv"
    cols = [
        "tp",
        "prompt",
        "gen",
        "batch",
        "wall_s",
        "ttft_s",
        "tpot_ms",
        "prefill_tps",
        "decode_tps",
        "total_tps",
        "req_latency_s",
        "gen_tok",
    ]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\n# CSV rows ({csv_path}):", flush=True)
    print(",".join(cols), flush=True)
    for r in rows:
        print(",".join(str(r[c]) for c in cols), flush=True)
    print(
        f"# sweep DONE tp={tp} runtag={runtag} load={load_s:.0f}s "
        f"load_used_MB={load_used} peak_used_MB={peak_used}",
        flush=True,
    )


# default matrix: representative, bounded for eager (~3 tok/s) within the Modal timeout.
_SWEEP_MATRIX = [
    (512, 32, 1),
    (512, 128, 1),
    (512, 32, 8),
    (512, 128, 8),
    (2048, 32, 1),
    (2048, 128, 1),
    (2048, 32, 8),
    (512, 512, 1),
    (8192, 32, 1),
]


@app.function(
    gpu="RTX-PRO-6000:2",
    timeout=150 * MIN,
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def sweep2(instr: bool = True, tax: bool = True, max_len: int = 8960) -> None:
    _sweep_impl(2, "tp2", _SWEEP_MATRIX, instr, tax, max_len)


@app.function(
    gpu="RTX-PRO-6000:4",
    timeout=150 * MIN,
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def sweep4(instr: bool = True, tax: bool = True, max_len: int = 8960) -> None:
    _sweep_impl(4, "tp4", _SWEEP_MATRIX, instr, tax, max_len)


def _calib_impl(
    tp: int,
    tag: str,
    max_len: int,
    dump: bool = False,
    sparse_dump: bool = False,
    dense_layers: str = "",
    calib_file: str = "",
) -> None:
    """A2 Step-2 calibration: run DeepSeek-V4-Flash DENSE (coherent) over a text corpus and let the
    plugin accumulate per-expert per-projection column activation norms from REAL routed tokens,
    dumped per-rank to /cache/qb_calib_{tag}_dev*.pt for the Wanda 2:4 mask in a later sparse run.
    dump=True also captures per-sparse-layer MoE block I/O (x, routing, dense y) to
    /cache/qb_reconio_{tag}_dev*.pt for A3 layerwise repair (same teacher forward, no extra pass)."""
    import os
    import time

    import torch
    from vllm import LLM, SamplingParams

    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    os.environ["QB_DENSE"] = "nvfp4"
    if sparse_dump:
        # A3 reio2: run the exact A2-49 serving config (sparse MoE + dense anchors + Wanda mask) so
        # the dumped per-layer input x is the SERVE-CONSISTENT sparse trajectory, then dump it for
        # error-correcting recon. No QB_CALIB here (the mask already exists as calib_file).
        os.environ["QB_MOE"] = "sparse"
        os.environ["QB_CALIB_FILE"] = calib_file
        os.environ["QB_DENSE_LAYERS"] = dense_layers
        os.environ["QB_DUMP"] = "1"
    else:
        os.environ["QB_MOE"] = "dense"
        os.environ["QB_CALIB"] = "1"
        if dump:
            os.environ["QB_DUMP"] = "1"
    os.environ["QB_RUNTAG"] = tag
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    print(f"# calib tp={tp} tag={tag} on {torch.cuda.device_count()}x RTX-PRO-6000", flush=True)

    rope = {
        "rope_type": "yarn",
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "beta_fast": 32,
        "beta_slow": 1,
    }
    t0 = time.time()
    llm = LLM(
        model=MODEL,
        tokenizer_mode="deepseek_v4",
        tensor_parallel_size=tp,
        enforce_eager=True,
        trust_remote_code=True,
        max_model_len=max_len,
        gpu_memory_utilization=0.9,
        kv_cache_dtype="fp8",
        max_num_batched_tokens=max(2048, max_len),
        hf_overrides={"rope_scaling": rope},
    )
    print(f"  loaded in {time.time() - t0:.0f}s", flush=True)

    tok = llm.get_tokenizer()
    corpus = [
        "The history of science spans many centuries and cultures, from ancient Babylonian astronomy to "
        "modern quantum mechanics and relativity. Researchers formulate hypotheses, design controlled "
        "experiments, gather data, and revise theories as new evidence emerges. The scientific method "
        "relies on reproducibility, falsifiability, and rigorous peer review by the wider community.",
        "In computer science, algorithms and data structures form the foundation of efficient software. "
        "A sorting algorithm arranges elements in order; a hash table offers average constant-time "
        "lookup; a balanced tree keeps operations logarithmic. def quicksort(a):\n    if len(a) <= 1:\n"
        "        return a\n    p = a[len(a)//2]\n    lo = [x for x in a if x < p]\n    hi = [x for x in a "
        "if x > p]\n    return quicksort(lo) + [x for x in a if x == p] + quicksort(hi)\n\nclass Stack:\n"
        "    def __init__(self):\n        self.items = []\n    def push(self, x):\n        self.items.append(x)",
        "Economics studies how societies allocate scarce resources among competing uses. Supply and "
        "demand jointly determine prices in competitive markets, while central banks influence interest "
        "rates, employment, and inflation through monetary policy. International trade allows nations to "
        "specialize according to comparative advantage, raising aggregate output but creating winners and losers.",
        "The novel opened on a grey morning in a small coastal town, where the fishermen mended their "
        "nets and the gulls circled overhead, crying into the wind. She walked along the weathered pier, "
        "thinking of the letter she had never sent, the words still forming and dissolving in her mind "
        "like the restless tide against the pilings below.",
        "Photosynthesis is the process by which green plants convert light energy into chemical energy. "
        "Chlorophyll in the chloroplasts absorbs sunlight, water is split into hydrogen and oxygen, and "
        "atmospheric carbon dioxide is fixed into glucose through the Calvin cycle. This process releases "
        "the oxygen we breathe and sustains nearly all life on Earth through the food chain.",
        "Mathematics is the study of numbers, structure, space, and change. A prime number has exactly "
        "two positive divisors, one and itself. The Pythagorean theorem states that for a right triangle "
        "the square of the hypotenuse equals the sum of the squares of the other two sides. Calculus "
        "formalizes rates of change through derivatives and accumulation through integrals.",
        "The human body is composed of many interacting systems. The cardiovascular system circulates "
        "blood, delivering oxygen and nutrients while removing waste. The nervous system transmits "
        "electrical signals between the brain and the rest of the body. The immune system defends "
        "against pathogens using white blood cells, antibodies, and inflammatory responses.",
        "Constitutional law governs the relationship between the state and its citizens. Courts interpret "
        "statutes and precedents, balancing individual rights against public interest. The principle of "
        "separation of powers divides authority among the legislative, executive, and judicial branches "
        "to prevent the concentration of power and protect against tyranny.",
        "To make a classic risotto, warm the stock in a saucepan and keep it simmering. In a separate "
        "pan, soften diced onion in butter, add the rice, and toast it briefly. Add a ladle of stock at "
        "a time, stirring constantly until absorbed, and continue until the grains are creamy yet still "
        "firm at the centre. Finish with parmesan and a knob of cold butter.",
        "The orchestra tuned their instruments as the conductor stepped onto the podium. The symphony "
        "began softly with the strings, then the woodwinds entered, and finally the brass and percussion "
        "joined in a triumphant crescendo. Music theory describes harmony, melody, rhythm, and the "
        "intervals that give a chord its characteristic tension or resolution.",
        "Climate scientists study the long-term patterns of temperature, precipitation, and atmospheric "
        "composition. Rising concentrations of greenhouse gases trap heat, warming the oceans and melting "
        "polar ice. Feedback loops, such as reduced reflectivity from vanishing sea ice, can amplify these "
        "changes and shift weather patterns across entire continents.",
        "Ancient Rome grew from a small city-state on the Tiber into an empire spanning three continents. "
        "Its legions, roads, aqueducts, and legal code shaped the Mediterranean world for centuries. "
        "Latin, the language of Rome, evolved into the Romance languages and contributed an enormous "
        "vocabulary to English, science, and law.",
    ]
    # Dense expert coverage needs many tokens per routed expert (256 experts, top-6): the 12
    # inline paragraphs (~840 tok) give ~3 tok/expert -> colnorm is noise. Pull real diverse text
    # so per-expert routing stats stabilise. wikitext (prose) + a code slice for domain spread.
    try:
        from datasets import load_dataset  # noqa: PLC0415

        ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=True)
        extra: list[str] = []
        for row in ds:
            txt = row["text"].strip()
            if len(txt) < 400:
                continue
            extra.append(txt[:1400])  # ~300 tokens/chunk
            if len(extra) >= 300:  # ~90k tokens -> ~2000 routed tok/expert avg
                break
        corpus = corpus + extra
        print(f"  loaded {len(extra)} pile chunks", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  dataset load failed ({e}); using inline corpus only", flush=True)

    ids = [{"prompt_token_ids": tok.encode(c)} for c in corpus]
    ntok = sum(len(d["prompt_token_ids"]) for d in ids)
    print(
        f"  calibrating over {len(corpus)} chunks / {ntok} tokens (dense forwards)...", flush=True
    )
    llm.generate(ids, SamplingParams(temperature=0.0, max_tokens=1))
    # Each WORKER accumulates + dumps its own shard (periodically during the forwards + atexit on
    # shutdown); the driver has no model state. Free the engine to trigger worker shutdown, then check.
    del llm
    import gc
    import glob

    gc.collect()
    time.sleep(5)
    if not sparse_dump:
        files = sorted(glob.glob(f"/cache/qb_calib_{tag}_dev*.pt"))
        sizes = [f"{os.path.basename(f)}={os.path.getsize(f) // 1024}KB" for f in files]
        print(f"# calib DONE tag={tag}: per-rank files {sizes or 'MISSING'}", flush=True)
    if dump or sparse_dump:
        io = sorted(glob.glob(f"/cache/qb_reconio_{tag}_dev*.pt"))
        iosz = [f"{os.path.basename(f)}={os.path.getsize(f) // (1024 * 1024)}MB" for f in io]
        print(f"# reconio DONE tag={tag}: per-rank files {iosz or 'MISSING'}", flush=True)


def _qmap_impl(
    tp: int,
    tag: str,
    dense_layers: str,
    max_len: int,
    moe: str = "dense",
    probe_layers: str = "0,20,40",
    calib_file: str = "",
) -> None:
    """WORKSTREAM A0+A1#1: real-routed-activation quality map + dense-anchor-layer policy.
    Loads DeepSeek-V4-Flash-NVFP4 with QB_MOE=sparse, keeps NVFP4 resident on probe layers (and on
    dense-anchor layers), runs coherence prompts, and records per-layer *block* cosine + per-expert
    cos / route freq / weight / contribution norm on REAL routed activations. Reports whether the
    dense-anchor policy restores coherence and at what sparse-layer coverage."""
    import glob
    import json
    import math
    import os
    import statistics as st
    import subprocess
    import time

    import torch
    from vllm import LLM, SamplingParams

    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    # nvfp4 dense/attention (WS1: coherent on SM120) is ~3x smaller than the bf16 fallback, which
    # otherwise leaves zero KV headroom once probe-layer sparse codes are also resident -> OOM.
    os.environ["QB_DENSE"] = "nvfp4"
    # map: run MoE dense (coherent) so probe layers see HEALTHY activations; anchor-coherence tests
    # pass moe="sparse" + dense_layers to measure real generation quality under selective sparsity.
    os.environ["QB_MOE"] = moe
    # probe_layers="none" -> pure coherence run (no map probe, no resident codes on sparse layers)
    os.environ["QB_QMAP"] = "0" if probe_layers == "none" else "1"
    os.environ["QB_QMAP_LAYERS"] = "" if probe_layers == "none" else probe_layers
    os.environ["QB_INSTR"] = "1"  # captures the fire-once NaN diagnostics from the guardrail
    os.environ["QB_CALIB_FILE"] = calib_file  # A2: tag of a calib run -> Wanda 2:4 masks in pack()
    os.environ["QB_RUNTAG"] = tag
    os.environ["QB_DENSE_LAYERS"] = dense_layers
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    for f in glob.glob(f"/cache/qb_metrics_{tag}_dev*.json"):
        os.remove(f)

    ngpu = torch.cuda.device_count()
    print(
        f"# qmap tp={tp} tag={tag} moe={moe} probe_layers=[{probe_layers}] "
        f"dense_layers=[{dense_layers}] on {ngpu}x RTX-PRO-6000",
        flush=True,
    )

    def smi():
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        ).stdout
        return [int(ln.split(",")[0]) for ln in out.strip().splitlines()]

    rope = {
        "rope_type": "yarn",
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "beta_fast": 32,
        "beta_slow": 1,
    }
    t0 = time.time()
    llm = LLM(
        model=MODEL,
        tokenizer_mode="deepseek_v4",
        tensor_parallel_size=tp,
        enforce_eager=True,
        trust_remote_code=True,
        max_model_len=max_len,
        gpu_memory_utilization=0.95,
        kv_cache_dtype="fp8",
        max_num_batched_tokens=max(2048, max_len),
        hf_overrides={"rope_scaling": rope},
        disable_log_stats=False,
    )
    print(f"  loaded in {time.time() - t0:.0f}s; per-GPU used MB={smi()}", flush=True)

    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "The three primary colors are",
        "Water is made of hydrogen and",
        "The opposite of hot is",
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=40, min_tokens=8)
    outs = llm.generate(prompts, sp)
    print("# --- COHERENCE (generation sanity) ---", flush=True)
    for p, o in zip(prompts, outs, strict=False):
        print(f"  [{p!r}] -> {o.outputs[0].text!r}", flush=True)

    # --- PPL: teacher-forced perplexity over a fixed held-out passage (quantifies the tax) ---
    passage = (
        "The mitochondria is the powerhouse of the cell. Photosynthesis converts sunlight, water, "
        "and carbon dioxide into glucose and oxygen. The Earth orbits the Sun once every year, and "
        "the Moon orbits the Earth roughly every twenty-eight days. Water boils at one hundred "
        "degrees Celsius at sea level and freezes at zero degrees. The human heart pumps blood "
        "through arteries and veins, delivering oxygen to every tissue in the body. Shakespeare "
        "wrote many famous plays, including Hamlet, Macbeth, and Romeo and Juliet. The speed of "
        "light in a vacuum is approximately three hundred thousand kilometres per second."
    )
    pids = llm.get_tokenizer().encode(passage)
    pout = llm.generate(
        [{"prompt_token_ids": pids}],
        SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0),
    )
    plp = pout[0].prompt_logprobs or []
    nlls = []
    for tid, d in zip(pids[1:], plp[1:], strict=False):
        if d and tid in d and math.isfinite(d[tid].logprob):
            nlls.append(-d[tid].logprob)
    ppl = math.exp(sum(nlls) / len(nlls)) if nlls else float("nan")
    print(
        f"# PPL over {len(nlls)}-token held-out passage: {ppl:.3f} "
        f"(mean NLL {sum(nlls) / len(nlls):.4f} nats)"
        if nlls
        else "# PPL: no valid logprobs",
        flush=True,
    )

    # aggregate the real-activation map + NaN diagnostics flushed by every worker
    rows, nan_diag = [], []
    for mf in sorted(glob.glob(f"/cache/qb_metrics_{tag}_dev*.json")):
        with open(mf) as f:
            d = json.load(f)
            rows += d.get("qmap", [])
            nan_diag += d.get("nan_diag", [])
    print("# --- NaN guardrail diagnostics (first nonfinite per layer/tensor) ---", flush=True)
    if nan_diag:
        for d in sorted(nan_diag, key=lambda r: (r["layer"], r["tensor"])):
            print(
                f"  layer {d['layer']:>3} {d['tensor']:<8} nonfinite={d['nonfinite']} "
                f"finite_max_abs={d['finite_max_abs']}",
                flush=True,
            )
    else:
        print("  (none — no nonfinite tensors intercepted in the sparse path)", flush=True)
    blk = {}  # layer -> [block_cos, ...]
    exp = {}  # layer -> [expert cos, ...]
    for r in rows:
        if r["expert"] == -1:
            if math.isfinite(r["block_cos"]):
                blk.setdefault(r["layer"], []).append(r["block_cos"])
        elif math.isfinite(r["cos"]):
            exp.setdefault(r["layer"], []).append(r["cos"])

    csv_path = f"/cache/qb_qmap_{tag}.csv"
    with open(csv_path, "w") as f:
        f.write("layer,expert,cos,freq,mean_w,contrib_norm,block_cos\n")
        for r in rows:
            if r["expert"] == -1:
                f.write(f"{r['layer']},-1,,,,,{r['block_cos']}\n")
            else:
                f.write(
                    f"{r['layer']},{r['expert']},{r['cos']},{r['freq']},"
                    f"{r['mean_w']},{r['contrib_norm']},\n"
                )

    print("# --- A0 real-activation quality map ---", flush=True)
    print("# layer  block_cos  expert_cos(median)  n_experts", flush=True)
    prod = 1.0
    all_exp = []
    for li in sorted(blk):
        bc = st.median(blk[li])
        ec = st.median(exp.get(li, [0.0]))
        all_exp += exp.get(li, [])
        prod *= bc
        print(f"  {li:>4}   {bc:.4f}     {ec:.4f}            {len(exp.get(li, []))}", flush=True)
    n_probe = len(blk)
    # extrapolate the probed block cosines over all sparse layers (probe covers every _TAX_LAYERS-th)
    dense_set = {int(x) for x in dense_layers.split(",") if x.strip()}
    n_sparse = 43 - len(dense_set)
    if n_probe:
        geo = math.exp(sum(math.log(max(1e-9, st.median(blk[li]))) for li in blk) / n_probe)
        pred_end = geo**n_sparse
        print(
            f"# probed {n_probe} layers; geomean block_cos={geo:.4f}; "
            f"predicted end-of-stack coherence signal geo^{n_sparse}={pred_end:.3e}",
            flush=True,
        )
    if all_exp:
        all_exp.sort()

        def q(p):
            return all_exp[min(len(all_exp) - 1, int(p * len(all_exp)))]

        print(
            f"# per-expert real-activation cos: median={st.median(all_exp):.4f} "
            f"p10={q(0.1):.4f} p90={q(0.9):.4f} min={all_exp[0]:.4f} max={all_exp[-1]:.4f}",
            flush=True,
        )
    print(
        f"# dense-anchor layers={sorted(dense_set)} ({len(dense_set)}/43 dense, "
        f"{n_sparse}/43 sparse = {100 * n_sparse / 43:.0f}% layers sparse)",
        flush=True,
    )
    print(f"# qmap CSV -> {csv_path} ({len(rows)} rows)", flush=True)


def _downstream_impl(
    tp: int,
    tag: str,
    moe: str,
    dense_layers: str,
    calib_file: str,
    limit: int,
    max_len: int,
    sparse_proj: str = "both",
    route_slot: int = 0,
    glm: bool = False,
    force_custom_ar: bool = False,
) -> None:
    """WS-A downstream capability validation. Loglikelihood MC eval (lm-eval-compatible
    acc/acc_norm: per choice, sum teacher-forced logprob of the continuation tokens via vLLM
    prompt_logprobs; pick argmax) over ARC-C / HellaSwag / PIQA / Winogrande / MMLU-subset. Same
    bespoke DeepSeek-V4-Flash init as _qmap_impl so the quadbit sparse plugin + fp8-KV + rope
    overrides are identical. GSM8K skipped (generative, ~2 tok/s sparse decode = not cheap).
    glm=True loads GLM-5.2-NVFP4 (8-GPU EP, no DeepSeek rope/tokenizer) instead; the MC eval is
    tokenizer-agnostic (uses llm.get_tokenizer()) so it transfers unchanged."""
    import math
    import os
    import subprocess
    import time

    from vllm import LLM, SamplingParams

    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    os.environ["QB_DENSE"] = "nvfp4"
    os.environ["QB_MOE"] = moe
    os.environ["QB_QMAP"] = "0"
    os.environ["QB_INSTR"] = "1"
    os.environ["QB_CALIB_FILE"] = calib_file
    os.environ["QB_RUNTAG"] = tag
    os.environ["QB_DENSE_LAYERS"] = dense_layers
    os.environ["QB_SPARSE_PROJ"] = sparse_proj
    os.environ["QB_ROUTE_SLOT"] = str(route_slot)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    # C6: exercise the C4 one-shot custom all-reduce (same env the speed row used). The plugin's
    # install() reads QB_FORCE_CUSTOM_AR, verifies the full P2P matrix, and (only if fully
    # connected) spoofs is_fully_connected + sets VLLM_SKIP_P2P_CHECK so vLLM enables
    # cross_device_reduce_1stage on the 4 PCIe GPUs. Graph capture is off here (enforce_eager) but
    # the AR kernel and its bf16 reduction order are byte-identical eager vs graph-replayed, so
    # downstream quality under this flag equals the captured speed row's. False = NCCL ring.
    # Clear on the False path too: a warm reused Modal container that previously ran a custom-AR row
    # would otherwise leak QB_FORCE_CUSTOM_AR into a later NCCL control and silently exercise the
    # one-shot path, breaking the controlled comparison.
    if force_custom_ar:
        os.environ["QB_FORCE_CUSTOM_AR"] = "1"
    else:
        os.environ.pop("QB_FORCE_CUSTOM_AR", None)

    def smi():
        q = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        out = subprocess.run(q, capture_output=True, text=True).stdout
        return [int(ln) for ln in out.strip().splitlines()]

    t0 = time.time()
    if glm:
        # GLM-5.2 keeps its own rope/config (1M ctx); no DeepSeek yarn override or tokenizer_mode.
        # 8-GPU EP, eager (GLM's EP MoE is not graph-capturable). Same QB env set above applies.
        llm = LLM(
            model=GLM_MODEL,
            tensor_parallel_size=tp,
            enforce_eager=True,
            trust_remote_code=True,
            max_model_len=max_len,
            gpu_memory_utilization=0.92,
            kv_cache_dtype="fp8",
            max_num_batched_tokens=max(2048, max_len),
            enable_expert_parallel=True,
            disable_log_stats=True,
        )
    else:
        rope = {
            "rope_type": "yarn",
            "factor": 16,
            "original_max_position_embeddings": 65536,
            "beta_fast": 32,
            "beta_slow": 1,
        }
        llm = LLM(
            model=MODEL,
            tokenizer_mode="deepseek_v4",
            tensor_parallel_size=tp,
            enforce_eager=True,
            trust_remote_code=True,
            max_model_len=max_len,
            gpu_memory_utilization=0.95,
            kv_cache_dtype="fp8",
            max_num_batched_tokens=max(2048, max_len),
            hf_overrides={"rope_scaling": rope},
            disable_log_stats=True,
        )
    tok = llm.get_tokenizer()
    mem = smi()
    print(
        f"# downstream tag={tag} moe={moe} dense_layers=[{dense_layers}] calib={calib_file} "
        f"limit={limit} loaded {time.time() - t0:.0f}s per-GPU-MB={mem}",
        flush=True,
    )

    # generation sanity (same 5 prompts as qmap)
    gp = [
        "The capital of France is",
        "def fibonacci(n):",
        "The three primary colors are",
        "Water is made of hydrogen and",
        "The opposite of hot is",
    ]
    gouts = llm.generate(gp, SamplingParams(temperature=0.0, max_tokens=24, min_tokens=6))
    print("# --- generation sanity ---", flush=True)
    for p, o in zip(gp, gouts, strict=False):
        print(f"  [{p!r}] -> {o.outputs[0].text!r}", flush=True)

    # PPL (self-report, must match the qmap Pareto number for this policy)
    passage = (
        "The mitochondria is the powerhouse of the cell. Photosynthesis converts sunlight, water, "
        "and carbon dioxide into glucose and oxygen. The Earth orbits the Sun once every year, and "
        "the Moon orbits the Earth roughly every twenty-eight days. Water boils at one hundred "
        "degrees Celsius at sea level and freezes at zero degrees. The human heart pumps blood "
        "through arteries and veins, delivering oxygen to every tissue in the body. Shakespeare "
        "wrote many famous plays, including Hamlet, Macbeth, and Romeo and Juliet. The speed of "
        "light in a vacuum is approximately three hundred thousand kilometres per second."
    )
    pids = tok.encode(passage)
    pout = llm.generate(
        [{"prompt_token_ids": pids}],
        SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0),
    )
    plp = pout[0].prompt_logprobs or []
    nlls = [
        -plp[i][t].logprob
        for i, t in enumerate(pids)
        if i > 0 and plp[i] and t in plp[i] and math.isfinite(plp[i][t].logprob)
    ]
    ppl = math.exp(sum(nlls) / len(nlls)) if nlls else float("nan")
    print(f"# PPL {ppl:.3f}", flush=True)

    # ---- loglikelihood MC scoring ----
    # request tuple: (item_key, choice_idx, prompt_token_ids, span_start, cont_char_len)
    def build(ctx: str, conts: list[str]):
        cids = tok.encode(ctx)
        out = []
        for c in conts:
            tail = tok.encode(c, add_special_tokens=False)
            out.append((cids + tail, len(cids), len(c)))
        return out

    def run_task(name: str, items: list):
        # items: list of (conts:list[str], gold:int, requests:list[(ids,start,clen)])
        reqs, meta = [], []
        for qi, (_, _, rq) in enumerate(items):
            for ci, (ids, start, clen) in enumerate(rq):
                reqs.append({"prompt_token_ids": ids})
                meta.append((qi, ci, start, clen))
        outs = llm.generate(reqs, SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0))
        scores: dict = {}
        for (qi, ci, start, clen), o in zip(meta, outs, strict=False):
            lp = o.prompt_logprobs or []
            # prompt_logprobs=0 -> each position dict holds only the ACTUAL token's logprob;
            # sum over the continuation span [start:] (position 0 is None/BOS, always < start).
            s = sum(next(iter(d.values())).logprob for i, d in enumerate(lp) if i >= start and d)
            scores.setdefault(qi, {})[ci] = (s, clen)
        acc = accn = 0
        for qi, (_, gold, _) in enumerate(items):
            sc = scores.get(qi, {})
            if not sc:
                continue
            best = max(sc, key=lambda c: sc[c][0])
            bestn = max(sc, key=lambda c: sc[c][0] / max(1, sc[c][1]))
            acc += int(best == gold)
            accn += int(bestn == gold)
        n = len(items)
        print(f"# TASK {name} n={n} acc={acc / n:.4f} acc_norm={accn / n:.4f}", flush=True)
        return {"task": name, "n": n, "acc": acc / n, "acc_norm": accn / n}

    from datasets import load_dataset  # noqa: PLC0415

    results = []

    def try_load(cands):
        for repo, kw in cands:
            try:
                return load_dataset(repo, **kw), repo
            except Exception as e:  # noqa: BLE001
                print(f"  load {repo} failed: {str(e)[:120]}", flush=True)
        return None, None

    # ARC-Challenge (acc_norm primary). template: "Question: {q}\nAnswer:" cont=" {choice}"
    ds, src = try_load([("allenai/ai2_arc", {"name": "ARC-Challenge", "split": "test"})])
    if ds is not None:
        items = []
        for r in list(ds)[:limit]:
            ch = r["choices"]["text"]
            labels = r["choices"]["label"]
            gold = labels.index(r["answerKey"]) if r["answerKey"] in labels else 0
            conts = [" " + t for t in ch]
            items.append((conts, gold, build(f"Question: {r['question']}\nAnswer:", conts)))
        results.append(run_task(f"arc_c[{src}]", items))

    # HellaSwag (acc_norm primary). lm-eval preprocess + ctx = activity + ctx_a/ctx_b
    ds, src = try_load([("Rowan/hellaswag", {"split": "validation"})])
    if ds is not None:
        import re

        def pp(t):
            t = t.strip().replace(" [title]", ". ")
            t = re.sub("\\[.*?\\]", "", t)
            return t.replace("  ", " ")

        items = []
        for r in list(ds)[:limit]:
            ctx = pp(r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize())
            conts = [" " + pp(e) for e in r["endings"]]
            gold = int(r["label"]) if r["label"] != "" else 0
            items.append((conts, gold, build(ctx, conts)))
        results.append(run_task(f"hellaswag[{src}]", items))

    # PIQA is intentionally excluded: ybisk/piqa needs gated trust_remote_code loading that is not
    # available on the serve image, so it never loaded and never entered any recorded average. Every
    # documented downstream number (DeepSeek deepseek_final.csv and GLM glm_results.md) is a 4-task
    # average; dropping the block makes that deterministic and reproducible from the manifest command.

    # Winogrande (acc). partial scoring: ctx=prefix+option, cont=suffix after blank
    ds, src = try_load(
        [
            (
                "allenai/winogrande",
                {"name": "winogrande_xl", "split": "validation", "trust_remote_code": True},
            )
        ]
    )
    if ds is not None:
        items = []
        for r in list(ds)[:limit]:
            s = r["sentence"]
            idx = s.index("_")
            suffix = s[idx + 1 :]
            reqs, conts = [], []
            for opt in (r["option1"], r["option2"]):
                cids = tok.encode(s[:idx] + opt)
                tail = tok.encode(suffix, add_special_tokens=False)
                reqs.append((cids + tail, len(cids), len(suffix)))
                conts.append(suffix)
            gold = int(r["answer"]) - 1
            items.append((conts, gold, reqs))
        results.append(run_task(f"winogrande[{src}]", items))

    # MMLU subset (acc), 0-shot
    subjects = [
        "abstract_algebra",
        "college_computer_science",
        "high_school_world_history",
        "professional_medicine",
        "moral_scenarios",
    ]
    mmlu_scores = []
    for subj in subjects:
        ds, src = try_load([("cais/mmlu", {"name": subj, "split": "test"})])
        if ds is None:
            continue
        readable = subj.replace("_", " ")
        items = []
        for r in list(ds)[: max(50, limit // 4)]:
            q = r["question"]
            opts = r["choices"]
            head = (
                f"The following are multiple choice questions (with answers) about "
                f"{readable}.\n\n{q}\n"
            )
            for i, o in enumerate(opts):
                head += f"{chr(65 + i)}. {o}\n"
            head += "Answer:"
            conts = [f" {chr(65 + i)}" for i in range(len(opts))]
            items.append((conts, int(r["answer"]), build(head, conts)))
        rr = run_task(f"mmlu:{subj}", items)
        mmlu_scores.append(rr["acc"])
    if mmlu_scores:
        mmlu_avg = sum(mmlu_scores) / len(mmlu_scores)
        print(f"# TASK mmlu_subset n_subj={len(mmlu_scores)} acc={mmlu_avg:.4f}", flush=True)
        results.append(
            {"task": "mmlu_subset", "n": len(mmlu_scores), "acc": mmlu_avg, "acc_norm": mmlu_avg}
        )

    # primary metric per task (acc_norm for arc/hellaswag/piqa, acc for winogrande/mmlu)
    def primary(r):
        return r["acc"] if r["task"].startswith(("winogrande", "mmlu")) else r["acc_norm"]

    prim = [primary(r) for r in results]
    avg = sum(prim) / len(prim) if prim else float("nan")
    print("# ================ DOWNSTREAM SUMMARY ================", flush=True)
    print(f"# tag={tag} PPL={ppl:.3f} per-GPU-MB={mem}", flush=True)
    for r in results:
        print(
            f"#   {r['task']:<28} acc={r['acc']:.4f} acc_norm={r['acc_norm']:.4f} "
            f"primary={primary(r):.4f} (n={r['n']})",
            flush=True,
        )
    print(f"# AVG normalized primary = {avg:.4f}", flush=True)
    with open(f"/cache/qb_downstream_{tag}.csv", "w") as f:
        f.write("task,n,acc,acc_norm,primary\n")
        for r in results:
            f.write(f"{r['task']},{r['n']},{r['acc']:.4f},{r['acc_norm']:.4f},{primary(r):.4f}\n")
        f.write(f"AVG,,,,{avg:.4f}\n")
    print(f"# downstream CSV -> /cache/qb_downstream_{tag}.csv", flush=True)


@app.function(
    gpu="RTX-PRO-6000:2",
    timeout=120 * MIN,
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def downstream(
    tag: str = "ds",
    moe: str = "dense",
    dense_layers: str = "",
    calib_file: str = "",
    limit: int = 400,
    max_len: int = 2048,
    sparse_proj: str = "both",
    route_slot: int = 0,
) -> None:
    _downstream_impl(2, tag, moe, dense_layers, calib_file, limit, max_len, sparse_proj, route_slot)


@app.function(
    gpu="RTX-PRO-6000:4",
    timeout=120 * MIN,
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def downstream4(
    tag: str = "ds",
    moe: str = "dense",
    dense_layers: str = "",
    calib_file: str = "",
    limit: int = 400,
    max_len: int = 2048,
    sparse_proj: str = "both",
    route_slot: int = 0,
    force_custom_ar: bool = False,
) -> None:
    # 4-GPU variant: route-slot keeps raw NVFP4 + packed codes resident, so experts must shard over
    # more GPUs to fit (tp=2 OOMs at the first-22 anchor). force_custom_ar routes the attention TP
    # all-reduce through C4's one-shot custom AR (C6 validation of the speed-row collective).
    _downstream_impl(
        4, tag, moe, dense_layers, calib_file, limit, max_len, sparse_proj, route_slot,
        force_custom_ar=force_custom_ar,
    )


@app.function(
    gpu="RTX-PRO-6000:8",
    timeout=360 * MIN,
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def glm_downstream(
    tag: str = "glm",
    moe: str = "dense",
    dense_layers: str = "",
    calib_file: str = "",
    limit: int = 200,
    max_len: int = 2048,
    sparse_proj: str = "both",
    route_slot: int = 0,
) -> None:
    # GLM-5.2 downstream smoke (P1): same MC eval as DeepSeek, 8-GPU EP, GLM load (glm=True).
    _downstream_impl(
        8, tag, moe, dense_layers, calib_file, limit, max_len, sparse_proj, route_slot, glm=True
    )


@app.function(
    gpu="RTX-PRO-6000:2",
    timeout=360 * MIN,
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def recon(
    tag: str = "rc",
    dense_layers: str = "",
    calib_file: str = "cal4",
    recon_io: str = "",
    recon_layers: str = "",
    steps: int = 200,
    scale_only: bool = False,
    limit: int = 400,
    max_len: int = 2048,
) -> None:
    """A3 layerwise repair: QB_RECON trains the listed sparse MoE layers vs the dumped teacher I/O
    during load, packs the repaired weights, then runs downstream eval on the repaired model in the
    same job. Repaired weights persist to /cache/qb_reconw_{tag}_dev*.pt for later serving."""
    import os

    import moe_recon

    moe_recon._selfcheck()  # fail fast on a trainer logic bug before the 8-min model load
    os.environ["QB_RECON"] = "1"
    os.environ["QB_RECON_IO"] = recon_io
    os.environ["QB_RECON_LAYERS"] = recon_layers
    os.environ["QB_RECON_STEPS"] = str(steps)
    if scale_only:
        os.environ["QB_RECON_SCALE_ONLY"] = "1"
    _downstream_impl(2, tag, "sparse", dense_layers, calib_file, limit, max_len)


@app.function(
    gpu="RTX-PRO-6000:2",
    timeout=90 * MIN,
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def qmap(
    tag: str = "qm",
    dense_layers: str = "",
    max_len: int = 2048,
    moe: str = "dense",
    probe_layers: str = "0,20,40",
    calib_file: str = "",
) -> None:
    _qmap_impl(2, tag, dense_layers, max_len, moe, probe_layers, calib_file)


@app.function(
    gpu="RTX-PRO-6000:2",
    timeout=90 * MIN,
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def calib(
    tag: str = "cal1",
    max_len: int = 2048,
    dump: bool = False,
    sparse_dump: bool = False,
    dense_layers: str = "",
    calib_file: str = "",
) -> None:
    _calib_impl(
        2,
        tag,
        max_len,
        dump=dump,
        sparse_dump=sparse_dump,
        dense_layers=dense_layers,
        calib_file=calib_file,
    )


@app.local_entrypoint()
def main(
    mode: str = "baseline",
    tp: int = 2,
    eager: bool = False,
    max_len: int = 4096,
    dense: str = "bf16",
    kv: str = "fp8",
    moe: str = "off",
    tag: str = "qm",
    dense_layers: str = "",
    probe_layers: str = "0,20,40",
    calib_file: str = "",
    limit: int = 400,
    recon_io: str = "",
    recon_layers: str = "",
    steps: int = 200,
    scale_only: bool = False,
    sparse_proj: str = "both",
    route_slot: int = 0,
) -> None:
    if mode == "test_so":
        test_so.remote()
    elif mode == "glm_inspect":
        glm_inspect.remote()
    elif mode == "glm_baseline":
        # main defaults eager=False, but GLM's EP MoE path is NOT graph-capturable (the plugin's
        # local-expert loop host-syncs via torch.unique(...).tolist(), illegal under stream capture),
        # so graph capture aborts the run. Force eager -- it is currently the only working GLM path;
        # revisit if/when the plugin's expert loop is made capture-safe.
        glm_baseline.remote(
            tp=tp,
            eager=True,
            max_len=max_len,
            dense=(dense if dense in ("nvfp4", "bf16") else "nvfp4"),
            moe=(moe if moe != "off" else "dense"),
            dense_layers=dense_layers,
            sparse_proj=sparse_proj,
            route_slot=route_slot,
        )
    elif mode == "versions":
        versions.remote()
    elif mode == "dumpsrc":
        dumpsrc.remote()
    elif mode == "probe":
        probe.remote()
    elif mode == "inspect":
        inspect_moe.remote(tp=tp, max_len=max_len)
    elif mode == "quadbit":
        quadbit.remote(tp=tp, eager=eager, max_len=max_len)
    elif mode == "unblock":
        # .remote() (not .spawn()) so the FULL worker log streams to this client; launch under
        # `modal run --detach` so the job still survives client disconnect. spawn's detached logs
        # get truncated to a short tail once the app stops, hiding the worker root-cause traceback.
        unblock.remote(tp=tp, eager=eager, max_len=max_len, dense=dense, kv=kv, moe=moe)
    elif mode == "sweep2":
        sweep2.remote()
    elif mode == "sweep4":
        sweep4.remote()
    elif mode == "qmap":
        qmap.remote(
            tag=tag,
            dense_layers=dense_layers,
            max_len=max_len,
            moe=(moe if moe != "off" else "dense"),
            probe_layers=probe_layers,
            calib_file=calib_file,
        )
    elif mode == "calib":
        calib.remote(tag=tag, max_len=max_len)
    elif mode == "dumpio":
        calib.remote(tag=tag, max_len=max_len, dump=True)
    elif mode == "dumpio2":
        # A3 reio2: sparse-trajectory (serve-consistent) I/O dump under the A2-49 policy.
        calib.remote(
            tag=tag,
            max_len=max_len,
            sparse_dump=True,
            dense_layers=dense_layers,
            calib_file=(calib_file or "cal4"),
        )
    elif mode == "downstream":
        downstream.remote(
            tag=tag,
            moe=(moe if moe != "off" else "dense"),
            dense_layers=dense_layers,
            calib_file=calib_file,
            limit=limit,
            max_len=max_len,
            sparse_proj=sparse_proj,
            route_slot=route_slot,
        )
    elif mode == "glm_downstream":
        glm_downstream.remote(
            tag=tag,
            moe=(moe if moe != "off" else "dense"),
            dense_layers=dense_layers,
            calib_file=calib_file,
            limit=limit,
            max_len=max_len,
            sparse_proj=sparse_proj,
            route_slot=route_slot,
        )
    elif mode == "downstream4":
        downstream4.remote(
            tag=tag,
            moe=(moe if moe != "off" else "dense"),
            dense_layers=dense_layers,
            calib_file=calib_file,
            limit=limit,
            max_len=max_len,
            sparse_proj=sparse_proj,
            route_slot=route_slot,
        )
    elif mode == "recon":
        recon.remote(
            tag=tag,
            dense_layers=dense_layers,
            calib_file=(calib_file or "cal4"),
            recon_io=recon_io,
            recon_layers=recon_layers,
            steps=steps,
            scale_only=scale_only,
            limit=limit,
            max_len=max_len,
        )
    else:
        baseline.remote(tp=tp, eager=eager, max_len=max_len)
