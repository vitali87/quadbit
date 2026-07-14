"""C7 DP-attention worker. One OS process per DP rank: tensor_parallel_size=1 so attention runs
replicated per rank (NO per-layer TP all-reduce, the 94.5% decode floor) while experts stay
EP-sharded (the MoE EP all-to-all becomes the sole cross-GPU collective). Rank 0 reports
single-request batch=1 decode tok/s, PPL, coherence, and the per-token all-reduce / all-to-all
COUNT from a torch trace.

Launch protocol (fixed vs C5): vLLM 0.24 LLM(data_parallel_size>1) hard-raises
(entrypoints/llm.py:293) unless distributed_executor_backend=="external_launcher". The supported
offline path is env-driven SPMD: set VLLM_DP_* + CUDA_VISIBLE_DEVICES per rank and call a plain
LLM(tensor_parallel_size=1) with NO data_parallel_* kwargs (config/parallel.py:852 reads DP
size/rank from env; is_moe_model clears the dense-model guard). C5 failed because it passed
data_parallel_size=/data_parallel_rank= as LLM() kwargs.

Lockstep safety: the EP all-to-all is collective over the DP group, so every generate() must be
matched across all ranks or rank 0 deadlocks on idle peers. So ALL ranks run IDENTICAL generate
calls; only rank 0 profiles and prints. Each rank is its own batch=1 request, so rank 0's per-token
latency is a genuine single-request decode latency (NOT aggregate replica QPS)."""

from __future__ import annotations

import glob
import gzip
import json
import math
import os
import time


def dp_worker(dp_rank, dp, port, model, cap, max_seqs, max_len, gpu_mem, baseline, eager):
    # Pin one physical GPU per rank BEFORE importing torch; each process sees a single cuda:0 and
    # the NCCL DP/EP group spans the 4 processes via the master ip/port. RANK_LOCAL=0 = that device.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(dp_rank)
    os.environ["VLLM_DP_RANK"] = str(dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = "0"
    os.environ["VLLM_DP_SIZE"] = str(dp)
    os.environ["VLLM_DP_MASTER_IP"] = "127.0.0.1"
    os.environ["VLLM_DP_MASTER_PORT"] = str(port)
    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    os.environ["QB_DENSE"] = "nvfp4"
    os.environ["QB_MOE"] = "off" if baseline == "dense_nvfp4" else "sparse"
    os.environ["QB_GRAPH"] = "0" if eager else "1"
    os.environ["QB_GRAPH_CAP"] = str(cap)
    os.environ["QB_DENSE_BACKEND"] = "native_nvfp4"

    profile = os.environ.get("QB_DP_PROFILE", "1") == "1"
    prof_dir = f"/tmp/qb_dpprof_dev{dp_rank}"
    if profile:
        os.makedirs(prof_dir, exist_ok=True)
        os.environ["VLLM_TORCH_PROFILER_DIR"] = prof_dir  # set before LLM() init to arm profiler

    import torch
    from vllm import LLM, SamplingParams

    rope = {"rope_type": "yarn", "factor": 16, "original_max_position_embeddings": 65536,
            "beta_fast": 32, "beta_slow": 1}
    # Env-driven SPMD DP: NO data_parallel_* kwargs (they trigger the single-process raise). vLLM
    # reads VLLM_DP_SIZE/RANK from env. tp=1 => attention replicated (no attention all-reduce);
    # enable_expert_parallel=True => experts EP-sharded (all-to-all across the DP group).
    kw = dict(model=model, tensor_parallel_size=1, enforce_eager=eager, trust_remote_code=True,
              max_model_len=max_len, gpu_memory_utilization=gpu_mem, kv_cache_dtype="fp8",
              max_num_batched_tokens=max(2048, max_len), max_num_seqs=max_seqs,
              enable_expert_parallel=True, hf_overrides={"rope_scaling": rope})
    try:
        llm = LLM(tokenizer_mode="deepseek_v4", **kw)
    except Exception as ex:  # noqa: BLE001
        print(f"[dp{dp_rank}] deepseek_v4 kw rejected: {type(ex).__name__}: {ex}", flush=True)
        llm = LLM(**kw)
    print(f"[dp{dp_rank}] LLM constructed dp={dp} tp=1 EP=on device=cuda:{dp_rank}", flush=True)

    tids = llm.get_tokenizer().encode("The history of the Roman empire spans many centuries and")

    def _wall(n):
        torch.cuda.synchronize()
        t = time.time()
        llm.generate([{"prompt_token_ids": tids}], SamplingParams(temperature=0.0, max_tokens=n))
        torch.cuda.synchronize()
        return time.time() - t

    # ALL ranks run identical work (lockstep for the EP all-to-all). Only rank 0 profiles/prints.
    _wall(4)                          # warm
    w1, w64 = _wall(1), _wall(64)     # measurement: batch=1, TTFT-subtracted decode
    dtps = 63.0 / (w64 - w1) if w64 > w1 else float("nan")

    # Profiled pass (all ranks step together; only rank 0's trace is parsed for the AR/a2a count).
    if profile:
        torch.cuda.synchronize()
        llm.start_profile()
        _wall(16)
        llm.stop_profile()
        torch.cuda.synchronize()

    if dp_rank != 0:
        return

    if profile:
        _report_collectives(prof_dir, model)

    # coherence smoke (rank 0)
    for p in ["The capital of France is", "def fibonacci(n):", "The three primary colors are"]:
        pid = llm.get_tokenizer().encode(p)
        sp = SamplingParams(temperature=0.0, max_tokens=24)
        out = llm.generate([{"prompt_token_ids": pid}], sp)
        print(f"# GEN {p!r} -> {out[0].outputs[0].text!r}", flush=True)

    # teacher-forced PPL on the mito80 passage (same protocol as the other rows)
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
    print(f"# C7-DP decode tok/s: {dtps:.3f} (wall1={w1:.3f}s wall64={w64:.3f}s) dp={dp} tp=1 "
          f"ppl={ppl:.4f} graph={'eager' if eager else 'captured'}", flush=True)
    print(f"# C7-DP PASS dp={dp} decode_tps={dtps:.3f} ppl={ppl:.4f}", flush=True)


def _report_collectives(prof_dir, model):
    """Parse rank 0's torch trace and count cross-GPU collective kernels per decode token, reusing
    floor_profile counting (cross_device_reduce / allreduce_ = TP all-reduce; alltoall = EP;
    sparse_mla_decode = one per layer per forward, the per-token anchor). AR/token ~0 proves DP
    removed the attention all-reduce; the a2a count shows what replaced it."""
    traces = []
    for _ in range(20):  # stop_profile flush can lag a beat
        traces = sorted(glob.glob(f"{prof_dir}/*.json*"))
        if traces:
            break
        time.sleep(0.5)
    if not traces:
        print("# C7-DP NO TRACE WRITTEN — profiler dir empty", flush=True)
        return
    t = traces[0]
    opener = gzip.open if t.endswith(".gz") else open
    with opener(t, "rt") as fh:
        data = json.loads(fh.read())
    evs = data.get("traceEvents", data) if isinstance(data, dict) else data
    cnt_by_name: dict = {}
    for e in evs:
        if not isinstance(e, dict) or e.get("ph") != "X":
            continue
        if str(e.get("cat", "")) not in ("kernel", "gpu_op", "Kernel"):
            continue
        nm = str(e.get("name", ""))
        cnt_by_name[nm] = cnt_by_name.get(nm, 0) + 1

    def cnt(*subs):
        return sum(c for k, c in cnt_by_name.items() if any(s in k.lower() for s in subs))
    n_customar = cnt("cross_device_reduce")
    n_ring_ar = cnt("allreduce_") + cnt("allreduce ")
    n_a2a = cnt("alltoall", "all_to_all", "all2all")
    n_dsa = cnt("sparse_mla_decode", "sparse_mla")
    n_layers = 0
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model, trust_remote_code=True, local_files_only=True)
        n_layers = int(getattr(cfg, "num_hidden_layers", 0))
    except Exception:  # noqa: BLE001
        pass
    fwd = (n_dsa / n_layers) if (n_dsa and n_layers) else 0.0
    ar = n_customar or n_ring_ar
    ar_tok = (ar / fwd) if fwd else 0.0
    a2a_tok = (n_a2a / fwd) if fwd else 0.0
    print(f"# C7-DP COLLECTIVES trace={os.path.basename(t)} n_layers={n_layers} "
          f"decode_fwd~{fwd:.1f}", flush=True)
    print(f"# C7-DP all-reduce: custom={n_customar} ring={n_ring_ar} per-token={ar_tok:.1f} | "
          f"all-to-all={n_a2a} per-token={a2a_tok:.1f} | DSA-decode={n_dsa}", flush=True)
