"""C5 DP-attention worker (installed into the plugin package so multiprocessing.spawn can import it in the
child without re-importing the Modal-decorated harness module, which is not picklable). One process per DP
rank: DP attention + EP MoE, tensor_parallel_size=1, so attention runs replicated per rank (NO per-layer TP
all-reduce, the 94.5% decode floor) while experts stay EP-sharded. Rank 0 reports decode tok/s."""

from __future__ import annotations

import math
import os
import time


def dp_worker(dp_rank, dp, port, model, cap, max_seqs, max_len, gpu_mem, baseline, eager):
    import torch
    from vllm import LLM, SamplingParams

    os.environ["VLLM_DP_RANK"] = str(dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp)
    os.environ["VLLM_DP_MASTER_IP"] = "127.0.0.1"
    os.environ["VLLM_DP_MASTER_PORT"] = str(port)
    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    os.environ["QB_DENSE"] = "nvfp4"
    os.environ["QB_MOE"] = "off" if baseline == "dense_nvfp4" else "sparse"
    os.environ["QB_GRAPH"] = "0" if eager else "1"
    os.environ["QB_GRAPH_CAP"] = str(cap)
    os.environ["QB_DENSE_BACKEND"] = "native_nvfp4"

    rope = {"rope_type": "yarn", "factor": 16, "original_max_position_embeddings": 65536,
            "beta_fast": 32, "beta_slow": 1}
    kw = dict(model=model, tensor_parallel_size=1, data_parallel_size=dp, enforce_eager=eager,
              trust_remote_code=True, max_model_len=max_len, gpu_memory_utilization=gpu_mem,
              kv_cache_dtype="fp8", max_num_batched_tokens=max(2048, max_len), max_num_seqs=max_seqs,
              enable_expert_parallel=True, hf_overrides={"rope_scaling": rope})
    try:
        llm = LLM(tokenizer_mode="deepseek_v4", **kw)
    except Exception:  # noqa: BLE001
        llm = LLM(**kw)

    tids = llm.get_tokenizer().encode("The history of the Roman empire spans many centuries and")

    def _wall(n):
        torch.cuda.synchronize()
        t = time.time()
        llm.generate([{"prompt_token_ids": tids}], SamplingParams(temperature=0.0, max_tokens=n))
        torch.cuda.synchronize()
        return time.time() - t

    _wall(4)                         # warm (all ranks step together: EP all-to-all needs every rank live)
    w1, w64 = _wall(1), _wall(64)
    dtps = 63.0 / (w64 - w1) if w64 > w1 else float("nan")
    if dp_rank == 0:
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
        print(f"# C5-DP decode tok/s: {dtps:.3f} (wall1={w1:.3f}s wall64={w64:.3f}s) dp={dp} tp=1 "
              f"ppl={ppl:.4f} graph={'eager' if eager else 'captured'}", flush=True)
        print(f"# C5-DP PASS dp={dp} decode_tps={dtps:.3f} ppl={ppl:.4f}", flush=True)
