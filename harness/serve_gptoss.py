"""M1-serving gate: run gpt-oss-20b in vLLM 0.24 with quadbit 2:4-sparse-FP4 experts (QB_MOE=sparse)
and confirm (a) the GptOssMxfp4MoEMethod patch fires, (b) the segmented sparse kernel actually runs
(SPARSE_EXPERT_CALLS>0), (c) output is finite + coherent, (d) a teacher-forced PPL. QB_MOE=off gives
the stock-MXFP4 baseline. No recovery yet, so sparse PPL will be degraded-but-coherent (the 2:4 tax the
recovery recipe closes next). Offline math already verified in gptoss_prep.py (M0/M2/M1/M1.5)."""
import math
import os
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
MODEL = "openai/gpt-oss-20b"
MIN = 60

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.12")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64", "HF_HOME": "/cache", "HF_XET_HIGH_PERFORMANCE": "1"})
    .pip_install("vllm", "huggingface_hub", "datasets")
    .run_commands("pip install --force-reinstall --no-deps flashinfer-python==0.6.14 flashinfer-cubin==0.6.13")
    .env({"FLASHINFER_DISABLE_VERSION_CHECK": "1"})
    .add_local_dir(str(ROOT / "cuda"), "/root/cuda", copy=True)
    .add_local_dir(str(ROOT / "harness" / "qb_vllm_plugin"), "/opt/qb_plugin", copy=True)
    .run_commands("pip install --force-reinstall --no-deps /opt/qb_plugin", force_build=True)
)
app = modal.App("quadbit-serve-gptoss", image=image)
vol = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)

PASSAGE = (
    "The mitochondria is the powerhouse of the cell. Photosynthesis converts sunlight, water, "
    "and carbon dioxide into glucose and oxygen. The Earth orbits the Sun once every year, and "
    "the Moon orbits the Earth roughly every twenty-eight days. Water boils at one hundred "
    "degrees Celsius at sea level and freezes at zero degrees. The human heart pumps blood "
    "through arteries and veins, delivering oxygen to every tissue in the body. Shakespeare "
    "wrote many famous plays, including Hamlet, Macbeth, and Romeo and Juliet. The speed of "
    "light in a vacuum is approximately three hundred thousand kilometres per second."
)


def _ensure_so() -> None:
    # plugin loads /cache/sparse_fp4.so; build it if the volume lacks it. CUDA-13 base -> gencode
    # compute_120a,sm_120a (a plain -arch=sm_120a drops the 'a' and ptxas rejects the block-scale mma).
    import subprocess
    so = "/cache/sparse_fp4.so"
    # Always rebuild: the .so lives on the volume and outlives image rebuilds, and add_local_dir mtimes
    # are unreliable as a freshness gate. ~90s compile is cheap insurance that kernel edits take effect.
    print(f"# building {so} (CUDA-13 gencode compute_120a,sm_120a)", flush=True)
    c = subprocess.run(["nvcc", "-gencode=arch=compute_120a,code=sm_120a", "-O3", "-shared",
                        "-Xcompiler", "-fPIC", "-o", so, "/root/cuda/sparse_fp4_lib.cu", "-lcuda"],
                       capture_output=True, text=True)
    if c.returncode != 0:
        print(c.stderr, flush=True); raise SystemExit(1)
    vol.commit()


@app.function(gpu="RTX-PRO-6000", timeout=30 * MIN, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(moe: str = "sparse", proj: str = "both", sparse_from: int = 0, max_len: int = 2048,
        batch_prefill: int = 16, max_batched: int = 16384, bench_only: bool = False,
        graph: bool = False, graph_cap: int = 256, ablate: str = "", torchprof: bool = False,
        prof: bool = False, fused: bool = True, ab: bool = False) -> None:
    import time

    import torch
    from vllm import LLM, SamplingParams

    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    os.environ["QB_MOE"] = moe
    os.environ["QB_SPARSE_PROJ"] = proj          # both|down|gateup (tax lives in gate_up -> down recovers)
    os.environ["QB_SPARSE_FROM"] = str(sparse_from)  # prefix-optimal: sparsify layers >= this, anchor earlier
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    if graph:  # capture-legal MoE path + let vLLM CUDA-graph the forward
        os.environ["QB_GRAPH"] = "1"
        os.environ["QB_GRAPH_CAP"] = str(graph_cap)
    if ablate:
        os.environ["QB_GPTOSS_ABLATE"] = ablate
    if torchprof:
        os.environ["QB_TORCHPROF"] = "1"
    if prof:
        os.environ["QB_GPTOSS_PROF"] = "1"
    os.environ["QB_GPTOSS_FUSED"] = "1" if fused else "0"   # fully-fused MoE (GEMM1 + down-scatter)
    if ab:  # same-invocation A/B: plugin reads /dev/shm/qb_fused per-apply (must be set BEFORE worker fork)
        os.environ["QB_AB"] = "1"
    _ensure_so()
    print(f"# gpt-oss serve: {MODEL} QB_MOE={moe} proj={proj} sparse_from={sparse_from} "
          f"graph={graph} cap={graph_cap if graph else 0} on {torch.cuda.device_count()}x RTX-PRO-6000", flush=True)

    kw = dict(model=MODEL, tensor_parallel_size=1, enforce_eager=not graph, trust_remote_code=True,
              max_model_len=max_len, gpu_memory_utilization=0.90, max_num_batched_tokens=max_batched)
    t0 = time.time()
    llm = LLM(**kw)
    print(f"  load+init ok in {time.time() - t0:.0f}s", flush=True)

    tok = llm.get_tokenizer()
    base = tok.encode(PASSAGE)
    ppl = float("nan")
    if not bench_only:
        prompts = ["The capital of France is", "def fibonacci(n):",
                   "The three primary colors are", "Water is made of hydrogen and"]
        outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=32))
        for o in outs:
            print(f"  [gen] {o.prompt!r} -> {o.outputs[0].text!r}", flush=True)

        pout = llm.generate([{"prompt_token_ids": base}],
                            SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0))
        plp = pout[0].prompt_logprobs or []
        nlls = [-d[tid].logprob for tid, d in zip(base[1:], plp[1:], strict=False)
                if d and tid in d and math.isfinite(d[tid].logprob)]
        ppl = math.exp(sum(nlls) / len(nlls)) if nlls else float("nan")
        print(f"# PPL over {len(nlls)}-token passage: {ppl:.3f}", flush=True)

        # --- throughput (prefill + decode, B=1) ---
        for plen in (512, 1536):
            gen = 64
            plen = min(plen, max_len - gen - 8)
            if plen <= 0:
                continue
            ptoks = (base * ((plen // len(base)) + 1))[:plen]
            try:
                tp0 = time.time()
                llm.generate([{"prompt_token_ids": ptoks}], SamplingParams(temperature=0.0, max_tokens=1))
                prefill_s = time.time() - tp0
                td0 = time.time()
                dout = llm.generate([{"prompt_token_ids": ptoks}], SamplingParams(temperature=0.0, max_tokens=gen))
                dec_s = time.time() - td0
                gtok = len(dout[0].outputs[0].token_ids)
                dec_only = max(1e-6, dec_s - prefill_s)
                steps = max(1, gtok - 1)
                print(f"# serve B=1 prompt={plen}: prefill {plen / prefill_s:.0f} tok/s, "
                      f"decode {steps / dec_only:.1f} tok/s", flush=True)
            except Exception as e:  # noqa: BLE001 - timing must not abort the eval
                print(f"# timing skipped prompt={plen} ({type(e).__name__}: {e})", flush=True)

    # --- large-batch prefill throughput: the COMPUTE-BOUND regime where 2:4 sparse wins (many
    # tokens/expert -> large-M MoE GEMMs). B=1 above is memory-bound + overhead-bound (our worst case). ---
    S = max_len - 16  # leave room for the +1 output token (prompt+out must be <= max_model_len)
    big = (base * ((S // len(base)) + 1))[:S]
    reqs = [{"prompt_token_ids": big} for _ in range(batch_prefill)]
    ntok = batch_prefill * S
    one = SamplingParams(temperature=0.0, max_tokens=1)
    if os.environ.get("QB_TORCHPROF") == "1":
        from torch.profiler import ProfilerActivity, profile
        llm.generate(reqs, one)  # warm
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            llm.generate(reqs, one)
        print("# ===== TORCH PROFILER (large-batch prefill, top CUDA ops) =====", flush=True)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30), flush=True)
    iters = 20
    try:
        for _ in range(3):  # warm (kernel JIT, autotune, allocator)
            llm.generate(reqs, one)
        times = []
        for _ in range(iters):
            t0 = time.time()
            llm.generate(reqs, one)
            times.append(time.time() - t0)
        times.sort()
        med = times[len(times) // 2]
        print(f"# LARGE-BATCH prefill B={batch_prefill} S={S} ({ntok} tok, ~{ntok // 32} tok/expert), "
              f"median of {iters}: {ntok / med:.0f} tok/s (med {med * 1000:.1f}ms, "
              f"min {min(times) * 1000:.1f} / max {max(times) * 1000:.1f})", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"# large-batch prefill skipped ({type(e).__name__}: {e})", flush=True)

    # --- SAME-INVOCATION A/B: interleave fused vs unfused large-batch prefill (alternate every iter so GPU
    # clock drift affects both equally -> the delta is real, unlike ±6% cross-run noise). The plugin worker
    # reads /dev/shm/qb_fused per apply; we toggle it from the driver (shared tmpfs, same container). ---
    if ab:
        def _setflag(v):
            with open("/dev/shm/qb_fused", "w") as f:
                f.write(v)
        try:
            for v in ("1", "0"):  # warm both paths (JIT/autotune/allocator)
                _setflag(v)
                for _ in range(3):
                    llm.generate(reqs, one)
            tf, tu = [], []
            for _ in range(iters):
                _setflag("1"); t0 = time.time(); llm.generate(reqs, one); tf.append(time.time() - t0)
                _setflag("0"); t0 = time.time(); llm.generate(reqs, one); tu.append(time.time() - t0)
            tf.sort(); tu.sort()
            mf, mu = tf[len(tf) // 2], tu[len(tu) // 2]
            print(f"# ===== SAME-INVOCATION A/B (interleaved, same clock, {iters} each) =====", flush=True)
            print(f"#   FUSED (monolith)  {ntok / mf:.0f} tok/s ({mf * 1000:.1f}ms med, "
                  f"{min(tf) * 1000:.1f}/{max(tf) * 1000:.1f} min/max)", flush=True)
            print(f"#   UNFUSED (fast_gu) {ntok / mu:.0f} tok/s ({mu * 1000:.1f}ms med, "
                  f"{min(tu) * 1000:.1f}/{max(tu) * 1000:.1f} min/max)", flush=True)
            print(f"#   fused/unfused speedup = {mu / mf:.3f}x  (RELIABLE: same GPU clock, interleaved)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"# A/B skipped ({type(e).__name__}: {e})", flush=True)

    # --- decode throughput (B=1): the launch-overhead-bound regime graph capture accelerates ---
    dprompt = base[:32]
    G = 128
    try:
        llm.generate([{"prompt_token_ids": dprompt}], SamplingParams(temperature=0.0, max_tokens=8))  # warm
        rates = []
        for _ in range(5):
            td0 = time.time()
            dout = llm.generate([{"prompt_token_ids": dprompt}], SamplingParams(temperature=0.0, max_tokens=G))
            n = max(1, len(dout[0].outputs[0].token_ids))
            rates.append(n / (time.time() - td0))
        rates.sort()
        print(f"# DECODE B=1 gen={G}: {rates[len(rates) // 2]:.1f} tok/s (median of 5)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"# decode bench skipped ({type(e).__name__}: {e})", flush=True)

    try:
        import qb_sm120_plugin as qb
        print(f"# STATS moe_calls={qb.STATS.get('moe_calls')} "
              f"sparse_expert_calls={qb.STATS.get('sparse_expert_calls')}", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"# STATS unavailable: {type(ex).__name__}: {ex}", flush=True)
    print(f"# gpt-oss serve done (QB_MOE={moe} proj={proj} sparse_from={sparse_from}, PPL={ppl:.3f})", flush=True)
