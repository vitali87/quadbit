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
    if os.path.exists(so):
        print(f"# {so} present", flush=True)
        return
    print(f"# building {so} (CUDA-13 gencode compute_120a,sm_120a)", flush=True)
    c = subprocess.run(["nvcc", "-gencode=arch=compute_120a,code=sm_120a", "-O3", "-shared",
                        "-Xcompiler", "-fPIC", "-o", so, "/root/cuda/sparse_fp4_lib.cu", "-lcuda"],
                       capture_output=True, text=True)
    if c.returncode != 0:
        print(c.stderr, flush=True); raise SystemExit(1)
    vol.commit()


@app.function(gpu="RTX-PRO-6000", timeout=30 * MIN, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(moe: str = "sparse", max_len: int = 2048) -> None:
    import time

    import torch
    from vllm import LLM, SamplingParams

    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    os.environ["QB_MOE"] = moe
    os.environ["QB_SPARSE_PROJ"] = "both"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    _ensure_so()
    print(f"# gpt-oss serve: {MODEL} QB_MOE={moe} on {torch.cuda.device_count()}x RTX-PRO-6000", flush=True)

    kw = dict(model=MODEL, tensor_parallel_size=1, enforce_eager=True, trust_remote_code=True,
              max_model_len=max_len, gpu_memory_utilization=0.90, max_num_batched_tokens=max_len)
    t0 = time.time()
    llm = LLM(**kw)
    print(f"  load+init ok in {time.time() - t0:.0f}s", flush=True)

    prompts = ["The capital of France is", "def fibonacci(n):",
               "The three primary colors are", "Water is made of hydrogen and"]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=32))
    for o in outs:
        print(f"  [gen] {o.prompt!r} -> {o.outputs[0].text!r}", flush=True)

    pids = llm.get_tokenizer().encode(PASSAGE)
    pout = llm.generate([{"prompt_token_ids": pids}],
                        SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0))
    plp = pout[0].prompt_logprobs or []
    nlls = [-d[tid].logprob for tid, d in zip(pids[1:], plp[1:], strict=False)
            if d and tid in d and math.isfinite(d[tid].logprob)]
    ppl = math.exp(sum(nlls) / len(nlls)) if nlls else float("nan")
    print(f"# PPL over {len(nlls)}-token passage: {ppl:.3f}", flush=True)

    try:
        import qb_sm120_plugin as qb
        print(f"# STATS moe_calls={qb.STATS.get('moe_calls')} "
              f"sparse_expert_calls={qb.STATS.get('sparse_expert_calls')}", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"# STATS unavailable: {type(ex).__name__}: {ex}", flush=True)
    print(f"# gpt-oss serve done (QB_MOE={moe}, PPL={ppl:.3f})", flush=True)
