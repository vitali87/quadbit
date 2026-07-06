"""P0: quadbit sparse FP4 INSIDE a real serving engine (vLLM), prefill-integrated path.

Goal: turn the GEMM-level sparse-beats-FlashInfer-dense Pareto win into a measured serving row.
Reuse vLLM for tokenizer / paged attention / KV cache / batching; swap ONLY the MLP linears to
quadbit sparse on large-M (prefill); small-M (decode) falls back to dense bf16. Attention stays
vLLM bf16. Honestly labeled "quadbit sparse prefill path + dense decode fallback".

TOOLCHAIN (measured 2026-07-06, see memory quadbit-fp4-leaderboard): our sm_120a block-scale mma
assembles ONLY under nvcc <=12.8; vLLM 0.21 needs the 12.9.0 base (12.8.1 base dies in FlashInfer's
sm75 probe). So STAGED build: `store_so` compiles the .so in a 12.8.1 image onto the shared volume;
the 12.9 vLLM function ctypes-loads it (driver API + libcudart.so.12 forward-compat within CUDA 12.x).
vLLM 0.21 is V1 (EngineCore subprocess) -> set VLLM_ENABLE_V1_MULTIPROCESSING=0 to run in-process so
a post-load monkeypatch reaches the model; enforce_eager so the ctypes kernel is not inside a CUDA
graph / torch.compile region.

Run:  uv run modal run harness/quadbit_serve.py --mode store_so   # once, populates the volume
      uv run modal run harness/quadbit_serve.py --mode smoke      # de-risk .so-in-vllm + patch surface
"""

import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
BASE = "meta-llama/Meta-Llama-3-8B"
SO_PATH = "/cache/sparse_fp4_sm120.so"

# 12.8.1: the ONLY toolchain that assembles our block-scale sparse mma. Compiles the .so -> volume.
build_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64"})
    .add_local_dir((ROOT / "cuda").as_posix(), "/root/cuda")
)
# 12.9.0: the base the working vLLM serving baseline uses (vllm_nvfp4.py). Runs the engine + our .so.
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "huggingface_hub", "pyarrow")
    .env({"HF_HOME": "/cache", "HF_XET_HIGH_PERFORMANCE": "1",
          "VLLM_ENABLE_V1_MULTIPROCESSING": "0"})  # engine core in-process -> post-load patch reaches model
)
app = modal.App("quadbit-serve", image=vllm_image)
vol = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)


@app.function(image=build_image, timeout=1200, volumes={"/cache": vol})
def store_so() -> None:
    c = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
                        "--cudart", "shared",  # resolve libcudart.so.12 at runtime (12.9 provides it)
                        "-o", SO_PATH, "/root/cuda/sparse_fp4_lib.cu", "-lcuda"],
                       capture_output=True, text=True)
    print(f"nvcc rc={c.returncode}", flush=True)
    if c.returncode != 0:
        print(c.stderr[-4000:], flush=True); raise RuntimeError("nvcc failed")
    vol.commit()
    print(f"STORED {SO_PATH}", flush=True)


@app.function(gpu="RTX-PRO-6000", timeout=3600, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def smoke() -> None:
    import ctypes
    import os

    import torch
    import vllm

    print(f"vllm {vllm.__version__}  torch {torch.__version__}  {torch.cuda.get_device_name(0)} "
          f"sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}  "
          f"V1_MP={os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING')}", flush=True)

    # 1) cross-version ABI: does the 12.8-compiled .so load + run inside the 12.9 vLLM process?
    lib = ctypes.CDLL(SO_PATH)
    lib.sparse_fp4_mm_2lvl.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    dev = torch.device("cuda")
    out_f, in_f, M = 256, 256, 128
    Ac = torch.zeros((out_f, (in_f // 128) * 32), dtype=torch.uint8, device=dev)
    meta = torch.zeros((in_f // 128, out_f, 2), dtype=torch.int32, device=dev)
    scaleA = torch.ones((in_f // 128, out_f, 4), dtype=torch.uint8, device=dev)
    Bb = torch.zeros((M, in_f // 2), dtype=torch.uint8, device=dev)
    sB = torch.ones((in_f // 128, M, 4), dtype=torch.uint8, device=dev)
    gA = torch.ones((out_f,), dtype=torch.float32, device=dev)
    gB = torch.ones((M,), dtype=torch.float32, device=dev)
    C = torch.empty((out_f, M), dtype=torch.bfloat16, device=dev)
    lib.sparse_fp4_mm_2lvl(Ac.data_ptr(), Bb.data_ptr(), scaleA.data_ptr(), sB.data_ptr(),
                           meta.data_ptr(), C.data_ptr(), out_f, M, in_f, gA.data_ptr(), gB.data_ptr())
    torch.cuda.synchronize()
    print(f"KERNEL_IN_VLLM_PROCESS_OK C{tuple(C.shape)}", flush=True)

    # 2) in-process model reachability + MLP structure (with V1_MP=0)
    from vllm import LLM, SamplingParams
    llm = LLM(model=BASE, enforce_eager=True, max_model_len=2048, gpu_memory_utilization=0.85,
              dtype="bfloat16")
    print("LLM loaded (bf16, eager)", flush=True)

    model = None
    for path in ("llm_engine.model_executor.driver_worker.model_runner.model",
                 "llm_engine.model_executor.driver_worker.worker.model_runner.model",
                 "llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner.model"):
        try:
            obj = llm
            for a in path.split("."):
                obj = getattr(obj, a)
            model = obj; print(f"MODEL_REACHABLE via {path}: {type(model).__name__}", flush=True); break
        except Exception as e:
            print(f"  path {path}: {type(e).__name__}", flush=True)

    if model is not None:
        mlp = model.model.layers[0].mlp
        attrs = [a for a in ("gate_up_proj", "gate_proj", "up_proj", "down_proj") if hasattr(mlp, a)]
        print(f"MLP {type(mlp).__name__}; linears {attrs}", flush=True)
        for a in attrs:
            lin = getattr(mlp, a); w = getattr(lin, "weight", None)
            print(f"  {a}: {type(lin).__name__} weight={tuple(w.shape) if w is not None else None} "
                  f"{w.dtype if w is not None else None}", flush=True)

    out = llm.generate(["The capital of France is"], SamplingParams(temperature=0, max_tokens=8))
    print(f"GEN {out[0].outputs[0].text!r}", flush=True)
    print("SMOKE_OK", flush=True)


@app.local_entrypoint()
def main(mode: str = "smoke") -> None:
    fn = {"store_so": store_so, "smoke": smoke}.get(mode, smoke)
    call = fn.spawn()
    print(f"SPAWN_ID {call.object_id}", flush=True)
