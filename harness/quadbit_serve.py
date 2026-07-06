"""P0: quadbit sparse FP4 INSIDE a real serving engine (vLLM), prefill-integrated path.

Goal: turn the GEMM-level sparse-beats-FlashInfer-dense Pareto win into a measured serving row.
Reuse vLLM for tokenizer / paged attention / KV cache / batching; swap ONLY the MLP linears to
quadbit sparse on large-M (prefill); small-M (decode) falls back to dense bf16. Attention stays
vLLM bf16. Honestly labeled "quadbit sparse prefill path + dense decode fallback".

Integration approach: load Llama-3-8B as bf16 (no vLLM quant plumbing), monkeypatch each decoder
layer's MLP so gate_up/down dispatch on token count M -> sparse kernel (M>=threshold, padded to
%128) or dense bf16 (M<threshold). Prefill runs eager (enforce_eager) so the ctypes kernel is not
inside a CUDA graph / torch.compile region.

`smoke` first DE-RISKS the unknowns before the full patch: vLLM version, whether the model is
reachable/monkeypatchable in-process, and whether our nvcc-12.8 .so runs inside the vLLM process
on SM120. Run that, read the log, then wire the patch.

Run:  uv run modal run harness/quadbit_serve.py --mode smoke
"""

import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
BASE = "meta-llama/Meta-Llama-3-8B"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64", "HF_HOME": "/cache"})
    .uv_pip_install("vllm==0.21.0", "huggingface_hub", "pyarrow")
    .add_local_dir((ROOT / "cuda").as_posix(), "/root/cuda")
)
app = modal.App("quadbit-serve", image=image)
vol = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)

# 12.9 base = the CUDA the working vLLM serving baseline uses (vllm_nvfp4.py). Test whether nvcc 12.9
# still assembles our sm_120a block-scale mma; if it does, vLLM + kernel share ONE image.
image_129 = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64"})
    .add_local_dir((ROOT / "cuda").as_posix(), "/root/cuda")
)


@app.function(image=image_129, timeout=1200)
def compile_check() -> None:
    import subprocess as sp
    v = sp.run(["nvcc", "--version"], capture_output=True, text=True).stdout
    print(v, flush=True)
    c = sp.run(["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
                "-o", "/root/sparse_fp4.so", "/root/cuda/sparse_fp4_lib.cu", "-lcuda"],
               capture_output=True, text=True)
    print(f"NVCC_129_RC {c.returncode}", flush=True)
    if c.returncode != 0:
        print("STDERR:\n" + c.stderr[-4000:], flush=True)
    else:
        print("NVCC_129_ASSEMBLES_OK", flush=True)


def _compile_so() -> str:
    so = "/root/sparse_fp4.so"
    c = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
                        "-o", so, "/root/cuda/sparse_fp4_lib.cu", "-lcuda"], capture_output=True, text=True)
    print(f"nvcc rc={c.returncode}", flush=True)
    if c.returncode != 0:
        print(c.stderr, flush=True)
        raise RuntimeError("nvcc failed inside vLLM image")
    return so


@app.function(gpu="RTX-PRO-6000", timeout=3600, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def smoke() -> None:
    import ctypes
    import os

    import torch
    import vllm

    print(f"vllm {vllm.__version__}  torch {torch.__version__}  "
          f"{torch.cuda.get_device_name(0)} sm_{torch.cuda.get_device_capability(0)[0]}"
          f"{torch.cuda.get_device_capability(0)[1]}  VLLM_USE_V1={os.environ.get('VLLM_USE_V1')}",
          flush=True)

    # 1) does our nvcc-12.8 sparse kernel compile + run inside the vLLM (torch cu12x) process?
    so = _compile_so()
    lib = ctypes.CDLL(so)
    lib.sparse_fp4_mm_2lvl.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    lib.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    dev = torch.device("cuda")
    # trivial shape check: out=256,in=256,M=128 (all tile-legal); just confirm it launches w/o error
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
    print(f"KERNEL_IN_PROCESS_OK C{tuple(C.shape)} sum={C.float().sum().item():.1f}", flush=True)

    # 2) load 8B bf16 eager; probe whether the model is reachable + monkeypatchable in-process
    from vllm import LLM, SamplingParams
    llm = LLM(model=BASE, enforce_eager=True, max_model_len=2048, gpu_memory_utilization=0.85,
              dtype="bfloat16")
    print("LLM loaded (bf16, eager)", flush=True)

    model = None
    for path in ("llm_engine.model_executor.driver_worker.model_runner.model",
                 "llm_engine.model_executor.driver_worker.worker.model_runner.model"):
        try:
            obj = llm
            for a in path.split("."):
                obj = getattr(obj, a)
            model = obj; print(f"MODEL_REACHABLE via {path}: {type(model).__name__}", flush=True); break
        except Exception as e:
            print(f"  path {path} failed: {type(e).__name__} {e}", flush=True)

    if model is not None:
        try:
            layer0 = model.model.layers[0]
            mlp = layer0.mlp
            print(f"MLP type {type(mlp).__name__}; attrs: "
                  f"{[a for a in ('gate_up_proj','gate_proj','up_proj','down_proj') if hasattr(mlp, a)]}",
                  flush=True)
            for a in ("gate_up_proj", "down_proj"):
                if hasattr(mlp, a):
                    lin = getattr(mlp, a)
                    w = getattr(lin, "weight", None)
                    print(f"  {a}: {type(lin).__name__} weight={tuple(w.shape) if w is not None else None} "
                          f"dtype={w.dtype if w is not None else None} "
                          f"quant_method={type(getattr(lin,'quant_method',None)).__name__}", flush=True)
        except Exception as e:
            print(f"  MLP introspection failed: {type(e).__name__} {e}", flush=True)

    out = llm.generate(["The capital of France is"], SamplingParams(temperature=0, max_tokens=8))
    print(f"GEN {out[0].outputs[0].text!r}", flush=True)
    print("SMOKE_OK", flush=True)


@app.local_entrypoint()
def main(mode: str = "smoke") -> None:
    fn = {"smoke": smoke, "compile_check": compile_check}.get(mode, smoke)
    call = fn.spawn()
    print(f"SPAWN_ID {call.object_id}", flush=True)
