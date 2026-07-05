"""Serving-table row: does SGLang run NATIVE NVFP4 (W4A4) on SM120 (RTX PRO 6000), and at what
throughput? SGLang docs list modelopt_fp4 as native FP4 on Blackwell and expose --fp4-gemm-backend
{auto, flashinfer_cutlass, cutlass, flashinfer_cudnn}; auto is documented to pick FlashInfer CUTLASS
on SM120 with an SGLang-CUTLASS fallback.

smoke: load the official NVFP4 checkpoint via the offline Engine API with a chosen fp4 backend,
generate, and report the ACTUAL selected backend + quant method from the logs (not the requested
flag). bench: prefill/decode throughput sweep matching the vLLM row. If SGLang fails to build/run on
SM120, the failure is captured as the artifact (do not block the paper on it).

Run:  uv run modal run harness/sglang_fp4.py --backend auto
      uv run modal run harness/sglang_fp4.py --mode bench --backend flashinfer_cutlass
"""

import modal

MODEL = "nvidia/Llama-3.1-8B-Instruct-NVFP4"
MINUTES = 60

# SGLang for Blackwell/SM120: needs CUDA >= 12.9 (SM 12.x device-capability probe + flashinfer JIT
# fail on 12.8 with "SM 12.x requires CUDA >= 12.9"). 12.9.0 base, same as the working vLLM image.
sgl_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .pip_install("sglang[all]", "flashinfer-python", "huggingface_hub", "pyarrow")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
app = modal.App("quadbit-sglang-fp4", image=sgl_image)


@app.function(gpu="RTX-PRO-6000", timeout=30 * MINUTES,
              volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[modal.Secret.from_name("huggingface")])
def run(mode: str = "smoke", backend: str = "auto") -> None:
    import time

    import torch

    print(f"{torch.cuda.get_device_name(0)}  sm_{torch.cuda.get_device_capability(0)[0]}"
          f"{torch.cuda.get_device_capability(0)[1]}  requested fp4_gemm_backend={backend}", flush=True)

    import sglang as sgl

    # fp4_gemm_backend is the Engine kwarg form of --fp4-gemm-backend; auto omits it (let SGLang pick).
    kw = {} if backend == "auto" else {"fp4_gemm_backend": backend}
    try:
        engine = sgl.Engine(model_path=MODEL, quantization="modelopt_fp4",
                            mem_fraction_static=0.85, disable_cuda_graph=(mode == "smoke"), **kw)
    except Exception as e:
        print(f"SGLANG_FAIL init backend={backend}: {type(e).__name__}: {e}", flush=True)
        raise

    out = engine.generate("The capital of France is",
                          {"temperature": 0.0, "max_new_tokens": 16})
    print(f"GEN: {out['text']!r}", flush=True)

    if mode == "smoke":
        print("SGLANG_SMOKE_OK (selected backend + quant method in the init logs above)", flush=True)
        engine.shutdown(); return

    # bench: prefill (S=2048, 1 token) + decode (short prompt, GEN=128) at B=1/8/32/64, matching vLLM
    import subprocess

    def gpu_mib():
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.strip().splitlines()
        return int(out[0])

    S, GEN = 2048, 128
    long_ids = list(range(1, S + 1))
    short = "Explain how a transformer works."
    engine.generate(short, {"temperature": 0.0, "max_new_tokens": 8})  # warmup
    mem_warm = gpu_mib(); mem_peak = mem_warm

    print(f"{'B':>4} | {'prefill tok/s':>14} | {'decode tok/s':>13}", flush=True)
    for B in (1, 8, 32, 64):
        t = time.perf_counter()
        engine.generate(input_ids=[long_ids] * B, sampling_params={"temperature": 0.0, "max_new_tokens": 1})
        pf = B * S / (time.perf_counter() - t)
        t = time.perf_counter()
        engine.generate([short] * B, sampling_params={"temperature": 0.0, "max_new_tokens": GEN, "ignore_eos": True})
        dc = B * GEN / (time.perf_counter() - t)
        mem_peak = max(mem_peak, gpu_mib())
        print(f"{B:>4} | {pf:>14.0f} | {dc:>13.0f}", flush=True)
    print(f"RESULT sglang_fp4 backend={backend}: device MiB post-warmup {mem_warm}, peak {mem_peak} "
          f"(nvidia-smi; total incl KV pool @ mem_fraction 0.85). selected backend in init logs.", flush=True)
    engine.shutdown()


@app.local_entrypoint()
def main(mode: str = "smoke", backend: str = "auto") -> None:
    call = run.spawn(mode, backend)
    print(f"SPAWN_ID {call.object_id}", flush=True)
    call.get()
