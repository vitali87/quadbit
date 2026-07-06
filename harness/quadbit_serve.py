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


def _patch_mlp_sparse(model, lib, torch, thresh: int):
    """Replace each LlamaMLP.forward: large-M (prefill) -> quadbit sparse two-level kernel,
    small-M (decode) -> original dense bf16 path. Packs sparse weights from the loaded bf16
    gate_up_proj / down_proj once, here. Returns (#patched, sparse_param_frac)."""
    import torch.nn.functional as F
    dev = torch.device("cuda")
    FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
    BND = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], device=dev)
    _cc = torch.arange(128, device=dev); _e, _m = (_cc >> 3) & 0xf, _cc & 7
    UE4M3 = torch.where(_e == 0, _m.float() * 0.001953125, (1.0 + _m.float() / 8.0) * torch.exp2((_e - 7).float()))

    def q_fp4(v):
        return torch.bucketize(v.abs(), BND) | ((v < 0).long() << 3)

    def enc_ue4m3_t(s):
        mant_f, e = torch.frexp(s.clamp_min(1e-30)); mm = 2.0 * mant_f
        biased = (e - 1) + 7; mant = torch.round((mm - 1.0) * 8.0).long(); carry = mant == 8
        mant = torch.where(carry, torch.zeros_like(mant), mant); biased = torch.where(carry, biased + 1, biased)
        code = (biased.long() << 3) | mant
        code = torch.where(biased < 1, torch.ones_like(code), code)
        code = torch.where(biased > 15, torch.full_like(code, 0x7f), code)
        code = torch.where(s >= 480.0, torch.full_like(code, 0x7f), code)
        return torch.where(s > 0, code, torch.zeros_like(code))

    class QBSparse:  # two-level pair-2:4 sparse FP4 (magnitude prune + per-16 ue4m3 local + per-row gA)
        def __init__(self, W):
            out_f, in_f = W.shape; ks = in_f // 128
            self.out_f, self.in_f, self.ks = out_f, in_f, ks
            Wg = W.float().to(dev).view(out_f, ks, 16, 4, 2)
            i01, _ = Wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
            keptW = torch.gather(Wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
            gA = (keptW.abs().amax(dim=(1, 2, 3, 4), keepdim=True) / 2688.0).clamp_min(1e-30).reshape(out_f, 1, 1)
            blk = keptW.reshape(out_f, ks, 4, 8, 2)
            scode = enc_ue4m3_t((blk.abs().amax(dim=(3, 4)) / 6.0) / gA)
            sdeq = UE4M3[scode] * gA
            kc = q_fp4(blk / sdeq[..., None, None])
            self.Ac = (kc[..., 0] | (kc[..., 1] << 4)).reshape(out_f, ks * 32).to(torch.uint8).contiguous()
            nib = (i01[..., 0] | (i01[..., 1] << 2)).view(out_f, ks, 2, 8)
            sh = (torch.arange(8, device=dev) * 4).view(1, 1, 1, 8)
            self.meta = (nib << sh).sum(-1).to(torch.int32).permute(1, 0, 2).contiguous()
            self.scaleA = scode.to(torch.uint8).permute(1, 0, 2).contiguous()
            self.gA = gA.reshape(out_f).float().contiguous()

        def forward(self, x):
            lead = x.shape[:-1]; x2 = x.reshape(-1, self.in_f).to(torch.bfloat16)
            t = x2.shape[0]; pad = (-t) % 128
            if pad:
                x2 = torch.cat([x2, x2.new_zeros(pad, self.in_f)], 0)
            x2 = x2.contiguous(); tp = t + pad
            Bb = torch.empty((tp, self.in_f // 2), dtype=torch.uint8, device=dev)
            sB = torch.empty((self.ks, tp, 4), dtype=torch.uint8, device=dev)
            gB = torch.empty((tp,), dtype=torch.float32, device=dev)
            lib.quantize_act_nvfp4_2lvl(x2.data_ptr(), Bb.data_ptr(), sB.data_ptr(), gB.data_ptr(), tp, self.in_f)
            C = torch.empty((self.out_f, tp), dtype=torch.bfloat16, device=dev)
            lib.sparse_fp4_mm_2lvl(self.Ac.data_ptr(), Bb.data_ptr(), self.scaleA.data_ptr(),
                                   sB.data_ptr(), self.meta.data_ptr(), C.data_ptr(),
                                   self.out_f, tp, self.in_f, self.gA.data_ptr(), gB.data_ptr())
            return C.t()[:t].reshape(*lead, self.out_f).to(x.dtype)

    n = 0; sparse_params = 0; total_params = 0
    for layer in model.model.layers:
        for nm2 in ("q_proj", "k_proj", "v_proj", "o_proj"):
            total_params += getattr(layer.self_attn, nm2).weight.numel()
        mlp = layer.mlp
        total_params += mlp.gate_up_proj.weight.numel() + mlp.down_proj.weight.numel()
        gu = QBSparse(mlp.gate_up_proj.weight.data); dn = QBSparse(mlp.down_proj.weight.data)
        sparse_params += mlp.gate_up_proj.weight.numel() + mlp.down_proj.weight.numel()
        mlp._qb_gu, mlp._qb_dn, mlp._qb_orig = gu, dn, mlp.forward

        def make_fwd(mlp_ref):
            def fwd(x):
                m = x.numel() // x.shape[-1]
                if m >= thresh:  # prefill / large batch -> sparse kernel
                    return mlp_ref._qb_dn.forward(mlp_ref.act_fn(mlp_ref._qb_gu.forward(x)))
                return mlp_ref._qb_orig(x)  # decode small-M -> dense bf16 fallback
            return fwd
        mlp.forward = make_fwd(mlp); n += 1
    return n, sparse_params / total_params


@app.function(gpu="RTX-PRO-6000", timeout=3600, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def serve(sparse: bool = True, thresh: int = 256) -> None:
    # Serving-table row on the SAME protocol as vllm_nvfp4.serve: prefill (B prompts x S=2048, 1 out)
    # and decode (B short prompts, GEN=128) tok/s at B=1/8/32/64. sparse=True patches MLP to quadbit
    # sparse (prefill) + dense bf16 (decode); sparse=False is the bf16 baseline in the SAME harness.
    import ctypes
    import subprocess
    import time

    import torch
    from vllm import LLM, SamplingParams

    def gpu_mib():
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.strip().splitlines()
        return int(out[0])

    S, GEN = 2048, 128
    llm = LLM(model=BASE, enforce_eager=True, max_model_len=S + GEN + 16,
              kv_cache_dtype="auto", gpu_memory_utilization=0.9, dtype="bfloat16")
    tag = "quadbit-sparse-prefill" if sparse else "bf16-baseline"
    frac = 0.0
    if sparse:
        lib = ctypes.CDLL(SO_PATH)
        lib.sparse_fp4_mm_2lvl.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
        lib.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        npatched, frac = _patch_mlp_sparse(model, lib, torch, thresh)
        print(f"patched {npatched} MLP layers to sparse (thresh M>={thresh}); sparse FLOP frac {frac:.3f}", flush=True)
    print(f"[{tag}] {torch.cuda.get_device_name(0)}", flush=True)
    mem_load = gpu_mib()

    def distinct_ids(i):
        return [((j * 131 + i * 7919) % 128000) + 1 for j in range(S)]

    def distinct_short(i):
        return f"Explain concept {i}: how does mechanism {i * 7 + 3} operate in practice, step by step?"

    def tps(prompts, sp):
        torch.cuda.synchronize(); t = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        return outs, time.perf_counter() - t

    llm.generate([distinct_short(0)], SamplingParams(temperature=0, max_tokens=8), use_tqdm=False)  # warmup
    mem_warm = gpu_mib(); mem_peak = mem_warm
    print(f"{'B':>4} | {'prefill tok/s':>14} | {'decode tok/s':>13}", flush=True)
    for B in (1, 8, 32, 64):
        _, pf_dt = tps([{"prompt_token_ids": distinct_ids(i)} for i in range(B)],
                       SamplingParams(temperature=0, max_tokens=1))
        prefill_tps = B * S / pf_dt
        outs, dc_dt = tps([distinct_short(i) for i in range(B)],
                          SamplingParams(temperature=0, max_tokens=GEN, ignore_eos=True))
        gen = sum(len(o.outputs[0].token_ids) for o in outs)
        mem_peak = max(mem_peak, gpu_mib())
        print(f"{B:>4} | {prefill_tps:>14.0f} | {gen / dc_dt:>13.0f}", flush=True)
    print(f"RESULT [{tag}] sparse_frac={frac:.3f} thresh={thresh} device_MiB load={mem_load} "
          f"warm={mem_warm} peak={mem_peak} (nvidia-smi, incl KV pool @util0.9)", flush=True)


@app.local_entrypoint()
def main(mode: str = "smoke", sparse: bool = True, thresh: int = 256) -> None:
    if mode == "serve":
        call = serve.spawn(sparse, thresh)
    else:
        call = {"store_so": store_so, "smoke": smoke}.get(mode, smoke).spawn()
    print(f"SPAWN_ID {call.object_id}", flush=True)
