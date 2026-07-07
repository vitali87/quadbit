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
    # --default-stream per-thread: the kernels' <<<>>> (stream 0) then bind to the CALLING THREAD's current
    # stream, i.e. vLLM's stream, not the legacy default stream. Removes the cross-stream hazard that
    # otherwise forced torch.cuda.synchronize + a re-materializing copy in every MLP forward (which erased
    # the batch speedup). With per-thread streams the kernel is naturally ordered on vLLM's stream.
    c = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
                        "--default-stream", "per-thread",
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
    lib.sparse_fp4_mm_2lvl_t.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
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
            # guard all-zero scale-blocks: sdeq==0 -> blk/sdeq = 0/0 = NaN -> q_fp4(NaN) buckets to code 7
            # (value 6.0, MAX magnitude) instead of 0. Dead blocks (common in PRUNED/recovered weights,
            # never in dense) would otherwise become maximal garbage. blk is 0 there, so code 0 is correct.
            kc = q_fp4(blk / sdeq[..., None, None].clamp_min(1e-30))
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
            # .so compiled with --default-stream per-thread -> these <<<>>> kernels run on vLLM's current
            # stream (same as the torch ops above/below), so they are naturally ordered; no explicit sync.
            lib.quantize_act_nvfp4_2lvl(x2.data_ptr(), Bb.data_ptr(), sB.data_ptr(), gB.data_ptr(), tp, self.in_f)
            # ZERO-COPY: [tp, out_f] token-major output (outT=1) -> C[:t] is contiguous for vLLM, no copy.
            C = torch.empty((tp, self.out_f), dtype=torch.bfloat16, device=dev)
            lib.sparse_fp4_mm_2lvl_t(self.Ac.data_ptr(), Bb.data_ptr(), self.scaleA.data_ptr(),
                                     sB.data_ptr(), self.meta.data_ptr(), C.data_ptr(),
                                     self.out_f, tp, self.in_f, self.gA.data_ptr(), gB.data_ptr())
            return C[:t].reshape(*lead, self.out_f)

    n = 0; sparse_params = 0; total_params = 0
    for layer in model.model.layers:
        attn = layer.self_attn  # vLLM fuses QKV into qkv_proj; count whatever attn linears exist
        for nm2 in ("qkv_proj", "q_proj", "k_proj", "v_proj", "o_proj"):
            lin = getattr(attn, nm2, None)
            if lin is not None and hasattr(lin, "weight"):
                total_params += lin.weight.numel()
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
        lib.sparse_fp4_mm_2lvl_t.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
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


def _qbsparse_factory(torch, lib, dev):
    """QBSparse class shared by serve/serve_hybrid: two-level pair-2:4 sparse FP4 (magnitude prune +
    per-16 ue4m3 local + per-row fp32 global). Prunes the given weight internally."""
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

    class QBSparse:
        def __init__(self, W, keep_wdq=False):
            out_f, in_f = W.shape; ks = in_f // 128
            self.out_f, self.in_f, self.ks = out_f, in_f, ks
            Wg = W.float().to(dev).view(out_f, ks, 16, 4, 2)
            i01, _ = Wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
            keptW = torch.gather(Wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
            gA = (keptW.abs().amax(dim=(1, 2, 3, 4), keepdim=True) / 2688.0).clamp_min(1e-30).reshape(out_f, 1, 1)
            blk = keptW.reshape(out_f, ks, 4, 8, 2)
            scode = enc_ue4m3_t((blk.abs().amax(dim=(3, 4)) / 6.0) / gA)
            sdeq = UE4M3[scode] * gA
            # guard all-zero scale-blocks: sdeq==0 -> blk/sdeq = 0/0 = NaN -> q_fp4(NaN) buckets to code 7
            # (value 6.0, MAX magnitude) instead of 0. Dead blocks (common in PRUNED/recovered weights,
            # never in dense) would otherwise become maximal garbage. blk is 0 there, so code 0 is correct.
            kc = q_fp4(blk / sdeq[..., None, None].clamp_min(1e-30))
            self.Ac = (kc[..., 0] | (kc[..., 1] << 4)).reshape(out_f, ks * 32).to(torch.uint8).contiguous()
            nib = (i01[..., 0] | (i01[..., 1] << 2)).view(out_f, ks, 2, 8)
            sh = (torch.arange(8, device=dev) * 4).view(1, 1, 1, 8)
            self.meta = (nib << sh).sum(-1).to(torch.int32).permute(1, 0, 2).contiguous()
            self.scaleA = scode.to(torch.uint8).permute(1, 0, 2).contiguous()
            self.gA = gA.reshape(out_f).float().contiguous()
            if keep_wdq:  # dense fake-quant weight the kernel REPRESENTS (for kernel-correctness test)
                kept_dq = (FP4[kc] * sdeq[..., None, None]).reshape(out_f, ks, 16, 2, 2)
                Wdq_g = torch.zeros(out_f, ks, 16, 4, 2, device=dev)
                Wdq_g.scatter_(3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2), kept_dq)
                self.Wdq = Wdq_g.reshape(out_f, in_f)

        def forward(self, x):
            lead = x.shape[:-1]; x2 = x.reshape(-1, self.in_f).to(torch.bfloat16)
            t = x2.shape[0]; pad = (-t) % 128
            if pad:
                x2 = torch.cat([x2, x2.new_zeros(pad, self.in_f)], 0)
            x2 = x2.contiguous(); tp = t + pad
            Bb = torch.empty((tp, self.in_f // 2), dtype=torch.uint8, device=dev)
            sB = torch.empty((self.ks, tp, 4), dtype=torch.uint8, device=dev)
            gB = torch.empty((tp,), dtype=torch.float32, device=dev)
            # .so compiled with --default-stream per-thread -> these <<<>>> kernels run on vLLM's current
            # stream (same as the torch ops above/below), so they are naturally ordered; no explicit sync.
            lib.quantize_act_nvfp4_2lvl(x2.data_ptr(), Bb.data_ptr(), sB.data_ptr(), gB.data_ptr(), tp, self.in_f)
            # ZERO-COPY: [tp, out_f] token-major output (outT=1) -> C[:t] is contiguous for vLLM, no copy.
            C = torch.empty((tp, self.out_f), dtype=torch.bfloat16, device=dev)
            lib.sparse_fp4_mm_2lvl_t(self.Ac.data_ptr(), Bb.data_ptr(), self.scaleA.data_ptr(),
                                     sB.data_ptr(), self.meta.data_ptr(), C.data_ptr(),
                                     self.out_f, tp, self.in_f, self.gA.data_ptr(), gB.data_ptr())
            return C[:t].reshape(*lead, self.out_f)

    return QBSparse


# Route B proof-of-life. Option A (Dynamo graph break) is IMPOSSIBLE under vLLM V1: it compiles the
# model with aot_compile_fullgraph, which forbids graph breaks (`torch.compiler.disable` -> Unsupported).
# The sanctioned way to put an opaque ctypes kernel into a fullgraph is a torch.library custom op: Dynamo
# emits it as an op NODE (no inline, no break) and cudagraph bakes the kernel launches the op makes AT
# CAPTURE TIME. So the sparse weights MUST be registered BEFORE LLM() (vLLM captures once during init;
# a post-init patch is invisible to the frozen graph -> the 7.97-dense trap). Each LlamaMLP instance is
# tagged with its layer index at CONSTRUCTION (patched __init__, in-order 0..NL-1, before capture) and
# passes it to the op as an explicit arg -- a per-instance constant Dynamo bakes into the graph, so layer
# resolution never depends on a runtime call-order counter (which Dynamo dummy/out-of-order passes could
# desync). g["call"] is kept only as the SPARSE_CALLS proof-of-life counter.
_QB_OP = {"reg": None, "lib": None, "call": 0, "ctor": 0, "nl": 0, "splits": 8, "registered": False, "init_patched": False}


def _install_graph_customop(torch, lib, reg, dense=None, dense_thresh=512):
    """reg = [(gu_QBSparse, dn_QBSparse)] * NL, built BEFORE LLM(). Registers quadbit::fused_mlp, tags each
    LlamaMLP with its construction-order layer index, and class-patches LlamaMLP.forward to call the op.

    dense (Track 4C phase-adaptive): [(guq, gu_sf, gu_gs, dnq, dn_sf, dn_gs)] * NL = the SAME recovered
    weights re-quantized to dense NVFP4 (flashinfer layout). When set, the op runs the production
    flashinfer cutlass NVFP4 GEMM for tp >= dense_thresh (prefill/large-M) and the sparse split-K path
    for tp < dense_thresh (decode/small-M). Same weights, layout chosen by effective token count."""
    from vllm.model_executor.models.llama import LlamaMLP
    import os
    g = _QB_OP
    g["reg"], g["lib"], g["call"], g["ctor"], g["nl"] = reg, lib, 0, 0, len(reg)
    g["splits"] = int(os.environ.get("QB_SK_SPLITS", "8"))  # 0 -> disable sk-down (plain fused everywhere)
    g["dense"], g["dense_thresh"], g["call_dense"] = dense, dense_thresh, 0
    if dense is not None:  # phase-adaptive: bind the flashinfer NVFP4 GEMM + swizzled-scale layout enum
        import flashinfer as _fi
        from flashinfer import SfLayout as _SfL
        g["fi"], g["sfl"] = _fi, _SfL.layout_128x4
    out_f = reg[0][1].out_f

    if not g["init_patched"]:  # tag each MLP with its layer index as it is built (in order, pre-capture)
        _orig_mlp_init = LlamaMLP.__init__

        def _mlp_init(self, *a, **k):
            _orig_mlp_init(self, *a, **k)
            self._qb_lidx = g["ctor"] % g["nl"]
            g["ctor"] += 1
        LlamaMLP.__init__ = _mlp_init
        g["init_patched"] = True

    if not g["registered"]:
        @torch.library.custom_op("quadbit::fused_mlp", mutates_args=())
        def qb_fused_mlp(x: torch.Tensor, layer_idx: int, out_f: int) -> torch.Tensor:
            lead = x.shape[:-1]
            x2f = x.reshape(-1, x.shape[-1])
            t = x2f.shape[0]
            tp = t + (-t) % 128
            gu, dn = g["reg"][layer_idx]
            dev = x.device
            H, Iw = gu.in_f, dn.in_f
            if g["dense"] is not None and tp >= g["dense_thresh"]:
                # PHASE-ADAPTIVE PREFILL/large-M: production flashinfer cutlass NVFP4 over the SAME
                # recovered weights (dense-zero), matching baseline prefill speed. do_shuffle=False +
                # backend=cutlass is the SM120-viable recipe (trtllm refuses sm_120); alpha=1/(gsa*gsw).
                import torch.nn.functional as F
                fi_, sfl = g["fi"], g["sfl"]
                guq, gu_sf, gu_gs, dnq, dn_sf, dn_gs = g["dense"][layer_idx]
                xin = x2f.to(torch.bfloat16).contiguous()  # real t rows (cutlass handles arbitrary M)
                # bf16 abs/max directly (same dynamic range as fp32); .float() here would allocate a
                # multi-GB temp per layer in the prefill hot path. clamp_min guards div-by-zero.
                gsa = (448.0 * 6.0) / xin.abs().max().clamp_min(1e-12)
                xq, x_sf = fi_.nvfp4_quantize(xin, gsa, sfLayout=sfl, do_shuffle=False)
                y = fi_.mm_fp4(xq, guq.T, x_sf, gu_sf.T, (1.0 / (gsa * gu_gs)), torch.bfloat16, None, backend="cutlass")
                h = (F.silu(y[:, :Iw]) * y[:, Iw:]).contiguous()
                gsh = (448.0 * 6.0) / h.abs().max().clamp_min(1e-12)
                hq, h_sf = fi_.nvfp4_quantize(h, gsh, sfLayout=sfl, do_shuffle=False)
                out = fi_.mm_fp4(hq, dnq.T, h_sf, dn_sf.T, (1.0 / (gsh * dn_gs)), torch.bfloat16, None, backend="cutlass")
                g["call_dense"] += 1
                return out.reshape(*lead, dn.out_f).contiguous()
            g["call"] += 1
            x2 = x2f.to(torch.bfloat16)
            if tp != t:
                x2 = torch.cat([x2, x2.new_zeros(tp - t, H)], 0)
            x2 = x2.contiguous()
            Bb = torch.empty((tp, H // 2), dtype=torch.uint8, device=dev)
            sBg = torch.empty((gu.ks, tp, 4), dtype=torch.uint8, device=dev)
            gBg = torch.empty((tp,), dtype=torch.float32, device=dev)
            Cgu = torch.empty((gu.out_f, tp), dtype=torch.bfloat16, device=dev)
            Hb = torch.empty((tp, Iw // 2), dtype=torch.uint8, device=dev)
            sH = torch.empty((Iw // 128, tp, 4), dtype=torch.uint8, device=dev)
            gH = torch.empty((tp,), dtype=torch.float32, device=dev)
            Cout = torch.empty((tp, dn.out_f), dtype=torch.bfloat16, device=dev)
            strm = torch.cuda.current_stream().cuda_stream
            if tp <= 128 and g["splits"] > 0:  # DECODE: split-K down fixes the 16-CTA underfill (sk@8 ~2x)
                Cf = torch.empty((dn.out_f * tp,), dtype=torch.float32, device=dev)
                g["lib"].fused_mlp_2lvl_skdown(
                    x2.data_ptr(), gu.Ac.data_ptr(), gu.scaleA.data_ptr(), gu.meta.data_ptr(),
                    gu.gA.data_ptr(), dn.Ac.data_ptr(), dn.scaleA.data_ptr(), dn.meta.data_ptr(),
                    dn.gA.data_ptr(), Bb.data_ptr(), sBg.data_ptr(), gBg.data_ptr(), Cgu.data_ptr(),
                    Hb.data_ptr(), sH.data_ptr(), gH.data_ptr(), Cout.data_ptr(), Cf.data_ptr(),
                    tp, H, Iw, gu.out_f, dn.out_f, g["splits"], strm)
            else:  # PREFILL: plain fused (enough token-tiles to fill the GPU already)
                g["lib"].fused_mlp_2lvl(
                    x2.data_ptr(), gu.Ac.data_ptr(), gu.scaleA.data_ptr(), gu.meta.data_ptr(),
                    gu.gA.data_ptr(), dn.Ac.data_ptr(), dn.scaleA.data_ptr(), dn.meta.data_ptr(),
                    dn.gA.data_ptr(), Bb.data_ptr(), sBg.data_ptr(), gBg.data_ptr(), Cgu.data_ptr(),
                    Hb.data_ptr(), sH.data_ptr(), gH.data_ptr(), Cout.data_ptr(),
                    tp, H, Iw, gu.out_f, dn.out_f, strm)
            return Cout[:t].reshape(*lead, dn.out_f).contiguous()

        @qb_fused_mlp.register_fake
        def _(x, layer_idx, out_f):
            return x.new_empty((*x.shape[:-1], out_f))

        g["registered"] = True

    def _forward(self, x):
        return torch.ops.quadbit.fused_mlp(x, self._qb_lidx, out_f)

    LlamaMLP.forward = _forward
    return LlamaMLP


def _fused_mlp_fwd(mlp, torch, lib, dev, graph=False):
    """Assign mlp.forward to the FUSED sparse MLP: gate_up sparse GEMM -> C[28672,tp] (no transpose)
    -> fused_swiglu_quant (silu(g)*u + NVFP4 quant, one pass) -> down sparse GEMM. Correct (cos 0.997
    vs two-level; do NOT reuse the model's SiluAndMul on sparse bf16) and fast (~1.25x vs native NVFP4
    MLP at chunk M). Uses mlp._qb_gu / mlp._qb_dn (QBSparse). down activation is single-level (gB=1)."""
    import os
    gu, dn = mlp._qb_gu, mlp._qb_dn
    H, Iw = gu.in_f, dn.in_f  # 4096, 14336
    _fused_single = os.environ.get("QB_FUSED_SINGLE", "1") == "1"  # 1 ctypes call, 0 syncs; "0" -> 6-call A/B
    _persist = os.environ.get("QB_PERSIST_WS", "1") == "1"         # persistent per-shape workspace (capture-safe)
    mlp._qb_ws = {}

    def _workspace(tp):  # persistent per-tp buffers: allocation-free steady state + CUDA-graph-capturable
        ws = mlp._qb_ws.get(tp)
        if ws is None:
            ws = (torch.zeros(tp, H, dtype=torch.bfloat16, device=dev),          # x2 (input staging; pad rows stay 0)
                  torch.empty((tp, H // 2), dtype=torch.uint8, device=dev),      # Bb
                  torch.empty((gu.ks, tp, 4), dtype=torch.uint8, device=dev),    # sBg
                  torch.empty((tp,), dtype=torch.float32, device=dev),           # gBg
                  torch.empty((gu.out_f, tp), dtype=torch.bfloat16, device=dev), # Cgu
                  torch.empty((tp, Iw // 2), dtype=torch.uint8, device=dev),     # Hb
                  torch.empty((Iw // 128, tp, 4), dtype=torch.uint8, device=dev),# sH
                  torch.empty((tp,), dtype=torch.float32, device=dev),           # gH (fused_mlp_2lvl zeroes it)
                  torch.empty((tp, dn.out_f), dtype=torch.bfloat16, device=dev)) # Cout ([tp,out_f] zero-copy)
            mlp._qb_ws[tp] = ws
        return ws

    def fwd(x):
        lead = x.shape[:-1]; x2f = x.reshape(-1, H)
        t = x2f.shape[0]; pad = (-t) % 128; tp = t + pad
        # Persist ONLY the small decode shapes (tp<=512): those are what vLLM's decode CUDA graphs capture
        # and are reused every step. Large prefill tp uses per-call alloc (freed immediately) so persistent
        # Cgu[28672,tp] buffers don't accumulate across chunk shapes and OOM (prefill runs eager anyway).
        if _fused_single and _persist and tp <= 512:
            # PERSISTENT + CAPTURE-SAFE: stage input into a stable buffer, then one no-alloc no-sync fused
            # call on the current (capture) stream. Padded rows [t:tp] produce sliced-off output (each token
            # is independent), so they need no zeroing. Cout is persistent -> consumed by the layer residual
            # before this layer's next call (and is exactly the static-output vLLM CUDA-graph capture needs).
            x2, Bb, sBg, gBg, Cgu, Hb, sH, gH, Cout = _workspace(tp)
            x2[:t].copy_(x2f if x2f.dtype == torch.bfloat16 else x2f.to(torch.bfloat16))
            lib.fused_mlp_2lvl(x2.data_ptr(), gu.Ac.data_ptr(), gu.scaleA.data_ptr(), gu.meta.data_ptr(),
                               gu.gA.data_ptr(), dn.Ac.data_ptr(), dn.scaleA.data_ptr(), dn.meta.data_ptr(),
                               dn.gA.data_ptr(), Bb.data_ptr(), sBg.data_ptr(), gBg.data_ptr(), Cgu.data_ptr(),
                               Hb.data_ptr(), sH.data_ptr(), gH.data_ptr(), Cout.data_ptr(),
                               tp, H, Iw, gu.out_f, dn.out_f, torch.cuda.current_stream().cuda_stream)
            return Cout[:t].reshape(*lead, dn.out_f)
        x2 = x2f.to(torch.bfloat16)
        if pad:
            x2 = torch.cat([x2, x2.new_zeros(pad, H)], 0)
        x2 = x2.contiguous()
        Bb = torch.empty((tp, H // 2), dtype=torch.uint8, device=dev)
        sBg = torch.empty((gu.ks, tp, 4), dtype=torch.uint8, device=dev)
        gBg = torch.empty((tp,), dtype=torch.float32, device=dev)
        Cgu = torch.empty((gu.out_f, tp), dtype=torch.bfloat16, device=dev)
        Hb = torch.empty((tp, Iw // 2), dtype=torch.uint8, device=dev)
        sH = torch.empty((Iw // 128, tp, 4), dtype=torch.uint8, device=dev)
        gH = torch.empty((tp,), dtype=torch.float32, device=dev)   # per-token global (two-level); fused_mlp_2lvl zeroes it (cudaMemsetAsync)
        # ZERO-COPY: down GEMM writes [tp, out_f] token-major (outT=1 epilogue) -> Cout[:t] is a
        # CONTIGUOUS, storage_offset-0 tensor returnable to vLLM with no transpose+copy pass.
        Cout = torch.empty((tp, dn.out_f), dtype=torch.bfloat16, device=dev)
        if _fused_single:
            # ONE ctypes crossing, ZERO device syncs: the whole two-level sparse MLP streams back-to-back
            # on vLLM's per-thread stream (removes 5/6 Python crossings + 2 cudaDeviceSynchronize per layer).
            lib.fused_mlp_2lvl(x2.data_ptr(), gu.Ac.data_ptr(), gu.scaleA.data_ptr(), gu.meta.data_ptr(),
                               gu.gA.data_ptr(), dn.Ac.data_ptr(), dn.scaleA.data_ptr(), dn.meta.data_ptr(),
                               dn.gA.data_ptr(), Bb.data_ptr(), sBg.data_ptr(), gBg.data_ptr(), Cgu.data_ptr(),
                               Hb.data_ptr(), sH.data_ptr(), gH.data_ptr(), Cout.data_ptr(),
                               tp, H, Iw, gu.out_f, dn.out_f, torch.cuda.current_stream().cuda_stream)
            return Cout[:t].reshape(*lead, dn.out_f)
        # per-thread-stream .so: kernels run on vLLM's stream, ordered with the torch ops; no explicit sync
        lib.quantize_act_nvfp4_2lvl(x2.data_ptr(), Bb.data_ptr(), sBg.data_ptr(), gBg.data_ptr(), tp, H)
        lib.sparse_fp4_mm_2lvl(gu.Ac.data_ptr(), Bb.data_ptr(), gu.scaleA.data_ptr(), sBg.data_ptr(),
                               gu.meta.data_ptr(), Cgu.data_ptr(), gu.out_f, tp, H,
                               gu.gA.data_ptr(), gBg.data_ptr())
        # TWO-LEVEL fused swiglu: emits per-token global gH so the down GEMM is two-level (gB=gH), not
        # single-level (gB=1) -- closes the ~11 vs 8.95 fused-path accuracy gap.
        lib.fused_swiglu_quant_2lvl(Cgu.data_ptr(), Cgu.data_ptr() + Iw * tp * 2, Hb.data_ptr(),
                                    sH.data_ptr(), gH.data_ptr(), tp, Iw)
        lib.sparse_fp4_mm_2lvl_t(dn.Ac.data_ptr(), Hb.data_ptr(), dn.scaleA.data_ptr(), sH.data_ptr(),
                                 dn.meta.data_ptr(), Cout.data_ptr(), dn.out_f, tp, Iw,
                                 dn.gA.data_ptr(), gH.data_ptr())
        return Cout[:t].reshape(*lead, dn.out_f)
    if graph:  # class-level graph-break dispatch reads _qb_run; do NOT set instance .forward
        mlp._qb_run = fwd
    else:
        mlp.forward = fwd


NVFP4_CKPT = "nvidia/Llama-3.1-8B-Instruct-NVFP4"
BF16_CKPT = "meta-llama/Llama-3.1-8B-Instruct"
RECOVERED_CKPT = "/cache/recovered_Meta-Llama-3-8B_P30000_p25000_2sh_lr3e-05.pt"  # base (bf16 non-MLP) for serve_recovered
# recovered-Instruct MLP weights, matched to the NVFP4 Instruct model serve_densify/serve_hybrid load
RECOVERED_INSTRUCT_CKPT = "/cache/recovered_Llama-3.1-8B-Instruct_P30000_p25000_2sh_lr3e-05.pt"


@app.function(gpu="RTX-PRO-6000", timeout=7200, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def serve_hybrid(do_ppl: bool = True, util: float = 0.8, instrument: bool = False,
                 fused: bool = True, ppl_only: bool = False, baseline: bool = False,
                 recovered_ckpt: str = "", graph: bool = False, splits: int = 8,
                 crossover: bool = False, versweep: bool = False,
                 phase_adaptive: bool = False, dense_thresh: int = 512) -> None:
    import os
    # The dense-prefill branch lives ONLY in the graph custom op (installed under graph+fused+not baseline).
    # Without those, phase_adaptive would silently run pure sparse yet still tag/write phaseadaptive output,
    # mislabeling all-sparse data. Fail fast instead so the CSV/tag can never misrepresent the run.
    if phase_adaptive and not (graph and fused and not baseline):
        raise ValueError("phase_adaptive requires graph=True, fused=True, baseline=False "
                         "(the dense-prefill branch is only installed in the graph custom op)")
    os.environ["QB_SK_SPLITS"] = str(splits)  # sk-down split factor read by _install_graph_customop
    # Option 1: vLLM native NVFP4 for ALL non-MLP linears (attention/qkv/o/lm_head 4-bit), quadbit
    # sparse two-level for the MLP on EVERY M (prefill + decode; same weights, no mode-dependence).
    # Non-MLP stays NVFP4 (log confirms modelopt_fp4). MLP weights are the true bf16 Instruct weights,
    # magnitude pair-2:4 pruned (no recovery here -> PPL is the magnitude number, reported honestly;
    # the recovered accuracy is the base-model 8.47 measured through-kernel elsewhere).
    import ctypes
    import gc
    import math
    import subprocess
    import time

    import torch
    from vllm import LLM, SamplingParams

    def gpu_mib():
        return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True).stdout.strip().splitlines()[0])

    # 1) side-load true bf16 Instruct MLP weights on CPU (same checkpoint family as the NVFP4 model).
    # baseline=True: skip side-load + patch entirely -> pure vLLM native NVFP4 (the speed comparand,
    # measured in THIS identical harness/util so the fused-vs-NVFP4 delta is apples-to-apples).
    mlpw = []
    if not baseline and recovered_ckpt:
        # UNIFIED row: recovered-Instruct sparse MLP (correct accuracy) + NVFP4 non-MLP, ONE checkpoint.
        rec = torch.load(recovered_ckpt, map_location="cpu", weights_only=True)["weights"]  # [gate,up,down]*32
        for li in range(len(rec) // 3):
            gu = torch.cat([rec[3 * li], rec[3 * li + 1]], 0).clone()
            mlpw.append((gu, rec[3 * li + 2].clone()))
        del rec; gc.collect()
        print(f"loaded RECOVERED-Instruct MLP weights for {len(mlpw)} layers from {recovered_ckpt}", flush=True)
    elif not baseline:
        from transformers import AutoModelForCausalLM
        src = AutoModelForCausalLM.from_pretrained(BF16_CKPT, dtype=torch.bfloat16, low_cpu_mem_usage=True)
        for layer in src.model.layers:
            gu = torch.cat([layer.mlp.gate_proj.weight.data, layer.mlp.up_proj.weight.data], 0).clone()
            mlpw.append((gu, layer.mlp.down_proj.weight.data.clone()))
        del src; gc.collect()
        print(f"side-loaded bf16 MLP weights for {len(mlpw)} layers", flush=True)

    # 2) vLLM native NVFP4 (non-MLP linears 4-bit). Lower util to leave room for sparse MLP buffers.
    # graph=True: enforce_eager=False so vLLM captures the WHOLE forward as CUDA graphs (Route B). The
    # patched MLP is capture-safe (persistent workspaces, explicit-stream kernels). Whether vLLM's V1
    # compile/piecewise-cudagraph path admits the ctypes MLP is the open question this measures.
    S, GEN = 2048, 128
    # crossover (Track 4) sweeps prompt up to 8192 + gen up to 1024; size the context window for it.
    mm_len = (8192 + 1024 + 16) if crossover else (S + GEN + 16)
    if graph and fused and not baseline:
        # Build QBSparse for all layers and register the custom op BEFORE LLM() so vLLM's one-shot init
        # cudagraph capture records SPARSE per-layer kernels (see _install_graph_customop).
        lib = ctypes.CDLL(SO_PATH)
        lib.fused_mlp_2lvl.argtypes = [ctypes.c_void_p] * 17 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
        lib.fused_mlp_2lvl_skdown.argtypes = [ctypes.c_void_p] * 18 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
        lib.qb_init_sk_attrs()  # set matmul_sp_sk smem attr pre-capture too (same reason as below)
        lib.qb_init_func_attrs()  # set matmul_sp smem attr NOW (pre-capture): cudaFuncSetAttribute is a
        # host call illegal inside vLLM's CUDA-graph capture, and the lazy run_sp_mm path would otherwise
        # first hit it during capture. call_once in the .so makes this the single warmup that marks the flag.
        dev = torch.device("cuda")
        QBSparse = _qbsparse_factory(torch, lib, dev)
        # Track 4C: dense NVFP4 (flashinfer, do_shuffle=False + cutlass) over the SAME recovered weights,
        # for the prefill/large-M branch. Quantized ONCE here; the op's global scale = (448*6)/amax.
        _fi = _SfL = None
        if phase_adaptive:
            import flashinfer as _fi
            from flashinfer import SfLayout as _SfL

            def _dq(W):  # -> (packed_fp4, swizzled_scale, global_scale=(448*6)/amax)
                gsw = (448.0 * 6.0) / W.abs().max().clamp_min(1e-12)
                wq, w_sf = _fi.nvfp4_quantize(W, gsw, sfLayout=_SfL.layout_128x4, do_shuffle=False)
                return wq, w_sf, gsw
        reg = []
        dense_reg = [] if phase_adaptive else None
        for li in range(len(mlpw)):
            gu, dn = mlpw[li]
            gud, dnd = gu.to(dev), dn.to(dev)
            reg.append((QBSparse(gud), QBSparse(dnd)))
            if phase_adaptive:
                guq, gu_sf, gu_gs = _dq(gud)
                dnq, dn_sf, dn_gs = _dq(dnd)
                dense_reg.append((guq, gu_sf, gu_gs, dnq, dn_sf, dn_gs))
            mlpw[li] = None
        _install_graph_customop(torch, lib, reg, dense=dense_reg, dense_thresh=dense_thresh)
        gc.collect(); torch.cuda.empty_cache()
        print(f"registered quadbit::fused_mlp custom op with {len(reg)} layers (pre-LLM capture); "
              f"QB_SK_SPLITS={_QB_OP['splits']} phase_adaptive={phase_adaptive} dense_thresh={dense_thresh}", flush=True)
    # crossover measures per-request prefill+decode latency, so prefix caching MUST be off: otherwise
    # the second (max_tokens=G) generate reuses the first's (max_tokens=1) prompt KV and skips the real
    # prefill, corrupting the decode = total - ttft decomposition (and hiding sparse's prefill deficit).
    llm = LLM(model=NVFP4_CKPT, enforce_eager=not graph, max_model_len=mm_len,
              kv_cache_dtype="auto", gpu_memory_utilization=util,
              enable_prefix_caching=not (crossover or versweep))
    print(f"NON_MLP_QUANT = {llm.llm_engine.model_config.quantization}; graph={graph}", flush=True)
    mem_load = gpu_mib()
    if graph and fused and not baseline:  # custom op already registered pre-LLM; skip the eager patch loop
        mem_patched = mem_load
        print(f"graph mode: {_QB_OP['nl']} MLPs via quadbit::fused_mlp custom op (captured in cudagraph)", flush=True)
        _graph_customop = True
    else:
        _graph_customop = False

    lib = ctypes.CDLL(SO_PATH)
    lib.sparse_fp4_mm_2lvl.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    lib.sparse_fp4_mm_2lvl_t.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    lib.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    lib.fused_swiglu_quant.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    lib.fused_swiglu_quant_2lvl.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 2
    lib.fused_mlp_2lvl.argtypes = [ctypes.c_void_p] * 17 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
    dev = torch.device("cuda")
    QBSparse = _qbsparse_factory(torch, lib, dev)
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    # FUSED sparse MLP: gate_up sparse GEMM -> C[28672,tp] (no transpose) -> fused_swiglu_quant
    # (silu(g)*u + NVFP4 quant, one pass, no bf16 intermediate) -> down sparse GEMM. This is BOTH the
    # correct path (cos 0.997 vs unfused two-level; the model's SiluAndMul custom op must NOT be reused
    # on sparse bf16 output -> that produced the 129854-PPL garbage) AND the fast one (~1.25x vs native
    # NVFP4 MLP at chunk M). down activation is single-level (gB=1) as fused_swiglu_quant emits.
    def patch(mlp):  # fused (default) or unfused two-level plain-SwiGLU (discriminating/accuracy control)
        if fused:
            _fused_mlp_fwd(mlp, torch, lib, dev, graph=graph); return
        gu, dn, Iw = mlp._qb_gu, mlp._qb_dn, mlp._qb_dn.in_f

        def fwd_unf(x):  # plain SwiGLU (NOT the model's SiluAndMul -> that was the cos~0 bug), two-level
            y = gu.forward(x)
            return dn.forward(torch.nn.functional.silu(y[..., :Iw]) * y[..., Iw:])
        if graph:
            mlp._qb_run = fwd_unf
        else:
            mlp.forward = fwd_unf

    for li, layer in enumerate([] if _graph_customop else (model.model.layers if not baseline else [])):
        gu, dn = mlpw[li]
        mlp = layer.mlp
        mlp._qb_gu = QBSparse(gu.to(dev)); mlp._qb_dn = QBSparse(dn.to(dev))
        mlpw[li] = None  # free CPU copy as we go
        patch(mlp)
    if not _graph_customop:
        del mlpw; gc.collect(); torch.cuda.empty_cache()
        mem_patched = gpu_mib()
        npatched = 0 if baseline else len(model.model.layers)
        print(f"patched {npatched} MLPs to sparse (all M); device MiB load={mem_load} "
              f"post-patch={mem_patched} (+{mem_patched - mem_load} for sparse MLP buffers)", flush=True)

    if instrument:  # Option 3: explain the parity - is per-call overhead eating the GEMM win?
        import torch.nn.functional as F
        gu = model.model.layers[0].mlp._qb_gu  # gate_up: out=28672 in=4096
        Wb = torch.randn(gu.out_f, gu.in_f, device=dev, dtype=torch.bfloat16)  # dense bf16 proxy

        def ev(fn, it=50):
            for _ in range(8):
                fn()
            torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True)
            s.record()
            for _ in range(it):
                fn()
            e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it

        print("INSTRUMENT gate_up MLP (out=28672,in=4096) full sparse fwd vs mma-only vs bf16 dense:", flush=True)
        for M in (256, 2048, 8192):
            x = torch.randn(M, gu.in_f, device=dev, dtype=torch.bfloat16)
            # mma-only: prepack activations once, time just the kernel
            xp = x.contiguous(); Bb = torch.empty((M, gu.in_f // 2), dtype=torch.uint8, device=dev)
            sB = torch.empty((gu.ks, M, 4), dtype=torch.uint8, device=dev); gB = torch.empty((M,), dtype=torch.float32, device=dev)
            lib.quantize_act_nvfp4_2lvl(xp.data_ptr(), Bb.data_ptr(), sB.data_ptr(), gB.data_ptr(), M, gu.in_f)
            Cc = torch.empty((gu.out_f, M), dtype=torch.bfloat16, device=dev)

            def mma_only():
                lib.sparse_fp4_mm_2lvl(gu.Ac.data_ptr(), Bb.data_ptr(), gu.scaleA.data_ptr(),
                                       sB.data_ptr(), gu.meta.data_ptr(), Cc.data_ptr(),
                                       gu.out_f, M, gu.in_f, gu.gA.data_ptr(), gB.data_ptr())

            def quant_only():
                lib.quantize_act_nvfp4_2lvl(xp.data_ptr(), Bb.data_ptr(), sB.data_ptr(), gB.data_ptr(), M, gu.in_f)

            t_full = ev(lambda: gu.forward(x)); t_mma = ev(mma_only); t_q = ev(quant_only)
            t_bf16 = ev(lambda: torch.matmul(x, Wb.t()))
            print(f"  M={M:5d}: full {t_full:.3f}ms  mma {t_mma:.3f}  quant {t_q:.3f}  "
                  f"alloc/py {t_full - t_mma - t_q:.3f}  | bf16-dense {t_bf16:.3f}  "
                  f"(full/bf16 {t_full / t_bf16:.2f}x, mma/bf16 {t_mma / t_bf16:.2f}x)", flush=True)
        print("INSTRUMENT_DONE", flush=True); return

    def distinct_ids(i):
        return [((j * 131 + i * 7919) % 128000) + 1 for j in range(S)]

    def distinct_short(i):
        return f"Explain concept {i}: how does mechanism {i * 7 + 3} operate in practice, step by step?"

    def tps(prompts, sp):
        torch.cuda.synchronize(); t = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        return outs, time.perf_counter() - t

    if do_ppl:  # PPL through the REAL serving path (prefill M=2048 -> sparse MLP + NVFP4 non-MLP)
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(BF16_CKPT)
        ids = tok("\n\n".join(pq.read_table(hf_hub_download(
            "Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
            repo_type="dataset")).column("text").to_pylist())).input_ids
        wins = [ids[i:i + S] for i in range(0, min(len(ids), 16 * S) - S, S)]
        outs = llm.generate([{"prompt_token_ids": w} for w in wins],
                            SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1), use_tqdm=False)
        nll = n = 0
        for w, o in zip(wins, outs):
            plp = o.prompt_logprobs
            nll += -sum(plp[j][w[j]].logprob for j in range(1, S)); n += S - 1
        wtag = f"recovered ({recovered_ckpt.split('/')[-1]})" if recovered_ckpt else "magnitude pair-2:4, no recovery"
        mlptag = ("phase-adaptive dense-NVFP4 (prefill M=%d>=thresh %d)" % (S, dense_thresh)) if phase_adaptive \
            else ("fused two-level" if fused else "unfused two-level")
        print(f"PPL_THROUGH_SERVING {math.exp(nll / n):.4f} (MLP={wtag}; non-MLP NVFP4; {mlptag}; {len(wins)}x{S})", flush=True)
        if _graph_customop:  # phase-adaptive: prefill PPL exercises the DENSE branch (dense_calls>0, sparse=0)
            print(f"PPL_PATH SPARSE_CALLS={_QB_OP['call']} DENSE_CALLS={_QB_OP['call_dense']}", flush=True)
    if ppl_only:
        print("PPL_ONLY_DONE", flush=True); return

    if crossover:  # Track 4: end-to-end request-regime crossover matrix (this row = NVFP4 baseline
        # or sparse+split-K, decided by how the harness was launched). Per (B, prompt, gen) cell:
        # TTFT (prefill+1st tok), total request latency, decode time, TPOT, and throughputs. TTFT is
        # a separate max_tokens=1 pass; total is the full ignore_eos generation. Infeasible cells
        # (KV pool can't hold B*(prompt+gen)) are caught and skipped so the matrix completes.
        llm.generate([distinct_short(0)], SamplingParams(temperature=0, max_tokens=8), use_tqdm=False)  # warmup
        mem_peak = gpu_mib()

        def ids_len(i, P):  # P-dependent so different prompt lengths are NOT prefixes of each other
            return [((j * 131 + i * 7919 + P * 104729) % 128000) + 1 for j in range(P)]

        # persist every cell to the shared volume too: Modal log retention keeps only the tail, so the
        # early (small-B) cells scroll off. The CSV on /cache is the authoritative complete matrix.
        row_tag = "nvfp4" if baseline else ("phaseadaptive" if phase_adaptive else "sparse")
        csv_path = f"/cache/crossover_{row_tag}.csv"
        hdr = "B,prompt,gen,ttft_s,total_s,decode_s,tpot_ms,prefill_tps,decode_tps,out_tps,total_tps"
        csv = open(csv_path, "w")
        csv.write(hdr + "\n"); csv.flush()
        print("CX_HDR " + hdr, flush=True)
        for B in (1, 8, 32, 64):
            for P in (128, 512, 2048, 8192):
                prompts = [{"prompt_token_ids": ids_len(i, P)} for i in range(B)]
                # TTFT (prefill + 1st token) is gen-independent, so measure it ONCE per (B,P). With prefix
                # caching off, each total-latency call below re-prefills the same prompt cold, so
                # decode = total - ttft isolates the (G-1) decode steps against a consistent prefill.
                # Warm THIS (B,P) shape first (graph replay / lazy init) so the cold-vs-warm mismatch does
                # not inflate ttft and understate decode for the first gen in the block.
                try:
                    llm.generate(prompts, SamplingParams(temperature=0, max_tokens=2), use_tqdm=False)
                    _, ttft = tps(prompts, SamplingParams(temperature=0, max_tokens=1))
                except Exception as e:
                    print(f"CX {B},{P},*,SKIP {type(e).__name__}: {str(e)[:80]}", flush=True)
                    csv.write(f"{B},{P},all,SKIP\n"); csv.flush()
                    torch.cuda.empty_cache(); continue
                for G in (16, 32, 64, 128, 256, 512, 1024):
                    try:
                        outs, total = tps(prompts, SamplingParams(temperature=0, max_tokens=G, ignore_eos=True))
                        gen = sum(len(o.outputs[0].token_ids) for o in outs)
                        mem_peak = max(mem_peak, gpu_mib())
                        decode = max(total - ttft, 1e-6)
                        tpot = decode / max(gen / B - 1, 1) * 1000  # avg per-token decode latency (ms)
                        row = (f"{B},{P},{G},{ttft:.4f},{total:.4f},{decode:.4f},{tpot:.3f},"
                               f"{B * P / ttft:.0f},{gen / decode:.0f},{gen / total:.0f},{B * (P + G) / total:.0f}")
                        print("CX " + row, flush=True)
                        csv.write(row + "\n"); csv.flush()
                    except Exception as e:
                        print(f"CX {B},{P},{G},SKIP {type(e).__name__}: {str(e)[:80]}", flush=True)
                        csv.write(f"{B},{P},{G},SKIP\n"); csv.flush()
                        torch.cuda.empty_cache()
        csv.close()
        tag = ("full-NVFP4-baseline" if baseline else
               ("phase-adaptive (dense-prefill/sparse-decode, thresh=%d)" % dense_thresh if phase_adaptive else
                "nvfp4-base+sparse-MLP" + ("" if fused else "-unfused2lvl")))
        print(f"CROSSOVER_DONE [{tag}] util={util} device_MiB load={mem_load} peak={mem_peak}", flush=True)
        if _graph_customop:
            print(f"SPARSE_CALLS = {_QB_OP['call']} DENSE_CALLS = {_QB_OP['call_dense']} "
                  f"(both >0 in phase-adaptive proves the split ran)", flush=True)
        return

    if versweep:  # Track 4B: decode throughput vs effective M (verification/multi-token shape M = B*k).
        # A speculative step verifies k candidate tokens per sequence, so the MLP sees M = B*k rows in one
        # decode forward. Sweeping the decode batch is exactly that shape. Tests whether the sparse split-K
        # decode margin over NVFP4 EXPANDS with M (the hypothesis) or shrinks (the small-M underfill fix
        # fading as NVFP4 also fills the GPU). Fixed short prompt, gen=256; decode isolated via ttft.
        P, G = 128, 256
        llm.generate([distinct_short(0)], SamplingParams(temperature=0, max_tokens=8), use_tqdm=False)
        mem_peak = gpu_mib()

        def ids_v(i):
            return [((j * 131 + i * 7919 + 7) % 128000) + 1 for j in range(P)]

        csv_path = f"/cache/versweep_{'nvfp4' if baseline else 'sparse'}.csv"
        vc = open(csv_path, "w"); vc.write("M,ttft_s,total_s,decode_s,decode_tps,tpot_ms\n"); vc.flush()
        print("VS_HDR M,ttft_s,total_s,decode_s,decode_tps,tpot_ms", flush=True)
        for M in (1, 8, 16, 32, 64, 128, 256, 512, 1024):
            prompts = [{"prompt_token_ids": ids_v(i)} for i in range(M)]
            try:
                llm.generate(prompts, SamplingParams(temperature=0, max_tokens=2), use_tqdm=False)  # warm shape
                _, ttft = tps(prompts, SamplingParams(temperature=0, max_tokens=1))
                outs, total = tps(prompts, SamplingParams(temperature=0, max_tokens=G, ignore_eos=True))
                gen = sum(len(o.outputs[0].token_ids) for o in outs)
                mem_peak = max(mem_peak, gpu_mib())
                decode = max(total - ttft, 1e-6)
                tpot = decode / max(gen / M - 1, 1) * 1000
                row = f"{M},{ttft:.4f},{total:.4f},{decode:.4f},{gen / decode:.0f},{tpot:.3f}"
                print("VS " + row, flush=True); vc.write(row + "\n"); vc.flush()
            except Exception as e:
                print(f"VS {M},SKIP {type(e).__name__}: {str(e)[:80]}", flush=True)
                vc.write(f"{M},SKIP\n"); vc.flush(); torch.cuda.empty_cache()
        vc.close()
        tag = "full-NVFP4-baseline" if baseline else "nvfp4-base+sparse-MLP"
        print(f"VERSWEEP_DONE [{tag}] util={util} peak={mem_peak}", flush=True)
        if _graph_customop:
            print(f"SPARSE_CALLS = {_QB_OP['call']}", flush=True)
        return

    llm.generate([distinct_short(0)], SamplingParams(temperature=0, max_tokens=8), use_tqdm=False)  # warmup
    mem_peak = gpu_mib()
    print(f"{'B':>4} | {'prefill tok/s':>14} | {'decode tok/s':>13}", flush=True)
    for B in (1, 8, 32, 64):
        _, pf_dt = tps([{"prompt_token_ids": distinct_ids(i)} for i in range(B)],
                       SamplingParams(temperature=0, max_tokens=1))
        outs, dc_dt = tps([distinct_short(i) for i in range(B)],
                          SamplingParams(temperature=0, max_tokens=GEN, ignore_eos=True))
        gen = sum(len(o.outputs[0].token_ids) for o in outs)
        mem_peak = max(mem_peak, gpu_mib())
        print(f"{B:>4} | {B * S / pf_dt:>14.0f} | {gen / dc_dt:>13.0f}", flush=True)
    tag = "full-NVFP4-baseline" if baseline else ("nvfp4-base+sparse-MLP" + ("" if fused else "-unfused2lvl"))
    print(f"RESULT [{tag}] util={util} device_MiB load={mem_load} "
          f"post-patch={mem_patched} peak={mem_peak} (nvidia-smi, incl KV pool).",
          flush=True)
    if _graph_customop:  # HARD proof the sparse custom op executed (not silently bypassed to dense NVFP4)
        print(f"SPARSE_CALLS = {_QB_OP['call']} (>0 proves quadbit::fused_mlp ran; "
              f"PPL must read ~10.27 sparse, NOT ~7.97 dense)", flush=True)


@app.function(gpu="RTX-PRO-6000", timeout=3600, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def serve_gu_sparse(util: float = 0.8, do_ppl: bool = True) -> None:
    # BRANCH A (correct fallback serving row): vLLM native NVFP4 EVERYWHERE (attn/qkv/o + down_proj all
    # 4-bit) + quadbit sparse gate/up (the two GEMMs that WORK through vLLM), native NVFP4 dense down_proj
    # (the accuracy-sensitive matrix + the in-vLLM down-sparse bug -> keep it native). Correct output,
    # keeps the gate/up sparse speedup. gate/up sparse from bf16 Instruct weights (magnitude 2:4, no
    # recovery -> PPL reported honestly). Only gate_up.forward is patched; down stays vLLM modelopt_fp4.
    import ctypes
    import gc
    import math
    import subprocess
    import time

    import torch
    from vllm import LLM, SamplingParams

    def gpu_mib():
        return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True).stdout.strip().splitlines()[0])

    from transformers import AutoModelForCausalLM
    src = AutoModelForCausalLM.from_pretrained(BF16_CKPT, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    guw = [torch.cat([layer.mlp.gate_proj.weight.data, layer.mlp.up_proj.weight.data], 0).clone()
           for layer in src.model.layers]
    del src; gc.collect()
    print(f"side-loaded bf16 gate_up for {len(guw)} layers", flush=True)

    S, GEN = 2048, 128
    llm = LLM(model=NVFP4_CKPT, enforce_eager=True, max_model_len=S + GEN + 16,
              kv_cache_dtype="auto", gpu_memory_utilization=util)
    print(f"NON_MLP_QUANT = {llm.llm_engine.model_config.quantization}", flush=True)
    mem_load = gpu_mib()
    lib = ctypes.CDLL(SO_PATH)
    lib.sparse_fp4_mm_2lvl.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    lib.sparse_fp4_mm_2lvl_t.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    lib.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    dev = torch.device("cuda")
    QBSparse = _qbsparse_factory(torch, lib, dev)
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    I = 14336
    for li, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        mlp._qb_gu = QBSparse(guw[li].to(dev)); guw[li] = None
        native_down = mlp.down_proj  # vLLM modelopt_fp4 RowParallelLinear (kept)

        def make_fwd(m, dn):
            def fwd(x):  # sparse gate/up -> SwiGLU -> native NVFP4 down
                y = m._qb_gu.forward(x)
                h = torch.nn.functional.silu(y[..., :I]) * y[..., I:]
                out = dn(h)
                return out[0] if isinstance(out, tuple) else out
            return fwd
        mlp.forward = make_fwd(mlp, native_down)
    del guw; gc.collect(); torch.cuda.empty_cache()
    mem_patched = gpu_mib()
    # gate+up are 2 of 3 MLP GEMMs; per Llama-3-8B FLOPs gate_up = 2*4096*14336, down = 4096*14336 ->
    # gate/up = 2/3 of MLP matmul FLOPs sparsified.
    print(f"patched {len(model.model.layers)} MLPs: sparse gate/up + native NVFP4 down; "
          f"MiB load={mem_load} post-patch={mem_patched}", flush=True)

    def distinct_ids(i):
        return [((j * 131 + i * 7919) % 128000) + 1 for j in range(S)]

    def distinct_short(i):
        return f"Explain concept {i}: how does mechanism {i * 7 + 3} operate in practice, step by step?"

    def tps(prompts, sp):
        torch.cuda.synchronize(); t = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        return outs, time.perf_counter() - t

    if do_ppl:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(BF16_CKPT)
        ids = tok("\n\n".join(pq.read_table(hf_hub_download(
            "Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
            repo_type="dataset")).column("text").to_pylist())).input_ids
        wins = [ids[i:i + S] for i in range(0, min(len(ids), 16 * S) - S, S)]
        outs = llm.generate([{"prompt_token_ids": w} for w in wins],
                            SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1), use_tqdm=False)
        nll = n = 0
        for w, o in zip(wins, outs):
            plp = o.prompt_logprobs
            nll += -sum(plp[j][w[j]].logprob for j in range(1, S)); n += S - 1
        print(f"PPL_THROUGH_SERVING {math.exp(nll / n):.4f} (sparse gate/up magnitude-2:4 no-recovery, "
              f"native NVFP4 down; {len(wins)}x{S})", flush=True)

    llm.generate([distinct_short(0)], SamplingParams(temperature=0, max_tokens=8), use_tqdm=False)
    mem_peak = gpu_mib()
    print(f"{'B':>4} | {'prefill tok/s':>14} | {'decode tok/s':>13}", flush=True)
    for B in (1, 8, 32, 64):
        _, pf_dt = tps([{"prompt_token_ids": distinct_ids(i)} for i in range(B)],
                       SamplingParams(temperature=0, max_tokens=1))
        outs, dc_dt = tps([distinct_short(i) for i in range(B)],
                          SamplingParams(temperature=0, max_tokens=GEN, ignore_eos=True))
        gen = sum(len(o.outputs[0].token_ids) for o in outs)
        mem_peak = max(mem_peak, gpu_mib())
        print(f"{B:>4} | {B * S / pf_dt:>14.0f} | {gen / dc_dt:>13.0f}", flush=True)
    print(f"RESULT [gu-sparse+native-down] util={util} MiB load={mem_load} post-patch={mem_patched} "
          f"peak={mem_peak}. Sparse gate/up (2/3 of MLP matmul FLOPs), native NVFP4 down.", flush=True)


@app.function(gpu="RTX-PRO-6000", timeout=3600, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def serve_densify(policy: str = "none", recovered_ckpt: str = "", util: float = 0.8,
                  do_ppl: bool = True, speed: bool = False) -> None:
    # TRACK 3: reverse hybrid densification accuracy Pareto. Start from the all-MLP sparse recovered
    # Instruct ckpt (PPL 10.2709) and REVERT selected projections to the model's stock dense NVFP4
    # weights (accuracy back toward 7.97), keeping the rest sparse. Measures PPL through the deployed
    # serving path (+ optional tok/s) for a densification policy. PPL is fusion/graph-invariant, so
    # this eager unfused mixed path reports the deployed accuracy.
    #   policy grammar (comma-separated tokens; a projection is DENSE if any token selects it):
    #     none                -> all sparse (reproduces 10.2709 baseline)
    #     all                 -> all dense NVFP4 (reproduces ~7.97 sanity)
    #     down                -> down_proj dense (all layers), gate_up sparse
    #     gateup              -> gate_up dense (all layers), down sparse
    #     L<a>-<b>            -> whole MLP dense for layers a..b inclusive (rest sparse)
    #     gu:<a>-<b>          -> gate_up dense only for layers a..b (down stays sparse)
    #     dn:<a>-<b>          -> down dense only for layers a..b (gate_up stays sparse)
    #   e.g. "down,L22-31" = dense down everywhere + fully dense late layers;
    #        "gu:22-31" = dense gate_up in the late layers only, everything else sparse.
    import ctypes
    import gc
    import math
    import subprocess
    import time

    import torch
    from vllm import LLM, SamplingParams

    def gpu_mib():
        return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True).stdout.strip().splitlines()[0])

    toks = [t for t in policy.split(",") if t]

    def is_dense(li: int, proj: str) -> bool:
        for t in toks:
            if t == "all":
                return True
            if t == "down" and proj == "dn":
                return True
            if t == "gateup" and proj == "gu":
                return True
            if t.startswith("gu:") and proj == "gu":
                a, b = t[3:].split("-")
                if int(a) <= li <= int(b):
                    return True
            if t.startswith("dn:") and proj == "dn":
                a, b = t[3:].split("-")
                if int(a) <= li <= int(b):
                    return True
            if t.startswith("L") and "-" in t and ":" not in t:
                a, b = t[1:].split("-")
                if int(a) <= li <= int(b):
                    return True
        return False

    # sparse projections use recovered (pruned+QAT) weights; dense projections use vLLM's stock NVFP4.
    ckpt = recovered_ckpt or RECOVERED_INSTRUCT_CKPT  # Instruct-matched: this fn loads NVFP4_CKPT (Instruct)
    rec = torch.load(ckpt, map_location="cpu", weights_only=True)["weights"]  # [gate,up,down]*32
    nl = len(rec) // 3
    guw = [torch.cat([rec[3 * li], rec[3 * li + 1]], 0).clone() for li in range(nl)]
    dnw = [rec[3 * li + 2].clone() for li in range(nl)]
    del rec; gc.collect()
    print(f"loaded recovered MLP weights for {nl} layers from {ckpt}; policy='{policy}'", flush=True)

    S, GEN = 2048, 128
    llm = LLM(model=NVFP4_CKPT, enforce_eager=True, max_model_len=S + GEN + 16,
              kv_cache_dtype="auto", gpu_memory_utilization=util)
    print(f"NON_MLP_QUANT = {llm.llm_engine.model_config.quantization}", flush=True)
    mem_load = gpu_mib()
    lib = ctypes.CDLL(SO_PATH)
    lib.sparse_fp4_mm_2lvl.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    lib.sparse_fp4_mm_2lvl_t.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    lib.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    dev = torch.device("cuda")
    QBSparse = _qbsparse_factory(torch, lib, dev)
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    I = 14336
    n_gu_dense = n_dn_dense = 0
    for li, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        gu_dense = is_dense(li, "gu")
        dn_dense = is_dense(li, "dn")
        n_gu_dense += gu_dense
        n_dn_dense += dn_dense
        gu_sp = None if gu_dense else QBSparse(guw[li].to(dev))
        dn_sp = None if dn_dense else QBSparse(dnw[li].to(dev))
        guw[li] = dnw[li] = None
        native_gu, native_dn = mlp.gate_up_proj, mlp.down_proj  # vLLM modelopt_fp4 (stock dense NVFP4)

        def make_fwd(m, gu_sp, dn_sp, ngu, ndn):
            def fwd(x):
                if gu_sp is not None:
                    y = gu_sp.forward(x)
                else:
                    y = ngu(x)
                    y = y[0] if isinstance(y, tuple) else y
                h = torch.nn.functional.silu(y[..., :I]) * y[..., I:]
                if dn_sp is not None:
                    return dn_sp.forward(h)
                out = ndn(h)
                return out[0] if isinstance(out, tuple) else out
            return fwd
        mlp.forward = make_fwd(mlp, gu_sp, dn_sp, native_gu, native_dn)
    del guw, dnw; gc.collect(); torch.cuda.empty_cache()
    mem_patched = gpu_mib()
    print(f"policy='{policy}': {n_gu_dense}/{nl} gate_up dense, {n_dn_dense}/{nl} down dense "
          f"(rest sparse); MiB load={mem_load} post-patch={mem_patched}", flush=True)

    def distinct_ids(i):
        return [((j * 131 + i * 7919) % 128000) + 1 for j in range(S)]

    def distinct_short(i):
        return f"Explain concept {i}: how does mechanism {i * 7 + 3} operate in practice, step by step?"

    def tps(prompts, sp):
        torch.cuda.synchronize(); t = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        return outs, time.perf_counter() - t

    if do_ppl:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(BF16_CKPT)
        ids = tok("\n\n".join(pq.read_table(hf_hub_download(
            "Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
            repo_type="dataset")).column("text").to_pylist())).input_ids
        wins = [ids[i:i + S] for i in range(0, min(len(ids), 16 * S) - S, S)]
        outs = llm.generate([{"prompt_token_ids": w} for w in wins],
                            SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1), use_tqdm=False)
        nll = n = 0
        for w, o in zip(wins, outs):
            plp = o.prompt_logprobs
            nll += -sum(plp[j][w[j]].logprob for j in range(1, S)); n += S - 1
        print(f"DENSIFY_PPL policy='{policy}' PPL={math.exp(nll / n):.4f} "
              f"(gu_dense={n_gu_dense}/{nl} dn_dense={n_dn_dense}/{nl})", flush=True)

    if speed:
        llm.generate([distinct_short(0)], SamplingParams(temperature=0, max_tokens=8), use_tqdm=False)
        mem_peak = gpu_mib()
        print(f"{'B':>4} | {'prefill tok/s':>14} | {'decode tok/s':>13}", flush=True)
        for B in (1, 8, 32, 64):
            _, pf_dt = tps([{"prompt_token_ids": distinct_ids(i)} for i in range(B)],
                           SamplingParams(temperature=0, max_tokens=1))
            outs, dc_dt = tps([distinct_short(i) for i in range(B)],
                              SamplingParams(temperature=0, max_tokens=GEN, ignore_eos=True))
            gen = sum(len(o.outputs[0].token_ids) for o in outs)
            mem_peak = max(mem_peak, gpu_mib())
            print(f"{B:>4} | {B * S / pf_dt:>14.0f} | {gen / dc_dt:>13.0f}", flush=True)
        print(f"DENSIFY_SPEED policy='{policy}' util={util} MiB load={mem_load} "
              f"post-patch={mem_patched} peak={mem_peak}", flush=True)
    print(f"DENSIFY_DONE policy='{policy}'", flush=True)


@app.function(gpu="RTX-PRO-6000", timeout=3600, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def serve_recovered(ckpt: str = RECOVERED_CKPT, util: float = 0.85, fused: bool = True,
                    diag: bool = False) -> None:
    # Deliverable #5: RECOVERED-weight PPL through the FUSED serving path. The recovered checkpoint is
    # base Meta-Llama-3-8B, all-MLP SparseGPT-pruned + QAT-recovered (through-kernel ~8.30 via two-level
    # QuadbitLinear). Here we run those SAME recovered weights through the FUSED single-level serving
    # kernel to confirm it preserves the recovered accuracy end-to-end. Base has no NVFP4 ckpt so non-MLP
    # is bf16 (this run is the ACCURACY control; the SPEED headline is the NVFP4-Instruct config).
    import ctypes
    import gc
    import math
    import subprocess

    import torch
    from vllm import LLM, SamplingParams

    S = 2048
    llm = LLM(model=BASE, enforce_eager=True, max_model_len=S + 128 + 16, dtype="bfloat16",
              gpu_memory_utilization=util)
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    lib = ctypes.CDLL(SO_PATH)
    lib.sparse_fp4_mm_2lvl.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    lib.sparse_fp4_mm_2lvl_t.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    lib.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    lib.fused_swiglu_quant.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    lib.fused_swiglu_quant_2lvl.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 2
    lib.fused_mlp_2lvl.argtypes = [ctypes.c_void_p] * 17 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
    dev = torch.device("cuda")
    QBSparse = _qbsparse_factory(torch, lib, dev)

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    ids = tok("\n\n".join(pq.read_table(hf_hub_download(
        "Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
        repo_type="dataset")).column("text").to_pylist())).input_ids
    wins = [ids[i:i + S] for i in range(0, min(len(ids), 16 * S) - S, S)]

    def ppl(tag):
        outs = llm.generate([{"prompt_token_ids": w} for w in wins],
                            SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1), use_tqdm=False)
        nll = n = 0
        for w, o in zip(wins, outs):
            plp = o.prompt_logprobs
            nll += -sum(plp[j][w[j]].logprob for j in range(1, S)); n += S - 1
        print(f"PPL[{tag}] {math.exp(nll / n):.4f} ({len(wins)}x{S})", flush=True)

    ppl("base-unpatched")  # CONTROL: base Meta-Llama-3-8B through this exact harness (expect ~6)

    rec = torch.load(ckpt, map_location="cpu", weights_only=True)["weights"]  # [gate,up,down]*32 recovered bf16
    nl = len(model.model.layers)
    assert len(rec) == 3 * nl, f"recovered weights {len(rec)} != 3*{nl}"

    for li, layer in enumerate(model.model.layers):
        g, u, d = rec[3 * li], rec[3 * li + 1], rec[3 * li + 2]
        mlp = layer.mlp
        mlp._qb_gu = QBSparse(torch.cat([g, u], 0).to(dev))
        mlp._qb_dn = QBSparse(d.to(dev))
    del rec; gc.collect()

    if diag:  # VALIDATE the provenance fix end-to-end: ALL layers full sparse via the fixed QBSparse.forward
        # (unfused two-level: sparse gate_up -> silu -> sparse down). Expect sane PPL (~11 single-level-ish)
        # + coherent generation now that the returned tensor is re-materialized via torch.empty+copy_.
        from vllm import SamplingParams
        for layer in model.model.layers:
            mlp = layer.mlp
            def unf(x, g_=mlp._qb_gu, d_=mlp._qb_dn, iw=mlp._qb_dn.in_f):
                y = g_.forward(x)
                return d_.forward(torch.nn.functional.silu(y[..., :iw]) * y[..., iw:])
            mlp.forward = unf
        torch.cuda.empty_cache()
        prompts = ["The capital of France is", "Water is made of hydrogen and",
                   "To sort a list in Python, you can use the", "The theory of relativity was developed by"]
        outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=24), use_tqdm=False)
        for p, o in zip(prompts, outs):
            print(f"GEN-FIXED {p!r} -> {o.outputs[0].text!r}", flush=True)
        ppl("all-sparse-kernel-provenance-fixed")
        print("DIAG_DONE", flush=True); return

    # ACCURACY+SPEED row (recovered base, zero-copy epilogue): all-MLP sparse via the kernel (fused or
    # unfused two-level), PPL through serving + prefill/decode tok/s + memory. Non-MLP is bf16 (base has no
    # NVFP4), so this is the accuracy proof + the sparse-MLP speedup over the bf16 all-dense path.
    import time
    for layer in model.model.layers:
        mlp = layer.mlp
        if fused:
            _fused_mlp_fwd(mlp, torch, lib, dev)
        else:
            def unf(x, g_=mlp._qb_gu, d_=mlp._qb_dn, iw=mlp._qb_dn.in_f):
                y = g_.forward(x)
                return d_.forward(torch.nn.functional.silu(y[..., :iw]) * y[..., iw:])
            mlp.forward = unf
    torch.cuda.empty_cache()
    ppl(f"recovered-all-MLP-sparse-{'fused' if fused else '2lvl'}")

    def gmib():
        return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True).stdout.strip().splitlines()[0])

    def distinct_ids(i):
        return [((j * 131 + i * 7919) % 128000) + 1 for j in range(S)]

    def distinct_short(i):
        return f"Explain concept {i}: how does mechanism {i * 7 + 3} operate in practice, step by step?"

    llm.generate([distinct_short(0)], SamplingParams(temperature=0, max_tokens=8), use_tqdm=False)
    mem_peak = gmib()
    print(f"{'B':>4} | {'prefill tok/s':>14} | {'decode tok/s':>13}", flush=True)
    for B in (1, 8, 32, 64):
        torch.cuda.synchronize(); t = time.perf_counter()
        llm.generate([{"prompt_token_ids": distinct_ids(i)} for i in range(B)],
                     SamplingParams(temperature=0, max_tokens=1), use_tqdm=False)
        pf = B * S / (time.perf_counter() - t)
        t = time.perf_counter()
        outs = llm.generate([distinct_short(i) for i in range(B)],
                            SamplingParams(temperature=0, max_tokens=128, ignore_eos=True), use_tqdm=False)
        gen = sum(len(o.outputs[0].token_ids) for o in outs); dc = gen / (time.perf_counter() - t)
        mem_peak = max(mem_peak, gmib())
        print(f"{B:>4} | {pf:>14.0f} | {dc:>13.0f}", flush=True)
    print(f"RESULT [recovered-all-MLP-sparse-{'fused' if fused else '2lvl'}] peak_MiB={mem_peak} "
          f"(non-MLP bf16; accuracy proof + sparse-MLP speedup over bf16-dense)", flush=True)


@app.function(gpu="RTX-PRO-6000", timeout=3600, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def bench_mlp(util: float = 0.55, verify: bool = False) -> None:
    # Track A: exact MLP head-to-head. Times vLLM native NVFP4 MLP vs quadbit sparse MLP with CUDA
    # events (p50/p95/min), per component, over a shape sweep + real serving prefill shapes; then
    # measures the empirical MLP fraction of a real prefill forward and prints the Amdahl ceiling
    # table (sparse-MLP speedup needed for +5%/+10% whole-prefill; projected from the measured speedup).
    # Everything eager (sparse ctypes kernel can't be graph-captured yet -> B6). gate+up are FUSED in
    # both vLLM (MergedColumnParallelLinear) and our kernel; separate gate/up are not addressable in
    # the merged layout, reported as one fused unit.
    import ctypes
    import gc

    import torch
    from vllm import LLM, SamplingParams

    H, I = 4096, 14336  # llama-3.1-8B hidden / intermediate
    # ponytail: cap at 65536; the sparse kernel indexes out_f*M in int32 and M=131072 x out=28672
    # overflows 2^31 (illegal access). No real serving call hits M=131072 in one GEMM (vLLM chunks
    # prefill), so cap + note rather than an int64 recompile. Upgrade path: widen kernel indices to long.
    SHAPES = (256, 2048, 8192, 16384, 65536)
    dev = torch.device("cuda")

    def evstats(fn, it=40, warm=8):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        ss = [torch.cuda.Event(True) for _ in range(it)]
        ee = [torch.cuda.Event(True) for _ in range(it)]
        for k in range(it):
            ss[k].record(); fn(); ee[k].record()
        torch.cuda.synchronize()
        ts = sorted(ss[k].elapsed_time(ee[k]) for k in range(it))
        return ts[len(ts) // 2], ts[min(len(ts) - 1, int(0.95 * len(ts)))], ts[0]

    # side-load true bf16 Instruct MLP weights for the sparse path (same family as the NVFP4 model)
    from transformers import AutoModelForCausalLM
    src = AutoModelForCausalLM.from_pretrained(BF16_CKPT, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    gu0 = torch.cat([src.model.layers[0].mlp.gate_proj.weight.data,
                     src.model.layers[0].mlp.up_proj.weight.data], 0).clone()
    dn0 = src.model.layers[0].mlp.down_proj.weight.data.clone()
    del src; gc.collect()

    llm = LLM(model=NVFP4_CKPT, enforce_eager=True, max_model_len=2048 + 128 + 16,
              kv_cache_dtype="auto", gpu_memory_utilization=util)
    print(f"NON_MLP_QUANT = {llm.llm_engine.model_config.quantization}", flush=True)
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    nmlp = model.model.layers[0].mlp  # native NVFP4 MLP (modelopt_fp4)

    lib = ctypes.CDLL(SO_PATH)
    lib.sparse_fp4_mm_2lvl.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    lib.sparse_fp4_mm_2lvl_t.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    lib.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    lib.fused_swiglu_quant.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    lib.fused_swiglu_quant_2lvl.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 2
    lib.fused_mlp_2lvl.argtypes = [ctypes.c_void_p] * 17 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
    QBSparse = _qbsparse_factory(torch, lib, dev)
    qb_gu = QBSparse(gu0.to(dev), keep_wdq=verify)
    qb_dn = QBSparse(dn0.to(dev), keep_wdq=verify)

    if verify:  # CORRECTNESS GATE for the B4 fused path (before trusting its 1.25x speed)
        M = 2048
        x = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
        Wgu, Wdn = gu0.to(dev).float(), dn0.to(dev).float()
        gu_bf = x.float() @ Wgu.t()                    # [M,28672] bf16-input fp32-acc reference
        g_ref, u_ref = gu_bf[:, :I], gu_bf[:, I:]
        h_ref = (g_ref * torch.sigmoid(g_ref)) * u_ref  # silu(gate)*up
        ref = h_ref @ Wdn.t()                           # [M,4096]

        # KERNEL FAITHFULNESS: does the staged 12.9 .so compute what the packed weights represent?
        # Compare qb_gu kernel output to x @ Wdq.t() (Wdq = the dense fake-quant weight the pack encodes,
        # activation full precision). cos ~0.97+ => kernel faithful (only act-fp4 error). Low => .so wrong.
        kgu = qb_gu.forward(x).float()
        fq_gu = x.float() @ qb_gu.Wdq.t()

        def _cmp(a, b):
            return (torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
                    ((a - b).abs().max() / b.abs().max().clamp_min(1e-9)).item())

        c, m = _cmp(kgu, fq_gu)
        print(f"VERIFY kernel(gate_up) vs its-own-Wdq (act full-prec): cos {c:.5f}  maxrel {m:.4f}", flush=True)
        c, m = _cmp(fq_gu, gu_bf)
        print(f"VERIFY Wdq(gate_up) vs bf16-dense (weight fake-quant only): cos {c:.5f}  maxrel {m:.4f}", flush=True)
        print(f"VERIFY act_fn type = {type(nmlp.act_fn).__name__}", flush=True)

        def plain_act(y):  # explicit SwiGLU, not the NVFP4 model's (possibly quant-fused) act_fn
            return torch.nn.functional.silu(y[:, :I]) * y[:, I:]

        uns_af = qb_dn.forward(nmlp.act_fn(qb_gu.forward(x))).float()   # via NVFP4 model act_fn (suspect)
        uns = qb_dn.forward(plain_act(qb_gu.forward(x))).float()        # via plain SwiGLU (unfused two-level)
        # fused single-level path
        Bb = torch.empty((M, H // 2), dtype=torch.uint8, device=dev)
        sBg = torch.empty((H // 128, M, 4), dtype=torch.uint8, device=dev)
        gBg = torch.empty((M,), dtype=torch.float32, device=dev)
        Cgu = torch.empty((qb_gu.out_f, M), dtype=torch.bfloat16, device=dev)
        Hb = torch.empty((M, I // 2), dtype=torch.uint8, device=dev)
        sH = torch.empty((I // 128, M, 4), dtype=torch.uint8, device=dev)
        gB1 = torch.ones((M,), dtype=torch.float32, device=dev)
        Cout = torch.empty((qb_dn.out_f, M), dtype=torch.bfloat16, device=dev)
        lib.quantize_act_nvfp4_2lvl(x.contiguous().data_ptr(), Bb.data_ptr(), sBg.data_ptr(), gBg.data_ptr(), M, H)
        lib.sparse_fp4_mm_2lvl(qb_gu.Ac.data_ptr(), Bb.data_ptr(), qb_gu.scaleA.data_ptr(), sBg.data_ptr(),
                               qb_gu.meta.data_ptr(), Cgu.data_ptr(), qb_gu.out_f, M, H,
                               qb_gu.gA.data_ptr(), gBg.data_ptr())
        lib.fused_swiglu_quant(Cgu.data_ptr(), Cgu.data_ptr() + 14336 * M * 2, Hb.data_ptr(), sH.data_ptr(), M, I)
        lib.sparse_fp4_mm_2lvl(qb_dn.Ac.data_ptr(), Hb.data_ptr(), qb_dn.scaleA.data_ptr(), sH.data_ptr(),
                               qb_dn.meta.data_ptr(), Cout.data_ptr(), qb_dn.out_f, M, I,
                               qb_dn.gA.data_ptr(), gB1.data_ptr())
        fused = Cout.t()[:M].float()

        def cmp(a, b):
            cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
            mr = ((a - b).abs().max() / b.abs().max().clamp_min(1e-9)).item()
            return cos, mr

        for nm, t in (("unfused-plain", uns), ("unfused-act_fn", uns_af), ("fused", fused)):
            cos, mr = cmp(t, ref)
            print(f"VERIFY {nm:>15} vs bf16-ref: cos {cos:.5f}  maxrel {mr:.4f}", flush=True)
        cos, mr = cmp(fused, uns)
        print(f"VERIFY {'fused':>15} vs unfused-plain: cos {cos:.5f}  maxrel {mr:.4f}", flush=True)
        print("VERIFY_DONE", flush=True); return
    del gu0, dn0; gc.collect(); torch.cuda.empty_cache()

    def nat_gu(x):
        return nmlp.gate_up_proj(x)[0]

    def nat_dn(y):
        return nmlp.down_proj(y)[0]

    print("\n=== MLP HEAD-TO-HEAD (ms, p50/p95/min via CUDA events; gate+up FUSED) ===", flush=True)
    print(f"{'M':>7} | {'native gate_up':>22} | {'sparse gate_up':>22} | "
          f"{'native full MLP':>22} | {'sparse full MLP':>22} | {'MLP speedup':>11}", flush=True)
    speedup_at = {}; nfull_at = {}
    for M in SHAPES:
        try:
            x = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
            ngu = evstats(lambda: nat_gu(x))
            sgu = evstats(lambda: qb_gu.forward(x))
            nfull = evstats(lambda: nat_dn(nmlp.act_fn(nat_gu(x))))
            sfull = evstats(lambda: qb_dn.forward(nmlp.act_fn(qb_gu.forward(x))))
            sp = nfull[0] / sfull[0]
            speedup_at[M] = sp; nfull_at[M] = nfull[0]
            print(f"{M:>7} | {ngu[0]:7.3f}/{ngu[1]:6.3f}/{ngu[2]:6.3f} | "
                  f"{sgu[0]:7.3f}/{sgu[1]:6.3f}/{sgu[2]:6.3f} | "
                  f"{nfull[0]:7.3f}/{nfull[1]:6.3f}/{nfull[2]:6.3f} | "
                  f"{sfull[0]:7.3f}/{sfull[1]:6.3f}/{sfull[2]:6.3f} | {sp:8.3f}x", flush=True)
            del x; torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"{M:>7} | OOM (skip, reported honestly)", flush=True)
            torch.cuda.empty_cache()

    # B4: FUSED SwiGLU+quant path. gate_up sparse GEMM -> C[28672,M] (no transpose); fused_swiglu_quant
    # reads gate rows [0,14336) + up rows [14336,28672) straight from C, does silu(g)*u + NVFP4 quant in
    # one pass (no bf16 intermediate, no transpose, no separate quant); down sparse GEMM consumes it.
    # NOTE: fused_swiglu_quant is SINGLE-level (per-32 ue4m3, no per-token gB) -> down activation is
    # single-level here (gB=1), a small accuracy regression vs the two-level path; this measures the
    # SPEED ceiling of fusion. If it clears s>=1.09, a two-level fused variant recovers the accuracy.
    print("\n=== B4 FUSED SwiGLU+quant full MLP (prealloc scratch; single-level down act) ===", flush=True)
    print(f"{'M':>7} | {'fused full MLP':>22} | {'vs native':>10} | {'vs unfused sparse':>17}", flush=True)
    for M in SHAPES:
        try:
            x = torch.randn(M, H, device=dev, dtype=torch.bfloat16).contiguous()
            Bb = torch.empty((M, H // 2), dtype=torch.uint8, device=dev)
            sBg = torch.empty((H // 128, M, 4), dtype=torch.uint8, device=dev)
            gBg = torch.empty((M,), dtype=torch.float32, device=dev)
            Cgu = torch.empty((qb_gu.out_f, M), dtype=torch.bfloat16, device=dev)
            Hb = torch.empty((M, I // 2), dtype=torch.uint8, device=dev)
            sH = torch.empty((I // 128, M, 4), dtype=torch.uint8, device=dev)
            gB1 = torch.ones((M,), dtype=torch.float32, device=dev)
            Cout = torch.empty((qb_dn.out_f, M), dtype=torch.bfloat16, device=dev)
            u_off = 14336 * M * 2  # bf16 bytes: up rows start at row 14336 of C[28672,M]

            def fused_full():
                lib.quantize_act_nvfp4_2lvl(x.data_ptr(), Bb.data_ptr(), sBg.data_ptr(), gBg.data_ptr(), M, H)
                lib.sparse_fp4_mm_2lvl(qb_gu.Ac.data_ptr(), Bb.data_ptr(), qb_gu.scaleA.data_ptr(),
                                       sBg.data_ptr(), qb_gu.meta.data_ptr(), Cgu.data_ptr(),
                                       qb_gu.out_f, M, H, qb_gu.gA.data_ptr(), gBg.data_ptr())
                lib.fused_swiglu_quant(Cgu.data_ptr(), Cgu.data_ptr() + u_off,
                                       Hb.data_ptr(), sH.data_ptr(), M, I)
                lib.sparse_fp4_mm_2lvl(qb_dn.Ac.data_ptr(), Hb.data_ptr(), qb_dn.scaleA.data_ptr(),
                                       sH.data_ptr(), qb_dn.meta.data_ptr(), Cout.data_ptr(),
                                       qb_dn.out_f, M, I, qb_dn.gA.data_ptr(), gB1.data_ptr())

            ff = evstats(fused_full)
            nf = nfull_at.get(M)
            vn = f"{nf / ff[0]:.3f}x" if nf else "n/a"
            vs = f"{(nf / speedup_at[M]) / ff[0]:.3f}x" if nf and M in speedup_at else "n/a"
            print(f"{M:>7} | {ff[0]:7.3f}/{ff[1]:6.3f}/{ff[2]:6.3f} | {vn:>10} | {vs:>17}", flush=True)
            del x, Bb, sBg, gBg, Cgu, Hb, sH, gB1, Cout; torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"{M:>7} | OOM (skip)", flush=True); torch.cuda.empty_cache()

    # empirical MLP fraction of a REAL prefill forward, native vs sparse patched, via event accumulators
    class Acc:
        def __init__(self):
            self.p = []

        def wrap(self, fn):
            def g(*a, **k):
                s = torch.cuda.Event(True); e = torch.cuda.Event(True)
                s.record(); r = fn(*a, **k); e.record(); self.p.append((s, e)); return r
            return g

        def ms(self):
            torch.cuda.synchronize(); return sum(s.elapsed_time(e) for s, e in self.p)

    S = 2048

    def distinct_ids(i):
        return [((j * 131 + i * 7919) % 128000) + 1 for j in range(S)]

    orig_attn = [layer.self_attn.forward for layer in model.model.layers]
    orig_mlp = [layer.mlp.forward for layer in model.model.layers]

    def prefill_fracs(B, patch_sparse):
        am, mm = Acc(), Acc()
        for i, layer in enumerate(model.model.layers):  # always wrap pristine originals (no stacking)
            layer.self_attn.forward = am.wrap(orig_attn[i])
            if patch_sparse:
                mlp = layer.mlp
                mlp._qb_gu, mlp._qb_dn = qb_gu, qb_dn  # share (weights identical for timing)
                f = (lambda m: (lambda x: m._qb_dn.forward(m.act_fn(m._qb_gu.forward(x)))))(mlp)
                layer.mlp.forward = mm.wrap(f)
            else:
                layer.mlp.forward = mm.wrap(orig_mlp[i])
        wall_s = torch.cuda.Event(True); wall_e = torch.cuda.Event(True)
        torch.cuda.synchronize(); wall_s.record()
        llm.generate([{"prompt_token_ids": distinct_ids(i)} for i in range(B)],
                     SamplingParams(temperature=0, max_tokens=1), use_tqdm=False)
        wall_e.record(); torch.cuda.synchronize()
        wall = wall_s.elapsed_time(wall_e)
        return am.ms(), mm.ms(), wall

    print("\n=== EMPIRICAL PREFILL FRACTION (native NVFP4, real forward) ===", flush=True)
    frac_mlp = {}
    for B in (1, 8, 32, 64):
        attn_ms, mlp_ms, wall = prefill_fracs(B, patch_sparse=False)
        f = mlp_ms / wall
        frac_mlp[B] = f
        print(f"B={B:>3} M={B * S:>6}: mlp {mlp_ms:8.2f}ms  attn {attn_ms:8.2f}ms  wall {wall:8.2f}ms  "
              f"| MLP frac of wall {f:5.3f}  attn frac {attn_ms / wall:5.3f}", flush=True)

    print("\n=== AMDAHL CEILING (sparse-MLP speedup needed vs measured) ===", flush=True)
    print(f"{'B':>4} | {'MLP frac f':>10} | {'s for +5%':>10} | {'s for +10%':>11} | "
          f"{'measured s':>10} | {'proj total':>10}", flush=True)
    for B in (1, 8, 32, 64):
        f = frac_mlp[B]
        M = B * S
        s_meas = speedup_at.get(M, speedup_at.get(min(speedup_at, key=lambda k: abs(k - M)), 1.0))

        def need(G):
            d = 1.0 / G - (1.0 - f)
            return f / d if d > 1e-9 else float("inf")

        proj = 1.0 / ((1.0 - f) + f / s_meas)
        n5, n10 = need(1.05), need(1.10)
        print(f"{B:>4} | {f:10.3f} | {('inf' if n5 == float('inf') else f'{n5:.2f}x'):>10} | "
              f"{('inf' if n10 == float('inf') else f'{n10:.2f}x'):>11} | {s_meas:8.3f}x | "
              f"{proj:8.3f}x", flush=True)
    print("BENCH_MLP_DONE", flush=True)


@app.function(gpu="RTX-PRO-6000", timeout=1800, volumes={"/cache": vol})
def graph_probe() -> None:
    # Milestone 2: capture ONE fused_mlp_2lvl block per fixed shape as a CUDA graph; verify replay ==
    # eager (cos/relL2/maxabs/nonfinite) and time eager-vs-graph (p50/p95/min + CPU submit). Decode B<=128
    # all pad to tp=128 (one graph covers all decode batches); prefill chunks 2048/8192/16384/65536.
    import ctypes
    import statistics
    import time

    import torch

    lib = ctypes.CDLL(SO_PATH)
    lib.sparse_fp4_mm_2lvl.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    lib.fused_mlp_2lvl.argtypes = [ctypes.c_void_p] * 17 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
    lib.qb_init_func_attrs()  # set matmul_sp smem attr ONCE (before any capture)
    dev = torch.device("cuda")
    QBSparse = _qbsparse_factory(torch, lib, dev)

    H, Iw, GU, DN = 4096, 14336, 28672, 4096
    torch.manual_seed(0)
    gu = QBSparse((torch.randn(GU, H, device=dev) * 0.02).bfloat16())   # gate_up: out 28672, in 4096
    dn = QBSparse((torch.randn(DN, Iw, device=dev) * 0.02).bfloat16())  # down:    out 4096,  in 14336
    strm = torch.cuda.current_stream().cuda_stream

    def call(tp, bufs):
        x2, Bb, sBg, gBg, Cgu, Hb, sH, gH, Cout = bufs
        lib.fused_mlp_2lvl(x2.data_ptr(), gu.Ac.data_ptr(), gu.scaleA.data_ptr(), gu.meta.data_ptr(),
                           gu.gA.data_ptr(), dn.Ac.data_ptr(), dn.scaleA.data_ptr(), dn.meta.data_ptr(),
                           dn.gA.data_ptr(), Bb.data_ptr(), sBg.data_ptr(), gBg.data_ptr(), Cgu.data_ptr(),
                           Hb.data_ptr(), sH.data_ptr(), gH.data_ptr(), Cout.data_ptr(),
                           tp, H, Iw, GU, DN, torch.cuda.current_stream().cuda_stream)

    def stats_ms(fn, it=200):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(it):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        return statistics.median(ts), ts[int(0.95 * it)], ts[0]

    print(f"{'shape':>8} | {'cos':>8} {'relL2':>9} {'maxabs':>9} {'nonfin':>6} | "
          f"{'eager p50':>9} {'graph p50':>9} {'speedup':>7} | {'eagerCPU':>8} {'graphCPU':>8}", flush=True)
    for tp in (128, 2048, 8192, 16384, 65536):
        x2 = (torch.randn(tp, H, device=dev) * 0.5).bfloat16()
        Bb = torch.empty((tp, H // 2), dtype=torch.uint8, device=dev)
        sBg = torch.empty((H // 128, tp, 4), dtype=torch.uint8, device=dev)
        gBg = torch.empty((tp,), dtype=torch.float32, device=dev)
        Cgu = torch.empty((GU, tp), dtype=torch.bfloat16, device=dev)
        Hb = torch.empty((tp, Iw // 2), dtype=torch.uint8, device=dev)
        sH = torch.empty((Iw // 128, tp, 4), dtype=torch.uint8, device=dev)
        gH = torch.empty((tp,), dtype=torch.float32, device=dev)
        Cout = torch.empty((tp, DN), dtype=torch.bfloat16, device=dev)
        bufs = (x2, Bb, sBg, gBg, Cgu, Hb, sH, gH, Cout)

        call(tp, bufs); torch.cuda.synchronize()          # eager reference
        ref = Cout.clone()

        # capture: warmup on a side stream, then record on the graph's capture stream
        side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                call(tp, bufs)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(g):
                call(tp, bufs)
        except Exception as ex:
            print(f"{tp:>8} | CAPTURE FAILED: {type(ex).__name__} {str(ex)[:120]}", flush=True); continue
        Cout.zero_(); g.replay(); torch.cuda.synchronize()  # replay must reproduce eager ref
        rep = Cout
        num = (rep.float() - ref.float())
        cos = torch.nn.functional.cosine_similarity(rep.float().flatten(), ref.float().flatten(), dim=0).item()
        rell2 = (num.norm() / ref.float().norm().clamp_min(1e-9)).item()
        mx = num.abs().max().item()
        nonfin = int((~torch.isfinite(rep.float())).sum().item())

        e50, e95, emin = stats_ms(lambda: call(tp, bufs))
        g50, g95, gmin = stats_ms(lambda: g.replay())
        # CPU submit time (no sync): how long the host spends launching
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(100):
            call(tp, bufs)
        ecpu = (time.perf_counter() - t) / 100 * 1e3
        t = time.perf_counter()
        for _ in range(100):
            g.replay()
        gcpu = (time.perf_counter() - t) / 100 * 1e3
        torch.cuda.synchronize()
        print(f"{tp:>8} | {cos:8.5f} {rell2:9.5f} {mx:9.4f} {nonfin:>6} | "
              f"{e50:9.3f} {g50:9.3f} {e50 / g50:7.2f}x | {ecpu:8.3f} {gcpu:8.3f}", flush=True)
        del x2, Bb, sBg, gBg, Cgu, Hb, sH, gH, Cout, bufs, g, ref
        torch.cuda.empty_cache()
    print("GRAPH_PROBE_DONE", flush=True)


@app.function(gpu="RTX-PRO-6000", timeout=1800, volumes={"/cache": vol})
def profile_decode() -> None:
    # ONE bounded profiling pass (kernel-level, CUDA events). Kernel times are GRAPH-INVARIANT (a cudagraph
    # removes launch gaps, not kernel duration), so this explains the graph-vs-graph decode loss directly.
    # Per-kernel breakdown of the sparse MLP at decode (M=8/32/64) + prefill (2048/4096), vs a dense-bf16
    # reference. dense-bf16 is 4x the BYTES of dense-FP4, so if bf16-dense already BEATS sparse-FP4 at
    # decode, a dense-NVFP4 decode fallback (0.25x bf16 bytes) would clearly win -> that is the ablation.
    # Weights random 2:4 (GEMM timing is value-independent; the 2:4 density is fixed).
    import ctypes
    import statistics

    import torch

    lib = ctypes.CDLL(SO_PATH)
    lib.sparse_fp4_mm_2lvl.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    lib.sparse_fp4_mm_2lvl_t.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    lib.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    lib.fused_swiglu_quant_2lvl.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 2
    lib.fused_mlp_2lvl.argtypes = [ctypes.c_void_p] * 17 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
    lib.sparse_down_sk_2lvl.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2 + [ctypes.c_int]
    lib.qb_init_func_attrs()
    lib.qb_init_sk_attrs()
    dev = torch.device("cuda")
    QBSparse = _qbsparse_factory(torch, lib, dev)
    H, Iw, GU, DN = 4096, 14336, 28672, 4096
    torch.manual_seed(0)
    gu = QBSparse((torch.randn(GU, H, device=dev) * 0.02).bfloat16())
    dn = QBSparse((torch.randn(DN, Iw, device=dev) * 0.02).bfloat16())
    strm = torch.cuda.current_stream().cuda_stream
    Wgu = (torch.randn(GU, H, device=dev) * 0.02).bfloat16()   # dense bf16 gate_up ref
    Wdn = (torch.randn(DN, Iw, device=dev) * 0.02).bfloat16()  # dense bf16 down ref

    def us(fn, it=300):  # median us over it iters (per-call sync)
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(it):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) * 1e3)
        return statistics.median(ts)

    print(f"{'M':>6} | {'quant':>7} {'gateup':>7} {'swiglu':>7} {'down':>7} {'ksum':>7} {'full':>7} | "
          f"{'bf16gu':>7} {'bf16dn':>7} {'bf16sum':>8} | {'full/bf16':>9}", flush=True)
    for M in (8, 32, 64, 2048, 4096):
        tp = M + (-M) % 128
        x2 = (torch.randn(tp, H, device=dev) * 0.5).bfloat16()
        Bb = torch.empty((tp, H // 2), dtype=torch.uint8, device=dev)
        sBg = torch.empty((H // 128, tp, 4), dtype=torch.uint8, device=dev)
        gBg = torch.empty((tp,), dtype=torch.float32, device=dev)
        Cgu = torch.empty((GU, tp), dtype=torch.bfloat16, device=dev)
        Hb = torch.empty((tp, Iw // 2), dtype=torch.uint8, device=dev)
        sH = torch.empty((Iw // 128, tp, 4), dtype=torch.uint8, device=dev)
        gH = torch.zeros((tp,), dtype=torch.float32, device=dev)
        Cout = torch.empty((tp, DN), dtype=torch.bfloat16, device=dev)
        xd = (torch.randn(M, H, device=dev) * 0.5).bfloat16()
        hd = (torch.randn(M, Iw, device=dev) * 0.5).bfloat16()

        t_q = us(lambda: lib.quantize_act_nvfp4_2lvl(x2.data_ptr(), Bb.data_ptr(), sBg.data_ptr(), gBg.data_ptr(), tp, H))
        t_gu = us(lambda: lib.sparse_fp4_mm_2lvl(gu.Ac.data_ptr(), Bb.data_ptr(), gu.scaleA.data_ptr(), sBg.data_ptr(),
                                                 gu.meta.data_ptr(), Cgu.data_ptr(), GU, tp, H, gu.gA.data_ptr(), gBg.data_ptr()))
        t_sw = us(lambda: lib.fused_swiglu_quant_2lvl(Cgu.data_ptr(), Cgu.data_ptr() + Iw * tp * 2, Hb.data_ptr(),
                                                      sH.data_ptr(), gH.data_ptr(), tp, Iw))
        t_dn = us(lambda: lib.sparse_fp4_mm_2lvl_t(dn.Ac.data_ptr(), Hb.data_ptr(), dn.scaleA.data_ptr(), sH.data_ptr(),
                                                   dn.meta.data_ptr(), Cout.data_ptr(), DN, tp, Iw, dn.gA.data_ptr(), gH.data_ptr()))
        t_full = us(lambda: lib.fused_mlp_2lvl(x2.data_ptr(), gu.Ac.data_ptr(), gu.scaleA.data_ptr(), gu.meta.data_ptr(),
                                               gu.gA.data_ptr(), dn.Ac.data_ptr(), dn.scaleA.data_ptr(), dn.meta.data_ptr(),
                                               dn.gA.data_ptr(), Bb.data_ptr(), sBg.data_ptr(), gBg.data_ptr(), Cgu.data_ptr(),
                                               Hb.data_ptr(), sH.data_ptr(), gH.data_ptr(), Cout.data_ptr(),
                                               tp, H, Iw, GU, DN, strm))
        t_bgu = us(lambda: torch.matmul(xd, Wgu.t()))
        t_bdn = us(lambda: torch.matmul(hd, Wdn.t()))
        ksum = t_q + t_gu + t_sw + t_dn
        print(f"{M:>6} | {t_q:7.1f} {t_gu:7.1f} {t_sw:7.1f} {t_dn:7.1f} {ksum:7.1f} {t_full:7.1f} | "
              f"{t_bgu:7.1f} {t_bdn:7.1f} {t_bgu + t_bdn:8.1f} | {t_full / (t_bgu + t_bdn):9.2f}", flush=True)
        del x2, Bb, sBg, gBg, Cgu, Hb, sH, gH, Cout, xd, hd
        torch.cuda.empty_cache()

    # ---- Track 1: split-K DECODE down (sparse_down_sk_2lvl) vs plain two-level down, per split factor.
    # Correctness: sk-down output must match the plain two-level down (same weights/act/scales) within
    # atomic-reduction noise. Speed: does split-K fill the 16-CTA underfill and beat plain down 111us?
    print("\nSPLIT-K DOWN (out_f=4096, K=14336): plain vs sk@splits; cos/relL2 vs plain two-level down", flush=True)
    print(f"{'M':>6} | {'plain':>7} | {'sk@4':>7} {'sk@8':>7} {'sk@16':>7} {'sk@32':>7} | {'cos':>8} {'relL2':>8}", flush=True)
    for M in (8, 32, 64, 128):
        tp = M + (-M) % 128
        h = (torch.randn(tp, Iw, device=dev) * 0.5).bfloat16()
        Hb = torch.empty((tp, Iw // 2), dtype=torch.uint8, device=dev)
        sH = torch.empty((Iw // 128, tp, 4), dtype=torch.uint8, device=dev)
        gHt = torch.empty((tp,), dtype=torch.float32, device=dev)
        lib.quantize_act_nvfp4_2lvl(h.data_ptr(), Hb.data_ptr(), sH.data_ptr(), gHt.data_ptr(), tp, Iw)
        Cpl = torch.empty((tp, DN), dtype=torch.bfloat16, device=dev)
        Csk = torch.empty((tp, DN), dtype=torch.bfloat16, device=dev)
        Cf = torch.empty((DN * tp,), dtype=torch.float32, device=dev)

        def plain():
            lib.sparse_fp4_mm_2lvl_t(dn.Ac.data_ptr(), Hb.data_ptr(), dn.scaleA.data_ptr(), sH.data_ptr(),
                                     dn.meta.data_ptr(), Cpl.data_ptr(), DN, tp, Iw, dn.gA.data_ptr(), gHt.data_ptr())

        def sk(sp):
            lib.sparse_down_sk_2lvl(dn.Ac.data_ptr(), Hb.data_ptr(), dn.scaleA.data_ptr(), sH.data_ptr(),
                                    dn.meta.data_ptr(), Csk.data_ptr(), Cf.data_ptr(), DN, tp, Iw,
                                    dn.gA.data_ptr(), gHt.data_ptr(), sp)
        t_pl = us(plain)
        tsk = {sp: us(lambda sp=sp: sk(sp)) for sp in (4, 8, 16, 32)}
        plain(); sk(16); torch.cuda.synchronize()  # correctness: sk@16 vs plain
        a, b = Csk.float().flatten(), Cpl.float().flatten()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        rel = ((a - b).norm() / b.norm().clamp_min(1e-9)).item()
        print(f"{M:>6} | {t_pl:7.1f} | {tsk[4]:7.1f} {tsk[8]:7.1f} {tsk[16]:7.1f} {tsk[32]:7.1f} | {cos:8.5f} {rel:8.5f}", flush=True)
        del h, Hb, sH, gHt, Cpl, Csk, Cf
        torch.cuda.empty_cache()
    print("PROFILE_DONE (us; ksum=sum of 4 sub-kernels, full=fused single call, full/bf16<1 => sparse beats "
          "dense-bf16; >1 at decode => dense-NVFP4 fallback would win). sk cos~1/relL2~0 => correct; "
          "sk << plain 111us => split-K fixes the down underfill.", flush=True)


@app.function(gpu="RTX-PRO-6000", timeout=1800, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def phase_probe() -> None:
    # TRACK 4C de-risk: can we run a PRODUCTION NVFP4 dense GEMM over ARBITRARY (recovered pruned,
    # dense-zero) weights, matching baseline speed + correct math? Enumerate the NVFP4 quantize/mm APIs
    # vLLM ships (vllm._custom_ops, flashinfer), numerically validate a dense-zero NVFP4 GEMM vs bf16 ref
    # (cos>0.99), and confirm dense-zero-NVFP4(W) ~= sparse-FP4(W) for the SAME recovered W (the phase
    # boundary must not switch weight semantics). If this passes, the dense-prefill branch is buildable
    # inside the custom op with the production kernel; if not, 4C hits the "toolchain research" stop-cond.
    import ctypes

    import torch

    dev = torch.device("cuda")

    def cos(a, b):
        a, b = a.float().reshape(-1), b.float().reshape(-1)
        return (a @ b / (a.norm() * b.norm()).clamp_min(1e-12)).item()

    # 1) enumerate available production NVFP4 GEMM APIs
    ops = fi = None
    try:
        import vllm._custom_ops as ops
        print("VLLM_OPS", [n for n in ("scaled_fp4_quant", "cutlass_scaled_fp4_mm") if hasattr(ops, n)], flush=True)
    except Exception as e:
        print("VLLM_OPS_IMPORT_FAIL", repr(e), flush=True)
    try:
        import flashinfer as fi
        print("FLASHINFER", getattr(fi, "__version__", "?"),
              [n for n in ("mm_fp4", "nvfp4_quantize", "fp4_quantize") if hasattr(fi, n)], flush=True)
    except Exception as e:
        print("FLASHINFER_IMPORT_FAIL", repr(e), flush=True)

    FP4_MAX, FP8_MAX = 6.0, 448.0

    # 2) SELF-CONTAINED flashinfer path (no vLLM layer layout dependency): quantize BOTH operands with
    #    flashinfer's OWN nvfp4_quantize (self-consistent with mm_fp4) per the documented-verified recipe:
    #    nvfp4_quantize(t, (448*6)/amax, sfLayout=layout_128x4, do_shuffle=False); alpha=1/(gsa*gsb);
    #    mm_fp4(a, b.T, a_s, b_s.T, alpha, block_size=16, use_8x4_sf_layout=False).
    # EXACT flashinfer docstring recipe: SfLayout from top-level; weight nvfp4_quantize with do_shuffle
    # matched to the backend; mm_fp4(a_fp4, b_fp4.T, a_sf, b_sf.T, 1/(gsa*gsw)). Grid backend x weight
    # shuffle to find the SM120-viable combo (trtllm refuses SM120 -> need cutlass/auto).
    from flashinfer import SfLayout
    torch.manual_seed(0)
    H, N = 4096, 512
    Wz = (torch.randn(N, H, device=dev) * 0.02).bfloat16()
    x = (torch.randn(256, H, device=dev) * 0.1).bfloat16()
    ref = x.float() @ Wz.float().t()
    gsa = (FP8_MAX * FP4_MAX) / x.float().abs().nan_to_num().max()
    gsw = (FP8_MAX * FP4_MAX) / Wz.float().abs().nan_to_num().max()
    alpha = (1.0 / (gsa * gsw))
    aq, a_sf = fi.nvfp4_quantize(x, gsa, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    for wshuf in (True, False):
        bq, b_sf = fi.nvfp4_quantize(Wz, gsw, sfLayout=SfLayout.layout_128x4, do_shuffle=wshuf)
        for backend in ("auto", "cutlass", "trtllm"):
            try:
                out = fi.mm_fp4(aq, bq.T, a_sf, b_sf.T, alpha, torch.bfloat16, None, backend=backend)
                print(f"FI wshuf={wshuf} backend={backend} cos={cos(out, ref):.5f} shape={tuple(out.shape)}", flush=True)
            except Exception as e:
                print(f"FI wshuf={wshuf} backend={backend} FAIL {repr(e)[:75]}", flush=True)
    print("PHASE_PROBE_DONE", flush=True)


@app.function(gpu="RTX-PRO-6000", timeout=1800, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def phase_bench() -> None:
    # TRACK 4C root-cause: the phase-adaptive dense-prefill ran ~1.7x SLOWER than NVFP4/sparse. The sparse
    # path (also eager inside the opaque custom op) hits prefill parity via ONE fused ctypes call, so the
    # penalty must be in the DENSE path's structure (2x flashinfer nvfp4_quantize + 2x mm_fp4 + eager silu).
    # Break down us/layer at prefill M and compare to (a) vLLM's native NVFP4 MLP (baseline kernel, eager)
    # and (b) the sparse fused single call. Tells us whether dense-in-op can EVER hit prefill parity.
    import ctypes

    import torch
    import torch.nn.functional as F
    from flashinfer import SfLayout
    from vllm import LLM

    dev = torch.device("cuda")
    FP4M, FP8M = 6.0, 448.0
    import flashinfer as fi
    llm = LLM(model=NVFP4_CKPT, enforce_eager=True, max_model_len=4096, gpu_memory_utilization=0.8)
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    mlp0 = model.model.layers[0].mlp
    gu_proj, dn_proj, act = mlp0.gate_up_proj, mlp0.down_proj, mlp0.act_fn  # native NVFP4 (stock weights)
    H, Iw = 4096, 14336

    lib = ctypes.CDLL(SO_PATH)
    lib.fused_mlp_2lvl.argtypes = [ctypes.c_void_p] * 17 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
    lib.qb_init_func_attrs()
    QBSparse = _qbsparse_factory(torch, lib, dev)
    torch.manual_seed(0)
    guW = (torch.randn(28672, H, device=dev) * 0.02).bfloat16()
    dnW = (torch.randn(4096, Iw, device=dev) * 0.02).bfloat16()
    gu_sp, dn_sp = QBSparse(guW), QBSparse(dnW)

    def dq(W):
        gsw = (FP8M * FP4M) / W.float().abs().nan_to_num().max()
        wq, wsf = fi.nvfp4_quantize(W, gsw, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
        return wq, wsf, gsw
    guq, gu_sf, gu_gs = dq(guW)
    dnq, dn_sf, dn_gs = dq(dnW)

    def q(t):
        gs = (FP8M * FP4M) / t.float().abs().nan_to_num().max()
        tq, tsf = fi.nvfp4_quantize(t, gs, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
        return tq, tsf, gs

    def ev(fn, it=30):
        for _ in range(8):
            fn()
        torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(it):
            fn()
        e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it * 1000  # us

    print(f"{'M':>6} | {'q_x':>6} {'mm_gu':>6} {'silu':>6} {'q_h':>6} {'mm_dn':>6} {'DENSE':>6} | "
          f"{'nativeMLP':>9} {'sparse1call':>11}", flush=True)
    for M in (2048, 8192, 16384):
        x = (torch.randn(M, H, device=dev) * 0.1).bfloat16()

        def q_x():
            q(x)

        xq, xsf, xgs = q(x)

        def mm_gu():
            fi.mm_fp4(xq, guq.T, xsf, gu_sf.T, (1.0 / (xgs * gu_gs)), torch.bfloat16, None, backend="cutlass")
        y = fi.mm_fp4(xq, guq.T, xsf, gu_sf.T, (1.0 / (xgs * gu_gs)), torch.bfloat16, None, backend="cutlass")

        def silu():
            (F.silu(y[:, :Iw]) * y[:, Iw:]).contiguous()
        h = (F.silu(y[:, :Iw]) * y[:, Iw:]).contiguous()

        def q_h():
            q(h)
        hq, hsf, hgs = q(h)

        def mm_dn():
            fi.mm_fp4(hq, dnq.T, hsf, dn_sf.T, (1.0 / (hgs * dn_gs)), torch.bfloat16, None, backend="cutlass")

        def dense():
            xq2, xsf2, xgs2 = q(x)
            y2 = fi.mm_fp4(xq2, guq.T, xsf2, gu_sf.T, (1.0 / (xgs2 * gu_gs)), torch.bfloat16, None, backend="cutlass")
            h2 = (F.silu(y2[:, :Iw]) * y2[:, Iw:]).contiguous()
            hq2, hsf2, hgs2 = q(h2)
            fi.mm_fp4(hq2, dnq.T, hsf2, dn_sf.T, (1.0 / (hgs2 * dn_gs)), torch.bfloat16, None, backend="cutlass")

        def native():
            g = gu_proj(x); g = g[0] if isinstance(g, tuple) else g
            hh = act(g)
            o = dn_proj(hh); return o[0] if isinstance(o, tuple) else o

        tp = M + (-M) % 128
        Bb = torch.empty((tp, H // 2), dtype=torch.uint8, device=dev)
        sBg = torch.empty((gu_sp.ks, tp, 4), dtype=torch.uint8, device=dev)
        gBg = torch.empty((tp,), dtype=torch.float32, device=dev)
        Cgu = torch.empty((28672, tp), dtype=torch.bfloat16, device=dev)
        Hb = torch.empty((tp, Iw // 2), dtype=torch.uint8, device=dev)
        sH = torch.empty((Iw // 128, tp, 4), dtype=torch.uint8, device=dev)
        gH = torch.empty((tp,), dtype=torch.float32, device=dev)
        Cout = torch.empty((tp, 4096), dtype=torch.bfloat16, device=dev)
        x2 = x.contiguous()

        def sparse1():
            lib.fused_mlp_2lvl(x2.data_ptr(), gu_sp.Ac.data_ptr(), gu_sp.scaleA.data_ptr(), gu_sp.meta.data_ptr(),
                               gu_sp.gA.data_ptr(), dn_sp.Ac.data_ptr(), dn_sp.scaleA.data_ptr(), dn_sp.meta.data_ptr(),
                               dn_sp.gA.data_ptr(), Bb.data_ptr(), sBg.data_ptr(), gBg.data_ptr(), Cgu.data_ptr(),
                               Hb.data_ptr(), sH.data_ptr(), gH.data_ptr(), Cout.data_ptr(),
                               tp, H, Iw, 28672, 4096, torch.cuda.current_stream().cuda_stream)
        print(f"{M:>6} | {ev(q_x):>6.1f} {ev(mm_gu):>6.1f} {ev(silu):>6.1f} {ev(q_h):>6.1f} {ev(mm_dn):>6.1f} "
              f"{ev(dense):>6.1f} | {ev(native):>9.1f} {ev(sparse1):>11.1f}", flush=True)
    print("PHASE_BENCH_DONE (us/layer; DENSE=my flashinfer 5-op path, nativeMLP=vLLM NVFP4 MLP eager, "
          "sparse1=one fused ctypes call). DENSE>>native => hand-rolled overhead; DENSE~native => flashinfer "
          "mm_fp4 itself is the floor.", flush=True)


@app.local_entrypoint()
def main(mode: str = "smoke", sparse: bool = True, thresh: int = 256, do_ppl: bool = True,
         util: float = 0.8, instrument: bool = False, fused: bool = True, ppl_only: bool = False,
         baseline: bool = False, recovered_ckpt: str = "", graph: bool = False,
         splits: int = 8, crossover: bool = False, policy: str = "none", speed: bool = False,
         versweep: bool = False, phase_adaptive: bool = False, dense_thresh: int = 512) -> None:
    if mode == "graph_probe":
        call = graph_probe.spawn()
        print(f"SPAWN_ID {call.object_id}", flush=True); return
    if mode == "profile_decode":
        call = profile_decode.spawn()
        print(f"SPAWN_ID {call.object_id}", flush=True); return
    if mode == "phase_probe":
        call = phase_probe.spawn()
        print(f"SPAWN_ID {call.object_id}", flush=True); return
    if mode == "phase_bench":
        call = phase_bench.spawn()
        print(f"SPAWN_ID {call.object_id}", flush=True); return
    if mode == "serve":
        call = serve.spawn(sparse, thresh)
    elif mode == "hybrid":
        call = serve_hybrid.spawn(do_ppl, util, instrument, fused, ppl_only, baseline, recovered_ckpt, graph, splits, crossover, versweep, phase_adaptive, dense_thresh)
    elif mode == "recovered":
        call = serve_recovered.spawn(RECOVERED_CKPT, util, fused, instrument)  # --instrument -> diag
    elif mode == "gusparse":
        call = serve_gu_sparse.spawn(util, do_ppl)
    elif mode == "densify":
        call = serve_densify.spawn(policy, recovered_ckpt, util, do_ppl, speed)
    elif mode == "bench":
        call = bench_mlp.spawn(util, instrument)  # reuse --instrument flag as the verify gate
    else:
        call = {"store_so": store_so, "smoke": smoke}.get(mode, smoke).spawn()
    print(f"SPAWN_ID {call.object_id}", flush=True)
