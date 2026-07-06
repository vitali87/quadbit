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
            torch.cuda.synchronize()  # ctypes kernels are on the default stream; vLLM forward is on another
            lib.quantize_act_nvfp4_2lvl(x2.data_ptr(), Bb.data_ptr(), sB.data_ptr(), gB.data_ptr(), tp, self.in_f)
            C = torch.empty((self.out_f, tp), dtype=torch.bfloat16, device=dev)
            lib.sparse_fp4_mm_2lvl(self.Ac.data_ptr(), Bb.data_ptr(), self.scaleA.data_ptr(),
                                   sB.data_ptr(), self.meta.data_ptr(), C.data_ptr(),
                                   self.out_f, tp, self.in_f, self.gA.data_ptr(), gB.data_ptr())
            torch.cuda.synchronize()
            # Re-materialize the result through a plain torch.empty+copy_ (NOT .clone()). The kernel-derived
            # tensor (view into the ctypes-written C buffer) has storage/provenance that vLLM's later residual
            # /logprob ops mishandle -> garbage, even after .clone(). Proven: return kernel .clone() -> PPL
            # 11594; copy same values into a fresh torch.empty -> 8.93. Allocate the output the way vLLM does.
            res = C.t()[:t].reshape(*lead, self.out_f)
            out = torch.empty(res.shape, dtype=x.dtype, device=dev)
            out.copy_(res)
            return out

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
            torch.cuda.synchronize()  # ctypes kernels are on the default stream; vLLM forward is on another
            lib.quantize_act_nvfp4_2lvl(x2.data_ptr(), Bb.data_ptr(), sB.data_ptr(), gB.data_ptr(), tp, self.in_f)
            C = torch.empty((self.out_f, tp), dtype=torch.bfloat16, device=dev)
            lib.sparse_fp4_mm_2lvl(self.Ac.data_ptr(), Bb.data_ptr(), self.scaleA.data_ptr(),
                                   sB.data_ptr(), self.meta.data_ptr(), C.data_ptr(),
                                   self.out_f, tp, self.in_f, self.gA.data_ptr(), gB.data_ptr())
            torch.cuda.synchronize()
            # Re-materialize the result through a plain torch.empty+copy_ (NOT .clone()). The kernel-derived
            # tensor (view into the ctypes-written C buffer) has storage/provenance that vLLM's later residual
            # /logprob ops mishandle -> garbage, even after .clone(). Proven: return kernel .clone() -> PPL
            # 11594; copy same values into a fresh torch.empty -> 8.93. Allocate the output the way vLLM does.
            res = C.t()[:t].reshape(*lead, self.out_f)
            out = torch.empty(res.shape, dtype=x.dtype, device=dev)
            out.copy_(res)
            return out

    return QBSparse


def _fused_mlp_fwd(mlp, torch, lib, dev):
    """Assign mlp.forward to the FUSED sparse MLP: gate_up sparse GEMM -> C[28672,tp] (no transpose)
    -> fused_swiglu_quant (silu(g)*u + NVFP4 quant, one pass) -> down sparse GEMM. Correct (cos 0.997
    vs two-level; do NOT reuse the model's SiluAndMul on sparse bf16) and fast (~1.25x vs native NVFP4
    MLP at chunk M). Uses mlp._qb_gu / mlp._qb_dn (QBSparse). down activation is single-level (gB=1)."""
    gu, dn = mlp._qb_gu, mlp._qb_dn
    H, Iw = gu.in_f, dn.in_f  # 4096, 14336

    def fwd(x):
        lead = x.shape[:-1]; x2 = x.reshape(-1, H).to(torch.bfloat16)
        t = x2.shape[0]; pad = (-t) % 128
        if pad:
            x2 = torch.cat([x2, x2.new_zeros(pad, H)], 0)
        x2 = x2.contiguous(); tp = t + pad
        Bb = torch.empty((tp, H // 2), dtype=torch.uint8, device=dev)
        sBg = torch.empty((gu.ks, tp, 4), dtype=torch.uint8, device=dev)
        gBg = torch.empty((tp,), dtype=torch.float32, device=dev)
        Cgu = torch.empty((gu.out_f, tp), dtype=torch.bfloat16, device=dev)
        Hb = torch.empty((tp, Iw // 2), dtype=torch.uint8, device=dev)
        sH = torch.empty((Iw // 128, tp, 4), dtype=torch.uint8, device=dev)
        gB1 = torch.ones((tp,), dtype=torch.float32, device=dev)
        Cout = torch.empty((dn.out_f, tp), dtype=torch.bfloat16, device=dev)
        torch.cuda.synchronize()  # torch writes (x2, gB1) done before default-stream kernels read them
        lib.quantize_act_nvfp4_2lvl(x2.data_ptr(), Bb.data_ptr(), sBg.data_ptr(), gBg.data_ptr(), tp, H)
        lib.sparse_fp4_mm_2lvl(gu.Ac.data_ptr(), Bb.data_ptr(), gu.scaleA.data_ptr(), sBg.data_ptr(),
                               gu.meta.data_ptr(), Cgu.data_ptr(), gu.out_f, tp, H,
                               gu.gA.data_ptr(), gBg.data_ptr())
        lib.fused_swiglu_quant(Cgu.data_ptr(), Cgu.data_ptr() + Iw * tp * 2, Hb.data_ptr(), sH.data_ptr(), tp, Iw)
        lib.sparse_fp4_mm_2lvl(dn.Ac.data_ptr(), Hb.data_ptr(), dn.scaleA.data_ptr(), sH.data_ptr(),
                               dn.meta.data_ptr(), Cout.data_ptr(), dn.out_f, tp, Iw,
                               dn.gA.data_ptr(), gB1.data_ptr())
        torch.cuda.synchronize()  # all kernels done before torch reads Cout
        # re-materialize via torch.empty+copy_ (see QBSparse.forward): kernel-derived tensor storage breaks
        # vLLM's downstream ops; a plain torch allocation does not.
        res = Cout.t()[:t].reshape(*lead, dn.out_f)
        out = torch.empty(res.shape, dtype=x.dtype, device=dev)
        out.copy_(res)
        return out
    mlp.forward = fwd


NVFP4_CKPT = "nvidia/Llama-3.1-8B-Instruct-NVFP4"
BF16_CKPT = "meta-llama/Llama-3.1-8B-Instruct"
RECOVERED_CKPT = "/cache/recovered_Meta-Llama-3-8B_P30000_p25000_2sh_lr3e-05.pt"


@app.function(gpu="RTX-PRO-6000", timeout=3600, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def serve_hybrid(do_ppl: bool = True, util: float = 0.8, instrument: bool = False,
                 fused: bool = True, ppl_only: bool = False, baseline: bool = False) -> None:
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
    if not baseline:
        from transformers import AutoModelForCausalLM
        src = AutoModelForCausalLM.from_pretrained(BF16_CKPT, dtype=torch.bfloat16, low_cpu_mem_usage=True)
        for layer in src.model.layers:
            gu = torch.cat([layer.mlp.gate_proj.weight.data, layer.mlp.up_proj.weight.data], 0).clone()
            mlpw.append((gu, layer.mlp.down_proj.weight.data.clone()))
        del src; gc.collect()
        print(f"side-loaded bf16 MLP weights for {len(mlpw)} layers", flush=True)

    # 2) vLLM native NVFP4 (non-MLP linears 4-bit). Lower util to leave room for sparse MLP buffers.
    S, GEN = 2048, 128
    llm = LLM(model=NVFP4_CKPT, enforce_eager=True, max_model_len=S + GEN + 16,
              kv_cache_dtype="auto", gpu_memory_utilization=util)
    print(f"NON_MLP_QUANT = {llm.llm_engine.model_config.quantization}", flush=True)  # expect modelopt_fp4
    mem_load = gpu_mib()

    lib = ctypes.CDLL(SO_PATH)
    lib.sparse_fp4_mm_2lvl.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    lib.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    lib.fused_swiglu_quant.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
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
            _fused_mlp_fwd(mlp, torch, lib, dev); return
        gu, dn, Iw = mlp._qb_gu, mlp._qb_dn, mlp._qb_dn.in_f

        def fwd_unf(x):  # plain SwiGLU (NOT the model's SiluAndMul -> that was the cos~0 bug), two-level
            y = gu.forward(x)
            return dn.forward(torch.nn.functional.silu(y[..., :Iw]) * y[..., Iw:])
        mlp.forward = fwd_unf

    for li, layer in enumerate(model.model.layers if not baseline else []):
        gu, dn = mlpw[li]
        mlp = layer.mlp
        mlp._qb_gu = QBSparse(gu.to(dev)); mlp._qb_dn = QBSparse(dn.to(dev))
        mlpw[li] = None  # free CPU copy as we go
        patch(mlp)
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
        print(f"PPL_THROUGH_SERVING {math.exp(nll / n):.4f} (magnitude pair-2:4 MLP, no recovery; "
              f"non-MLP NVFP4; {'fused single-level' if fused else 'unfused two-level'}; {len(wins)}x{S})", flush=True)
    if ppl_only:
        print("PPL_ONLY_DONE", flush=True); return

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

    import torch
    from vllm import LLM, SamplingParams

    S = 2048
    llm = LLM(model=BASE, enforce_eager=True, max_model_len=S + 16, dtype="bfloat16",
              gpu_memory_utilization=util)
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    lib = ctypes.CDLL(SO_PATH)
    lib.sparse_fp4_mm_2lvl.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    lib.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    lib.fused_swiglu_quant.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
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
        mlp._qb_gu = QBSparse(torch.cat([g, u], 0).to(dev), keep_wdq=True)
        mlp._qb_dn = QBSparse(d.to(dev), keep_wdq=True)
        # CONTROL: recovered weights as DENSE bf16 (no sparse/quant) via my monkeypatch -> isolates whether
        # the patch MECHANISM/full-forward is broken (dense should reproduce the masked-recovered model ~8-9).
        gg = torch.cat([g, u], 0).to(dev).bfloat16(); dd = d.to(dev).bfloat16()
        mlp.forward = (lambda guw, dw, iw: (lambda x: torch.nn.functional.linear(
            torch.nn.functional.silu(torch.nn.functional.linear(x, guw)[..., :iw])
            * torch.nn.functional.linear(x, guw)[..., iw:], dw)))(gg, dd, d.shape[1])
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

    ppl("recovered-dense-bf16-via-patch")  # if this is garbage, the patch mechanism/full-forward is the bug

    # K-SWEEP: patch down-sparse for the FIRST K layers only, dense-Wdq for the rest. Reveals whether the
    # 121k is accumulation (PPL grows smoothly with K) or divergence/instability (explodes past some K),
    # or per-layer catastrophic (already bad at K=1, which would contradict the cos-0.995 probe).
    def dense_fwd(mlp):
        gw, dw, iw = mlp._qb_gu.Wdq, mlp._qb_dn.Wdq, mlp._qb_dn.in_f
        return lambda x: torch.nn.functional.linear(
            torch.nn.functional.silu((x.float() @ gw.t())[..., :iw]) * (x.float() @ gw.t())[..., iw:],
            dw).to(x.dtype)

    def sparse_fwd(mlp):  # unfused TWO-LEVEL (QBSparse.forward for gate_up and down; plain SwiGLU)
        g_, d_, iw = mlp._qb_gu, mlp._qb_dn, mlp._qb_dn.in_f

        def fwd(x):
            y = g_.forward(x)
            return d_.forward(torch.nn.functional.silu(y[..., :iw]) * y[..., iw:])
        return fwd

    # DEFINITIVE: capture the REAL layer-0 mlp input x0 in-run, then compare sparse_fwd(x0) vs dense_fwd(x0)
    # on the EXACT same input (removes all "which h" ambiguity). Low cos => the in-run sparse MLP output is
    # genuinely wrong for the real input (vs faithful on captured/dense h) -> the real x0 has a property the
    # probes missed. Also dumps x0 dtype/shape/contiguity/max (vLLM may hand a surprising layout).
    m0 = model.model.layers[0].mlp
    capx = {}
    hk0 = m0.register_forward_pre_hook(lambda m, a: capx.setdefault("x", a[0].detach().clone()))
    llm.generate([{"prompt_token_ids": wins[0]}], SamplingParams(temperature=0, max_tokens=1), use_tqdm=False)
    hk0.remove()
    x0 = capx["x"]
    so = sparse_fwd(m0)(x0).float(); do = dense_fwd(m0)(x0).float()
    rc = torch.nn.functional.cosine_similarity(so, do, dim=-1).reshape(-1)
    print(f"DEFINITIVE x0 shape {tuple(x0.shape)} dtype {x0.dtype} contig {x0.is_contiguous()} max {x0.abs().max().item():.2f} | "
          f"sparse-vs-dense cos {torch.nn.functional.cosine_similarity(so.flatten(), do.flatten(), dim=0).item():.5f} "
          f"per-row mean {rc.mean().item():.4f} min {rc.min().item():.3f} sparse-max {so.abs().max().item():.1f} "
          f"dense-max {do.abs().max().item():.1f} sparse-NaN {torch.isnan(so).any().item()}", flush=True)
    del x0, so, do, rc, capx; torch.cuda.empty_cache()

    for K in (1, 2, 4, 32):
        for li, layer in enumerate(model.model.layers):
            mlp = layer.mlp
            if li < K:
                if fused:
                    _fused_mlp_fwd(mlp, torch, lib, dev)
                else:
                    mlp.forward = sparse_fwd(mlp)
            else:
                mlp.forward = dense_fwd(mlp)  # dense Wdq
        torch.cuda.empty_cache()
        ppl(f"MLP-sparse[{'fused' if fused else '2lvl'}] first K={K:2d} (rest dense-Wdq)")


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
    lib.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    lib.fused_swiglu_quant.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
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


@app.local_entrypoint()
def main(mode: str = "smoke", sparse: bool = True, thresh: int = 256, do_ppl: bool = True,
         util: float = 0.8, instrument: bool = False, fused: bool = True, ppl_only: bool = False,
         baseline: bool = False) -> None:
    if mode == "serve":
        call = serve.spawn(sparse, thresh)
    elif mode == "hybrid":
        call = serve_hybrid.spawn(do_ppl, util, instrument, fused, ppl_only, baseline)
    elif mode == "recovered":
        call = serve_recovered.spawn(RECOVERED_CKPT, util, fused, instrument)  # --instrument -> diag
    elif mode == "gusparse":
        call = serve_gu_sparse.spawn(util, do_ppl)
    elif mode == "bench":
        call = bench_mlp.spawn(util, instrument)  # reuse --instrument flag as the verify gate
    else:
        call = {"store_so": store_so, "smoke": smoke}.get(mode, smoke).spawn()
    print(f"SPAWN_ID {call.object_id}", flush=True)
