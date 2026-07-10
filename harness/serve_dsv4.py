"""M3.6 substrate + M4 baseline: serve DeepSeek-V4-Flash NVFP4 in vLLM on RTX-PRO-6000 (sm_120).

DeepSeek-V4-Flash NVFP4 (~142GB) exceeds one 102GB GPU, so >=2 GPUs are required just to LOAD it
(expert/tensor parallel). This harness first proves the native stack loads + generates + graph-captures
(the M4 dense-NVFP4 baseline row and the M3.6 integration substrate). The quadbit sparse MoE monkeypatch
(M3.6) is layered on top once this passes.

Reports: load success, backend/quant method actually selected (fallback detection), a sanity generation,
prefill/decode timing, per-GPU memory, graph-capture (enforce_eager=False) status.
"""

import time
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
MODEL = "nvidia/DeepSeek-V4-Flash-NVFP4"
MIN = 60

# Let vLLM resolve its own torch + flashinfer (pinning them conflicts). vLLM >=0.20 ships the SM120
# Blackwell FP4 kernels and the deepseek_v4 model. CUDA 13 base for the runtime libs it expects.
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.12")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64", "HF_HOME": "/cache",
          "HF_XET_HIGH_PERFORMANCE": "1"})
    .pip_install("vllm", "huggingface_hub", "datasets")
    # vLLM 0.24.0's deepseek_v4 sparse-MLA path calls flashinfer's newer
    # trtllm_batch_decode_sparse_mla_dsv4(swa_topk_lens=..., extra_sparse_indices=...) API, but
    # 0.24.0 exact-pins flashinfer 0.6.12 (older sparse_topk_lens/seq_lens sig -> TypeError). Force
    # 0.6.14 with --no-deps so the resolver can't backtrack vLLM to 0.11.0 (which lacks deepseek_v4).
    # The swa_topk_lens/extra_sparse_* API vLLM 0.24.0 calls exists ONLY in flashinfer-python 0.6.14,
    # but flashinfer-cubin stops at 0.6.13. Use python 0.6.14 (for the API) + cubin 0.6.13 (latest
    # precompiled kernels) and bypass the python/cubin version check -- the sparse-MLA kernel JITs at
    # runtime regardless. --no-deps keeps vLLM 0.24.0 pinned (a plain pin backtracks it to 0.11.0).
    .run_commands("pip install --force-reinstall --no-deps "
                  "flashinfer-python==0.6.14 flashinfer-cubin==0.6.13")
    .env({"FLASHINFER_DISABLE_VERSION_CHECK": "1"})
    # SM120 dense/attention unblock plugin: registered as a vllm.general_plugins entry point so the
    # monkeypatch runs in every spawned worker (an imperative patch in the driver does not survive).
    .add_local_dir(str(ROOT / "harness" / "qb_vllm_plugin"), "/opt/qb_plugin", copy=True)
    # force_build so a plugin edit always redeploys (Modal's layer cache has served stale bytecode
    # here despite --force-reinstall); ~10s cost, buys reliable deploys.
    .run_commands("pip install --force-reinstall --no-deps /opt/qb_plugin", force_build=True)
)
app = modal.App("quadbit-serve", image=image)
vol = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)


@app.function(gpu="RTX-PRO-6000:2", timeout=60 * MIN, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def unblock(tp: int = 2, eager: bool = True, max_len: int = 2048, dense: str = "bf16",
            kv: str = "fp8", moe: str = "off") -> None:
    """WS0/WS1 end-to-end unblock: SM120-safe dense/attention (dense='bf16' or 'nvfp4') + native
    NVFP4 MoE. The replacement is installed by the qb_sm120 vLLM plugin (survives worker spawn); we
    only select it via QB_DENSE here. eager=True first (correctness), then flip for graph-capture."""
    import os

    import torch
    from vllm import LLM, SamplingParams

    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    os.environ["QB_DENSE"] = dense  # read by the qb_sm120 plugin in every spawned worker
    os.environ["QB_MOE"] = moe  # off|dense|sparse: quadbit sparse-expert injection selector
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # quiet load bars so worker tracebacks survive
    print(f"# WS0/1 unblock: dense={dense} tp={tp} eager={eager} on "
          f"{torch.cuda.device_count()}x RTX-PRO-6000", flush=True)

    rope = {"rope_type": "yarn", "factor": 16, "original_max_position_embeddings": 65536,
            "beta_fast": 32, "beta_slow": 1}
    # bf16 fallback keeps fp8 originals (MLA absorption) + bf16 copies -> dense weights ~3x; push
    # gpu_mem_util high and keep batched tokens modest so KV cache memory stays positive.
    kw = dict(model=MODEL, tensor_parallel_size=tp, enforce_eager=eager, trust_remote_code=True,
              max_model_len=max_len, gpu_memory_utilization=0.95, kv_cache_dtype=kv,
              max_num_batched_tokens=max_len, hf_overrides={"rope_scaling": rope})
    t0 = time.time()
    try:
        llm = LLM(tokenizer_mode="deepseek_v4", **kw)
    except Exception as ex:  # noqa: BLE001
        # only fall back when the tokenizer_mode itself is the problem; never mask a forward/init error
        if "tokenizer_mode" not in str(ex) and "deepseek_v4" not in str(ex).lower():
            raise
        print(f"  (deepseek_v4 tokenizer_mode rejected: {type(ex).__name__}; retrying default)", flush=True)
        llm = LLM(**kw)
    print(f"  load+init forward ok in {time.time() - t0:.0f}s (the SM120 wall is cleared)", flush=True)

    prompts = ["The capital of France is", "def fibonacci(n):", "The three primary colors are"]
    sp = SamplingParams(temperature=0.0, max_tokens=32)
    t1 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t1
    for o in outs:
        print(f"  [gen] {o.prompt!r} -> {o.outputs[0].text!r}", flush=True)
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"  generated {ntok} tok in {dt:.2f}s ({ntok / dt:.1f} tok/s)", flush=True)

    import subprocess
    mem = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    print(f"  per-GPU mem used (MB): {mem.split(chr(10))}", flush=True)
    # audit the actual dense path taken (nvfp4 vs per-layer bf16 fallback) for honest labeling
    try:
        import qb_sm120_plugin as _p

        print(f"  qb STATS (driver view): {_p.STATS}", flush=True)
    except Exception:  # noqa: BLE001
        pass
    ok = ntok > 0 and all(o.outputs[0].text.strip() for o in outs)
    label = {"bf16": "bf16 dense fallback", "nvfp4": "NVFP4 dense (mm_fp4 cutlass)",
             "off": "native SM120 path"}.get(dense, dense)
    moe_label = {"off": "native NVFP4 MoE", "dense": "NVFP4->bf16 dequant experts (dense)",
                 "sparse": "quadbit 2:4 sparse-FP4 experts"}.get(moe, moe)
    print(f"# unblock {'PASS' if ok else 'FAIL'} dense={dense} moe={moe} "
          f"({label} + {moe_label} + bf16 DSA indexer, "
          f"graph={'eager' if eager else 'captured'})", flush=True)


@app.function(gpu="RTX-PRO-6000:2", timeout=60 * MIN, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def baseline(tp: int = 2, eager: bool = False, max_len: int = 4096) -> None:
    import torch
    from vllm import LLM, SamplingParams

    import os
    # DeepGEMM's ue8m0 scale-factor transform asserts on SM120 ("Unknown SF transformation"); disable it
    # so vLLM routes FP8/scaled GEMMs to the FlashInfer/CUTLASS path that SM120 supports.
    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    print(f"# M4 baseline: {MODEL} tp={tp} eager={eager} on {torch.cuda.device_count()}x RTX-PRO-6000", flush=True)
    # DeepSeek-V4 config uses rope_scaling {type: yarn}; newer configs want rope_type -> patch via hf_overrides.
    rope = {"rope_type": "yarn", "factor": 16, "original_max_position_embeddings": 65536,
            "beta_fast": 32, "beta_slow": 1}
    t0 = time.time()
    kw = dict(model=MODEL, tensor_parallel_size=tp, enforce_eager=eager, trust_remote_code=True,
              max_model_len=max_len, gpu_memory_utilization=0.9, kv_cache_dtype="fp8",
              hf_overrides={"rope_scaling": rope})
    try:
        llm = LLM(tokenizer_mode="deepseek_v4", **kw)
    except Exception as ex:  # noqa: BLE001 -- deepseek_v4 tokenizer mode may not exist in this build
        print(f"  (deepseek_v4 tokenizer_mode rejected: {type(ex).__name__}; retrying default) ", flush=True)
        llm = LLM(**kw)
    print(f"  load ok in {time.time() - t0:.0f}s", flush=True)

    # which MoE / quant method actually got selected (fallback detection)
    try:
        eng = llm.llm_engine.model_executor.driver_worker.model_runner.model
        seen = set()
        for n, m in eng.named_modules():
            qm = type(getattr(m, "quant_method", None)).__name__
            if "moe" in type(m).__name__.lower() or "expert" in n.lower():
                seen.add(f"{type(m).__name__}:{qm}")
        print(f"  MoE modules/methods: {sorted(seen)[:8]}", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"  (module introspection failed: {type(ex).__name__}: {ex})", flush=True)

    prompts = ["The capital of France is", "def fibonacci(n):"]
    sp = SamplingParams(temperature=0.0, max_tokens=32)
    t1 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t1
    for o in outs:
        print(f"  [gen] {o.prompt!r} -> {o.outputs[0].text!r}", flush=True)
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"  generated {ntok} tok in {dt:.2f}s ({ntok / dt:.1f} tok/s)", flush=True)

    import subprocess
    mem = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    print(f"  per-GPU mem used (MB): {mem.split(chr(10))}", flush=True)
    print(f"# baseline {'PASS' if ntok > 0 else 'FAIL'} (graph_capture={'eager-OFF' if not eager else 'eager'})", flush=True)


@app.function(gpu="RTX-PRO-6000:2", timeout=60 * MIN, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def inspect_moe(tp: int = 2, max_len: int = 2048) -> None:
    # M3.6 recon: dump the exact FusedMoE structure so the sparse injection targets real attrs (model
    # is volume-cached after the first baseline load -> this loads fast).
    import inspect
    import os

    import torch
    from vllm import LLM

    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    os.environ.setdefault("QB_DENSE", "bf16")  # qb_sm120 plugin (bf16 dense) makes SM120 init work
    rope = {"rope_type": "yarn", "factor": 16, "original_max_position_embeddings": 65536,
            "beta_fast": 32, "beta_slow": 1}
    llm = LLM(model=MODEL, tensor_parallel_size=tp, enforce_eager=True, trust_remote_code=True,
              max_model_len=max_len, gpu_memory_utilization=0.95, kv_cache_dtype="fp8",
              max_num_batched_tokens=max_len, hf_overrides={"rope_scaling": rope},
              tokenizer_mode="deepseek_v4")
    m = llm.llm_engine.model_executor.driver_worker.model_runner.model
    shown = 0
    for name, mod in m.named_modules():
        cls = type(mod).__name__
        if "fusedmoe" in cls.lower() or "experts" in cls.lower() or name.endswith("experts"):
            if shown >= 2:
                break
            shown += 1
            print(f"\n=== {name}  ({cls}) ===", flush=True)
            qm = getattr(mod, "quant_method", None)
            print(f"  quant_method: {type(qm).__name__}", flush=True)
            for an in dir(mod):
                if an.startswith("_"):
                    continue
                v = getattr(mod, an, None)
                if isinstance(v, torch.Tensor):
                    print(f"  tensor {an}: shape={tuple(v.shape)} dtype={v.dtype}", flush=True)
                elif isinstance(v, (int, bool)) and an in (
                        "top_k", "num_experts", "global_num_experts", "local_num_experts",
                        "intermediate_size_per_partition", "hidden_size", "renormalize",
                        "use_grouped_topk", "num_expert_group", "topk_group"):
                    print(f"  attr {an} = {v}", flush=True)
            if qm is not None and hasattr(qm, "apply"):
                try:
                    print(f"  apply{inspect.signature(qm.apply)}", flush=True)
                except (ValueError, TypeError):
                    pass
    print("\n# inspect_moe done", flush=True)


@app.function(gpu="RTX-PRO-6000", timeout=20 * MIN, volumes={"/cache": vol})
def test_so() -> None:
    # Step A de-risk: does the staged sm_120 quadbit kernel .so (built under CUDA 12.8) ctypes-load
    # and RUN on the CUDA-13 serve image / sm_120? Port moe_layer.py's pack/quant_act/seg_gemm and
    # run one synthetic segmented MoE matmul vs a bf16 reference. Expect finite output + the ~0.88
    # 2:4-FP4 accuracy tax (NOT ~1.0 -- sparse is lossy by design). Crash/garbage => must rebuild .so.
    import ctypes

    import torch
    import torch.nn.functional as F

    bn = 128
    dev = torch.device("cuda")
    print(f"# test_so: torch {torch.__version__} cuda {torch.version.cuda} "
          f"cap {torch.cuda.get_device_capability()}", flush=True)
    so = None
    for cand in ("/cache/sparse_fp4_sm120.so", "/cache/sparse_fp4.so"):
        try:
            lib = ctypes.CDLL(cand)
            so = cand
            break
        except Exception as ex:  # noqa: BLE001
            print(f"  CDLL {cand} failed: {type(ex).__name__}: {ex}", flush=True)
    if so is None:
        print("# test_so FAIL: no loadable .so", flush=True)
        return
    print(f"  loaded {so}", flush=True)
    lib.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    lib.sparse_moe_mm_2lvl.argtypes = ([ctypes.c_void_p] * 6 + [ctypes.c_int] * 4
                                       + [ctypes.c_void_p] * 3 + [ctypes.c_int] + [ctypes.c_void_p])
    lib.qb_init_moe_attrs()
    torch.manual_seed(0)

    fp4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
    bnd = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], device=dev)
    cc = torch.arange(128, device=dev)
    e_, m_ = (cc >> 3) & 0xf, cc & 7
    ue4m3 = torch.where(e_ == 0, m_.float() * 0.001953125,
                        (1.0 + m_.float() / 8.0) * torch.exp2((e_ - 7).float()))

    def q_fp4(v):
        return torch.bucketize(v.abs(), bnd) | ((v < 0).long() << 3)

    def enc(s):
        mant_f, e = torch.frexp(s.clamp_min(1e-30))
        mm = 2.0 * mant_f
        biased = (e - 1) + 7
        mant = torch.round((mm - 1.0) * 8.0).long()
        carry = mant == 8
        mant = torch.where(carry, torch.zeros_like(mant), mant)
        biased = torch.where(carry, biased + 1, biased)
        code = (biased.long() << 3) | mant
        code = torch.where(biased < 1, torch.ones_like(code), code)
        code = torch.where(biased > 15, torch.full_like(code, 0x7f), code)
        code = torch.where(s >= 480.0, torch.full_like(code, 0x7f), code)
        return torch.where(s > 0, code, torch.zeros_like(code))

    def pack(w):
        out_f, in_f = w.shape
        ks = in_f // 128
        wg = w.float().to(dev).view(out_f, ks, 16, 4, 2)
        i01, _ = wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
        kept = torch.gather(wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
        ga = (kept.abs().amax(dim=(1, 2, 3, 4), keepdim=True) / 2688.0).clamp_min(1e-30).reshape(out_f, 1, 1)
        blk = kept.reshape(out_f, ks, 4, 8, 2)
        scode = enc((blk.abs().amax(dim=(3, 4)) / 6.0) / ga)
        sdeq = ue4m3[scode] * ga
        kc = q_fp4(blk / sdeq.clamp_min(1e-30)[..., None, None])
        ac = (kc[..., 0] | (kc[..., 1] << 4)).reshape(out_f, ks * 32).to(torch.uint8)
        nib = (i01[..., 0] | (i01[..., 1] << 2)).view(out_f, ks, 2, 8)
        sh = (torch.arange(8, device=dev) * 4).view(1, 1, 1, 8)
        meta = (nib << sh).sum(-1).to(torch.int32).permute(1, 0, 2).contiguous()
        return (ac.contiguous(), meta, scode.to(torch.uint8).permute(1, 0, 2).contiguous(),
                ga.reshape(out_f).float().contiguous())

    def quant_act(x):
        r, in_f = x.shape
        ks = in_f // 128
        x = x.to(torch.bfloat16).contiguous()
        bb = torch.empty((r, in_f // 2), dtype=torch.uint8, device=dev)
        sb = torch.empty((ks, r, 4), dtype=torch.uint8, device=dev)
        gb = torch.empty((r,), dtype=torch.float32, device=dev)
        lib.quantize_act_nvfp4_2lvl(x.data_ptr(), bb.data_ptr(), sb.data_ptr(), gb.data_ptr(), r, in_f)
        return bb, sb, gb

    # single-expert segmented call (eblk all-zero -> one expert, Mpe=M) as the minimal kernel exercise
    m_out, k_in, rows = 512, 1024, 256
    w = (torch.randn(m_out, k_in, device=dev) * (k_in ** -0.5)).to(torch.bfloat16)
    x = (torch.randn(rows, k_in, device=dev) * (k_in ** -0.5) * 4).to(torch.bfloat16)
    ac, meta, scale_a, ga = pack(w)
    bb, sb, gb = quant_act(x)
    c = torch.empty((rows, m_out), dtype=torch.bfloat16, device=dev)
    eblk = torch.zeros(rows // bn, dtype=torch.int32, device=dev)
    lib.sparse_moe_mm_2lvl(ac.data_ptr(), bb.data_ptr(), scale_a.data_ptr(), sb.data_ptr(),
                           meta.data_ptr(), c.data_ptr(), ac.shape[0], m_out, rows, k_in,
                           ga.data_ptr(), gb.data_ptr(), eblk.data_ptr(), 1, 0)
    torch.cuda.synchronize()
    ref = F.linear(x.float(), w.float())
    nonfin = int((~torch.isfinite(c)).sum().item())
    cos = F.cosine_similarity(c.float().flatten(), ref.flatten(), dim=0).item()
    print(f"  seg_gemm ran: out={tuple(c.shape)} nonfin={nonfin} cos(seg,dense-bf16)={cos:.4f}", flush=True)
    ok = nonfin == 0 and cos > 0.8
    print(f"# test_so {'PASS' if ok else 'FAIL'} (staged sm_120 .so runs on CUDA-13 serve image)", flush=True)


@app.function(gpu="RTX-PRO-6000:2", timeout=60 * MIN, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def quadbit(tp: int = 2, eager: bool = False, max_len: int = 2048) -> None:
    # M3.6: inject the segmented sparse-FP4 MoE op into vLLM's FusedMoE layers. Loads the staged .so
    # (built under CUDA 12.8 by build_so.py -> /cache/sparse_fp4.so) since the sparse mma won't assemble
    # under the serve image's CUDA 13 ptxas. Structure-specific wiring (expert-weight dequant + attr
    # names) is finalized against inspect_moe's dump; this run reports SPARSE_EXPERT_CALLS + fallbacks.
    import ctypes
    import os

    import torch
    from vllm import LLM, SamplingParams

    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    so = "/cache/sparse_fp4.so"
    lib = ctypes.CDLL(so)
    lib.sparse_moe_mm_2lvl.argtypes = ([ctypes.c_void_p] * 6 + [ctypes.c_int] * 4 +
                                       [ctypes.c_void_p] * 3 + [ctypes.c_int] + [ctypes.c_void_p])
    lib.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    lib.qb_init_moe_attrs()
    print(f"# M3.6 quadbit sparse-MoE injection: loaded {so}", flush=True)

    rope = {"rope_type": "yarn", "factor": 16, "original_max_position_embeddings": 65536,
            "beta_fast": 32, "beta_slow": 1}
    llm = LLM(model=MODEL, tensor_parallel_size=tp, enforce_eager=eager, trust_remote_code=True,
              max_model_len=max_len, gpu_memory_utilization=0.9, kv_cache_dtype="fp8",
              hf_overrides={"rope_scaling": rope}, tokenizer_mode="deepseek_v4")
    m = llm.llm_engine.model_executor.driver_worker.model_runner.model

    # locate FusedMoE layers (finalized against inspect_moe output)
    moe = [(n, mod) for n, mod in m.named_modules()
           if "fusedmoe" in type(mod).__name__.lower() and hasattr(mod, "quant_method")]
    print(f"  found {len(moe)} FusedMoE layers", flush=True)
    # The sparse injection (dequant experts NVFP4->bf16, 2:4 pack, replace quant_method.apply with a
    # sparse_moe_mm_2lvl path) is NOT yet wired: it needs the expert weight/scale attr names from
    # `inspect_moe`, and on SM120 this model's FP8 attention GEMM has no kernel so LLM() cannot even
    # finish init (see docs/paper.md Section 10). We deliberately do NOT run llm.generate() here -- that
    # would exercise the NATIVE vLLM MoE path and could be mistaken for a sparse-served result. This
    # entry point is ready to complete on a host whose FP8-attention backend supports SM120 (or Hopper).
    _ = SamplingParams  # referenced once wiring lands
    raise NotImplementedError(
        "quadbit sparse-MoE injection not wired: replace each FusedMoE.quant_method.apply with the "
        "segmented sparse_moe_mm_2lvl path using per-expert weights from `inspect` mode. This does NOT "
        "serve sparse yet; running native generate here would misrepresent the result.")


@app.function(image=image)
def versions() -> None:
    # CPU-only: pin down the vLLM<->FlashInfer skew behind the swa_topk_lens TypeError.
    import inspect
    from importlib.metadata import version

    for pkg in ["vllm", "flashinfer-python", "flashinfer", "torch"]:
        try:
            print(f"{pkg} == {version(pkg)}", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"{pkg}: {type(ex).__name__}", flush=True)
    try:
        import flashinfer

        print(f"flashinfer.__version__ = {getattr(flashinfer, '__version__', '?')}", flush=True)
        fn = getattr(flashinfer, "trtllm_batch_decode_sparse_mla_dsv4", None)
        if fn is None:
            import flashinfer.decode as fd

            fn = getattr(fd, "trtllm_batch_decode_sparse_mla_dsv4", None)
        print(f"trtllm_batch_decode_sparse_mla_dsv4 sig: "
              f"{inspect.signature(fn) if fn else 'NOT FOUND'}", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"flashinfer introspection failed: {type(ex).__name__}: {ex}", flush=True)


@app.function(image=image)
def dumpsrc() -> None:
    # CPU-only: read the exact source of the deepseek_v4 o_proj DeepGEMM path so the SM120-safe
    # replacement is faithful (deep_gemm_fp8_o_proj bypasses Fp8LinearMethod -> our linear patch
    # never sees it; the DeepGEMM fp8_einsum asserts t.dim()==N on sm_120).
    import vllm

    base = Path(vllm.__file__).parent
    def show_full(label, p):
        print(f"\n===== {label}  ({p}) =====", flush=True)
        if not p.exists():
            print(f"  MISSING {p}", flush=True)
            return
        text = p.read_text().splitlines()
        print("\n".join(f"{i + 1:4} {ln}" for i, ln in enumerate(text)), flush=True)

    def show_defs(label, p, keys):
        print(f"\n===== {label}  ({p}) =====", flush=True)
        if not p.exists():
            print(f"  MISSING {p}", flush=True)
            return
        text = p.read_text().splitlines()
        for key in keys:
            hits = [i for i, ln in enumerate(text) if key in ln]
            for h in hits[:4]:
                lo, hi = max(0, h - 1), min(len(text), h + 45)
                print(f"  --- '{key}' @ line {h + 1} ---", flush=True)
                print("\n".join(f"{i + 1:4} {text[i]}" for i in range(lo, hi)), flush=True)

    show_defs("deep_gemm.py mqa-logits",
              base / "utils/deep_gemm.py",
              ["def get_paged_mqa_logits_metadata", "def fp8_paged_mqa_logits",
               "def _lazy_init", "_get_paged_mqa_logits_metadata_impl =",
               "_fp8_paged_mqa_logits_impl =", "def _missing"])
    show_defs("indexer.py backend",
              base / "v1/attention/backends/mla/indexer.py",
              ["get_paged_mqa_logits_metadata", "fp8_paged_mqa_logits", "def build",
               "scheduler_metadata_buffer"])
    show_defs("flashinfer_sparse_mla_warmup.py",
              base / "model_executor/warmup/flashinfer_sparse_mla_warmup.py",
              ["def deepseek_v4_sparse_mla_attention_warmup"])

    import subprocess

    def show_range(label, p, lo, hi):
        print(f"\n===== {label} ({p}) [{lo}-{hi}] =====", flush=True)
        text = p.read_text().splitlines()
        print("\n".join(f"{i + 1:4} {text[i]}" for i in range(lo - 1, min(hi, len(text)))), flush=True)

    def show_range2(label, p, lo, hi):
        print(f"\n===== {label} [{lo}-{hi}] =====", flush=True)
        text = p.read_text().splitlines()
        print("\n".join(f"{i + 1:4} {text[i]}" for i in range(lo - 1, min(hi, len(text)))), flush=True)

    import subprocess as _sp

    # SharedExperts.forward wants an `order` positional arg. Dump its signature + how the
    # native modelopt apply / model calls it so our patched_moe_apply passes `order` correctly.
    dv = base / "models/deepseek_v4/nvidia/model.py"
    dtext = dv.read_text().splitlines()
    for key in ["class SharedExperts", "def _forward_fused_moe", "shared_experts(",
                "SharedExperts(", "order"]:
        hits = [i for i, ln in enumerate(dtext) if key in ln]
        for h in hits[:6]:
            lo, hi = max(0, h - 1), min(len(dtext), h + 20)
            print(f"\n--- model.py '{key}' @ {h + 1} ---", flush=True)
            print("\n".join(f"{i + 1:4} {dtext[i]}" for i in range(lo, hi)), flush=True)
    # find the SharedExperts class def + its forward(order) signature, tree-wide
    r = _sp.run(["grep", "-rn", r"class SharedExperts", str(base)], capture_output=True, text=True)
    print(f"\n### class SharedExperts defs ###\n{r.stdout}", flush=True)
    for ln in r.stdout.strip().splitlines():
        fp = ln.split(":")[0]
        if not fp.endswith(".py"):
            continue
        stext = Path(fp).read_text().splitlines()
        cs = [i for i, l in enumerate(stext) if "class SharedExperts" in l]
        for c in cs:
            print(f"\n--- {fp} SharedExperts [{c + 1}] ---", flush=True)
            print("\n".join(f"{i + 1:4} {stext[i]}" for i in range(c, min(len(stext), c + 55))), flush=True)
    # full shared_experts.py (forward + retrieval) and moe_runner apply/_maybe_apply 500-585
    se = base / "model_executor/layers/fused_moe/runner/shared_experts.py"
    stext = se.read_text().splitlines()
    print(f"\n### shared_experts.py [90-{len(stext)}] ###", flush=True)
    print("\n".join(f"{i + 1:4} {stext[i]}" for i in range(89, len(stext))), flush=True)
    mr = base / "model_executor/layers/fused_moe/runner/moe_runner.py"
    mt = mr.read_text().splitlines()
    print(f"\n### moe_runner.py [520-590] ###", flush=True)
    print("\n".join(f"{i + 1:4} {mt[i]}" for i in range(519, min(len(mt), 590))), flush=True)
    return  # SharedExperts recon only this run

    show_range2("flashinfer_sparse.py _forward_prefill call", base
                / "models/deepseek_v4/nvidia/flashinfer_sparse.py", 820, 895)
    show_defs("vllm flashinfer.py wrapper",
              base / "utils/flashinfer.py",
              ["trtllm_batch_decode_sparse_mla_dsv4", "def has_flashinfer_sparse_mla_sm120"])
    import subprocess

    r = subprocess.run(["pip", "index", "versions", "flashinfer-python"],
                       capture_output=True, text=True)
    print(f"\n### pip index versions flashinfer-python:\n{r.stdout}\n{r.stderr}", flush=True)
    r2 = subprocess.run(["grep", "-rn", "flashinfer", str(base.parent / "vllm-0.24.0.dist-info")
                        if (base.parent / "vllm-0.24.0.dist-info").exists() else str(base)],
                       capture_output=True, text=True)
    print(f"\n### vllm flashinfer pin refs:\n{r2.stdout[:1500]}", flush=True)


@app.function(image=image)
def probe() -> None:
    # CPU-only: confirm which plugin code is actually baked into the image (guards against pip
    # skipping a same-version reinstall). Prints the installed version + the dequant source.
    import inspect
    from importlib.metadata import entry_points, version

    import torch

    import qb_sm120_plugin as p

    print(f"# qb-sm120-plugin version={version('qb-sm120-plugin')} file={p.__file__}", flush=True)
    eps = [e.value for e in entry_points(group="vllm.general_plugins")]
    print(f"# vllm.general_plugins entry points: {eps}", flush=True)
    print(inspect.getsource(p._dequant_block), flush=True)
    # call the ACTUAL installed function on the real failing shape (w[16384,1024], s[128,8])
    for sshape in [(128, 8), (8, 128), (12, 32)]:
        wh = (16384, 1024) if sshape != (12, 32) else (1536, 4096)
        w = torch.randint(-7, 8, wh, dtype=torch.float32)
        s = torch.rand(*sshape) + 0.1
        try:
            out = p._dequant_block(w, s, (128, 128))
            print(f"  CALL p._dequant_block w={wh} s={sshape} -> {tuple(out.shape)} OK", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"  CALL p._dequant_block w={wh} s={sshape} -> {type(ex).__name__}: {ex}", flush=True)


def _sweep_impl(tp: int, runtag: str, matrix: list, instr: bool, tax: bool,
                max_len: int) -> None:
    """WS3: quadbit 2:4 sparse-FP4 expert performance sweep. Loads DeepSeek-V4-Flash-NVFP4 once with
    QB_MOE=sparse on `tp` GPUs, then times a prompt x gen x batch matrix, emitting figure-ready CSV
    rows + a compute/comm breakdown (from the plugin's per-worker instrumentation flushed to /cache)
    + the in-situ per-layer tax probe. Eager only (graph capture unproven on this SM120 path)."""
    import glob
    import json
    import os
    import statistics as st
    import subprocess
    import time

    import torch
    from vllm import LLM, SamplingParams

    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    os.environ["QB_DENSE"] = "bf16"
    os.environ["QB_MOE"] = "sparse"
    os.environ["QB_RUNTAG"] = runtag
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    if instr:
        os.environ["QB_INSTR"] = "1"
    if tax:
        os.environ["QB_TAXPROBE"] = "1"
    # clear stale per-worker metric files for this runtag
    for f in glob.glob(f"/cache/qb_metrics_{runtag}_dev*.json"):
        os.remove(f)

    ngpu = torch.cuda.device_count()
    print(f"# WS3 sweep tp={tp} runtag={runtag} instr={instr} tax={tax} on {ngpu}x RTX-PRO-6000; "
          f"commit=$(git) matrix={matrix}", flush=True)

    def smi():
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        used, free = [], []
        for line in out.strip().splitlines():
            u, fr = line.split(",")
            used.append(int(u))
            free.append(int(fr))
        return used, free

    rope = {"rope_type": "yarn", "factor": 16, "original_max_position_embeddings": 65536,
            "beta_fast": 32, "beta_slow": 1}
    kw = dict(model=MODEL, tensor_parallel_size=tp, enforce_eager=True, trust_remote_code=True,
              max_model_len=max_len, gpu_memory_utilization=0.95, kv_cache_dtype="fp8",
              max_num_batched_tokens=max(2048, max_len), hf_overrides={"rope_scaling": rope},
              disable_log_stats=False)
    t0 = time.time()
    llm = LLM(tokenizer_mode="deepseek_v4", **kw)
    load_s = time.time() - t0
    load_used, load_free = smi()
    print(f"  loaded in {load_s:.0f}s; per-GPU used MB={load_used} free MB={load_free}", flush=True)

    tok = llm.get_tokenizer()
    base = tok.encode("The quick brown fox jumps over the lazy dog and then keeps running. ")

    def make_prompt(n):
        ids = (base * (n // len(base) + 1))[:n]
        return {"prompt_token_ids": ids}

    # single global warmup so per-config timings exclude first-call JIT/autotune
    llm.generate([make_prompt(128)], SamplingParams(temperature=0.0, max_tokens=8, min_tokens=8))

    rows = []
    peak_used = list(load_used)
    def med(x):
        return st.median(x) if x else 0.0

    for (P, G, B) in matrix:
        prompts = [make_prompt(P) for _ in range(B)]
        sp = SamplingParams(temperature=0.0, max_tokens=G, min_tokens=G, ignore_eos=True)
        torch.cuda.synchronize()
        t1 = time.time()
        try:
            outs = llm.generate(prompts, sp)
        except Exception as e:
            # engine death (e.g. long-prefill scheduler KeyError) must not lose completed rows
            print(f"  [cfg] P={P} G={G} B={B}: FAILED {type(e).__name__}: {str(e)[:160]}", flush=True)
            break
        wall = time.time() - t1
        u, _fr = smi()
        peak_used = [max(a, b) for a, b in zip(peak_used, u)]
        # decode time from the monotonic (*_ts) pair only; arrival_time is unix-epoch (different clock),
        # so TTFT is derived from wall - decode instead of mixing clocks.
        decs, tpots = [], []
        gen_tok = 0
        for o in outs:
            gen_tok += len(o.outputs[0].token_ids)
            m = o.metrics
            if m is None:
                continue
            ft = getattr(m, "first_token_time", None) or getattr(m, "first_token_ts", None)
            lt = getattr(m, "last_token_time", None) or getattr(m, "last_token_ts", None)
            fin = getattr(m, "finished_time", None) or getattr(m, "finished_ts", None) or lt
            if fin and ft and G > 1:
                d = fin - ft
                decs.append(d)
                tpots.append(d / (G - 1))

        dec_wall = med(decs)
        ttft = max(0.0, wall - dec_wall)  # B=1: exact; B>1: proxy (requests overlap)
        tpot = med(tpots)
        prefill_tps = (B * P) / ttft if ttft > 0 else 0.0
        dec_tps = (B * (G - 1)) / dec_wall if dec_wall > 0 else 0.0
        tot_tps = gen_tok / wall if wall > 0 else 0.0
        row = {"tp": tp, "prompt": P, "gen": G, "batch": B, "wall_s": round(wall, 3),
               "ttft_s": round(ttft, 4), "tpot_ms": round(tpot * 1000, 3),
               "prefill_tps": round(prefill_tps, 1), "decode_tps": round(dec_tps, 1),
               "total_tps": round(tot_tps, 1), "req_latency_s": round(wall, 3),
               "gen_tok": gen_tok}
        rows.append(row)
        print(f"  [cfg] P={P} G={G} B={B}: wall={wall:.1f}s ttft={ttft:.3f}s "
              f"tpot={tpot * 1000:.1f}ms prefill={prefill_tps:.1f}tok/s decode={dec_tps:.1f}tok/s "
              f"total={tot_tps:.1f}tok/s", flush=True)

    print(f"  peak per-GPU used MB={peak_used}", flush=True)

    # aggregate per-worker instrumentation + tax from /cache
    metric_files = sorted(glob.glob(f"/cache/qb_metrics_{runtag}_dev*.json"))
    agg = {"expert_ms": 0.0, "dense_ms": 0.0, "expert_calls": 0, "sparse_expert_calls": 0,
           "imbalance_mean": [], "imbalance_max": [], "tax_cos": []}
    for mf in metric_files:
        with open(mf) as f:
            d = json.load(f)
        agg["expert_ms"] += d["t_ms"].get("expert", 0.0)
        agg["dense_ms"] += d["t_ms"].get("dense", 0.0)
        agg["expert_calls"] += d["counts"].get("expert_calls", 0)
        agg["sparse_expert_calls"] += d["stats"].get("sparse_expert_calls", 0)
        agg["imbalance_mean"].append(d.get("imbalance_mean", 0.0))
        agg["imbalance_max"].append(d.get("imbalance_max", 0.0))
        agg["tax_cos"].extend(d.get("tax_cos", []))
    print(f"\n# --- instrumentation (ranks={len(metric_files)}) ---", flush=True)
    print(f"  expert-kernel ms(sum over ranks)={agg['expert_ms']:.1f} "
          f"dense/attn ms={agg['dense_ms']:.1f} expert_calls={agg['expert_calls']} "
          f"SPARSE_EXPERT_CALLS={agg['sparse_expert_calls']}", flush=True)
    if agg["imbalance_mean"]:
        print(f"  expert imbalance (max/mean tokens-per-expert): "
              f"mean={st.mean(agg['imbalance_mean']):.3f} max={max(agg['imbalance_max']):.3f}",
              flush=True)
    cosv = sorted(c["cos"] for c in agg["tax_cos"])
    if cosv:
        def pct(p):
            return cosv[min(len(cosv) - 1, int(p * len(cosv)))]
        worst = min(agg["tax_cos"], key=lambda c: c["cos"])
        print(f"\n# --- in-situ per-layer tax cos(sparse,dense) (n={len(cosv)}) ---", flush=True)
        print(f"  median={st.median(cosv):.4f} p10={pct(0.1):.4f} p90={pct(0.9):.4f} "
              f"min={cosv[0]:.4f} max={cosv[-1]:.4f}", flush=True)
        print(f"  worst: layer={worst['layer']} expert={worst['expert']} cos={worst['cos']:.4f}",
              flush=True)

    # figure-ready CSV to /cache
    csv_path = f"/cache/qb_sweep_{runtag}.csv"
    cols = ["tp", "prompt", "gen", "batch", "wall_s", "ttft_s", "tpot_ms", "prefill_tps",
            "decode_tps", "total_tps", "req_latency_s", "gen_tok"]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\n# CSV rows ({csv_path}):", flush=True)
    print(",".join(cols), flush=True)
    for r in rows:
        print(",".join(str(r[c]) for c in cols), flush=True)
    print(f"# sweep DONE tp={tp} runtag={runtag} load={load_s:.0f}s "
          f"load_used_MB={load_used} peak_used_MB={peak_used}", flush=True)


# default matrix: representative, bounded for eager (~3 tok/s) within the Modal timeout.
_SWEEP_MATRIX = [(512, 32, 1), (512, 128, 1), (512, 32, 8), (512, 128, 8),
                 (2048, 32, 1), (2048, 128, 1), (2048, 32, 8),
                 (512, 512, 1), (8192, 32, 1)]


@app.function(gpu="RTX-PRO-6000:2", timeout=150 * MIN, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def sweep2(instr: bool = True, tax: bool = True, max_len: int = 8960) -> None:
    _sweep_impl(2, "tp2", _SWEEP_MATRIX, instr, tax, max_len)


@app.function(gpu="RTX-PRO-6000:4", timeout=150 * MIN, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def sweep4(instr: bool = True, tax: bool = True, max_len: int = 8960) -> None:
    _sweep_impl(4, "tp4", _SWEEP_MATRIX, instr, tax, max_len)


def _calib_impl(tp: int, tag: str, max_len: int, dump: bool = False,
                sparse_dump: bool = False, dense_layers: str = "", calib_file: str = "") -> None:
    """A2 Step-2 calibration: run DeepSeek-V4-Flash DENSE (coherent) over a text corpus and let the
    plugin accumulate per-expert per-projection column activation norms from REAL routed tokens,
    dumped per-rank to /cache/qb_calib_{tag}_dev*.pt for the Wanda 2:4 mask in a later sparse run.
    dump=True also captures per-sparse-layer MoE block I/O (x, routing, dense y) to
    /cache/qb_reconio_{tag}_dev*.pt for A3 layerwise repair (same teacher forward, no extra pass)."""
    import os
    import time

    import torch
    from vllm import LLM, SamplingParams

    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    os.environ["QB_DENSE"] = "nvfp4"
    if sparse_dump:
        # A3 reio2: run the exact A2-49 serving config (sparse MoE + dense anchors + Wanda mask) so
        # the dumped per-layer input x is the SERVE-CONSISTENT sparse trajectory, then dump it for
        # error-correcting recon. No QB_CALIB here (the mask already exists as calib_file).
        os.environ["QB_MOE"] = "sparse"
        os.environ["QB_CALIB_FILE"] = calib_file
        os.environ["QB_DENSE_LAYERS"] = dense_layers
        os.environ["QB_DUMP"] = "1"
    else:
        os.environ["QB_MOE"] = "dense"
        os.environ["QB_CALIB"] = "1"
        if dump:
            os.environ["QB_DUMP"] = "1"
    os.environ["QB_RUNTAG"] = tag
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    print(f"# calib tp={tp} tag={tag} on {torch.cuda.device_count()}x RTX-PRO-6000", flush=True)

    rope = {"rope_type": "yarn", "factor": 16, "original_max_position_embeddings": 65536,
            "beta_fast": 32, "beta_slow": 1}
    t0 = time.time()
    llm = LLM(model=MODEL, tokenizer_mode="deepseek_v4", tensor_parallel_size=tp,
              enforce_eager=True, trust_remote_code=True, max_model_len=max_len,
              gpu_memory_utilization=0.9, kv_cache_dtype="fp8",
              max_num_batched_tokens=max(2048, max_len), hf_overrides={"rope_scaling": rope})
    print(f"  loaded in {time.time() - t0:.0f}s", flush=True)

    tok = llm.get_tokenizer()
    corpus = [
        "The history of science spans many centuries and cultures, from ancient Babylonian astronomy to "
        "modern quantum mechanics and relativity. Researchers formulate hypotheses, design controlled "
        "experiments, gather data, and revise theories as new evidence emerges. The scientific method "
        "relies on reproducibility, falsifiability, and rigorous peer review by the wider community.",
        "In computer science, algorithms and data structures form the foundation of efficient software. "
        "A sorting algorithm arranges elements in order; a hash table offers average constant-time "
        "lookup; a balanced tree keeps operations logarithmic. def quicksort(a):\n    if len(a) <= 1:\n"
        "        return a\n    p = a[len(a)//2]\n    lo = [x for x in a if x < p]\n    hi = [x for x in a "
        "if x > p]\n    return quicksort(lo) + [x for x in a if x == p] + quicksort(hi)\n\nclass Stack:\n"
        "    def __init__(self):\n        self.items = []\n    def push(self, x):\n        self.items.append(x)",
        "Economics studies how societies allocate scarce resources among competing uses. Supply and "
        "demand jointly determine prices in competitive markets, while central banks influence interest "
        "rates, employment, and inflation through monetary policy. International trade allows nations to "
        "specialize according to comparative advantage, raising aggregate output but creating winners and losers.",
        "The novel opened on a grey morning in a small coastal town, where the fishermen mended their "
        "nets and the gulls circled overhead, crying into the wind. She walked along the weathered pier, "
        "thinking of the letter she had never sent, the words still forming and dissolving in her mind "
        "like the restless tide against the pilings below.",
        "Photosynthesis is the process by which green plants convert light energy into chemical energy. "
        "Chlorophyll in the chloroplasts absorbs sunlight, water is split into hydrogen and oxygen, and "
        "atmospheric carbon dioxide is fixed into glucose through the Calvin cycle. This process releases "
        "the oxygen we breathe and sustains nearly all life on Earth through the food chain.",
        "Mathematics is the study of numbers, structure, space, and change. A prime number has exactly "
        "two positive divisors, one and itself. The Pythagorean theorem states that for a right triangle "
        "the square of the hypotenuse equals the sum of the squares of the other two sides. Calculus "
        "formalizes rates of change through derivatives and accumulation through integrals.",
        "The human body is composed of many interacting systems. The cardiovascular system circulates "
        "blood, delivering oxygen and nutrients while removing waste. The nervous system transmits "
        "electrical signals between the brain and the rest of the body. The immune system defends "
        "against pathogens using white blood cells, antibodies, and inflammatory responses.",
        "Constitutional law governs the relationship between the state and its citizens. Courts interpret "
        "statutes and precedents, balancing individual rights against public interest. The principle of "
        "separation of powers divides authority among the legislative, executive, and judicial branches "
        "to prevent the concentration of power and protect against tyranny.",
        "To make a classic risotto, warm the stock in a saucepan and keep it simmering. In a separate "
        "pan, soften diced onion in butter, add the rice, and toast it briefly. Add a ladle of stock at "
        "a time, stirring constantly until absorbed, and continue until the grains are creamy yet still "
        "firm at the centre. Finish with parmesan and a knob of cold butter.",
        "The orchestra tuned their instruments as the conductor stepped onto the podium. The symphony "
        "began softly with the strings, then the woodwinds entered, and finally the brass and percussion "
        "joined in a triumphant crescendo. Music theory describes harmony, melody, rhythm, and the "
        "intervals that give a chord its characteristic tension or resolution.",
        "Climate scientists study the long-term patterns of temperature, precipitation, and atmospheric "
        "composition. Rising concentrations of greenhouse gases trap heat, warming the oceans and melting "
        "polar ice. Feedback loops, such as reduced reflectivity from vanishing sea ice, can amplify these "
        "changes and shift weather patterns across entire continents.",
        "Ancient Rome grew from a small city-state on the Tiber into an empire spanning three continents. "
        "Its legions, roads, aqueducts, and legal code shaped the Mediterranean world for centuries. "
        "Latin, the language of Rome, evolved into the Romance languages and contributed an enormous "
        "vocabulary to English, science, and law.",
    ]
    # Dense expert coverage needs many tokens per routed expert (256 experts, top-6): the 12
    # inline paragraphs (~840 tok) give ~3 tok/expert -> colnorm is noise. Pull real diverse text
    # so per-expert routing stats stabilise. wikitext (prose) + a code slice for domain spread.
    try:
        from datasets import load_dataset  # noqa: PLC0415

        ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=True)
        extra: list[str] = []
        for row in ds:
            txt = row["text"].strip()
            if len(txt) < 400:
                continue
            extra.append(txt[:1400])  # ~300 tokens/chunk
            if len(extra) >= 300:  # ~90k tokens -> ~2000 routed tok/expert avg
                break
        corpus = corpus + extra
        print(f"  loaded {len(extra)} pile chunks", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  dataset load failed ({e}); using inline corpus only", flush=True)

    ids = [{"prompt_token_ids": tok.encode(c)} for c in corpus]
    ntok = sum(len(d["prompt_token_ids"]) for d in ids)
    print(f"  calibrating over {len(corpus)} chunks / {ntok} tokens (dense forwards)...", flush=True)
    llm.generate(ids, SamplingParams(temperature=0.0, max_tokens=1))
    # Each WORKER accumulates + dumps its own shard (periodically during the forwards + atexit on
    # shutdown); the driver has no model state. Free the engine to trigger worker shutdown, then check.
    del llm
    import gc
    import glob

    gc.collect()
    time.sleep(5)
    if not sparse_dump:
        files = sorted(glob.glob(f"/cache/qb_calib_{tag}_dev*.pt"))
        sizes = [f"{os.path.basename(f)}={os.path.getsize(f) // 1024}KB" for f in files]
        print(f"# calib DONE tag={tag}: per-rank files {sizes or 'MISSING'}", flush=True)
    if dump or sparse_dump:
        io = sorted(glob.glob(f"/cache/qb_reconio_{tag}_dev*.pt"))
        iosz = [f"{os.path.basename(f)}={os.path.getsize(f) // (1024 * 1024)}MB" for f in io]
        print(f"# reconio DONE tag={tag}: per-rank files {iosz or 'MISSING'}", flush=True)


def _qmap_impl(tp: int, tag: str, dense_layers: str, max_len: int, moe: str = "dense",
               probe_layers: str = "0,20,40", calib_file: str = "") -> None:
    """WORKSTREAM A0+A1#1: real-routed-activation quality map + dense-anchor-layer policy.
    Loads DeepSeek-V4-Flash-NVFP4 with QB_MOE=sparse, keeps NVFP4 resident on probe layers (and on
    dense-anchor layers), runs coherence prompts, and records per-layer *block* cosine + per-expert
    cos / route freq / weight / contribution norm on REAL routed activations. Reports whether the
    dense-anchor policy restores coherence and at what sparse-layer coverage."""
    import glob
    import json
    import math
    import os
    import statistics as st
    import subprocess
    import time

    import torch
    from vllm import LLM, SamplingParams

    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    # nvfp4 dense/attention (WS1: coherent on SM120) is ~3x smaller than the bf16 fallback, which
    # otherwise leaves zero KV headroom once probe-layer sparse codes are also resident -> OOM.
    os.environ["QB_DENSE"] = "nvfp4"
    # map: run MoE dense (coherent) so probe layers see HEALTHY activations; anchor-coherence tests
    # pass moe="sparse" + dense_layers to measure real generation quality under selective sparsity.
    os.environ["QB_MOE"] = moe
    # probe_layers="none" -> pure coherence run (no map probe, no resident codes on sparse layers)
    os.environ["QB_QMAP"] = "0" if probe_layers == "none" else "1"
    os.environ["QB_QMAP_LAYERS"] = "" if probe_layers == "none" else probe_layers
    os.environ["QB_INSTR"] = "1"  # captures the fire-once NaN diagnostics from the guardrail
    os.environ["QB_CALIB_FILE"] = calib_file  # A2: tag of a calib run -> Wanda 2:4 masks in pack()
    os.environ["QB_RUNTAG"] = tag
    os.environ["QB_DENSE_LAYERS"] = dense_layers
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    for f in glob.glob(f"/cache/qb_metrics_{tag}_dev*.json"):
        os.remove(f)

    ngpu = torch.cuda.device_count()
    print(f"# qmap tp={tp} tag={tag} moe={moe} probe_layers=[{probe_layers}] "
          f"dense_layers=[{dense_layers}] on {ngpu}x RTX-PRO-6000", flush=True)

    def smi():
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        return [int(ln.split(",")[0]) for ln in out.strip().splitlines()]

    rope = {"rope_type": "yarn", "factor": 16, "original_max_position_embeddings": 65536,
            "beta_fast": 32, "beta_slow": 1}
    t0 = time.time()
    llm = LLM(model=MODEL, tokenizer_mode="deepseek_v4", tensor_parallel_size=tp,
              enforce_eager=True, trust_remote_code=True, max_model_len=max_len,
              gpu_memory_utilization=0.95, kv_cache_dtype="fp8",
              max_num_batched_tokens=max(2048, max_len), hf_overrides={"rope_scaling": rope},
              disable_log_stats=False)
    print(f"  loaded in {time.time() - t0:.0f}s; per-GPU used MB={smi()}", flush=True)

    prompts = ["The capital of France is", "def fibonacci(n):",
               "The three primary colors are", "Water is made of hydrogen and",
               "The opposite of hot is"]
    sp = SamplingParams(temperature=0.0, max_tokens=40, min_tokens=8)
    outs = llm.generate(prompts, sp)
    print("# --- COHERENCE (generation sanity) ---", flush=True)
    for p, o in zip(prompts, outs, strict=False):
        print(f"  [{p!r}] -> {o.outputs[0].text!r}", flush=True)

    # --- PPL: teacher-forced perplexity over a fixed held-out passage (quantifies the tax) ---
    passage = (
        "The mitochondria is the powerhouse of the cell. Photosynthesis converts sunlight, water, "
        "and carbon dioxide into glucose and oxygen. The Earth orbits the Sun once every year, and "
        "the Moon orbits the Earth roughly every twenty-eight days. Water boils at one hundred "
        "degrees Celsius at sea level and freezes at zero degrees. The human heart pumps blood "
        "through arteries and veins, delivering oxygen to every tissue in the body. Shakespeare "
        "wrote many famous plays, including Hamlet, Macbeth, and Romeo and Juliet. The speed of "
        "light in a vacuum is approximately three hundred thousand kilometres per second.")
    pids = llm.get_tokenizer().encode(passage)
    pout = llm.generate([{"prompt_token_ids": pids}],
                        SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0))
    plp = pout[0].prompt_logprobs or []
    nlls = []
    for tid, d in zip(pids[1:], plp[1:], strict=False):
        if d and tid in d and math.isfinite(d[tid].logprob):
            nlls.append(-d[tid].logprob)
    ppl = math.exp(sum(nlls) / len(nlls)) if nlls else float("nan")
    print(f"# PPL over {len(nlls)}-token held-out passage: {ppl:.3f} "
          f"(mean NLL {sum(nlls) / len(nlls):.4f} nats)" if nlls else "# PPL: no valid logprobs",
          flush=True)

    # aggregate the real-activation map + NaN diagnostics flushed by every worker
    rows, nan_diag = [], []
    for mf in sorted(glob.glob(f"/cache/qb_metrics_{tag}_dev*.json")):
        with open(mf) as f:
            d = json.load(f)
            rows += d.get("qmap", [])
            nan_diag += d.get("nan_diag", [])
    print("# --- NaN guardrail diagnostics (first nonfinite per layer/tensor) ---", flush=True)
    if nan_diag:
        for d in sorted(nan_diag, key=lambda r: (r["layer"], r["tensor"])):
            print(f"  layer {d['layer']:>3} {d['tensor']:<8} nonfinite={d['nonfinite']} "
                  f"finite_max_abs={d['finite_max_abs']}", flush=True)
    else:
        print("  (none — no nonfinite tensors intercepted in the sparse path)", flush=True)
    blk = {}   # layer -> [block_cos, ...]
    exp = {}   # layer -> [expert cos, ...]
    for r in rows:
        if r["expert"] == -1:
            if math.isfinite(r["block_cos"]):
                blk.setdefault(r["layer"], []).append(r["block_cos"])
        elif math.isfinite(r["cos"]):
            exp.setdefault(r["layer"], []).append(r["cos"])

    csv_path = f"/cache/qb_qmap_{tag}.csv"
    with open(csv_path, "w") as f:
        f.write("layer,expert,cos,freq,mean_w,contrib_norm,block_cos\n")
        for r in rows:
            if r["expert"] == -1:
                f.write(f"{r['layer']},-1,,,,,{r['block_cos']}\n")
            else:
                f.write(f"{r['layer']},{r['expert']},{r['cos']},{r['freq']},"
                        f"{r['mean_w']},{r['contrib_norm']},\n")

    print("# --- A0 real-activation quality map ---", flush=True)
    print("# layer  block_cos  expert_cos(median)  n_experts", flush=True)
    prod = 1.0
    all_exp = []
    for li in sorted(blk):
        bc = st.median(blk[li])
        ec = st.median(exp.get(li, [0.0]))
        all_exp += exp.get(li, [])
        prod *= bc
        print(f"  {li:>4}   {bc:.4f}     {ec:.4f}            {len(exp.get(li, []))}", flush=True)
    n_probe = len(blk)
    # extrapolate the probed block cosines over all sparse layers (probe covers every _TAX_LAYERS-th)
    dense_set = {int(x) for x in dense_layers.split(",") if x.strip()}
    n_sparse = 43 - len(dense_set)
    if n_probe:
        geo = math.exp(sum(math.log(max(1e-9, st.median(blk[li]))) for li in blk) / n_probe)
        pred_end = geo ** n_sparse
        print(f"# probed {n_probe} layers; geomean block_cos={geo:.4f}; "
              f"predicted end-of-stack coherence signal geo^{n_sparse}={pred_end:.3e}", flush=True)
    if all_exp:
        all_exp.sort()

        def q(p):
            return all_exp[min(len(all_exp) - 1, int(p * len(all_exp)))]
        print(f"# per-expert real-activation cos: median={st.median(all_exp):.4f} "
              f"p10={q(0.1):.4f} p90={q(0.9):.4f} min={all_exp[0]:.4f} max={all_exp[-1]:.4f}",
              flush=True)
    print(f"# dense-anchor layers={sorted(dense_set)} ({len(dense_set)}/43 dense, "
          f"{n_sparse}/43 sparse = {100*n_sparse/43:.0f}% layers sparse)", flush=True)
    print(f"# qmap CSV -> {csv_path} ({len(rows)} rows)", flush=True)


def _downstream_impl(tp: int, tag: str, moe: str, dense_layers: str, calib_file: str,
                     limit: int, max_len: int, sparse_proj: str = "both",
                     route_slot: int = 0) -> None:
    """WS-A downstream capability validation. Loglikelihood MC eval (lm-eval-compatible
    acc/acc_norm: per choice, sum teacher-forced logprob of the continuation tokens via vLLM
    prompt_logprobs; pick argmax) over ARC-C / HellaSwag / PIQA / Winogrande / MMLU-subset. Same
    bespoke DeepSeek-V4-Flash init as _qmap_impl so the quadbit sparse plugin + fp8-KV + rope
    overrides are identical. GSM8K skipped (generative, ~2 tok/s sparse decode = not cheap)."""
    import math
    import os
    import subprocess
    import time

    from vllm import LLM, SamplingParams

    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    os.environ["QB_DENSE"] = "nvfp4"
    os.environ["QB_MOE"] = moe
    os.environ["QB_QMAP"] = "0"
    os.environ["QB_INSTR"] = "1"
    os.environ["QB_CALIB_FILE"] = calib_file
    os.environ["QB_RUNTAG"] = tag
    os.environ["QB_DENSE_LAYERS"] = dense_layers
    os.environ["QB_SPARSE_PROJ"] = sparse_proj
    os.environ["QB_ROUTE_SLOT"] = str(route_slot)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    def smi():
        q = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        out = subprocess.run(q, capture_output=True, text=True).stdout
        return [int(ln) for ln in out.strip().splitlines()]

    rope = {"rope_type": "yarn", "factor": 16, "original_max_position_embeddings": 65536,
            "beta_fast": 32, "beta_slow": 1}
    t0 = time.time()
    llm = LLM(model=MODEL, tokenizer_mode="deepseek_v4", tensor_parallel_size=tp,
              enforce_eager=True, trust_remote_code=True, max_model_len=max_len,
              gpu_memory_utilization=0.95, kv_cache_dtype="fp8",
              max_num_batched_tokens=max(2048, max_len), hf_overrides={"rope_scaling": rope},
              disable_log_stats=True)
    tok = llm.get_tokenizer()
    mem = smi()
    print(f"# downstream tag={tag} moe={moe} dense_layers=[{dense_layers}] calib={calib_file} "
          f"limit={limit} loaded {time.time() - t0:.0f}s per-GPU-MB={mem}", flush=True)

    # generation sanity (same 5 prompts as qmap)
    gp = ["The capital of France is", "def fibonacci(n):", "The three primary colors are",
          "Water is made of hydrogen and", "The opposite of hot is"]
    gouts = llm.generate(gp, SamplingParams(temperature=0.0, max_tokens=24, min_tokens=6))
    print("# --- generation sanity ---", flush=True)
    for p, o in zip(gp, gouts, strict=False):
        print(f"  [{p!r}] -> {o.outputs[0].text!r}", flush=True)

    # PPL (self-report, must match the qmap Pareto number for this policy)
    passage = (
        "The mitochondria is the powerhouse of the cell. Photosynthesis converts sunlight, water, "
        "and carbon dioxide into glucose and oxygen. The Earth orbits the Sun once every year, and "
        "the Moon orbits the Earth roughly every twenty-eight days. Water boils at one hundred "
        "degrees Celsius at sea level and freezes at zero degrees. The human heart pumps blood "
        "through arteries and veins, delivering oxygen to every tissue in the body. Shakespeare "
        "wrote many famous plays, including Hamlet, Macbeth, and Romeo and Juliet. The speed of "
        "light in a vacuum is approximately three hundred thousand kilometres per second.")
    pids = tok.encode(passage)
    pout = llm.generate([{"prompt_token_ids": pids}],
                        SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0))
    plp = pout[0].prompt_logprobs or []
    nlls = [-plp[i][t].logprob for i, t in enumerate(pids)
            if i > 0 and plp[i] and t in plp[i] and math.isfinite(plp[i][t].logprob)]
    ppl = math.exp(sum(nlls) / len(nlls)) if nlls else float("nan")
    print(f"# PPL {ppl:.3f}", flush=True)

    # ---- loglikelihood MC scoring ----
    # request tuple: (item_key, choice_idx, prompt_token_ids, span_start, cont_char_len)
    def build(ctx: str, conts: list[str]):
        cids = tok.encode(ctx)
        out = []
        for c in conts:
            tail = tok.encode(c, add_special_tokens=False)
            out.append((cids + tail, len(cids), len(c)))
        return out

    def run_task(name: str, items: list):
        # items: list of (conts:list[str], gold:int, requests:list[(ids,start,clen)])
        reqs, meta = [], []
        for qi, (_, _, rq) in enumerate(items):
            for ci, (ids, start, clen) in enumerate(rq):
                reqs.append({"prompt_token_ids": ids})
                meta.append((qi, ci, start, clen))
        outs = llm.generate(reqs, SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0))
        scores: dict = {}
        for (qi, ci, start, clen), o in zip(meta, outs, strict=False):
            lp = o.prompt_logprobs or []
            # prompt_logprobs=0 -> each position dict holds only the ACTUAL token's logprob;
            # sum over the continuation span [start:] (position 0 is None/BOS, always < start).
            s = sum(next(iter(d.values())).logprob for i, d in enumerate(lp) if i >= start and d)
            scores.setdefault(qi, {})[ci] = (s, clen)
        acc = accn = 0
        for qi, (_, gold, _) in enumerate(items):
            sc = scores.get(qi, {})
            if not sc:
                continue
            best = max(sc, key=lambda c: sc[c][0])
            bestn = max(sc, key=lambda c: sc[c][0] / max(1, sc[c][1]))
            acc += int(best == gold)
            accn += int(bestn == gold)
        n = len(items)
        print(f"# TASK {name} n={n} acc={acc / n:.4f} acc_norm={accn / n:.4f}", flush=True)
        return {"task": name, "n": n, "acc": acc / n, "acc_norm": accn / n}

    from datasets import load_dataset  # noqa: PLC0415

    results = []

    def try_load(cands):
        for repo, kw in cands:
            try:
                return load_dataset(repo, **kw), repo
            except Exception as e:  # noqa: BLE001
                print(f"  load {repo} failed: {str(e)[:120]}", flush=True)
        return None, None

    # ARC-Challenge (acc_norm primary). template: "Question: {q}\nAnswer:" cont=" {choice}"
    ds, src = try_load([("allenai/ai2_arc", {"name": "ARC-Challenge", "split": "test"})])
    if ds is not None:
        items = []
        for r in list(ds)[:limit]:
            ch = r["choices"]["text"]
            labels = r["choices"]["label"]
            gold = labels.index(r["answerKey"]) if r["answerKey"] in labels else 0
            conts = [" " + t for t in ch]
            items.append((conts, gold, build(f"Question: {r['question']}\nAnswer:", conts)))
        results.append(run_task(f"arc_c[{src}]", items))

    # HellaSwag (acc_norm primary). lm-eval preprocess + ctx = activity + ctx_a/ctx_b
    ds, src = try_load([("Rowan/hellaswag", {"split": "validation"})])
    if ds is not None:
        import re

        def pp(t):
            t = t.strip().replace(" [title]", ". ")
            t = re.sub("\\[.*?\\]", "", t)
            return t.replace("  ", " ")
        items = []
        for r in list(ds)[:limit]:
            ctx = pp(r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize())
            conts = [" " + pp(e) for e in r["endings"]]
            gold = int(r["label"]) if r["label"] != "" else 0
            items.append((conts, gold, build(ctx, conts)))
        results.append(run_task(f"hellaswag[{src}]", items))

    # PIQA (acc_norm primary)
    ds, src = try_load([("ybisk/piqa", {"split": "validation", "trust_remote_code": True}),
                        ("piqa", {"split": "validation", "trust_remote_code": True})])
    if ds is not None:
        items = []
        for r in list(ds)[:limit]:
            conts = [" " + r["sol1"], " " + r["sol2"]]
            items.append((conts, int(r["label"]), build(f"Question: {r['goal']}\nAnswer:", conts)))
        results.append(run_task(f"piqa[{src}]", items))

    # Winogrande (acc). partial scoring: ctx=prefix+option, cont=suffix after blank
    ds, src = try_load([("allenai/winogrande", {"name": "winogrande_xl", "split": "validation",
                                                "trust_remote_code": True})])
    if ds is not None:
        items = []
        for r in list(ds)[:limit]:
            s = r["sentence"]
            idx = s.index("_")
            suffix = s[idx + 1:]
            reqs, conts = [], []
            for opt in (r["option1"], r["option2"]):
                cids = tok.encode(s[:idx] + opt)
                tail = tok.encode(suffix, add_special_tokens=False)
                reqs.append((cids + tail, len(cids), len(suffix)))
                conts.append(suffix)
            gold = int(r["answer"]) - 1
            items.append((conts, gold, reqs))
        results.append(run_task(f"winogrande[{src}]", items))

    # MMLU subset (acc), 0-shot
    subjects = ["abstract_algebra", "college_computer_science", "high_school_world_history",
                "professional_medicine", "moral_scenarios"]
    mmlu_scores = []
    for subj in subjects:
        ds, src = try_load([("cais/mmlu", {"name": subj, "split": "test"})])
        if ds is None:
            continue
        readable = subj.replace("_", " ")
        items = []
        for r in list(ds)[: max(50, limit // 4)]:
            q = r["question"]
            opts = r["choices"]
            head = (f"The following are multiple choice questions (with answers) about "
                    f"{readable}.\n\n{q}\n")
            for i, o in enumerate(opts):
                head += f"{chr(65 + i)}. {o}\n"
            head += "Answer:"
            conts = [f" {chr(65 + i)}" for i in range(len(opts))]
            items.append((conts, int(r["answer"]), build(head, conts)))
        rr = run_task(f"mmlu:{subj}", items)
        mmlu_scores.append(rr["acc"])
    if mmlu_scores:
        mmlu_avg = sum(mmlu_scores) / len(mmlu_scores)
        print(f"# TASK mmlu_subset n_subj={len(mmlu_scores)} acc={mmlu_avg:.4f}", flush=True)
        results.append({"task": "mmlu_subset", "n": len(mmlu_scores), "acc": mmlu_avg,
                        "acc_norm": mmlu_avg})

    # primary metric per task (acc_norm for arc/hellaswag/piqa, acc for winogrande/mmlu)
    def primary(r):
        return r["acc"] if r["task"].startswith(("winogrande", "mmlu")) else r["acc_norm"]
    prim = [primary(r) for r in results]
    avg = sum(prim) / len(prim) if prim else float("nan")
    print("# ================ DOWNSTREAM SUMMARY ================", flush=True)
    print(f"# tag={tag} PPL={ppl:.3f} per-GPU-MB={mem}", flush=True)
    for r in results:
        print(f"#   {r['task']:<28} acc={r['acc']:.4f} acc_norm={r['acc_norm']:.4f} "
              f"primary={primary(r):.4f} (n={r['n']})", flush=True)
    print(f"# AVG normalized primary = {avg:.4f}", flush=True)
    with open(f"/cache/qb_downstream_{tag}.csv", "w") as f:
        f.write("task,n,acc,acc_norm,primary\n")
        for r in results:
            f.write(f"{r['task']},{r['n']},{r['acc']:.4f},{r['acc_norm']:.4f},{primary(r):.4f}\n")
        f.write(f"AVG,,,,{avg:.4f}\n")
    print(f"# downstream CSV -> /cache/qb_downstream_{tag}.csv", flush=True)


@app.function(gpu="RTX-PRO-6000:2", timeout=120 * MIN, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def downstream(tag: str = "ds", moe: str = "dense", dense_layers: str = "", calib_file: str = "",
               limit: int = 400, max_len: int = 2048, sparse_proj: str = "both",
               route_slot: int = 0) -> None:
    _downstream_impl(2, tag, moe, dense_layers, calib_file, limit, max_len, sparse_proj, route_slot)


@app.function(gpu="RTX-PRO-6000:2", timeout=360 * MIN, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def recon(tag: str = "rc", dense_layers: str = "", calib_file: str = "cal4", recon_io: str = "",
          recon_layers: str = "", steps: int = 200, scale_only: bool = False, limit: int = 400,
          max_len: int = 2048) -> None:
    """A3 layerwise repair: QB_RECON trains the listed sparse MoE layers vs the dumped teacher I/O
    during load, packs the repaired weights, then runs downstream eval on the repaired model in the
    same job. Repaired weights persist to /cache/qb_reconw_{tag}_dev*.pt for later serving."""
    import os

    import moe_recon

    moe_recon._selfcheck()  # fail fast on a trainer logic bug before the 8-min model load
    os.environ["QB_RECON"] = "1"
    os.environ["QB_RECON_IO"] = recon_io
    os.environ["QB_RECON_LAYERS"] = recon_layers
    os.environ["QB_RECON_STEPS"] = str(steps)
    if scale_only:
        os.environ["QB_RECON_SCALE_ONLY"] = "1"
    _downstream_impl(2, tag, "sparse", dense_layers, calib_file, limit, max_len)


@app.function(gpu="RTX-PRO-6000:2", timeout=90 * MIN, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def qmap(tag: str = "qm", dense_layers: str = "", max_len: int = 2048, moe: str = "dense",
         probe_layers: str = "0,20,40", calib_file: str = "") -> None:
    _qmap_impl(2, tag, dense_layers, max_len, moe, probe_layers, calib_file)


@app.function(gpu="RTX-PRO-6000:2", timeout=90 * MIN, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def calib(tag: str = "cal1", max_len: int = 2048, dump: bool = False,
          sparse_dump: bool = False, dense_layers: str = "", calib_file: str = "") -> None:
    _calib_impl(2, tag, max_len, dump=dump, sparse_dump=sparse_dump,
                dense_layers=dense_layers, calib_file=calib_file)


@app.local_entrypoint()
def main(mode: str = "baseline", tp: int = 2, eager: bool = False, max_len: int = 4096,
         dense: str = "bf16", kv: str = "fp8", moe: str = "off",
         tag: str = "qm", dense_layers: str = "", probe_layers: str = "0,20,40",
         calib_file: str = "", limit: int = 400, recon_io: str = "", recon_layers: str = "",
         steps: int = 200, scale_only: bool = False, sparse_proj: str = "both",
         route_slot: int = 0) -> None:
    if mode == "test_so":
        test_so.remote()
    elif mode == "versions":
        versions.remote()
    elif mode == "dumpsrc":
        dumpsrc.remote()
    elif mode == "probe":
        probe.remote()
    elif mode == "inspect":
        inspect_moe.remote(tp=tp, max_len=max_len)
    elif mode == "quadbit":
        quadbit.remote(tp=tp, eager=eager, max_len=max_len)
    elif mode == "unblock":
        # .remote() (not .spawn()) so the FULL worker log streams to this client; launch under
        # `modal run --detach` so the job still survives client disconnect. spawn's detached logs
        # get truncated to a short tail once the app stops, hiding the worker root-cause traceback.
        unblock.remote(tp=tp, eager=eager, max_len=max_len, dense=dense, kv=kv, moe=moe)
    elif mode == "sweep2":
        sweep2.remote()
    elif mode == "sweep4":
        sweep4.remote()
    elif mode == "qmap":
        qmap.remote(tag=tag, dense_layers=dense_layers, max_len=max_len,
                    moe=(moe if moe != "off" else "dense"), probe_layers=probe_layers,
                    calib_file=calib_file)
    elif mode == "calib":
        calib.remote(tag=tag, max_len=max_len)
    elif mode == "dumpio":
        calib.remote(tag=tag, max_len=max_len, dump=True)
    elif mode == "dumpio2":
        # A3 reio2: sparse-trajectory (serve-consistent) I/O dump under the A2-49 policy.
        calib.remote(tag=tag, max_len=max_len, sparse_dump=True,
                     dense_layers=dense_layers, calib_file=(calib_file or "cal4"))
    elif mode == "downstream":
        downstream.remote(tag=tag, moe=(moe if moe != "off" else "dense"),
                          dense_layers=dense_layers, calib_file=calib_file, limit=limit,
                          max_len=max_len, sparse_proj=sparse_proj, route_slot=route_slot)
    elif mode == "recon":
        recon.remote(tag=tag, dense_layers=dense_layers, calib_file=(calib_file or "cal4"),
                     recon_io=recon_io, recon_layers=recon_layers, steps=steps,
                     scale_only=scale_only, limit=limit, max_len=max_len)
    else:
        baseline.remote(tp=tp, eager=eager, max_len=max_len)
