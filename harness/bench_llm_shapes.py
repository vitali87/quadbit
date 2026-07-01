"""Dense + 2:4-sparse FP4 vs cuBLAS bf16 at REAL Llama-3-8B GEMM shapes (not square).
Square M=N=K is a benchmark artifact; inference runs rectangular shapes, and the biggest
FP4 win is memory-bound DECODE (small M = few tokens): 4-bit weights stream ~4x less DRAM
than bf16, so decode should show ~4x regardless of compute. Prefill (large M) is
compute-bound and tracks the square story. hidden=4096, FFN intermediate=14336.

Run:  uv run modal run harness/bench_llm_shapes.py
"""

import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64"})
    .pip_install("torch", index_url="https://download.pytorch.org/whl/nightly/cu128", pre=True)
    .add_local_dir((ROOT / "cuda").as_posix(), "/root/cuda")
)
app = modal.App("quadbit-llm-bench", image=image)

# (label, M, N, K). Llama-3-8B: hidden 4096, FFN 14336. M = tokens in the batch.
SHAPES = [
    ("prefill attn  qkv/o", 4096, 4096, 4096),
    ("prefill ffn   up/gate", 4096, 14336, 4096),
    ("prefill ffn   down", 4096, 4096, 14336),
    ("decode  attn  qkv/o", 128, 4096, 4096),
    ("decode  ffn   up/gate", 128, 14336, 4096),
    ("decode  ffn   down", 128, 4096, 14336),
]


@app.function(gpu="RTX-PRO-6000", timeout=1800)
def run() -> None:
    import ctypes

    import torch

    outs = {}
    for name, src in (("sp", "sparse_fp4_lib"), ("de", "dense_fp4_lib"), ("sk", "dense_sk_lib"),
                      ("dc", "dense_decode_lib"), ("ss", "sparse_sk_lib")):
        so = f"/root/{name}.so"
        c = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
                            "-o", so, f"/root/cuda/{src}.cu", "-lcuda"], capture_output=True, text=True)
        if c.returncode != 0:
            print(c.stderr, flush=True); return
        outs[name] = ctypes.CDLL(so)
    sp, de, sk, dc = outs["sp"], outs["de"], outs["sk"], outs["dc"]
    sp.sparse_fp4_mm.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3
    de.dense_fp4_mm.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3
    sk.dense_fp4_mm_sk.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4
    dc.dense_fp4_decode.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3
    dc.qb_encode_map.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 3
    dc.qb_encode_map.restype = ctypes.c_void_p
    dc.qb_free_map.argtypes = [ctypes.c_void_p]
    dc.dense_fp4_decode_cached.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3
    dc.dense_fp4_decode_cached_async.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3
    dc.dense_fp4_decode_cached_async.restype = None
    dc.qb_decode_tn.argtypes = [ctypes.c_int]
    dc.qb_decode_tn.restype = ctypes.c_int
    ss = outs["ss"]
    ss.sparse_fp4_mm_sk.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 4

    print(f"{torch.cuda.get_device_name(0)}\n", flush=True)

    def tms(fn, it=50):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(it):
            fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / it

    hdr = (f"{'shape':>22} | {'M/N/K':>16} | {'bf16':>9} | {'plain':>8} | "
           f"{'split-K':>8} | {'decode':>8} | {'dec-cache':>9} | {'BEST dense':>18} | {'2:4 FP4':>15}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    for label, M, N, K in SHAPES:
        flop = 2.0 * M * N * K
        Ab = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        Bb = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        ms_bf16 = tms(lambda: torch.matmul(Ab, Bb.t()))

        Ad = torch.full((M, K // 2), 0x22, dtype=torch.uint8, device="cuda")
        Bd = torch.full((N, K // 2), 0x22, dtype=torch.uint8, device="cuda")
        Cd = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
        ms_de = tms(lambda: de.dense_fp4_mm(Ad.data_ptr(), Bd.data_ptr(), Cd.data_ptr(), M, N, K))

        # split-K: fill idle SMs when the data-parallel grid (N/256 x M/128) is small.
        # sweep splits (bounded by ksteps=K/128) and keep the fastest.
        Cf = torch.empty((M, N), dtype=torch.float32, device="cuda")
        ntiles = (N // 256) * (M // 128)
        ksteps = K // 128
        cands = sorted({1, 2, 3, 4, 6, 8, max(1, min(ksteps, -(-188 // max(1, ntiles))))})
        best_sk, best_sp = 1e9, 1
        for spl in cands:
            if spl > ksteps:
                continue
            t = tms(lambda: sk.dense_fp4_mm_sk(Ad.data_ptr(), Bd.data_ptr(), Cd.data_ptr(),
                                               Cf.data_ptr(), M, N, K, spl))
            if t < best_sk:
                best_sk, best_sp = t, spl

        # decode kernel (split-N direct bf16): best for small-M low-K
        ms_dc = tms(lambda: dc.dense_fp4_decode(Ad.data_ptr(), Bd.data_ptr(), Cd.data_ptr(), M, N, K))
        # cached-map decode: build both TMA maps ONCE (deployment: weight+reused-act buffer), time launch only
        mapA_h = dc.qb_encode_map(Ad.data_ptr(), M, K, 128)
        mapB_h = dc.qb_encode_map(Bd.data_ptr(), N, K, dc.qb_decode_tn(N))
        # async (no per-call sync) matches how torch.matmul is timed -> fair kernel throughput
        ms_dcc = tms(lambda: dc.dense_fp4_decode_cached_async(mapA_h, mapB_h, Cd.data_ptr(), M, N, K))
        dc.qb_free_map(mapA_h); dc.qb_free_map(mapB_h)
        # best FP4 dense strategy for this shape
        best_dense = min(ms_de, best_sk, ms_dc, ms_dcc)

        # sparse needs M % 256 (bm256 tiling); skip small-M decode rows
        sp_str = "  n/a (M<256)"
        if M % 256 == 0:
            ks = K // 128
            Ac = torch.full((M, K // 4), 0x22, dtype=torch.uint8, device="cuda")
            Bs = torch.full((N, K // 2), 0x22, dtype=torch.uint8, device="cuda")
            sA = torch.full((ks, M, 4), 0x38, dtype=torch.uint8, device="cuda")
            sB = torch.full((ks, N, 4), 0x38, dtype=torch.uint8, device="cuda")
            mt = torch.full((ks, M, 2), 0x44444444, dtype=torch.int32, device="cuda")
            Cs = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
            ms_sp = tms(lambda: sp.sparse_fp4_mm(Ac.data_ptr(), Bs.data_ptr(), sA.data_ptr(),
                                                 sB.data_ptr(), mt.data_ptr(), Cs.data_ptr(), M, N, K))
            sp_str = f"{ms_sp*1e3:6.1f}us {ms_bf16/ms_sp:4.2f}x"

        # SPARSE weight-stationary decode: orient C[out,tok]=W[out,in]@X[tok,in]^T so the 2:4
        # weight is the compressed mma-A (M=out, large -> fills SMs), tok=128 is the thin N.
        sp_dec = ""
        if N % 256 == 0:
            Mo, No = N, M  # out (large) x tok
            ks = K // 128
            Acw = torch.full((Mo, K // 4), 0x22, dtype=torch.uint8, device="cuda")
            Bxw = torch.full((No, K // 2), 0x22, dtype=torch.uint8, device="cuda")
            sAw = torch.full((ks, Mo, 4), 0x38, dtype=torch.uint8, device="cuda")
            sBw = torch.full((ks, No, 4), 0x38, dtype=torch.uint8, device="cuda")
            mtw = torch.full((ks, Mo, 2), 0x44444444, dtype=torch.int32, device="cuda")
            Csw = torch.empty((Mo, No), dtype=torch.bfloat16, device="cuda")
            Cfw = torch.empty((Mo, No), dtype=torch.float32, device="cuda")
            ktot = K // 256  # chunks (128*WK)
            best_ss, best_ssp = 1e9, 1
            for spl in sorted({1, 2, 3, 4, 6}):
                if spl > ktot:
                    continue
                try:
                    t = tms(lambda: ss.sparse_fp4_mm_sk(Acw.data_ptr(), Bxw.data_ptr(), sAw.data_ptr(),
                                                        sBw.data_ptr(), mtw.data_ptr(), Csw.data_ptr(),
                                                        Cfw.data_ptr(), Mo, No, K, spl))
                    if t < best_ss:
                        best_ss, best_ssp = t, spl
                except Exception as ex:
                    sp_dec = f"  spDEC err {ex}"; break
            if best_ss < 1e9:
                sp_dec = f"  spDEC {ms_bf16/best_ss:4.2f}x s={best_ssp}"

        def cell(ms):
            return f"{ms_bf16/ms:5.2f}x"
        best_kind = {ms_de: "plain", best_sk: f"splitK{best_sp}", ms_dc: "decode",
                     ms_dcc: "dec-cache"}[best_dense]
        print(f"{label:>22} | {f'{M}/{N}/{K}':>16} | {ms_bf16*1e3:6.1f}us | "
              f"{cell(ms_de):>8} | {cell(best_sk):>8} | {cell(ms_dc):>8} | {cell(ms_dcc):>9} | "
              f"{ms_bf16/best_dense:5.2f}x {best_kind:>9} | {sp_str:>15}{sp_dec}", flush=True)


@app.local_entrypoint()
def main() -> None:
    run.remote()
