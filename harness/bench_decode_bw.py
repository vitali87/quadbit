"""Decode memory-bandwidth ceiling probe. For each LLM decode shape it reports the ACHIEVED
DRAM bandwidth (TB/s) vs the card's measured peak, so we can tell latency-bound shapes (far from
peak, headroom in occupancy) from truly bandwidth-bound ones (at peak, no kernel win left).

C[M,N] = A[M,K] @ B[N,K]^T. Decode = small M (tokens). DRAM-relevant traffic (weight read once +
output write): bytes = N*K/2 (FP4 weight) + M*N*2 (bf16 out). Activation A is tiny and L2-resident
(re-read by N-blocks -> L2 not DRAM), so it is reported separately. bf16 traffic = N*K*2 + M*N*2.

Run:  uv run modal run harness/bench_decode_bw.py
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
app = modal.App("quadbit-decode-bw", image=image)

# (label, M tokens, N out, K in). Real decoders FUSE Q+K+V (N=12288 for H=4096, or 6144 GQA) and
# gate+up; the isolated 4096x4096 is only o_proj (the cheapest op). Testing fused vs isolated.
SHAPES = [
    ("o_proj (isolated)  128x 4096x 4096", 128, 4096, 4096),
    ("fused QKV GQA      128x 6144x 4096", 128, 6144, 4096),
    ("fused QKV MHA      128x12288x 4096", 128, 12288, 4096),
    ("ffn  up            128x14336x 4096", 128, 14336, 4096),
    ("ffn  down          128x 4096x14336", 128, 4096, 14336),
    ("fused gate+up      128x28672x 4096", 128, 28672, 4096),
]


@app.function(gpu="RTX-PRO-6000", timeout=1800)
def run() -> None:
    import ctypes

    import torch

    def build(src, so):
        c = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
                            "-o", so, f"/root/cuda/{src}", "-lcuda"], capture_output=True, text=True)
        if c.returncode != 0:
            print(c.stderr, flush=True); raise SystemExit
        return ctypes.CDLL(so)

    lib = build("dense_decode_lib.cu", "/root/dec.so")
    lib.qb_encode_map.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 3
    lib.qb_encode_map.restype = ctypes.c_void_p
    lib.qb_decode_tn.argtypes = [ctypes.c_int]; lib.qb_decode_tn.restype = ctypes.c_int
    lib.dense_fp4_decode_cached_async.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3
    lib.qb_free_map.argtypes = [ctypes.c_void_p]
    sk = build("dense_sk_lib.cu", "/root/sk.so")
    sk.dense_fp4_mm_sk.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4
    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    print(name, flush=True)

    def tms(fn, it=50):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(it):
            fn()
        b.record(); torch.cuda.synchronize()
        return a.elapsed_time(b) / it  # ms

    # measured peak DRAM BW: large d2d copy (read+write both count)
    big = torch.empty(1 << 28, dtype=torch.float32, device=dev)  # 1 GiB
    dst = torch.empty_like(big)
    tcopy = tms(lambda: dst.copy_(big), it=30)
    peak = 2 * big.numel() * 4 / (tcopy / 1e3) / 1e12  # TB/s (read src + write dst)
    print(f"measured peak DRAM BW (d2d copy): {peak:.2f} TB/s\n", flush=True)

    hdr = (f"{'shape':>30} | {'bf16 us':>8} | {'fp4 us':>7} | {'speedup':>7} | "
           f"{'fp4 TB/s':>8} | {'%peak':>6} | {'ideal us':>8}")
    print(hdr + "\n" + "-" * len(hdr), flush=True)
    for label, M, N, K in SHAPES:
        Ab = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        Bb = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
        tb = tms(lambda: torch.matmul(Ab, Bb.t()))

        Kb = K // 2
        Ad = torch.full((M, Kb), 0x22, dtype=torch.uint8, device=dev)
        Bd = torch.full((N, Kb), 0x22, dtype=torch.uint8, device=dev)
        C = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
        mapA = lib.qb_encode_map(Ad.data_ptr(), M, K, 128)
        mapB = lib.qb_encode_map(Bd.data_ptr(), N, K, lib.qb_decode_tn(N))
        tf = tms(lambda: lib.dense_fp4_decode_cached_async(mapA, mapB, C.data_ptr(), M, N, K))

        dram_bytes = N * K / 2 + M * N * 2          # weight read once + bf16 write
        achieved = dram_bytes / (tf / 1e3) / 1e12
        ideal_us = dram_bytes / (peak * 1e12) * 1e6
        print(f"{label:>30} | {tb*1e3:7.1f} | {tf*1e3:6.1f} | {tb/tf:6.2f}x | "
              f"{achieved:7.2f} | {100*achieved/peak:5.1f}% | {ideal_us:7.1f}", flush=True)

    # split-K sweep on the two headroom shapes (adds z-blocks to fill SMs; f32 workspace + convert)
    print("\nsplit-K sweep (dense_fp4_mm_sk, end-to-end incl. workspace+convert+sync):", flush=True)
    for label, M, N, K in [("attn qkv/o 128x4096x4096", 128, 4096, 4096),
                           ("ffn down  128x4096x14336", 128, 4096, 14336)]:
        Ab = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        Bb = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
        tb = tms(lambda: torch.matmul(Ab, Bb.t()))
        Ad = torch.full((M, K // 2), 0x22, dtype=torch.uint8, device=dev)
        Bd = torch.full((N, K // 2), 0x22, dtype=torch.uint8, device=dev)
        C = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
        Cf = torch.empty((M, N), dtype=torch.float32, device=dev)
        row = f"{label:>26} (bf16 {tb*1e3:.1f}us):"
        for sp in (1, 2, 4, 8, 12, 16):
            ts = tms(lambda: sk.dense_fp4_mm_sk(Ad.data_ptr(), Bd.data_ptr(), C.data_ptr(),
                                                Cf.data_ptr(), M, N, K, sp))
            row += f"  s{sp}={tb/ts:.2f}x({ts*1e3:.0f}us)"
        print(row, flush=True)
    # NOTE: split-K loses for small-N decode (attn) in every reduction form tried (global-atomic AND
    # Marlin-style per-tile semaphore, tested in-tree then reverted): the f32-partial reduction
    # overhead exceeds the SM-fill gain because the 8 MB fp4 weight is too small to amortize it. It
    # only wins long-K ffn-down (s8 1.52x). Small-N attn is fill/latency-bound at the shape's ceiling.

    # batched-decode M-sweep. HYPOTHESIS (refuted): batching serving requests fills the SM array and
    # climbs %peak. MEASURED: %peak FALLS (o_proj 39->11%) and speedup plateaus ~1.45x. Root cause:
    # this split-N kernel uses fixed BM=128 blocks, so each 128-row block RE-READS the full weight ->
    # batching multiplies weight traffic instead of amortizing it (weight not stationary across M).
    # So the remedy for batched M is NOT this kernel: large M is prefill, use the weight-stationary
    # prefill kernel (matmul_fp4_pp, 3.6-5x). Decode kernel's job is small-M, where it's ceiling-bound.
    print("\nbatched-decode M-sweep (speedup vs bf16 / %DRAM-peak; fill recovers as M grows):", flush=True)
    for label, N, K in [("o_proj      4096x4096", 4096, 4096), ("fused QKV GQA 6144x4096", 6144, 4096)]:
        Bb = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
        Bd = torch.full((N, K // 2), 0x22, dtype=torch.uint8, device=dev)
        mapB = lib.qb_encode_map(Bd.data_ptr(), N, K, lib.qb_decode_tn(N))
        row = f"{label:>24}:"
        for M in (128, 256, 512, 1024, 2048):
            Ab = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            tb = tms(lambda: torch.matmul(Ab, Bb.t()))
            Ad = torch.full((M, K // 2), 0x22, dtype=torch.uint8, device=dev)
            C = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
            mapA = lib.qb_encode_map(Ad.data_ptr(), M, K, 128)
            tf = tms(lambda: lib.dense_fp4_decode_cached_async(mapA, mapB, C.data_ptr(), M, N, K))
            lib.qb_free_map(mapA)
            pk = 100 * (N * K / 2 + M * N * 2) / (tf / 1e3) / 1e12 / peak
            row += f"  M{M}={tb / tf:.2f}x/{pk:.0f}%"
        lib.qb_free_map(mapB)
        print(row, flush=True)


@app.local_entrypoint()
def main() -> None:
    run.remote()
