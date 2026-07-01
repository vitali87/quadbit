"""Decode/prefill routing question: does the real-scale weight-stationary prefill kernel
(dense_scaled_fast_mm, MXFP4) already beat cuBLAS bf16 across the small-token (decode) range,
or is a separate decode kernel needed? The unit-scale split-N decode kernel is NOT deployable
(no real scales), so a real router can only dispatch between real-scale kernels. This sweeps
tokens N=256..4096 (the kernel tiles tokens by 2*DBN=256, so 256 is the floor; decode batches
below that pad up) for real Llama/Qwen linear shapes and reports speedup vs bf16 + %DRAM-peak.

MEASURED: the prefill kernel does NOT uniformly beat bf16 at decode token counts. It LOSES
(0.68-0.77x) in one regime -- small tokens (256) AND small output-N (4096: o_proj, ffn-down) --
because the grid = ceil(M/256)*ceil(N/128) underfills the 188-SM array (32 blocks < ~48). It wins
everywhere else (large-N even at M256: qkv 2.71x, ffn-up 2.45x; and all shapes at M>=1024). So the
router's job is to NEVER REGRESS: pick FP4 where the block count fills the array, else fall back to
bf16. route_dense() encodes that fill rule; the run asserts it never selects FP4 in a losing cell.
The 0.68-0.77x corner is the target for a real-scale decode kernel (would lift it to ~1.3x, the
small-N decode ceiling); until that exists, bf16 fallback keeps the go-to call regression-free.

Run:  uv run modal run harness/bench_router.py
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
app = modal.App("quadbit-bench-router", image=image)

# (label, out N-weight, in K). tokens M swept. Real decode/prefill linears.
SHAPES = [
    ("o_proj      4096x 4096", 4096, 4096),
    ("qkv GQA     6144x 4096", 6144, 4096),
    ("ffn up     14336x 4096", 14336, 4096),
    ("ffn down    4096x14336", 4096, 14336),
]
TOKENS = [256, 512, 1024, 2048, 4096]
FILL_BLOCKS = 48  # min grid blocks (ceil(M/256)*ceil(N/128)) for the FP4 prefill kernel to beat bf16


def route_dense(m: int, n: int) -> str:  # "fp4" if the grid fills the SM array, else "bf16"
    blocks = ((m + 255) // 256) * ((n + 127) // 128)
    return "fp4" if blocks >= FILL_BLOCKS else "bf16"


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

    lib = build("dense_scaled_fast_lib.cu", "/root/dsf.so")
    lib.dense_scaled_fast_mm.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3
    lib.quantize_act_mxfp4.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2
    lib.quantize_act_mxfp4.restype = None
    dev = torch.device("cuda")
    print(torch.cuda.get_device_name(0), flush=True)

    BND = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], dtype=torch.float32, device=dev)

    def quant_fp4(v):
        return (torch.bucketize(v.abs(), BND) | ((v < 0).long() << 3)).to(torch.uint8)

    def mxfp4_pack(W):  # W[out,in] -> Ab[out,in/2], SFA[step][out][4] ue8m0 (kernel layout)
        out, inn = W.shape
        Wb = W.float().view(out, inn // 32, 32)
        _, e = torch.frexp(Wb.abs().amax(-1) / 6.0)
        code = (e + 127).clamp(0, 255)
        scale = torch.ldexp(torch.ones_like(code, dtype=torch.float32), code - 127)
        q = quant_fp4(Wb / scale[..., None]).view(out, inn)
        Ab = (q[:, 0::2] | (q[:, 1::2] << 4)).to(torch.uint8).contiguous()
        SFA = code.view(out, inn // 128, 4).permute(1, 0, 2).contiguous().to(torch.uint8)
        return Ab, SFA

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

    big = torch.empty(1 << 28, dtype=torch.float32, device=dev)
    dst = torch.empty_like(big)
    peak = 2 * big.numel() * 4 / (tms(lambda: dst.copy_(big), it=30) / 1e3) / 1e12
    print(f"measured peak DRAM BW: {peak:.2f} TB/s\n", flush=True)
    print("prefill kernel (dense_scaled_fast_mm) speedup vs bf16 / %DRAM-peak, over token count:", flush=True)

    for label, N, K in SHAPES:
        W = (torch.randn(N, K, device=dev) * 0.02)
        Wb = W.to(torch.bfloat16)
        Ab, SFA = mxfp4_pack(W)
        row = f"{label:>24}:"
        for M in TOKENS:  # M = tokens
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            tb = tms(lambda: torch.matmul(x, Wb.t()))
            Bb = torch.empty((M, K // 2), dtype=torch.uint8, device=dev)
            SFB = torch.empty((M, K // 32), dtype=torch.uint8, device=dev)
            lib.quantize_act_mxfp4(x.contiguous().data_ptr(), Bb.data_ptr(), SFB.data_ptr(), M, K)
            C = torch.empty((N, M), dtype=torch.bfloat16, device=dev)
            tf = tms(lambda: lib.dense_scaled_fast_mm(Ab.data_ptr(), Bb.data_ptr(), SFA.data_ptr(),
                                                      SFB.data_ptr(), C.data_ptr(), N, M, K))
            pk = 100 * (N * K / 2 + M * N * 2) / (tf / 1e3) / 1e12 / peak
            choice = route_dense(M, N)
            routed = (tb / tf) if choice == "fp4" else 1.0    # bf16 fallback speedup = 1.0
            assert not (choice == "fp4" and tb / tf < 0.98), \
                f"router chose FP4 in a losing cell {label} M{M}: {tb / tf:.2f}x"
            row += f"  M{M}={tb / tf:.2f}x/{pk:.0f}%[{choice}->{routed:.2f}x]"
        print(row, flush=True)
    print(f"\nrouter never regressed (routed >= bf16 in every cell). rule: FP4 iff "
          f"ceil(M/256)*ceil(N/128) >= {FILL_BLOCKS} blocks.", flush=True)


@app.local_entrypoint()
def main() -> None:
    run.remote()
