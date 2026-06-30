"""QuadbitLinear: the 2:4-sparse FP4 kernel as a drop-in for torch nn.Linear on SM120.

Proves the kernel is usable end-to-end from PyTorch: take a normal dense weight, prune
it 2:4 by magnitude + pack to NVFP4 in the kernel's exact layouts (in torch), quantize
activations per-forward, run the sparse_fp4 kernel, and compare accuracy + speed to a
bf16 nn.Linear. Validation runs with UNIT scales (the metadata/2:4/compress packing is the
novel torch code; the ue4m3 scale path is already verified maxrel-0 in the CUDA probes), so
the only error vs the reference is bf16 output rounding.

Mapping: nn.Linear y = x @ W^T. Kernel computes C[m,n] = sum_k A[m,k]*B[n,k] (A 2:4-sparse).
So A = W [out,in] (sparse weight, pruned offline), B = x [batch,in] (dense activations),
C = W @ x^T = y^T, shape [out, batch]. Tile constraints: out%256, batch%128, in%256.

Run:  uv run modal run harness/quadbit_linear.py
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

app = modal.App("quadbit-linear", image=image)


@app.function(gpu="RTX-PRO-6000", timeout=1800)
def run() -> None:
    so = "/root/sparse_fp4.so"
    c = subprocess.run(
        ["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
         "-o", so, "/root/cuda/sparse_fp4_lib.cu", "-lcuda"],
        capture_output=True, text=True,
    )
    print(c.stdout + c.stderr, flush=True)
    if c.returncode != 0:
        print(">>> nvcc failed", flush=True)
        return

    import ctypes

    import torch

    print(f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}", flush=True)
    lib = ctypes.CDLL(so)
    lib.sparse_fp4_mm.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3
    lib.sparse_fp4_mm.restype = ctypes.c_int
    dev = torch.device("cuda")

    # fp4 e2m1 code(0..15) -> value, and round-to-nearest fp4 (unit scale).
    FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
                       dtype=torch.float32, device=dev)
    BND = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], dtype=torch.float32, device=dev)

    def quant_fp4(v):
        """fp32 -> fp4 code (uint8), unit scale (round to nearest e2m1 grid)."""
        idx = torch.bucketize(v.abs(), BND).to(torch.uint8)  # 0..7 magnitude
        return idx | ((v < 0).to(torch.uint8) << 3)

    class QuadbitLinear:
        """Drop-in for nn.Linear(in, out, bias=False): packs W once, sparse-FP4 forward."""

        def __init__(self, W):  # W: [out, in] fp32
            out_f, in_f = W.shape
            assert out_f % 256 == 0 and in_f % 256 == 0, "out%256, in%256"
            ks = in_f // 128
            # W -> [out, ks, 16 groups, 4 pairs, 2 (lo/hi fp4)]
            codes = quant_fp4(W).view(out_f, ks, 16, 4, 2)
            pmag = FP4[codes.long()].abs().sum(-1)            # [out,ks,16,4] pair magnitude
            top2 = pmag.topk(2, dim=-1).indices               # [out,ks,16,2]
            i01, _ = top2.sort(dim=-1)                         # ascending: i0<i1
            i0, i1 = i01[..., 0], i01[..., 1]                  # [out,ks,16]
            # compressed A: slot0=pair i0, slot1=pair i1 ; byte = lo | hi<<4
            kept = torch.stack([i0, i1], dim=-1)               # [out,ks,16,2]
            kc = torch.gather(codes, 3, kept.unsqueeze(-1).expand(-1, -1, -1, -1, 2).long())
            Abytes = (kc[..., 0] | (kc[..., 1] << 4)).reshape(out_f, ks * 32).contiguous()
            # metadata: nib = i0|(i1<<2) ; u32 per half (g/8), nibble g%8 at shift (g%8)*4
            nib = (i0 | (i1 << 2)).view(out_f, ks, 2, 8).long()   # [out,ks,half,j]
            sh = (torch.arange(8, device=dev) * 4).view(1, 1, 1, 8)
            meta = (nib << sh).sum(-1).to(torch.int32)            # [out,ks,2]
            meta = meta.permute(1, 0, 2).contiguous()             # step-major [ks,out,2]
            self.out_f, self.in_f, self.ks = out_f, in_f, ks
            self.Ac = Abytes.to(torch.uint8)
            self.meta = meta
            self.scaleA = torch.full((ks, out_f, 4), 0x38, dtype=torch.uint8, device=dev)
            # dense dequant of the pruned weight (unit scale) for the reference
            mask = torch.zeros(out_f, ks, 16, 4, device=dev)
            mask.scatter_(3, kept.long(), 1.0)
            self.W_deq = (FP4[codes.long()] * mask.unsqueeze(-1)).reshape(out_f, in_f)

        def forward(self, x):  # x: [batch, in] fp32 -> y: [batch, out] bf16
            batch = x.shape[0]
            assert batch % 128 == 0 and x.shape[1] == self.in_f, "batch%128, in matches"
            xc = quant_fp4(x)
            Bbytes = (xc.view(batch, self.in_f // 2, 2)[..., 0]
                      | (xc.view(batch, self.in_f // 2, 2)[..., 1] << 4)).to(torch.uint8).contiguous()
            scaleB = torch.full((self.ks, batch, 4), 0x38, dtype=torch.uint8, device=dev)
            C = torch.empty((self.out_f, batch), dtype=torch.bfloat16, device=dev)  # [out,batch]
            rc = lib.sparse_fp4_mm(self.Ac.data_ptr(), Bbytes.data_ptr(), self.scaleA.data_ptr(),
                                   scaleB.data_ptr(), self.meta.data_ptr(), C.data_ptr(),
                                   self.out_f, batch, self.in_f)
            if rc != 0:
                raise RuntimeError(f"kernel cuda error {rc}")
            self.x_deq = FP4[xc.long()]            # dense dequant of activations (unit scale)
            return C.t()                            # [batch, out]

    torch.manual_seed(0)
    for (out_f, in_f, batch) in [(512, 512, 256), (4096, 4096, 512), (8192, 8192, 512)]:
        W = torch.randn(out_f, in_f, device=dev)
        x = torch.randn(batch, in_f, device=dev)
        ql = QuadbitLinear(W)
        y = ql.forward(x)                           # [batch, out] bf16
        ref = (ql.x_deq @ ql.W_deq.t())             # [batch, out] fp32, pruned+fp4 dense matmul
        rel = ((y.float() - ref).abs() / (ref.abs() + 1.0)).max().item()
        ok = "PASS" if rel < 6e-3 else "FAIL"

        it = 30
        for _ in range(5):
            ql.forward(x)
        torch.cuda.synchronize()
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(it):
            ql.forward(x)
        en.record(); torch.cuda.synchronize()
        ms = st.elapsed_time(en) / it
        # bf16 dense baseline (same logical GEMM out x batch x in)
        Wb, xb = W.bfloat16(), x.bfloat16()
        for _ in range(5):
            torch.matmul(Wb, xb.t())
        torch.cuda.synchronize()
        st.record()
        for _ in range(it):
            torch.matmul(Wb, xb.t())
        en.record(); torch.cuda.synchronize()
        ms_bf16 = st.elapsed_time(en) / it
        gf = 2.0 * out_f * batch * in_f / (ms / 1e3) / 1e9
        print(f"QuadbitLinear out={out_f} in={in_f} batch={batch}: {ms:.3f} ms {gf:.0f} GFLOP/s "
              f"{ok} (maxrel {rel:.4f}) | torch bf16 {ms_bf16:.3f} ms -> {ms_bf16/ms:.2f}x", flush=True)


@app.local_entrypoint()
def main() -> None:
    run.remote()
