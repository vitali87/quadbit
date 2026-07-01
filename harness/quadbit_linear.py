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
    lib.quantize_act_nvfp4.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2
    lib.quantize_act_nvfp4.restype = None
    dev = torch.device("cuda")

    # fp4 e2m1 code(0..15) -> value, and round-to-nearest fp4 (unit scale, weight side).
    FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
                       dtype=torch.float32, device=dev)
    BND = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], dtype=torch.float32, device=dev)
    # ue4m3 code(0..127) -> value (for decoding the fused quantizer's per-block scales).
    _c = torch.arange(128, device=dev)
    _e, _m = (_c >> 3) & 0xf, _c & 7
    UE4M3 = torch.where(_e == 0, _m.float() * 0.001953125,
                        (1.0 + _m.float() / 8.0) * torch.exp2((_e - 7).float()))
    _MID = (UE4M3[:-1] + UE4M3[1:]) / 2  # 127 monotonic midpoints for nearest-ue4m3 encode

    def quant_fp4(v):
        """fp32 -> fp4 code (uint8), unit scale (round to nearest e2m1 grid)."""
        idx = torch.bucketize(v.abs(), BND).to(torch.uint8)  # 0..7 magnitude
        return idx | ((v < 0).to(torch.uint8) << 3)

    def enc_ue4m3(s):
        """positive scale -> nearest ue4m3 code (uint8, 0..127)."""
        return torch.bucketize(s, _MID).to(torch.uint8)

    class QuadbitLinear:
        """Drop-in for nn.Linear(in, out, bias=False): packs W once, sparse-FP4 forward."""

        def __init__(self, W):  # W: [out, in] fp32
            out_f, in_f = W.shape
            assert out_f % 256 == 0 and in_f % 256 == 0, "out%256, in%256"
            ks = in_f // 128
            # W -> [out, ks, 16 groups, 4 pairs, 2 (lo/hi fp4)]; prune 2:4 by raw magnitude
            Wg = W.view(out_f, ks, 16, 4, 2)
            pmag = Wg.abs().sum(-1)                            # [out,ks,16,4] pair magnitude
            top2 = pmag.topk(2, dim=-1).indices                # [out,ks,16,2]
            i01, _ = top2.sort(dim=-1)                          # ascending: i0<i1
            i0, i1 = i01[..., 0], i01[..., 1]                   # [out,ks,16]
            kept = torch.stack([i0, i1], dim=-1)                # [out,ks,16,2]
            keptW = torch.gather(Wg, 3, kept.unsqueeze(-1).expand(-1, -1, -1, -1, 2).long())
            # real per-block ue4m3 scales: 4 scaleA blocks/step, 8 compressed bytes each
            blk = keptW.reshape(out_f, ks, 4, 8, 2)             # (block, byte-in-block, lo/hi)
            scode = enc_ue4m3(blk.abs().amax(dim=(3, 4)) / 6.0)  # [out,ks,4] ue4m3 codes
            sdeq = UE4M3[scode.long()]                          # [out,ks,4]
            kc = quant_fp4(blk / sdeq[..., None, None])         # [out,ks,4,8,2] fp4 codes
            Abytes = (kc[..., 0] | (kc[..., 1] << 4)).reshape(out_f, ks * 32).contiguous()
            # metadata: nib = i0|(i1<<2) ; u32 per half (g/8), nibble g%8 at shift (g%8)*4
            nib = (i0 | (i1 << 2)).view(out_f, ks, 2, 8).long()   # [out,ks,half,j]
            sh = (torch.arange(8, device=dev) * 4).view(1, 1, 1, 8)
            meta = (nib << sh).sum(-1).to(torch.int32).permute(1, 0, 2).contiguous()  # [ks,out,2]
            self.out_f, self.in_f, self.ks = out_f, in_f, ks
            self.Ac = Abytes.to(torch.uint8)
            self.meta = meta
            self.scaleA = scode.to(torch.uint8).permute(1, 0, 2).contiguous()  # step-major [ks,out,4]
            # dense dequant of the pruned+scaled weight for the reference
            kd = (FP4[kc.long()] * sdeq[..., None, None]).reshape(out_f, ks, 16, 2, 2)
            Wd = torch.zeros(out_f, ks, 16, 4, 2, device=dev)
            Wd.scatter_(3, kept.unsqueeze(-1).expand(-1, -1, -1, -1, 2).long(), kd)
            self.W_deq = Wd.reshape(out_f, in_f)

        def pack_act(self, x):  # x: [batch, in] bf16 -> (Bbytes, scaleB), fused real ue4m3 scales
            batch = x.shape[0]
            assert batch % 128 == 0 and x.shape[1] == self.in_f, "batch%128, in matches"
            x = x.to(torch.bfloat16).contiguous()
            Bbytes = torch.empty((batch, self.in_f // 2), dtype=torch.uint8, device=dev)
            scaleB = torch.empty((self.ks, batch, 4), dtype=torch.uint8, device=dev)
            lib.quantize_act_nvfp4(x.data_ptr(), Bbytes.data_ptr(), scaleB.data_ptr(),
                                   batch, self.in_f)
            return Bbytes, scaleB

        def deq_act(self, Bbytes, scaleB):      # read packed activations back to dense (for ref)
            batch = Bbytes.shape[0]
            codes = torch.stack([(Bbytes & 0xf).long(), (Bbytes >> 4).long()], -1).view(batch, self.in_f)
            sB = UE4M3[scaleB.long()].permute(1, 0, 2).reshape(batch, self.ks * 4)
            return FP4[codes] * sB.repeat_interleave(32, dim=1)

        def run(self, Bbytes, scaleB, batch):       # packed activations -> C [out, batch] bf16
            C = torch.empty((self.out_f, batch), dtype=torch.bfloat16, device=dev)
            rc = lib.sparse_fp4_mm(self.Ac.data_ptr(), Bbytes.data_ptr(), self.scaleA.data_ptr(),
                                   scaleB.data_ptr(), self.meta.data_ptr(), C.data_ptr(),
                                   self.out_f, batch, self.in_f)
            if rc != 0:
                raise RuntimeError(f"kernel cuda error {rc}")
            return C

        def forward(self, x):  # x: [batch, in] fp32 -> y: [batch, out] bf16
            Bbytes, scaleB = self.pack_act(x)
            return self.run(Bbytes, scaleB, x.shape[0]).t()

    def time_ms(fn, it=30):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(it):
            fn()
        en.record(); torch.cuda.synchronize()
        return st.elapsed_time(en) / it

    torch.manual_seed(0)
    print("# correctness (maxrel vs packed ref = layout check; err vs true fp32 = quant accuracy)")
    for (out_f, in_f, batch, wscale) in [(512, 512, 256, 1.0), (8192, 8192, 512, 1.0),
                                         (4096, 4096, 256, 0.02)]:  # 0.02 = real-weight magnitude
        W = torch.randn(out_f, in_f, device=dev) * wscale
        x = torch.randn(batch, in_f, device=dev)
        ql = QuadbitLinear(W)
        Bb, sB = ql.pack_act(x)
        y = ql.run(Bb, sB, batch).t()
        ref = ql.deq_act(Bb, sB) @ ql.W_deq.t()               # self-consistent packed ref
        rel = ((y.float() - ref).abs() / (ref.abs() + 1.0)).max().item()
        true = x @ W.t()                                      # true fp32 dense (2:4 + FP4 error)
        acc = ((y.float() - true).norm() / true.norm()).item()
        print(f"  out={out_f} in={in_f} batch={batch} wscale={wscale}: "
              f"{'PASS' if rel < 6e-3 else 'FAIL'} (maxrel {rel:.4f}, err-vs-fp32 {acc:.3f})",
              flush=True)

    # speed vs torch bf16 across the prefill batch dimension (out=in=8192, the kernel's strong size)
    print("# speed @ out=in=8192 (full=quant+kernel, kern=kernel only, vs torch bf16 dense)", flush=True)
    out_f = in_f = 8192
    W = torch.randn(out_f, in_f, device=dev)
    Wb = W.bfloat16()
    ql = QuadbitLinear(W)
    for batch in (512, 1024, 2048, 4096, 8192):
        x = torch.randn(batch, in_f, device=dev)
        xb = x.bfloat16()
        Bb, sB = ql.pack_act(xb)
        ms_full = time_ms(lambda: ql.forward(xb))
        ms_kern = time_ms(lambda: ql.run(Bb, sB, batch))
        ms_bf16 = time_ms(lambda: torch.matmul(Wb, xb.t()))
        gf = 2.0 * out_f * batch * in_f / (ms_kern / 1e3) / 1e9
        print(f"  batch={batch:5d}: full {ms_full:.3f} kern {ms_kern:.3f} ({gf:.0f} GF/s) "
              f"bf16 {ms_bf16:.3f} | kern {ms_bf16/ms_kern:.2f}x  full {ms_bf16/ms_full:.2f}x", flush=True)


@app.local_entrypoint()
def main() -> None:
    run.remote()
