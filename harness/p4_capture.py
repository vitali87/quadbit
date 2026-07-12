"""P4 M2-seg: standalone graph-capture of the segmented sparse-FP4 MoE path.

Proves the MoE seg path (quantize_act_nvfp4_2lvl_s + sparse_moe_mm_2lvl) is CUDA-graph-capturable with
ZERO host sync in the captured region, before touching vLLM. Differences vs the frozen moe_seg.py capture:
  * uses the additive stream-safe quantizer quantize_act_nvfp4_2lvl_s (default-stream quantizer never
    enters the captured region);
  * fixed-capacity DEVICE routing (no per-expert .item(), no host loop, no data-dependent alloc) -- eblk
    is a compile-time constant; overflow rows are dropped deterministically;
  * full metric matrix (cos/relL2/maxabs/maxrel/nonfinite, eager vs graph latency p50/p95/min, memory),
    decode + prefill shapes, and pathological routings.
Compiles the branch .cu (with the new _s symbol) fresh, like moe_seg.py. Does not touch the frozen path.
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
    .pip_install("numpy")
    .add_local_dir((ROOT / "cuda").as_posix(), "/root/cuda")
)
app = modal.App("quadbit-p4capture", image=image)

BN = 128


@app.function(gpu="RTX-PRO-6000", timeout=1800)
def validate() -> None:
    import ctypes
    import statistics

    import torch
    import torch.nn.functional as F

    so = "/root/sparse_fp4.so"
    c = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
                        "-o", so, "/root/cuda/sparse_fp4_lib.cu", "-lcuda"], capture_output=True, text=True)
    if c.returncode != 0:
        print(c.stderr, flush=True)
        raise SystemExit(1)
    lib = ctypes.CDLL(so)
    lib.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    lib.quantize_act_nvfp4_2lvl_s.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
    lib.sparse_moe_mm_2lvl.argtypes = ([ctypes.c_void_p] * 6 + [ctypes.c_int] * 4 +
                                       [ctypes.c_void_p] * 3 + [ctypes.c_int] + [ctypes.c_void_p])
    lib.sparse_moe_mm_2lvl.restype = ctypes.c_int
    lib.qb_init_moe_attrs()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    print(f"# torch {torch.__version__} | {torch.cuda.get_device_name(0)} | "
          f"cap {torch.cuda.get_device_capability(0)} | driver {torch.version.cuda}", flush=True)

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

    def pack(W):  # bf16 [out,in] -> (Ac, meta, scaleA, gA); M1-validated layout (see moe_seg.py)
        out_f, in_f = W.shape; ks = in_f // 128
        Wg = W.float().to(dev).view(out_f, ks, 16, 4, 2)
        i01, _ = Wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
        keptW = torch.gather(Wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
        gA = (keptW.abs().amax(dim=(1, 2, 3, 4), keepdim=True) / 2688.0).clamp_min(1e-30).reshape(out_f, 1, 1)
        blk = keptW.reshape(out_f, ks, 4, 8, 2)
        scode = enc_ue4m3_t((blk.abs().amax(dim=(3, 4)) / 6.0) / gA)
        sdeq = UE4M3[scode] * gA
        kc = q_fp4(blk / sdeq.clamp_min(1e-30)[..., None, None])
        Ac = (kc[..., 0] | (kc[..., 1] << 4)).reshape(out_f, ks * 32).to(torch.uint8)
        nib = (i01[..., 0] | (i01[..., 1] << 2)).view(out_f, ks, 2, 8)
        sh = (torch.arange(8, device=dev) * 4).view(1, 1, 1, 8)
        meta = (nib << sh).sum(-1).to(torch.int32).permute(1, 0, 2).contiguous()
        return (Ac.contiguous(), meta, scode.to(torch.uint8).permute(1, 0, 2).contiguous(),
                gA.reshape(out_f).float().contiguous())

    def stack(packs):
        return (torch.cat([p[0] for p in packs], 0).contiguous(),
                torch.cat([p[1] for p in packs], 1).contiguous(),
                torch.cat([p[2] for p in packs], 1).contiguous(),
                torch.cat([p[3] for p in packs], 0).contiguous())

    def route_fixed_cap(assign, E, cap):
        """Fixed-capacity DEVICE routing. assign:[R] expert per routed row. Returns src:[E*cap] (token
        index per padded slot, -1 = pad), eblk:[E*cap/BN] (CONSTANT: block b -> b//(cap/BN)), and the
        dropped-row count. No .item(), no host loop, no data-dependent shape. Overflow (>cap rows for an
        expert) is dropped deterministically (lowest sorted rows kept)."""
        R = assign.shape[0]
        order = torch.argsort(assign, stable=True)          # rows grouped by expert
        sa = assign[order]                                  # sorted expert ids
        counts = torch.bincount(assign, minlength=E)
        offs = torch.cumsum(counts, 0) - counts             # exclusive prefix (device)
        within = torch.arange(R, device=dev) - offs[sa]     # rank within its expert
        keep = within < cap                                 # deterministic overflow drop
        dest = sa * cap + within                            # destination slot
        src = torch.full((E * cap,), -1, dtype=torch.long, device=dev)
        src[dest[keep]] = order[keep]
        eblk = (torch.arange(E * cap // BN, device=dev) // (cap // BN)).to(torch.int32)
        dropped = R - int(keep.sum()) if not keep.all() else 0
        return src, eblk, dropped

    st_ptr = lambda: torch.cuda.current_stream().cuda_stream

    def quant_s(x, Bb, sB, gB):  # stream-safe quantizer into preallocated buffers
        R, in_f = x.shape
        lib.quantize_act_nvfp4_2lvl_s(x.data_ptr(), Bb.data_ptr(), sB.data_ptr(), gB.data_ptr(),
                                      R, in_f, st_ptr())

    def seg(W, Bb, sB, gB, C, Mpe, R, in_f, eblk):  # sparse_moe_mm_2lvl on the CURRENT stream
        Ac, meta, scaleA, gA = W
        lib.sparse_moe_mm_2lvl(Ac.data_ptr(), Bb.data_ptr(), scaleA.data_ptr(), sB.data_ptr(),
                               meta.data_ptr(), C.data_ptr(), Ac.shape[0], Mpe, R, in_f, gA.data_ptr(),
                               gB.data_ptr(), eblk.data_ptr(), 1, st_ptr())

    def metrics(a, b):
        a, b = a.float(), b.float()
        cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        rel = (a - b).norm().item() / max(b.norm().item(), 1e-12)
        mx = (a - b).abs().max().item()
        mxr = ((a - b).abs() / b.abs().clamp_min(1e-3)).max().item()
        nf = int((~torch.isfinite(a)).sum().item())
        return cos, rel, mx, mxr, nf

    def stats_ms(fn, it=100):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(it):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        return statistics.median(ts), ts[min(int(0.95 * it), it - 1)], ts[0]

    H, I, E, topk = 4096, 2048, 8, 6           # deepseek-ish expert shape, small E for a fast standalone
    Wg = [torch.randn(I, H, device=dev) * (H ** -0.5) for _ in range(E)]
    Wu = [torch.randn(I, H, device=dev) * (H ** -0.5) for _ in range(E)]
    Wd = [torch.randn(H, I, device=dev) * (I ** -0.5) for _ in range(E)]
    gu_W = stack([pack(torch.cat([Wg[e], Wu[e]], 0)) for e in range(E)])
    dn_W = stack([pack(Wd[e]) for e in range(E)])
    ksH, ksI = H // 128, I // 128

    def run_shape(tag, T, assign, cap):
        """Preallocate everything, define a sync-free one_pass over the CURRENT stream, warm on a side
        stream, capture, replay, and compare graph-vs-eager on the valid rows. Returns a result dict."""
        Rp = E * cap
        src, eblk, dropped = route_fixed_cap(assign, E, cap)
        valid = src >= 0
        srcc = src.clamp_min(0)
        Xtok = (torch.randn(max(T, 1), H, device=dev, dtype=torch.bfloat16) * 0.1)
        Xs = (Xtok[srcc % Xtok.shape[0]] * valid[:, None]).contiguous()
        Bb = torch.empty((Rp, H // 2), dtype=torch.uint8, device=dev)
        sB = torch.empty((ksH, Rp, 4), dtype=torch.uint8, device=dev)
        gB = torch.empty((Rp,), dtype=torch.float32, device=dev)
        GU = torch.empty((Rp, 2 * I), dtype=torch.bfloat16, device=dev)
        Hb = torch.empty((Rp, I), dtype=torch.bfloat16, device=dev)
        Bb2 = torch.empty((Rp, I // 2), dtype=torch.uint8, device=dev)
        sB2 = torch.empty((ksI, Rp, 4), dtype=torch.uint8, device=dev)
        gB2 = torch.empty((Rp,), dtype=torch.float32, device=dev)
        D = torch.empty((Rp, H), dtype=torch.bfloat16, device=dev)

        def one_pass():
            quant_s(Xs, Bb, sB, gB)
            seg(gu_W, Bb, sB, gB, GU, 2 * I, Rp, H, eblk)
            Hb.copy_((F.silu(GU[:, :I].float()) * GU[:, I:].float()).to(torch.bfloat16))
            quant_s(Hb, Bb2, sB2, gB2)
            seg(dn_W, Bb2, sB2, gB2, D, H, Rp, I, eblk)

        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                one_pass()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        eager = D.clone()
        eln = stats_ms(one_pass)

        res = {"tag": tag, "T": T, "Rp": Rp, "cap": cap, "dropped": dropped, "valid": int(valid.sum())}
        try:
            torch.cuda.reset_peak_memory_stats()
            gph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gph):
                one_pass()
            for _ in range(3):
                gph.replay()
            torch.cuda.synchronize()
            graph_out = D.clone()
            cos, rel, mx, mxr, nf = metrics(graph_out[valid], eager[valid])
            gln = stats_ms(lambda: gph.replay())
            res.update({"capture": "OK", "cos": cos, "relL2": rel, "maxabs": mx, "maxrel": mxr,
                        "nonfin": nf, "eager_p50": eln[0], "graph_p50": gln[0],
                        "mem_mb": torch.cuda.max_memory_allocated() / 1e6})
        except Exception as ex:  # noqa: BLE001
            res.update({"capture": f"FAIL: {type(ex).__name__}: {ex}"})
        return res

    def rup(x):
        return ((x + BN - 1) // BN) * BN

    print("\n# ===== decode + prefill shapes (uniform routing) =====", flush=True)
    print(f"{'tag':>16} {'T':>7} {'Rp':>8} {'drop':>6} | {'cap':>6} {'cos':>9} {'relL2':>9} "
          f"{'maxrel':>8} {'nf':>4} | {'eagerms':>8} {'graphms':>8} {'memMB':>8} | capture", flush=True)
    rows = []
    for kind, T in [("decode-B1", 1), ("decode-B8", 8), ("decode-B32", 32), ("decode-B64", 64),
                    ("prefill-2048", 2048), ("prefill-8192", 8192), ("prefill-16384", 16384),
                    ("prefill-65536", 65536)]:
        R = T * topk
        g = torch.Generator(device=dev).manual_seed(T)
        assign = torch.randint(0, E, (R,), device=dev, generator=g)
        cap = max(BN, rup(2 * R // E))          # 2x headroom over the uniform mean
        rows.append(run_shape(kind, T, assign, cap))

    print("\n# ===== pathological routings (T=4096, topk=6) =====", flush=True)
    Tp = 4096; Rp0 = Tp * topk
    cap_norm = max(BN, rup(2 * Rp0 // E))
    paths = [
        ("all->expert0", torch.zeros(Rp0, dtype=torch.long, device=dev), cap_norm),   # overflow -> drop
        ("uniform", torch.randint(0, E, (Rp0,), device=dev, generator=torch.Generator(device=dev).manual_seed(7)), cap_norm),
        ("empty-experts", torch.randint(0, 2, (Rp0,), device=dev, generator=torch.Generator(device=dev).manual_seed(8)), cap_norm),
        ("near-capacity", torch.arange(Rp0, device=dev) % E, cap_norm),
    ]
    for tag, assign, cap in paths:
        rows.append(run_shape(tag, Tp, assign, cap))

    def fmt(r):
        if r["capture"] != "OK":
            return (f"{r['tag']:>16} {r['T']:>7} {r['Rp']:>8} {r['dropped']:>6} | {r['cap']:>6} "
                    f"{'-':>9} {'-':>9} {'-':>8} {'-':>4} | {'-':>8} {'-':>8} {'-':>8} | {r['capture']}")
        return (f"{r['tag']:>16} {r['T']:>7} {r['Rp']:>8} {r['dropped']:>6} | {r['cap']:>6} "
                f"{r['cos']:>9.6f} {r['relL2']:>9.2e} {r['maxrel']:>8.2e} {r['nonfin']:>4} | "
                f"{r['eager_p50']:>8.3f} {r['graph_p50']:>8.3f} {r['mem_mb']:>8.1f} | {r['capture']}")

    print("\n# ===== M2-seg verdict =====", flush=True)
    for r in rows:
        print(fmt(r), flush=True)
    ok = [r for r in rows if r["capture"] == "OK"]
    passed = [r for r in ok if r["cos"] > 0.9999 and r["nonfin"] == 0]
    print(f"\n# capture OK {len(ok)}/{len(rows)}; graph==eager (cos>0.9999, nonfin=0) {len(passed)}/{len(ok)}",
          flush=True)
    print(f"# M2-SEG {'PASS' if len(passed) == len(rows) else 'PARTIAL/FAIL'}", flush=True)


@app.local_entrypoint()
def main() -> None:
    validate.remote()
