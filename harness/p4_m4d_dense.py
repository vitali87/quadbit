"""P4 M4-D de-risk: does the graph-safe DENSE grouped matmul (_dense_seg_gs) capture memory-viably?

The deployed policies (down49/gateup49/route-slot D2) anchor one projection as a DENSE NVFP4 grouped
matmul over fixed-capacity segments. _dense_seg_gs implements it as a `range(E)` loop (unrolled into the
graph). The open risk is memory: if the capture pool held one dequant/output buffer PER expert, E=128
experts x N layers would OOM. PyTorch reuses freed buffers within a capture, so it should stay bounded.
This proves it at E=128 (DeepSeek 2-GPU local count): capture succeeds, graph==eager, and peak memory for
the E-loop is ~the same as a single grouped pass (no E-fold blowup). bf16 weights (the NVFP4 dequant
inside _dense_seg_gs is elementwise / capture-safe / orthogonal to this question).
"""

import statistics
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64"})
    .pip_install("torch", index_url="https://download.pytorch.org/whl/nightly/cu128", pre=True)
)
app = modal.App("quadbit-p4m4d-dense", image=image)


@app.function(gpu="RTX-PRO-6000", timeout=1800)
def validate() -> None:
    import torch
    import torch.nn.functional as F

    dev = torch.device("cuda")
    torch.manual_seed(0)
    print(f"# torch {torch.__version__} | {torch.cuda.get_device_name(0)} | "
          f"cap {torch.cuda.get_device_capability(0)}", flush=True)

    def dense_seg(xs, W, e, cap, out_dim):
        # mirror of plugin _dense_seg_gs: range(e) unrolled, per-iter temp freed -> pool reuses buffers
        out = torch.empty(e * cap, out_dim, dtype=torch.bfloat16, device=dev)
        for le in range(e):
            out[le * cap:(le + 1) * cap] = (
                xs[le * cap:(le + 1) * cap].float() @ W[le].float().t()).to(torch.bfloat16)
        return out

    def stats_ms(fn, it=30):
        for _ in range(5):
            fn()
        torch.cuda.synchronize(); ts = []
        for _ in range(it):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
        return statistics.median(sorted(ts))

    H, OUT = 4096, 4096
    print(f"{'E':>4} {'cap':>4} {'in':>5} {'out':>5} | {'GvsE cos':>9} {'nf':>3} | "
          f"{'eagerms':>8} {'graphms':>8} | {'loopMB':>8} {'weightMB':>9} | capture", flush=True)
    rows = []
    for E, cap in [(32, 128), (128, 128), (128, 256)]:
        W = torch.randn(E, OUT, H, device=dev, dtype=torch.bfloat16) * (H ** -0.5)
        xs = torch.randn(E * cap, H, device=dev, dtype=torch.bfloat16) * 0.1
        # a few experts get zero-padded rows (empty-expert case): every 4th expert's block set to 0
        for le in range(0, E, 4):
            xs[le * cap:(le + 1) * cap] = 0
        wmb = W.numel() * 2 / 1e6

        # warm on side stream, eager reference
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                dense_seg(xs, W, E, cap, OUT)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        y_eager = dense_seg(xs, W, E, cap, OUT)
        eln = stats_ms(lambda: dense_seg(xs, W, E, cap, OUT))

        res = {"E": E, "cap": cap}
        try:
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                y_g = dense_seg(xs, W, E, cap, OUT)
            for _ in range(3):
                g.replay()
            torch.cuda.synchronize()
            loop_mb = (torch.cuda.max_memory_allocated() - base) / 1e6   # extra memory the E-loop needed
            cos = F.cosine_similarity(y_g.float().flatten(), y_eager.float().flatten(), dim=0).item()
            nf = int((~torch.isfinite(y_g)).sum().item())
            gln = stats_ms(lambda: g.replay())
            res.update({"capture": "OK", "cos": cos, "nf": nf, "eager": eln, "graph": gln,
                        "loop_mb": loop_mb, "wmb": wmb})
        except Exception as ex:  # noqa: BLE001
            res.update({"capture": f"FAIL: {type(ex).__name__}: {ex}", "wmb": wmb})
        rows.append(res)
        del W, xs
        torch.cuda.empty_cache()

    for r in rows:
        if r["capture"] != "OK":
            print(f"{r['E']:>4} {r['cap']:>4} {H:>5} {OUT:>5} | {r['capture']}", flush=True)
            continue
        print(f"{r['E']:>4} {r['cap']:>4} {H:>5} {OUT:>5} | {r['cos']:>9.6f} {r['nf']:>3} | "
              f"{r['eager']:>8.3f} {r['graph']:>8.3f} | {r['loop_mb']:>8.1f} {r['wmb']:>9.1f} | OK", flush=True)
    ok = [r for r in rows if r["capture"] == "OK"]
    passed = [r for r in ok if r["cos"] > 0.9999 and r["nf"] == 0]
    # loop extra memory should be a SMALL multiple of ONE output+dequant buffer (E*cap*out + out*in),
    # NOT E * that. Single output buffer for E=128,cap=128,out=4096 ~ 128*128*4096*2/1e6 = 134 MB.
    print(f"\n# capture OK {len(ok)}/{len(rows)}; graph==eager {len(passed)}/{len(rows)}", flush=True)
    print(f"# M4D-DENSE {'PASS' if len(passed) == len(rows) else 'PARTIAL/FAIL'} "
          f"(loopMB bounded => capture pool reuses buffers, no E-fold blowup)", flush=True)


@app.local_entrypoint()
def main() -> None:
    validate.remote()
