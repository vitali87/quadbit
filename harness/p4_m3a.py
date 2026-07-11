"""P4 M3-A: plugin-only in-process graph capture of the sparse-FP4 MoE seg path.

Unlike M2 (p4_capture.py, private harness copies of pack/quant/seg/route), M3-A imports the ACTUAL plugin
module `qb_sm120_plugin` and drives its real `_load_sparse_moe()` helpers -- so it proves the shipped
plugin functions are graph-capturable, in-process, with a fixed route and persistent workspace. No vLLM
scheduler yet. Three variants per shape, compared in TOKEN space (apples-to-apples):
  * OLD eager   : plugin build_routing + quant_act (alloc) + seg_gemm (stream 0, forced sync)  [frozen]
  * NEW eager   : plugin route_fixed_cap + quant_into + seg_into (current stream, preallocated) [M3]
  * NEW graph   : the NEW path captured as a CUDA graph and replayed.
Reports cos/relL2/maxabs/nonfinite for NEW-eager-vs-OLD and NEW-graph-vs-NEW-eager, capture status,
eager/graph latency, memory, and kernels-per-pass (graph collapses them to one host launch).
"""

import subprocess
import sys
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
    .add_local_dir((ROOT / "harness" / "qb_vllm_plugin").as_posix(), "/root/qb_vllm_plugin")
)
app = modal.App("quadbit-p4m3a", image=image)


@app.function(gpu="RTX-PRO-6000", timeout=1800)
def validate() -> None:
    import statistics

    import torch
    import torch.nn.functional as F

    # build the branch .so at the path the plugin ctypes-loads from
    Path("/cache").mkdir(exist_ok=True)
    so = "/cache/sparse_fp4.so"
    c = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
                        "-o", so, "/root/cuda/sparse_fp4_lib.cu", "-lcuda"], capture_output=True, text=True)
    if c.returncode != 0:
        print(c.stderr, flush=True)
        raise SystemExit(1)

    sys.path.insert(0, "/root/qb_vllm_plugin")
    import qb_sm120_plugin as P  # the real plugin module (top-level import is os-only, no vLLM)

    sp = P._load_sparse_moe()          # the ACTUAL plugin helpers
    dev = torch.device("cuda")
    torch.manual_seed(0)
    print(f"# torch {torch.__version__} | {torch.cuda.get_device_name(0)} | "
          f"cap {torch.cuda.get_device_capability(0)} | driver {torch.version.cuda}", flush=True)
    print(f"# plugin helpers: {[k for k in vars(sp) if not k.startswith('_')]}", flush=True)

    H, I, E, topk = 4096, 2048, 8, 6
    Wg = [torch.randn(I, H, device=dev) * (H ** -0.5) for _ in range(E)]
    Wu = [torch.randn(I, H, device=dev) * (H ** -0.5) for _ in range(E)]
    Wd = [torch.randn(H, I, device=dev) * (I ** -0.5) for _ in range(E)]
    gu_W = sp.stack([sp.pack(torch.cat([Wg[e], Wu[e]], 0)) for e in range(E)])
    dn_W = sp.stack([sp.pack(Wd[e]) for e in range(E)])
    ksH, ksI = H // 128, I // 128
    KPP = 4  # kernels per pass: 2 quantize + 2 seg-gemm; graph replay = 1 host launch

    def metrics(a, b):
        a, b = a.float(), b.float()
        return (F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
                (a - b).norm().item() / max(b.norm().item(), 1e-12),
                (a - b).abs().max().item(), int((~torch.isfinite(a)).sum().item()))

    def stats_ms(fn, it=100):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(it):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        return statistics.median(sorted(ts))

    def run_shape(tag, T, assign, cap):
        R = assign.shape[0]
        tok = (torch.arange(R, device=dev) % T)           # routed row -> token (fixed pattern)
        rw = torch.full((R,), 1.0 / topk, device=dev)     # fixed per-slot weight
        Xtok = torch.randn(T, H, device=dev, dtype=torch.bfloat16) * 0.1

        # ---- OLD eager: frozen plugin path (build_routing + quant_act + seg_gemm) ----
        src_o, eblk_o, _rp = sp.build_routing(assign, E)
        valid_o = src_o >= 0
        srcc_o = src_o.clamp_min(0)
        xs_o = (Xtok[tok[srcc_o]] * valid_o[:, None]).contiguous()
        gu_o = sp.seg_gemm(xs_o, gu_W, 2 * I, H, eblk_o)
        hh_o = (F.silu(gu_o[:, :I].float()) * gu_o[:, I:].float()).to(torch.bfloat16)
        d_o = sp.seg_gemm(hh_o, dn_W, H, I, eblk_o)
        y_old = torch.zeros(T, H, device=dev, dtype=torch.bfloat16)
        y_old.index_add_(0, tok[srcc_o], (d_o.float() * (valid_o.float() * rw[srcc_o])[:, None]).to(torch.bfloat16))

        # ---- NEW path: route_fixed_cap + preallocated + current-stream launches ----
        src, eblk, dropped = sp.route_fixed_cap(assign, E, cap)
        Rp = E * cap
        valid = src >= 0
        srcc = src.clamp_min(0)
        Xs = (Xtok[tok[srcc]] * valid[:, None]).contiguous()
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
            sp.quant_into(Xs, Bb, sB, gB)
            sp.seg_into(gu_W, Bb, sB, gB, GU, 2 * I, H, eblk)
            Hb.copy_((F.silu(GU[:, :I].float()) * GU[:, I:].float()).to(torch.bfloat16))
            sp.quant_into(Hb, Bb2, sB2, gB2)
            sp.seg_into(dn_W, Bb2, sB2, gB2, D, H, I, eblk)

        def scatter():
            y = torch.zeros(T, H, device=dev, dtype=torch.bfloat16)
            y.index_add_(0, tok[srcc], (D.float() * (valid.float() * rw[srcc])[:, None]).to(torch.bfloat16))
            return y

        # warm on a side stream, then eager
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                one_pass()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        one_pass()
        y_new_eager = scatter()
        eln = stats_ms(one_pass)

        res = {"tag": tag, "T": T, "Rp": Rp, "cap": cap, "dropped": dropped}
        try:
            torch.cuda.reset_peak_memory_stats()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                one_pass()
            for _ in range(3):
                g.replay()
            torch.cuda.synchronize()
            y_new_graph = scatter()
            # NEW-eager vs OLD (graph-safe path matches frozen eager) and NEW-graph vs NEW-eager (capture exact)
            c1 = metrics(y_new_eager, y_old)
            c2 = metrics(y_new_graph, y_new_eager)
            gln = stats_ms(lambda: g.replay())
            res.update({"capture": "OK", "n_cos": c1[0], "n_rel": c1[1], "n_max": c1[2], "n_nf": c1[3],
                        "g_cos": c2[0], "g_rel": c2[1], "g_nf": c2[3], "eager": eln, "graph": gln,
                        "mem": torch.cuda.max_memory_allocated() / 1e6})
        except Exception as ex:  # noqa: BLE001
            res.update({"capture": f"FAIL: {type(ex).__name__}: {ex}"})
        return res

    def rup(x):
        return ((x + 128 - 1) // 128) * 128

    rows = []
    for kind, T in [("decode-B1", 1), ("decode-B8", 8), ("decode-B32", 32), ("decode-B64", 64),
                    ("prefill-2048", 2048), ("prefill-8192", 8192)]:
        R = T * topk
        g = torch.Generator(device=dev).manual_seed(T)
        assign = torch.randint(0, E, (R,), device=dev, generator=g)
        rows.append(run_shape(kind, T, assign, max(128, rup(2 * R // E))))

    # pathological single-rank routes
    Tp, Rp0 = 4096, 4096 * topk
    capn = max(128, rup(2 * Rp0 // E))
    for tag, assign in [("all->expert0", torch.zeros(Rp0, dtype=torch.long, device=dev)),
                        ("empty-experts", torch.randint(0, 2, (Rp0,), device=dev,
                                                        generator=torch.Generator(device=dev).manual_seed(8))),
                        ("near-capacity", torch.arange(Rp0, device=dev) % E)]:
        rows.append(run_shape(tag, Tp, assign, capn))

    print(f"\n# kernels/pass={KPP} (graph replay = 1 host launch)", flush=True)
    print(f"{'tag':>16} {'T':>6} {'Rp':>7} {'drop':>6} | {'NEWvsOLD cos':>12} {'rel':>9} {'nf':>3} | "
          f"{'GvsE cos':>9} {'nf':>3} | {'eagerms':>8} {'graphms':>8} {'memMB':>8} | capture", flush=True)
    for r in rows:
        if r["capture"] != "OK":
            print(f"{r['tag']:>16} {r['T']:>6} {r['Rp']:>7} {r['dropped']:>6} | {r['capture']}", flush=True)
            continue
        print(f"{r['tag']:>16} {r['T']:>6} {r['Rp']:>7} {r['dropped']:>6} | {r['n_cos']:>12.6f} "
              f"{r['n_rel']:>9.2e} {r['n_nf']:>3} | {r['g_cos']:>9.6f} {r['g_nf']:>3} | "
              f"{r['eager']:>8.3f} {r['graph']:>8.3f} {r['mem']:>8.1f} | {r['capture']}", flush=True)
    ok = [r for r in rows if r["capture"] == "OK"]
    # graph==eager must hold for ALL captured shapes; NEW==OLD only when nothing was dropped (overflow
    # drop is a deliberate NEW semantic, so drop>0 shapes are expected to diverge from build_routing).
    graph_ok = [r for r in ok if r["g_cos"] > 0.9999 and r["g_nf"] == 0]
    nodrop = [r for r in ok if r["dropped"] == 0]
    eq_ok = [r for r in nodrop if r["n_cos"] > 0.999 and r["n_nf"] == 0]
    print(f"\n# capture OK {len(ok)}/{len(rows)}; graph==eager {len(graph_ok)}/{len(ok)}; "
          f"NEW==OLD (no-drop) {len(eq_ok)}/{len(nodrop)}", flush=True)
    ap = len(ok) == len(rows) and len(graph_ok) == len(ok) and len(eq_ok) == len(nodrop)
    print(f"# M3-A {'PASS' if ap else 'PARTIAL/FAIL'}", flush=True)


@app.local_entrypoint()
def main() -> None:
    validate.remote()
