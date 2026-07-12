"""P4 M3-B: one real MoE layer through the plugin's graph-safe apply.

Builds a real MoE layer (E experts, a real gate -> topk router with softmax weights) and drives it three
ways per routing regime, comparing the full per-token output y[T,H]:
  * OLD eager  : the frozen plugin `both`-branch math (build_routing + seg_gemm), driven by the real router;
  * NEW eager  : plugin `_seg_apply_gs` (route_fixed_cap + quant_into/seg_into), run eager;
  * NEW graph  : `_seg_apply_gs` captured as a CUDA graph and replayed.
Regimes: single-expert (topk=1, all->e0, overflow), uniform, empty-experts (only 4 reachable),
near-capacity (tight cap), overflow (cap < mean load). Reports NEWvsOLD / GraphvsEager cos, drop count,
capture status, eager/graph latency, memory. Single rank (identity expert_map); EP off-rank is M3-C.
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
app = modal.App("quadbit-p4m3b", image=image)


@app.function(gpu="RTX-PRO-6000", timeout=1800)
def validate() -> None:
    import statistics

    import torch
    import torch.nn.functional as F

    Path("/cache").mkdir(exist_ok=True)
    so = "/cache/sparse_fp4.so"
    c = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
                        "-o", so, "/root/cuda/sparse_fp4_lib.cu", "-lcuda"], capture_output=True, text=True)
    if c.returncode != 0:
        print(c.stderr, flush=True)
        raise SystemExit(1)
    sys.path.insert(0, "/root/qb_vllm_plugin")
    import qb_sm120_plugin as P

    sp = P._load_sparse_moe()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    print(f"# torch {torch.__version__} | {torch.cuda.get_device_name(0)} | "
          f"cap {torch.cuda.get_device_capability(0)} | driver {torch.version.cuda}", flush=True)

    H, I, E = 4096, 2048, 8
    Wg = [torch.randn(I, H, device=dev) * (H ** -0.5) for _ in range(E)]
    Wu = [torch.randn(I, H, device=dev) * (H ** -0.5) for _ in range(E)]
    Wd = [torch.randn(H, I, device=dev) * (I ** -0.5) for _ in range(E)]
    gu_W = sp.stack([sp.pack(torch.cat([Wg[e], Wu[e]], 0)) for e in range(E)])
    dn_W = sp.stack([sp.pack(Wd[e]) for e in range(E)])

    def metrics(a, b):
        a, b = a.float(), b.float()
        return (F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
                (a - b).norm().item() / max(b.norm().item(), 1e-12),
                int((~torch.isfinite(a)).sum().item()))

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

    def old_eager(x, tok_of, assign, w_of, on_input):
        # frozen `both`-branch math driven by the real router (the reference)
        src, eblk, _r = sp.build_routing(assign, E)
        valid = src >= 0
        srcc = src.clamp_min(0)
        xs = (x[tok_of[srcc]].to(torch.bfloat16) * valid[:, None])
        if on_input:
            xs = xs * w_of[srcc][:, None].to(torch.bfloat16)
        xs = xs.contiguous()
        gu = sp.seg_gemm(xs, gu_W, 2 * I, H, eblk)
        hh = (F.silu(gu[:, :I].float()) * gu[:, I:].float()).to(torch.bfloat16)
        d = sp.seg_gemm(hh, dn_W, H, I, eblk)
        y = torch.zeros(x.shape[0], H, device=dev, dtype=torch.bfloat16)
        rw = valid.float() if on_input else (w_of[srcc] * valid.float())
        y.index_add_(0, tok_of[srcc], (d.float() * rw[:, None]).to(torch.bfloat16))
        return y

    def rup(v):
        return ((v + 128 - 1) // 128) * 128

    # Hoist ALL randomness before any capture: torch.randn AFTER a graph capture trips the philox
    # offset bookkeeping ("Offset increment outside graph capture"). x and the gate projection are
    # fixed across cases; regimes vary only gate_bias / topk / cap.
    T = 4096
    x0 = torch.randn(T, H, device=dev, dtype=torch.bfloat16) * 0.1
    Wgate = torch.randn(H, E, device=dev) * (H ** -0.5)
    gate0 = x0.float() @ Wgate

    def run(tag, T, gate_bias, topk, cap, on_input=False):
        x = x0
        gate = gate0 + gate_bias
        tw, ti = torch.topk(gate, topk, dim=1)
        tw = torch.softmax(tw, dim=1)
        assign = ti.reshape(-1).to(torch.long)
        tok_of = torch.arange(T, device=dev).repeat_interleave(topk)
        w_of = tw.reshape(-1).to(torch.float32)

        y_old = old_eager(x, tok_of, assign, w_of, on_input)

        y = torch.zeros(T, H, device=dev, dtype=torch.bfloat16)

        def one_pass():
            y.zero_()
            return P._seg_apply_gs(sp, x, tok_of, assign, w_of, gu_W, dn_W, I, H, E, cap, on_input, y)

        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                one_pass()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        dropped = int(one_pass())            # device scalar -> host int (outside capture)
        y_new_eager = y.clone()
        eln = stats_ms(one_pass)

        res = {"tag": tag, "T": T, "topk": topk, "cap": cap, "Rp": E * cap, "dropped": dropped}
        try:
            torch.cuda.reset_peak_memory_stats()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                one_pass()
            for _ in range(3):
                g.replay()
            torch.cuda.synchronize()
            y_new_graph = y.clone()
            c1 = metrics(y_new_eager, y_old)
            c2 = metrics(y_new_graph, y_new_eager)
            gln = stats_ms(lambda: g.replay())
            res.update({"capture": "OK", "n_cos": c1[0], "n_rel": c1[1], "n_nf": c1[2],
                        "g_cos": c2[0], "g_nf": c2[2], "eager": eln, "graph": gln,
                        "mem": torch.cuda.max_memory_allocated() / 1e6})
        except Exception as ex:  # noqa: BLE001
            res.update({"capture": f"FAIL: {type(ex).__name__}: {ex}"})
        return res

    T = 4096
    mean = rup(2 * T * 6 // E)            # topk=6 uniform mean load, 2x headroom
    e0bias = torch.zeros(E, device=dev); e0bias[0] = 1e4
    only4 = torch.zeros(E, device=dev); only4[4:] = -1e4     # experts 4..7 unreachable
    cases = [
        ("single-e0", T, e0bias, 1, max(128, rup(T // 4))),          # all top-1 -> e0, overflow drop
        ("uniform", T, torch.zeros(E, device=dev), 6, mean),         # no drop
        ("empty-experts", T, only4, 6, mean),                        # only 4 experts busy
        ("near-capacity", T, torch.zeros(E, device=dev), 6, rup(int(1.1 * 2 * T * 6 / E))),
        ("overflow", T, torch.zeros(E, device=dev), 6, max(128, rup(2 * T * 6 // E // 3))),  # cap << load
        ("uniform-oninput", T, torch.zeros(E, device=dev), 6, mean),
    ]
    rows = [run(*a[:5], on_input=(a[0].endswith("oninput"))) for a in cases]

    print(f"\n{'tag':>16} {'T':>5} {'tk':>3} {'cap':>6} {'drop':>6} | {'NEWvsOLD':>9} {'rel':>9} {'nf':>3} | "
          f"{'GvsE':>9} {'nf':>3} | {'eagerms':>8} {'graphms':>8} {'memMB':>8} | capture", flush=True)
    for r in rows:
        if r["capture"] != "OK":
            print(f"{r['tag']:>16} {r['T']:>5} {r['topk']:>3} {r['cap']:>6} {r['dropped']:>6} | {r['capture']}", flush=True)
            continue
        print(f"{r['tag']:>16} {r['T']:>5} {r['topk']:>3} {r['cap']:>6} {r['dropped']:>6} | "
              f"{r['n_cos']:>9.6f} {r['n_rel']:>9.2e} {r['n_nf']:>3} | {r['g_cos']:>9.6f} {r['g_nf']:>3} | "
              f"{r['eager']:>8.3f} {r['graph']:>8.3f} {r['mem']:>8.1f} | OK", flush=True)
    ok = [r for r in rows if r["capture"] == "OK"]
    graph_ok = [r for r in ok if r["g_cos"] > 0.9999 and r["g_nf"] == 0]
    nodrop = [r for r in ok if r["dropped"] == 0]
    eq_ok = [r for r in nodrop if r["n_cos"] > 0.999 and r["n_nf"] == 0]
    print(f"\n# capture OK {len(ok)}/{len(rows)}; graph==eager {len(graph_ok)}/{len(ok)}; "
          f"NEW==OLD (no-drop) {len(eq_ok)}/{len(nodrop)}", flush=True)
    ap = len(ok) == len(rows) and len(graph_ok) == len(ok) and len(eq_ok) == len(nodrop)
    print(f"# M3-B {'PASS' if ap else 'PARTIAL/FAIL'}", flush=True)


@app.local_entrypoint()
def main() -> None:
    validate.remote()
