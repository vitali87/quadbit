"""P4 M3-C: expert-parallel (EP) graph safety + routing-capacity stability of the sparse-FP4 MoE path.

Proves the graph-safe apply works PER EP SHARD with off-rank slots sink-routed (so the shape stays static
despite a data-dependent on-rank count), and that summing the shards reconstructs the full MoE output.
Single process, single GPU, NO NCCL: each of `world` shards is simulated sequentially (its own local
experts + expert_map), and the cross-shard reduce is an in-process sum -- which is exactly what an
all-reduce computes. This isolates the two questions this rung owns (multi-rank plugin graph safety;
per-rank capacity stability) from the NCCL collective, which is deferred to M4 where vLLM's known-good EP
runs it. (A standalone torch.distributed collective hangs on Modal's mp.spawn bootstrap -- a container
transport issue, NOT a capture/SM120 issue: the FIRST plain-eager all_reduce hung, before any graph.)

Per shard compared (LOCAL contribution y[T,H]):
  * OLD eager : frozen EP both-branch (compact on-rank slots + build_routing + seg);
  * NEW eager : `_seg_apply_gs` with a valid mask (sink routing), run eager;
  * NEW graph : the NEW path captured as a CUDA graph and replayed.
Plus an EP-reduce check: sum of all shard outputs vs a single all-experts reference.
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
app = modal.App("quadbit-p4m3c", image=image)

H, I, EG, TOPK, T, WORLD = 2048, 1024, 8, 6, 4096, 2   # EG global experts, WORLD shards, EL=EG/WORLD


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
    EL = EG // WORLD
    print(f"# torch {torch.__version__} | {torch.cuda.get_device_name(0)} | cap {torch.cuda.get_device_capability(0)} "
          f"| EP world={WORLD} EG={EG} ({EL}/shard) T={T} topk={TOPK}", flush=True)

    # global model + inputs, all hoisted before any capture (post-capture randn trips philox offset)
    Wg = [torch.randn(I, H, device=dev) * (H ** -0.5) for _ in range(EG)]
    Wu = [torch.randn(I, H, device=dev) * (H ** -0.5) for _ in range(EG)]
    Wd = [torch.randn(H, I, device=dev) * (I ** -0.5) for _ in range(EG)]
    packG = [sp.pack(torch.cat([Wg[e], Wu[e]], 0)) for e in range(EG)]
    packD = [sp.pack(Wd[e]) for e in range(EG)]
    x = torch.randn(T, H, device=dev, dtype=torch.bfloat16) * 0.1
    gate = x.float() @ (torch.randn(H, EG, device=dev) * (H ** -0.5))
    tw, ti = torch.topk(gate, TOPK, dim=1)
    tw = torch.softmax(tw, dim=1)
    gids = ti.reshape(-1).to(torch.long)
    tok_of = torch.arange(T, device=dev).repeat_interleave(TOPK)
    w_of = tw.reshape(-1).to(torch.float32)
    cap = max(128, ((2 * T * TOPK // EG + 127) // 128) * 128)

    def cos(a, b):
        return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()

    def ms(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize(); ts = []
        for _ in range(50):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
        return statistics.median(sorted(ts))

    def old_eager(gu_W, dn_W, local, valid, el):
        a = local[valid]; t_ = tok_of[valid]; wv = w_of[valid]
        src, eblk, _r = sp.build_routing(a, el)
        vv = src >= 0; sc = src.clamp_min(0)
        xs = (x[t_[sc]].to(torch.bfloat16) * vv[:, None]).contiguous()
        gu = sp.seg_gemm(xs, gu_W, 2 * I, H, eblk)
        hh = (F.silu(gu[:, :I].float()) * gu[:, I:].float()).to(torch.bfloat16)
        d = sp.seg_gemm(hh, dn_W, H, I, eblk)
        y = torch.zeros(T, H, device=dev, dtype=torch.bfloat16)
        y.index_add_(0, t_[sc], (d.float() * (wv[sc] * vv.float())[:, None]).to(torch.bfloat16))
        return y

    def run_shard(tag, lo, el):
        gu_W = sp.stack([packG[e] for e in range(lo, lo + el)])
        dn_W = sp.stack([packD[e] for e in range(lo, lo + el)])
        valid = (gids >= lo) & (gids < lo + el)
        local = (gids - lo).clamp(0, el - 1)
        on_rank = int(valid.sum())
        cnt = torch.zeros(el, device=dev, dtype=torch.long).scatter_add_(
            0, local[valid], torch.ones(on_rank, device=dev, dtype=torch.long)).tolist()

        y_old = old_eager(gu_W, dn_W, local, valid, el)
        y = torch.zeros(T, H, device=dev, dtype=torch.bfloat16)

        def one_pass():
            y.zero_()
            return P._seg_apply_gs(sp, x, tok_of, local, w_of, gu_W, dn_W, I, H, el, cap, False, y, valid)

        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                one_pass()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        dropped = int(one_pass())
        y_new_eager = y.clone()

        res = {"tag": tag, "lo": lo, "el": el, "cap": cap, "on_rank": on_rank, "counts": cnt, "dropped": dropped}
        try:
            torch.cuda.reset_peak_memory_stats()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                one_pass()
            for _ in range(3):
                g.replay()
            torch.cuda.synchronize()
            y_new_graph = y.clone()
            res.update({"capture": "OK", "n_cos": cos(y_new_eager, y_old), "g_cos": cos(y_new_graph, y_new_eager),
                        "g_nf": int((~torch.isfinite(y_new_graph)).sum().item()),
                        "eager": ms(one_pass), "graph": ms(lambda: g.replay()),
                        "mem": torch.cuda.max_memory_allocated() / 1e6, "y": y_new_eager})
        except Exception as ex:  # noqa: BLE001
            res.update({"capture": f"FAIL: {type(ex).__name__}: {ex}", "y": None})
        return res

    rows = [run_shard(f"shard{s}", s * EL, EL) for s in range(WORLD)]
    full = run_shard("full-8", 0, EG)          # single all-experts reference (EL=EG)

    print(f"\n{'shard':>7} {'lo':>3} {'el':>3} {'cap':>6} {'onrank':>7} {'drop':>5} | {'counts':>22} | "
          f"{'NEWvsOLD':>9} {'GvsE':>9} {'nf':>3} | {'eagerms':>8} {'graphms':>8} {'memMB':>8} | capture", flush=True)
    for r in rows + [full]:
        if r["capture"] != "OK":
            print(f"{r['tag']:>7} {r['lo']:>3} {r['el']:>3} {r['cap']:>6} {r['on_rank']:>7} {r['dropped']:>5} | "
                  f"{str(r['counts']):>22} | {r['capture']}", flush=True)
            continue
        print(f"{r['tag']:>7} {r['lo']:>3} {r['el']:>3} {r['cap']:>6} {r['on_rank']:>7} {r['dropped']:>5} | "
              f"{str(r['counts']):>22} | {r['n_cos']:>9.6f} {r['g_cos']:>9.6f} {r['g_nf']:>3} | "
              f"{r['eager']:>8.3f} {r['graph']:>8.3f} {r['mem']:>8.1f} | OK", flush=True)

    # EP-reduce check: sum of shard outputs (in-process stand-in for the all-reduce) vs all-experts ref
    ep_cos = None
    if all(r["capture"] == "OK" for r in rows) and full["capture"] == "OK":
        ysum = sum(r["y"] for r in rows)
        ep_cos = cos(ysum, full["y"])
        print(f"\n# EP-reduce: cos(sum_shards, full-8) = {ep_cos:.6f} "
              f"(shards partition the {EG} experts; in-process sum stands in for all-reduce)", flush=True)

    ok = [r for r in rows + [full] if r["capture"] == "OK"]
    graph_ok = [r for r in ok if r["g_cos"] > 0.9999 and r["g_nf"] == 0]
    eq_ok = [r for r in ok if r["dropped"] > 0 or r["n_cos"] > 0.999]
    ep_ok = ep_cos is not None and ep_cos > 0.999
    print(f"\n# capture OK {len(ok)}/{len(rows) + 1}; graph==eager {len(graph_ok)}/{len(ok)}; "
          f"NEW==OLD (no-drop) {len(eq_ok)}/{len(ok)}; EP-reduce {'OK' if ep_ok else 'FAIL'}", flush=True)
    ap = len(ok) == len(rows) + 1 and len(graph_ok) == len(ok) and len(eq_ok) == len(ok) and ep_ok
    print(f"# M3-C {'PASS' if ap else 'PARTIAL/FAIL'}", flush=True)


@app.local_entrypoint()
def main() -> None:
    validate.remote()
