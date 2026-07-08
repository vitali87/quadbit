"""M3.7: distributed (expert-parallel) sparse-FP4 MoE across multiple RTX-PRO-6000 (no NVLink -> PCIe).

Shards the 256 experts across N ranks (expert parallelism); each rank runs the validated segmented
routed-row kernel (matmul_sp_moe) over the tokens routed to ITS local experts, then an all_reduce combines
the per-rank expert contributions into the full MoE output. Measures per-rank expert-kernel time vs
communication time vs routing imbalance across world sizes 1/2/4 -- the external-validity result the paper
needs (on no-NVLink PCIe, communication may dominate; that is a finding, not a failure).

Correctness: rank-0 combined output vs a single-process all-experts reference (cos ~1.0).
DeepSeek-V4-Flash expert shapes (H=4096, I=2048, 256 experts, top-6). Synthetic weights (mechanics+timing).
"""

import os
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
BN = 128
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64"})
    .pip_install("torch", index_url="https://download.pytorch.org/whl/nightly/cu128", pre=True)
    .pip_install("numpy")
    .add_local_dir((ROOT / "cuda").as_posix(), "/root/cuda")
)
app = modal.App("quadbit-moedist", image=image)


def _worker(rank, world, H, I, E, T, topk, so, retq):
    import ctypes

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
    torch.manual_seed(0)  # identical weights/routing on every rank (deterministic)

    lib = ctypes.CDLL(so)
    lib.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    lib.sparse_moe_mm_2lvl.argtypes = ([ctypes.c_void_p] * 6 + [ctypes.c_int] * 4 +
                                       [ctypes.c_void_p] * 3 + [ctypes.c_int] + [ctypes.c_void_p])
    lib.qb_init_moe_attrs()

    FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
    BND = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], device=dev)
    _cc = torch.arange(128, device=dev); _e, _m = (_cc >> 3) & 0xf, _cc & 7
    UE4M3 = torch.where(_e == 0, _m.float() * 0.001953125, (1.0 + _m.float() / 8.0) * torch.exp2((_e - 7).float()))

    def q_fp4(v):
        return torch.bucketize(v.abs(), BND) | ((v < 0).long() << 3)

    def enc(s):
        mf, e = torch.frexp(s.clamp_min(1e-30)); mm = 2.0 * mf
        b = (e - 1) + 7; mant = torch.round((mm - 1.0) * 8.0).long(); carry = mant == 8
        mant = torch.where(carry, torch.zeros_like(mant), mant); b = torch.where(carry, b + 1, b)
        code = (b.long() << 3) | mant
        code = torch.where(b < 1, torch.ones_like(code), code)
        code = torch.where(b > 15, torch.full_like(code, 0x7f), code)
        return torch.where(s > 0, torch.where(s >= 480.0, torch.full_like(code, 0x7f), code), torch.zeros_like(code))

    def pack(W):
        out_f, in_f = W.shape; ks = in_f // 128
        Wg = W.float().view(out_f, ks, 16, 4, 2)
        i01, _ = Wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
        kept = torch.gather(Wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
        gA = (kept.abs().amax(dim=(1, 2, 3, 4), keepdim=True) / 2688.0).clamp_min(1e-30).reshape(out_f, 1, 1)
        blk = kept.reshape(out_f, ks, 4, 8, 2)
        sc = enc((blk.abs().amax(dim=(3, 4)) / 6.0) / gA); sdeq = UE4M3[sc] * gA
        kc = q_fp4(blk / sdeq.clamp_min(1e-30)[..., None, None])
        Ac = (kc[..., 0] | (kc[..., 1] << 4)).reshape(out_f, ks * 32).to(torch.uint8)
        nib = (i01[..., 0] | (i01[..., 1] << 2)).view(out_f, ks, 2, 8)
        sh = (torch.arange(8, device=dev) * 4).view(1, 1, 1, 8)
        meta = (nib << sh).sum(-1).to(torch.int32).permute(1, 0, 2).contiguous()
        return (Ac.contiguous(), meta, sc.to(torch.uint8).permute(1, 0, 2).contiguous(), gA.reshape(out_f).float().contiguous())

    def stack(ps):
        return (torch.cat([p[0] for p in ps], 0).contiguous(), torch.cat([p[1] for p in ps], 1).contiguous(),
                torch.cat([p[2] for p in ps], 1).contiguous(), torch.cat([p[3] for p in ps], 0).contiguous())

    def quant(x):
        R, in_f = x.shape; ks = in_f // 128
        x = x.to(torch.bfloat16).contiguous()
        Bb = torch.empty((R, in_f // 2), dtype=torch.uint8, device=dev)
        sB = torch.empty((ks, R, 4), dtype=torch.uint8, device=dev); gB = torch.empty((R,), dtype=torch.float32, device=dev)
        lib.quantize_act_nvfp4_2lvl(x.data_ptr(), Bb.data_ptr(), sB.data_ptr(), gB.data_ptr(), R, in_f)
        return Bb, sB, gB

    def seg(x, W, Mpe, in_f, eblk):
        R = x.shape[0]; Ac, meta, scaleA, gA = W
        Bb, sB, gB = quant(x)
        C = torch.empty((R, Mpe), dtype=torch.bfloat16, device=dev)
        lib.sparse_moe_mm_2lvl(Ac.data_ptr(), Bb.data_ptr(), scaleA.data_ptr(), sB.data_ptr(), meta.data_ptr(),
                               C.data_ptr(), Ac.shape[0], Mpe, R, in_f, gA.data_ptr(), gB.data_ptr(),
                               eblk.data_ptr(), 1, torch.cuda.current_stream().cuda_stream)
        return C

    def build_routing(assign, Elocal, local_ids):  # assign restricted to local experts -> local eblk
        loc = {g: i for i, g in enumerate(local_ids)}
        mask = torch.tensor([int(a.item()) in loc for a in assign], device=dev)
        la = torch.tensor([loc.get(int(a.item()), 0) for a in assign], device=dev)
        order = torch.argsort(torch.where(mask, la, torch.full_like(la, 10**9)), stable=True)
        cnt = torch.bincount(la[mask], minlength=Elocal)
        padc = (cnt + BN - 1) // BN * BN; R_pad = int(padc.sum().item())
        src = torch.full((R_pad,), -1, dtype=torch.long, device=dev)
        eblk = torch.zeros(max(R_pad // BN, 1), dtype=torch.int32, device=dev)
        off = 0; oi = 0; nlocal = int(mask.sum().item())
        srt = order[:nlocal]
        for e in range(Elocal):
            ce = int(cnt[e].item()); pe = int(padc[e].item())
            if pe == 0:
                continue
            src[off:off + ce] = srt[oi:oi + ce]; oi += ce
            eblk[off // BN:(off + pe) // BN] = e; off += pe
        return src, eblk, R_pad

    # weights: all E experts (seeded identically); each rank keeps only its slice
    Ke = E // world; local_ids = list(range(rank * Ke, (rank + 1) * Ke))
    gu, dn = [], []
    for e in range(E):
        g = torch.randn(I, H, device=dev) * (H ** -0.5); u = torch.randn(I, H, device=dev) * (H ** -0.5)
        d = torch.randn(H, I, device=dev) * (I ** -0.5)
        if e in local_ids:
            gu.append(pack(torch.cat([g, u], 0))); dn.append(pack(d))
    gu_W, dn_W = stack(gu), stack(dn)

    X = torch.randn(T, H, device=dev, dtype=torch.bfloat16) * (H ** -0.5) * 4
    logits = torch.randn(T, E, device=dev)
    tw, tidx = logits.softmax(-1).topk(topk, dim=-1)
    tw = tw / tw.sum(-1, keepdim=True) * 1.5
    assign = tidx.reshape(-1); tok_of = torch.arange(T, device=dev).repeat_interleave(topk); w_of = tw.reshape(-1)

    src, eblk, R_pad = build_routing(assign, Ke, local_ids)
    valid = src >= 0; srcc = src.clamp_min(0)
    Xs = X[tok_of[srcc]] * valid[:, None]

    def compute_local():
        GU = seg(Xs, gu_W, 2 * I, H, eblk)
        Hh = (F.silu(GU[:, :I].float()) * GU[:, I:].float()).to(torch.bfloat16)
        D = seg(Hh, dn_W, H, I, eblk)
        y = torch.zeros(T, H, device=dev)
        y.index_add_(0, tok_of[srcc][valid], D[valid].float() * w_of[srcc][valid, None])
        return y

    def timed(fn, iters=20):
        torch.cuda.synchronize(); dist.barrier()
        st = torch.cuda.Event(True); en = torch.cuda.Event(True); st.record()
        for _ in range(iters):
            fn()
        en.record(); torch.cuda.synchronize()
        return st.elapsed_time(en) / iters

    y_local = compute_local()
    t_comp = timed(compute_local)
    buf = y_local.clone()
    t_comm = timed(lambda: torch.distributed.all_reduce(buf))
    torch.distributed.all_reduce(y_local)  # final combined output

    rows_local = int(valid.sum().item())
    stats = torch.tensor([rows_local, float(t_comp), float(t_comm)], device=dev)
    allrows = [torch.zeros_like(stats) for _ in range(world)]
    dist.all_gather(allrows, stats)
    if rank == 0:
        rl = [int(a[0].item()) for a in allrows]
        comp = max(a[1].item() for a in allrows); comm = allrows[0][2].item()
        imbal = max(rl) / (sum(rl) / len(rl)) if sum(rl) else 1.0
        retq.put({"world": world, "rows_per_rank": rl, "routing_imbalance": round(imbal, 3),
                  "expert_kernel_ms": round(comp, 3), "allreduce_comm_ms": round(comm, 3),
                  "y_checksum": round(float(y_local.float().abs().mean().item()), 6)})
    dist.destroy_process_group()


@app.function(gpu="RTX-PRO-6000:4", timeout=3600)
def run(H: int = 4096, I: int = 2048, E: int = 256, T: int = 256, topk: int = 6) -> None:
    import torch.multiprocessing as mp

    so = "/root/sparse_fp4.so"
    c = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
                        "-o", so, "/root/cuda/sparse_fp4_lib.cu", "-lcuda"], capture_output=True, text=True)
    if c.returncode != 0:
        print(c.stderr, flush=True)
        raise SystemExit(1)
    print("# M3.7 distributed expert-parallel sparse-FP4 MoE (RTX-PRO-6000, PCIe/no-NVLink)", flush=True)
    print(f"# DeepSeek-V4-Flash shapes: H={H} I={I} E={E} top-{topk}, T={T} tokens", flush=True)
    ref = None
    for world in (1, 2, 4):
        ctx = mp.get_context("spawn"); q = ctx.Queue()
        procs = [ctx.Process(target=_worker, args=(r, world, H, I, E, T, topk, so, q)) for r in range(world)]
        for p in procs:
            p.start()
        res = q.get()
        for p in procs:
            p.join()
        if ref is None:
            ref = res["y_checksum"]; comp1 = res["expert_kernel_ms"]
        ok = abs(res["y_checksum"] - ref) < 1e-4
        speedup = comp1 / res["expert_kernel_ms"]
        print(f"  world={world}: rows/rank={res['rows_per_rank']} imbalance={res['routing_imbalance']}  "
              f"kernel_speedup_vs_1gpu={speedup:.2f}x  "
              f"expert_kernel={res['expert_kernel_ms']}ms  all_reduce_comm={res['allreduce_comm_ms']}ms  "
              f"checksum={res['y_checksum']} {'OK' if ok else 'MISMATCH'}", flush=True)
    print("# M3.7 done (expert-parallel sparse MoE runs across ranks; comm vs compute split reported)", flush=True)


@app.local_entrypoint()
def main() -> None:
    run.remote()
