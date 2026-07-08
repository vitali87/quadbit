"""M3.2/M3.3: segmented routed-row sparse-FP4 MoE operator -- synthetic correctness.

ONE graph-capturable launch per GEMM: routed rows are sorted by expert and padded so every BN(=128)
column block is single-expert; eblk[block]=expert. The kernel (matmul_sp_moe) offsets weight-side
indices by expert*Mpe into STACKED expert weights. Scheduling unit = routed row, NOT expert -> no
per-expert launch, fixed grid, capturable.

Path: quant(X_sorted) -> gate/up seg-GEMM [R,2I] -> SwiGLU+quant -> down seg-GEMM [R,H] -> combine
(expert_out[token,slot,H] then weighted sum + shared expert). Validated vs the per-expert reference
(the M1 driver) under pathological routing, then graph capture+replay.
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
app = modal.App("quadbit-moeseg", image=image)

BN = 128  # kernel token-tile; each padded col-block is single-expert and a multiple of BN


@app.function(gpu="RTX-PRO-6000", timeout=1800)
def validate() -> None:
    import ctypes

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
    lib.sparse_moe_mm_2lvl.argtypes = ([ctypes.c_void_p] * 6 + [ctypes.c_int] * 4 +
                                       [ctypes.c_void_p] * 3 + [ctypes.c_int] + [ctypes.c_void_p])
    lib.sparse_moe_mm_2lvl.restype = ctypes.c_int
    lib.sparse_fp4_mm_2lvl_t.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    lib.sparse_fp4_mm_2lvl_t.restype = ctypes.c_int
    lib.qb_init_moe_attrs()
    dev = torch.device("cuda")
    torch.manual_seed(0)

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

    def sparse_fp4_dequant(W):
        out_f, in_f = W.shape; ks = in_f // 128
        Wg = W.view(out_f, ks, 16, 4, 2)
        i01, _ = Wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
        keptW = torch.gather(Wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
        gA = (keptW.abs().amax(dim=(1, 2, 3, 4), keepdim=True) / 2688.0).clamp_min(1e-30).reshape(out_f, 1, 1)
        blk = keptW.reshape(out_f, ks, 4, 8, 2)
        sdeq = UE4M3[enc_ue4m3_t((blk.abs().amax(dim=(3, 4)) / 6.0) / gA)] * gA
        kd = (FP4[q_fp4(blk / sdeq.clamp_min(1e-30)[..., None, None])] * sdeq[..., None, None]).reshape(out_f, ks, 16, 2, 2)
        Wd = torch.zeros(out_f, ks, 16, 4, 2, device=dev)
        Wd.scatter_(3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2), kd)
        return Wd.reshape(out_f, in_f)

    def act_fp4_dequant(x):
        lead = x.shape[:-1]; i = x.shape[-1]
        b = x.to(torch.bfloat16).float().reshape(-1, i)
        gB = (b.abs().amax(-1, keepdim=True) / 2688.0).clamp_min(1e-30)
        bb = b.reshape(b.shape[0], i // 32, 32)
        sdeq = UE4M3[enc_ue4m3_t((bb.abs().amax(-1) / 6.0) / gB)] * gB
        return (FP4[q_fp4(bb / sdeq.clamp_min(1e-30)[..., None])] * sdeq[..., None]).reshape(*lead, i)

    def pack(W):  # bf16 [out,in] -> (Ac[out,ks*32], meta[ks,out,2], scaleA[ks,out,4], gA[out]) (M1-validated)
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
        return (Ac.contiguous(), meta, scode.to(torch.uint8).permute(1, 0, 2).contiguous(), gA.reshape(out_f).float().contiguous())

    def stack(packs):  # list of per-expert packs -> stacked contiguous buffers over Mtot=E*out
        Ac = torch.cat([p[0] for p in packs], 0).contiguous()          # [E*out, ks*32]
        meta = torch.cat([p[1] for p in packs], 1).contiguous()        # [ks, E*out, 2]
        scaleA = torch.cat([p[2] for p in packs], 1).contiguous()      # [ks, E*out, 4]
        gA = torch.cat([p[3] for p in packs], 0).contiguous()          # [E*out]
        return Ac, meta, scaleA, gA

    def quant_act(x):  # bf16 [R,in] (R%128==0) -> (Bb, sB, gB) via the kernel quantizer
        R, in_f = x.shape; ks = in_f // 128
        x = x.to(torch.bfloat16).contiguous()
        Bb = torch.empty((R, in_f // 2), dtype=torch.uint8, device=dev)
        sB = torch.empty((ks, R, 4), dtype=torch.uint8, device=dev)
        gB = torch.empty((R,), dtype=torch.float32, device=dev)
        lib.quantize_act_nvfp4_2lvl(x.data_ptr(), Bb.data_ptr(), sB.data_ptr(), gB.data_ptr(), R, in_f)
        return Bb, sB, gB

    def seg_gemm(Bb, sB, gB, W, R, Mpe, in_f, eblk, C):  # W=(Ac,meta,scaleA,gA) stacked; C=[R,Mpe] out
        Ac, meta, scaleA, gA = W; Mtot = Ac.shape[0]
        lib.sparse_moe_mm_2lvl(Ac.data_ptr(), Bb.data_ptr(), scaleA.data_ptr(), sB.data_ptr(), meta.data_ptr(),
                               C.data_ptr(), Mtot, Mpe, R, in_f, gA.data_ptr(), gB.data_ptr(),
                               eblk.data_ptr(), 1, 0)

    def build_routing(assign, E):  # assign: [R] expert per routed row -> sorted order, padded, eblk
        order = torch.argsort(assign, stable=True)
        counts = torch.bincount(assign, minlength=E)
        padc = ((counts + BN - 1) // BN * BN)  # pad each expert segment to a multiple of BN
        R_pad = int(padc.sum().item())
        sorted_src = torch.full((R_pad,), -1, dtype=torch.long, device=dev)  # -1 = padding row
        eblk = torch.empty(R_pad // BN, dtype=torch.int32, device=dev)
        off = 0; oi = 0
        for e in range(E):
            ce = int(counts[e].item()); pe = int(padc[e].item())
            if pe == 0:
                continue
            seg = order[oi:oi + ce]; oi += ce
            sorted_src[off:off + ce] = seg
            eblk[off // BN:(off + pe) // BN] = e
            off += pe
        return sorted_src, eblk, R_pad

    def cos(a, b):
        return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()

    def expert_mlp_ref(x, Wg, Wu, Wd):  # fakequant reference for one expert on rows x (informational)
        xq = act_fp4_dequant(x.float())
        g = F.linear(xq, sparse_fp4_dequant(Wg.float())); u = F.linear(xq, sparse_fp4_dequant(Wu.float()))
        h = act_fp4_dequant(F.silu(g) * u)
        return F.linear(h, sparse_fp4_dequant(Wd.float()))

    def one_lin(x, packe, M, K):  # single-expert KERNEL GEMM (outT=1 -> [n,M]); pads rows to 128
        Ac, meta, scaleA, gA = packe
        n = x.shape[0]; padn = (-n) % 128
        xp = torch.cat([x, x.new_zeros(padn, K)], 0).to(torch.bfloat16).contiguous() if padn else x.to(torch.bfloat16).contiguous()
        Bb, sB, gB = quant_act(xp)
        C = torch.empty((n + padn, M), dtype=torch.bfloat16, device=dev)
        lib.sparse_fp4_mm_2lvl_t(Ac.data_ptr(), Bb.data_ptr(), scaleA.data_ptr(), sB.data_ptr(), meta.data_ptr(),
                                 C.data_ptr(), M, n + padn, K, gA.data_ptr(), gB.data_ptr())
        return C[:n]

    def expert_mlp_kernel(x, gu_pack, dn_pack, I, H):  # per-expert KERNEL reference (the M1 path)
        GU = one_lin(x, gu_pack, 2 * I, H)
        h = (F.silu(GU[:, :I].float()) * GU[:, I:].float()).to(torch.bfloat16)
        return one_lin(h, dn_pack, H, I)

    print("# M3.2 segmented routed-row sparse-FP4 MoE correctness (RTX-PRO-6000)", flush=True)
    for name, (H, I, E, T, k) in [("tiny", (512, 256, 8, 384, 2)), ("deepseek-shape", (4096, 2048, 8, 256, 6))]:
        # per-expert weights
        Wg = [torch.randn(I, H, device=dev) * (H ** -0.5) for _ in range(E)]
        Wu = [torch.randn(I, H, device=dev) * (H ** -0.5) for _ in range(E)]
        Wd = [torch.randn(H, I, device=dev) * (I ** -0.5) for _ in range(E)]
        # stacked packs: gate/up concatenated on out dim (2I), down (H)
        gu_packs = [pack(torch.cat([Wg[e], Wu[e]], 0)) for e in range(E)]
        dn_packs = [pack(Wd[e]) for e in range(E)]
        gu_W = stack(gu_packs); dn_W = stack(dn_packs)

        def routings():
            g = torch.Generator(device=dev).manual_seed(1)
            uni = torch.randint(0, E, (T * k,), device=dev, generator=g)
            yield "uniform", uni
            yield "all->expert0", torch.zeros(T * k, dtype=torch.long, device=dev)
            yield "one-per-expert", torch.arange(E, device=dev).repeat_interleave(max(1, (T * k) // E))[:T * k]
            imbal = torch.cat([torch.zeros(T * k - E + 1, dtype=torch.long, device=dev),
                               torch.arange(1, E, device=dev)])
            yield "imbalanced", imbal

        allok = True
        for rname, assign in routings():
            R = assign.shape[0]
            Xr = torch.randn(R, H, device=dev, dtype=torch.bfloat16) * 0.1  # one activation per routed row
            src, eblk, R_pad = build_routing(assign, E)
            valid = src >= 0; srcc = src.clamp_min(0)
            Xs = Xr[srcc] * valid[:, None]
            Bb, sB, gB = quant_act(Xs)
            GU = torch.empty((R_pad, 2 * I), dtype=torch.bfloat16, device=dev)
            seg_gemm(Bb, sB, gB, gu_W, R_pad, 2 * I, H, eblk, GU)
            g, u = GU[:, :I], GU[:, I:]
            Hh = (F.silu(g.float()) * u.float()).to(torch.bfloat16)
            Bb2, sB2, gB2 = quant_act(Hh)
            D = torch.empty((R_pad, H), dtype=torch.bfloat16, device=dev)
            seg_gemm(Bb2, sB2, gB2, dn_W, R_pad, H, I, eblk, D)

            # references on the SAME sorted rows: per-expert KERNEL (tight gate) + fakequant (informational)
            Dkref = torch.zeros(R_pad, H, device=dev); Dfref = torch.zeros(R_pad, H, device=dev)
            for e in range(E):
                rows = ((eblk == e).repeat_interleave(BN)) & valid
                if rows.any():
                    Dkref[rows] = expert_mlp_kernel(Xs[rows], gu_packs[e], dn_packs[e], I, H).float()
                    Dfref[rows] = expert_mlp_ref(Xs[rows], Wg[e], Wu[e], Wd[e])
            m = valid  # compare only real (non-pad) rows
            cK = cos(D[m], Dkref[m]); cF = cos(D[m], Dfref[m])
            nonfin = int((~torch.isfinite(D[m])).sum().item())
            ok = cK > 0.9999 and nonfin == 0
            allok = allok and ok
            print(f"  [{name:14s} {rname:16s}] R={R} R_pad={R_pad}  cos(seg,per-expert-kernel)={cK:.6f}  "
                  f"cos(seg,fakequant)={cF:.5f}  nonfin={nonfin}  -> {'PASS' if ok else 'FAIL'}", flush=True)

        # graph capture + replay on a fixed routing -- ALL buffers preallocated, NO allocation in captured region
        g = torch.Generator(device=dev).manual_seed(2)
        assign = torch.randint(0, E, (T * k,), device=dev, generator=g)
        src, eblk, R_pad = build_routing(assign, E)
        valid = src >= 0
        Xr = (torch.randn(R_pad, H, device=dev, dtype=torch.bfloat16) * 0.1).contiguous()
        ksH, ksI = H // 128, I // 128
        Bb = torch.empty((R_pad, H // 2), dtype=torch.uint8, device=dev)
        sB = torch.empty((ksH, R_pad, 4), dtype=torch.uint8, device=dev)
        gB = torch.empty((R_pad,), dtype=torch.float32, device=dev)
        GU = torch.empty((R_pad, 2 * I), dtype=torch.bfloat16, device=dev)
        Hb = torch.empty((R_pad, I), dtype=torch.bfloat16, device=dev)
        Bb2 = torch.empty((R_pad, I // 2), dtype=torch.uint8, device=dev)
        sB2 = torch.empty((ksI, R_pad, 4), dtype=torch.uint8, device=dev)
        gB2 = torch.empty((R_pad,), dtype=torch.float32, device=dev)
        Dg = torch.empty((R_pad, H), dtype=torch.bfloat16, device=dev)
        # gate/up input is fixed -> quantize once outside the captured region
        lib.quantize_act_nvfp4_2lvl(Xr.data_ptr(), Bb.data_ptr(), sB.data_ptr(), gB.data_ptr(), R_pad, H)

        def one_pass():
            st = torch.cuda.current_stream().cuda_stream
            Ac, meta, scaleA, gA = gu_W
            lib.sparse_moe_mm_2lvl(Ac.data_ptr(), Bb.data_ptr(), scaleA.data_ptr(), sB.data_ptr(), meta.data_ptr(),
                                   GU.data_ptr(), Ac.shape[0], 2 * I, R_pad, H, gA.data_ptr(), gB.data_ptr(),
                                   eblk.data_ptr(), 1, st)
            Hb.copy_((F.silu(GU[:, :I].float()) * GU[:, I:].float()).to(torch.bfloat16))
            lib.quantize_act_nvfp4_2lvl(Hb.data_ptr(), Bb2.data_ptr(), sB2.data_ptr(), gB2.data_ptr(), R_pad, I)
            Ad, md, sd, gd = dn_W
            lib.sparse_moe_mm_2lvl(Ad.data_ptr(), Bb2.data_ptr(), sd.data_ptr(), sB2.data_ptr(), md.data_ptr(),
                                   Dg.data_ptr(), Ad.shape[0], H, R_pad, I, gd.data_ptr(), gB2.data_ptr(),
                                   eblk.data_ptr(), 1, st)

        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                one_pass()
        torch.cuda.current_stream().wait_stream(s)
        eager = Dg.clone()
        try:
            gph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gph):
                one_pass()
            gph.replay(); torch.cuda.synchronize()
            cg = cos(Dg[valid], eager[valid])
            print(f"  [{name:14s} graph-capture   ] replay cos(graph,eager)={cg:.6f}  -> {'PASS' if cg > 0.999 else 'FAIL'}", flush=True)
            allok = allok and cg > 0.999
        except Exception as ex:  # noqa: BLE001
            print(f"  [{name:14s} graph-capture   ] FAILED: {type(ex).__name__}: {ex}", flush=True)
            allok = False
        print(f"# {name}: {'ALL PASS' if allok else 'FAIL'}", flush=True)


@app.local_entrypoint()
def main() -> None:
    validate.remote()
