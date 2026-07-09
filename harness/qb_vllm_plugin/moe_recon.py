"""A3 layerwise sparse-FP4 repair (local knowledge distillation).

The training-free Pareto (WS-A) showed the downstream gap is broad and shallow: every 2:4-sparse MoE
layer drifts a little from its dense output, and adding dense anchors plateaus around -3.3pt (the
anchor sweep). Wanda only picks *which* 2-of-4 survive; it never *updates* them. A3 does the update.

Per sparse layer we have the DENSE (teacher) forward's MoE-block input x, routing, and output y
(dumped by the plugin). We rebuild each expert's surviving 2:4-FP4 weights so the sparse block out
matches teacher y, one layer at a time (no global backward, no teacher co-residency).

Faithfulness trick: the differentiable "served weight" is the REAL serving quantizer (2:4 pick +
two-level NVFP4) wrapped in a straight-through estimator, so train==serve numerically. Dropped
positions are held at 0, so the serving `pack()` re-selects the same mask (imp=0 there) for free.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# e2m1 FP4 value LUT (codes 0-15, sign bit 8) + the bucketize boundaries the kernel's q_fp4 uses.
_FP4_VALS = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
_FP4_BND = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
_GLOBAL_DIV = 2688.0  # rowamax / 2688 -> per-row global scale (matches pack + quant_act)
_BLK = 32  # two-level NVFP4 local-scale block


def _tables(dev: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fp4 = torch.tensor(_FP4_VALS, device=dev)
    bnd = torch.tensor(_FP4_BND, device=dev)
    cc = torch.arange(128, device=dev)
    e_, m_ = (cc >> 3) & 0xF, cc & 7
    ue4m3 = torch.where(
        e_ == 0, m_.float() * 0.001953125, (1.0 + m_.float() / 8.0) * torch.exp2((e_ - 7).float())
    )
    return fp4, bnd, ue4m3


def _enc(s: torch.Tensor) -> torch.Tensor:
    # fp32 -> ue4m3 code (mirror of the plugin's enc).
    mant_f, e = torch.frexp(s.clamp_min(1e-30))
    biased = (e - 1) + 7
    mant = torch.round((2.0 * mant_f - 1.0) * 8.0).long()
    carry = mant == 8
    mant = torch.where(carry, torch.zeros_like(mant), mant)
    biased = torch.where(carry, biased + 1, biased)
    code = (biased.long() << 3) | mant
    code = torch.where(biased < 1, torch.ones_like(code), code)
    code = torch.where(biased > 15, torch.full_like(code, 0x7F), code)
    code = torch.where(s >= 480.0, torch.full_like(code, 0x7F), code)
    return torch.where(s > 0, code, torch.zeros_like(code))


def served_weight(w: torch.Tensor, colnorm: torch.Tensor | None) -> torch.Tensor:
    """Exact serving weight [out,in]: 2:4 pick (Wanda if colnorm else magnitude) + two-level NVFP4,
    dropped positions -> 0. Differentiable via STE (forward = quantized, backward flows to w)."""
    fp4, bnd, ue4m3 = _tables(w.device)
    out_f, in_f = w.shape
    ks = in_f // 128
    wg = w.float().view(out_f, ks, 16, 4, 2)
    if colnorm is None or not bool((colnorm != 0).any()):
        imp = wg.abs().sum(-1)
    else:
        imp = (wg.abs() * colnorm.view(1, ks, 16, 4, 2)).sum(-1)
    i01, _ = imp.topk(2, dim=-1).indices.sort(dim=-1)
    kept = torch.gather(wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))  # (out,ks,16,2,2)
    ga = (
        (kept.abs().amax(dim=(1, 2, 3, 4), keepdim=True) / _GLOBAL_DIV)
        .clamp_min(1e-30)
        .reshape(out_f, 1, 1)
    )
    blk = kept.reshape(out_f, ks, 4, 8, 2)
    scode = _enc((blk.abs().amax(dim=(3, 4)) / 6.0) / ga)
    sdeq = ue4m3[scode] * ga  # (out,ks,4)
    codes = torch.bucketize((blk / sdeq[..., None, None].clamp_min(1e-30)).abs(), bnd)
    codes = codes | ((blk < 0).long() << 3)
    deq = fp4[codes] * sdeq[..., None, None]  # (out,ks,4,8,2) survivor values
    deq = deq.reshape(out_f, ks, 16, 2, 2)
    served = torch.zeros_like(wg)
    served.scatter_(3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2), deq)
    served = served.reshape(out_f, in_f)
    return w + (served - w).detach()  # STE


def fq_act(x: torch.Tensor) -> torch.Tensor:
    """Two-level NVFP4 fakequant of an activation row-block (mirror of quantize_act_nvfp4_2lvl):
    per-row global scale = rowamax/2688, per-32 ue4m3 local scale, E2M1 mantissa."""
    fp4, bnd, ue4m3 = _tables(x.device)
    r, in_f = x.shape
    xf = x.float()
    ga = (xf.abs().amax(dim=1, keepdim=True) / _GLOBAL_DIV).clamp_min(1e-30)  # (r,1)
    blk = xf.view(r, in_f // _BLK, _BLK)
    scode = _enc((blk.abs().amax(dim=2) / 6.0) / ga)
    sdeq = ue4m3[scode] * ga  # (r, in/32)
    codes = torch.bucketize((blk / sdeq[..., None].clamp_min(1e-30)).abs(), bnd)
    codes = codes | ((blk < 0).long() << 3)
    return (fp4[codes] * sdeq[..., None]).reshape(r, in_f).to(x.dtype)


def _dense_expert(
    x: torch.Tensor, wg: torch.Tensor, wu: torch.Tensor, wd: torch.Tensor
) -> torch.Tensor:
    # dense (teacher) expert operator == the plugin dense apply path (plain float matmul, no quant).
    return (F.silu(x @ wg.t()) * (x @ wu.t())) @ wd.t()


def _sparse_expert(
    x: torch.Tensor,
    wg: torch.Tensor,
    wu: torch.Tensor,
    wd: torch.Tensor,
    cgu: torch.Tensor,
    cdn: torch.Tensor,
) -> torch.Tensor:
    # student: fakequant activations + served (2:4-FP4) weights, mirroring the seg-kernel operator.
    # Matmuls run in float32 (served_weight is float32 via its STE); fq_act quantizes in bf16 then casts.
    xq = fq_act(x.to(torch.bfloat16)).float()
    gu = F.silu(xq @ served_weight(wg, cgu).t()) * (xq @ served_weight(wu, cgu).t())
    return fq_act(gu.to(torch.bfloat16)).float() @ served_weight(wd, cdn).t()


def train_expert(
    x: torch.Tensor,
    wg: torch.Tensor,
    wu: torch.Tensor,
    wd: torch.Tensor,
    cgu: torch.Tensor,
    cdn: torch.Tensor,
    steps: int,
    lr: float,
    scale_only: bool,
) -> dict:
    """Fit ONE expert surviving 2:4-FP4 weights (dropped held 0) to its dense output over its routed
    tokens x. Teacher = dense operator; student = fakequant-sparse. Memory-trivial (one expert)."""
    dev = wg.device
    mg = (served_weight(wg, cgu) != 0).float()  # keep-mask (dropped positions -> 0)
    mu = (served_weight(wu, cgu) != 0).float()
    md = (served_weight(wd, cdn) != 0).float()
    with torch.no_grad():
        teacher = _dense_expert(x.float(), wg.float(), wu.float(), wd.float())
    tn = teacher.norm().clamp_min(1e-6)

    if scale_only:
        s = [torch.ones(1, device=dev, requires_grad=True) for _ in range(3)]
        params, base = s, (wg.float() * mg, wu.float() * mu, wd.float() * md)
    else:
        pg = (wg.float() * mg).requires_grad_(True)
        pu = (wu.float() * mu).requires_grad_(True)
        pd = (wd.float() * md).requires_grad_(True)
        params = [pg, pu, pd]
    opt = torch.optim.Adam(params, lr=lr)
    trace = []
    for step in range(1, steps + 1):
        if scale_only:
            wg_, wu_, wd_ = base[0] * s[0], base[1] * s[1], base[2] * s[2]
        else:
            wg_, wu_, wd_ = pg * mg, pu * mu, pd * md
        pred = _sparse_expert(x, wg_, wu_, wd_, cgu, cdn)
        loss = (pred - teacher).pow(2).mean() + 0.1 * (
            1 - F.cosine_similarity(pred, teacher, dim=1).mean()
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, steps // 5) == 0 or step == steps:
            with torch.no_grad():
                rel = ((_sparse_expert(x, wg_, wu_, wd_, cgu, cdn) - teacher).norm() / tn).item()
            trace.append((step, round(rel, 4)))
    with torch.no_grad():
        if scale_only:
            wg_, wu_, wd_ = base[0] * s[0], base[1] * s[1], base[2] * s[2]
        else:
            wg_, wu_, wd_ = pg * mg, pu * mu, pd * md
    return {"wg": wg_.detach(), "wu": wu_.detach(), "wd": wd_.detach(), "trace": trace}


def train_layer(
    io: dict,
    w13: torch.Tensor,
    w2: torch.Tensor,
    cgu: torch.Tensor,
    cdn: torch.Tensor,
    inter: int,
    steps: int = 400,
    lr: float = 1e-3,
    scale_only: bool = False,
) -> dict:
    """Repair one MoE layer: per routed expert, fit its sparse weights to dense output. io={x,tid}.
    Returns updated w13/w2 (dropped=0, folded scales); unrouted experts keep their dense weights."""
    dev = w13.device
    x, tid = io["x"].to(dev), io["tid"].to(dev)
    w13n, w2n = w13.float().clone(), w2.float().clone()
    tok = torch.arange(x.shape[0], device=dev).repeat_interleave(tid.shape[1])
    flat = tid.reshape(-1).long()
    traces = {}
    for le in torch.unique(flat).tolist():
        if le < 0:
            continue
        rows = tok[flat == le]
        if rows.numel() < 8:
            continue
        r = train_expert(
            x[rows],
            w13[le, :inter],
            w13[le, inter:],
            w2[le],
            cgu[le],
            cdn[le],
            steps,
            lr,
            scale_only,
        )
        w13n[le, :inter], w13n[le, inter:], w2n[le] = r["wg"], r["wu"], r["wd"]
        traces[le] = r["trace"][-1] if r["trace"] else None
    rels = [t[1] for t in traces.values() if t]
    return {
        "w13": w13n,
        "w2": w2n,
        "n_experts": len(traces),
        "rel_mean": round(sum(rels) / len(rels), 4) if rels else -1.0,
    }


def _selfcheck() -> None:
    # tiny CPU check: served_weight is 2:4 (half the pairs zero), STE keeps the forward value + a
    # gradient, and a per-expert fit drives the dense-match relative error down.
    torch.manual_seed(0)
    out_f, in_f = 8, 256
    w = torch.randn(out_f, in_f)
    sw = served_weight(w, None)
    nz = (sw.view(out_f, in_f // 128, 16, 4, 2).abs().sum(-1) > 0).float().mean().item()
    assert abs(nz - 0.5) < 1e-6, f"2:4 kept fraction {nz} != 0.5"
    assert torch.allclose(served_weight(w, None), sw), "served_weight not deterministic"
    wp = w.clone().requires_grad_(True)
    served_weight(wp, None).sum().backward()
    assert wp.grad is not None and wp.grad.abs().sum() > 0, "STE broke the gradient"

    # single expert: teacher = dense; fit sparse to match, expect the relative error to fall.
    # (served_weight needs in_f % 128 == 0 -> h, inter both multiples of 128, as in the model)
    h, inter, n = 256, 128, 512
    w13 = torch.randn(2, 2 * inter, h) * 0.05
    w2 = torch.randn(2, h, inter) * 0.05
    x = torch.randn(n, h)
    tid = torch.randint(0, 2, (n, 1))
    cgu, cdn = x.new_ones(2, h), x.new_ones(2, inter)
    out = train_layer({"x": x, "tid": tid}, w13, w2, cgu, cdn, inter, steps=300, lr=3e-3)
    r0 = train_expert(x, w13[0, :inter], w13[0, inter:], w2[0], cgu[0], cdn[0], 1, 3e-3, False)[
        "trace"
    ][-1][1]
    assert out["rel_mean"] < r0, (
        f"per-expert fit did not improve: start {r0} -> end {out['rel_mean']}"
    )
    print(
        f"selfcheck OK: 2:4 kept={nz:.3f}, dense-match rel {r0:.4f} -> {out['rel_mean']:.4f} "
        f"over {out['n_experts']} experts"
    )


if __name__ == "__main__":
    _selfcheck()
