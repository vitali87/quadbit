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


class SparseExpertsRepair(torch.nn.Module):
    """One MoE layer's routed experts as a differentiable fakequant-sparse block. Trainable = the
    surviving 2:4 weights (dropped held 0). scale_only=True freezes weights, trains only scales."""

    def __init__(
        self,
        w13: torch.Tensor,
        w2: torch.Tensor,
        colnorm_gu: torch.Tensor,
        colnorm_dn: torch.Tensor,
        inter: int,
        scale_only: bool = False,
    ):
        super().__init__()
        e = w13.shape[0]
        self.inter = inter
        self.scale_only = scale_only
        self.cgu = colnorm_gu  # (E, H)   per-expert gate/up input col rms
        self.cdn = colnorm_dn  # (E, I)   per-expert down  input col rms
        # keep-mask from the initial Wanda pick; hold dropped at 0 so serving re-selects it.
        self.w13 = torch.nn.Parameter(w13.float(), requires_grad=not scale_only)
        self.w2 = torch.nn.Parameter(w2.float(), requires_grad=not scale_only)
        self.register_buffer("m13", self._keepmask(w13.float(), colnorm_gu))
        self.register_buffer("m2", self._keepmask(w2.float(), colnorm_dn))
        with torch.no_grad():
            self.w13 *= self.m13
            self.w2 *= self.m2
        self.s_gate = torch.nn.Parameter(
            torch.ones(e)
        )  # per-expert output affine (always trainable)
        self.s_up = torch.nn.Parameter(torch.ones(e))
        self.s_dn = torch.nn.Parameter(torch.ones(e))

    @staticmethod
    def _keepmask(w: torch.Tensor, colnorm: torch.Tensor | None) -> torch.Tensor:
        out_f, in_f = w.shape[-2], w.shape[-1]
        masks = []
        for e in range(w.shape[0]):
            wg = w[e].view(out_f, in_f // 128, 16, 4, 2)
            cn = colnorm[e] if colnorm is not None else None
            imp = (
                wg.abs().sum(-1)
                if cn is None
                else (wg.abs() * cn.view(1, in_f // 128, 16, 4, 2)).sum(-1)
            )
            i01 = imp.topk(2, dim=-1).indices
            mk = torch.zeros_like(wg[..., 0])
            mk.scatter_(3, i01, 1.0)
            masks.append(mk.unsqueeze(-1).expand(-1, -1, -1, -1, 2).reshape(out_f, in_f))
        return torch.stack(masks)

    def forward(self, x: torch.Tensor, tid: torch.Tensor, tw: torch.Tensor) -> torch.Tensor:
        t, h = x.shape
        topk = tid.shape[1]
        y = torch.zeros(t, h, dtype=torch.float32, device=x.device)
        tok = torch.arange(t, device=x.device).repeat_interleave(topk)
        flat = tid.reshape(-1).long()
        fw = tw.reshape(-1).float()
        ii = self.inter
        for le in torch.unique(flat).tolist():
            if le < 0:
                continue
            sel = flat == le
            rows = tok[sel]
            xe = fq_act(x[rows].to(torch.bfloat16))
            wg = (
                served_weight((self.w13[le, :ii] * self.m13[le, :ii]), self.cgu[le])
                * self.s_gate[le]
            )
            wu = (
                served_weight((self.w13[le, ii:] * self.m13[le, ii:]), self.cgu[le]) * self.s_up[le]
            )
            wd = served_weight((self.w2[le] * self.m2[le]), self.cdn[le]) * self.s_dn[le]
            hh = fq_act((F.silu(xe @ wg.t()) * (xe @ wu.t())).to(torch.bfloat16))
            oe = (hh @ wd.t()) * fw[sel][:, None]
            y.index_add_(0, rows, oe.float())
        return y


def train_layer(
    io: dict,
    w13: torch.Tensor,
    w2: torch.Tensor,
    cgu: torch.Tensor,
    cdn: torch.Tensor,
    inter: int,
    steps: int = 2000,
    lr: float = 1e-3,
    scale_only: bool = False,
    ckpt_steps: tuple[int, ...] = (500, 1000, 2000, 5000),
) -> dict:
    """Fit one layer's surviving sparse-FP4 weights to the teacher MoE output. io = {x,tid,tw,y}.
    Returns {"w13","w2"} updated (dropped=0) + a loss trace + per-ckpt snapshots."""
    dev = w13.device
    x, tid, tw, y = io["x"].to(dev), io["tid"].to(dev), io["tw"].to(dev), io["y"].to(dev).float()
    m = SparseExpertsRepair(w13, w2, cgu, cdn, inter, scale_only).to(dev)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
    yn = y.norm().clamp_min(1e-6)
    trace, snaps = [], {}
    bs = min(4096, x.shape[0])
    for step in range(1, max(steps, max(ckpt_steps)) + 1):
        idx = torch.randint(0, x.shape[0], (bs,), device=dev)
        pred = m(x[idx], tid[idx], tw[idx])
        loss = (pred - y[idx]).pow(2).mean() + 0.1 * (
            1 - F.cosine_similarity(pred, y[idx], dim=1).mean()
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():  # keep dropped positions pinned at 0
            m.w13 *= m.m13
            m.w2 *= m.m2
        if step % 100 == 0:
            rel = (
                (m(x[idx], tid[idx], tw[idx]) - y[idx]).norm()
                / yn
                * (yn / y[idx].norm().clamp_min(1e-6))
            ).item()
            trace.append((step, round(loss.item(), 6), round(rel, 4)))
        if step in ckpt_steps:
            with torch.no_grad():
                snaps[step] = {
                    "w13": (m.w13 * m.m13 * m.s_gate.new_ones(1)).cpu().clone(),
                    "w2": (m.w2 * m.m2).cpu().clone(),
                }
    return {
        "w13": (m.w13 * m.m13).detach(),
        "w2": (m.w2 * m.m2).detach(),
        "s_gate": m.s_gate.detach(),
        "s_up": m.s_up.detach(),
        "s_dn": m.s_dn.detach(),
        "trace": trace,
        "snaps": snaps,
    }


def _selfcheck() -> None:
    # tiny CPU check: served_weight is 2:4 (half the pairs zero), STE keeps the forward value + a
    # gradient, and a short fit on synthetic teacher data drives the loss down.
    torch.manual_seed(0)
    out_f, in_f = 8, 256
    w = torch.randn(out_f, in_f)
    sw = served_weight(w, None)
    pairs = sw.view(out_f, in_f // 128, 16, 4, 2)
    nz = (pairs.abs().sum(-1) > 0).float().mean().item()
    assert abs(nz - 0.5) < 1e-6, f"2:4 kept fraction {nz} != 0.5"
    assert torch.allclose(served_weight(w, None), sw), "served_weight not deterministic"
    # STE: gradient reaches w
    wp = w.clone().requires_grad_(True)
    served_weight(wp, None).sum().backward()
    assert wp.grad is not None and wp.grad.abs().sum() > 0, "STE broke the gradient"

    # synthetic single-expert layer: teacher = a random dense expert; fit sparse to match.
    e, h, inter = 2, 128, 64
    w13 = torch.randn(e, 2 * inter, h) * 0.05
    w2 = torch.randn(e, h, inter) * 0.05
    n = 512
    x = torch.randn(n, h)
    tid = torch.randint(0, e, (n, 1))
    tw = torch.ones(n, 1)
    cgu = x.new_ones(e, h)
    cdn = x.new_ones(e, inter)
    teacher = SparseExpertsRepair(w13, w2, cgu, cdn, inter)
    with torch.no_grad():
        y = teacher(x, tid, tw)
    out = train_layer(
        {"x": x, "tid": tid, "tw": tw, "y": y},
        torch.randn_like(w13) * 0.05,
        torch.randn_like(w2) * 0.05,
        cgu,
        cdn,
        inter,
        steps=400,
        lr=3e-3,
        ckpt_steps=(400,),
    )
    first, last = out["trace"][0][1], out["trace"][-1][1]
    assert last < first, f"loss did not decrease: {first} -> {last}"
    steps = out["trace"][-1][0]
    print(f"selfcheck OK: 2:4 kept={nz:.3f}, fit loss {first:.5f} -> {last:.5f} over {steps} steps")


if __name__ == "__main__":
    _selfcheck()
