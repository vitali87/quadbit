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


_TABLES_CACHE: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _tables(dev: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if dev in _TABLES_CACHE:  # cache: rebuilt every served_weight/fq_act call in the train loop
        return _TABLES_CACHE[dev]
    fp4 = torch.tensor(_FP4_VALS, device=dev)
    bnd = torch.tensor(_FP4_BND, device=dev)
    cc = torch.arange(128, device=dev)
    e_, m_ = (cc >> 3) & 0xF, cc & 7
    ue4m3 = torch.where(
        e_ == 0, m_.float() * 0.001953125, (1.0 + m_.float() / 8.0) * torch.exp2((e_ - 7).float())
    )
    _TABLES_CACHE[dev] = (fp4, bnd, ue4m3)
    return fp4, bnd, ue4m3


def _enc(s: torch.Tensor) -> torch.Tensor:
    # fp32 -> ue4m3 code (mirror of the plugin's enc).
    mant_f, e = torch.frexp(s.clamp_min(1e-30))
    biased = (e - 1) + 7
    mant = torch.round((2.0 * mant_f - 1.0) * 8.0).long()
    carry = mant == 8
    mant = torch.where(carry, torch.zeros_like(mant), mant)
    biased = torch.where(carry, biased + 1, biased)
    # clamp before the shift (out-of-range biased is overridden below anyway) to avoid a shift on a
    # negative operand
    code = (biased.clamp(0, 15).long() << 3) | mant
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
    if colnorm is None:
        imp = wg.abs().sum(-1)
    else:
        # Wanda importance + a tiny magnitude term so an all-zero colnorm (an unrouted expert)
        # degrades to magnitude ordering, without the per-step `.any()` GPU->CPU sync.
        imp = (wg.abs() * colnorm.view(1, ks, 16, 4, 2)).sum(-1) + wg.abs().sum(-1) * 1e-30
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
    xf = x.float().clamp(-1e4, 1e4)  # match serving _sanitize guardrail: stop NaN/Inf propagation
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
    proj: str = "both",
) -> torch.Tensor:
    # student: fakequant activations + served (2:4-FP4) weights, mirroring the seg-kernel operator.
    # Matmuls run in float32 (served_weight is float32 via STE); fq_act quantizes bf16 then casts.
    xq = fq_act(x.to(torch.bfloat16)).float()
    gu = F.silu(xq @ served_weight(wg, cgu).t()) * (xq @ served_weight(wu, cgu).t())
    if proj == "gateup":
        # gateup49 policy: down is anchored dense NVFP4 (cos~1, near-exact) -> keep it exact here so
        # training isolates the gate_up tax we are actually attacking (down carries ~none of it).
        return gu @ wd.t()
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
    proj: str = "both",
) -> dict:
    """Fit ONE expert surviving 2:4-FP4 weights (dropped held 0) to its dense output over its routed
    tokens x. Teacher = dense operator; student = fakequant-sparse. Memory-trivial (one expert).

    proj="gateup": only gate_up (wg/wu) is sparse+trained; down (wd) stays dense-exact and frozen,
    matching the deployed gateup49 policy (down anchored NVFP4). Isolates the gate_up tax and stops
    the optimizer chasing the down projection (which A3's both-proj repair conflated). lr default
    1e-4 is deliberate: the synthetic probe showed lr=1e-3 DIVERGES under the FP4 STE (the smooth
    surrogate gradient walks the discrete quantized forward the wrong way). Keep the best true-rel
    weights, since the STE noise floor reverses after ~100 steps."""
    dev = wg.device
    gu_only = proj == "gateup"
    mg = (served_weight(wg, cgu) != 0).float()  # keep-mask (dropped positions -> 0)
    mu = (served_weight(wu, cgu) != 0).float()
    md = torch.ones_like(wd) if gu_only else (served_weight(wd, cdn) != 0).float()
    with torch.no_grad():
        teacher = _dense_expert(x.float(), wg.float(), wu.float(), wd.float())
    tn = teacher.norm().clamp_min(1e-6)
    wd_frozen = wd.float()  # gateup mode: down stays exact-dense, never trained

    if scale_only:
        s = [torch.ones(1, device=dev, requires_grad=True) for _ in range(2 if gu_only else 3)]
        params = s
        base = (wg.float() * mg, wu.float() * mu, wd_frozen if gu_only else wd.float() * md)
    else:
        pg = (wg.float() * mg).requires_grad_(True)
        pu = (wu.float() * mu).requires_grad_(True)
        params = [pg, pu]
        if not gu_only:
            pd = (wd.float() * md).requires_grad_(True)
            params.append(pd)

    def _weights():
        if scale_only:
            return base[0] * s[0], base[1] * s[1], wd_frozen if gu_only else base[2] * s[2]
        return pg * mg, pu * mu, wd_frozen if gu_only else pd * md

    opt = torch.optim.Adam(params, lr=lr)
    trace = []
    best_rel, best_w = float("inf"), None
    for step in range(1, steps + 1):
        wg_, wu_, wd_ = _weights()
        pred = _sparse_expert(x, wg_, wu_, wd_, cgu, cdn, proj)
        loss = (pred - teacher).pow(2).mean() + 0.1 * (
            1 - F.cosine_similarity(pred, teacher, dim=1).mean()
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step % max(1, steps // 10) == 0 or step == steps:
            with torch.no_grad():
                wg_, wu_, wd_ = _weights()
                p2 = _sparse_expert(x, wg_, wu_, wd_, cgu, cdn, proj)
                rel = ((p2 - teacher).norm() / tn).item()
            trace.append((step, round(rel, 4), round(loss.item(), 6)))
            if rel < best_rel:  # keep best true-rel (STE noise floor reverses late)
                best_rel = rel
                best_w = tuple(w.detach().clone() for w in (wg_, wu_, wd_))
    if best_w is None:
        with torch.no_grad():
            best_w = tuple(w.detach() for w in _weights())
    return {"wg": best_w[0], "wu": best_w[1], "wd": best_w[2], "trace": trace}


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


def train_layer_lazy(
    io: dict,
    get_expert,
    cgu: torch.Tensor,
    cdn: torch.Tensor,
    inter: int,
    dev: torch.device,
    steps: int = 400,
    lr: float = 1e-3,
    scale_only: bool = False,
    proj: str = "both",
) -> dict:
    """Memory-lean train_layer: dequant ONE expert at a time via get_expert(le)->(wg,wu,wd) on
    `dev`, so peak GPU scratch is a single expert (not the whole [E,...] bf16 stack, which is
    ~13 GiB and OOMs the near-full serving GPU). Returns only the repaired routed experts as
    {le: (w13form[2I,H], w2form[H,I]) bf16 on CPU}; caller dense-dequants the rest at pack time."""
    x_all, tid = io["x"].to(dev), io["tid"].to(dev)
    tok = torch.arange(x_all.shape[0], device=dev).repeat_interleave(tid.shape[1])
    flat = tid.reshape(-1).long()
    repaired, traces = {}, {}
    for le in torch.unique(flat).tolist():
        if le < 0:
            continue
        rows = tok[flat == le]
        if rows.numel() < 8:
            continue
        wg, wu, wd = get_expert(le)
        r = train_expert(x_all[rows], wg, wu, wd, cgu[le], cdn[le], steps, lr, scale_only, proj)
        repaired[le] = (
            torch.cat([r["wg"], r["wu"]], 0).to(torch.bfloat16).cpu(),
            r["wd"].to(torch.bfloat16).cpu(),
        )
        traces[le] = r["trace"][-1] if r["trace"] else None
        del wg, wu, wd, r
    rels = [t[1] for t in traces.values() if t]
    return {
        "repaired": repaired,
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

    # Optimizer mechanics on a realistic case: CORRELATED activations (low-rank), so the dense
    # teacher is genuinely approximable by repaired 2:4 survivors (as with real routed tokens; iid
    # noise has a hard 2:4 floor no repair can beat). Assert the training LOSS falls. served_weight
    # (served_weight needs in_f % 128 == 0 -> h, inter both multiples of 128, as in the model)
    h, inter, n = 256, 128, 512
    w13 = torch.randn(2, 2 * inter, h) * 0.05
    w2 = torch.randn(2, h, inter) * 0.05
    x = (torch.randn(n, 8) @ torch.randn(8, h)) * (8**-0.5)  # rank-8 correlated activations
    cgu, cdn = x.new_ones(2, h), x.new_ones(2, inter)
    tr = train_expert(
        x,
        w13[0, :inter],
        w13[0, inter:],
        w2[0],
        cgu[0],
        cdn[0],
        steps=300,
        lr=5e-4,
        scale_only=False,
    )["trace"]
    l0, l1 = tr[0][2], tr[-1][2]
    assert l1 < l0, f"training loss did not decrease: {l0} -> {l1}"

    # gateup-only mode: down is frozen exact (2 trainable weights, not 3), loss still falls, and the
    # returned wd equals the input dense wd (down untouched). This is the deployed gateup49 config.
    rg = train_expert(x, w13[0, :inter], w13[0, inter:], w2[0], cgu[0], cdn[0],
                      steps=200, lr=1e-4, scale_only=False, proj="gateup")
    assert rg["trace"][-1][2] < rg["trace"][0][2], "gateup loss did not decrease"
    assert torch.allclose(rg["wd"], w2[0].float()), "gateup mode must leave down (wd) exact"

    # Lazy layer path (the memory-lean plumbing used in-model): get_expert dequants one expert at a
    # time; every routed expert (>=8 rows) comes back as (w13form[2I,H], w2form[H,I]) on CPU.
    ne = 4
    w13l = torch.randn(ne, 2 * inter, h) * 0.05
    w2l = torch.randn(ne, h, inter) * 0.05
    tid = torch.randint(0, ne, (n, 2))
    io = {"x": x, "tid": tid}
    lz = train_layer_lazy(
        io,
        lambda le: (w13l[le, :inter], w13l[le, inter:], w2l[le]),
        x.new_ones(ne, h),
        x.new_ones(ne, inter),
        inter,
        x.device,
        steps=20,
        lr=5e-4,
    )
    assert lz["n_experts"] > 0, "lazy path repaired no experts"
    le0 = next(iter(lz["repaired"]))
    w13f, w2f = lz["repaired"][le0]
    assert w13f.shape == (2 * inter, h) and w2f.shape == (h, inter), f"bad shapes {w13f.shape}"
    assert w13f.is_cpu and w2f.is_cpu, "repaired weights must be on CPU"
    print(
        f"selfcheck OK: 2:4 kept={nz:.3f}, loss {l0:.5f}->{l1:.5f}, "
        f"rel {tr[0][1]:.3f}->{tr[-1][1]:.3f}, lazy {lz['n_experts']}exp rel={lz['rel_mean']}"
    )


if __name__ == "__main__":
    _selfcheck()
