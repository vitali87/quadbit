"""HYBRID sparse QAT pilot: sparsify ONLY a chosen subset of MLP matrices (mask), keep the rest
dense W4A4, attention bf16; short QAT repair from the dense checkpoint; trace the recovered Pareto.

The training-free hybrid sweep (`harness/sensitivity_sparse.py`) said no free lunch: errors compound,
+0.05 PPL buys ~3% of MLP FLOPs sparse. This asks the follow-on question the sweep could not: does
QAT recovery move that frontier? Bounded pilot, one mask per launch, stop-gated so it earns compute.

Masks (matrix types sparsified; attention always bf16/dense, per the sweep's attn-is-dangerous result):
  A: down_proj            (~27% of linear FLOPs; down_proj = most sparse-tolerant type in both sweeps)
  B: down_proj,gate_proj  (~54%; gate ranked more tolerant than up)
  C: down_proj,gate_proj,up_proj (all-MLP; the recovered control = 8.47 from finetune_pair)
Non-masked MLP matrices are FROZEN dense W4A4 two-level fake-quant, so the delta isolates the
sparsification effect against a consistent in-harness dense-W4A4-MLP baseline.

STOP GATES (reported, applied by the launcher): kill if after 1k QAT steps still >0.75 PPL over the
dense-MLP baseline and flat; kill if <8% est prefill speedup; continue only if >=25% sparse FLOPs,
deploy gap <=0.02, PPL within ~+0.25-0.50 of the dense-MLP baseline.

Run:  uv run modal run --detach harness/finetune_hybrid.py --mask down_proj --model meta-llama/Meta-Llama-3-8B
"""

import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
MODEL = "meta-llama/Meta-Llama-3-8B"
SP_RATIO = 1.33  # deployed sparse-vs-dense roofline ratio (paper §5); caps hybrid speedup

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64", "HF_HOME": "/cache",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .pip_install("torch", index_url="https://download.pytorch.org/whl/nightly/cu128", pre=True)
    .pip_install("transformers", "huggingface_hub", "safetensors", "sentencepiece", "pyarrow", "bitsandbytes", "numpy")
    .add_local_dir((ROOT / "cuda").as_posix(), "/root/cuda")
)
app = modal.App("quadbit-finetune-hybrid", image=image)
vol = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)


@app.function(gpu="RTX-PRO-6000", timeout=86400, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(model: str = MODEL, mask: str = "down_proj", p1: int = 3000, p2: int = 2000,
        lr_max: float = 3e-5, p2_lr: float = 1e-5) -> float:
    import ctypes
    import math

    import pyarrow.parquet as pq
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    so = "/root/sparse_fp4.so"
    c = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
                        "-o", so, "/root/cuda/sparse_fp4_lib.cu", "-lcuda"], capture_output=True, text=True)
    if c.returncode != 0:
        print(c.stderr, flush=True); return -1.0
    lib = ctypes.CDLL(so)
    lib.sparse_fp4_mm_2lvl.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    lib.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    dev = torch.device("cuda")

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

    def sparse_fp4_dequant(W):  # pair-2:4 magnitude + per-16 two-level (per-out-row global gA)
        out_f, in_f = W.shape; ks = in_f // 128
        Wg = W.view(out_f, ks, 16, 4, 2)
        i01, _ = Wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
        keptW = torch.gather(Wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
        gA = (keptW.abs().amax(dim=(1, 2, 3, 4), keepdim=True) / 2688.0).clamp_min(1e-30).reshape(out_f, 1, 1)
        blk = keptW.reshape(out_f, ks, 4, 8, 2)
        sdeq = UE4M3[enc_ue4m3_t((blk.abs().amax(dim=(3, 4)) / 6.0) / gA)] * gA
        kd = (FP4[q_fp4(blk / sdeq[..., None, None])] * sdeq[..., None, None]).reshape(out_f, ks, 16, 2, 2)
        Wd = torch.zeros(out_f, ks, 16, 4, 2, device=dev)
        Wd.scatter_(3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2), kd)
        return Wd.reshape(out_f, in_f)

    def dense_fp4_dequant(W):  # NO prune: per-16 two-level (per-out-row global gA) = deployed dense W4A4
        out_f, in_f = W.shape
        gA = (W.abs().amax(-1, keepdim=True) / 2688.0).clamp_min(1e-30)
        b = W.view(out_f, in_f // 16, 16)
        sdeq = UE4M3[enc_ue4m3_t((b.abs().amax(-1) / 6.0) / gA)] * gA
        return (FP4[q_fp4(b / sdeq[..., None])] * sdeq[..., None]).reshape(out_f, in_f)

    def act_fp4_dequant(x):  # per-32 two-level (matched to the sparse mma B operand)
        lead = x.shape[:-1]; i = x.shape[-1]
        b = x.to(torch.bfloat16).float().reshape(-1, i)
        gB = (b.abs().amax(-1, keepdim=True) / 2688.0).clamp_min(1e-30)
        bb = b.reshape(b.shape[0], i // 32, 32)
        sdeq = UE4M3[enc_ue4m3_t((bb.abs().amax(-1) / 6.0) / gB)] * gB
        return (FP4[q_fp4(bb / sdeq[..., None])] * sdeq[..., None]).reshape(*lead, i)

    class SparseQAT(nn.Module):  # trainable, pruned, sparse two-level STE (train == deploy)
        def __init__(self, weight):
            super().__init__()
            self.weight = nn.Parameter(weight.clone()); self.qat = False

        def forward(self, x):
            if not self.qat:
                return F.linear(x, self.weight)
            Wf = self.weight.float()
            Wq = Wf + (sparse_fp4_dequant(Wf) - Wf).detach()
            xf = x.float(); xq = xf + (act_fp4_dequant(xf) - xf).detach()
            return F.linear(xq, Wq).to(x.dtype)

    class DenseW4A4(nn.Module):  # FROZEN dense W4A4 two-level fake-quant (the "rest stays dense" reference)
        def __init__(self, weight):
            super().__init__()
            self.register_buffer("w", weight.clone())

        def forward(self, x):
            Wq = dense_fp4_dequant(self.w.float())
            xf = x.float(); xq = xf + (act_fp4_dequant(xf) - xf).detach()
            return F.linear(xq, Wq).to(x.dtype)

    class QuadbitLinear(nn.Module):  # sparse two-level KERNEL (for the through-kernel deploy eval)
        def __init__(self, W):
            super().__init__()
            out_f, in_f = W.shape; ks = in_f // 128
            self.out_f, self.in_f, self.ks = out_f, in_f, ks
            Wg = W.float().to(dev).view(out_f, ks, 16, 4, 2)
            i01, _ = Wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
            keptW = torch.gather(Wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
            gA = (keptW.abs().amax(dim=(1, 2, 3, 4), keepdim=True) / 2688.0).clamp_min(1e-30).reshape(out_f, 1, 1)
            blk = keptW.reshape(out_f, ks, 4, 8, 2)
            scode = enc_ue4m3_t((blk.abs().amax(dim=(3, 4)) / 6.0) / gA)
            sdeq = UE4M3[scode] * gA
            kc = q_fp4(blk / sdeq[..., None, None])
            Ac = (kc[..., 0] | (kc[..., 1] << 4)).reshape(out_f, ks * 32).to(torch.uint8)
            nib = (i01[..., 0] | (i01[..., 1] << 2)).view(out_f, ks, 2, 8)
            sh = (torch.arange(8, device=dev) * 4).view(1, 1, 1, 8)
            meta = (nib << sh).sum(-1).to(torch.int32).permute(1, 0, 2).contiguous()
            self.register_buffer("Ac", Ac.contiguous()); self.register_buffer("meta", meta)
            self.register_buffer("scaleA", scode.to(torch.uint8).permute(1, 0, 2).contiguous())
            self.register_buffer("gA", gA.reshape(out_f).float().contiguous())

        def forward(self, x):
            lead = x.shape[:-1]; x2 = x.reshape(-1, self.in_f).to(torch.bfloat16)
            t = x2.shape[0]; pad = (-t) % 128
            if pad:
                x2 = torch.cat([x2, x2.new_zeros(pad, self.in_f)], 0)
            x2 = x2.contiguous(); tp = t + pad
            Bb = torch.empty((tp, self.in_f // 2), dtype=torch.uint8, device=dev)
            sB = torch.empty((self.ks, tp, 4), dtype=torch.uint8, device=dev)
            gB = torch.empty((tp,), dtype=torch.float32, device=dev)
            lib.quantize_act_nvfp4_2lvl(x2.data_ptr(), Bb.data_ptr(), sB.data_ptr(), gB.data_ptr(), tp, self.in_f)
            C = torch.empty((self.out_f, tp), dtype=torch.bfloat16, device=dev)
            lib.sparse_fp4_mm_2lvl(self.Ac.data_ptr(), Bb.data_ptr(), self.scaleA.data_ptr(),
                                   sB.data_ptr(), self.meta.data_ptr(), C.data_ptr(),
                                   self.out_f, tp, self.in_f, self.gA.data_ptr(), gB.data_ptr())
            return C.t()[:t].reshape(*lead, self.out_f).to(x.dtype)

    def sparsegpt_pair24(W, H, blocksize=128, percdamp=0.01):
        W = W.float().clone(); cols = W.shape[1]
        d = torch.diag(H) == 0; H[d, d] = 1.0; W[:, d] = 0.0
        H[range(cols), range(cols)] += percdamp * torch.mean(torch.diag(H))
        Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
        rows = W.shape[0]; ri = torch.arange(rows, device=dev)
        for i in range(0, cols, blocksize):
            e = min(i + blocksize, cols); B = e - i
            W1 = W[:, i:e].clone(); Err = torch.zeros_like(W1)
            H1 = Hinv[i:e, i:e]; dinv = torch.diag(H1); cm = None
            for j in range(B):
                if (j % 8) == 0:
                    tmp = (W1[:, j:j + 8] ** 2) / (dinv[j:j + 8] ** 2)[None, :]
                    pm = tmp[:, 0::2] + tmp[:, 1::2]; pr = pm.topk(2, dim=1, largest=False).indices
                    cm = torch.ones(rows, 8, dtype=torch.bool, device=dev)
                    cm[ri[:, None], pr * 2] = False; cm[ri[:, None], pr * 2 + 1] = False
                w = W1[:, j]; q = torch.where(cm[:, j % 8], w, torch.zeros_like(w))
                err = (w - q) / dinv[j]; W1[:, j:] -= err[:, None] * H1[j, j:][None, :]
                Err[:, j] = err; W1[:, j] = q
            W[:, i:e] = W1; W[:, e:] -= Err @ Hinv[i:e, e:]
        return W

    tok = AutoTokenizer.from_pretrained(model)

    def load():
        return AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).to(dev)

    def wikitext(cfg, fn):
        p = hf_hub_download("Salesforce/wikitext", f"{cfg}/{fn}", repo_type="dataset")
        return tok("\n\n".join(pq.read_table(p).column("text").to_pylist())[:200_000_000], return_tensors="pt").input_ids[0]

    test_ids = wikitext("wikitext-2-raw-v1", "test-00000-of-00001.parquet")
    train_ids = wikitext("wikitext-103-raw-v1", "train-00000-of-00002.parquet")
    print(f"train tokens: {len(train_ids)}", flush=True)

    def ppl(m, windows=16, seq=2048):
        m.eval(); nll, n = 0.0, 0
        for i in range(0, min(len(test_ids), windows * seq) - seq, seq):
            w = test_ids[i:i + seq].unsqueeze(0).to(dev)
            with torch.no_grad():
                nll += m(w, labels=w).loss.item() * (seq - 1); n += seq - 1
        return math.exp(nll / n)

    def mlp_lins(m):
        for layer in m.model.layers:
            for nm in ("gate_proj", "up_proj", "down_proj"):
                lin = getattr(layer.mlp, nm)
                if lin.weight.shape[0] % 256 == 0 and lin.weight.shape[1] % 256 == 0:
                    yield layer.mlp, nm, lin

    mask_set = set(mask.split(","))
    print(f"HYBRID mask={sorted(mask_set)}  model={model}", flush=True)

    teacher = load().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"PPL dense bf16 teacher: {ppl(teacher):.3f}", flush=True)

    # in-harness DENSE-W4A4-MLP baseline (all MLP dense W4A4, attn bf16) -> the honest delta reference
    base = load().eval()
    for mlp, nm, lin in mlp_lins(base):
        setattr(mlp, nm, DenseW4A4(lin.weight.data).to(dev))
    dense_mlp_ppl = ppl(base)
    del base; torch.cuda.empty_cache()
    print(f"PPL dense-W4A4-MLP baseline (all MLP dense, attn bf16): {dense_mlp_ppl:.3f}", flush=True)

    student = load()
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.config.use_cache = False

    # split MLP targets by mask; calibrate H only for the sparse ones; prune+freeze mask
    targets = list(mlp_lins(student))
    sparse_t = [(mlp, nm, lin) for (mlp, nm, lin) in targets if nm in mask_set]
    dense_t = [(mlp, nm, lin) for (mlp, nm, lin) in targets if nm not in mask_set]
    print(f"sparse matrices: {len(sparse_t)}   dense matrices: {len(dense_t)}", flush=True)

    Hs, ns = {}, {}

    def hook(mod, inp, _o):
        x = inp[0].detach().float().reshape(-1, inp[0].shape[-1]); k = id(mod); nn_ = x.shape[0]
        if k not in Hs:
            Hs[k] = torch.zeros(x.shape[1], x.shape[1], device=dev); ns[k] = 0
        Hs[k] *= ns[k] / (ns[k] + nn_)
        Hs[k] += (x * math.sqrt(2.0 / (ns[k] + nn_))).t() @ (x * math.sqrt(2.0 / (ns[k] + nn_)))
        ns[k] += nn_

    hs = [lin.register_forward_hook(hook) for _, _, lin in sparse_t]
    with torch.no_grad():
        for i in range(0, 16 * 2048, 2048):
            student(train_ids[i:i + 2048].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()

    masks_d, sparse_qats = {}, []
    frac_sparse_params = 0
    total_lin_params = 0
    for mlp, nm, lin in targets:
        total_lin_params += lin.weight.numel()
    # attention params also count toward the FLOP fraction denominator (they stay dense)
    for layer in student.model.layers:
        for nm2 in ("q_proj", "k_proj", "v_proj", "o_proj"):
            total_lin_params += getattr(layer.self_attn, nm2).weight.numel()

    for mlp, nm, lin in sparse_t:
        Wp = sparsegpt_pair24(lin.weight.data, Hs.pop(id(lin)))
        q = SparseQAT(Wp.to(torch.bfloat16)).to(dev)
        masks_d[q] = (Wp != 0).to(dev); setattr(mlp, nm, q); sparse_qats.append(q)
        frac_sparse_params += lin.weight.numel()
    for mlp, nm, lin in dense_t:
        setattr(mlp, nm, DenseW4A4(lin.weight.data).to(dev))
    frac = frac_sparse_params / total_lin_params
    est_speedup = 1.0 / ((1.0 - frac) + frac / SP_RATIO)
    print(f"sparse FLOP fraction {frac:.3f}  est prefill speedup {est_speedup:.3f}x "
          f"(SP_RATIO {SP_RATIO})", flush=True)
    print(f"PPL one-shot hybrid (pre-QAT, sparse fake-quant): {ppl(student):.3f}  "
          f"(dense-MLP baseline {dense_mlp_ppl:.3f})", flush=True)

    for p in student.parameters():
        p.requires_grad_(False)
    params = []
    for q in sparse_qats:
        q.weight.requires_grad_(True); params.append(q.weight)
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(params, lr=lr_max, betas=(0.9, 0.95), weight_decay=0.0)

    T, seq = 2.0, 1024
    P1, P2, wu = p1, p2, 300
    total = P1 + P2; lr_min = lr_max / 20.0
    starts = list(range(0, len(train_ids) - seq, seq))

    def kd_step(step):
        for g in opt.param_groups:
            if step < P1:
                g["lr"] = (lr_max * (step + 1) / wu if step < wu
                           else lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * (step - wu) / max(1, P1 - wu))))
            else:
                s = step - P1; p2wu = 100
                g["lr"] = (p2_lr * (s + 1) / p2wu if s < p2wu
                           else lr_min + 0.5 * (p2_lr - lr_min) * (1 + math.cos(math.pi * (s - p2wu) / max(1, P2 - p2wu))))
        i = starts[(step * 2654435761) % len(starts)]
        w = train_ids[i:i + seq].unsqueeze(0).to(dev)
        with torch.no_grad():
            tl = teacher(w).logits.reshape(-1, teacher.config.vocab_size)
        logits = student(w).logits.reshape(-1, teacher.config.vocab_size)
        kl = F.kl_div(F.log_softmax(logits / T, -1), F.softmax(tl / T, -1), reduction="batchmean") * (T * T)
        opt.zero_grad(); kl.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        with torch.no_grad():
            for q, mk in masks_d.items():
                q.weight.data *= mk
        return kl.item()

    student.train()
    for step in range(P1):  # phase-1: bf16 masked recovery
        loss = kd_step(step)
        if (step + 1) % 1000 == 0:
            print(f"  P1 {step + 1}/{P1} KD {loss:.4f} PPL(bf16) {ppl(student, windows=8):.3f}", flush=True)
            student.train()
    print(f"PPL after phase-1 bf16 recovery: {ppl(student):.3f}", flush=True)

    for q in sparse_qats:
        q.qat = True
    student.train()
    ppl_at = {}
    for step in range(P1, total):  # phase-2: QAT, eval at 500/1000/2000
        loss = kd_step(step)
        s = step + 1 - P1
        if s in (500, 1000, 2000) or (step + 1) == total:
            fq = ppl(student, windows=16); ppl_at[s] = fq
            print(f"  P2 {s}/{P2} KD {loss:.4f} PPL(fake-quant FP4) {fq:.3f} "
                  f"dPPL {fq - dense_mlp_ppl:+.3f} vs dense-MLP", flush=True)
            student.train()
    fake_quant_ppl = ppl(student)
    print(f"PPL after phase-2 QAT (fake-quant): {fake_quant_ppl:.3f}", flush=True)

    # through-kernel deploy eval: sparse matrices -> QuadbitLinear kernel; dense stay DenseW4A4
    for mlp, nm, lin in targets:
        if nm in mask_set:
            q = getattr(mlp, nm)
            setattr(mlp, nm, QuadbitLinear(q.weight.data).to(dev))
    kernel_ppl = ppl(student)
    deploy_gap = kernel_ppl - fake_quant_ppl
    dppl = kernel_ppl - dense_mlp_ppl
    print(f"PPL through-kernel (deploy): {kernel_ppl:.3f}  deploy_gap {deploy_gap:+.3f}  "
          f"dPPL_vs_dense_MLP {dppl:+.3f}", flush=True)

    # stop-gate verdict
    gate_flops = frac >= 0.25
    gate_gap = abs(deploy_gap) <= 0.02
    gate_ppl = dppl <= 0.50
    gate_speed = est_speedup >= 1.08
    verdict = "EXPAND" if (gate_flops and gate_gap and gate_ppl and gate_speed and dppl <= 0.35) else \
              ("MARGINAL" if (gate_flops and gate_ppl) else "KILL")
    print(f"RESULT mask={mask} frac={frac:.3f} est_speedup={est_speedup:.3f}x "
          f"dense_mlp_baseline={dense_mlp_ppl:.3f} fake_quant={fake_quant_ppl:.3f} "
          f"kernel={kernel_ppl:.3f} deploy_gap={deploy_gap:+.3f} dPPL={dppl:+.3f} "
          f"gates[flops>=.25={gate_flops},gap<=.02={gate_gap},dPPL<=.5={gate_ppl},speed>=1.08x={gate_speed}] "
          f"VERDICT={verdict}", flush=True)
    return kernel_ppl


@app.local_entrypoint()
def main(model: str = MODEL, mask: str = "down_proj", p1: int = 3000, p2: int = 2000,
         lr_max: float = 3e-5, p2_lr: float = 1e-5) -> None:
    call = run.spawn(model=model, mask=mask, p1=p1, p2=p2, lr_max=lr_max, p2_lr=p2_lr)
    print(f"SPAWN_ID {call.object_id}", flush=True)
