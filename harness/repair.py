"""Workstream A: accuracy repair tournament for the all-sparse split-K serving row.

Goal: shrink the +2.3 PPL sparse tax (through-kernel ~10.03, serving 10.27 vs production NVFP4 7.97)
while keeping the serving wins. Four modes, all on Llama-3.1-8B-Instruct + the recovered-Instruct
sparse MLP checkpoint (the banked all-sparse student). Every mode reports through-fake-quant PPL and,
where a deploy artifact exists, through-KERNEL PPL (what serving actually runs).

  calib   (A1): zero-runtime per-layer scalar (y = a_l * y_sparse) and per-channel affine
                (y_j = a_lj * y_sparse_j + b_lj) MLP-output correction, closed-form least squares vs a
                dense-NVFP4 teacher MLP evaluated on the STUDENT's own hidden states. Foldable into a
                graph-capturable elementwise op; no training, no runtime tensor-core cost.
  lowrank (A2): low-rank residual adapters after each sparse MLP block, y = sparse_mlp(x) + U_l(V_l x).
                KD-train adapters (sparse weights frozen). One rank per launch (--rank 16/32/64).
  mask    (A3): activation-aware (Wanda-pair) 2:4 masks recomputed from the ORIGINAL weights, then QAT
                recovery; PPL logged at 0/500/1k/2k steps so the launcher can stop on a flat curve.
  distill (A4): KD (logits KL + CE + optional MLP-output MSE), train sparse nonzero weights + a
                learnable per-out-channel scale, from the recovered init; eval at 500/1k/2k/5k.

Run (all four in parallel, separate GPUs):
  uv run modal run --detach harness/repair.py --mode calib
  uv run modal run --detach harness/repair.py --mode lowrank --rank 32
  uv run modal run --detach harness/repair.py --mode mask   --steps 2000
  uv run modal run --detach harness/repair.py --mode distill --steps 5000
"""

import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
RECOVERED_INSTRUCT_CKPT = "/cache/recovered_Llama-3.1-8B-Instruct_P30000_p25000_2sh_lr3e-05.pt"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64", "HF_HOME": "/cache",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .pip_install("torch", index_url="https://download.pytorch.org/whl/nightly/cu128", pre=True)
    .pip_install("transformers", "huggingface_hub", "safetensors", "sentencepiece", "pyarrow", "bitsandbytes", "numpy")
    .add_local_dir((ROOT / "cuda").as_posix(), "/root/cuda")
)
app = modal.App("quadbit-repair", image=image)
vol = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)


@app.function(gpu="RTX-PRO-6000", timeout=86400, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(mode: str = "calib", model: str = MODEL, recovered_ckpt: str = RECOVERED_INSTRUCT_CKPT,
        rank: int = 32, steps: int = 2000, p1: int = 2000, lr: float = 2e-5, mse_w: float = 0.0,
        calib_windows: int = 32) -> float:
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

    class SparseQAT(nn.Module):  # trainable, pruned, sparse two-level STE (train == deploy); optional learnable per-out scale
        def __init__(self, weight, learn_scale=False):
            super().__init__()
            self.weight = nn.Parameter(weight.clone()); self.qat = True
            self.scale = nn.Parameter(torch.ones(weight.shape[0], device=dev)) if learn_scale else None

        def forward(self, x):
            if not self.qat:
                y = F.linear(x, self.weight)
            else:
                Wf = self.weight.float()
                Wq = Wf + (sparse_fp4_dequant(Wf) - Wf).detach()
                xf = x.float(); xq = xf + (act_fp4_dequant(xf) - xf).detach()
                y = F.linear(xq, Wq).to(x.dtype)
            if self.scale is not None:
                y = y * self.scale.to(y.dtype)
            return y

    class QuadbitLinear(nn.Module):  # sparse two-level KERNEL (through-kernel deploy eval); optional folded out-scale
        def __init__(self, W, scale=None):
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
            self.register_buffer("oscale", (scale if scale is not None else torch.ones(out_f, device=dev)).float())

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
            y = (C.t()[:t] * self.oscale.to(C.dtype)).reshape(*lead, self.out_f)
            return y.to(x.dtype)

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

    def wanda_pair24(W, colnorm):  # activation-aware pair-2:4: keep 2 of 4 pairs by sum_pair |W|*||X_col||
        out_f, in_f = W.shape; ks = in_f // 128
        score = (W.float().abs() * colnorm.view(1, in_f)).view(out_f, ks, 16, 4, 2).sum(-1)  # [out,ks,16,4]
        keep = score.topk(2, dim=-1).indices  # top-2 of the 4 pairs
        m = torch.zeros(out_f, ks, 16, 4, dtype=torch.bool, device=dev)
        m.scatter_(-1, keep, True)
        return (W.view(out_f, ks, 16, 4, 2) * m.unsqueeze(-1)).reshape(out_f, in_f)

    tok = AutoTokenizer.from_pretrained(model)

    def load():
        return AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).to(dev)

    def wikitext(cfg, fn):
        p = hf_hub_download("Salesforce/wikitext", f"{cfg}/{fn}", repo_type="dataset")
        return tok("\n\n".join(pq.read_table(p).column("text").to_pylist())[:200_000_000], return_tensors="pt").input_ids[0]

    test_ids = wikitext("wikitext-2-raw-v1", "test-00000-of-00001.parquet")
    train_ids = wikitext("wikitext-103-raw-v1", "train-00000-of-00002.parquet")

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

    rec = torch.load(recovered_ckpt, map_location="cpu", weights_only=True)["weights"]  # [gate,up,down]*32
    print(f"MODE={mode} model={model} recovered_ckpt={recovered_ckpt} rank={rank} steps={steps}", flush=True)

    teacher = load().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"PPL dense bf16 teacher: {ppl(teacher):.4f}", flush=True)

    # ---- Workstream A3: recompute masks from ORIGINAL weights (activation-aware), QAT recover, save ckpt.
    if mode == "mask":
        student = load()
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        student.config.use_cache = False
        targets = list(mlp_lins(student))
        # calibrate per-column activation L2 (Wanda) for each MLP linear
        col2 = {}

        def hook(mod, inp, _o):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1]); k = id(mod)
            s = (x * x).sum(0)
            col2[k] = s if k not in col2 else col2[k] + s

        hs = [lin.register_forward_hook(hook) for _, _, lin in targets]
        with torch.no_grad():
            for i in range(0, calib_windows * 2048, 2048):
                student(train_ids[i:i + 2048].unsqueeze(0).to(dev))
        for h in hs:
            h.remove()
        masks_d, sparse_qats = {}, []
        for mlp, nm, lin in targets:
            colnorm = col2[id(lin)].sqrt().clamp_min(1e-8)
            Wp = wanda_pair24(lin.weight.data, colnorm)
            q = SparseQAT(Wp.to(torch.bfloat16)).to(dev)
            masks_d[q] = (Wp != 0).to(dev); setattr(mlp, nm, q); sparse_qats.append(q)
        print(f"PPL one-shot Wanda-pair (pre-QAT, fake-quant): {ppl(student):.4f}", flush=True)
        for p in student.parameters():
            p.requires_grad_(False)
        params = []
        for q in sparse_qats:
            q.weight.requires_grad_(True); params.append(q.weight)
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
        T, seq, wu = 2.0, 1024, 200
        starts = list(range(0, len(train_ids) - seq, seq))
        student.train()
        for step in range(steps):
            for g in opt.param_groups:
                g["lr"] = (lr * (step + 1) / wu if step < wu else
                           lr / 20 + 0.5 * (lr - lr / 20) * (1 + math.cos(math.pi * (step - wu) / max(1, steps - wu))))
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
            if (step + 1) in (500, 1000, 2000, 3000, 5000) or (step + 1) == steps:
                fq = ppl(student, windows=16)
                print(f"  mask QAT {step + 1}/{steps} KD {kl.item():.4f} PPL(fake-quant) {fq:.4f}", flush=True)
                student.train()
        weights = [q.weight.data.to(torch.bfloat16).cpu() for q in sparse_qats]
        outp = f"/cache/repair_mask_wanda_{steps}.pt"
        torch.save({"weights": weights}, outp)
        for (mlp, nm, _lin), q in zip(targets, sparse_qats):
            setattr(mlp, nm, QuadbitLinear(q.weight.data).to(dev))
        kp = ppl(student)
        print(f"RESULT mode=mask steps={steps} kernel_ppl={kp:.4f} ckpt={outp}", flush=True)
        return kp

    # ---- calib / lowrank / distill all start from the recovered sparse student.
    student = load()
    targets = list(mlp_lins(student))
    assert len(targets) == len(rec), f"targets {len(targets)} != rec {len(rec)}"
    layer_of = {}  # sparse module -> layer index (down output correction is per llama layer)
    sparse_qats = []
    for idx, (mlp, nm, lin) in enumerate(targets):
        learn = (mode == "distill" and nm == "down_proj")
        q = SparseQAT(rec[idx].to(dev).to(torch.bfloat16), learn_scale=learn).to(dev)
        setattr(mlp, nm, q); sparse_qats.append(q); layer_of[id(mlp)] = idx // 3
    base_fq = ppl(student)
    print(f"PPL all-sparse recovered (fake-quant, baseline to beat): {base_fq:.4f}", flush=True)

    if mode == "calib":
        # dense-NVFP4 teacher MLP on the STUDENT's own hidden states, per llama layer. The student's
        # gate/up/down are now SparseQAT, so read the original dense weights from the frozen teacher.
        dense_dq = {}
        for li, layer in enumerate(teacher.model.layers):
            dense_dq[li] = (dense_fp4_dequant(layer.mlp.gate_proj.weight.data.float()),
                            dense_fp4_dequant(layer.mlp.up_proj.weight.data.float()),
                            dense_fp4_dequant(layer.mlp.down_proj.weight.data.float()))

        def teacher_block(x, li):
            g, u, d = dense_dq[li]
            xq = act_fp4_dequant(x)
            h = F.silu(F.linear(xq, g)) * F.linear(xq, u)
            return F.linear(act_fp4_dequant(h), d)

        NL = len(student.model.layers)
        H = student.config.hidden_size
        S1 = torch.zeros(NL, H, device=dev); T1 = torch.zeros(NL, H, device=dev)
        SS = torch.zeros(NL, H, device=dev); ST = torch.zeros(NL, H, device=dev); NCT = 0

        def cap_hook(mlp, inp, out):
            nonlocal NCT
            li = layer_of[id(mlp)]
            x = inp[0].detach()
            s = out.detach().float().reshape(-1, H)
            t = teacher_block(x, li).float().reshape(-1, H)
            S1[li] += s.sum(0); T1[li] += t.sum(0)
            SS[li] += (s * s).sum(0); ST[li] += (s * t).sum(0)
            if li == 0:
                NCT += s.shape[0]

        hooks = [layer.mlp.register_forward_hook(cap_hook) for layer in student.model.layers]
        with torch.no_grad():
            for i in range(0, calib_windows * 2048, 2048):
                student(train_ids[i:i + 2048].unsqueeze(0).to(dev))
        for h in hooks:
            h.remove()
        N = float(NCT)
        denom = (N * SS - S1 * S1).clamp_min(1e-8)
        alpha = (N * ST - S1 * T1) / denom             # per-channel affine slope
        beta = (T1 - alpha * S1) / N                    # per-channel affine intercept
        alpha_s = (ST.sum(1) / SS.sum(1).clamp_min(1e-8)).view(NL, 1)  # per-layer scalar (beta=0)

        # correction wrapper on each layer's mlp output
        corr = {"mode": "none", "a": None, "b": None}
        NLm = {id(layer.mlp): li for li, layer in enumerate(student.model.layers)}

        def corr_hook(mlp, inp, out):
            if corr["a"] is None:
                return out
            li = NLm[id(mlp)]
            a = corr["a"][li].to(out.dtype)
            y = out * a
            if corr["b"] is not None:
                y = y + corr["b"][li].to(out.dtype)
            return y

        chooks = [layer.mlp.register_forward_hook(corr_hook) for layer in student.model.layers]
        corr["a"] = None; ppl_none = ppl(student)
        corr["a"] = alpha_s; corr["b"] = None; ppl_scalar = ppl(student)
        corr["a"] = alpha; corr["b"] = beta; ppl_affine = ppl(student)
        for h in chooks:
            h.remove()
        print(f"PPL calib: none={ppl_none:.4f} scalar_alpha={ppl_scalar:.4f} perchan_affine={ppl_affine:.4f}", flush=True)

        # through-KERNEL deploy PPL with the best (affine) correction folded onto the mlp output.
        for idx, (mlp, nm, _lin) in enumerate(targets):
            q = getattr(mlp, nm)
            setattr(mlp, nm, QuadbitLinear(q.weight.data).to(dev))
        kbest = min((ppl_none, "none"), (ppl_scalar, "scalar"), (ppl_affine, "affine"))
        ab = {"none": (None, None), "scalar": (alpha_s, None), "affine": (alpha, beta)}[kbest[1]]
        corr["a"], corr["b"] = ab
        khooks = [layer.mlp.register_forward_hook(corr_hook) for layer in student.model.layers]
        kernel_ppl = ppl(student)
        for h in khooks:
            h.remove()
        outp = "/cache/repair_calib_affine.pt"
        torch.save({"alpha": alpha.cpu(), "beta": beta.cpu(), "alpha_scalar": alpha_s.cpu()}, outp)
        print(f"RESULT mode=calib base_fq={base_fq:.4f} none={ppl_none:.4f} scalar={ppl_scalar:.4f} "
              f"affine={ppl_affine:.4f} best={kbest[1]} kernel_ppl={kernel_ppl:.4f} ckpt={outp}", flush=True)
        return kernel_ppl

    if mode == "lowrank":
        class Adapter(nn.Module):
            def __init__(self, mlp, li, r):
                super().__init__()
                self.mlp = mlp; self.li = li
                self.V = nn.Linear(H_, r, bias=False, device=dev, dtype=torch.bfloat16)
                self.U = nn.Linear(r, H_, bias=False, device=dev, dtype=torch.bfloat16)
                nn.init.normal_(self.V.weight, std=1e-3); nn.init.zeros_(self.U.weight)

            def forward(self, x):
                return self.mlp(x) + self.U(self.V(x))

        H_ = student.config.hidden_size
        adapters = []
        for li, layer in enumerate(student.model.layers):
            a = Adapter(layer.mlp, li, rank).to(dev); layer.mlp = a; adapters.append(a)
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        student.config.use_cache = False
        for p in student.parameters():
            p.requires_grad_(False)
        params = []
        for a in adapters:
            for p in (a.U.weight, a.V.weight):
                p.requires_grad_(True); params.append(p)
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
        T, seq, wu = 2.0, 1024, 200
        starts = list(range(0, len(train_ids) - seq, seq))
        student.train()
        for step in range(steps):
            for g in opt.param_groups:
                g["lr"] = (lr * (step + 1) / wu if step < wu else
                           lr / 20 + 0.5 * (lr - lr / 20) * (1 + math.cos(math.pi * (step - wu) / max(1, steps - wu))))
            i = starts[(step * 2654435761) % len(starts)]
            w = train_ids[i:i + seq].unsqueeze(0).to(dev)
            with torch.no_grad():
                tl = teacher(w).logits.reshape(-1, teacher.config.vocab_size)
            logits = student(w).logits.reshape(-1, teacher.config.vocab_size)
            kl = F.kl_div(F.log_softmax(logits / T, -1), F.softmax(tl / T, -1), reduction="batchmean") * (T * T)
            opt.zero_grad(); kl.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            if (step + 1) in (500, 1000, 2000) or (step + 1) == steps:
                fq = ppl(student, windows=16)
                print(f"  lowrank r={rank} {step + 1}/{steps} KD {kl.item():.4f} PPL(fake-quant) {fq:.4f}", flush=True)
                student.train()
        adapter_params = sum(p.numel() for p in params)
        # through-KERNEL: swap SparseQAT inside each adapter's wrapped mlp for QuadbitLinear
        for a in adapters:
            for nm in ("gate_proj", "up_proj", "down_proj"):
                q = getattr(a.mlp, nm); setattr(a.mlp, nm, QuadbitLinear(q.weight.data).to(dev))
        kp = ppl(student)
        outp = f"/cache/repair_lowrank_r{rank}_{steps}.pt"
        torch.save({"U": [a.U.weight.data.cpu() for a in adapters],
                    "V": [a.V.weight.data.cpu() for a in adapters], "rank": rank}, outp)
        print(f"RESULT mode=lowrank rank={rank} steps={steps} base_fq={base_fq:.4f} kernel_ppl={kp:.4f} "
              f"adapter_params={adapter_params} extra_MiB={adapter_params * 2 / 2**20:.1f} ckpt={outp}", flush=True)
        return kp

    if mode == "distill":
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        student.config.use_cache = False
        masks_d = {q: (q.weight.data != 0).to(dev) for q in sparse_qats}
        for p in student.parameters():
            p.requires_grad_(False)
        params = []
        for q in sparse_qats:
            q.weight.requires_grad_(True); params.append(q.weight)
            if q.scale is not None:
                q.scale.requires_grad_(True); params.append(q.scale)
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
        T, seq, wu = 2.0, 1024, 200
        starts = list(range(0, len(train_ids) - seq, seq))
        student.train()
        best = 1e9
        for step in range(steps):
            for g in opt.param_groups:
                g["lr"] = (lr * (step + 1) / wu if step < wu else
                           lr / 20 + 0.5 * (lr - lr / 20) * (1 + math.cos(math.pi * (step - wu) / max(1, steps - wu))))
            i = starts[(step * 2654435761) % len(starts)]
            w = train_ids[i:i + seq].unsqueeze(0).to(dev)
            with torch.no_grad():
                tl = teacher(w).logits.reshape(-1, teacher.config.vocab_size)
            logits = student(w).logits.reshape(-1, teacher.config.vocab_size)
            kl = F.kl_div(F.log_softmax(logits / T, -1), F.softmax(tl / T, -1), reduction="batchmean") * (T * T)
            ce = F.cross_entropy(logits[:-1], w.reshape(-1)[1:])
            loss = kl + 0.1 * ce
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            with torch.no_grad():
                for q, mk in masks_d.items():
                    q.weight.data *= mk
            if (step + 1) in (500, 1000, 2000, 3000, 5000) or (step + 1) == steps:
                fq = ppl(student, windows=16)
                print(f"  distill {step + 1}/{steps} KL {kl.item():.4f} CE {ce.item():.4f} PPL(fake-quant) {fq:.4f}", flush=True)
                if fq < best:
                    best = fq
                    torch.save({"weights": [q.weight.data.to(torch.bfloat16).cpu() for q in sparse_qats],
                                "scales": [q.scale.data.cpu() if q.scale is not None else None for q in sparse_qats]},
                               f"/cache/repair_distill_{steps}.pt")
                student.train()
        for (mlp, nm, _lin), q in zip(targets, sparse_qats):
            setattr(mlp, nm, QuadbitLinear(q.weight.data, scale=q.scale.data if q.scale is not None else None).to(dev))
        kp = ppl(student)
        print(f"RESULT mode=distill steps={steps} base_fq={base_fq:.4f} best_fq={best:.4f} kernel_ppl={kp:.4f} "
              f"ckpt=/cache/repair_distill_{steps}.pt", flush=True)
        return kp

    print(f"unknown mode {mode}", flush=True)
    return -1.0


@app.local_entrypoint()
def main(mode: str = "calib", model: str = MODEL, recovered_ckpt: str = RECOVERED_INSTRUCT_CKPT,
         rank: int = 32, steps: int = 2000, p1: int = 2000, lr: float = 2e-5, mse_w: float = 0.0,
         calib_windows: int = 32) -> None:
    call = run.spawn(mode=mode, model=model, recovered_ckpt=recovered_ckpt, rank=rank, steps=steps,
                     p1=p1, lr=lr, mse_w=mse_w, calib_windows=calib_windows)
    print(f"SPAWN_ID {call.object_id}", flush=True)
