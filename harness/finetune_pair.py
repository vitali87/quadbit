"""Pair-granular 2:4 RECOVERY FINE-TUNE: make the sparse FP4 kernel production-accurate.

Pipeline: dense teacher -> SparseGPT-pair one-shot prune (mask frozen) -> knowledge-distill
the surviving weights from the dense teacher on text -> the weights stay pair-granular 2:4
(re-masked each step) -> FP4-quantize -> run through our sparse kernel. This is exactly how
Neural Magic recovered element-2:4 (distillation), retargeted to Blackwell's pair-granularity.
Reports the PPL trajectory: one-shot ~22 -> fine-tuned -> and the final number through the
actual FP4 kernel. Small model (TinyLlama-1.1B, MLP dims %256) to fit one GPU session.

Run:  uv run modal run harness/finetune_pair.py
"""

import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
MODEL = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"  # dense base; MLP 5632x2048 %256
# 8B-scale probe target: Qwen2.5-7B is ungated (Apache-2.0), MLP 18944x3584 both %256.

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64", "HF_HOME": "/cache",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})  # cut fragmentation for the 8B QAT graph
    .pip_install("torch", index_url="https://download.pytorch.org/whl/nightly/cu128", pre=True)
    .pip_install("transformers", "huggingface_hub", "safetensors", "sentencepiece", "pyarrow", "bitsandbytes")
    .add_local_dir((ROOT / "cuda").as_posix(), "/root/cuda")
)
app = modal.App("quadbit-finetune", image=image)
vol = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)


@app.function(gpu="RTX-PRO-6000", timeout=86400, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])  # HF_TOKEN for gated Llama-3
def run(model: str = MODEL, p1: int = 30000, p2: int = 2000, both_shards: bool = False,
        lr_max: float = 2e-4) -> float:  # 2e-4 tuned for TinyLlama-1.1B; 8B needs ~3e-5 (higher diverges)
    import ctypes
    import math
    import os

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
        print(c.stderr, flush=True); return
    lib = ctypes.CDLL(so)
    lib.sparse_fp4_mm.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3
    lib.quantize_act_nvfp4.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2
    dev = torch.device("cuda")

    FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
                       dtype=torch.float32, device=dev)
    BND = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], dtype=torch.float32, device=dev)
    _cc = torch.arange(128, device=dev)
    _e, _m = (_cc >> 3) & 0xf, _cc & 7
    UE4M3 = torch.where(_e == 0, _m.float() * 0.001953125,
                        (1.0 + _m.float() / 8.0) * torch.exp2((_e - 7).float()))
    _MID = (UE4M3[:-1] + UE4M3[1:]) / 2

    def q_fp4(v):
        return torch.bucketize(v.abs(), BND) | ((v < 0).long() << 3)

    def enc(s):
        return torch.bucketize(s, _MID)

    def sparse_fp4_dequant(W):  # pair-2:4 magnitude + per-16 TWO-LEVEL NVFP4 (per-out-row global gA)
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

    def enc_ue4m3_t(s):  # torch replica of the kernel's enc_ue4m3: no denormals, min-normal clamp
        mant_f, e = torch.frexp(s.clamp_min(1e-30))    # s = mant_f * 2^e, mant_f in [0.5,1)
        mm = 2.0 * mant_f                              # [1,2)
        biased = (e - 1) + 7
        mant = torch.round((mm - 1.0) * 8.0).long()    # 0..8
        carry = mant == 8
        mant = torch.where(carry, torch.zeros_like(mant), mant)
        biased = torch.where(carry, biased + 1, biased)
        code = (biased.long() << 3) | mant
        code = torch.where(biased < 1, torch.ones_like(code), code)
        code = torch.where(biased > 15, torch.full_like(code, 0x7f), code)
        code = torch.where(s >= 480.0, torch.full_like(code, 0x7f), code)
        return torch.where(s > 0, code, torch.zeros_like(code))

    def act_fp4_dequant(x):  # per-16 TWO-LEVEL NVFP4 (matches modelopt/quant_act_nv2_k): per-row
        lead = x.shape[:-1]; i = x.shape[-1]           # global gB=rowamax/2688 + per-16 local ue4m3
        b = x.to(torch.bfloat16).float().reshape(-1, i)                 # kernel reads activations as bf16
        gB = (b.abs().amax(-1, keepdim=True) / 2688.0).clamp_min(1e-30)  # 2688 = e4m3max 448 * e2m1max 6
        bb = b.reshape(b.shape[0], i // 16, 16)
        sdeq = UE4M3[enc_ue4m3_t((bb.abs().amax(-1) / 6.0) / gB)] * gB
        return (FP4[q_fp4(bb / sdeq[..., None])] * sdeq[..., None]).reshape(*lead, i)

    class QATLinear(nn.Module):  # phase1: plain masked bf16 (fast); phase2 (qat=True): FP4 STE
        def __init__(self, weight):
            super().__init__()
            self.weight = nn.Parameter(weight.clone())
            self.qat = False

        def forward(self, x):
            if not self.qat:
                return F.linear(x, self.weight)                # masked bf16 (weight kept sparse)
            Wf = self.weight.float()
            Wq = Wf + (sparse_fp4_dequant(Wf) - Wf).detach()   # STE weight fake-quant
            xf = x.float()
            xq = xf + (act_fp4_dequant(xf) - xf).detach()      # STE activation fake-quant
            return F.linear(xq, Wq).to(x.dtype)

    class QuadbitLinear(nn.Module):
        def __init__(self, W):
            super().__init__()
            out_f, in_f = W.shape
            ks = in_f // 128
            self.out_f, self.in_f, self.ks = out_f, in_f, ks
            Wg = W.float().to(dev).view(out_f, ks, 16, 4, 2)
            i01, _ = Wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
            keptW = torch.gather(Wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
            blk = keptW.reshape(out_f, ks, 4, 8, 2)
            scode = enc(blk.abs().amax(dim=(3, 4)) / 6.0)
            sdeq = UE4M3[scode]
            kc = q_fp4(blk / sdeq[..., None, None])
            Ac = (kc[..., 0] | (kc[..., 1] << 4)).reshape(out_f, ks * 32).to(torch.uint8)
            nib = (i01[..., 0] | (i01[..., 1] << 2)).view(out_f, ks, 2, 8)
            sh = (torch.arange(8, device=dev) * 4).view(1, 1, 1, 8)
            meta = (nib << sh).sum(-1).to(torch.int32).permute(1, 0, 2).contiguous()
            self.register_buffer("Ac", Ac.contiguous())
            self.register_buffer("meta", meta)
            self.register_buffer("scaleA", scode.to(torch.uint8).permute(1, 0, 2).contiguous())

        def forward(self, x):
            lead = x.shape[:-1]
            x2 = x.reshape(-1, self.in_f).to(torch.bfloat16)
            t = x2.shape[0]; pad = (-t) % 128
            if pad:
                x2 = torch.cat([x2, x2.new_zeros(pad, self.in_f)], 0)
            x2 = x2.contiguous(); tp = t + pad
            Bb = torch.empty((tp, self.in_f // 2), dtype=torch.uint8, device=dev)
            sB = torch.empty((self.ks, tp, 4), dtype=torch.uint8, device=dev)
            lib.quantize_act_nvfp4(x2.data_ptr(), Bb.data_ptr(), sB.data_ptr(), tp, self.in_f)
            C = torch.empty((self.out_f, tp), dtype=torch.bfloat16, device=dev)
            lib.sparse_fp4_mm(self.Ac.data_ptr(), Bb.data_ptr(), self.scaleA.data_ptr(),
                              sB.data_ptr(), self.meta.data_ptr(), C.data_ptr(),
                              self.out_f, tp, self.in_f)
            return C.t()[:t].reshape(*lead, self.out_f).to(x.dtype)

    def sparsegpt_pair24(W, H, blocksize=128, percdamp=0.01):
        W = W.float().clone()
        cols = W.shape[1]
        d = torch.diag(H) == 0
        H[d, d] = 1.0; W[:, d] = 0.0
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
                    pm = tmp[:, 0::2] + tmp[:, 1::2]
                    pr = pm.topk(2, dim=1, largest=False).indices
                    cm = torch.ones(rows, 8, dtype=torch.bool, device=dev)
                    cm[ri[:, None], pr * 2] = False; cm[ri[:, None], pr * 2 + 1] = False
                w = W1[:, j]; q = torch.where(cm[:, j % 8], w, torch.zeros_like(w))
                err = (w - q) / dinv[j]
                W1[:, j:] -= err[:, None] * H1[j, j:][None, :]
                Err[:, j] = err; W1[:, j] = q
            W[:, i:e] = W1; W[:, e:] -= Err @ Hinv[i:e, e:]
        return W

    tok = AutoTokenizer.from_pretrained(model)

    def load():
        return AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).to(dev)

    def wikitext(cfg, fn):
        p = hf_hub_download("Salesforce/wikitext", f"{cfg}/{fn}", repo_type="dataset")
        txt = "\n\n".join(pq.read_table(p).column("text").to_pylist())[:200_000_000]
        return tok(txt, return_tensors="pt").input_ids[0]

    test_ids = wikitext("wikitext-2-raw-v1", "test-00000-of-00001.parquet")   # comparable eval
    train_ids = wikitext("wikitext-103-raw-v1", "train-00000-of-00002.parquet")  # big, no overfit
    if both_shards:  # real-data-scale probe: full wikitext-103 train (both shards)
        train_ids = torch.cat([train_ids, wikitext("wikitext-103-raw-v1", "train-00001-of-00002.parquet")])
    print(f"train tokens: {len(train_ids)}", flush=True)

    def ppl(m, windows=16, seq=2048):
        m.eval()
        nll, n = 0.0, 0
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

    teacher = load().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"PPL dense fp16 teacher: {ppl(teacher):.3f}", flush=True)

    student = load()
    # phase-2 QAT builds the full two-level fake-quant graph per MLP layer; without this the 8B
    # activation graph + AdamW states OOM a 96GB card. Recompute activations in backward instead.
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.config.use_cache = False

    T, seq = 2.0, 1024
    P1, P2, wu = p1, p2, 500  # phase-1 full (still descending); phase-2 QAT converges by ~1k steps
    total = P1 + P2
    lr_min = lr_max / 20.0
    starts = list(range(0, len(train_ids) - seq, seq))
    targets = list(mlp_lins(student))  # (mlp, name, lin) in deterministic order; checkpoint key basis
    sh_tag = "_2sh" if both_shards else ""
    ck = f"/cache/phase1_{model.split('/')[-1]}_P{P1}{sh_tag}_lr{lr_max:.0e}.pt"  # phase-1 cached on volume (lr in key)

    CKPT_EVERY = 2500  # periodic phase-1 save so an infra preemption mid-30k doesn't lose hours
    masks, qats, start_step = {}, [], 0
    if os.path.exists(ck):
        ckd = torch.load(ck, map_location=dev, weights_only=True)  # {"step", "weights"} (targets order)
        start_step = int(ckd["step"])
        for (mlp, nm, _), W in zip(targets, ckd["weights"]):
            qat = QATLinear(W.to(torch.bfloat16)).to(dev)
            masks[qat] = (W != 0).to(dev); setattr(mlp, nm, qat); qats.append(qat)
        print(f"resumed phase-1 checkpoint {ck} at step {start_step}", flush=True)
    else:
        # H calibration on the student MLP inputs, then SparseGPT-pair prune + freeze mask
        Hs, ns = {}, {}

        def hook(mod, inp, _o):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1]); k = id(mod); nn_ = x.shape[0]
            if k not in Hs:
                Hs[k] = torch.zeros(x.shape[1], x.shape[1], device=dev); ns[k] = 0
            Hs[k] *= ns[k] / (ns[k] + nn_)
            Hs[k] += (x * math.sqrt(2.0 / (ns[k] + nn_))).t() @ (x * math.sqrt(2.0 / (ns[k] + nn_)))
            ns[k] += nn_

        hs = [lin.register_forward_hook(hook) for _, _, lin in targets]
        with torch.no_grad():
            for i in range(0, 16 * 2048, 2048):
                student(train_ids[i:i + 2048].unsqueeze(0).to(dev))
        for h in hs:
            h.remove()
        for mlp, nm, lin in targets:
            Wp = sparsegpt_pair24(lin.weight.data, Hs.pop(id(lin)))
            qat = QATLinear(Wp.to(torch.bfloat16)).to(dev)
            masks[qat] = (Wp != 0).to(dev); setattr(mlp, nm, qat); qats.append(qat)
        # student forward runs through the FP4 fake-quant, so this == the kernel's one-shot PPL
        print(f"PPL one-shot pair-2:4 FP4 (QAT-equivalent): {ppl(student):.3f}", flush=True)

    # KD recovery: only surviving MLP weights, re-mask each step. Phase 1 = fast bf16 masked
    # (recovers the sparsity, the big error); Phase 2 = QAT to adapt to weight+act FP4.
    for p in student.parameters():
        p.requires_grad_(False)
    params = []
    for q in qats:
        q.weight.requires_grad_(True); params.append(q.weight)
    import bitsandbytes as bnb  # 8-bit AdamW: optimizer states ~11GB vs bf16's ~23GB -> fits 8B in 96GB
    opt = bnb.optim.AdamW8bit(params, lr=lr_max, betas=(0.9, 0.95), weight_decay=0.0)

    def kd_step(step):
        for g in opt.param_groups:                    # warmup + cosine decay
            if step < wu:
                g["lr"] = lr_max * (step + 1) / wu
            else:
                t = (step - wu) / max(1, total - wu)
                g["lr"] = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t))
        i = starts[(step * 2654435761) % len(starts)]  # shuffled window pick
        w = train_ids[i:i + seq].unsqueeze(0).to(dev)
        with torch.no_grad():
            tl = teacher(w).logits.reshape(-1, teacher.config.vocab_size)
        sl = student(w).logits.reshape(-1, teacher.config.vocab_size)
        loss = F.kl_div(F.log_softmax(sl / T, -1), F.softmax(tl / T, -1), reduction="batchmean") * (T * T)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        with torch.no_grad():
            for q, mk in masks.items():
                q.weight.data *= mk
        return loss.item()

    def save_ck(step):
        torch.save({"step": step, "weights": [q.weight.data.cpu() for q in qats]}, ck); vol.commit()

    if start_step < P1:
        student.train()
        for step in range(start_step, P1):            # phase 1: bf16 masked (fast), resumable
            loss = kd_step(step)
            if (step + 1) % 1000 == 0:
                print(f"  P1 {step + 1}/{P1} KD {loss:.4f} PPL(bf16) {ppl(student, windows=8):.3f}", flush=True)
                student.train()
            if (step + 1) % CKPT_EVERY == 0:
                save_ck(step + 1)                     # periodic: resume from here if preempted
        save_ck(P1)                                   # phase-1 done marker
        print(f"saved phase-1 checkpoint {ck}", flush=True)
    print(f"PPL after phase-1 bf16 recovery: {ppl(student):.3f}", flush=True)

    for q in qats:
        q.qat = True                                  # phase 2: QAT (weight+act FP4 STE)
    student.train()
    for step in range(P1, total):
        loss = kd_step(step)
        if (step + 1) % 500 == 0:
            print(f"  P2 {step + 1 - P1}/{P2} KD {loss:.4f} PPL(FP4) {ppl(student, windows=8):.3f}", flush=True)
            student.train()
    print(f"PPL after phase-2 QAT recovery (FP4): {ppl(student):.3f}", flush=True)

    # final: build the actual sparse-FP4 KERNEL modules from the fine-tuned weights
    for layer in student.model.layers:
        for nm in ("gate_proj", "up_proj", "down_proj"):
            q = getattr(layer.mlp, nm)
            setattr(layer.mlp, nm, QuadbitLinear(q.weight.data).to(dev))
    kernel_ppl = ppl(student)
    print(f"PPL through 2:4-sparse FP4 KERNEL (final): {kernel_ppl:.3f}", flush=True)
    return kernel_ppl


@app.local_entrypoint()
def main(model: str = MODEL, p1: int = 30000, p2: int = 2000, both_shards: bool = False,
         lr_max: float = 2e-4) -> None:
    # spawn (not remote): compute survives local-client disconnect. See memory.
    call = run.spawn(model=model, p1=p1, p2=p2, both_shards=both_shards, lr_max=lr_max)
    print(f"SPAWN_ID {call.object_id}", flush=True)
    print(f"RESULT {call.get():.3f}", flush=True)  # blocks; if this waiter dies, recover via SPAWN_ID
