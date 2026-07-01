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

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64", "HF_HOME": "/cache"})
    .pip_install("torch", index_url="https://download.pytorch.org/whl/nightly/cu128", pre=True)
    .pip_install("transformers", "huggingface_hub", "safetensors", "sentencepiece", "pyarrow")
    .add_local_dir((ROOT / "cuda").as_posix(), "/root/cuda")
)
app = modal.App("quadbit-finetune", image=image)
vol = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)


@app.function(gpu="RTX-PRO-6000", timeout=3600, volumes={"/cache": vol})
def run() -> None:
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

    def sparse_fp4_dequant(W):  # EXACT kernel dequant: pair-2:4 magnitude + FP4, returns float
        out_f, in_f = W.shape; ks = in_f // 128
        Wg = W.view(out_f, ks, 16, 4, 2)
        i01, _ = Wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
        keptW = torch.gather(Wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
        blk = keptW.reshape(out_f, ks, 4, 8, 2)
        sdeq = UE4M3[enc(blk.abs().amax(dim=(3, 4)) / 6.0)]
        kd = (FP4[q_fp4(blk / sdeq[..., None, None])] * sdeq[..., None, None]).reshape(out_f, ks, 16, 2, 2)
        Wd = torch.zeros(out_f, ks, 16, 4, 2, device=dev)
        Wd.scatter_(3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2), kd)
        return Wd.reshape(out_f, in_f)

    def act_fp4_dequant(x):  # per-32-block NVFP4 fake-quant of activations (what the kernel does)
        lead = x.shape[:-1]; i = x.shape[-1]
        b = x.reshape(-1, i // 32, 32)
        s = UE4M3[enc(b.abs().amax(-1) / 6.0)]
        return (FP4[q_fp4(b / s[..., None])] * s[..., None]).reshape(*lead, i)

    class QATLinear(nn.Module):  # trains against the EXACT sparse-FP4 kernel: weight AND act quant
        def __init__(self, weight):
            super().__init__()
            self.weight = nn.Parameter(weight.clone())

        def forward(self, x):
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

    tok = AutoTokenizer.from_pretrained(MODEL)

    def load():
        return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)

    def wikitext(split):
        p = hf_hub_download("Salesforce/wikitext", f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet",
                            repo_type="dataset")
        return tok("\n\n".join(pq.read_table(p).column("text").to_pylist()), return_tensors="pt").input_ids[0]

    test_ids, train_ids = wikitext("test"), wikitext("train")

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

    # H calibration on the student MLP inputs, then SparseGPT-pair prune + freeze mask
    Hs, ns = {}, {}

    def hook(mod, inp, _o):
        x = inp[0].detach().float().reshape(-1, inp[0].shape[-1]); k = id(mod); nn_ = x.shape[0]
        if k not in Hs:
            Hs[k] = torch.zeros(x.shape[1], x.shape[1], device=dev); ns[k] = 0
        Hs[k] *= ns[k] / (ns[k] + nn_)
        Hs[k] += (x * math.sqrt(2.0 / (ns[k] + nn_))).t() @ (x * math.sqrt(2.0 / (ns[k] + nn_)))
        ns[k] += nn_

    hs = [lin.register_forward_hook(hook) for _, _, lin in mlp_lins(student)]
    with torch.no_grad():
        for i in range(0, 16 * 2048, 2048):
            student(train_ids[i:i + 2048].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()

    # SparseGPT-pair prune, then wrap each MLP linear in a QAT linear (forward = exact FP4 dequant)
    masks, qats = {}, []
    for mlp, nm, lin in mlp_lins(student):
        Wp = sparsegpt_pair24(lin.weight.data, Hs.pop(id(lin)))
        qat = QATLinear(Wp.to(torch.bfloat16)).to(dev)
        masks[qat] = (Wp != 0).to(dev)
        setattr(mlp, nm, qat); qats.append(qat)
    # student forward now runs through the FP4 fake-quant, so this == the kernel's one-shot PPL
    print(f"PPL one-shot pair-2:4 FP4 (QAT-equivalent): {ppl(student):.3f}", flush=True)

    # knowledge-distillation recovery: train only surviving MLP weights, re-mask each step
    for p in student.parameters():
        p.requires_grad_(False)
    params = []
    for q in qats:
        q.weight.requires_grad_(True); params.append(q.weight)
    opt = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0)
    T, seq, steps = 2.0, 1024, 1500
    starts = list(range(0, len(train_ids) - seq, seq))
    student.train()
    for step in range(steps):
        w = train_ids[starts[step % len(starts)]:starts[step % len(starts)] + seq].unsqueeze(0).to(dev)
        with torch.no_grad():
            tl = teacher(w).logits.reshape(-1, teacher.config.vocab_size)
        sl = student(w).logits.reshape(-1, teacher.config.vocab_size)
        loss = F.kl_div(F.log_softmax(sl / T, -1), F.softmax(tl / T, -1), reduction="batchmean") * (T * T)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        with torch.no_grad():
            for q, mk in masks.items():
                q.weight.data *= mk             # keep pair-2:4 structure exact
        if (step + 1) % 300 == 0:
            print(f"  step {step + 1}/{steps} KD {loss.item():.4f}  PPL {ppl(student, windows=8):.3f}", flush=True)
            student.train()

    print(f"PPL after QAT recovery fine-tune (FP4): {ppl(student):.3f}", flush=True)

    # final: build the actual sparse-FP4 KERNEL modules from the fine-tuned weights
    for layer in student.model.layers:
        for nm in ("gate_proj", "up_proj", "down_proj"):
            q = getattr(layer.mlp, nm)
            setattr(layer.mlp, nm, QuadbitLinear(q.weight.data).to(dev))
    print(f"PPL through 2:4-sparse FP4 KERNEL (final): {ppl(student):.3f}", flush=True)


@app.local_entrypoint()
def main() -> None:
    run.remote()
