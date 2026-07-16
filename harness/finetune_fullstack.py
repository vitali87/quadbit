"""Gap C: FULL-STACK capability QAT for 2:4-sparse NVFP4.

Difference from finetune_pair.py (which recovers PPL only): this fork attacks downstream CAPABILITY,
the paper's open frontier. Two load-bearing changes over every prior recovery run, plus honest
checkpoint selection:

  1. WIDEN the sparsified + trainable + matched-STE set from MLP-only to MLP + attention q/k/v/o.
     Prior runs froze attention dense; the in-context reasoning ARC/HellaSwag test lives there.
  2. CAPABILITY corpus, not web text. Pass --corpus /cache/corpus_capability_llama3 (built by
     build_capability_corpus.py from the downstream TRAIN splits). WikiText/C4 provably buy PPL not
     capability (docs/paper_notes.md:623-630); this is in-distribution KD signal.
  3. SELECT on downstream, not PPL. PPL and capability diverge, so a capped 0-shot multiple-choice
     scorer picks the best-capability checkpoint; the final publication number still goes through the
     canonical lm-eval downstream_eval.py on full test sets.

Everything else (SparseGPT-pair mask, matched two-level STE == cuda/sparse_fp4_lib.cu, 8-bit AdamW,
gradient checkpointing, phase-1 bf16 -> phase-2 QAT warm-restart, resume) is inherited unchanged.

Run:  uv run modal run --detach harness/finetune_fullstack.py --model meta-llama/Llama-3.1-8B-Instruct \
          --corpus /cache/corpus_capability_llama3 --p1 30000 --p2 3000
      RED de-risk: uv run modal run harness/finetune_fullstack.py --model meta-llama/Llama-3.2-1B-Instruct \
          --corpus /cache/corpus_capability_llama3_smoke --p1 40 --p2 40 --sel-limit 50   # proves the loop
          # NOTE: the capability corpus is Llama-3-tokenized, so the proxy MUST share that vocab
          # (Llama-3.2-1B-Instruct), NOT TinyLlama. The MODEL default below only fits the no-corpus path.
"""

import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
MODEL = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"  # dense base; MLP + attn dims %256

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64", "HF_HOME": "/cache",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .pip_install("torch", index_url="https://download.pytorch.org/whl/nightly/cu128", pre=True)
    .pip_install("transformers", "huggingface_hub", "safetensors", "sentencepiece", "pyarrow",
                 "bitsandbytes", "numpy", "datasets")  # datasets: in-loop downstream selection splits
    .add_local_dir((ROOT / "cuda").as_posix(), "/root/cuda")
)
app = modal.App("quadbit-fullstack", image=image)
vol = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)


@app.function(gpu="RTX-PRO-6000", timeout=86400, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(model: str = MODEL, p1: int = 30000, p2: int = 3000, both_shards: bool = False,
        lr_max: float = 2e-4, p2_lr: float = 1e-5, ce_alpha: float = 0.1, corpus: str = "",
        attn: bool = True, sel_limit: int = 200) -> float:
    # attn: also sparsify+train attention q/k/v/o (the full-stack lever). ce_alpha default 0.1: a
    # little hard-label CE alongside KD sharpens the argmax that MC accuracy reads. sel_limit: items
    # per task for the in-loop downstream selector (0 disables selection, keeps final-step weights).
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
    lib.sparse_fp4_mm_2lvl.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    lib.quantize_act_nvfp4.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2
    lib.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
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
        mant_f, e = torch.frexp(s.clamp_min(1e-30))
        mm = 2.0 * mant_f
        biased = (e - 1) + 7
        mant = torch.round((mm - 1.0) * 8.0).long()
        carry = mant == 8
        mant = torch.where(carry, torch.zeros_like(mant), mant)
        biased = torch.where(carry, biased + 1, biased)
        code = (biased.long() << 3) | mant
        code = torch.where(biased < 1, torch.ones_like(code), code)
        code = torch.where(biased > 15, torch.full_like(code, 0x7f), code)
        code = torch.where(s >= 480.0, torch.full_like(code, 0x7f), code)
        return torch.where(s > 0, code, torch.zeros_like(code))

    def act_fp4_dequant(x):  # per-32 TWO-LEVEL NVFP4 MATCHED to the sparse mma B-operand. Train == deploy.
        lead = x.shape[:-1]; i = x.shape[-1]
        b = x.to(torch.bfloat16).float().reshape(-1, i)
        gB = (b.abs().amax(-1, keepdim=True) / 2688.0).clamp_min(1e-30)
        bb = b.reshape(b.shape[0], i // 32, 32)
        sdeq = UE4M3[enc_ue4m3_t((bb.abs().amax(-1) / 6.0) / gB)] * gB
        return (FP4[q_fp4(bb / sdeq[..., None])] * sdeq[..., None]).reshape(*lead, i)

    class QATLinear(nn.Module):  # phase1: plain masked bf16 (fast); phase2 (qat=True): FP4 STE
        def __init__(self, weight):
            super().__init__()
            self.weight = nn.Parameter(weight.clone())
            self.qat = False

        def forward(self, x):
            if not self.qat:
                return F.linear(x, self.weight)
            Wf = self.weight.float()
            Wq = Wf + (sparse_fp4_dequant(Wf) - Wf).detach()   # STE weight fake-quant
            xf = x.float()
            xq = xf + (act_fp4_dequant(xf) - xf).detach()      # STE activation fake-quant
            return F.linear(xq, Wq).to(x.dtype)

    class QuadbitLinear(nn.Module):  # deployable through the real sparse-FP4 KERNEL; matches sparse_fp4_dequant
        def __init__(self, W):
            super().__init__()
            out_f, in_f = W.shape
            ks = in_f // 128
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
            self.register_buffer("Ac", Ac.contiguous())
            self.register_buffer("meta", meta)
            self.register_buffer("scaleA", scode.to(torch.uint8).permute(1, 0, 2).contiguous())
            self.register_buffer("gA", gA.reshape(out_f).float().contiguous())

        def forward(self, x):
            lead = x.shape[:-1]
            x2 = x.reshape(-1, self.in_f).to(torch.bfloat16)
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

    test_ids = wikitext("wikitext-2-raw-v1", "test-00000-of-00001.parquet")   # fluency reference (PPL)
    if corpus:  # capability corpus (or C4) pre-packed by build_capability_corpus / build_corpus
        import glob
        import numpy as np
        shards = sorted(glob.glob(f"{corpus}/shard_*.npy"))
        train_ids = torch.from_numpy(np.concatenate([np.load(s) for s in shards]).reshape(-1)).long()
        print(f"corpus {corpus}: {len(shards)} shards", flush=True)
    else:
        train_ids = wikitext("wikitext-103-raw-v1", "train-00000-of-00002.parquet")
        if both_shards:
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

    # --- in-loop downstream SELECTION: capped 0-shot loglikelihood multiple-choice accuracy ---
    # NOT the publication number (that stays lm-eval / downstream_eval.py on full sets); this only
    # picks the best-CAPABILITY checkpoint, since PPL and capability diverge (the documented trap).
    def _mc_items(limit):
        from datasets import load_dataset
        items = []  # (context, [continuation,...], gold_idx)
        try:
            for ex in load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test").select(range(limit)):
                t, l, k = ex["choices"]["text"], ex["choices"]["label"], ex["answerKey"]
                if k in l:
                    items.append((f"Question: {ex['question']}\nAnswer:", [" " + c for c in t], l.index(k)))
        except Exception as e:
            print(f"  [sel] arc unavailable: {e}", flush=True)
        try:
            for ex in load_dataset("Rowan/hellaswag", split="validation").select(range(limit)):
                items.append((ex["ctx"], [" " + e for e in ex["endings"]], int(ex["label"])))
        except Exception as e:
            print(f"  [sel] hellaswag unavailable: {e}", flush=True)
        try:
            for ex in load_dataset("allenai/winogrande", "winogrande_xl",
                                   split="validation").select(range(limit)):
                if ex["answer"] in ("1", "2") and "_" in ex["sentence"]:
                    a, b = ex["sentence"].split("_", 1)
                    items.append((a, [ex["option1"] + b, ex["option2"] + b], int(ex["answer"]) - 1))
        except Exception as e:
            print(f"  [sel] winogrande unavailable: {e}", flush=True)
        return items

    sel_items = _mc_items(sel_limit) if sel_limit else []
    print(f"downstream selection set: {len(sel_items)} items", flush=True)

    def downstream_acc(m):
        if not sel_items:
            return 0.0
        m.eval(); correct = 0
        with torch.no_grad():
            for ctx, conts, gold in sel_items:
                cids = tok(ctx, return_tensors="pt").input_ids.to(dev)
                best, arg = -1e30, 0
                for j, cont in enumerate(conts):
                    full = tok(ctx + cont, return_tensors="pt").input_ids.to(dev)
                    lg = m(full).logits[0, :-1].float().log_softmax(-1)
                    tgt = full[0, 1:]
                    lp = lg[torch.arange(tgt.shape[0], device=dev), tgt]
                    score = lp[cids.shape[1] - 1:].mean().item()  # mean logprob of the continuation
                    if score > best:
                        best, arg = score, j
                correct += int(arg == gold)
        return correct / len(sel_items)

    def target_lins(m):  # WIDENED: MLP gate/up/down + (if attn) self_attn q/k/v/o, dims %256 for the kernel
        specs = [("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj")]
        if attn:
            specs += [("self_attn", n) for n in ("q_proj", "k_proj", "v_proj", "o_proj")]
        for layer in m.model.layers:
            for pname, nm in specs:
                parent = getattr(layer, pname)
                lin = getattr(parent, nm)
                if lin.weight.shape[0] % 256 == 0 and lin.weight.shape[1] % 256 == 0:
                    yield parent, nm, lin

    teacher = load().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"PPL dense fp16 teacher: {ppl(teacher):.3f}  downstream(sel): {downstream_acc(teacher):.4f}", flush=True)

    student = load()
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.config.use_cache = False

    T, seq = 2.0, 1024
    P1, P2, wu = p1, p2, 500
    total = P1 + P2
    lr_min = lr_max / 20.0
    starts = list(range(0, len(train_ids) - seq, seq))
    targets = list(target_lins(student))
    tag = ("_" + corpus.rstrip("/").split("/")[-1]) if corpus else ("_2sh" if both_shards else "")
    tag += "_fs" + ("A" if attn else "M")  # full-stack marker: A=+attn, M=mlp-only, distinct from finetune_pair keys
    ck = f"/cache/phase1_{model.split('/')[-1]}_P{P1}{tag}_lr{lr_max:.0e}.pt"
    ck2 = f"/cache/phase2_{model.split('/')[-1]}_P{P1}_p2{P2}{tag}_lr{lr_max:.0e}.pt"   # durable phase-2 resume (survives Modal preemption)
    bcap = f"/cache/best_cap_{model.split('/')[-1]}_P{P1}_p2{P2}{tag}_lr{lr_max:.0e}.pt"

    CKPT_EVERY = 2500
    masks, qats, start_step = {}, [], 0
    best_acc, best_path, from_p2 = -1.0, None, False   # SELECT ON DOWNSTREAM, not PPL; best restored across preemption below

    def _mem(tag):  # GB alloc/reserved/free/peak — proves the phase-1 checkpoint is not duplicated on GPU
        free, _tot = torch.cuda.mem_get_info(dev)
        print(f"  [mem] {tag}: alloc {torch.cuda.memory_allocated(dev) / 2**30:.2f} GB  "
              f"reserved {torch.cuda.memory_reserved(dev) / 2**30:.2f} GB  "
              f"peak {torch.cuda.max_memory_allocated(dev) / 2**30:.2f} GB  free {free / 2**30:.2f} GB", flush=True)

    resume = ck2 if os.path.exists(ck2) else (ck if os.path.exists(ck) else None)
    if resume is not None:
        from_p2 = (resume == ck2)   # ck2 wins: a preemption mid/post phase-2 continues where it left off
        ckd = torch.load(resume, map_location="cpu", weights_only=True)  # CPU: never hold a duplicate copy on the GPU
        start_step = int(ckd["step"])
        _mem(f"after checkpoint load (cpu): {resume.split('/')[-1]}")
        for (parent, nm, _), W in zip(targets, ckd["weights"]):
            qat = QATLinear(W.to(torch.bfloat16)).to(dev)  # only the live module weight lands on GPU
            masks[qat] = (W != 0).to(dev); setattr(parent, nm, qat); qats.append(qat)
        _mem("after qats built")
        if from_p2:  # restore best-capability selection so preemption does not lose the best snapshot
            best_acc = float(ckd.get("best_acc", -1.0))
            if best_acc > -1.0 and os.path.exists(bcap):
                best_path = bcap
        del ckd
        import gc
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
        _mem("after ckd deleted")
        print(f"resumed {'phase-2' if from_p2 else 'phase-1'} checkpoint {resume} at step {start_step}"
              + (f" (best-cap sel {best_acc:.4f})" if from_p2 else ""), flush=True)
    else:
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
        for parent, nm, lin in targets:
            Wp = sparsegpt_pair24(lin.weight.data, Hs.pop(id(lin)))
            qat = QATLinear(Wp.to(torch.bfloat16)).to(dev)
            masks[qat] = (Wp != 0).to(dev); setattr(parent, nm, qat); qats.append(qat)
        print(f"PPL one-shot pair-2:4 FP4 (QAT-equiv): {ppl(student):.3f}  "
              f"downstream(sel): {downstream_acc(student):.4f}", flush=True)

    for p in student.parameters():
        p.requires_grad_(False)
    params = []
    for q in qats:
        q.weight.requires_grad_(True); params.append(q.weight)
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(params, lr=lr_max, betas=(0.9, 0.95), weight_decay=0.0)

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
        logits = student(w).logits
        sl = logits.reshape(-1, teacher.config.vocab_size)
        kl = F.kl_div(F.log_softmax(sl / T, -1), F.softmax(tl / T, -1), reduction="batchmean") * (T * T)
        loss = kl if ce_alpha == 0.0 else (1 - ce_alpha) * kl + ce_alpha * F.cross_entropy(logits[0, :-1], w[0, 1:])
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
        for step in range(start_step, P1):
            loss = kd_step(step)
            if (step + 1) % 1000 == 0:
                print(f"  P1 {step + 1}/{P1} KD {loss:.4f} PPL(bf16) {ppl(student, windows=8):.3f}", flush=True)
                student.train()
            if (step + 1) % CKPT_EVERY == 0:
                save_ck(step + 1)
        save_ck(P1)
        print(f"saved phase-1 checkpoint {ck}", flush=True)
    if not from_p2:  # skip the phase-1 baseline eval when resuming straight into phase-2
        print(f"PPL after phase-1 bf16 recovery: {ppl(student):.3f}  "
              f"downstream(sel): {downstream_acc(student):.4f}", flush=True)

    for q in qats:
        q.qat = True                                  # phase 2: QAT (weight+act FP4 STE)
    student.train()
    p2_start = max(P1, start_step)                      # resume phase-2 where preemption left off (durable ck2)
    _mem(f"before P2 step {p2_start - P1 + 1}")
    for step in range(p2_start, total):
        loss = kd_step(step)
        if (step + 1) % 500 == 0 or (step + 1) == total:
            acc = downstream_acc(student)
            print(f"  P2 {step + 1 - P1}/{P2} KD {loss:.4f} PPL(FP4) {ppl(student, windows=8):.3f} "
                  f"downstream(sel) {acc:.4f}", flush=True)
            _mem(f"P2 {step + 1 - P1}")
            if acc >= best_acc:  # snapshot best-capability weights to /cache as CPU tensors — no GPU-resident copy
                best_acc = acc
                torch.save([q.weight.detach().cpu() for q in qats], bcap); vol.commit()
                best_path = bcap
                print(f"  best-capability checkpoint -> {bcap} (sel {best_acc:.4f})", flush=True)
            torch.save({"step": step + 1, "weights": [q.weight.data.cpu() for q in qats],
                        "best_acc": best_acc}, ck2); vol.commit()  # durable: resume phase-2 here after preemption
            student.train()
    if best_path is not None:  # restore best-capability checkpoint before packing
        bw = torch.load(best_path, map_location="cpu", weights_only=True)
        for q, w in zip(qats, bw):
            q.weight.data.copy_(w.to(dev))
        del bw
        print(f"restored best-downstream(sel) checkpoint: {best_acc:.4f} from {best_path}", flush=True)
    print(f"PPL after phase-2 QAT recovery (FP4): {ppl(student):.3f}", flush=True)

    rck = f"/cache/recovered_{model.split('/')[-1]}_P{P1}_p2{P2}{tag}_lr{lr_max:.0e}.pt"
    torch.save({"weights": [q.weight.data.cpu() for q in qats], "sel_acc": best_acc}, rck); vol.commit()
    print(f"saved recovered phase-2 weights {rck}", flush=True)

    # final: build the actual sparse-FP4 KERNEL modules from ALL fine-tuned targets (MLP + attn)
    for parent, nm, _ in targets:
        q = getattr(parent, nm)
        setattr(parent, nm, QuadbitLinear(q.weight.data).to(dev))
    kernel_ppl = ppl(student)
    kernel_acc = downstream_acc(student)
    print(f"PPL through 2:4-sparse FP4 KERNEL (final): {kernel_ppl:.3f}  "
          f"downstream(sel) through kernel: {kernel_acc:.4f}", flush=True)
    return kernel_ppl


@app.local_entrypoint()
def main(model: str = MODEL, p1: int = 30000, p2: int = 3000, both_shards: bool = False,
         lr_max: float = 2e-4, p2_lr: float = 1e-5, ce_alpha: float = 0.1, corpus: str = "",
         attn: bool = True, sel_limit: int = 200) -> None:
    call = run.spawn(model=model, p1=p1, p2=p2, both_shards=both_shards, lr_max=lr_max,
                     p2_lr=p2_lr, ce_alpha=ce_alpha, corpus=corpus, attn=attn, sel_limit=sel_limit)
    print(f"SPAWN_ID {call.object_id}", flush=True)
    print(f"RESULT {call.get():.3f}", flush=True)
