"""A/B SEMANTICS on the recovered pair-2:4 checkpoint: matched fake-quant vs the OLD single-level
sparse kernel vs the NEW two-level sparse kernel, all on the SAME fine-tuned weights.

For each mode: PPL + mean NLL on WikiText-2 test (windows=16, held-out), NLL delta vs fake-quant,
and a fixed-batch logit comparison vs fake-quant (RMSE, max abs, cosine, top-1 agreement). This is
the semantic proof that the two-level kernel's outputs == what QAT trained against, while the
single-level kernel diverges -- the deploy gap, made concrete at the logit level.

Reads the recovered weights saved by finetune_pair (recovered_*.pt on the volume). The weight/act
fake-quant and kernel packing are lifted verbatim from finetune_pair / verify_sparse_2lvl.

Run:  uv run modal run harness/ab_sparse_semantics.py
"""

import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
MODEL = "meta-llama/Meta-Llama-3-8B"
RCK = "/cache/recovered_Meta-Llama-3-8B_P30000_p25000_2sh_lr3e-05.pt"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64", "HF_HOME": "/cache",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .pip_install("torch", index_url="https://download.pytorch.org/whl/nightly/cu128", pre=True)
    .pip_install("transformers", "huggingface_hub", "safetensors", "sentencepiece", "pyarrow", "numpy")
    .add_local_dir((ROOT / "cuda").as_posix(), "/root/cuda")
)
app = modal.App("quadbit-ab-semantics", image=image)
vol = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)


@app.function(gpu="RTX-PRO-6000", timeout=3600, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(model: str = MODEL, rck: str = RCK) -> None:
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

    if not os.path.exists(rck):
        print(f"MISSING recovered checkpoint {rck} -- run finetune_pair first", flush=True); return

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

    def enc_ue4m3_t(s):
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

    def pair24(W):
        out_f, in_f = W.shape; ks = in_f // 128
        Wg = W.view(out_f, ks, 16, 4, 2)
        i01, _ = Wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
        keptW = torch.gather(Wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
        return Wg, i01, keptW, ks

    def sparse_fp4_dequant_2lvl(W):  # two-level weight fake-quant (== finetune)
        out_f, in_f = W.shape
        Wg, i01, keptW, ks = pair24(W)
        gA = (keptW.abs().amax(dim=(1, 2, 3, 4), keepdim=True) / 2688.0).clamp_min(1e-30).reshape(out_f, 1, 1)
        blk = keptW.reshape(out_f, ks, 4, 8, 2)
        sdeq = UE4M3[enc_ue4m3_t((blk.abs().amax(dim=(3, 4)) / 6.0) / gA)] * gA
        kd = (FP4[q_fp4(blk / sdeq[..., None, None])] * sdeq[..., None, None]).reshape(out_f, ks, 16, 2, 2)
        Wd = torch.zeros(out_f, ks, 16, 4, 2, device=dev)
        Wd.scatter_(3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2), kd)
        return Wd.reshape(out_f, in_f)

    def act_2lvl_deq_p32(x):  # two-level per-32 activation fake-quant (matches sparse mma B-side)
        lead = x.shape[:-1]; i = x.shape[-1]
        b = x.to(torch.bfloat16).float().reshape(-1, i)
        gB = (b.abs().amax(-1, keepdim=True) / 2688.0).clamp_min(1e-30)
        bb = b.reshape(b.shape[0], i // 32, 32)
        sdeq = UE4M3[enc_ue4m3_t((bb.abs().amax(-1) / 6.0) / gB)] * gB
        return (FP4[q_fp4(bb / sdeq[..., None])] * sdeq[..., None]).reshape(*lead, i)

    class FakeQuantLinear(nn.Module):  # matched two-level weight+act STE (== what QAT trained against)
        def __init__(self, W):
            super().__init__()
            self.register_buffer("Wq", sparse_fp4_dequant_2lvl(W.float()))

        def forward(self, x):
            return F.linear(act_2lvl_deq_p32(x.float()), self.Wq).to(x.dtype)

    class Kernel2lvl(nn.Module):  # new two-level kernel (verbatim from finetune_pair.QuadbitLinear)
        def __init__(self, W):
            super().__init__()
            out_f, in_f = W.shape; ks = in_f // 128
            self.out_f, self.in_f, self.ks = out_f, in_f, ks
            Wg, i01, keptW, _ = pair24(W.float().to(dev))
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

    class Kernel1lvl(nn.Module):  # OLD single-level kernel (absolute per-block scale, no global)
        def __init__(self, W):
            super().__init__()
            out_f, in_f = W.shape; ks = in_f // 128
            self.out_f, self.in_f, self.ks = out_f, in_f, ks
            Wg, i01, keptW, _ = pair24(W.float().to(dev))
            blk = keptW.reshape(out_f, ks, 4, 8, 2)
            scode = torch.bucketize(blk.abs().amax(dim=(3, 4)) / 6.0, _MID)  # single-level absolute
            kc = q_fp4(blk / UE4M3[scode][..., None, None])
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
                              sB.data_ptr(), self.meta.data_ptr(), C.data_ptr(), self.out_f, tp, self.in_f)
            return C.t()[:t].reshape(*lead, self.out_f).to(x.dtype)

    tok = AutoTokenizer.from_pretrained(model)

    def wikitext(cfg, fn):
        p = hf_hub_download("Salesforce/wikitext", f"{cfg}/{fn}", repo_type="dataset")
        txt = "\n\n".join(pq.read_table(p).column("text").to_pylist())[:200_000_000]
        return tok(txt, return_tensors="pt").input_ids[0]

    test_ids = wikitext("wikitext-2-raw-v1", "test-00000-of-00001.parquet")

    def mlp_lins(m):
        for layer in m.model.layers:
            for nm in ("gate_proj", "up_proj", "down_proj"):
                lin = getattr(layer.mlp, nm)
                if lin.weight.shape[0] % 256 == 0 and lin.weight.shape[1] % 256 == 0:
                    yield layer.mlp, nm, lin

    ckd = torch.load(rck, map_location="cpu", weights_only=True)
    rec_ws = ckd["weights"]  # recovered bf16 sparse weights, in mlp_lins order
    print(f"loaded {len(rec_ws)} recovered MLP weights from {rck}", flush=True)

    def build(mode):  # fresh model, swap each MLP linear with the chosen sparse module
        m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).to(dev)
        m.config.use_cache = False
        cls = {"fake": FakeQuantLinear, "1lvl": Kernel1lvl, "2lvl": Kernel2lvl}[mode]
        for (mlp, nm, _), W in zip(list(mlp_lins(m)), rec_ws):
            setattr(mlp, nm, cls(W.to(dev)).to(dev))
        return m.eval()

    seq, windows = 2048, 16

    def eval_ppl_and_logits(m):  # mean NLL + PPL, plus logits on the first fixed window
        nll, n = 0.0, 0
        first_logits = None
        for i in range(0, min(len(test_ids), windows * seq) - seq, seq):
            w = test_ids[i:i + seq].unsqueeze(0).to(dev)
            with torch.no_grad():
                out = m(w, labels=w)
            nll += out.loss.item() * (seq - 1); n += seq - 1
            if first_logits is None:
                first_logits = out.logits[0].float().cpu()  # (seq, vocab)
        return nll / n, math.exp(nll / n), first_logits

    rows = {}
    for mode in ("fake", "1lvl", "2lvl"):
        m = build(mode)
        mean_nll, ppl, logits = eval_ppl_and_logits(m)
        rows[mode] = (mean_nll, ppl, logits)
        print(f"[{mode}] PPL {ppl:.4f}  meanNLL {mean_nll:.5f}", flush=True)
        del m; torch.cuda.empty_cache()

    ref_nll, ref_ppl, ref_logits = rows["fake"]
    print("\nmode   PPL       meanNLL    dNLL_vs_fake   logit-RMSE   logit-maxabs   cosine     top1-agree", flush=True)
    print("-" * 100, flush=True)
    for mode in ("fake", "1lvl", "2lvl"):
        mean_nll, ppl, logits = rows[mode]
        d = logits - ref_logits
        rmse = d.pow(2).mean().sqrt().item()
        maxabs = d.abs().max().item()
        a, b = logits.reshape(-1), ref_logits.reshape(-1)
        cos = (a @ b / (a.norm() * b.norm())).item()
        top1 = (logits.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
        print(f"{mode:<5}  {ppl:8.4f}  {mean_nll:8.5f}  {mean_nll - ref_nll:+12.5f}   {rmse:10.5f}   "
              f"{maxabs:11.5f}   {cos:8.5f}   {top1:9.4f}", flush=True)
    print("\nfake == 2lvl (kernel reproduces QAT target); 1lvl diverges = the single-level deploy gap.", flush=True)


@app.local_entrypoint()
def main(model: str = MODEL, rck: str = RCK) -> None:
    call = run.spawn(model=model, rck=rck)
    print(f"SPAWN_ID {call.object_id}", flush=True)
    call.get()
