"""Workstream C downstream-quality table: is the +2.3 WT-2 PPL sparse tax broad quality loss or
WT-2-specific? Runs lm-eval-harness (0-shot) on HellaSwag, ARC-Challenge, PIQA, Winogrande (+ MMLU
behind --mmlu) for the MLP variants, holding non-MLP at bf16 so the delta isolates MLP sparsity:

  bf16     : dense bf16 Instruct MLP (reference ceiling).
  nvfp4    : dense-NVFP4 two-level fake-quant MLP (the production dense quality).
  sparse   : recovered all-MLP 2:4 sparse through the KERNEL (the banked serving row, PPL ~10.03).
  repaired : a repair-tournament ckpt through the KERNEL (+ optional per-channel affine correction).

Run:
  uv run modal run --detach harness/downstream_eval.py --variant nvfp4
  uv run modal run --detach harness/downstream_eval.py --variant sparse
  uv run modal run --detach harness/downstream_eval.py --variant repaired --ckpt /cache/repair_distill_5000.pt
"""

import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
RECOVERED_INSTRUCT_CKPT = "/cache/recovered_Llama-3.1-8B-Instruct_P30000_p25000_2sh_lr3e-05.pt"
TASKS = "hellaswag,arc_challenge,piqa,winogrande"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64", "HF_HOME": "/cache",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .pip_install("torch", index_url="https://download.pytorch.org/whl/nightly/cu128", pre=True)
    .pip_install("transformers", "huggingface_hub", "safetensors", "sentencepiece", "numpy",
                 "lm-eval==0.4.5", "datasets")
    .add_local_dir((ROOT / "cuda").as_posix(), "/root/cuda")
)
app = modal.App("quadbit-downstream", image=image)
vol = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)


@app.function(gpu="RTX-PRO-6000", timeout=86400, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(variant: str = "sparse", model: str = MODEL, ckpt: str = "", tasks: str = TASKS,
        mmlu: bool = False, calib: str = "") -> dict:
    import ctypes

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    so = "/root/sparse_fp4.so"
    c = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
                        "-o", so, "/root/cuda/sparse_fp4_lib.cu", "-lcuda"], capture_output=True, text=True)
    if c.returncode != 0:
        print(c.stderr, flush=True); return {}
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

    def dense_fp4_dequant(W):
        out_f, in_f = W.shape
        gA = (W.abs().amax(-1, keepdim=True) / 2688.0).clamp_min(1e-30)
        b = W.view(out_f, in_f // 16, 16)
        sdeq = UE4M3[enc_ue4m3_t((b.abs().amax(-1) / 6.0) / gA)] * gA
        return (FP4[q_fp4(b / sdeq[..., None])] * sdeq[..., None]).reshape(out_f, in_f)

    def act_fp4_dequant(x):
        lead = x.shape[:-1]; i = x.shape[-1]
        b = x.to(torch.bfloat16).float().reshape(-1, i)
        gB = (b.abs().amax(-1, keepdim=True) / 2688.0).clamp_min(1e-30)
        bb = b.reshape(b.shape[0], i // 32, 32)
        sdeq = UE4M3[enc_ue4m3_t((bb.abs().amax(-1) / 6.0) / gB)] * gB
        return (FP4[q_fp4(bb / sdeq[..., None])] * sdeq[..., None]).reshape(*lead, i)

    class DenseW4A4(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.register_buffer("w", weight.clone())

        def forward(self, x):
            Wq = dense_fp4_dequant(self.w.float())
            xf = x.float(); xq = xf + (act_fp4_dequant(xf) - xf).detach()
            return F.linear(xq, Wq).to(x.dtype)

    class QuadbitLinear(nn.Module):
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

    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).to(dev).eval()

    def mlp_lins():
        for layer in m.model.layers:
            for nm in ("gate_proj", "up_proj", "down_proj"):
                lin = getattr(layer.mlp, nm)
                if lin.weight.shape[0] % 256 == 0 and lin.weight.shape[1] % 256 == 0:
                    yield layer.mlp, nm, lin

    targets = list(mlp_lins())
    if variant == "bf16":
        pass
    elif variant == "nvfp4":
        for mlp, nm, lin in targets:
            setattr(mlp, nm, DenseW4A4(lin.weight.data).to(dev))
    elif variant in ("sparse", "repaired"):
        src = ckpt or RECOVERED_INSTRUCT_CKPT
        rec = torch.load(src, map_location="cpu", weights_only=True)["weights"]
        assert len(rec) == len(targets), f"weights {len(rec)} != targets {len(targets)}"
        for (mlp, nm, _lin), w in zip(targets, rec):
            setattr(mlp, nm, QuadbitLinear(w.to(dev)).to(dev))
        if calib:  # per-channel affine MLP-output correction from repair.py calib mode
            cd = torch.load(calib, map_location="cpu", weights_only=True)
            alpha = cd["alpha"].to(dev); beta = cd["beta"].to(dev)
            for li, layer in enumerate(m.model.layers):
                a, b = alpha[li], beta[li]
                layer.mlp.register_forward_hook(lambda _mo, _i, o, a=a, b=b: o * a.to(o.dtype) + b.to(o.dtype))
    else:
        print(f"unknown variant {variant}", flush=True); return {}

    m.config.use_cache = True
    task_list = tasks.split(",") + (["mmlu"] if mmlu else [])
    print(f"VARIANT={variant} ckpt={ckpt or '-'} calib={calib or '-'} tasks={task_list}", flush=True)
    lm = HFLM(pretrained=m, tokenizer=tok, batch_size=16)
    res = simple_evaluate(model=lm, tasks=task_list, num_fewshot=0, bootstrap_iters=0)
    out = {}
    for t, r in res["results"].items():
        acc = r.get("acc_norm,none", r.get("acc,none"))
        out[t] = round(float(acc), 4) if acc is not None else None
    print(f"RESULT variant={variant} ckpt={ckpt or '-'} " +
          " ".join(f"{k}={v}" for k, v in out.items()), flush=True)
    return out


@app.local_entrypoint()
def main(variant: str = "sparse", model: str = MODEL, ckpt: str = "", tasks: str = TASKS,
         mmlu: bool = False, calib: str = "") -> None:
    call = run.spawn(variant=variant, model=model, ckpt=ckpt, tasks=tasks, mmlu=mmlu, calib=calib)
    print(f"SPAWN_ID {call.object_id}", flush=True)
