"""Gap #3 (b): quadbit dense W4A4 FULL-MODEL PREFILL throughput + memory + through-kernel PPL.

Swaps every transformer-block linear in a real Llama-3.1-8B-Instruct with the dense two-level
NVFP4 kernel (fused activation quantizer per linear), runs a batched PREFILL forward, and times it.
This measures the DROP-IN INTEGRATION overhead (fused quantizer + per-linear kernel call + layout
conversion) at full-model scale -- what the isolated GEMM sweep (cutlass_shapes) does NOT capture.

HARD LABEL (do not misuse): this is a full-model PREFILL-forward number, GEMM+activation-quantizer
bound. It is NOT a serving-engine number -- quadbit ships a kernel + nn.Linear drop-in, not paged
attention / continuous batching / a decode scheduler. It must NOT share a tok/s column with vLLM's
serving throughput (2724 tok/s). Expect it to trail vLLM's dense NVFP4 (which rides the CUTLASS-79b
class) on the same rectangular shapes cutlass_shapes showed (0.89-1.01x). Reported honestly:
full serving parity is future work through a real engine integration.

Run:  uv run modal run harness/dense_e2e.py
"""

import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
MODEL = "meta-llama/Llama-3.1-8B-Instruct"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64", "HF_HOME": "/cache"})
    .pip_install("torch", index_url="https://download.pytorch.org/whl/nightly/cu128", pre=True)
    .pip_install("transformers", "huggingface_hub", "safetensors", "pyarrow")
    .add_local_dir((ROOT / "cuda").as_posix(), "/root/cuda")
)
app = modal.App("quadbit-dense-e2e", image=image)
vol = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)


@app.function(gpu="RTX-PRO-6000", timeout=3600, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run() -> None:
    import ctypes
    import math
    import time

    import pyarrow.parquet as pq
    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    so = "/root/dense_nvfp4.so"
    c = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
                        "-o", so, "/root/cuda/dense_nvfp4_fast_lib.cu", "-lcuda"], capture_output=True, text=True)
    if c.returncode != 0:
        print(c.stderr, flush=True); return
    nv = ctypes.CDLL(so)
    nv.dense_nvfp4_mm.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    nv.quantize_act_nvfp4_2lvl.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    nv.quantize_act_nvfp4_2lvl.restype = None
    dev = torch.device("cuda")

    FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
    BND = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], device=dev)
    _c = torch.arange(128, device=dev); _e, _m = (_c >> 3) & 0xf, _c & 7
    UE4M3 = torch.where(_e == 0, _m.float() * 0.001953125, (1.0 + _m.float() / 8.0) * torch.exp2((_e - 7).float()))

    def quant_fp4(v):
        return torch.bucketize(v.abs(), BND) | ((v < 0).long() << 3)

    def enc_ue4m3(s):  # matches real_model / dense_nvfp4 kernel (no-denormal ue4m3)
        mf, e = torch.frexp(s.clamp_min(1e-30)); mant = torch.round((2.0 * mf - 1.0) * 8.0).long()
        biased = (e - 1) + 7; carry = mant == 8
        mant = torch.where(carry, torch.zeros_like(mant), mant); biased = torch.where(carry, biased + 1, biased)
        code = (biased << 3) | mant
        code = torch.where(biased < 1, torch.ones_like(code), code)
        return torch.where((biased > 15) | (s >= 480.0), torch.full_like(code, 0x7f),
                           torch.where(s > 0, code, torch.zeros_like(code)))

    def nvfp4_pack(W):  # two-level: Ab[out,in/2], SFA[step][out][8] ue4m3 local, gA[out] fp32 global
        out, inn = W.shape; Wf = W.float().to(dev)
        gA = (Wf.abs().amax(-1) / 2688.0).clamp(min=1e-30)
        Wb = Wf.view(out, inn // 16, 16)
        scode = enc_ue4m3((Wb.abs().amax(-1) / 6.0) / gA[:, None])
        sdeq = UE4M3[scode.long()] * gA[:, None]
        q = quant_fp4(Wb / sdeq[..., None]).view(out, inn)
        Ab = (q[:, 0::2] | (q[:, 1::2] << 4)).to(torch.uint8).contiguous()
        SFA = scode.view(out, inn // 128, 8).permute(1, 0, 2).contiguous().to(torch.uint8)
        return Ab, SFA, gA.contiguous()

    class DenseNVLinear(nn.Module):  # nn.Linear drop-in: fused act-quant + dense NVFP4 kernel
        def __init__(self, weight):
            super().__init__()
            out, inn = weight.shape
            Ab, SFA, gA = nvfp4_pack(weight)
            self.register_buffer("Ab", Ab); self.register_buffer("SFA", SFA); self.register_buffer("gA", gA)
            self.out, self.inn = out, inn
            self.wbytes = Ab.numel() + SFA.numel() + gA.numel() * 4

        def forward(self, x):
            lead = x.shape[:-1]
            x2 = x.reshape(-1, self.inn).to(torch.bfloat16)
            T = x2.shape[0]; pad = (-T) % 256
            if pad:
                x2 = torch.cat([x2, x2.new_zeros(pad, self.inn)], 0)
            x2 = x2.contiguous(); tp = T + pad
            Bb = torch.empty((tp, self.inn // 2), dtype=torch.uint8, device=dev)
            SFB = torch.empty((self.inn // 128, tp, 8), dtype=torch.uint8, device=dev)
            gB = torch.empty((tp,), dtype=torch.float32, device=dev)
            nv.quantize_act_nvfp4_2lvl(x2.data_ptr(), Bb.data_ptr(), SFB.data_ptr(), gB.data_ptr(), tp, self.inn)
            C = torch.empty((self.out, tp), dtype=torch.bfloat16, device=dev)
            nv.dense_nvfp4_mm(self.Ab.data_ptr(), Bb.data_ptr(), self.SFA.data_ptr(), SFB.data_ptr(),
                              C.data_ptr(), self.out, tp, self.inn, self.gA.data_ptr(), gB.data_ptr())
            return C.t()[:T].reshape(*lead, self.out).to(x.dtype)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    wbytes = 0
    for layer in model.model.layers:
        for parent, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                              (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
            for nm in names:
                lin = getattr(parent, nm)
                if lin.weight.shape[0] % 256 == 0 and lin.weight.shape[1] % 256 == 0:
                    dl = DenseNVLinear(lin.weight.data).to(dev); wbytes += dl.wbytes
                    setattr(parent, nm, dl)
    torch.cuda.empty_cache()
    print(f"{torch.cuda.get_device_name(0)}  |  quadbit dense W4A4 weights: {wbytes / 1e9:.2f} GiB", flush=True)

    # through-kernel PPL (correctness + deployed accuracy), same wikitext-2 windows as the vLLM ppl run
    ids = tok("\n\n".join(pq.read_table(hf_hub_download(
        "Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
        repo_type="dataset")).column("text").to_pylist())).input_ids
    seq, windows = 2048, 16
    nll = n = 0
    for i in range(0, min(len(ids), windows * seq) - seq, seq):
        w = torch.tensor(ids[i:i + seq], device=dev).unsqueeze(0)
        with torch.no_grad():
            nll += model(w, labels=w).loss.item() * (seq - 1); n += seq - 1
    ppl = math.exp(nll / n)

    # full-model PREFILL throughput: batched forward, GEMM+quantizer bound (NOT serving)
    B, S = 8, 2048
    inp = torch.randint(0, 32000, (B, S), device=dev)
    for _ in range(2):
        with torch.no_grad():
            model(inp)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
    with torch.no_grad():
        for _ in range(5):
            model(inp)
    torch.cuda.synchronize(); dt = (time.time() - t) / 5
    tps = B * S / dt
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"RESULT quadbit dense W4A4 [FULL-MODEL PREFILL, GEMM+quantizer bound, NOT a serving number]: "
          f"{tps:.0f} tok/s (prefill B={B} S={S}); through-kernel PPL {ppl:.3f}; weights {wbytes / 1e9:.2f} GiB; "
          f"peak {peak:.1f} GiB. vLLM's 2724 tok/s is full serving-stack (paged attn+decode) -> separate row, "
          f"not comparable; full serving parity = future engine integration.", flush=True)


@app.local_entrypoint()
def main() -> None:
    run.remote()
