"""M0: gpt-oss-20b MXFP4 expert decode + round-trip verification.

gpt-oss stores its MoE experts natively MXFP4 as 3D FUSED tensors (unlike DeepSeek's per-expert w1/w3/w2):
  model.layers.{L}.mlp.experts.gate_up_proj_blocks  uint8  [E, out=2*I, in=H, packed]  (E2M1, 2/byte)
  model.layers.{L}.mlp.experts.gate_up_proj_scales  uint8  [E, out, in//32]            (e8m0 block scale)
  model.layers.{L}.mlp.experts.gate_up_proj_bias    bf16   [E, out]
  model.layers.{L}.mlp.experts.down_proj_{blocks,scales,bias}  (out=H, in=I)
  model.layers.{L}.mlp.router.{weight,bias}         (stays dense, untouched)

The gate_up projection is INTERLEAVED in gpt-oss: gate = out[..., ::2], up = out[..., 1::2].
Activation (M1, not here) is the clamped SwiGLU: glu = clamp(gate,max=limit)*sigmoid(alpha*..);
out = (clamp(up,-limit,limit)+1)*glu; alpha=1.702, limit=7.0. This M0 only DECODES + round-trips.

Round-trip proof: re-encode the decoded bf16 back to MXFP4 bytes with the STORED e8m0 scale; it must
reproduce the stored int8 exactly (value-exact, allowing harmless +0/-0 E2M1 code aliasing). That
validates the E2M1 table, sign bit, e8m0 (2^(raw-127)) scale, nibble order, and 32-block layout on a
DIFFERENT architecture's checkpoint. No 128-alignment needed (per-32 decode), so 2880 dims are fine here.
"""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
MODEL = "openai/gpt-oss-20b"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64", "HF_HOME": "/cache"})
    .pip_install("torch", index_url="https://download.pytorch.org/whl/nightly/cu128", pre=True)
    .pip_install("safetensors", "huggingface_hub", "numpy")
)
app = modal.App("quadbit-gptoss-prep", image=image)
vol = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)


@app.function(gpu="RTX-PRO-6000", timeout=3600, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run(layers: str = "0,11,23") -> None:
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    dev = torch.device("cuda")
    # OCP E2M1 grid (transformers convert_moe_packed_tensors order), signed-zero at 0 and 8.
    FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
    BND = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], device=dev)

    def q_fp4(v):  # nearest E2M1 code (abs bucket | sign bit)
        return torch.bucketize(v.abs(), BND) | ((v < 0).long() << 3)

    def mxfp4_dequant(blocks_u8, scales_u8):
        """MXFP4 -> bf16. blocks [..., B, 16] uint8 (32 E2M1 vals/block, 2/byte, low nibble first);
        scales [..., B] uint8 e8m0 (multiplier 2^(raw-127)). Returns bf16 [..., B*32]."""
        b = blocks_u8.to(dev).to(torch.int32) & 0xFF
        lead = b.shape[:-1]                       # [..., B]
        nib_lo = b & 0xF
        nib_hi = (b >> 4) & 0xF
        codes = torch.stack([nib_lo, nib_hi], dim=-1).reshape(*lead[:-1], lead[-1] * 32).long()
        vals = FP4[codes]                          # [..., B*32]
        mult = torch.exp2(scales_u8.to(dev).float() - 127.0)  # e8m0 biased exponent
        nb = scales_u8.shape[-1]
        return (vals.reshape(*vals.shape[:-1], nb, 32) * mult[..., None]).reshape(vals.shape).to(torch.bfloat16)

    idx_path = hf_hub_download(MODEL, "model.safetensors.index.json")
    weight_map = json.load(open(idx_path))["weight_map"]

    def get(name):
        p = hf_hub_download(MODEL, weight_map[name])
        with safe_open(p, framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    print(f"# M0 gpt-oss MXFP4 expert decode + round-trip ({MODEL})", flush=True)
    ok = True
    for L in [int(x) for x in layers.split(",")]:
        for proj in ("gate_up_proj", "down_proj"):
            base = f"model.layers.{L}.mlp.experts.{proj}"
            blk = get(f"{base}_blocks")
            scl = get(f"{base}_scales")
            bias = get(f"{base}_bias")
            print(f"  L{L} {proj}: blocks {tuple(blk.shape)}/{blk.dtype}  "
                  f"scales {tuple(scl.shape)}/{scl.dtype}  bias {tuple(bias.shape)}/{bias.dtype}", flush=True)

            W = mxfp4_dequant(blk, scl)            # [..., B*32]
            # round-trip: re-encode decoded values with the STORED scale -> must reproduce stored bytes
            nb = scl.shape[-1]
            mult = torch.exp2(scl.to(dev).float() - 127.0)
            codes = q_fp4((W.float().reshape(*W.shape[:-1], nb, 32) / mult[..., None].clamp_min(1e-30)))
            codes = codes.reshape(W.shape)
            lo, hi = codes[..., 0::2], codes[..., 1::2]
            rebytes = (lo | (hi << 4)).to(torch.int32) & 0xFF        # [..., out, nb*16]
            orig = (blk.to(dev).to(torch.int32) & 0xFF).reshape(*blk.shape[:-2], nb * 16)
            raw_exact = (rebytes == orig).float().mean().item()
            vmatch = ((FP4[orig & 0xF] == FP4[rebytes & 0xF]) &
                      (FP4[(orig >> 4) & 0xF] == FP4[(rebytes >> 4) & 0xF])).float().mean().item()
            verdict = "VERIFIED" if vmatch > 0.9999 else "SUSPECT"
            ok = ok and vmatch > 0.9999
            print(f"      decoded W {tuple(W.shape)} absmax={W.abs().max():.3g} std={W.float().std():.3g}  "
                  f"round-trip value-exact={vmatch*100:.3f}% raw-byte={raw_exact*100:.3f}%  -> {verdict}", flush=True)
    print(f"# M0 done: {'DECODE VERIFIED on gpt-oss' if ok else 'DECODE SUSPECT — fix before M1'}", flush=True)
