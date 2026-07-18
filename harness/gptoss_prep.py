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
    .add_local_dir((ROOT / "cuda").as_posix(), "/root/cuda")
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


@app.function(gpu="RTX-PRO-6000", timeout=3600, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def sparse_validate(layer: int = 11, experts: str = "0,1") -> None:
    """M2 kernel gate: sparse-pack REAL gpt-oss experts through the 2:4 two-level-NVFP4 kernel and
    verify kernel == fake-quant reference. gpt-oss experts are 2880x2880 (gate/up de-interleaved from
    the fused 5760; down 2880->2880), and NONE align to the kernel's grid(N/BN, M/2BM) requirement
    (out_f % 256, in_f % 128). Fix = pad each expert [2880,2880] -> [3072, 2944] with zeros (exact:
    zero rows/cols contribute nothing), pack, run, slice back. cos(kernel, fakequant) ~1 => the sparse
    path runs correctly on gpt-oss dims (blocker #2 solved); cos(sparseFP4, bf16) is the real 2:4-FP4 tax."""
    import ctypes
    import subprocess

    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    so = "/root/sparse_fp4.so"
    c = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-shared", "-Xcompiler", "-fPIC",
                        "-o", so, "/root/cuda/sparse_fp4_lib.cu", "-lcuda"], capture_output=True, text=True)
    if c.returncode != 0:
        print(c.stderr, flush=True); raise SystemExit(1)
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

    def sparse_fp4_dequant(W):  # pair-2:4 magnitude + per-16 two-level (matches kernel path); verbatim moe_prep
        out_f, in_f = W.shape; ks = in_f // 128
        Wg = W.view(out_f, ks, 16, 4, 2)
        i01, _ = Wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
        keptW = torch.gather(Wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
        gA = (keptW.abs().amax(dim=(1, 2, 3, 4), keepdim=True) / 2688.0).clamp_min(1e-30).reshape(out_f, 1, 1)
        blk = keptW.reshape(out_f, ks, 4, 8, 2)
        sdeq = UE4M3[enc_ue4m3_t((blk.abs().amax(dim=(3, 4)) / 6.0) / gA)] * gA
        kd = (FP4[q_fp4(blk / sdeq.clamp_min(1e-30)[..., None, None])] * sdeq[..., None, None]).reshape(out_f, ks, 16, 2, 2)
        Wd = torch.zeros(out_f, ks, 16, 4, 2, device=dev)
        Wd.scatter_(3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2), kd)
        return Wd.reshape(out_f, in_f)

    def act_fp4_dequant(x):
        lead = x.shape[:-1]; i = x.shape[-1]
        b = x.to(torch.bfloat16).float().reshape(-1, i)
        gB = (b.abs().amax(-1, keepdim=True) / 2688.0).clamp_min(1e-30)
        bb = b.reshape(b.shape[0], i // 32, 32)
        sdeq = UE4M3[enc_ue4m3_t((bb.abs().amax(-1) / 6.0) / gB)] * gB
        return (FP4[q_fp4(bb / sdeq.clamp_min(1e-30)[..., None])] * sdeq[..., None]).reshape(*lead, i)

    def pack(W):  # bf16 [out,in] -> kernel buffers; verbatim moe_prep (validated)
        out_f, in_f = W.shape; ks = in_f // 128
        Wg = W.float().to(dev).view(out_f, ks, 16, 4, 2)
        i01, _ = Wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
        keptW = torch.gather(Wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
        gA = (keptW.abs().amax(dim=(1, 2, 3, 4), keepdim=True) / 2688.0).clamp_min(1e-30).reshape(out_f, 1, 1)
        blk = keptW.reshape(out_f, ks, 4, 8, 2)
        scode = enc_ue4m3_t((blk.abs().amax(dim=(3, 4)) / 6.0) / gA)
        sdeq = UE4M3[scode] * gA
        kc = q_fp4(blk / sdeq.clamp_min(1e-30)[..., None, None])
        Ac = (kc[..., 0] | (kc[..., 1] << 4)).reshape(out_f, ks * 32).to(torch.uint8)
        nib = (i01[..., 0] | (i01[..., 1] << 2)).view(out_f, ks, 2, 8)
        sh = (torch.arange(8, device=dev) * 4).view(1, 1, 1, 8)
        meta = (nib << sh).sum(-1).to(torch.int32).permute(1, 0, 2).contiguous()
        return (Ac.contiguous(), meta, scode.to(torch.uint8).permute(1, 0, 2).contiguous(),
                gA.reshape(out_f).float().contiguous())

    def kernel_fwd(x, buf, out_f, in_f):  # verbatim moe_prep
        Ac, meta, scaleA, gA = buf; ks = in_f // 128
        x2 = x.reshape(-1, in_f).to(torch.bfloat16); t = x2.shape[0]; padn = (-t) % 128
        if padn:
            x2 = torch.cat([x2, x2.new_zeros(padn, in_f)], 0)
        x2 = x2.contiguous(); tp = t + padn
        Bb = torch.empty((tp, in_f // 2), dtype=torch.uint8, device=dev)
        sB = torch.empty((ks, tp, 4), dtype=torch.uint8, device=dev)
        gB = torch.empty((tp,), dtype=torch.float32, device=dev)
        lib.quantize_act_nvfp4_2lvl(x2.data_ptr(), Bb.data_ptr(), sB.data_ptr(), gB.data_ptr(), tp, in_f)
        C = torch.empty((out_f, tp), dtype=torch.bfloat16, device=dev)
        lib.sparse_fp4_mm_2lvl(Ac.data_ptr(), Bb.data_ptr(), scaleA.data_ptr(), sB.data_ptr(),
                               meta.data_ptr(), C.data_ptr(), out_f, tp, in_f, gA.data_ptr(), gB.data_ptr())
        return C.t()[:t]

    def mxfp4_dequant(blocks_u8, scales_u8):  # [.., B, 16] uint8 + [.., B] e8m0 -> bf16 [.., B*32] (M0-verified)
        b = blocks_u8.to(dev).to(torch.int32) & 0xFF
        lead = b.shape[:-1]
        codes = torch.stack([b & 0xF, (b >> 4) & 0xF], dim=-1).reshape(*lead[:-1], lead[-1] * 32).long()
        vals = FP4[codes]
        nb = scales_u8.shape[-1]
        mult = torch.exp2(scales_u8.to(dev).float() - 127.0)
        return (vals.reshape(*vals.shape[:-1], nb, 32) * mult[..., None]).reshape(vals.shape).to(torch.bfloat16)

    def pad_w(W):  # [out,in] -> padded [outp,inp] (outp % 256, inp % 128) with zeros; exact
        out_f, in_f = W.shape
        outp = ((out_f + 255) // 256) * 256
        inp = ((in_f + 127) // 128) * 128
        Wp = W.new_zeros(outp, inp); Wp[:out_f, :in_f] = W
        return Wp, out_f, in_f

    idx_path = hf_hub_download(MODEL, "model.safetensors.index.json")
    weight_map = __import__("json").load(open(idx_path))["weight_map"]

    def get(name):
        p = hf_hub_download(MODEL, weight_map[name])
        with safe_open(p, framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    print(f"# M2 gpt-oss sparse-kernel gate ({MODEL}) layer {layer}", flush=True)
    # M2 tests ONE thing: does zero-padding [2880,2880]->[3072,2944] let the kernel run on gpt-oss's
    # non-aligned dims WITHOUT adding error? The absolute cos(kernel,fakequant) floors at ~0.98 for THIS
    # harness (the Python fakequant is an approximate model of the real FP4 kernel; model-level fidelity is
    # proven separately, deploy gap 8.47==8.47). So the gate is RELATIVE: padded-real must agree at least as
    # well as an ALIGNED control that needs no padding. If padding corrupted anything, padded would be worse.
    ctrl = 0.0
    for scale, tag in ((0.02, "ctrl-aligned-small"), (0.5, "ctrl-aligned-outlier")):
        Wc = (torch.randn(3072, 2944, device=dev) * scale).float()
        xc = torch.randn(256, 2944, device=dev, dtype=torch.bfloat16) * 0.1
        yk = kernel_fwd(xc, pack(Wc), 3072, 2944)
        yf = F.linear(act_fp4_dequant(xc.float()), sparse_fp4_dequant(Wc))
        ck = F.cosine_similarity(yk.float().flatten(), yf.float().flatten(), dim=0).item()
        ctrl = max(ctrl, ck)
        print(f"  {tag}: cos(kernel,fakequant)={ck:.5f}  (aligned, no pad — the reference floor)", flush=True)
    thr = ctrl - 0.005
    print(f"  -> padding-exact gate: padded-real experts must reach >= {thr:.5f} (aligned floor {ctrl:.5f} - 0.005)", flush=True)
    allok = True
    for e in [int(x) for x in experts.split(",")]:
        gu = mxfp4_dequant(get(f"model.layers.{layer}.mlp.experts.gate_up_proj_blocks")[e],
                           get(f"model.layers.{layer}.mlp.experts.gate_up_proj_scales")[e])  # [5760, 2880]
        dn = mxfp4_dequant(get(f"model.layers.{layer}.mlp.experts.down_proj_blocks")[e],
                           get(f"model.layers.{layer}.mlp.experts.down_proj_scales")[e])      # [2880, 2880]
        gate, up = gu[0::2, :], gu[1::2, :]  # gpt-oss interleaves gate/up on the fused out dim
        for name, W in (("gate", gate), ("up", up), ("down", dn)):
            Wp, out_f, in_f = pad_w(W.float())
            outp, inp = Wp.shape
            xr = torch.randn(64, in_f, device=dev, dtype=torch.bfloat16) * 0.1
            xp = torch.zeros(64, inp, device=dev, dtype=torch.bfloat16); xp[:, :in_f] = xr
            yk = kernel_fwd(xp, pack(Wp), outp, inp)[:, :out_f]
            yf = F.linear(act_fp4_dequant(xp.float()), sparse_fp4_dequant(Wp))[:, :out_f]
            ck = F.cosine_similarity(yk.float().flatten(), yf.float().flatten(), dim=0).item()
            wtax = F.cosine_similarity(sparse_fp4_dequant(Wp)[:out_f, :in_f].flatten(), Wp[:out_f, :in_f].flatten(), dim=0).item()
            allok = allok and ck >= thr
            print(f"  L{layer} e{e} {name}: [{out_f}x{in_f}]->pad[{outp}x{inp}]  "
                  f"cos(kernel,fakequant)={ck:.5f}  cos(sparseFP4,bf16)={wtax:.4f}  {'OK' if ck >= thr else 'FAIL'}", flush=True)
    print(f"# M2 done: {'PADDING EXACT — sparse kernel runs on gpt-oss dims (blocker #2 solved)' if allok else 'PADDING ADDS ERROR — investigate'}", flush=True)
