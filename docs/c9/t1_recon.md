# C9 T1 recon: MTP speculative decoding is available on this stack

Branch `feat/mtp-spec-decode` off `main`. Goal of C9: attack the SM120 decode floor from the one
angle C4-C7 did not touch. C4-C7 attacked collective **cost per token** and hit the PCIe
sync-latency wall (~374us/layer, no NVLink, sequential, non-overlappable at batch=1). Speculative
decoding attacks collective **frequency per token**: verify k drafted tokens in one target forward,
so the per-layer collective floor is paid once per ~k accepted tokens instead of once per token.
Motivated by DSpark (arXiv 2607.05147) + the sentdex measurement of ~+50% from DSpark on the same
2x RTX PRO 6000 box.

## T1 question: does the stack expose DeepSeek-V4-Flash MTP spec-decode, and does the NVFP4 checkpoint ship the head?

**Both YES.**

### 1. vLLM supports MTP spec-decode, auto-detected, no separate checkpoint
- `speculative_config={"method": "mtp", "num_speculative_tokens": N}`. vLLM detects the MTP head
  from the model config and activates it; `num_speculative_tokens` can auto-fill from the head's
  n_predict. (vLLM speculative-decoding docs; DeepSeek-V4 recipes list MTP.)
- Known caveat (not blocking a batch=1 decode-latency benchmark): MTP + prefix caching truncates the
  reported prefix-cache hit length (32K->16K, vllm-ascend #9247). Irrelevant at short-context batch=1.

### 2. The NVFP4 checkpoint (`nvidia/DeepSeek-V4-Flash-NVFP4`) ships the MTP head
- `config.json`: `num_nextn_predict_layers: 1`, `architectures: [DeepseekV4ForCausalLM]`,
  `model_type: deepseek_v4`. `mtp.*` is in the quantization **ignore** list (kept unquantized, not dropped).
- `model.safetensors.index.json`: **1575 `mtp.0.*` tensors** = a full decoder block used as the draft head:
  - MLA attention: `mtp.0.attn.{wkv,wo_a,wo_b,wq_a,wq_b}.{weight,scale}`, `kv_norm`, `q_norm`,
    `attn_sink`, `attn_norm`.
  - MoE ffn: `mtp.0.ffn.experts.{0..255}.w{1,2,3}.{weight,scale}` (same 256-expert layout as main layers).
  - `mtp.0.e_proj.{weight,scale}`, `mtp.0.enorm.weight`.
  - No `layers.>=43` (the MTP block lives under `mtp.0.`, not as layer 43).

## Consequence for integration (feeds T2/T3)

- The MTP draft block reuses the **same** MLA/DSA attention module + Fp8/NVFP4 quant classes as the main
  model. The plugin's SM120 unblock patches are **class-level** (Fp8LinearMethod, DSA indexer,
  attention), so they should cover the MTP block automatically — to be verified on the first load (T2).
- The MTP head is **not cheap**: it is a full MoE block (attn + 256-expert ffn), not a lightweight
  linear. Draft-forward cost is a real term in the `L = (T_draft + T_verify)/tau` budget; the amortization
  win must clear it. This is exactly the T3(b) risk on a no-NVLink box (cf. C7's DP-attention negative:
  a structurally-correct floor attack that lost on wall-clock).

## Verdict

**T1 PASS.** No blocker to trying MTP spec-decode. Proceed to T2: enable
`speculative_config` on the **dense** NVFP4 serve path (dense is the C4 decode SOTA at 58.126 tok/s;
spec-decode is orthogonal to sparsity, so measure the multiplier on the SOTA first), captured, and
compare decode tok/s vs 58.126. Gate any further Modal spend on T2.

Open risks carried to T2/T3:
1. Does vLLM instantiate the MTP layer so the plugin's class-level SM120 patches apply, or via a path
   that bypasses them (would fail to load, like the base model did pre-unblock)?
2. Is the spec-decode accept/reject loop graph-capturable under the plugin, or does it add a host-sync?
3. Does the amortization beat the full-MoE-block draft cost on no-NVLink PCIe (the C7 failure mode)?
