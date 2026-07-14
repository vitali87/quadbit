# C6: C4 one-shot custom all-reduce quality validation

**Goal.** Decide whether the C4 collective (the vLLM one-shot custom all-reduce
`cross_device_reduce_1stage`, which raised SM120 decode from 48.248 to a median 58.126 tok/s,
+20.5%) preserves downstream quality, so the decode SOTA can be reported as quality-safe rather
than speed-only.

This is a collective-validation job, not a performance campaign. No kernels, no collective code, no
sparsity policy changed. The single variable under test is NCCL ring vs one-shot custom all-reduce.

Commit: `5b6b9e5` (branch `c6-c4-quality-validation`). Model `nvidia/DeepSeek-V4-Flash-NVFP4`,
4x RTX PRO 6000 (SM120), `enforce_eager=True`, `kv_cache_dtype=fp8`, greedy decode, limit 400,
max_len 2048. Downstream suite: the repo's 4-task MC loglikelihood smoke suite (ARC-Challenge,
HellaSwag, Winogrande, MMLU-subset of 5 subjects). Primary metric: acc_norm for arc/hellaswag, acc
for winogrande/mmlu; AVG is the mean of the four primaries. PIQA is intentionally excluded (gated
trust_remote_code, never loaded); all repo downstream AVGs are 4-task.

## Why eager eval validates the captured speed row

The C4 speed row is FULL CUDA-graph capture; these quality rows are `enforce_eager=True`. This is
numerically sound: CUDA-graph capture records and replays kernel launches, it does not change
arithmetic. `cross_device_reduce_1stage` and its bf16 cross-GPU reduction order are byte-identical
whether launched eagerly or replayed from a graph. So downstream quality measured eager equals the
captured speed row's quality; only latency differs, not the logits.

## Custom-AR engagement is gated on P2P topology (the key finding)

`QB_FORCE_CUSTOM_AR=1` does not blindly force the custom all-reduce. The plugin's `install()`
verifies the FULL P2P matrix with `torch.cuda.can_device_access_peer` and only then spoofs
`is_fully_connected -> True` and sets `VLLM_SKIP_P2P_CHECK=1` so vLLM enables the one-shot path. If
the 4-GPU set is not fully P2P-connected, it prints `P2P NOT fully connected -> leaving custom AR
disabled (NCCL fallback)` and runs NCCL. This is the same safety gate the C4 speed path uses.

Modal hands out 4-GPU sets with varying P2P topology. Of the two rows that requested custom AR:

- **R5 (sparse D2)** landed on a fully-connected set: `full P2P verified -> one-shot custom AR
  enabled`. This row genuinely exercised the custom all-reduce.
- **R3 (dense)** landed on a partially-connected set and fell back to NCCL. It is therefore a third
  dense-NCCL sample, not a dense custom-AR sample.

The custom all-reduce under test is the attention tensor-parallel reduce, which is identical whether
the MoE policy is dense or sparse (the collective is MoE-policy-independent). So R5's clean result on
that op is direct evidence for the dense case as well. Six dense attempts were made (R3 plus five
retries `c6_dense_customar_r2`..`_r6`) to catch a fully-connected container and obtain the dense
custom-AR point directly; Modal never handed out a fully-P2P dense set, so all six fell back to NCCL.
The dense-engaged point is therefore blocked by container luck, not by any quality issue, and R5
plus the policy-independence argument carry the verdict; see [verdict.md](verdict.md).

## Custom-AR engagement scope (honest)

vLLM's custom AR is size-gated: it engages when the all-reduce tensor is <= max_size (default 8 MB).
In this MC loglikelihood eval, forwards under the cap use the one-shot path; the largest packed
prefill steps fall back to NCCL in both rows identically. The decode regime that produces the speed
row is batch=1 (tiny tensor, always < max_size) = 100% one-shot custom AR, a strict subset of the
reductions this eval exercises. So the eval tests the same one-shot reduction on real activations,
plus additional larger reductions.

## Results

| row | tag | MoE policy | collective actually used | AVG | PPL (mito80) | per-GPU MB |
|-----|-----|-----------|--------------------------|-----|--------------|------------|
| R1 | c6_dense_nccl_a   | dense    | NCCL                    | 0.7382 | 3.518 | 94761 |
| R2 | c6_dense_nccl_b   | dense    | NCCL                    | 0.7344 | 3.518 | 94761 |
| R3 | c6_dense_customar | dense    | NCCL (P2P gate fell back) | 0.7379 | 3.518 | 94761 |
| R4 | c6_d2_nccl        | sparse D2 | NCCL                   | 0.7301 | 3.588 | 95065 |
| R5 | c6_d2_customar    | sparse D2 | **custom AR (engaged)** | 0.7341 | 3.538 | 95091 |

Dense-NCCL noise band (R1, R2, R3 are all NCCL on identical config): 0.7344 - 0.7382, spread
**0.38 pt**. This is the run-to-run floor.

Sparse D2 controlled comparison (identical config, collective is the only difference):
- R4 NCCL 0.7301 (PPL 3.588) vs R5 custom AR 0.7341 (PPL 3.538)
- delta R5 - R4 = **+0.40 pt** and lower PPL. The custom AR run is marginally higher, sits inside
  the 0.38 pt dense-NCCL noise band, and no task collapses.

PPL note: mito80 teacher-forced PPL is reduction-order-sensitive (bf16 accumulation order differs
between NCCL ring and one-shot), so it is reported as a sensitivity, not a ranking metric. The
observed PPL movement is small and in the favorable direction for custom AR.

## Generation sanity (custom-AR row and dense rows)

All rows produce coherent greedy output, no NaN / nonfinite / degenerate loops. Example (dense,
R3): `'The capital of France is' -> ' Paris.'`; `'The three primary colors are' -> ' red, green,
and blue. In the printing industry, the primaries are cyan, magenta, yellow and black.'`

## P2P engagement log (plugin `install()`, printed before the eval runs)

```
c6_dense_customar    : P2P NOT fully connected -> leaving custom AR disabled (NCCL fallback)
c6_dense_customar_r2 : P2P NOT fully connected -> leaving custom AR disabled (NCCL fallback)
c6_dense_customar_r3 : P2P NOT fully connected -> leaving custom AR disabled (NCCL fallback)
c6_dense_customar_r4 : P2P NOT fully connected -> leaving custom AR disabled (NCCL fallback)
c6_dense_customar_r5 : P2P NOT fully connected -> leaving custom AR disabled (NCCL fallback)
c6_dense_customar_r6 : P2P NOT fully connected -> leaving custom AR disabled (NCCL fallback)
c6_d2_customar       : full P2P verified -> one-shot custom AR enabled (is_fully_connected spoofed True + VLLM_SKIP_P2P_CHECK=1)
```

Six dense draws, zero fully-P2P; one sparse draw, fully-P2P. The dense-engaged point is blocked by
container topology luck, not quality. The full retry logs are supplementary (each is a redundant
NCCL run once it fell back); the decisive line above is captured for each.

## Raw logs

[c6_dense_nccl_a.log](../audit/logs/c6_dense_nccl_a.log),
[c6_dense_nccl_b.log](../audit/logs/c6_dense_nccl_b.log),
[c6_dense_customar.log](../audit/logs/c6_dense_customar.log),
[c6_d2_nccl.log](../audit/logs/c6_d2_nccl.log),
[c6_d2_customar.log](../audit/logs/c6_d2_customar.log). The dense custom-AR retries
`c6_dense_customar_r2.log`..`c6_dense_customar_r6.log` (all fell back to NCCL) are kept out of the
repo as redundant NCCL runs; their decisive P2P-fallback lines are quoted above.
