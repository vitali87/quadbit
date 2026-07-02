"""Step 3: stage a DIVERSE recovery corpus (C4 English) for full-scale sparse recovery.

Why: the 88M WikiText-103 corpus is narrow and phase-1 plateaued on it; some of the plateau may
be diversity starvation, not token count. This stages ~500M-1B tokens of diverse web text (C4 en),
tokenized with the RECOVERY TARGET's own tokenizer, packed to the recovery seq length.

CRITICAL — eval leakage guard: the recovery corpus MUST be disjoint from the WikiText-2 *test* set
we measure PPL on. Any test text leaking into recovery data contaminates the 9.01→7.x curve and
voids the experiment. We decontaminate by dropping any C4 doc sharing a 13-gram with the test set,
and VERIFY explicitly with a positive control (the test text itself must be caught by the guard).

Run:  uv run modal run harness/build_corpus.py --mode smoke   # tiny slice, validates end-to-end
      uv run modal run harness/build_corpus.py --mode full    # ~500M tokens staged
"""

import modal

TOKENIZER = "meta-llama/Meta-Llama-3-8B"  # recovery target's tokenizer (matches finetune_pair)
CORPUS_DIR = "/cache/corpus_c4_llama3"
SEQ = 1024          # recovery sequence length (matches finetune_pair)
NGRAM = 13          # decontamination n-gram (GPT-3 / C4 standard)
MINUTES = 60

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("datasets", "transformers", "huggingface_hub", "pyarrow", "numpy", "tqdm")
    .env({"HF_HOME": "/cache", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
)
vol = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)
app = modal.App("quadbit-corpus", image=image)


def word_ngrams(text: str, n: int):
    w = text.split()
    for i in range(len(w) - n + 1):
        yield hash(tuple(w[i:i + n]))  # process-local hash; test set + C4 checked in ONE process


@app.function(cpu=8.0, timeout=6 * 60 * MINUTES, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def build(target_tokens: int, smoke: bool) -> None:
    import json
    import os

    import numpy as np
    import pyarrow.parquet as pq
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    eos = tok.eos_token_id

    # --- build the WikiText-2 TEST 13-gram set (the exact text PPL is measured on) ---
    test_text = "\n\n".join(pq.read_table(hf_hub_download(
        "Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
        repo_type="dataset")).column("text").to_pylist())
    test_ngrams = set(word_ngrams(test_text, NGRAM))
    print(f"test-set {NGRAM}-grams: {len(test_ngrams):,}", flush=True)

    # --- POSITIVE CONTROL: the guard MUST catch the test text itself ---
    ctrl_hit = len(set(word_ngrams(test_text[:20000], NGRAM)) & test_ngrams)
    assert ctrl_hit > 0, "leakage guard positive-control FAILED: test text not caught by its own n-grams"
    print(f"leakage guard positive-control: PASS (test text shares {ctrl_hit:,} {NGRAM}-grams -> would be dropped)",
          flush=True)

    # --- stream C4 en, decontaminate + dedup, tokenize, accumulate ---
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    buf: list[int] = []
    seen: set[int] = set()
    kept = drop_contam = drop_dup = ntok = 0
    for ex in ds:
        text = ex.get("text", "")
        if not text:
            continue
        h = hash(text)
        if h in seen:
            drop_dup += 1
            continue
        seen.add(h)
        if set(word_ngrams(text, NGRAM)) & test_ngrams:  # THE GUARD: shares a 13-gram with test -> drop
            drop_contam += 1
            continue
        ids = tok(text, add_special_tokens=False).input_ids
        buf.append(eos)
        buf.extend(ids)
        ntok += len(ids) + 1
        kept += 1
        if kept % 20000 == 0:
            print(f"  kept {kept:,} docs, {ntok:,} tok, dropped {drop_contam:,} contam / {drop_dup:,} dup",
                  flush=True)
        if ntok >= target_tokens:
            break

    # --- pack to SEQ and shard to the volume as int32 ---
    n_seq = len(buf) // SEQ
    arr = np.asarray(buf[:n_seq * SEQ], dtype=np.int32).reshape(n_seq, SEQ)
    out = CORPUS_DIR + ("_smoke" if smoke else "")
    os.makedirs(out, exist_ok=True)
    shard = 500_000  # sequences per shard
    for s in range(0, n_seq, shard):
        np.save(f"{out}/shard_{s // shard:04d}.npy", arr[s:s + shard])

    # --- VERIFY disjointness on the packed output (sample), plus report the guard's real work ---
    leak = 0
    for row in arr[:min(500, n_seq)]:
        if set(word_ngrams(tok.decode(row, skip_special_tokens=True), NGRAM)) & test_ngrams:
            leak += 1
    manifest = {
        "tokenizer": TOKENIZER, "seq": SEQ, "ngram": NGRAM, "tokens": int(arr.size),
        "sequences": int(n_seq), "docs_kept": kept, "dropped_contam": drop_contam,
        "dropped_dup": drop_dup, "leak_in_sample": leak, "positive_control_hits": ctrl_hit,
    }
    with open(f"{out}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    vol.commit()
    print(f"MANIFEST {json.dumps(manifest)}", flush=True)
    assert leak == 0, f"LEAKAGE: {leak} packed sequences share a {NGRAM}-gram with the test set"
    print(f"RESULT staged {arr.size:,} tokens ({n_seq:,} seqs) at {out}, ZERO test leakage", flush=True)


@app.local_entrypoint()
def main(mode: str = "smoke") -> None:
    if mode == "full":  # long job -> spawn + `modal run --detach` so it survives local disconnect
        call = build.spawn(target_tokens=500_000_000, smoke=False)
        print(f"SPAWN_ID {call.object_id}", flush=True)
        call.get()
    else:
        build.remote(target_tokens=2_000_000, smoke=True)  # ~2M tokens, validates the pipeline end-to-end
