"""Gap C, step 1: stage a CAPABILITY-relevant recovery corpus (not web text).

Why: every prior recovery corpus is web text (WikiText-103 / C4). The notes prove web text
"buys perplexity, not capability" — even the 500M-token C4 lever did not move downstream
(docs/paper_notes.md:623-630, docs/standing.md:48). This stages an in-distribution corpus built
from the TRAIN splits of the four downstream tasks we score (ARC, HellaSwag, Winogrande, MMLU),
rendered as completion text and distilled from the dense teacher. That is the capability signal
the earlier campaigns never had.

CRITICAL — eval leakage guard: the corpus is built from each task's TRAIN split; PPL/downstream
are scored on the TEST/VALIDATION split. Those are disjoint by construction, but we ALSO drop any
train doc sharing a 13-gram with the rendered eval split, and prove the guard with a positive
control (the eval text must be caught by its own n-grams). A leak here would fake a capability win.

Run:  uv run modal run harness/build_capability_corpus.py --mode smoke   # tiny, validates pipeline
      uv run modal run harness/build_capability_corpus.py --mode full    # full train splits staged
"""

import modal

TOKENIZER = "meta-llama/Meta-Llama-3-8B"   # recovery target's tokenizer (matches finetune_fullstack)
CORPUS_DIR = "/cache/corpus_capability_llama3"
SEQ = 1024          # recovery sequence length (matches finetune_pair/fullstack)
NGRAM = 13          # decontamination n-gram (GPT-3 / C4 standard)
MINUTES = 60

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("datasets", "transformers", "huggingface_hub", "pyarrow", "numpy", "tqdm")
    .env({"HF_HOME": "/cache", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
)
vol = modal.Volume.from_name("quadbit-hf-cache", create_if_missing=True)
app = modal.App("quadbit-capability-corpus", image=image)


def word_ngrams(text: str, n: int):
    w = text.split()
    for i in range(len(w) - n + 1):
        yield hash(tuple(w[i:i + n]))  # process-local hash; train + eval checked in ONE process


# --- per-task renderers: turn a raw item into the completion text the model sees ---
# each returns "" for a malformed/uncertain item (skipped), never raises on a single row.

def _arc(ex):
    ch = ex.get("choices") or {}
    texts, labels = ch.get("text") or [], ch.get("label") or []
    key = ex.get("answerKey")
    if not texts or key not in labels:
        return ""
    return f"Question: {ex.get('question', '').strip()}\nAnswer: {texts[labels.index(key)].strip()}"


def _hellaswag(ex):
    endings = ex.get("endings") or []
    lab = ex.get("label")
    try:
        i = int(lab)
    except (TypeError, ValueError):
        return ""
    if not (0 <= i < len(endings)):
        return ""
    return f"{ex.get('ctx', '').strip()} {endings[i].strip()}"


def _winogrande(ex):
    ans = ex.get("answer")
    opt = ex.get("option1") if ans == "1" else ex.get("option2") if ans == "2" else None
    s = ex.get("sentence", "")
    if opt is None or "_" not in s:
        return ""
    return s.replace("_", opt.strip())


def _mmlu(ex):
    ch = ex.get("choices") or []
    a = ex.get("answer")
    if not isinstance(a, int) or not (0 <= a < len(ch)):
        return ""
    body = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(ch))
    return f"{ex.get('question', '').strip()}\n{body}\nAnswer: {ch[a].strip()}"


# (loader_args, split_train, split_eval, renderer). Loaded lazily so one bad hub name
# does not sink the whole corpus.
SOURCES = [
    (("allenai/ai2_arc", "ARC-Challenge"), "train", "test", _arc),
    (("allenai/ai2_arc", "ARC-Easy"), "train", "test", _arc),
    (("Rowan/hellaswag",), "train", "validation", _hellaswag),
    (("allenai/winogrande", "winogrande_xl"), "train", "validation", _winogrande),
    (("cais/mmlu", "all"), "auxiliary_train", "test", _mmlu),
]


@app.function(cpu=8.0, timeout=6 * 60 * MINUTES, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def build(smoke: bool, per_source_cap: int) -> None:
    import json
    import os

    import numpy as np
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    eos = tok.eos_token_id

    def load(args, split):
        return load_dataset(*args, split=split, trust_remote_code=True)

    # --- build the eval 13-gram guard set from the rendered EVAL splits of every source ---
    eval_ngrams: set[int] = set()
    ctrl_text = ""
    for args, _tr, ev, render in SOURCES:
        try:
            ds = load(args, ev)
        except Exception as e:  # a single unavailable eval split must not disable the guard for the rest
            print(f"WARN eval split {args}/{ev} unavailable ({e}); its train source is SKIPPED below too", flush=True)
            continue
        for ex in ds:
            t = render(ex)
            if t:
                eval_ngrams |= set(word_ngrams(t, NGRAM))
                if not ctrl_text:
                    ctrl_text = t
    print(f"eval-set {NGRAM}-grams: {len(eval_ngrams):,}", flush=True)

    # --- POSITIVE CONTROL: the guard MUST catch a rendered eval item ---
    ctrl_hit = len(set(word_ngrams(ctrl_text, NGRAM)) & eval_ngrams)
    assert ctrl_hit > 0, "leakage guard positive-control FAILED: eval text not caught by its own n-grams"
    print(f"leakage guard positive-control: PASS (eval item shares {ctrl_hit:,} {NGRAM}-grams -> dropped)", flush=True)

    # --- render TRAIN splits, decontaminate + dedup, tokenize, accumulate ---
    buf: list[int] = []
    seen: set[int] = set()
    kept = drop_contam = drop_dup = drop_empty = ntok = 0
    for args, tr, ev, render in SOURCES:
        try:
            ds = load(args, tr)
        except Exception as e:
            print(f"WARN train split {args}/{tr} unavailable ({e}); skipped", flush=True)
            continue
        src_kept = 0
        for ex in ds:
            if smoke and src_kept >= 500:
                break
            if per_source_cap and src_kept >= per_source_cap:
                break
            text = render(ex)
            if not text:
                drop_empty += 1
                continue
            h = hash(text)
            if h in seen:
                drop_dup += 1
                continue
            seen.add(h)
            if set(word_ngrams(text, NGRAM)) & eval_ngrams:  # THE GUARD: shares a 13-gram with eval -> drop
                drop_contam += 1
                continue
            ids = tok(text, add_special_tokens=False).input_ids
            buf.append(eos)
            buf.extend(ids)
            ntok += len(ids) + 1
            kept += 1
            src_kept += 1
        print(f"  {args[0]}/{args[-1]}: kept {src_kept:,} (tot {kept:,} docs, {ntok:,} tok)", flush=True)

    assert kept > 0, "no training docs survived — every source failed to load or render"

    # --- pack to SEQ and shard to the volume as int32 (same layout finetune consumes) ---
    n_seq = len(buf) // SEQ
    assert n_seq > 0, f"only {len(buf)} tokens rendered (< SEQ={SEQ}); raise per_source_cap or check sources"
    arr = np.asarray(buf[:n_seq * SEQ], dtype=np.int32).reshape(n_seq, SEQ)
    out = CORPUS_DIR + ("_smoke" if smoke else "")
    os.makedirs(out, exist_ok=True)
    shard = 500_000  # sequences per shard
    for s in range(0, n_seq, shard):
        np.save(f"{out}/shard_{s // shard:04d}.npy", arr[s:s + shard])

    # --- VERIFY disjointness on the packed output (sample) ---
    leak = 0
    for row in arr[:min(500, n_seq)]:
        if set(word_ngrams(tok.decode(row, skip_special_tokens=True), NGRAM)) & eval_ngrams:
            leak += 1
    manifest = {
        "tokenizer": TOKENIZER, "seq": SEQ, "ngram": NGRAM, "tokens": int(arr.size),
        "sequences": int(n_seq), "docs_kept": kept, "dropped_contam": drop_contam,
        "dropped_dup": drop_dup, "dropped_empty": drop_empty, "leak_in_sample": leak,
        "positive_control_hits": ctrl_hit, "sources": [s[0][0] + "/" + s[0][-1] for s in SOURCES],
    }
    with open(f"{out}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    vol.commit()
    print(f"MANIFEST {json.dumps(manifest)}", flush=True)
    assert leak == 0, f"LEAKAGE: {leak} packed sequences share a {NGRAM}-gram with an eval set"
    print(f"RESULT staged {arr.size:,} tokens ({n_seq:,} seqs) at {out}, ZERO eval leakage", flush=True)


@app.local_entrypoint()
def main(mode: str = "smoke") -> None:
    if mode == "full":  # long job -> spawn + `modal run --detach` so it survives local disconnect
        call = build.spawn(smoke=False, per_source_cap=0)
        print(f"SPAWN_ID {call.object_id}", flush=True)
        call.get()
    else:
        build.remote(smoke=True, per_source_cap=0)  # ~500 docs/source, validates the pipeline end-to-end
