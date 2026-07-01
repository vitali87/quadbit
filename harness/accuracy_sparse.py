"""Close the loop: measure the 2:4-sparse FP4 accuracy on a REAL 2:4-pretrained checkpoint
(a neuralmagic/RedHat Sparse-Llama *-2of4), not random or naively-pruned weights.

Key subtlety, reported honestly: Blackwell FP4 2:4 is PAIR-granular (mma.sp m16n8k128 selects
at b16 = fp4-pair granularity -> 2 of every 4 PAIRS kept), while existing fp16 2:4 checkpoints
are ELEMENT-granular (2 of every 4 elements). So we check (a) that the checkpoint really is
2:4-sparse, (b) whether its pattern is compatible with our pair-granular kernel (how much
nonzero energy our pair-2:4 selection retains), and (c) our FP4 reconstruction error on it.

Downloads only the shard(s) holding a couple of FFN layers (not the full 16GB).
Run:  uv run modal run harness/accuracy_sparse.py
"""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
CANDIDATES = [
    "neuralmagic/Sparse-Llama-3.1-8B-2of4",
    "nm-testing/SparseLlama-3-8B-pruned_50.2of4",
    "neuralmagic/Llama-2-7b-pruned2.4",
]

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/nightly/cu128", pre=True)
    .pip_install("huggingface_hub", "safetensors")
)
app = modal.App("quadbit-accuracy-sparse", image=image)


@app.function(gpu="RTX-PRO-6000", timeout=2400)
def run() -> None:
    import torch
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import GatedRepoError, EntryNotFoundError, RepositoryNotFoundError
    from safetensors.torch import load_file

    dev = torch.device("cuda")
    FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
                       dtype=torch.float32, device=dev)
    BND = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], dtype=torch.float32, device=dev)
    _c = torch.arange(128, device=dev)
    _e, _m = (_c >> 3) & 0xf, _c & 7
    UE4M3 = torch.where(_e == 0, _m.float() * 0.001953125,
                        (1.0 + _m.float() / 8.0) * torch.exp2((_e - 7).float()))
    _MID = (UE4M3[:-1] + UE4M3[1:]) / 2

    def q_fp4(v):
        return torch.bucketize(v.abs(), BND) | ((v < 0).long() << 3)

    def enc(s):
        return torch.bucketize(s, _MID)

    def sparse_fp4_dequant(W):  # our pair-granular 2:4-by-magnitude + NVFP4 (what the kernel packs)
        out_f, in_f = W.shape
        ks = in_f // 128
        Wg = W.view(out_f, ks, 16, 4, 2)
        i01, _ = Wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
        keptW = torch.gather(Wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
        blk = keptW.reshape(out_f, ks, 4, 8, 2)
        sdeq = UE4M3[enc(blk.abs().amax(dim=(3, 4)) / 6.0)]
        kd = (FP4[q_fp4(blk / sdeq[..., None, None])] * sdeq[..., None, None]).reshape(out_f, ks, 16, 2, 2)
        Wd = torch.zeros(out_f, ks, 16, 4, 2, device=dev)
        Wd.scatter_(3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2), kd)
        return Wd.reshape(out_f, in_f)

    def pair24_kept_energy(W):  # fraction of ||W||^2 our pair-2:4 selection keeps
        out_f, in_f = W.shape
        Wg = W.view(out_f, in_f // 128, 16, 4, 2)
        i01, _ = Wg.abs().sum(-1).topk(2, dim=-1).indices.sort(dim=-1)
        keptW = torch.gather(Wg, 3, i01.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
        return (keptW.pow(2).sum() / W.pow(2).sum()).item()

    def elem24_frac(W):  # fraction of 4-element groups that are exactly ELEMENT-2:4 (2 nonzero)
        g = (W.view(-1, 4) != 0).sum(-1)
        return (g == 2).float().mean().item(), (W == 0).float().mean().item()

    def rel(a, b):
        return (a - b).norm().item() / b.norm().item()

    model = None
    for m in CANDIDATES:
        try:
            idx_path = hf_hub_download(m, "model.safetensors.index.json")
            model = m
            print(f"using {m}", flush=True)
            break
        except (GatedRepoError, EntryNotFoundError, RepositoryNotFoundError) as e:
            print(f"skip {m}: {type(e).__name__}", flush=True)
    if model is None:
        print(">>> no ungated 2:4 checkpoint reachable; set HF_TOKEN or accept a license", flush=True)
        return

    wmap = json.load(open(idx_path))["weight_map"]
    # collect FFN weights for the first few layers, grouping by shard to bound the download
    want = [f"model.layers.{L}.mlp.{p}.weight" for L in range(4)
            for p in ("gate_proj", "up_proj", "down_proj")]
    want = [w for w in want if w in wmap]
    shards = {}
    for w in want:
        shards.setdefault(wmap[w], []).append(w)
    shards = dict(list(shards.items())[:1])  # just the first shard's worth (~5GB)

    print(f"\n{'tensor':<34}{'zeros':>8}{'elem2:4':>9}{'pairKeepE':>11}{'ourFP4err':>11}", flush=True)
    for shard, keys in shards.items():
        sd = load_file(hf_hub_download(model, shard))
        for w in keys:
            if w not in sd:
                continue
            W = sd[w].float().to(dev)
            if W.shape[1] % 256 or W.shape[0] % 256:
                print(f"{w:<34} skip (dims {tuple(W.shape)} not %256)", flush=True)
                continue
            e24, zf = elem24_frac(W)
            ke = pair24_kept_energy(W)
            err = rel(sparse_fp4_dequant(W), W)
            print(f"{w.replace('model.layers.', 'L'):<34}{zf:>8.3f}{e24:>9.3f}{ke:>11.3f}{err:>11.3f}",
                  flush=True)


@app.local_entrypoint()
def main() -> None:
    run.remote()
