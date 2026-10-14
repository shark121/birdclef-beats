"""Measure what a GPU actually does on this project's clip contract.

Frozen inference and a full fwd+bwd+AdamW step, with peak VRAM per batch size,
so a candidate box can be sized before committing to it.
"""
import sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.beats_model import BeatsEmbedder, fetch_code, fetch_checkpoint

N = int(config.CLIP_SECONDS * config.SAMPLE_RATE_HZ)
dev = "cuda"

print(f"GPU        : {torch.cuda.get_device_name(0)}")
print(f"clip       : {config.CLIP_SECONDS}s @ {config.SAMPLE_RATE_HZ}Hz -> {N} samples")

emb = BeatsEmbedder(device=dev)
nparam = sum(p.numel() for p in emb.model.parameters())
print(f"params     : {nparam/1e6:.1f}M   embed_dim={emb.embed_dim}")

# token count actually produced for a 5 s clip
with torch.inference_mode():
    w = torch.zeros(1, N, device=dev)
    f = emb.model.extract_features(w, padding_mask=torch.zeros(1, N, dtype=torch.bool, device=dev))[0]
print(f"tokens/clip: {tuple(f.shape)}")

def bench_infer(bs, iters=8):
    x = torch.randn(bs, N, device=dev) * 0.05
    m = torch.zeros(bs, N, dtype=torch.bool, device=dev)
    with torch.inference_mode():
        for _ in range(2): emb.model.extract_features(x, padding_mask=m)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(iters): emb.model.extract_features(x, padding_mask=m)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
    return bs * iters / dt

print("\n--- Phase 3a: frozen inference ---")
for bs in (16, 64, 128):
    try:
        torch.cuda.reset_peak_memory_stats()
        r = bench_infer(bs)
        print(f"  batch {bs:>3}: {r:7.1f} clips/s   peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GiB"
              f"   ({12007/r:.0f}s for 12,007 clips)")
    except RuntimeError as e:
        print(f"  batch {bs:>3}: FAILED {str(e)[:70]}")

print("\n--- Phase 3b: fine-tune step (fwd+bwd+AdamW, bf16 autocast) ---")
for bs in (16, 32, 64):
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        emb.model.train()
        opt = torch.optim.AdamW(emb.model.parameters(), lr=1e-5)
        x = torch.randn(bs, N, device=dev) * 0.05
        m = torch.zeros(bs, N, dtype=torch.bool, device=dev)
        def step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = emb.model.extract_features(x, padding_mask=m)[0]
                loss = out.float().mean()
            loss.backward(); opt.step()
        for _ in range(2): step()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(5): step()
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        r = bs * 5 / dt
        print(f"  batch {bs:>3}: {r:7.1f} clips/s   peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GiB"
              f"   ({12007/r/60:.1f} min/epoch @12k clips)")
        del opt
    except RuntimeError as e:
        print(f"  batch {bs:>3}: FAILED {str(e)[:70]}")
        torch.cuda.empty_cache()
