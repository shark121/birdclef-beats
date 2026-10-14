# Reproducing the Phase 3b corpus

Written 20 September 2026, when the RTX A4000 box used for this work was shut
down. Everything below is either archived in Azure blob storage or rebuildable
from the code in this repo.

## Where the artifacts live

Storage account `birdclefstore69916`, container `corpus`, prefix
`phase3b-20260920/`. Note the doubled path segment — azcopy appended the source
directory name to the destination, and 5,109 blobs were not worth re-shuffling
to make it pretty:

| what | blob path | size |
|---|---|---|
| raw audio, 5,109 recordings | `phase3b-20260920/raw/raw/inaturalist/<species>/<file>` | 4.8 GB |
| clip manifests + problem logs | `phase3b-20260920/interim/interim/` | 11 MB |
| BEATs embeddings + clip order | `phase3b-20260920/embeddings/embeddings/` | 68 MB |
| per-variant analyses | `phase3b-20260920/exports/exports/` | 12 KB |

The same embeddings, manifests and analyses are also checked out locally under
`data/` and `exports/`. `data/` is gitignored, so the blob copy is the durable one.

**Not archived, because it is deterministic:** `data/clips/` (3.3 GB of 5 s WAVs)
regenerates from the raw audio in about 16 minutes per variant, and the BEATs
checkpoint re-downloads from HuggingFace on first use.

## Two manifests, kept side by side

- `data/interim/recordings.orig-20260908.csv` — the original 5,146-recording corpus.
- `data/interim/recordings.csv` — the 20 September rebuild, 5,109 recordings.

The rebuild re-queried the iNaturalist API from scratch and landed within 0.7% of
the original, with the thinnest species matching (`laugul` 174 vs 176). It
reproduced the published numbers to within 2% — see the validation table below.
The fetcher selects by species, not by recording id, so a future re-fetch will
drift again; the blob copy of the raw audio is the only exact record.

## Rebuild from nothing

```bash
python src/fetch_inaturalist.py --per-species 300     # ~75 min, network-bound
python src/build_manifest.py
python src/preprocess.py --tag band400 --band-floor 400 --workers <n_cores>
python src/embed.py --corpus band400 --tag band400 --batch-size 64   # needs a GPU
python src/analyze_clusters.py --tag band400 --k 10
```

Or restore the raw audio from blob instead of the ~75 minute re-fetch, and start
at `preprocess.py`.

`scripts/run_fetch.sh` and `scripts/run_variants.sh` are the exact scripts that
produced this run; they assume `/workspace/birdclef` and 4 cores.

## Flags added during this run

`preprocess.py`
- `--tag <name>` — clips to `data/clips/<name>/`, manifest to `clips_<name>.csv`.
  Omitting it reproduces the original behaviour byte for byte.
- `--band-floor <Hz>` — overrides the low edge of `BIRD_BAND_HZ`.

`embed.py`
- `--corpus <name>` — read `clips_<name>.csv` instead of `clips.csv`.
- `--device cuda|cpu` — **previously `BeatsEmbedder` was constructed with no
  device and always ran on CPU**, which is why the Azure runs needed `--threads 16`.

## Validation against the published numbers

| gate | this rebuild | original | |
|---|---|---|---|
| RMS floor only | 11,757 clips / 3,880 rec / 26.2% purity / 5.0x | 12,007 / 3,958 / 25.5% / 4.9x | within 2% |
| + 1.5 kHz band | 3,774 clips / 1,473 rec / 33.0% purity / 5.3x | 3,809 / 1,502 / 32.3% / 5.2x | within 1% |

## Result: use `band400`

| variant | purity | chance | lift | clips |
|---|---|---|---|---|
| `nofilter` | 26.2% | 5.2% | 5.0x | 11,757 |
| **`band400`** | 32.3% | 5.6% | **5.8x** | 5,491 |
| `band1500` | 33.0% | 6.2% | 5.3x | 3,774 |

`band1500` has marginally higher raw purity but a higher chance rate, because it
discards so much. On lift over chance — the comparable figure — `band400` wins,
with 45% more clips.

FINDINGS section 3 is confirmed and understated. Under the 1.5 kHz floor,
**White-winged Dove reaches 4.6% purity against a 6.2% chance rate — below
chance**, meaning the embedding is worse than useless for that species. At 400 Hz
it recovers to 42.8%, 4th best of twenty. Mourning Dove goes 10.3% to 33.0%.
A third species was also being quietly destroyed and is not mentioned in
FINDINGS: **Downy Woodpecker, 18.1% unfiltered to 9.4% at 1.5 kHz** — low
frequency drumming, same mechanism as the dove coo.

`band400` improves 18 of 20 species over no filter at all. It *hurts* Eastern
Bluebird (14.0% to 9.4%), the one species that prefers `band1500`. So the
"a single global band is the wrong tool" conclusion still stands — 400 Hz is a
better global band, not a correct one.

## Hardware note

RTX A4000, 16 GB, compute 8.6, measured on the 5 s / 248-token clip contract:

- frozen inference: 125 clips/s, the full corpus in ~97 s
- fine-tune step (bf16 + AdamW): 86 clips/s at batch 64, **12.8 GiB peak**

Batch 64 is the practical ceiling — batch 128 will not fit. If Phase 3b uses a
contrastive objective that wants large batches, budget for gradient checkpointing
or a memory bank rather than assuming the batch can simply grow.
