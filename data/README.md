# data/

Manifests, metrics and logs for **BirdCLEF+ 2026** (Pantanal, Brazil), embedded
with BEATs and then fine-tuned for classification. Large files (model, embeddings,
Projector exports) are in Azure blob storage; see [Large files](#large-files-azure).

This replaces the earlier 20-species iNaturalist data (git history before this
commit, and Azure `corpus/phase3b-20260920/`). The two are separate tasks: only
House Sparrow appears in both.

## Results

Held-out **test** split (`viz`, 20,452 clips from 4,947 recordings never seen in
training). Best epoch picked on `val`.

| model | clip accuracy | macro-F1 | recording accuracy |
|---|---|---|---|
| frozen BEATs + linear head | 52.8% | 48.8% | 59.6% |
| frozen BEATs + MLP head (baseline) | 55.4% | 51.9% | 62.8% |
| **fine-tuned BEATs** | **72.3%** | **68.4%** | **81.2%** |

- *clip accuracy*: top-1 per 5 s clip
- *macro-F1*: every class weighs the same, so the 98% bird majority cannot hide failures on rare classes
- *recording accuracy*: logits averaged over a recording's clips, then top-1

Fine-tuning: BEATs `iter3_plus_AS2M`, all layers trained, LR 5e-5 at the top
layer × 0.75 per layer down (head 1e-3), 10 epochs, batch 128, sqrt-balanced
sampling, random shift + gain + mixup, label smoothing 0.1, bf16 on an RTX 5090
(62 min). Best epoch: 8. Every epoch: [`metrics/finetune_result.json`](metrics/finetune_result.json).

## The dataset

| | |
|---|---|
| classes | 234 in the competition taxonomy (162 birds, 35 frogs, 28 insects, 8 mammals, 1 reptile); **206 have training audio** |
| recordings | 35,549: 23,043 Xeno-canto, 12,506 iNaturalist |
| clips | **137,342** × 5 s from 33,012 recordings (2,537 skipped, see log) |
| splits | train 96,551 / val 20,339 / viz (test) 20,452 clips, assigned **per recording** so no recording spans two splits |
| clip contract | 16 kHz mono, silence-trimmed, max 8 non-overlapping 5 s windows per recording, recordings of 1–5 s zero-padded, no frequency-band filter |

The 28 classes without training audio (25 insects, 3 frogs) appear only in the
competition's labelled soundscapes and cannot be predicted yet.

## Files

| file | what it is |
|---|---|
| `recordings.csv` | one row per recording (35,549): label, source, location, recordist, licence, source URL |
| `clips_bc26.csv` | one row per 5 s clip (137,342). **Also the row order of both embedding files**: row *i* describes vector *i* |
| `species_bc26.json` | the 234 classes: code, scientific and common name, `taxon_class`, iNat taxon id |
| `preprocess_problems_bc26.log` | the 2,537 recordings that produced no clip, and why (2,157 too quiet, 380 under 1 s) |
| `metrics/analysis_frozen.md` | neighbour purity of the frozen embeddings: per class, per taxon class, source bias |
| `metrics/analysis_finetuned.md` | the same for the fine-tuned embeddings (inflated: includes the training clips) |
| `metrics/probe_*.json` | frozen-embedding baselines (linear / MLP, with and without class weighting) |
| `metrics/finetune_result.json` | fine-tuning: settings, every epoch's val scores, final test scores |
| `logs/run_bc26.log`, `logs/run_ft.log` | full console output of the preprocessing/embedding run and the fine-tuning run |

### Columns

`species_code` mixes eBird codes (`rufnig1`) with numeric iNaturalist taxon ids
(`244024`) for non-birds. **Read it as text**: `pd.read_csv(..., dtype={"species_code": str})`.

| column | meaning |
|---|---|
| `clip_id` | `<recording_id>_seg<NN>`; `seg02` = third 5 s window |
| `recording_id` | unique recording key and the unit for splitting |
| `split` | `train` / `val` / `viz` (test) |
| `taxon_class` | Aves / Amphibia / Insecta / Mammalia / Reptilia |
| `dataset_source` | `birdclef-xc` (Xeno-canto) or `birdclef-inat` (iNaturalist) |
| `location` | `latitude,longitude` |
| `quality_rating` | Xeno-canto rating 0–5; 0 for iNaturalist |
| `segment_start_sec` | window start in the trimmed recording |
| `recordist`, `license`, `source_url` | attribution, see [`ATTRIBUTION.md`](ATTRIBUTION.md) |

## Large files (Azure)

Storage account `birdclefstore69916`, container `corpus`, prefix
**`bc26-20261003/`** (private; ask for access). Verify with `sha256sum -c SHA256SUMS`.

| blob | size | contents |
|---|---|---|
| `model/bc26_ft_best.pt` | 362 MB | fine-tuned model: `{"model": state_dict, "classes": [...], "args", "epoch", "val"}` |
| `embeddings/embeddings_bc26.npy` | 422 MB | frozen embeddings, (137342, 768) float32 |
| `embeddings/embeddings_bc26_ft.npy` | 422 MB | fine-tuned embeddings, (137342, 768) float32 |
| `exports/bc26_viz/`, `exports/bc26_ft_viz/` | 153 MB each | Embedding Projector files for the test split, frozen and fine-tuned |
| `code/birdclef2.0_src_20261003.tar.gz` | | the pipeline, preprocessing, baseline and fine-tuning code |
| `reports/` | | progress reports 1 and 2 (PDF) |

```bash
az storage blob download-batch --account-name birdclefstore69916 --auth-mode key \
  -s corpus --pattern "bc26-20261003/*" -d ./azure
```

```python
import numpy as np, pandas as pd
X = np.load("embeddings_bc26_ft.npy")                                  # (137342, 768)
meta = pd.read_csv("data/clips_bc26.csv", dtype={"species_code": str})
assert len(X) == len(meta)
```

## Known caveats

- **Source bias**: iNaturalist clips have ~2.5× more iNaturalist neighbours than
  chance, partly because insects and most frogs come only from iNaturalist.
  Fine-tuning did not remove it.
- **Imbalance**: birds are 98% of clips; 13 classes have 1–3 recordings and 29 have 10 or fewer.
- **Not included**: the audio (re-download from Kaggle, `birdclef-2026`) and the
  5 s clip WAVs (regenerated by the preprocessing step).
