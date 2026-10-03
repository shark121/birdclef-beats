# data/

Manifests, clip lists and frozen-BEATs embeddings from the bird-song embedding
pipeline. **No audio is included**: the raw recordings (4.8 GB) and the 5 s clip
WAVs (3.3 GB) are too large for git and are archived in Azure blob storage
(`birdclefstore69916`, container `corpus`, prefix `phase3b-20260920/`). Every row
below points back to its source recording via `source_url` / `source_recording_id`,
so the audio can be re-fetched from iNaturalist.

## At a glance

| | |
|---|---|
| source | iNaturalist research-grade audio only (Xeno-canto / BirdCLEF / field not yet added) |
| species | 20 Texas Gulf Coast birds, eBird 6-letter codes (`normoc` = Northern Mockingbird) |
| recordings | 5,109 (20 Sep rebuild) · 5,146 (original 8 Sep run) |
| clip | 16 kHz mono, 5.0 s, non-overlapping, silence-trimmed, peak-normalised, max 8 per recording |
| splits | 70 / 15 / 15 `train` / `val` / `viz`, assigned **per recording** (SHA-1 of `recording_id`), so windows of one recording never cross splits |
| model | BEATs `iter3_plus_AS2M`, **frozen** (no fine-tuning), 768-d, mean over all 248 patch tokens |
| licences | CC0, CC-BY, CC-BY-NC, CC-BY-SA, CC-BY-NC-SA, per row in `license`. Credits: [`ATTRIBUTION.md`](ATTRIBUTION.md) |

## How the files connect

```
iNaturalist API
   │  fetch_inaturalist.py
   ▼
interim/records_inaturalist.jsonl   one line per downloaded recording (per source)
   │  build_manifest.py   (merge sources, drop duplicates, add recording_id)
   ▼
interim/recordings.csv              master list: one row per recording
   │  preprocess.py --tag <variant>  (trim, cut 5 s windows, quality filters, split)
   ▼
interim/clips_<variant>.csv         one row per kept clip   + preprocess_problems_<variant>.log
   │  embed.py   (BEATs, frozen)
   ▼
embeddings/embeddings_<variant>.npy   (N, 768) float32
embeddings/clip_order_<variant>.csv   N rows, row i describes vector i
```

## The five variants

All five use the same model and the same recordings. They differ only in the
quality filter applied to each 5 s window before embedding.

| variant | filter | clips | kNN species purity* | lift over chance |
|---|---|---|---|---|
| `baseline` | loudness floor only (8 Sep corpus) | 12,007 | 25.5% | 4.9× |
| `nofilter` | loudness floor only (20 Sep corpus) | 11,757 | 26.2% | 5.0× |
| **`band400`** | + ≥ 25% of energy in 400 Hz – 9 kHz | **5,491** | **32.3%** | **5.8×** |
| `filtered` | + ≥ 25% of energy in 1.5 – 9 kHz (8 Sep corpus) | 3,809 | 32.3% | 5.2× |
| `band1500` | + ≥ 25% of energy in 1.5 – 9 kHz (20 Sep corpus) | 3,774 | 33.0% | 5.3× |

\* share of each clip's 10 nearest neighbours (cosine, 768-d) that are the same
species, excluding neighbours from the same recording. Chance is ~5–6%.

**Use `band400`.** The 1.5 kHz floor throws away the low-pitched song of both
doves (~300–700 Hz) and Downy Woodpecker drumming; White-winged Dove drops
*below* chance. 400 Hz keeps them and still removes wind/traffic rumble.

Common filters in every variant: trim leading/trailing audio more than 35 dB
below the recording's peak; drop recordings shorter than 3 s after trimming;
drop windows with RMS < 0.0035 (≈ −49 dBFS).

## File reference

### `embeddings/`

| file | shape / size | what it is |
|---|---|---|
| `embeddings_band400.npy` | (5491, 768) float32, 17 MB | BEATs vectors for the `band400` clips |
| `embeddings_band1500.npy` | (3774, 768) float32, 12 MB | … for `band1500` |
| `embeddings_nofilter.npy` | (11757, 768) float32, 36 MB | … for `nofilter` |
| `clip_order_<variant>.csv` | one per variant | **the row index for the matching `.npy`**: row *i* describes vector *i*. Clips are sorted by `clip_id` |

`baseline` and `filtered` have a `clip_order` file but **no `.npy` here**. Their
vectors survive only as Projector TSVs in the original working copy; regenerate
with `embed.py` if needed (`nofilter` / `band1500` are near-identical replacements).

Load a variant:

```python
import numpy as np, pandas as pd
X = np.load("data/embeddings/embeddings_band400.npy")       # (5491, 768)
meta = pd.read_csv("data/embeddings/clip_order_band400.csv")
assert len(X) == len(meta)
train = meta["split"].eq("train").to_numpy()
X_train, y_train = X[train], meta.loc[train, "species_code"]
```

### `interim/`

| file | rows | what it is |
|---|---|---|
| `records_inaturalist.jsonl` | 5,826 lines | raw fetch log, one JSON object per downloaded recording. Contains repeats from re-runs; `build_manifest.py` de-duplicates them |
| `records_field.jsonl` | 0 | placeholder for your own field recordings; empty so far |
| `recordings.csv` | 5,109 | **master recording list** (20 Sep rebuild): every source merged and de-duplicated. Input to preprocessing |
| `recordings.orig-20260908.csv` | 5,146 | the original 8 Sep master list, kept for comparison |
| `clips_<variant>.csv` | 3,774 – 11,757 | one row per 5 s clip that passed that variant's filter. Same content as `embeddings/clip_order_<variant>.csv` |
| `preprocess_problems_<variant>.log` | 1 line per dropped recording | why a recording yielded no clips: `shorter than 3.0s after trim` or `no window passed the RMS + bird-band filters` (for `nofilter` this means the RMS floor alone) |
| `preprocess_problems.log`, `…orig-20260908.log` | 3,644 | the same log for the 8 Sep `filtered` run (the two files are identical) |
| `inat_taxon_ids.json` | 20 | cache: scientific name → iNaturalist taxon id |

There is no `clips_baseline.csv`; its clip list is `embeddings/clip_order_baseline.csv`.

## Column reference

**`recordings.csv`** (one row per recording)

| column | meaning |
|---|---|
| `recording_id` | globally unique key, `<source>_<source_recording_id>`. **The unit for splitting** |
| `source_recording_id` | id within the source, e.g. `inat100633812s327217` = iNaturalist observation 100633812, sound 327217 |
| `dataset_source` | `inaturalist` (later also `xeno-canto`, `birdclef`, `field`) |
| `species_code`, `scientific_name`, `common_name` | label; eBird code, Latin and English name |
| `xc_recording_id` | original Xeno-canto id, used to de-duplicate across sources; empty for iNaturalist |
| `country` | last token of iNaturalist's free-text place. **Messy**: `US`/`USA`/`United States` all appear, and some rows hold park names |
| `location` | free-text place string |
| `quality_rating` | `research` for every iNaturalist row |
| `recordist`, `license` | attribution; required by CC-BY licences |
| `audio_path` | path of the raw file relative to `data/raw/` (not included here) |
| `source_url` | the iNaturalist observation page |

**`clips_<variant>.csv` / `clip_order_<variant>.csv`** (one row per 5 s clip). Same
metadata columns as above, plus:

| column | meaning |
|---|---|
| `clip_id` | `<recording_id>_seg<NN>`; `seg02` is the 3rd window of that recording |
| `clip_path` | WAV path relative to `data/clips/` (not included here) |
| `split` | `train` / `val` / `viz`. `viz` is the held-out set |
| `segment_start_sec` | where the window starts in the trimmed recording |
| `duration_sec`, `sample_rate_hz` | always 5.0 and 16000 |
| `bird_band_fraction` | share of the window's spectral energy inside the band; not in `clip_order_baseline.csv` |

## Known quirks

- Some text has broken accented characters (`M�xico`) from an encoding issue in
  the iNaturalist fetch.
- `country` should be normalised before it is used as a feature or for colouring.
- The species are not perfectly balanced: in `band400`, Laughing Gull has 132
  clips and Northern Mockingbird 556.
