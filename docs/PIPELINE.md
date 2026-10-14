# Bird Song Embedding Visualization

Take raw bird song audio, extract embeddings with BEATs, and explore the
embedding space in the TensorFlow Embedding Projector.

Sub-project of *BEATs: Audio Pre-Training with Acoustic Tokenizers*.

---

## Status

| Phase | State |
|---|---|
| 1 — Data acquisition | **Done** for iNaturalist: 5,146 recordings, 20 species. Xeno-canto + BirdCLEF + field paths built, each waiting on one credential (below). |
| 2 — Preprocessing | **Done** — 16 kHz mono, silence-trimmed, 5 s fixed windows, recording-level splits. |
| 3a — Baseline embeddings | **Done** — BEATs `iter3_plus_AS2M`, 768-d, mean-pooled over time. |
| 3b — Bird-pretrained | **Blocked on GPU** — see *GPU quota* below. Same interface, `--tag birdpretrained`. |
| 4 — Projector export | **Done and validated** against the §3.5 checklist. |
| 5 — Visualization | **Done** — two validated exports, quantified in `docs/FINDINGS.md`. |
| 6 — Documentation | This file, `docs/FINDINGS.md`, `exports/ATTRIBUTION.md` (5,146 recordings, 5 licences). |

### Results at a glance

Two exports, both validated, meant to be loaded side by side:

| | `exports/baseline` | `exports/filtered` |
|---|---|---|
| points | 12,007 | 3,809 |
| Phase 2 gate | RMS floor only | RMS floor + ≥25% energy in 1.5–9 kHz |
| kNN species purity (k=10, same-recording neighbours excluded) | 25.5% (4.9× chance) | **32.3% (5.2× chance)** |

Off-the-shelf BEATs carries real species structure but is far from clean — which
is the number Phase 3b has to beat. Full write-up, including why the band filter
*harms* the two dove species, is in **[docs/FINDINGS.md](docs/FINDINGS.md)**.

### Three things need a credential you have and I don't

1. **Xeno-canto** — API v2 was retired (it now returns HTTP 404 with a pointer to
   v3), and v3 requires a free per-account key. Get one at
   <https://xeno-canto.org/account>, then:
   ```bash
   export XC_API_KEY=<key>          # or write it to config/xc_api_key.txt
   python src/fetch_xenocanto.py
   ```
2. **BirdCLEF** — needs a Kaggle token at `~/.kaggle/kaggle.json` and acceptance
   of the competition rules, then `python src/fetch_birdclef.py --competition birdclef-2026`.
3. **Field recordings** — drop audio under `data/raw/field/<species_code>/` and
   describe each file in `config/field_recordings.csv` (the template is already
   generated with the exact columns).

Adding any of these changes **no code**. Re-run the pipeline and the new rows
join the same manifest, the same splits, and the same export.

### GPU quota

Phase 3b needs a GPU and this subscription has **zero GPU quota in every region**
(all NC/ND/NV families report limit 0), so the tokenizer + encoder pretraining
cannot start until a quota increase is granted. Request one at
Azure Portal → Quotas → Compute → pick a region → request e.g. `Standard NCADSA10v4`
or `NCasT4_v3` cores. Everything else in this repo is CPU-only and already runs.

---

## The audio contract

Every source is normalised to one contract before it reaches the model
(`src/config.py` is the single source of truth):

| | |
|---|---|
| sample rate | 16 000 Hz, mono |
| clip length | 5.0 s, non-overlapping |
| silence trim | librosa, `top_db=35` |
| quality floor | RMS ≥ 0.0035 after trim; recordings < 3 s dropped |
| per-recording cap | 8 segments, so one long recording cannot dominate a species |
| splits | 70/15/15 train/val/viz, assigned **per recording** (SHA-1 of the id) |

The split is deterministic and recording-level, so clips from one recording can
never straddle two splits. `preprocess.py` asserts this rather than assuming it.

---

## Pipeline

```bash
python src/fetch_inaturalist.py --per-species 60   # Phase 1
python src/fetch_xenocanto.py                      #   (needs XC_API_KEY)
python src/fetch_birdclef.py                       #   (needs kaggle.json)
python src/fetch_field.py                          #   (needs your recordings)
python src/build_manifest.py                       #   merge + dedupe
python src/preprocess.py --workers 16              # Phase 2
python src/embed.py --tag baseline --threads 16    # Phase 3a
python src/export_projector.py --tag baseline      # Phase 4
python src/validate_export.py --tag baseline \
    --expect-order-from data/embeddings/clip_order_baseline.csv
python src/analyze_clusters.py --tag baseline      # Phase 5
python scripts/make_attribution.py                 # Phase 6
```

`scripts/run_pipeline.sh` runs the whole sequence with logging.

### Deduplication

BirdCLEF's training audio is largely re-hosted Xeno-canto, so the same recording
can arrive from two sources and would otherwise become two points in the
projector. `build_manifest.py` recovers the original Xeno-canto id (from the
BirdCLEF URL or filename) and keeps one canonical copy, preferring
field → xeno-canto → birdclef → inaturalist.

---

## Viewing the projection

1. Open <https://projector.tensorflow.org>.
2. **Load** → *Step 1* `exports/baseline/vectors.tsv`, *Step 2* `exports/baseline/metadata.tsv`.
3. Colour by `species_code`; switch to `dataset_source` to look for a batch effect.

Nothing is uploaded — the page is static HTML/JS and parses both files in your
browser. If that page ever disappears, the identical UI ships inside TensorBoard
(`tensorboard --logdir=exports/baseline` → PROJECTOR tab) and reads the same
`projector_config.json` written alongside the TSVs.

### Swapping in the Phase 3b model

The export step never changes — only the source of the vectors does:

```bash
python src/embed.py --checkpoint <bird-pretrained> --tag birdpretrained
python src/export_projector.py --tag birdpretrained
```

Keep both loadable side by side rather than replacing one with the other.

---

## Export format

Two files, no header on the vectors, **positional alignment only** — row *i* of
`metadata.tsv` describes the point at row *i* of `vectors.tsv`.

`export_projector.py` reads one ordered clip list and writes both files from it
without re-sorting in between; `validate_export.py` then proves it, checking
row counts, uniform column widths, no NaN/Inf, no stray tabs or carriage
returns, and that row *i* of the export really is row *i* of the embedding order.

> A real bug this caught: pandas writes CRLF on Windows, which glues a `\r` onto
> the last metadata column of every row. Both writers now force LF.

---

## Azure environment

Everything lives in one disposable resource group, `birdclef-rg` (East US),
isolated from the other workloads in this subscription.

| resource | purpose |
|---|---|
| `birdclef-vm` (`Standard_D16ads_v7`, 16 vCPU) | fetch, preprocess, BEATs inference |
| `birdclefstore69916` | blob containers `code`, `corpus`, `exports` |

SSH is closed by default; the VM is driven through `az vm run-command`. Note
that the Azure guest agent **redacts SAS tokens** out of run-command scripts, so
blob URLs with a `sig=` parameter arrive empty — use managed identity, or move
data over SSH.

**Teardown — this is the only thing that stops the meter:**

```bash
az vm deallocate -g birdclef-rg -n birdclef-vm    # stop compute billing
az group delete -n birdclef-rg --yes --no-wait    # remove everything
```
