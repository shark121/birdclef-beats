# Findings — first pass, 8 September 2026

Baseline (Phase 3a) embeddings only: a pretrained BEATs `iter3_plus_AS2M`
checkpoint used as a frozen feature extractor, mean-pooled over time, 768-d.
Corpus is iNaturalist research-grade audio for 20 Texas Gulf Coast species.
No bird-specific pretraining has happened yet — that is Phase 3b.

## Two exports, loadable side by side

| | `baseline` | `filtered` |
|---|---|---|
| points (clips) | 12,007 | 3,809 |
| recordings | 3,958 | 1,502 |
| Phase 2 quality gate | RMS floor only | RMS floor **+ ≥25% energy in 1.5–9 kHz** |
| kNN species purity (k=10) | 25.5% | **32.3%** |
| chance rate | 5.2% | 6.2% |
| lift over chance | 4.9× | **5.2×** |
| PC1–10 variance | 38.8% | 38.9% |

Neighbour tests **exclude same-recording neighbours**. Without that exclusion the
numbers look far better for a trivial reason: two 5 s clips cut from one
recording are near-duplicates, and scoring them as neighbours measures nothing
about generalisation.

## 1. Off-the-shelf BEATs already carries real species structure — but not enough

At 4.9–5.2× chance, a model that never saw a bird-specific objective clearly
encodes species-relevant acoustic structure. It is also nowhere near clean: the
best species reaches 68% neighbour purity and the worst sits near chance. This
is the expected "generic but informative" baseline, and it is the number Phase 3b
has to beat. **Keep both exports loaded when 3b lands** — the comparison is the
result, not the individual figure.

## 2. Data quality dominated everything

Two data bugs each cost more than any modelling choice available at this stage:

- **`.m4a` files silently failed to decode.** libsndfile has no AAC decoder and
  librosa 0.11 dropped the old audioread fallback, so `librosa.load()` rejected
  every `.m4a` iNaturalist serves — **770 of 1,200 recordings (64%)** vanished
  from the first corpus. Routing failures through ffmpeg recovered them and took
  the corpus from 1,390 to 12,007 clips.
- **An RMS floor does not mean "a bird is audible".** Wind, traffic and handling
  noise are loud and sail straight through it. A 20-clip spot-check found **8 of
  20** clips were sub-500 Hz rumble containing no song at all.

Adding the bird-band filter lifted median in-band energy from 53% to 67%, took
rumble-dominated clips in the spot-check from 8/20 to **0/20**, and raised
species purity by 6.8 points — while discarding 68% of the corpus.

## 3. The band filter systematically destroys low-pitched species

This is the finding to act on. The 1.5–9 kHz band suits passerines, but the two
*Columbidae* coo well below its floor (~300–700 Hz), so the filter throws away
their real song and keeps only atypical clips:

| species | clips kept | kNN purity, baseline → filtered |
|---|---|---|
| Mourning Dove | **12.1%** | 26.2% → **10.6%** |
| White-winged Dove | **13.0%** | 30.7% → **6.5%** |
| Northern Mockingbird | 52.1% | 31.2% → 44.2% |
| Carolina Wren | 46.4% | 23.8% → 31.6% |

Both doves have by far the lowest survival rates and are the only species whose
purity *collapsed* under filtering. A single global band is the wrong tool.

**Recommended fix:** lower the band floor to ~400 Hz, or make the band
species-aware, before treating `filtered` as the canonical corpus. Until then,
`baseline` is the fairer view for doves and `filtered` the better view for
everything else.

## 4. Per-species ordering is stable and ecologically sensible

Laughing Gull is the tightest cluster in both runs (54% / 68%) — a loud,
broadband, highly stereotyped call. American Crow and Great-tailed Grackle also
rank high. The species that overlap most are the quiet, tonal, narrow-band
singers — Tufted Titmouse, Eastern Bluebird, Red-bellied Woodpecker — which is
what you would expect from a model pretrained on general audio.

## 5. Not yet answerable

**Dataset-source batch effect.** Only one source is populated, so the test is
not meaningful yet. The check is implemented and runs automatically once
Xeno-canto, BirdCLEF or field audio is added. The thing to watch: if source
purity lands far above chance while species purity does not, the projection is
organised by *where the audio came from* rather than *which bird is singing*.

## Next actions

1. Add the Xeno-canto API v3 key → a second source, and the batch-effect test.
2. Lower the band floor to ~400 Hz and re-run; check whether the doves recover.
3. Request GPU quota (currently zero in every region) to start Phase 3b.
4. Collect the Galveston Bay field recordings — the species list was chosen so
   they drop into this same label space.
