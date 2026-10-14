# Embedding analysis -- tag `band1500`

- points: **3,774** clips from **1,473** recordings
- dimensions: **768**
- species: **20**
- sources: inaturalist

- PCA variance explained: PC1-2 **12.7%**, PC1-10 **39.0%**, PC1-50 **71.8%**

## Species separability (k=10 NN, same-recording neighbours excluded)

- **33.0%** of neighbours share the query's species (chance = 6.2%, so **5.3x** chance)

| rank | species | common name | kNN purity |
|---|---|---|---|
| 1 | `laugul` | Laughing Gull | 66.8% |
| 2 | `amecro` | American Crow | 53.7% |
| 3 | `normoc` | Northern Mockingbird | 48.8% |
| 4 | `grtgra` | Great-tailed Grackle | 42.8% |
| 5 | `blujay` | Blue Jay | 41.9% |
| 6 | `houspa` | House Sparrow | 37.9% |
| 7 | `barswa` | Barn Swallow | 37.2% |
| 8 | `grbher` | Great Blue Heron | 31.6% |
| 9 | `carwre` | Carolina Wren | 31.5% |
| 10 | `rewbla` | Red-winged Blackbird | 30.4% |
| 11 | `killde` | Killdeer | 27.5% |
| 12 | `norcar` | Northern Cardinal | 27.1% |
| 13 | `amerob` | American Robin | 25.6% |
| 14 | `rebwoo` | Red-bellied Woodpecker | 23.5% |
| 15 | `carchi` | Carolina Chickadee | 23.4% |
| 16 | `tuftit` | Tufted Titmouse | 22.8% |
| 17 | `easblu` | Eastern Bluebird | 10.9% |
| 18 | `moudov` | Mourning Dove | 10.3% |
| 19 | `dowwoo` | Downy Woodpecker | 9.4% |
| 20 | `whwdov` | White-winged Dove | 4.6% |

**Tightest:** laugul (67%), amecro (54%), normoc (49%)
**Most overlapped:** moudov (10%), dowwoo (9%), whwdov (5%)

## Dataset-source batch effect

Only one source present (`inaturalist`), so this test is not yet meaningful. Re-run once Xeno-canto, BirdCLEF or field audio is added: if source purity lands far above chance while species purity does not, the projection is partly organised by *where the audio came from* rather than by *which bird is singing*.

