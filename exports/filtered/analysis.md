# Embedding analysis -- tag `filtered`

- points: **3,809** clips from **1,502** recordings
- dimensions: **768**
- species: **20**
- sources: inaturalist

- PCA variance explained: PC1-2 **12.4%**, PC1-10 **38.9%**, PC1-50 **72.0%**

## Species separability (k=10 NN, same-recording neighbours excluded)

- **32.3%** of neighbours share the query's species (chance = 6.2%, so **5.2x** chance)

| rank | species | common name | kNN purity |
|---|---|---|---|
| 1 | `laugul` | Laughing Gull | 68.3% |
| 2 | `amecro` | American Crow | 45.8% |
| 3 | `normoc` | Northern Mockingbird | 44.2% |
| 4 | `houspa` | House Sparrow | 43.5% |
| 5 | `grtgra` | Great-tailed Grackle | 41.9% |
| 6 | `blujay` | Blue Jay | 39.6% |
| 7 | `barswa` | Barn Swallow | 37.3% |
| 8 | `rewbla` | Red-winged Blackbird | 31.9% |
| 9 | `grbher` | Great Blue Heron | 31.7% |
| 10 | `carwre` | Carolina Wren | 31.6% |
| 11 | `killde` | Killdeer | 29.1% |
| 12 | `norcar` | Northern Cardinal | 28.1% |
| 13 | `amerob` | American Robin | 27.5% |
| 14 | `carchi` | Carolina Chickadee | 21.8% |
| 15 | `rebwoo` | Red-bellied Woodpecker | 16.9% |
| 16 | `tuftit` | Tufted Titmouse | 16.7% |
| 17 | `easblu` | Eastern Bluebird | 14.6% |
| 18 | `dowwoo` | Downy Woodpecker | 13.7% |
| 19 | `moudov` | Mourning Dove | 10.6% |
| 20 | `whwdov` | White-winged Dove | 6.5% |

**Tightest:** laugul (68%), amecro (46%), normoc (44%)
**Most overlapped:** dowwoo (14%), moudov (11%), whwdov (6%)

## Dataset-source batch effect

Only one source present (`inaturalist`), so this test is not yet meaningful. Re-run once Xeno-canto, BirdCLEF or field audio is added: if source purity lands far above chance while species purity does not, the projection is partly organised by *where the audio came from* rather than by *which bird is singing*.

