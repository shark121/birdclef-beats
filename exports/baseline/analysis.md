# Embedding analysis -- tag `baseline`

- points: **12,007** clips from **3,958** recordings
- dimensions: **768**
- species: **20**
- sources: inaturalist

- PCA variance explained: PC1-2 **12.1%**, PC1-10 **38.8%**, PC1-50 **70.5%**

## Species separability (k=10 NN, same-recording neighbours excluded)

- **25.5%** of neighbours share the query's species (chance = 5.2%, so **4.9x** chance)

| rank | species | common name | kNN purity |
|---|---|---|---|
| 1 | `laugul` | Laughing Gull | 53.6% |
| 2 | `amecro` | American Crow | 36.4% |
| 3 | `grbher` | Great Blue Heron | 34.4% |
| 4 | `houspa` | House Sparrow | 33.8% |
| 5 | `normoc` | Northern Mockingbird | 31.2% |
| 6 | `grtgra` | Great-tailed Grackle | 30.9% |
| 7 | `whwdov` | White-winged Dove | 30.7% |
| 8 | `barswa` | Barn Swallow | 29.6% |
| 9 | `moudov` | Mourning Dove | 26.2% |
| 10 | `blujay` | Blue Jay | 26.1% |
| 11 | `carwre` | Carolina Wren | 23.8% |
| 12 | `carchi` | Carolina Chickadee | 23.7% |
| 13 | `rewbla` | Red-winged Blackbird | 22.2% |
| 14 | `killde` | Killdeer | 21.5% |
| 15 | `norcar` | Northern Cardinal | 19.3% |
| 16 | `amerob` | American Robin | 17.9% |
| 17 | `dowwoo` | Downy Woodpecker | 17.5% |
| 18 | `easblu` | Eastern Bluebird | 14.9% |
| 19 | `tuftit` | Tufted Titmouse | 12.2% |
| 20 | `rebwoo` | Red-bellied Woodpecker | 10.3% |

**Tightest:** laugul (54%), amecro (36%), grbher (34%)
**Most overlapped:** easblu (15%), tuftit (12%), rebwoo (10%)

## Dataset-source batch effect

Only one source present (`inaturalist`), so this test is not yet meaningful. Re-run once Xeno-canto, BirdCLEF or field audio is added: if source purity lands far above chance while species purity does not, the projection is partly organised by *where the audio came from* rather than by *which bird is singing*.

