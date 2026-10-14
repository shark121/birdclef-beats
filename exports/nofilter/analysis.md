# Embedding analysis -- tag `nofilter`

- points: **11,757** clips from **3,880** recordings
- dimensions: **768**
- species: **20**
- sources: inaturalist

- PCA variance explained: PC1-2 **12.5%**, PC1-10 **38.4%**, PC1-50 **70.3%**

## Species separability (k=10 NN, same-recording neighbours excluded)

- **26.2%** of neighbours share the query's species (chance = 5.2%, so **5.0x** chance)

| rank | species | common name | kNN purity |
|---|---|---|---|
| 1 | `laugul` | Laughing Gull | 52.8% |
| 2 | `amecro` | American Crow | 39.2% |
| 3 | `normoc` | Northern Mockingbird | 35.8% |
| 4 | `grbher` | Great Blue Heron | 35.3% |
| 5 | `grtgra` | Great-tailed Grackle | 32.2% |
| 6 | `whwdov` | White-winged Dove | 30.8% |
| 7 | `houspa` | House Sparrow | 30.7% |
| 8 | `barswa` | Barn Swallow | 28.8% |
| 9 | `blujay` | Blue Jay | 27.6% |
| 10 | `carwre` | Carolina Wren | 24.5% |
| 11 | `carchi` | Carolina Chickadee | 24.2% |
| 12 | `moudov` | Mourning Dove | 23.6% |
| 13 | `killde` | Killdeer | 23.3% |
| 14 | `rewbla` | Red-winged Blackbird | 23.3% |
| 15 | `norcar` | Northern Cardinal | 18.5% |
| 16 | `dowwoo` | Downy Woodpecker | 18.1% |
| 17 | `amerob` | American Robin | 17.1% |
| 18 | `tuftit` | Tufted Titmouse | 15.0% |
| 19 | `rebwoo` | Red-bellied Woodpecker | 14.2% |
| 20 | `easblu` | Eastern Bluebird | 14.0% |

**Tightest:** laugul (53%), amecro (39%), normoc (36%)
**Most overlapped:** tuftit (15%), rebwoo (14%), easblu (14%)

## Dataset-source batch effect

Only one source present (`inaturalist`), so this test is not yet meaningful. Re-run once Xeno-canto, BirdCLEF or field audio is added: if source purity lands far above chance while species purity does not, the projection is partly organised by *where the audio came from* rather than by *which bird is singing*.

