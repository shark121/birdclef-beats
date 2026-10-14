# Embedding analysis -- tag `band400`

- points: **5,491** clips from **2,082** recordings
- dimensions: **768**
- species: **20**
- sources: inaturalist

- PCA variance explained: PC1-2 **12.2%**, PC1-10 **38.2%**, PC1-50 **70.6%**

## Species separability (k=10 NN, same-recording neighbours excluded)

- **32.3%** of neighbours share the query's species (chance = 5.6%, so **5.8x** chance)

| rank | species | common name | kNN purity |
|---|---|---|---|
| 1 | `laugul` | Laughing Gull | 62.7% |
| 2 | `amecro` | American Crow | 51.4% |
| 3 | `normoc` | Northern Mockingbird | 44.9% |
| 4 | `whwdov` | White-winged Dove | 42.8% |
| 5 | `grbher` | Great Blue Heron | 39.2% |
| 6 | `grtgra` | Great-tailed Grackle | 37.8% |
| 7 | `houspa` | House Sparrow | 34.7% |
| 8 | `blujay` | Blue Jay | 33.9% |
| 9 | `barswa` | Barn Swallow | 33.4% |
| 10 | `moudov` | Mourning Dove | 33.0% |
| 11 | `rewbla` | Red-winged Blackbird | 31.1% |
| 12 | `carwre` | Carolina Wren | 30.9% |
| 13 | `killde` | Killdeer | 26.7% |
| 14 | `norcar` | Northern Cardinal | 23.6% |
| 15 | `carchi` | Carolina Chickadee | 23.4% |
| 16 | `dowwoo` | Downy Woodpecker | 22.8% |
| 17 | `amerob` | American Robin | 21.2% |
| 18 | `tuftit` | Tufted Titmouse | 20.0% |
| 19 | `rebwoo` | Red-bellied Woodpecker | 18.9% |
| 20 | `easblu` | Eastern Bluebird | 9.4% |

**Tightest:** laugul (63%), amecro (51%), normoc (45%)
**Most overlapped:** tuftit (20%), rebwoo (19%), easblu (9%)

## Dataset-source batch effect

Only one source present (`inaturalist`), so this test is not yet meaningful. Re-run once Xeno-canto, BirdCLEF or field audio is added: if source purity lands far above chance while species purity does not, the projection is partly organised by *where the audio came from* rather than by *which bird is singing*.

