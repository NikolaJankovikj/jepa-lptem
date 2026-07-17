# Data sources (verified 2026-07-17)

All datasets/weights are gitignored (`/data/`), all upstream code is gitignored (`/code/`).
Nothing here is committed to this repo.

## HuggingFace repos (verified via `list_repo_files` / `repo_info`)

### `papertemp/SAM-EM-Paper-PublicData` (dataset) — the authoritative SAM-EM data repo
Contains the segmentation oracle + synthetic + real clips. (`Siddarth-Sreeram/SAM-EM-Paper`
is a near-identical mirror missing `experimentalAggregation`; prefer `papertemp/…`.)

| File | Size | Split / role | Where |
|---|---|---|---|
| `ablationTrainingSet-2kDatasetSize.zip` | 64 GB | large ablation training set | **deferred / probably skip** |
| `trainset.zip` | 32 GB | full synthetic pretraining set | remote |
| `SAM-EMCheckpoint.pt` | 2.7 GB | trained segmentation model (eval oracle, Steps 11/13) | remote |
| `holdoutValidationSet.zip` | 2.3 GB | held-out synthetic + masks (probe train/eval) | remote |
| `ablationValidationSet.zip` | 96 MB | small synthetic + masks | **local dev subset** |
| `experimentalAuNanorods.zip` | 33 MB | real Au-nanorod clips (real-data validation) | local |
| `holdoutValidationSet-4ParticlesOverlapping` | 32 MB | small overlapping-particle holdout | local |
| `experimentalAggregation.zip` | 3 MB | real aggregation clips (event anticipation) | local |

### `JamaliLab/LEONDARDO` (dataset — note the upstream typo "LEONDARDO", it is real)
| File | Size | Role |
|---|---|---|
| `datasets.zip` | 175 MB | LEONARDO trajectory datasets (motion stats for simulator + comparison) | local |
| `particle_dose_size_experimental_trajectories.xlsx` | 13 MB | experimental trajectory table | local |

### `JamaliLab/LEONARDO` (model)
| File | Size | Role |
|---|---|---|
| `LEONARDOModel` | 106 MB | trained LEONARDO generative model | local |
| `datasets.zip` | 175 MB | (same trajectory data as above) | — |

### `facebook/vjepa2-vitl-fpc64-256` (model) — V-JEPA 2 ViT-L encoder
Loaded on demand via HF `transformers` (`AutoModel.from_pretrained`); cached under HF cache, not in `data/`.

## Upstream code repos (cloned shallow into `code/`, gitignored)
| Repo | Why we need it |
|---|---|
| `github.com/JamaliLab/SAM-EM` | The simulator/renderer we extend (Step 5) + segmentation model code |
| `github.com/JamaliLab/LEONARDO` | Trajectory motion sampling + reference model |
| `github.com/facebookresearch/vjepa2` | The official SSL pretraining loop for LoRA adaptation (Step 8) |

## Local footprint (dev subsets only, ≈ 460 MB total)
`ablationValidationSet.zip`, `experimentalAuNanorods.zip`, `experimentalAggregation.zip`,
`holdoutValidationSet-4ParticlesOverlapping`, LEONARDO `datasets.zip` + model + xlsx.
Everything ≥ 2 GB is pulled directly on the remote in Step 2/6.
