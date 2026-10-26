# SemMoic

Reference implementation of the method described in the accompanying paper.

This repository contains the proposed two-stage method.

## Method

SemMoic combines semantically diverse data enrichment with a two-stage
open intent classifier:

1. **Representation learning.** A BERT / Swin / WavLM encoder with a weighted
   fusion layer is trained with supervised contrastive learning and
   cross-entropy on the enriched training set (original samples, label-preserving
   augmentations, and unlabeled wild data).
2. **Source-aware ellipsoidal boundary.** The frozen encoder produces
   normalized representations. For each known intent, a class-specific
   ellipsoid is fitted with a convex objective whose positive support comes from
   original and augmented samples, and whose negative support comes from other
   known classes and target-specific adversarial samples. At test time the
   cosine classifier selects one known class; the sample is accepted only if it
   lies inside that class's ellipsoid, otherwise it is predicted as open.

## Repository layout

```
configs.py                  Fixed hyperparameters for the two datasets
train.py                    Stage 1: representation training
export.py                   Stage 2: frozen feature export
fit_boundary.py             Stage 3: boundary fitting and evaluation
run.py                      One-command orchestrator for the three stages
data/                       Dataset and feature loading
backbones/                  Multimodal encoder and fusion network
losses/                     Supervised contrastive loss
methods/semmoic/            Stage-1 model training loop
ood_detection/pba/          Source-aware ellipsoidal boundary solver
utils/                      Metrics, seeding, early stopping
examples/                   One-command reproduction scripts
DATA.md                     Expected dataset layout
```

## Environment

```bash
conda env create -f environment.yml -n SemMoic
conda activate SemMoic
```

The code runs on a single GPU. CPU-only execution is supported but slow.

## Data

The enriched datasets and precomputed WavLM / Swin features are required.
See [DATA.md](DATA.md) for the exact directory layout. By default the scripts
read from `./datasets`; set `DATA_ROOT` to use another location. Feature file names
are specified in `configs.py`; update them there if your copy uses different
names.

The BERT backbone defaults to `bert-base-uncased` from HuggingFace. Set
`BERT_PATH` to a local BERT directory to run offline.

## Reproducing the paper

Run both datasets with five seeds:

```bash
DATA_ROOT=/path/to/datasets bash examples/run_all.sh
```

Or run one dataset:

```bash
DATA_ROOT=/path/to/datasets bash examples/run_train_MIntRec.sh
DATA_ROOT=/path/to/datasets bash examples/run_train_MIntRec2.0.sh
```

Each command trains the representation model, exports frozen features, fits the
boundary, evaluates the test set, and writes the four metrics for every seed.
Use `GPU=n` to choose a GPU and `SEEDS=0,1` to run a subset.

The three steps can also be launched separately. Each step takes an explicit
seed and sets it before doing any work:

```bash
python train.py --dataset MIntRec --data_root /path/to/datasets --seed 0 \
  --run_dir outputs/MIntRec/seed_0 --gpu 0

python export.py --dataset MIntRec --data_root /path/to/datasets --seed 0 \
  --run_dir outputs/MIntRec/seed_0 --output outputs/MIntRec/seed_0/features.npz --gpu 0

python fit_boundary.py --dataset MIntRec --seed 0 \
  --inputs outputs/MIntRec/seed_0/features.npz \
  --output_dir outputs/MIntRec/seed_0 --results_dir results
```

## Outputs

```
results/<dataset>/seed_<k>.json    Per-seed metrics and boundary state path
results/<dataset>/summary.csv      Per-seed metrics plus mean and standard deviation
results/<dataset>/summary.json     Machine-readable summary
outputs/<dataset>/seed_<k>/        Selected encoder, classifier, features, and boundary state
logs/                              Training, export, fit, and run logs
```

All four metrics are reported in percent: `acc`, `f1` (all classes),
`f1-known`, and `f1-open`. Open intent samples appear only in the test sets.

## Expected results

Five seeds (0-4), mean ± standard deviation over seeds:

| Dataset | Acc | F1-All | F1-Known | F1-Open |
| --- | --- | --- | --- | --- |
| MIntRec | 67.60 ± 0.80 | 56.75 ± 0.44 | 55.89 ± 0.44 | 74.05 ± 0.92 |
| MIntRec2.0 | 51.49 ± 0.50 | 44.87 ± 0.50 | 44.48 ± 0.52 | 56.66 ± 0.74 |

## Reproducibility notes

- Runs are deterministic for a fixed seed, GPU model, and library stack. Each
  step is a separate script (`train.py`, `export.py`, `fit_boundary.py`) and
  sets the seed before building data or models, so the steps can be rerun
  independently and remain bit-identical.
- The boundary solver runs in float64 on CPU with single-threaded BLAS;
  `run.py` and `fit_boundary.py` set the corresponding environment variables
  before importing NumPy.
- The representation model uses 20 training epochs, early stopping on
  validation accuracy, and five random seeds. MIntRec uses patience 5 and
  selects the best epoch; MIntRec2.0 uses patience 3 and selects the best epoch
  at or after epoch 5.
- The released checkpoints are not included. The paper results are obtained by
  running the full pipeline from the enriched data.

## Citation

Citation information will be added after publication.
