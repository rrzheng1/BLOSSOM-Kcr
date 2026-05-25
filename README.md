# BLOSSOM-Kcr

BLOSSOM-Kcr is a deep learning baseline for predicting lysine crotonylation (Kcr) sites from protein sequences and protein structure embeddings. The model combines local BLOSUM62 sequence-window features with precomputed structure embeddings, then trains a CNN-BiLSTM-SE network with attention pooling for binary site classification.

This repository contains the training script, curated train/test splits, 5-fold cross-validation splits, structure embedding files, and example result figures.

## Highlights

- Uses a fixed local window around each candidate lysine site.
- Encodes sequence context with BLOSUM62.
- Integrates per-residue structure embeddings loaded from `.pt` files.
- Trains a CNN + BiLSTM + squeeze-and-excitation model with attention pooling.
- Supports 5-fold cross-validation and independent test-set evaluation.
- Reports AUC, AUPR, ACC, F1, MCC, precision, sensitivity and specificity.

## Repository Structure

```text
.
├── baseline.py                 # Main training, validation and independent-test script
├── environment.yml             # Conda environment based on the ptm_gpu runtime
├── data/
│   ├── train_80.csv            # Training set
│   ├── test_20.csv             # Independent test set
│   └── cv_folds/               # Protein-level 5-fold CV splits
├── structure_embedding/        # Precomputed per-protein structure embeddings
└── figure/                     # Example figures and model comparison plots
```

The included data currently contains:

- `data/train_80.csv`: 17,481 samples
- `data/test_20.csv`: 4,846 samples
- `data/cv_folds/`: 5 protein-disjoint cross-validation folds
- `structure_embedding/`: 3,066 `.pt` embedding files and 3,066 `.json` metadata files

## Model Overview

For each candidate residue, `baseline.py` builds a local residue window with size 31 by default:

1. BLOSUM62 features are extracted from the protein sequence.
2. Structure features are sliced from the corresponding per-residue `.pt` embedding.
3. The two feature blocks are concatenated along the feature dimension.
4. The concatenated window is passed through:
   - 1D convolution blocks
   - residual convolution
   - squeeze-and-excitation channel recalibration
   - bidirectional LSTM
   - attention pooling
   - fully connected binary classifier

The model is optimized with `BCEWithLogitsLoss`, AdamW, cosine annealing warm restarts, gradient clipping, optional AMP, and early stopping based on validation AUC.

## Data Format

Input CSV files should contain the following columns:

| Column | Description |
| --- | --- |
| `protein` | Protein identifier, matching the stem of a `.pt` file in `structure_embedding/` |
| `Position` | 1-based residue position |
| `Residue` | Residue symbol, usually `K` for lysine candidate sites |
| `y` | Binary label, `1` for Kcr site and `0` for non-Kcr site |
| `sequence` | Full protein sequence |

The loader also accepts common column-name variants such as `position`, `pos`, `label`, `class`, `seq` and `uniprot`.

Each structure embedding file should be named by protein ID, for example:

```text
structure_embedding/Q9NQ88.pt
```

The tensor is expected to represent per-residue features with shape:

```text
sequence_length x embedding_dim
```

## Installation

The recommended setup is to create the Conda environment from `environment.yml`. This file records the core packages from the original `ptm_gpu` runtime while omitting machine-specific paths and low-level transient packages from the full Conda export.

```bash
conda env create -f environment.yml
conda activate ptm_gpu
```

Verify that PyTorch can see the GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The provided environment uses PyTorch `2.5.1+cu121`. If your GPU driver or CUDA runtime requires a different PyTorch build, install the matching package from the official PyTorch distribution and keep the rest of the environment unchanged.

For a minimal CPU-only environment, install only the packages required by `baseline.py`:

```bash
conda create -n blossom-kcr-cpu python=3.10
conda activate blossom-kcr-cpu
pip install numpy pandas scipy scikit-learn torch
```

## Quick Start

Run 5-fold cross-validation and independent testing:

```bash
python baseline.py \
  --cv-dir data/cv_folds \
  --structure-dir structure_embedding \
  --test-csv data/test_20.csv \
  --output-dir outputs/baseline \
  --device cuda \
  --amp
```

Run on CPU:

```bash
python baseline.py \
  --cv-dir data/cv_folds \
  --structure-dir structure_embedding \
  --test-csv data/test_20.csv \
  --output-dir outputs/baseline_cpu \
  --device cpu
```

Skip the independent test and only run cross-validation:

```bash
python baseline.py \
  --cv-dir data/cv_folds \
  --structure-dir structure_embedding \
  --output-dir outputs/cv_only \
  --skip-independent-test
```

## Main Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--cv-dir` | `/data/cv_folds` | Directory containing `fold_*_train.csv` and `fold_*_val.csv` |
| `--structure-dir` | `/structure_embedding` | Directory containing `.pt` structure embeddings |
| `--test-csv` | `/data/test_20.csv` | Independent test CSV |
| `--output-dir` | `/` | Output directory |
| `--window-size` | `31` | Local sequence/structure window size |
| `--batch-size` | `32` | Batch size |
| `--epochs` | `50` | Maximum training epochs per fold |
| `--patience` | `10` | Early-stopping patience |
| `--lr` | `5e-5` | Learning rate |
| `--dropout` | `0.3` | Dropout rate |
| `--rnn-layers` | `2` | Number of BiLSTM layers |
| `--weight-decay` | `1e-4` | AdamW weight decay |
| `--device` | `cuda` | Use `cuda` when available, otherwise falls back to CPU |
| `--amp` | disabled | Enable automatic mixed precision |

For local use, pass explicit relative paths as shown in the quick-start commands because the script defaults are absolute paths used in the original runtime environment.

## Outputs

The script writes one directory per fold and summary files under `--output-dir`:

```text
outputs/baseline/
├── fold_1/
│   ├── best_model.pt
│   ├── best_val_predictions.csv
│   └── history.csv
├── ...
├── fold_5/
├── cv_summary.json
└── independent_test/
    ├── test_predictions.csv
    └── test_summary.json
```

Key output files:

- `best_model.pt`: best checkpoint for each fold.
- `history.csv`: validation metrics by epoch.
- `best_val_predictions.csv`: validation predictions from the best epoch.
- `cv_summary.json`: mean and standard deviation of CV metrics.
- `test_predictions.csv`: fold-level and ensemble test scores.
- `test_summary.json`: independent-test metrics.

## Example Figures

The `figure/` directory contains example visualizations, including ROC/AUPRC comparisons, bootstrap significance plots, t-SNE visualization and SHAP-based interpretation figures.

## Reproducibility Notes

- The script sets the random seed to `42`.
- Cross-validation splits are provided in `data/cv_folds/`.
- Folds are protein-disjoint according to `data/cv_folds/cv_summary.csv`.
- CUDA/cuDNN behavior may still introduce minor variation across hardware and software versions.

## Citation

If you use this repository in academic work, please cite the corresponding paper or project once available.

## License

No license file is currently included. Add a `LICENSE` file before public release to define how others may use, modify and redistribute the code and data.
