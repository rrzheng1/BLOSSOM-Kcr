#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline BLOSUM + Structure CNN-BiLSTM-SE - 5-Fold CV
"""

import os, gc, json, time, math, random, warnings
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, matthews_corrcoef, precision_score, recall_score,
    confusion_matrix
)

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# Constants
# =========================================================

BLOSUM62 = {
    'A': [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0],
    'R': [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3],
    'N': [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3],
    'D': [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3],
    'C': [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    'Q': [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2],
    'E': [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2],
    'G': [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3],
    'H': [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3],
    'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3],
    'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1],
    'K': [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2],
    'M': [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1],
    'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1],
    'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2],
    'S': [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2],
    'T': [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0],
    'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3],
    'Y': [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1],
    'V': [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4],
}

ZERO_BLOSUM = np.zeros(20, dtype=np.float32)

# =========================================================
# Utilities
# =========================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def tensor_to_float_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()

    if isinstance(x, dict):
        for k in ["embedding", "embeddings", "feature", "features", "feat", "tensor", "data", "x"]:
            if k in x:
                return tensor_to_float_tensor(x[k])

    if torch.is_tensor(x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        while x.dim() > 2 and x.shape[0] == 1:
            x = x.squeeze(0)
        return x.float().cpu()

    raise TypeError(f"Unsupported type: {type(x)}")


def build_pt_index(feature_dir):
    feature_dir = Path(feature_dir)
    index = {p.stem: str(p) for p in feature_dir.rglob("*.pt")}
    print(f"[Index] {feature_dir}: {len(index)} .pt files")
    return index


def standardize_df(df):
    col_map = {}

    for col in df.columns:
        cl = col.lower()
        if "protein" in cl or "uniprot" in cl:
            col_map[col] = "protein"
        elif "position" in cl or "pos" in cl:
            col_map[col] = "position"
        elif cl in ["y", "label", "class"]:
            col_map[col] = "y"
        elif "sequence" in cl or "seq" in cl:
            col_map[col] = "sequence"

    if len(col_map) < 4:
        cols = df.columns.tolist()
        if len(cols) >= 5:
            col_map = {
                cols[0]: "protein",
                cols[1]: "position",
                cols[3]: "y",
                cols[4]: "sequence",
            }

    df = df.rename(columns=col_map)
    df["position"] = pd.to_numeric(df["position"], errors="coerce").fillna(-1).astype(int)
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(int)

    return df


def blosum_window(seq, center_0, window_size):
    half = window_size // 2
    rows = []

    for i in range(center_0 - half, center_0 + half + 1):
        if i < 0 or i >= len(seq):
            rows.append(ZERO_BLOSUM)
        else:
            rows.append(np.asarray(BLOSUM62.get(seq[i].upper(), ZERO_BLOSUM), dtype=np.float32))

    return torch.tensor(np.stack(rows, axis=0), dtype=torch.float32)


def slice_window(feat, center_0, window_size):
    feat = tensor_to_float_tensor(feat)
    L, D = feat.shape

    if center_0 < 0 or center_0 >= L:
        return None

    half = window_size // 2
    out = torch.zeros(window_size, D, dtype=torch.float32)

    start = center_0 - half
    end = center_0 + half + 1

    src_start = max(start, 0)
    src_end = min(end, L)

    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)

    out[dst_start:dst_end] = feat[src_start:src_end]

    return out


def compute_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    metrics = {}

    if len(np.unique(y_true)) == 2:
        metrics["AUC"] = float(roc_auc_score(y_true, y_score))
        metrics["AUPR"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["AUC"] = np.nan
        metrics["AUPR"] = np.nan

    metrics["ACC"] = float(accuracy_score(y_true, y_pred))
    metrics["F1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["MCC"] = float(matthews_corrcoef(y_true, y_pred))
    metrics["Precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["SN"] = float(recall_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["SP"] = float(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    return metrics


# =========================================================
# Dataset
# =========================================================

class EnhancedDataset(Dataset):
    def __init__(self, csv_path, structure_index, window_size=31):
        raw = pd.read_csv(csv_path, low_memory=False)

        self.df = standardize_df(raw)
        self.structure_index = structure_index or {}
        self.window_size = window_size
        self.samples = []
        self._load = lru_cache(maxsize=512)(self._load_uncached)

        for _, row in self.df.iterrows():
            pid = str(row.get("protein", ""))
            pos = int(row.get("position", -1))
            seq = str(row.get("sequence", ""))
            center = pos - 1

            if center < 0 or center >= len(seq):
                continue

            if pid not in self.structure_index:
                continue

            path = self.structure_index[pid]

            try:
                feat = self._load(path)
            except Exception:
                continue

            if center >= feat.shape[0]:
                continue

            self.samples.append({
                "protein": pid,
                "position": pos,
                "center_0": center,
                "sequence": seq,
                "y": int(row.get("y", 0)),
                "struct_path": path,
            })

    def _load_uncached(self, path):
        return tensor_to_float_tensor(torch.load(path, map_location="cpu"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]

        blosum = blosum_window(
            s["sequence"],
            s["center_0"],
            self.window_size,
        )

        struct_raw = self._load(s["struct_path"])

        struct_win = slice_window(
            struct_raw,
            s["center_0"],
            self.window_size,
        )

        x = torch.cat([blosum, struct_win], dim=-1).float()
        y = torch.tensor(float(s["y"]), dtype=torch.float32)

        meta = {
            "protein": s["protein"],
            "position": s["position"],
            "y": s["y"],
        }

        return x, y, meta


def collate_fn(batch):
    xs, ys, metas = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(metas)


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


# =========================================================
# Model
# =========================================================

class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()

        h = max(channels // reduction, 4)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, h),
            nn.ReLU(inplace=True),
            nn.Linear(h, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.shape
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1)
        return x * w


class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()

        h = max(dim // 2, 1)

        self.attn = nn.Sequential(
            nn.Linear(dim, h),
            nn.Tanh(),
            nn.Linear(h, 1),
        )

    def forward(self, x):
        w = torch.softmax(self.attn(x).squeeze(-1), dim=1)
        pooled = torch.sum(x * w.unsqueeze(-1), dim=1)
        return pooled, w


class CNNBiLSTMSE(nn.Module):
    def __init__(
        self,
        input_dim,
        conv_dim=256,
        rnn_hidden=256,
        rnn_layers=2,
        dropout=0.3,
        se_reduction=8,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(input_dim, conv_dim, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(conv_dim)

        self.conv2 = nn.Conv1d(conv_dim, conv_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(conv_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.se = SEBlock1D(conv_dim, reduction=se_reduction)

        self.rnn = nn.LSTM(
            input_size=conv_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )

        rnn_out = rnn_hidden * 2
        self.pool = AttentionPooling(rnn_out)

        self.classifier = nn.Sequential(
            nn.Linear(rnn_out, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),

            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)

        h = self.relu(self.bn1(self.conv1(x)))
        h = self.dropout(h)

        residual = h

        h = self.relu(self.bn2(self.conv2(h)))
        h = h + residual
        h = self.relu(h)

        h = self.se(h)
        h = self.dropout(h)

        h = h.transpose(1, 2)
        h, _ = self.rnn(h)

        pooled, _ = self.pool(h)

        return self.classifier(pooled).squeeze(-1)


# =========================================================
# Training
# =========================================================

def predict(model, loader, device):
    model.eval()
    all_y = []
    all_score = []
    rows = []

    with torch.no_grad():
        for x, y, metas in loader:
            x = x.to(device)
            logits = model(x)
            scores = torch.sigmoid(logits).cpu().numpy().tolist()
            labels = y.numpy().tolist()

            all_score.extend(scores)
            all_y.extend(labels)

            for meta, label, score in zip(metas, labels, scores):
                rows.append({
                    "protein": meta["protein"],
                    "position": meta["position"],
                    "y_true": int(label),
                    "score": float(score),
                    "y_pred": int(score >= 0.5),
                })

    return all_y, all_score, pd.DataFrame(rows)


def train_fold(train_csv, val_csv, struct_idx, args, device, fold_id, fold_dir):
    print(f"\n  {'=' * 50}")
    print(f"  Baseline | Train: {Path(train_csv).name} | Val: {Path(val_csv).name}")
    print(f"  {'=' * 50}")

    train_ds = EnhancedDataset(
        csv_path=train_csv,
        structure_index=struct_idx,
        window_size=args.window_size,
    )

    val_ds = EnhancedDataset(
        csv_path=val_csv,
        structure_index=struct_idx,
        window_size=args.window_size,
    )

    if len(train_ds) == 0:
        raise RuntimeError(f"Train dataset is empty: {train_csv}")

    if len(val_ds) == 0:
        raise RuntimeError(f"Val dataset is empty: {val_csv}")

    x, _, _ = train_ds[0]
    input_dim = x.shape[-1]

    print(f"    Input dim: {input_dim} | Train: {len(train_ds)} | Val: {len(val_ds)}")

    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = fold_dir / "best_model.pt"

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = CNNBiLSTMSE(
        input_dim=input_dim,
        rnn_layers=args.rnn_layers,
        dropout=args.dropout,
        se_reduction=args.se_reduction,
    ).to(device)

    train_labels = [s["y"] for s in train_ds.samples]
    pos = sum(train_labels)
    neg = len(train_labels) - pos
    pos_weight = neg / max(pos, 1)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6,
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    best_auc = -math.inf
    patience = 0
    best_metrics = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)

            # Mixup data augmentation
            if torch.rand(1).item() < 0.5:
                idx = torch.randperm(x.size(0), device=device)
                lam_raw = np.random.beta(0.2, 0.2)
                lam = max(lam_raw, 1.0 - lam_raw)

                x = lam * x + (1.0 - lam) * x[idx]
                y = lam * y + (1.0 - lam) * y[idx]

            optimizer.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = criterion(logits, y)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()

            else:
                logits = model(x)
                loss = criterion(logits, y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        all_y, all_score, val_pred_df = predict(model, val_loader, device)
        val_metrics = compute_metrics(all_y, all_score)
        history.append({"epoch": epoch, **val_metrics})

        scheduler.step()

        if val_metrics["AUC"] > best_auc:
            best_auc = val_metrics["AUC"]
            best_metrics = val_metrics
            patience = 0
            torch.save(
                {
                    "fold": fold_id,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "args": vars(args),
                    "metrics": best_metrics,
                },
                best_model_path,
            )
            val_pred_df.to_csv(fold_dir / "best_val_predictions.csv", index=False)
        else:
            patience += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"    Epoch {epoch:3d}: "
                f"AUC={val_metrics['AUC']:.4f}  "
                f"MCC={val_metrics['MCC']:.4f}  "
                f"ACC={val_metrics['ACC']:.4f}  "
                f"F1={val_metrics['F1']:.4f}"
            )

        if patience >= args.patience:
            print(f"    Early stopping at epoch {epoch}")
            break

    pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)

    del model, train_loader, val_loader
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "input_dim": input_dim,
        "best_metrics": best_metrics,
        "model_path": str(best_model_path),
    }


def independent_test(test_csv, struct_idx, fold_results, args, device, output_dir):
    test_ds = EnhancedDataset(
        csv_path=test_csv,
        structure_index=struct_idx,
        window_size=args.window_size,
    )
    if len(test_ds) == 0:
        raise RuntimeError(f"Test dataset is empty: {test_csv}")

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    fold_scores = []
    base_rows = None
    fold_metrics = []

    for result in fold_results:
        ckpt = safe_torch_load(result["model_path"], map_location=device)
        model = CNNBiLSTMSE(
            input_dim=int(ckpt["input_dim"]),
            rnn_layers=args.rnn_layers,
            dropout=args.dropout,
            se_reduction=args.se_reduction,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        y_true, y_score, pred_df = predict(model, test_loader, device)
        fold_scores.append(np.asarray(y_score, dtype=float))
        fold_metrics.append({"fold": int(ckpt["fold"]), **compute_metrics(y_true, y_score)})

        if base_rows is None:
            base_rows = pred_df[["protein", "position", "y_true"]].copy()

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_score = np.mean(np.stack(fold_scores, axis=0), axis=0)
    test_metrics = compute_metrics(base_rows["y_true"].values, mean_score)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_df = base_rows.copy()
    for idx, scores in enumerate(fold_scores, start=1):
        out_df[f"fold_{idx}_score"] = scores
    out_df["score"] = mean_score
    out_df["y_pred"] = (mean_score >= 0.5).astype(int)
    out_df.to_csv(output_dir / "test_predictions.csv", index=False)

    summary = {
        "test_csv": str(test_csv),
        "n_samples": int(len(test_ds)),
        "ensemble_metrics": test_metrics,
        "fold_metrics": fold_metrics,
    }
    with open(output_dir / "test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# =========================================================
# Main
# =========================================================

def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cv-dir",
        default="/data/cv_folds",
    )

    parser.add_argument(
        "--structure-dir",
        default="/structure_embedding",
    )

    parser.add_argument(
        "--output-dir",
        default="/",
    )

    parser.add_argument(
        "--test-csv",
        default="/data/test_20.csv",
    )
    parser.add_argument("--skip-independent-test", action="store_true")

    parser.add_argument("--window-size", type=int, default=31)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--rnn-layers", type=int, default=2)
    parser.add_argument("--se-reduction", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")

    args = parser.parse_args()

    set_seed(42)

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BASELINE BLOSUM + STRUCTURE - 5-FOLD CV")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"lr: {args.lr}")
    print(f"dropout: {args.dropout}")
    print(f"window_size: {args.window_size}")
    print(f"rnn_layers: {args.rnn_layers}")
    print(f"se_reduction: {args.se_reduction}")
    print(f"batch_size: {args.batch_size}")
    print(f"weight_decay: {args.weight_decay}")

    struct_idx = build_pt_index(args.structure_dir)
    cv_dir = Path(args.cv_dir)

    folds = []

    for train_path in sorted(cv_dir.glob("fold_*_train.csv")):
        try:
            fold_id = int(train_path.name.split("_")[1])
        except Exception:
            continue

        val_path = cv_dir / f"fold_{fold_id}_val.csv"

        if val_path.exists():
            folds.append((fold_id, str(train_path), str(val_path)))

    folds = sorted(folds, key=lambda x: x[0])

    print(f"Folds: {[f[0] for f in folds]}")

    if len(folds) == 0:
        raise RuntimeError(f"No folds found in: {cv_dir}")

    all_results = {
        "Baseline": [],
    }

    for fold_id, train_csv, val_csv in folds:
        print(f"\n{'=' * 70}")
        print(f"FOLD {fold_id}")
        print(f"{'=' * 70}")

        result = train_fold(
            train_csv=train_csv,
            val_csv=val_csv,
            struct_idx=struct_idx,
            args=args,
            device=device,
            fold_id=fold_id,
            fold_dir=output_dir / f"fold_{fold_id}",
        )

        result["fold"] = fold_id
        all_results["Baseline"].append(result)

    print(f"\n{'=' * 90}")
    print("5-FOLD CV SUMMARY")
    print(f"{'=' * 90}")

    metric_names = [
        "ACC",
        "AUC",
        "AUPR",
        "MCC",
        "F1",
        "SN",
        "SP",
        "Precision",
    ]

    for method_name in ["Baseline"]:
        results = all_results[method_name]

        if not results:
            continue

        print(f"\n  {method_name}:")

        for r in results:
            m = r["best_metrics"]
            print(
                f"    Fold {r['fold']}: "
                f"AUC={m['AUC']:.4f}  "
                f"MCC={m['MCC']:.4f}  "
                f"ACC={m['ACC']:.4f}  "
                f"F1={m['F1']:.4f}"
            )

        for mn in metric_names:
            vals = [r["best_metrics"][mn] for r in results]
            print(f"    {mn}: {np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}")

    summary = {}

    for method_name in ["Baseline"]:
        results = all_results[method_name]

        if not results:
            continue

        summary[method_name] = {
            "folds": [
                {
                    **r["best_metrics"],
                    "fold": r["fold"],
                    "input_dim": r["input_dim"],
                    "model_path": r["model_path"],
                }
                for r in results
            ],
            "mean": {
                mn: float(np.mean([r["best_metrics"][mn] for r in results]))
                for mn in metric_names
            },
            "std": {
                mn: float(np.std([r["best_metrics"][mn] for r in results], ddof=1))
                for mn in metric_names
            },
        }

    with open(output_dir / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if not args.skip_independent_test:
        print(f"\n{'=' * 90}")
        print("INDEPENDENT TEST")
        print(f"{'=' * 90}")
        test_summary = independent_test(
            test_csv=args.test_csv,
            struct_idx=struct_idx,
            fold_results=all_results["Baseline"],
            args=args,
            device=device,
            output_dir=output_dir / "independent_test",
        )
        m = test_summary["ensemble_metrics"]
        print(
            "    Ensemble: "
            f"AUC={m['AUC']:.4f}  "
            f"AUPR={m['AUPR']:.4f}  "
            f"MCC={m['MCC']:.4f}  "
            f"ACC={m['ACC']:.4f}  "
            f"F1={m['F1']:.4f}"
        )

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
