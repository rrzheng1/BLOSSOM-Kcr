#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build balanced Crotonylation/Kcr site prediction dataset.

Pipeline:
1. Read positive Kcr sites with full-length sequences.
2. Validate that Position is K.
3. Run CD-HIT at 40% identity on protein sequences.
4. Keep CD-HIT representative proteins.
5. Check/download AlphaFold PDB files.
6. Keep proteins with available PDB files.
7. Generate negative samples from unannotated K residues in the same proteins.
8. Keep positive:negative = 1:1.
9. Split proteins into train/test = 8:2.
10. Build 5-fold CV from training proteins.

Input:
    /data/ranran/my_ptm/croton/data/kcr_positive_sites_with_sequence.validK.csv

Output:
    /data/ranran/my_ptm/croton/data/final_dataset_cdhit40_pdb/
        all_positive_filtered.csv
        all_balanced.csv
        train_80.csv
        test_20.csv
        cv_folds/
            fold_1_train.csv
            fold_1_val.csv
            ...
            fold_5_train.csv
            fold_5_val.csv
        proteins.fasta
        proteins_cdhit40.fasta
        proteins_cdhit40.clstr
        stats.csv
        failed_pdb_downloads.csv
        dataset_build_summary.txt
"""

from __future__ import annotations

import argparse
import random
import subprocess
import time
from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import KFold, train_test_split


OUTPUT_COLUMNS = ["protein", "Position", "Residue", "y", "sequence"]


# =========================================================
# Args
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CD-HIT40 + PDB-filtered balanced Kcr dataset with 8:2 split and 5-fold CV."
    )

    parser.add_argument(
        "--input",
        default="/data/ranran/my_ptm/croton/data/kcr_positive_sites_with_sequence.validK.csv",
        help="Input positive Kcr sites with full-length sequences.",
    )
    parser.add_argument(
        "--outdir",
        default="/data/ranran/my_ptm/croton/data/final_dataset_cdhit40_pdb",
        help="Output directory.",
    )

    parser.add_argument(
        "--cdhit",
        default="cd-hit",
        help="Path to cd-hit executable. Example: cd-hit",
    )
    parser.add_argument(
        "--identity",
        type=float,
        default=0.40,
        help="CD-HIT sequence identity threshold. Default: 0.40",
    )

    parser.add_argument(
        "--pdb-dir",
        default="/data/ranran/my_ptm/croton/structure_pipeline/pdb",
        help="Directory for local/downloaded PDB files.",
    )
    parser.add_argument(
        "--download-af",
        action="store_true",
        help="Try to download AlphaFold PDB files if local PDB is missing.",
    )
    parser.add_argument(
        "--af-version",
        default="v4",
        choices=["v4", "v6"],
        help="AlphaFold DB model version. Use v4 or v6.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Independent test protein ratio. Default: 0.2",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds from training proteins. Default: 5",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )

    return parser.parse_args()


# =========================================================
# Basic IO
# =========================================================
def ensure_write_allowed(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_positive_sites(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input missing columns: {missing}")

    df = df[OUTPUT_COLUMNS].copy()
    df["protein"] = df["protein"].astype(str).str.strip()
    df["Position"] = df["Position"].astype(int)
    df["Residue"] = "K"
    df["y"] = 1
    df["sequence"] = df["sequence"].astype(str).str.strip()

    df = df.drop_duplicates(subset=["protein", "Position"]).copy()

    # Check center K
    bad = []
    for i, row in df.iterrows():
        seq = row["sequence"]
        pos = int(row["Position"])
        if pos < 1 or pos > len(seq):
            bad.append(i)
        elif seq[pos - 1] != "K":
            bad.append(i)

    if bad:
        bad_df = df.loc[bad].copy()
        raise ValueError(
            f"{path}: {len(bad_df)} rows do not have K at Position. "
            f"Please use validK.csv or fix these rows first."
        )

    return df.reset_index(drop=True)


def write_fasta_from_df(df: pd.DataFrame, fasta_path: Path, overwrite: bool) -> None:
    ensure_write_allowed(fasta_path, overwrite)

    seqs = (
        df.drop_duplicates("protein")
        .set_index("protein")["sequence"]
        .to_dict()
    )

    with fasta_path.open("w", encoding="utf-8") as f:
        for protein, seq in sorted(seqs.items()):
            f.write(f">{protein}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i + 60] + "\n")


# =========================================================
# CD-HIT
# =========================================================
def run_cdhit(
    cdhit_exe: str,
    input_fasta: Path,
    output_fasta: Path,
    identity: float,
    overwrite: bool,
) -> None:
    ensure_write_allowed(output_fasta, overwrite)

    # CD-HIT word size recommendation:
    # for 0.4 identity, -n 2 is appropriate.
    if identity < 0.5:
        word_size = 2
    elif identity < 0.6:
        word_size = 3
    elif identity < 0.7:
        word_size = 4
    else:
        word_size = 5

    cmd = [
        cdhit_exe,
        "-i", str(input_fasta),
        "-o", str(output_fasta),
        "-c", str(identity),
        "-n", str(word_size),
        "-d", "0",
        "-M", "0",
        "-T", "8",
    ]

    print("\nRunning CD-HIT:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)


def parse_cdhit_representatives(cdhit_fasta: Path) -> set[str]:
    reps = set()
    with cdhit_fasta.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                protein = line[1:].strip().split()[0]
                reps.add(protein)
    return reps


# =========================================================
# PDB / AlphaFold
# =========================================================
def possible_local_pdb_paths(pdb_dir: Path, protein: str) -> list[Path]:
    return [
        pdb_dir / f"{protein}.pdb",
        pdb_dir / f"{protein}.ent",
        pdb_dir / f"AF-{protein}-F1-model_v4.pdb",
        pdb_dir / f"AF-{protein}-F1-model_v6.pdb",
    ]


def has_local_pdb(pdb_dir: Path, protein: str) -> tuple[bool, str]:
    for p in possible_local_pdb_paths(pdb_dir, protein):
        if p.exists() and p.stat().st_size > 0:
            return True, str(p)
    return False, ""


def alphafold_url(protein: str, version: str) -> str:
    return f"https://alphafold.ebi.ac.uk/files/AF-{protein}-F1-model_{version}.pdb"


def download_alphafold_pdb(
    protein: str,
    pdb_dir: Path,
    version: str = "v4",
    max_retry: int = 3,
    sleep_sec: float = 0.5,
) -> tuple[bool, str, str]:
    """
    Returns:
        success, pdb_path, error
    """
    safe_mkdir(pdb_dir)

    out_path = pdb_dir / f"AF-{protein}-F1-model_{version}.pdb"

    if out_path.exists() and out_path.stat().st_size > 0:
        return True, str(out_path), ""

    url = alphafold_url(protein, version)

    last_err = ""
    for attempt in range(1, max_retry + 1):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and r.text.startswith("HEADER"):
                out_path.write_text(r.text, encoding="utf-8")
                return True, str(out_path), ""

            if r.status_code == 404:
                return False, "", "not_found_404"

            last_err = f"http_{r.status_code}"

        except Exception as e:
            last_err = repr(e)

        if attempt < max_retry:
            time.sleep(sleep_sec * attempt)

    return False, "", last_err


def filter_proteins_with_pdb(
    proteins: list[str],
    pdb_dir: Path,
    download_af: bool,
    af_version: str,
) -> tuple[set[str], pd.DataFrame]:
    valid = set()
    failed_records = []

    safe_mkdir(pdb_dir)

    for idx, protein in enumerate(proteins, start=1):
        print(f"[PDB {idx}/{len(proteins)}] checking {protein}")

        ok, local_path = has_local_pdb(pdb_dir, protein)
        if ok:
            valid.add(protein)
            continue

        if download_af:
            ok, pdb_path, err = download_alphafold_pdb(
                protein=protein,
                pdb_dir=pdb_dir,
                version=af_version,
            )
            if ok:
                valid.add(protein)
            else:
                failed_records.append({
                    "protein": protein,
                    "reason": err,
                })
        else:
            failed_records.append({
                "protein": protein,
                "reason": "no_local_pdb",
            })

    failed_df = pd.DataFrame(failed_records)
    return valid, failed_df


# =========================================================
# Negative sampling
# =========================================================
def generate_negative_candidates(pos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate all negative candidates from unannotated K residues
    in the same proteins.

    Negative definition:
        same protein
        residue == K
        protein-position not in positive Kcr sites
    """
    used_positive_sites = set(zip(pos_df["protein"], pos_df["Position"]))

    seqs = (
        pos_df.drop_duplicates("protein")
        .set_index("protein")["sequence"]
        .to_dict()
    )

    records = []
    for protein, seq in seqs.items():
        for idx, aa in enumerate(seq, start=1):
            if aa != "K":
                continue
            if (protein, idx) in used_positive_sites:
                continue

            records.append({
                "protein": protein,
                "Position": idx,
                "Residue": "K",
                "y": 0,
                "sequence": seq,
            })

    neg_df = pd.DataFrame(records, columns=OUTPUT_COLUMNS)

    if neg_df.empty:
        raise RuntimeError("No negative candidates found.")

    return neg_df


def sample_balanced_by_protein(
    pos_df: pd.DataFrame,
    neg_candidates: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """
    Sample negatives globally with pos:neg = 1:1.
    Since train/test splitting is protein-level and all samples from a protein stay
    in the same split, we will rebalance each split separately later.
    """
    pos_n = len(pos_df)

    if len(neg_candidates) < pos_n:
        raise ValueError(
            f"Not enough negative candidates. Need {pos_n}, found {len(neg_candidates)}."
        )

    neg_sampled = neg_candidates.sample(n=pos_n, random_state=seed).copy()

    out = pd.concat([pos_df[OUTPUT_COLUMNS], neg_sampled[OUTPUT_COLUMNS]], ignore_index=True)
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)

    return out


def balance_split_from_proteins(
    pos_df: pd.DataFrame,
    neg_candidates: pd.DataFrame,
    proteins: set[str],
    seed: int,
) -> pd.DataFrame:
    """
    Build one split and keep pos:neg = 1:1 within this split.
    """
    pos = pos_df[pos_df["protein"].isin(proteins)].copy()
    neg_pool = neg_candidates[neg_candidates["protein"].isin(proteins)].copy()

    pos_n = len(pos)

    if pos_n == 0:
        raise ValueError("This split has zero positives.")

    if len(neg_pool) < pos_n:
        raise ValueError(
            f"Not enough negatives in split. positives={pos_n}, negative_candidates={len(neg_pool)}"
        )

    neg = neg_pool.sample(n=pos_n, random_state=seed).copy()

    out = pd.concat([pos[OUTPUT_COLUMNS], neg[OUTPUT_COLUMNS]], ignore_index=True)
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)

    return out


# =========================================================
# Split
# =========================================================
def protein_level_train_test_split(
    pos_df: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    """
    Split by proteins, not by sites.
    Stratification is approximated by splitting protein IDs only.
    """
    proteins = sorted(pos_df["protein"].unique())

    train_proteins, test_proteins = train_test_split(
        proteins,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    return set(train_proteins), set(test_proteins)


def write_5fold_cv(
    train_proteins: set[str],
    pos_df: pd.DataFrame,
    neg_candidates: pd.DataFrame,
    cv_dir: Path,
    n_folds: int,
    seed: int,
    overwrite: bool,
) -> list[dict]:
    safe_mkdir(cv_dir)

    proteins = sorted(train_proteins)

    if len(proteins) < n_folds:
        raise ValueError(f"Training proteins {len(proteins)} < n_folds {n_folds}")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    stats = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(proteins), start=1):
        fold_train_proteins = {proteins[i] for i in tr_idx}
        fold_val_proteins = {proteins[i] for i in val_idx}

        overlap = fold_train_proteins & fold_val_proteins
        if overlap:
            raise RuntimeError(f"Fold {fold} protein leakage: {len(overlap)}")

        fold_train = balance_split_from_proteins(
            pos_df=pos_df,
            neg_candidates=neg_candidates,
            proteins=fold_train_proteins,
            seed=seed + fold * 10 + 1,
        )
        fold_val = balance_split_from_proteins(
            pos_df=pos_df,
            neg_candidates=neg_candidates,
            proteins=fold_val_proteins,
            seed=seed + fold * 10 + 2,
        )

        train_path = cv_dir / f"fold_{fold}_train.csv"
        val_path = cv_dir / f"fold_{fold}_val.csv"

        ensure_write_allowed(train_path, overwrite)
        ensure_write_allowed(val_path, overwrite)

        fold_train[OUTPUT_COLUMNS].to_csv(train_path, index=False)
        fold_val[OUTPUT_COLUMNS].to_csv(val_path, index=False)

        stats.append(split_stats(f"fold_{fold}_train", fold_train))
        stats.append(split_stats(f"fold_{fold}_val", fold_val))

    return stats


# =========================================================
# Validation / stats
# =========================================================
def check_dataset(df: pd.DataFrame, name: str) -> None:
    missing = [c for c in OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

    duplicated = df.duplicated(subset=["protein", "Position"]).sum()
    if duplicated:
        raise ValueError(f"{name}: duplicated protein-position rows: {duplicated}")

    counts = df["y"].value_counts().to_dict()
    pos_n = int(counts.get(1, 0))
    neg_n = int(counts.get(0, 0))

    if pos_n != neg_n:
        raise ValueError(f"{name} is not balanced: positives={pos_n}, negatives={neg_n}")

    bad = []
    for i, row in df.iterrows():
        seq = row["sequence"]
        pos = int(row["Position"])
        if pos < 1 or pos > len(seq) or seq[pos - 1] != "K":
            bad.append(i)

    if bad:
        raise ValueError(f"{name}: {len(bad)} rows do not have K at Position")


def split_stats(name: str, df: pd.DataFrame) -> dict:
    labels = df["y"].value_counts().to_dict()
    return {
        "split": name,
        "rows": len(df),
        "proteins": df["protein"].nunique(),
        "positive": int(labels.get(1, 0)),
        "negative": int(labels.get(0, 0)),
    }


def check_protein_leakage(a: pd.DataFrame, b: pd.DataFrame, a_name: str, b_name: str) -> None:
    overlap = set(a["protein"]) & set(b["protein"])
    if overlap:
        examples = sorted(list(overlap))[:10]
        raise ValueError(
            f"Protein leakage between {a_name} and {b_name}: "
            f"{len(overlap)} proteins. Examples: {examples}"
        )


# =========================================================
# Main
# =========================================================
def main() -> None:
    args = parse_args()

    input_csv = Path(args.input)
    outdir = Path(args.outdir)
    pdb_dir = Path(args.pdb_dir)

    safe_mkdir(outdir)
    safe_mkdir(pdb_dir)

    fasta_path = outdir / "proteins.fasta"
    cdhit_fasta = outdir / "proteins_cdhit40.fasta"
    cdhit_clstr = outdir / "proteins_cdhit40.fasta.clstr"

    out_positive_filtered = outdir / "all_positive_filtered.csv"
    out_neg_candidates = outdir / "all_negative_candidates.csv"
    out_balanced = outdir / "all_balanced.csv"
    out_train = outdir / "train_80.csv"
    out_test = outdir / "test_20.csv"
    out_failed_pdb = outdir / "failed_pdb_downloads.csv"
    out_stats = outdir / "stats.csv"
    out_summary = outdir / "dataset_build_summary.txt"
    cv_dir = outdir / "cv_folds"

    # -----------------------------------------------------
    # 1. Read positives
    # -----------------------------------------------------
    print("\n[1] Reading positive Kcr sites")
    pos_raw = read_positive_sites(input_csv)

    print(f"Raw positive sites: {len(pos_raw)}")
    print(f"Raw positive proteins: {pos_raw['protein'].nunique()}")

    # -----------------------------------------------------
    # 2. Write FASTA
    # -----------------------------------------------------
    print("\n[2] Writing protein FASTA for CD-HIT")
    write_fasta_from_df(pos_raw, fasta_path, overwrite=args.overwrite)

    # -----------------------------------------------------
    # 3. Run CD-HIT
    # -----------------------------------------------------
    print("\n[3] Running CD-HIT")
    run_cdhit(
        cdhit_exe=args.cdhit,
        input_fasta=fasta_path,
        output_fasta=cdhit_fasta,
        identity=args.identity,
        overwrite=args.overwrite,
    )

    rep_proteins = parse_cdhit_representatives(cdhit_fasta)
    print(f"CD-HIT representative proteins: {len(rep_proteins)}")

    pos_cdhit = pos_raw[pos_raw["protein"].isin(rep_proteins)].copy()
    print(f"Positive sites after CD-HIT protein filtering: {len(pos_cdhit)}")
    print(f"Proteins after CD-HIT protein filtering: {pos_cdhit['protein'].nunique()}")

    # -----------------------------------------------------
    # 4. PDB / AlphaFold filtering
    # -----------------------------------------------------
    print("\n[4] Checking PDB / AlphaFold structure availability")
    proteins_after_cdhit = sorted(pos_cdhit["protein"].unique())

    pdb_valid_proteins, failed_pdb = filter_proteins_with_pdb(
        proteins=proteins_after_cdhit,
        pdb_dir=pdb_dir,
        download_af=args.download_af,
        af_version=args.af_version,
    )

    failed_pdb.to_csv(out_failed_pdb, index=False)
    print(f"Proteins with PDB/AlphaFold structure: {len(pdb_valid_proteins)}")
    print(f"Proteins without PDB/AlphaFold structure: {len(failed_pdb)}")

    pos_filtered = pos_cdhit[pos_cdhit["protein"].isin(pdb_valid_proteins)].copy()
    pos_filtered = pos_filtered.drop_duplicates(subset=["protein", "Position"]).copy()
    pos_filtered = pos_filtered[OUTPUT_COLUMNS].copy()

    ensure_write_allowed(out_positive_filtered, args.overwrite)
    pos_filtered.to_csv(out_positive_filtered, index=False)

    print(f"Final positive sites after CD-HIT + PDB filtering: {len(pos_filtered)}")
    print(f"Final proteins after CD-HIT + PDB filtering: {pos_filtered['protein'].nunique()}")

    if pos_filtered.empty:
        raise RuntimeError("No positive sites remain after CD-HIT + PDB filtering.")

    # -----------------------------------------------------
    # 5. Generate negatives
    # -----------------------------------------------------
    print("\n[5] Generating negative candidates from unannotated K sites")
    neg_candidates = generate_negative_candidates(pos_filtered)

    ensure_write_allowed(out_neg_candidates, args.overwrite)
    neg_candidates.to_csv(out_neg_candidates, index=False)

    print(f"Negative candidates: {len(neg_candidates)}")
    print(f"Negative candidate proteins: {neg_candidates['protein'].nunique()}")

    # -----------------------------------------------------
    # 6. Global balanced dataset
    # -----------------------------------------------------
    print("\n[6] Building global 1:1 balanced dataset")
    all_balanced = sample_balanced_by_protein(
        pos_df=pos_filtered,
        neg_candidates=neg_candidates,
        seed=args.seed,
    )

    check_dataset(all_balanced, "all_balanced")

    ensure_write_allowed(out_balanced, args.overwrite)
    all_balanced[OUTPUT_COLUMNS].to_csv(out_balanced, index=False)

    # -----------------------------------------------------
    # 7. Protein-level train/test split
    # -----------------------------------------------------
    print("\n[7] Protein-level train/test split = 8:2")
    train_proteins, test_proteins = protein_level_train_test_split(
        pos_df=pos_filtered,
        test_size=args.test_size,
        seed=args.seed,
    )

    train_df = balance_split_from_proteins(
        pos_df=pos_filtered,
        neg_candidates=neg_candidates,
        proteins=train_proteins,
        seed=args.seed + 1,
    )
    test_df = balance_split_from_proteins(
        pos_df=pos_filtered,
        neg_candidates=neg_candidates,
        proteins=test_proteins,
        seed=args.seed + 2,
    )

    check_dataset(train_df, "train_80")
    check_dataset(test_df, "test_20")
    check_protein_leakage(train_df, test_df, "train_80", "test_20")

    ensure_write_allowed(out_train, args.overwrite)
    ensure_write_allowed(out_test, args.overwrite)

    train_df[OUTPUT_COLUMNS].to_csv(out_train, index=False)
    test_df[OUTPUT_COLUMNS].to_csv(out_test, index=False)

    print(f"Train rows: {len(train_df)}, proteins: {train_df['protein'].nunique()}")
    print(f"Test rows: {len(test_df)}, proteins: {test_df['protein'].nunique()}")

    # -----------------------------------------------------
    # 8. 5-fold CV from training proteins
    # -----------------------------------------------------
    print("\n[8] Building 5-fold CV from training proteins")
    cv_stats = write_5fold_cv(
        train_proteins=train_proteins,
        pos_df=pos_filtered,
        neg_candidates=neg_candidates,
        cv_dir=cv_dir,
        n_folds=args.n_folds,
        seed=args.seed,
        overwrite=args.overwrite,
    )

    # -----------------------------------------------------
    # 9. Save stats
    # -----------------------------------------------------
    stats = [
        split_stats("all_positive_filtered", pos_filtered.assign(y=1)),
        split_stats("all_balanced", all_balanced),
        split_stats("train_80", train_df),
        split_stats("test_20", test_df),
    ]
    stats.extend(cv_stats)

    stats_df = pd.DataFrame(stats)

    ensure_write_allowed(out_stats, args.overwrite)
    stats_df.to_csv(out_stats, index=False)

    # -----------------------------------------------------
    # 10. Summary
    # -----------------------------------------------------
    summary = []
    summary.append(f"Input: {input_csv}")
    summary.append(f"Output dir: {outdir}")
    summary.append("")
    summary.append("Pipeline:")
    summary.append(f"1. CD-HIT identity: {args.identity}")
    summary.append(f"2. PDB dir: {pdb_dir}")
    summary.append(f"3. Download AlphaFold: {args.download_af}")
    summary.append(f"4. AlphaFold version: {args.af_version}")
    summary.append(f"5. Train/test split: {1 - args.test_size:.1f}:{args.test_size:.1f}")
    summary.append(f"6. CV folds: {args.n_folds}")
    summary.append("")
    summary.append("Counts:")
    summary.append(f"Raw positive sites: {len(pos_raw)}")
    summary.append(f"Raw positive proteins: {pos_raw['protein'].nunique()}")
    summary.append(f"CD-HIT representative proteins: {len(rep_proteins)}")
    summary.append(f"Positive sites after CD-HIT: {len(pos_cdhit)}")
    summary.append(f"Proteins with PDB/AlphaFold: {len(pdb_valid_proteins)}")
    summary.append(f"Final positive sites: {len(pos_filtered)}")
    summary.append(f"Final positive proteins: {pos_filtered['protein'].nunique()}")
    summary.append(f"Negative candidates: {len(neg_candidates)}")
    summary.append(f"All balanced rows: {len(all_balanced)}")
    summary.append(f"Train rows: {len(train_df)}")
    summary.append(f"Test rows: {len(test_df)}")
    summary.append("")
    summary.append("Output files:")
    summary.append(f"  {out_positive_filtered}")
    summary.append(f"  {out_neg_candidates}")
    summary.append(f"  {out_balanced}")
    summary.append(f"  {out_train}")
    summary.append(f"  {out_test}")
    summary.append(f"  {cv_dir}")
    summary.append(f"  {out_stats}")
    summary.append(f"  {out_failed_pdb}")

    out_summary.write_text("\n".join(summary), encoding="utf-8")

    print("\nDone.")
    print(f"Stats saved to: {out_stats}")
    print(f"Summary saved to: {out_summary}")
    print("\nStats:")
    print(stats_df)


if __name__ == "__main__":
    main()