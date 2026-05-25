#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch convert AlphaFold PDB files to DSSP files for crotonylation project.

功能：
1. 从 PDB 目录读取 .pdb 文件
2. 自动排除 AF-*-F1-model_v6.pdb 原始文件
3. 只处理简短蛋白名 PDB，例如 Q9Y6K9.pdb
4. 调用 mkdssp 生成对应 .dssp 文件
5. 记录失败日志 jsonl

推荐输入目录：
/data/ranran/my_ptm/croton/structure_pipeline/pdb

推荐输出目录：
/data/ranran/my_ptm/croton/structure_pipeline/dssp

注意：
请在 dssp_env 环境中运行，确保 mkdssp 可用。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def check_mkdssp() -> None:
    """Check whether mkdssp is available in current environment."""
    result = subprocess.run(
        ["mkdssp", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "mkdssp is not available in the current environment.\n"
            "Please run this script inside dssp_env, for example:\n"
            "conda run -n dssp_env python pdb_to_dssp_croton.py ...\n\n"
            f"STDOUT:\n{result.stdout.strip()}\n\n"
            f"STDERR:\n{result.stderr.strip()}"
        )

    version_text = result.stdout.strip() or result.stderr.strip()
    print(f"mkdssp check passed: {version_text}")


def is_valid_pdb(path: Path, min_size: int = 1024) -> bool:
    """Basic validation for PDB file."""
    if not path.exists() or path.stat().st_size < min_size:
        return False

    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return False

    return "ATOM" in text and ("END" in text or "CONECT" in text)


def load_target_proteins(input_csvs: list[str] | None) -> set[str] | None:
    """
    Load protein IDs from train/test CSV files.

    If input_csvs is None or empty, return None, meaning process all valid PDBs.
    """
    if not input_csvs:
        return None

    proteins: set[str] = set()

    for csv_path in input_csvs:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Input CSV not found: {path}")

        df = pd.read_csv(path, dtype=str)

        if "protein" not in df.columns:
            raise ValueError(f"{path} must contain a 'protein' column")

        proteins.update(
            df["protein"]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda s: (s != "") & (s.str.lower() != "nan")]
            .tolist()
        )

    return proteins


def collect_pdb_files(pdb_dir: Path, target_proteins: set[str] | None = None) -> list[Path]:
    """
    Collect PDB files to process.

    Important:
    - Exclude original AlphaFold names: AF-*-F1-model_v6.pdb
    - Only process short protein ID PDB files: Qxxxx.pdb
    - These can be real files or symlinks.
    """
    all_pdbs = sorted(pdb_dir.glob("*.pdb"))

    pdb_files: list[Path] = []

    for pdb_path in all_pdbs:
        name = pdb_path.name

        # 排除原始 AlphaFold 文件，避免重复处理
        if name.startswith("AF-") and "-F1-model_" in name:
            continue

        protein_id = pdb_path.stem

        if target_proteins is not None and protein_id not in target_proteins:
            continue

        pdb_files.append(pdb_path)

    return pdb_files


def run_mkdssp_one(
    pdb_path: Path,
    dssp_dir: Path,
    overwrite: bool = False,
) -> tuple[str, bool, str]:
    """
    Convert one PDB to DSSP.

    Returns:
        protein_id, success, reason
    """
    protein_id = pdb_path.stem
    dssp_path = dssp_dir / f"{protein_id}.dssp"

    if not is_valid_pdb(pdb_path):
        return protein_id, False, "missing_or_damaged_pdb"

    if not overwrite and dssp_path.exists() and dssp_path.stat().st_size > 0:
        return protein_id, True, "exists"

    dssp_dir.mkdir(parents=True, exist_ok=True)

    # 使用临时文件，避免中断时留下坏的 .dssp
    with tempfile.NamedTemporaryFile(
        prefix=f"{protein_id}.",
        suffix=".dssp.tmp",
        dir=str(dssp_dir),
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)

    result = subprocess.run(
        ["mkdssp", "-i", str(pdb_path), "-o", str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        reason = result.stderr.strip() or result.stdout.strip() or "mkdssp_failed"
        return protein_id, False, reason[-1000:]

    if not tmp_path.exists() or tmp_path.stat().st_size == 0:
        tmp_path.unlink(missing_ok=True)
        return protein_id, False, "empty_dssp"

    tmp_path.replace(dssp_path)

    return protein_id, True, "generated"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert crotonylation AlphaFold PDB files to DSSP files."
    )

    parser.add_argument(
        "--pdb-dir",
        default="/data/ranran/my_ptm/croton/structure_pipeline/pdb",
        help="Directory containing PDB files.",
    )

    parser.add_argument(
        "--dssp-dir",
        default="/data/ranran/my_ptm/croton/structure_pipeline/dssp",
        help="Output directory for DSSP files.",
    )

    parser.add_argument(
        "--input-csv",
        nargs="*",
        default=[
            "/data/ranran/my_ptm/croton/data_clean/train_80.csv",
            "/data/ranran/my_ptm/croton/data_clean/test_20.csv",
        ],
        help=(
            "CSV files containing a 'protein' column. "
            "Only proteins in these CSVs will be processed. "
            "Use --process-all-pdbs to ignore this filter."
        ),
    )

    parser.add_argument(
        "--process-all-pdbs",
        action="store_true",
        help="Process all short-name PDB files in pdb-dir, ignoring input CSVs.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel mkdssp workers.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing DSSP files.",
    )

    parser.add_argument(
        "--fail-log",
        default="/data/ranran/my_ptm/croton/structure_pipeline/logs/dssp_failed.jsonl",
        help="Failure log jsonl path.",
    )

    parser.add_argument(
        "--success-log",
        default="/data/ranran/my_ptm/croton/structure_pipeline/logs/dssp_success.jsonl",
        help="Success log jsonl path.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N PDB files for testing.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pdb_dir = Path(args.pdb_dir)
    dssp_dir = Path(args.dssp_dir)
    fail_log = Path(args.fail_log)
    success_log = Path(args.success_log)

    if not pdb_dir.exists():
        raise FileNotFoundError(f"PDB directory not found: {pdb_dir}")

    dssp_dir.mkdir(parents=True, exist_ok=True)
    fail_log.parent.mkdir(parents=True, exist_ok=True)

    check_mkdssp()

    if args.process_all_pdbs:
        target_proteins = None
        print("Protein filter: disabled, processing all short-name PDB files.")
    else:
        target_proteins = load_target_proteins(args.input_csv)
        print(f"Protein filter: enabled, target proteins = {len(target_proteins)}")

    pdb_files = collect_pdb_files(pdb_dir, target_proteins)

    if args.limit is not None:
        pdb_files = pdb_files[: args.limit]

    print("=" * 80)
    print("PDB to DSSP conversion")
    print("=" * 80)
    print(f"PDB directory:  {pdb_dir}")
    print(f"DSSP directory: {dssp_dir}")
    print(f"PDB files to process: {len(pdb_files)}")
    print(f"Workers: {args.workers}")
    print(f"Overwrite: {args.overwrite}")
    print("=" * 80)

    if not pdb_files:
        print("No PDB files to process.")
        write_jsonl(fail_log, [])
        write_jsonl(success_log, [])
        return

    failures: list[dict] = []
    successes: list[dict] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(run_mkdssp_one, pdb_path, dssp_dir, args.overwrite)
            for pdb_path in pdb_files
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="mkdssp"):
            protein_id, ok, reason = future.result()

            if ok:
                successes.append({"protein": protein_id, "status": reason})
            else:
                failures.append({"protein": protein_id, "reason": reason})

    write_jsonl(fail_log, failures)
    write_jsonl(success_log, successes)

    print("=" * 80)
    print(f"DSSP input PDBs: {len(pdb_files)}")
    print(f"DSSP successes:  {len(successes)}")
    print(f"DSSP failures:   {len(failures)}")
    print(f"Success log:     {success_log}")
    print(f"Failure log:     {fail_log}")
    print("=" * 80)

    if failures:
        print("First 10 failures:")
        for row in failures[:10]:
            print(row)


if __name__ == "__main__":
    main()