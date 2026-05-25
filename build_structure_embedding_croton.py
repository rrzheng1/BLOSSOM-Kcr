#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build full-length structure embeddings for crotonylation dataset.

Current project status:
1. PDB files already exist:
   /data/ranran/my_ptm/croton/structure_pipeline/pdb

2. DSSP files already exist:
   /data/ranran/my_ptm/croton/structure_pipeline/dssp

3. Clean data:
   /data/ranran/my_ptm/croton/data_clean/train_80.csv
   /data/ranran/my_ptm/croton/data_clean/test_20.csv

This script only extracts full-length structure embeddings:
PDB + DSSP + sequence -> [L, 112] tensor

Output:
   /data/ranran/my_ptm/croton/structure_embedding/{protein}.pt
   /data/ranran/my_ptm/croton/structure_embedding/{protein}.json
   /data/ranran/my_ptm/croton/structure_embedding/summary.csv

Feature layout:
   R_pro  : 34 dims
   R_aa   : 43 dims = 34 residue node + 9 edge mean
   R_atom : 35 dims = 26 atom node mean + 9 edge mean
   Total  : 112 dims
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# =========================================================
# Default paths for your current crotonylation project
# =========================================================
DEFAULT_INPUTS = [
    "/data/ranran/my_ptm/croton/data_clean/train_80.csv",
    "/data/ranran/my_ptm/croton/data_clean/test_20.csv",
]

DEFAULT_STRUCTURE_DIR = "/data/ranran/my_ptm/croton/structure_pipeline"
DEFAULT_OUTPUT_DIR = "/data/ranran/my_ptm/croton/structure_embedding"


# =========================================================
# Constants
# =========================================================
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWYX"
SS8_VOCAB = "HGIEBTSC"

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}

CHARGED = set("DEKRH")
POLAR = set("STNQCYW")
HYDROPHOBIC = set("AVLIMFP")
VALID_AA_RE = re.compile(r"^[A-Z]+$")


@dataclass(frozen=True)
class ResidueRecord:
    index: int
    aa: str
    ss8: str
    asa: float
    phi: float
    psi: float
    ca: tuple[float, float, float]


@dataclass(frozen=True)
class AtomRecord:
    residue_index: int
    residue_aa: str
    atom_name: str
    element: str
    coord: tuple[float, float, float]


def safe_name(protein_id: str, suffix: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(protein_id).strip()) + suffix


def load_unique_proteins(csv_paths: list[str]) -> list[tuple[str, str]]:
    """
    Load unique protein and sequence pairs from train/test CSV files.

    Required columns:
        protein
        sequence
    """
    proteins: dict[str, str] = {}

    for csv_path in csv_paths:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Input CSV not found: {path}")

        df = pd.read_csv(path, dtype=str)

        if "protein" not in df.columns or "sequence" not in df.columns:
            raise ValueError(f"{path} must contain 'protein' and 'sequence' columns")

        for protein_id, sequence in df[["protein", "sequence"]].itertuples(index=False):
            protein_id = str(protein_id).strip()
            sequence = str(sequence).strip().upper()

            if not protein_id or not sequence or sequence == "NAN":
                continue

            if not VALID_AA_RE.match(sequence):
                raise ValueError(
                    f"{path}: protein {protein_id} has illegal sequence characters"
                )

            old = proteins.get(protein_id)
            if old is not None and old != sequence:
                raise ValueError(
                    f"Protein {protein_id} has inconsistent sequences across CSV files"
                )

            proteins[protein_id] = sequence

    return sorted(proteins.items(), key=lambda item: item[0])


def is_complete_pdb(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False

    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return False

    return "ATOM" in text and ("END" in text or "CONECT" in text)


def parse_float(text: str, default: float = 0.0) -> float:
    try:
        value = float(text.strip())
    except ValueError:
        return default

    return value if math.isfinite(value) else default


def parse_int(text: str) -> int | None:
    try:
        return int(text.strip())
    except ValueError:
        return None


def normalize_dssp_aa(aa: str) -> str:
    aa = aa.strip()

    if not aa or aa == "!":
        return ""

    aa = aa.upper()

    return aa if aa in AA_VOCAB else "X"


def parse_dssp(path: Path) -> list[ResidueRecord]:
    """
    Parse DSSP file into residue records.
    """
    records: list[ResidueRecord] = []
    in_table = False

    for raw in path.read_text(errors="ignore").splitlines():
        if raw.lstrip().startswith("#"):
            in_table = True
            continue

        if not in_table or len(raw) < 40:
            continue

        aa = normalize_dssp_aa(raw[13:14])
        if not aa:
            continue

        index = parse_int(raw[5:10])
        if index is None:
            continue

        ss8 = raw[16:17].strip() or "C"
        if ss8 not in SS8_VOCAB:
            ss8 = "C"

        asa = parse_float(raw[34:38])

        phi = parse_float(raw[103:109]) if len(raw) >= 109 else 0.0
        psi = parse_float(raw[109:115]) if len(raw) >= 115 else 0.0

        if len(raw) >= 136:
            x = parse_float(raw[115:122])
            y = parse_float(raw[122:129])
            z = parse_float(raw[129:136])
        else:
            x = y = z = 0.0

        records.append(
            ResidueRecord(
                index=index,
                aa=aa,
                ss8=ss8,
                asa=asa,
                phi=phi,
                psi=psi,
                ca=(x, y, z),
            )
        )

    return records


def pdb_atom_element(line: str) -> str:
    element = line[76:78].strip() if len(line) >= 78 else ""

    if element:
        return element.upper()

    atom_name = line[12:16].strip()

    return re.sub(r"[^A-Za-z]", "", atom_name)[:1].upper() or "X"


def parse_pdb_atoms(
    path: Path,
) -> tuple[
    list[AtomRecord],
    dict[int, tuple[str, tuple[float, float, float]]],
    set[str],
    bool,
]:
    """
    Parse ATOM records from AlphaFold PDB.
    """
    atoms: list[AtomRecord] = []
    ca_by_index: dict[int, tuple[str, tuple[float, float, float]]] = {}
    chains: set[str] = set()
    has_insertion_code = False

    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith("ATOM"):
            continue

        atom_name = line[12:16].strip()
        residue_name = line[17:20].strip().upper()
        chain = line[21:22].strip() or "_"
        residue_index = parse_int(line[22:26])
        insertion_code = line[26:27].strip()

        if insertion_code:
            has_insertion_code = True
            continue

        if residue_index is None:
            continue

        aa = THREE_TO_ONE.get(residue_name, "X")
        if aa in {"U", "O"}:
            aa = "X"

        try:
            coord = (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
        except ValueError:
            continue

        chains.add(chain)

        atoms.append(
            AtomRecord(
                residue_index=residue_index,
                residue_aa=aa,
                atom_name=atom_name,
                element=pdb_atom_element(line),
                coord=coord,
            )
        )

        if atom_name == "CA":
            ca_by_index[residue_index] = (aa, coord)

    return atoms, ca_by_index, chains, has_insertion_code


def one_hot(value: str, vocab: str) -> np.ndarray:
    result = np.zeros(len(vocab), dtype=np.float32)
    index = vocab.find(value if value in vocab else "X")

    if index >= 0:
        result[index] = 1.0

    return result


def residue_node_feature(record: ResidueRecord) -> np.ndarray:
    """
    Residue node feature, 34 dims:
        phi/psi sin-cos: 4
        SS8 one-hot: 8
        normalized ASA: 1
        AA one-hot: 21
    """
    phi = math.radians(record.phi)
    psi = math.radians(record.psi)

    angle = np.array(
        [
            math.sin(phi),
            math.cos(phi),
            math.sin(psi),
            math.cos(psi),
        ],
        dtype=np.float32,
    )

    ss = one_hot(record.ss8, SS8_VOCAB)
    asa = np.array([min(max(record.asa, 0.0) / 300.0, 1.0)], dtype=np.float32)
    aa = one_hot(record.aa, AA_VOCAB)

    feature = np.concatenate([angle, ss, asa, aa]).astype(np.float32)

    if feature.shape[0] != 34:
        raise RuntimeError("Residue feature dimension must be 34")

    return feature


def atom_node_feature(atom: AtomRecord) -> np.ndarray:
    """
    Atom node feature, 26 dims:
        atom element one-hot: 5
        residue AA one-hot: 21
    """
    atom_vocab = ["C", "N", "O", "S", "X"]
    element = atom.element if atom.element in atom_vocab[:4] else "X"

    atom_one_hot = np.zeros(5, dtype=np.float32)
    atom_one_hot[atom_vocab.index(element)] = 1.0

    return np.concatenate(
        [
            atom_one_hot,
            one_hot(atom.residue_aa, AA_VOCAB),
        ]
    ).astype(np.float32)


def edge_feature(
    src_idx: int,
    dst_idx: int,
    src_aa: str,
    dst_aa: str,
    distance: float,
) -> np.ndarray:
    """
    Edge feature, 9 dims:
        sequence distance bins: 5
        residue relation features: 4
    """
    _ = distance

    seq_distance = abs(src_idx - dst_idx)

    seq_bins = np.zeros(5, dtype=np.float32)
    seq_bins[min(seq_distance, 4)] = 1.0

    relation = np.array(
        [
            1.0 if src_aa == dst_aa else 0.0,
            1.0 if src_aa in CHARGED and dst_aa in CHARGED else 0.0,
            1.0 if src_aa in POLAR and dst_aa in POLAR else 0.0,
            1.0 if src_aa in HYDROPHOBIC and dst_aa in HYDROPHOBIC else 0.0,
        ],
        dtype=np.float32,
    )

    feature = np.concatenate([seq_bins, relation]).astype(np.float32)

    if feature.shape[0] != 9:
        raise RuntimeError("Edge feature dimension must be 9")

    return feature


def build_neighbors(
    coords: np.ndarray,
    d_seq: int = 3,
    d_rad: float = 10.0,
    d_l: int = 5,
    knn_k: int = 10,
) -> list[list[int]]:
    """
    Build local residue neighbors for each residue.

    Includes:
        sequence neighbors within d_seq
        spatial radius neighbors
        kNN neighbors
    """
    length = coords.shape[0]
    neighbors: list[list[int]] = []

    for i in range(length):
        selected: set[int] = set()

        for offset in range(1, d_seq + 1):
            if i - offset >= 0:
                selected.add(i - offset)

            if i + offset < length:
                selected.add(i + offset)

        distances = np.linalg.norm(coords - coords[i], axis=1)
        spatial_order = [j for j in np.argsort(distances).tolist() if j != i]

        radius_hits = [j for j in spatial_order if distances[j] <= d_rad][:d_l]
        knn_hits = spatial_order[: min(knn_k, len(spatial_order))][:d_l]

        selected.update(radius_hits)
        selected.update(knn_hits)

        neighbors.append(sorted(selected))

    return neighbors


def validate_alignment(
    protein_id: str,
    sequence: str,
    dssp_records: list[ResidueRecord],
    ca_by_index: dict[int, tuple[str, tuple[float, float, float]]],
) -> None:
    """
    Strict alignment check:
        sequence length == DSSP length == PDB CA length
        residue index must be 1..L
        residue identity must match
    """
    if len(dssp_records) != len(sequence):
        raise ValueError(
            f"{protein_id}: DSSP length {len(dssp_records)} != sequence length {len(sequence)}"
        )

    expected = list(range(1, len(sequence) + 1))
    got = [record.index for record in dssp_records]

    if got != expected:
        raise ValueError(
            f"{protein_id}: DSSP residue indices are not exactly 1..L"
        )

    if sorted(ca_by_index) != expected:
        raise ValueError(
            f"{protein_id}: PDB CA residue indices are not exactly 1..L"
        )

    for pos, (seq_aa, record) in enumerate(zip(sequence, dssp_records), start=1):
        pdb_aa = ca_by_index[pos][0]

        if seq_aa != record.aa or seq_aa != pdb_aa:
            raise ValueError(
                f"{protein_id}: residue mismatch at {pos}: "
                f"sequence={seq_aa}, dssp={record.aa}, pdb={pdb_aa}"
            )


def make_structure_embedding(
    protein_id: str,
    sequence: str,
    pdb_path: Path,
    dssp_path: Path,
) -> tuple[np.ndarray, dict]:
    """
    Generate full-length [L,112] structure embedding for one protein.
    """
    if not is_complete_pdb(pdb_path):
        raise FileNotFoundError(f"{protein_id}: missing or damaged PDB: {pdb_path}")

    if not dssp_path.exists() or dssp_path.stat().st_size == 0:
        raise FileNotFoundError(f"{protein_id}: missing DSSP: {dssp_path}")

    dssp_records = parse_dssp(dssp_path)
    atoms, ca_by_index, chains, has_insertion_code = parse_pdb_atoms(pdb_path)

    if has_insertion_code:
        raise ValueError(f"{protein_id}: PDB contains insertion codes")

    if len(chains) != 1:
        raise ValueError(
            f"{protein_id}: expected exactly one PDB chain, found {sorted(chains)}"
        )

    validate_alignment(protein_id, sequence, dssp_records, ca_by_index)

    residue_features = np.stack(
        [residue_node_feature(record) for record in dssp_records]
    ).astype(np.float32)

    coords = np.asarray([record.ca for record in dssp_records], dtype=np.float32)

    if not np.isfinite(coords).all() or np.allclose(coords, 0.0):
        coords = np.asarray(
            [ca_by_index[i + 1][1] for i in range(len(sequence))],
            dtype=np.float32,
        )

    # Global protein-level feature, 34 dims
    r_pro = residue_features.mean(axis=0).astype(np.float32)

    # Local residue neighbors
    neighbors = build_neighbors(
        coords,
        d_seq=3,
        d_rad=10.0,
        d_l=5,
        knn_k=10,
    )

    atoms_by_residue: dict[int, list[AtomRecord]] = {}

    for atom in atoms:
        atoms_by_residue.setdefault(atom.residue_index, []).append(atom)

    rows = []

    for i, record in enumerate(dssp_records):
        neigh = neighbors[i]

        if neigh:
            edge_features = np.stack(
                [
                    edge_feature(
                        i + 1,
                        j + 1,
                        record.aa,
                        dssp_records[j].aa,
                        float(np.linalg.norm(coords[i] - coords[j])),
                    )
                    for j in neigh
                ]
            )

            aa_edge_mean = edge_features.mean(axis=0).astype(np.float32)
        else:
            aa_edge_mean = np.zeros(9, dtype=np.float32)

        # Residue-granularity feature: 34 + 9 = 43
        r_aa = np.concatenate(
            [
                residue_features[i],
                aa_edge_mean,
            ]
        ).astype(np.float32)

        local_atoms = list(atoms_by_residue.get(i + 1, []))

        for j in neigh:
            local_atoms.extend(atoms_by_residue.get(j + 1, []))

        if local_atoms:
            atom_nodes = np.stack([atom_node_feature(atom) for atom in local_atoms])
            atom_node_mean = atom_nodes.mean(axis=0).astype(np.float32)
        else:
            atom_node_mean = np.concatenate(
                [
                    np.array([0, 0, 0, 0, 1], dtype=np.float32),
                    one_hot(record.aa, AA_VOCAB),
                ]
            )

        # Atom-granularity feature: 26 + 9 = 35
        r_atom = np.concatenate(
            [
                atom_node_mean,
                aa_edge_mean,
            ]
        ).astype(np.float32)

        if r_aa.shape[0] != 43 or r_atom.shape[0] != 35:
            raise RuntimeError(f"{protein_id}: unexpected multi-granularity dimensions")

        # Total: 34 + 43 + 35 = 112
        rows.append(
            np.concatenate(
                [
                    r_pro,
                    r_aa,
                    r_atom,
                ]
            ).astype(np.float32)
        )

    embedding = np.stack(rows).astype(np.float32)

    if embedding.shape != (len(sequence), 112):
        raise RuntimeError(
            f"{protein_id}: embedding shape {embedding.shape} != ({len(sequence)}, 112)"
        )

    metadata = {
        "protein": protein_id,
        "length": len(sequence),
        "shape": list(embedding.shape),
        "chains": sorted(chains),
        "feature_layout": {
            "R_pro": [0, 34],
            "R_aa": [34, 77],
            "R_atom": [77, 112],
            "residue_node_dim": 34,
            "edge_dim": 9,
            "atom_node_dim": 26,
        },
        "neighbor_parameters": {
            "d_seq": 3,
            "d_rad_angstrom": 10.0,
            "d_l": 5,
            "knn_k": 10,
        },
    }

    return embedding, metadata


def embed_one(
    args: tuple[str, str, str, str, str, bool, str],
) -> tuple[str, bool, str, dict | None]:
    protein_id, sequence, pdb_dir, dssp_dir, output_dir, overwrite, dtype = args

    pdb_dir_path = Path(pdb_dir)
    dssp_dir_path = Path(dssp_dir)
    output_dir_path = Path(output_dir)

    output_dir_path.mkdir(parents=True, exist_ok=True)

    out_path = output_dir_path / safe_name(protein_id, ".pt")
    meta_path = output_dir_path / safe_name(protein_id, ".json")

    if not overwrite and out_path.exists() and out_path.stat().st_size > 0:
        return protein_id, True, "exists", None

    try:
        pdb_path = pdb_dir_path / safe_name(protein_id, ".pdb")
        dssp_path = dssp_dir_path / safe_name(protein_id, ".dssp")

        embedding, metadata = make_structure_embedding(
            protein_id=protein_id,
            sequence=sequence,
            pdb_path=pdb_path,
            dssp_path=dssp_path,
        )

        tensor = torch.from_numpy(embedding)

        if dtype == "float16":
            tensor = tensor.half()

        torch.save(tensor, out_path)

        meta_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )

        return protein_id, True, "embedded", metadata

    except Exception as exc:
        out_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)

        return protein_id, False, str(exc), None


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("protein,length,shape\n")
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "protein",
                "length",
                "shape",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate full-length [L,112] crotonylation structure embeddings."
    )

    parser.add_argument(
        "--input-csv",
        nargs="+",
        default=DEFAULT_INPUTS,
        help="Input train/test CSV files containing protein and sequence columns.",
    )

    parser.add_argument(
        "--structure-dir",
        default=DEFAULT_STRUCTURE_DIR,
        help="Structure pipeline directory containing pdb/ and dssp/.",
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for structure embedding .pt files.",
    )

    parser.add_argument(
        "--embed-workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) // 2)),
        help="Number of parallel embedding workers.",
    )

    parser.add_argument(
        "--dtype",
        choices=[
            "float32",
            "float16",
        ],
        default="float32",
        help="Output tensor dtype.",
    )

    parser.add_argument(
        "--overwrite-embedding",
        action="store_true",
        help="Overwrite existing .pt files.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N proteins for testing.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    structure_dir = Path(args.structure_dir)
    pdb_dir = structure_dir / "pdb"
    dssp_dir = structure_dir / "dssp"
    log_dir = structure_dir / "logs"
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    proteins = load_unique_proteins(args.input_csv)

    if args.limit is not None:
        proteins = proteins[: args.limit]

    print("=" * 80)
    print("Crotonylation structure embedding extraction")
    print("=" * 80)
    print(f"Input CSVs:       {args.input_csv}")
    print(f"Unique proteins:  {len(proteins)}")
    print(f"PDB directory:    {pdb_dir}")
    print(f"DSSP directory:   {dssp_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Log directory:    {log_dir}")
    print(f"Embed workers:    {args.embed_workers}")
    print(f"Dtype:            {args.dtype}")
    print("=" * 80)

    if not pdb_dir.exists():
        raise FileNotFoundError(f"PDB directory not found: {pdb_dir}")

    if not dssp_dir.exists():
        raise FileNotFoundError(f"DSSP directory not found: {dssp_dir}")

    jobs = [
        (
            protein_id,
            sequence,
            str(pdb_dir),
            str(dssp_dir),
            str(output_dir),
            args.overwrite_embedding,
            args.dtype,
        )
        for protein_id, sequence in proteins
    ]

    failures: list[dict] = []
    summaries: list[dict] = []

    with ProcessPoolExecutor(max_workers=args.embed_workers) as executor:
        for protein_id, ok, reason, metadata in tqdm(
            executor.map(embed_one, jobs),
            total=len(jobs),
            desc="Structure embeddings",
        ):
            if ok and metadata is not None:
                summaries.append(
                    {
                        "protein": protein_id,
                        "length": metadata["length"],
                        "shape": "x".join(map(str, metadata["shape"])),
                    }
                )
            elif not ok:
                failures.append(
                    {
                        "protein": protein_id,
                        "reason": reason,
                    }
                )

    write_jsonl(log_dir / "embedding_failed.jsonl", failures)
    write_summary_csv(output_dir / "summary.csv", summaries)

    print("=" * 80)
    print(f"Embedded proteins this run: {len(summaries)}")
    print(f"Embedding failures:        {len(failures)}")
    print(f"Summary CSV:               {output_dir / 'summary.csv'}")
    print(f"Failure log:               {log_dir / 'embedding_failed.jsonl'}")
    print("=" * 80)

    if failures:
        print("First 10 failures:")
        for row in failures[:10]:
            print(row)


if __name__ == "__main__":
    main()