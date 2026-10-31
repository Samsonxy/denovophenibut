#!/usr/bin/env python3

import os
import json
import time
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import AllChem

API_URL = "https://health.api.nvidia.com/v1/biology/mit/diffdock"
TIMEOUT = 300

# =========================================================
# EDIT THESE SETTINGS
# =========================================================

RECEPTOR_FILE = "prot.pdb"
COMPOUNDS_FILE = "phenibut_smiles.csv"
OUTPUT_DIR = "diffdock_results"

SMILES_COL = "smiles"
ID_COL = None

NUM_POSES = 10
TIME_DIVISIONS = 20
STEPS = 18
SAVE_TRAJECTORY = True
SLEEP_BETWEEN_CALLS = 0.2

EMBED_SEED = 0xF00D

# =========================================================


def read_protein_pdb(pdb_path: str) -> str:
    with open(pdb_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    protein_lines = [
        line.rstrip("\n")
        for line in lines
        if line.startswith(("ATOM", "HETATM"))
    ]

    if not protein_lines:
        raise ValueError(f"No ATOM/HETATM lines found in receptor file: {pdb_path}")

    return "\n".join(protein_lines)


def load_smiles_table(smiles_file: str, smiles_col: str = None, id_col: str = None):
    path = Path(smiles_file)

    if path.suffix.lower() in [".csv", ".tsv"]:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(smiles_file, sep=sep, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()

        print("Detected columns:", df.columns.tolist())

        if smiles_col is not None:
            smiles_col = smiles_col.strip()

        if id_col is not None:
            id_col = id_col.strip()

        if smiles_col is None or smiles_col not in df.columns:
            for candidate in ["smiles", "SMILES", "ligand", "compound_smiles"]:
                if candidate in df.columns:
                    smiles_col = candidate
                    break

        if smiles_col is None or smiles_col not in df.columns:
            raise ValueError(
                f"Could not find SMILES column. Available columns: {df.columns.tolist()}"
            )

        if id_col is not None and id_col not in df.columns:
            print(f"Warning: ID column '{id_col}' not found. Auto-generating IDs.")
            id_col = None

        if id_col is None:
            for candidate in ["id", "ID", "name", "compound_id", "compound"]:
                if candidate in df.columns:
                    id_col = candidate
                    break

        records = []
        for i, row in df.iterrows():
            smi = str(row[smiles_col]).strip()

            if not smi or smi.lower() == "nan":
                continue

            if id_col is not None and pd.notna(row[id_col]):
                cid = str(row[id_col]).strip()
                if not cid:
                    cid = f"cmpd_{i+1}"
            else:
                cid = f"cmpd_{i+1}"

            records.append({
                "compound_id": cid,
                "smiles": smi
            })

        return records

    else:
        records = []
        with open(smiles_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                parts = line.split(maxsplit=1)

                if len(parts) == 1:
                    cid = f"cmpd_{i}"
                    smi = parts[0]
                else:
                    cid, smi = parts[0], parts[1]

                records.append({
                    "compound_id": cid,
                    "smiles": smi
                })

        return records


def sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in str(name))


def smiles_to_sdf(smiles: str, compound_id: str, sdf_file: Path):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES for {compound_id}: {smiles}")

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = EMBED_SEED
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise RuntimeError(f"3D coordinate generation failed for {compound_id}")

    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            pass

    mol.SetProp("_Name", compound_id)
    mol.SetProp("SMILES", smiles)

    writer = Chem.SDWriter(str(sdf_file))
    writer.write(mol)
    writer.close()


def read_text_file(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def call_diffdock(session, api_key: str, protein_content: str, ligand_sdf_text: str):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "ligand": ligand_sdf_text,
        "ligand_file_type": "sdf",
        "protein": protein_content,
        "num_poses": NUM_POSES,
        "time_divisions": TIME_DIVISIONS,
        "steps": STEPS,
        "save_trajectory": SAVE_TRAJECTORY,
    }

    response = session.post(API_URL, headers=headers, json=payload, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def write_pose_confidence_file(compound_dir: Path, confidences):
    out_file = compound_dir / "pose_confidence.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("rank\tconfidence\n")
        for i, conf in enumerate(confidences, start=1):
            f.write(f"rank{i:02d}\t{conf}\n")
    return out_file


def write_rank_pose_files(compound_dir: Path, ligand_positions, position_confidence):
    saved = []

    for i, pose_text in enumerate(ligand_positions, start=1):
        rank_tag = f"rank{i:02d}"
        conf = position_confidence[i - 1] if i - 1 < len(position_confidence) else ""

        sdf_file = compound_dir / f"{rank_tag}_confidence.sdf"
        with open(sdf_file, "w", encoding="utf-8") as f:
            f.write(pose_text.rstrip() + "\n")

        reverse_pdb = compound_dir / f"{rank_tag}_reverseprocess.pdb"
        with open(reverse_pdb, "w", encoding="utf-8") as f:
            if isinstance(pose_text, str):
                f.write(pose_text.rstrip() + "\n")
            else:
                f.write("")

        saved.append({
            "rank": rank_tag,
            "confidence": conf,
            "confidence_sdf": str(sdf_file),
            "reverseprocess_pdb": str(reverse_pdb),
        })

    return saved


def write_rank_trajectory_files(compound_dir: Path, trajectory, n_ranks):
    if not trajectory:
        return

    for i in range(n_ranks):
        rank_tag = f"rank{i+1:02d}"
        traj_file = compound_dir / f"{rank_tag}_trajectory.txt"

        if i < len(trajectory):
            with open(traj_file, "w", encoding="utf-8") as f:
                f.write(str(trajectory[i]).rstrip() + "\n")
        else:
            with open(traj_file, "w", encoding="utf-8") as f:
                f.write("")


def main():
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "NVIDIA_API_KEY is not set. Run:\n"
            'export NVIDIA_API_KEY="your_api_key_here"'
        )

    receptor_path = Path(RECEPTOR_FILE)
    compounds_path = Path(COMPOUNDS_FILE)

    if not receptor_path.is_file():
        raise FileNotFoundError(f"Receptor file not found: {RECEPTOR_FILE}")

    if not compounds_path.is_file():
        raise FileNotFoundError(f"Compounds file not found: {COMPOUNDS_FILE}")

    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Reading receptor from: {RECEPTOR_FILE}")
    protein_content = read_protein_pdb(RECEPTOR_FILE)

    print(f"Reading compounds from: {COMPOUNDS_FILE}")
    compounds = load_smiles_table(COMPOUNDS_FILE, SMILES_COL, ID_COL)

    print(f"Loaded {len(compounds)} compounds")

    summary_rows = []
    session = requests.Session()

    for idx, rec in enumerate(compounds, start=1):
        compound_id = sanitize_name(rec["compound_id"])
        smiles = rec["smiles"]

        print(f"[{idx}/{len(compounds)}] Docking {compound_id}")

        compound_dir = output_root / compound_id
        compound_dir.mkdir(parents=True, exist_ok=True)

        try:
            input_sdf = compound_dir / f"{compound_id}.sdf"
            smiles_to_sdf(smiles, compound_id, input_sdf)
            ligand_sdf_text = read_text_file(input_sdf)

            result = call_diffdock(session, api_key, protein_content, ligand_sdf_text)

            out_json = compound_dir / f"{compound_id}.json"
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            ligand_positions = result.get("ligand_positions", []) or []
            position_confidence = result.get("position_confidence", []) or []
            trajectory = result.get("trajectory", []) or []

            pose_conf_file = write_pose_confidence_file(compound_dir, position_confidence)
            rank_files = write_rank_pose_files(compound_dir, ligand_positions, position_confidence)
            write_rank_trajectory_files(compound_dir, trajectory, len(ligand_positions))

            rank1_conf = position_confidence[0] if len(position_confidence) > 0 else ""
            rank1_sdf = str(compound_dir / "rank01_confidence.sdf") if len(ligand_positions) > 0 else ""
            rank1_reverse = str(compound_dir / "rank01_reverseprocess.pdb") if len(ligand_positions) > 0 else ""

            summary_rows.append({
                "compound_id": compound_id,
                "smiles": smiles,
                "status": result.get("status", "success"),
                "input_sdf": str(input_sdf),
                "pose_confidence_txt": str(pose_conf_file),
                "rank1_confidence": rank1_conf,
                "rank1_confidence_sdf": rank1_sdf,
                "rank1_reverseprocess_pdb": rank1_reverse,
                "n_ranks": len(rank_files),
                "json_file": str(out_json),
                "details": result.get("details", "")
            })

        except Exception as e:
            summary_rows.append({
                "compound_id": compound_id,
                "smiles": smiles,
                "status": "failed",
                "input_sdf": "",
                "pose_confidence_txt": "",
                "rank1_confidence": "",
                "rank1_confidence_sdf": "",
                "rank1_reverseprocess_pdb": "",
                "n_ranks": 0,
                "json_file": "",
                "details": str(e)
            })

        time.sleep(SLEEP_BETWEEN_CALLS)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_root / "screening_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"Finished. Results saved in: {OUTPUT_DIR}")
    print(f"Summary file: {summary_csv}")


if __name__ == "__main__":
    main()
