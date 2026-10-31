#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from vina import Vina
from meeko import MoleculePreparation, PDBQTWriterLegacy


def smiles_to_pdbqt(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if status != 0:
        return None

    try:
        AllChem.UFFOptimizeMolecule(mol)
    except Exception:
        pass

    preparator = MoleculePreparation()
    mol_setups = preparator.prepare(mol)

    if not mol_setups:
        return None

    pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(mol_setups[0])

    if not is_ok:
        return None

    return pdbqt_string


def dock_one_ligand(vina_obj, ligand_pdbqt: str, out_pdbqt: str, exhaustiveness: int, n_poses: int):
    vina_obj.set_ligand_from_string(ligand_pdbqt)
    vina_obj.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
    energies = vina_obj.energies(n_poses=n_poses)
    vina_obj.write_poses(out_pdbqt, n_poses=n_poses, overwrite=True)
    return energies


def main():
    parser = argparse.ArgumentParser(description="Batch docking of SMILES with AutoDock Vina")
    parser.add_argument("--input", required=True, help="Input CSV file containing SMILES")
    parser.add_argument("--smiles-col", default="smiles", help="Column name with SMILES")
    parser.add_argument("--id-col", default=None, help="Optional column with molecule IDs")
    parser.add_argument("--receptor", required=True, help="Prepared receptor PDBQT file")
    parser.add_argument("--outdir", default="vina_results", help="Output directory")
    parser.add_argument("--center", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"),
                        help="Docking box center")
    parser.add_argument("--size", nargs=3, type=float, required=True, metavar=("SX", "SY", "SZ"),
                        help="Docking box size")
    parser.add_argument("--exhaustiveness", type=int, default=8, help="Vina exhaustiveness")
    parser.add_argument("--n-poses", type=int, default=9, help="Number of poses to save")
    parser.add_argument("--cpu", type=int, default=1, help="Number of CPU threads")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.smiles_col not in df.columns:
        raise ValueError(f"Column '{args.smiles_col}' not found in {args.input}")

    outdir = Path(args.outdir)
    lig_dir = outdir / "ligands_pdbqt"
    pose_dir = outdir / "docked_poses"
    outdir.mkdir(parents=True, exist_ok=True)
    lig_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)

    v = Vina(sf_name="vina", cpu=args.cpu)
    v.set_receptor(args.receptor)
    v.compute_vina_maps(center=args.center, box_size=args.size)

    summary = []

    for i, row in df.iterrows():
        smiles = str(row[args.smiles_col]).strip()

        if args.id_col and args.id_col in df.columns:
            mol_id = str(row[args.id_col]).strip()
            if not mol_id:
                mol_id = f"mol_{i+1}"
        else:
            mol_id = f"mol_{i+1}"

        ligand_pdbqt_file = lig_dir / f"{mol_id}.pdbqt"
        pose_file = pose_dir / f"{mol_id}_out.pdbqt"

        try:
            ligand_pdbqt = smiles_to_pdbqt(smiles)
            if ligand_pdbqt is None:
                raise ValueError("Failed to generate ligand PDBQT from SMILES")

            with open(ligand_pdbqt_file, "w", encoding="utf-8") as f:
                f.write(ligand_pdbqt)

            energies = dock_one_ligand(
                vina_obj=v,
                ligand_pdbqt=ligand_pdbqt,
                out_pdbqt=str(pose_file),
                exhaustiveness=args.exhaustiveness,
                n_poses=args.n_poses,
            )

            best_score = float(energies[0][0]) if len(energies) > 0 else None

            summary.append({
                "molecule_id": mol_id,
                "smiles": smiles,
                "status": "success",
                "best_vina_kcal_mol": best_score,
                "ligand_pdbqt": str(ligand_pdbqt_file),
                "pose_file": str(pose_file),
                "error": ""
            })

            print(f"[OK] {mol_id}: {best_score:.3f} kcal/mol" if best_score is not None else f"[OK] {mol_id}")

        except Exception as e:
            summary.append({
                "molecule_id": mol_id,
                "smiles": smiles,
                "status": "failed",
                "best_vina_kcal_mol": None,
                "ligand_pdbqt": "",
                "pose_file": "",
                "error": str(e)
            })
            print(f"[FAIL] {mol_id}: {e}")

    summary_df = pd.DataFrame(summary)
    summary_file = outdir / "docking_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    print(f"\nDone. Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
