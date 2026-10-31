import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, Crippen, Lipinski, rdMolDescriptors

# Input and output files
input_csv = "smile_only"
output_csv = "smile_qed.csv"

# Read input
df = pd.read_csv(input_csv)

# Ensure required column exists
if "smiles" not in df.columns:
    raise ValueError("Input file must contain a 'smiles' column.")

def calc_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return pd.Series({
            "status": "invalid",
            "qed": None,
            "mol_wt": None,
            "logp": None,
            "hbd": None,
            "hba": None,
            "tpsa": None,
            "rot_bonds": None,
            "rings": None
        })

    return pd.Series({
        "status": "ok",
        "qed": QED.qed(mol),
        "mol_wt": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "rot_bonds": Lipinski.NumRotatableBonds(mol),
        "rings": Lipinski.RingCount(mol)
    })

# Calculate RDKit descriptors
props_df = df["smiles"].apply(calc_properties)

# Merge with original data
result_df = pd.concat([df, props_df], axis=1)

# Optional sorting: best MolMIM mean score first, then highest QED
sort_cols = [c for c in ["mean_score", "qed"] if c in result_df.columns]
result_df = result_df.sort_values(by=sort_cols, ascending=False)

# Save output
result_df.to_csv(output_csv, index=False)

print(f"Done. Results saved to {output_csv}")
print(result_df.head())
