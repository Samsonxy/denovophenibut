import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import pandas as pd


def load_smiles_set(file_path):
    """Loads a CSV file and extracts unique SMILES strings as a set."""
    try:
        # Read CSV file
        df = pd.read_csv(file_path)

        # Normalize column names to lowercase to prevent capitalization issues
        df.columns = df.columns.str.strip().str.lower()

        if "smiles" not in df.columns:
            print(
                f"Warning: 'smiles' column not found in {file_path}. Columns available: {list(df.columns)}"
            )
            return set()

        # Drop missing values and convert to a set of unique strings
        return set(df["smiles"].dropna().astype(str).str.strip())

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return set()


def main():
    # Define paths to your 3 CSV files and their display labels
    files = {
        "Database 1": "1.csv",  # <-- Replace with your actual filenames
        "Database 2": "2.csv",
        "Database 3": "3.csv",
    }

    # Load datasets into sets
    set1 = load_smiles_set(files["Database 1"])
    set2 = load_smiles_set(files["Database 2"])
    set3 = load_smiles_set(files["Database 3"])

    # Verify we have data to compare
    if not (set1 or set2 or set3):
        print("Error: All loaded sets are empty. Check your file paths/headers.")
        return

    # Calculate Set Mathematics
    intersection_all = set1.intersection(set2, set3)
    union_all = set1.union(set2, set3)

    # Print Text Analysis to the console
    print("=" * 50)
    print("             DATABASE ANALYSIS               ")
    print("=" * 50)
    print(f"{'Database':<15} | {'Unique SMILES Count':<20}")
    print("-" * 50)
    for label, path in files.items():
        current_set = (
            set1
            if label == "Database 1"
            else set2 if label == "Database 2" else set3
        )
        print(f"{label:<15} | {len(current_set):<20}")
    print("-" * 50)
    print(f"Total Unique Across All (Union):        {len(union_all)}")
    print(f"Common to All Three (Intersection):    {len(intersection_all)}")
    print("=" * 50)

    # Plotting the Venn Diagram
    plt.figure(figsize=(10, 8))

    # venn3 takes a list of sets and a tuple of labels
    venn = venn3(
        subsets=[set1, set2, set3], set_labels=list(files.keys())
    )

    # Add a custom title and layout adjustments
    plt.title(
        "SMILES Overlap Analysis Between 3 Databases",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Save diagram to disk and display it
    output_image = "smiles_venn_diagram.png"
    plt.savefig(output_image, dpi=300, bbox_inches="tight")
    print(f"Visual diagram saved successfully as: {output_image}")
    plt.show()


if __name__ == "__main__":
    main()

