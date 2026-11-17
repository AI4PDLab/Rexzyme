import pandas as pd
import matplotlib.pyplot as plt
import itertools
import glob
import os
import argparse
import numpy as np
import re

parser = argparse.ArgumentParser(
    description='Plot pLDDT',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    '--results_folder',
    '--input',
    dest='results_folder',
    required=True,
    type=str,
    help='Folder containing PDB files'
)
parser.add_argument(
    '--output',
    type=str,
    default=None,
    help='Output scatter-plot path (default: <results_folder>/average_scatter_plot.png)'
)
parser.add_argument(
    '--output-dir',
    dest='output_dir',
    type=str,
    default=None,
    help='Directory for per-reaction CSVs and the log file (default: --results_folder)'
)
parser.add_argument(
    '--top_k',
    default=15,
    type=int,
    help='To load the correct top_k results'
)
parser.add_argument(
    '--output_per_file',
    action='store_true',
    help='Give an output per csv instead of global one'
)
args = parser.parse_args()

# Generate CSV with avg pLDDT
from Bio.PDB import PDBParser


def compute_average_b_factor(pdb_file):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('', pdb_file)
        b_factors = [atom.get_bfactor() for atom in structure.get_atoms()]
        if b_factors:
            return sum(b_factors) / len(b_factors)
        else:
            return None
    except Exception as e:
        print(f"Error parsing {pdb_file}: {e}")
        return None


def extract_file_indices(directory, top_k):
    """Extract unique input idx values from filenames in the directory."""
    pdb_files = os.listdir(directory)

    patterns = [
        # New current naming format:
        # top_output_topk100_file_0.fasta_0_ppl_18.82.pdb
        rf"top_output_topk{top_k}_file_(\d+)\.fasta_\d+_ppl_[0-9.]+\.pdb$",

        # Slightly older naming format:
        # top_output_topk100_file-0.fasta_0.pdb
        rf"top_output_topk{top_k}_file-(\d+)\.fasta_\d+\.pdb$",

        # Old naming format:
        # top_top_0_smiles_results.fasta_0.pdb
        r"top_top_(\d+)_smiles_results\.fasta_\d+\.pdb$",
    ]

    indices = set()

    for filename in pdb_files:
        for pattern in patterns:
            match = re.match(pattern, filename)
            if match:
                indices.add(int(match.group(1)))
                break

    return sorted(list(indices))


def process_pdb_files(directory, idx):
    """Collect all PDBs belonging to one input index."""
    pdb_files = os.listdir(directory)

    patterns = [
        # New current naming format
        rf"top_output_topk{args.top_k}_file_{idx}\.fasta_\d+_ppl_[0-9.]+\.pdb$",

        # Older naming format with dash
        rf"top_output_topk{args.top_k}_file-{idx}\.fasta_\d+\.pdb$",

        # Old naming format
        rf"top_top_{idx}_smiles_results\.fasta_\d+\.pdb$",
    ]

    matched_pdb_files = []
    for pdb in pdb_files:
        if not pdb.endswith(".pdb"):
            continue
        for pattern in patterns:
            if re.match(pattern, pdb):
                matched_pdb_files.append(pdb)
                break

    if len(matched_pdb_files) == 0:
        print(f'No files found for input {idx}')
        return None
    else:
        print(f'{len(matched_pdb_files)} sequences for input {idx}')

    results = []

    for pdb_file in matched_pdb_files:
        pdb_path = os.path.join(directory, pdb_file)
        avg_b_factor = compute_average_b_factor(pdb_path)
        if avg_b_factor is not None:
            results.append({
                'file': pdb_file,
                'average_b_factor': avg_b_factor
            })

    df = pd.DataFrame(results)
    return df


def save_to_csv(df, output_csv):
    df.to_csv(output_csv, index=False)


output_dir = args.output_dir or args.results_folder
os.makedirs(output_dir, exist_ok=True)

# Extract actual indices from filenames
available_indices = extract_file_indices(args.results_folder, args.top_k)

if not available_indices:
    print("No valid PDB files found with recognized naming patterns.")

for index in available_indices:
    output_csv_file = f'average_plddt_{index}.csv'

    df = process_pdb_files(args.results_folder, index)

    if df is not None and not df.empty:
        save_to_csv(df, os.path.join(output_dir, output_csv_file))
        print(f'Average B-factors have been saved to {output_csv_file}')


results_folder = args.results_folder
log_file_path = os.path.join(output_dir, 'plddt_statistics.out')

if log_file_path:
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        log_file.write('')

csv_files = glob.glob(os.path.join(output_dir, '*.csv'))

# Colors for different datasets
colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])

# Initialize the plot
plt.figure(figsize=(10, 6))


def log_output():
    global all_values

    if len(all_values) == 0:
        messages = ["No pLDDT values found."]
        for line in messages:
            print(line)
        if log_file_path:
            with open(log_file_path, 'a', encoding='utf-8') as log_file:
                for line in messages:
                    log_file.write(f"{line}\n")
        return []

    messages = []

    if max(all_values) > 1:
        threshold_plddt = 70
        scaling = 1
    else:
        threshold_plddt = 0.7
        scaling = 100

    over70 = sum([1 for v in all_values if float(v) >= threshold_plddt])
    perc = 100 * over70 / len(all_values)
    messages.append(f'Number of pLDDT values over 70: {over70}/{len(all_values)} ({perc:.2f}%)')

    mean = np.array(all_values).mean() * scaling
    std = np.array(all_values).std() * scaling
    messages.append(f'Mean: {mean:.2f} STD: {std:.2f}')

    for line in messages:
        print(line)

    if log_file_path:
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            for line in messages:
                log_file.write(f"{line}\n")

    return []


# Loop through the CSV files and plot the data
all_values = []
last_df_len = None

for csv_file, color in zip(csv_files, colors):
    if os.stat(csv_file).st_size <= 1:
        continue

    df = pd.read_csv(csv_file)
    if df.empty:
        continue

    values = df['average_b_factor']
    all_values += values.to_list()
    last_df_len = len(df)

    plt.scatter(
        range(1, len(df) + 1),
        values,
        color=color,
        label=os.path.basename(csv_file).replace("average_plddt_", "").replace(".csv", "")
    )

    if args.output_per_file:
        print(f'Value for: {csv_file}')
        all_values = log_output()

if not args.output_per_file:
    log_output()

plt.xlabel('PDB File Index')
plt.ylabel('Average pLDDT')

if last_df_len is not None:
    plt.xticks(range(1, last_df_len + 1, max(1, last_df_len // 10)))

plt.legend()
plt.tight_layout()

output_image_file = args.output or os.path.join(output_dir, 'average_scatter_plot.png')
plt.savefig(output_image_file)

print(f'The plot has been saved to {output_image_file}')
