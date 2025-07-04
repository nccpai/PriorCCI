import os
import re
import numpy as np
import pandas as pd

def merge_gradcam_results(path, save_path):
    os.makedirs(save_path, exist_ok=True)

    # Collect all files starting with 'gcamplus_result_'
    result_files = sorted([
        f for f in os.listdir(path)
        if f.startswith("gcamplus_result_") and f.endswith(".txt")
    ])

    # Extract unique class names (e.g., B_TNK) from filenames
    class_names = []
    for filename in result_files:
        match = re.search(r'gcamplus_result_(.*)_v\d+\.txt', filename)
        if match:
            class_names.append(match.group(1))
        else:
            print(f"[WARNING] Failed to extract class name from: {filename}")
    class_names = sorted(list(set(class_names)))  # Remove duplicates

    # Process each class (e.g., B_TNK) separately
    for class_name in class_names:
        class_files = [
            f for f in result_files
            if f.startswith(f"gcamplus_result_{class_name}_v") and f.endswith(".txt")
        ]

        class_dfs = []

        for filename in sorted(class_files):
            df = pd.read_csv(os.path.join(path, filename), sep='\t')
            geneA_col, geneB_col, weight_col = df.columns.tolist()

            # Keep only the entry with the maximum weight for each gene pair
            df_unique = df.loc[
                df.groupby([geneA_col, geneB_col])[weight_col].idxmax()
            ]
            df_unique = df_unique.sort_values(by=weight_col, ascending=False).reset_index(drop=True)
            class_dfs.append(df_unique)

        # Count gene pair occurrences across all files
        gene_pairs = []
        for df in class_dfs:
            for _, row in df.iterrows():
                gene_pairs.append((row[geneA_col], row[geneB_col]))

        pair_df = pd.DataFrame(gene_pairs, columns=[geneA_col, geneB_col])
        pair_df['Count'] = pair_df.groupby([geneA_col, geneB_col])[geneB_col].transform('count')
        pair_df = pair_df.drop_duplicates().sort_values(by='Count', ascending=False).reset_index(drop=True)

        # Aggregate weights by gene pair
        gene_pair_weights = {}
        for df in class_dfs:
            for _, row in df.iterrows():
                pair = (row[geneA_col], row[geneB_col])
                gene_pair_weights.setdefault(pair, []).append(row[weight_col])

        # Calculate summary statistics for each gene pair
        stats = []
        for (geneA, geneB), weights in gene_pair_weights.items():
            weights = np.array(weights)
            mean_val = np.mean(weights)
            var_val = np.var(weights)
            std_val = np.std(weights)
            median_val = np.median(weights)
            cv_val = std_val / mean_val * 100 if mean_val != 0 else 0
            count_val = len(weights)

            stats.append({
                geneA_col: geneA,
                geneB_col: geneB,
                'Mean Normalized_Weight': mean_val,
                'Variance Normalized_Weight': var_val,
                'Std Dev Normalized_Weight': std_val,
                'Median Normalized_Weight': median_val,
                'CV Normalized_Weight': cv_val,
                'Count': count_val
            })

        # Save the result
        stats_df = pd.DataFrame(stats)
        stats_df = stats_df.sort_values(by='Count', ascending=False).reset_index(drop=True)
        output_file = os.path.join(save_path, f"gcam_{class_name}_res.csv")
        stats_df.to_csv(output_file, index=False)
        print(f"✅ Saved: {output_file}")
