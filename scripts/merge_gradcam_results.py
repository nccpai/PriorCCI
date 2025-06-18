import os
import re
import numpy as np
import pandas as pd

def merge_gradcam_results(path='gcam_res/', save_path='CCI_res/'):
    """
    Merge GradCAM++ results per class and compute statistical summaries of L-R gene pair importance.

    Parameters
    ----------
    path : str
        Directory containing GradCAM++ result .txt files.
    save_path : str
        Directory to save merged and summarized results.
    """
    os.makedirs(save_path, exist_ok=True)

    # List of result files
    file_list = sorted([f for f in os.listdir(path) if f.startswith("gcamplus_result_") and f.endswith(".txt")])

    # Extract class names from filenames
    class_names = list(set(
        re.search(r'gcamplus_result_(.*)_v\d+\.txt', f).group(1)
        for f in file_list if re.search(r'gcamplus_result_(.*)_v\d+\.txt', f)
    ))
    class_names.sort()

    for cell in class_names:
        cell_files = [f for f in file_list if f.startswith(f"gcamplus_result_{cell}_v")]
        dfs = []

        for file in sorted(cell_files):
            df = pd.read_csv(os.path.join(path, file), sep='\t')

            # For duplicated (A, B) pairs, keep the one with the highest weight
            df_unique = df.loc[df.groupby(['A', 'B'])['Normalized_Weight'].idxmax()]
            df_unique = df_unique.sort_values(by='Normalized_Weight', ascending=False).reset_index(drop=True)
            dfs.append(df_unique)

        # Count occurrences of each gene pair
        gene_pairs = [(row['A'], row['B']) for df in dfs for _, row in df.iterrows()]
        pair_counts = pd.Series(gene_pairs).value_counts().reset_index()
        pair_counts.columns = ['A_B_pair', 'Count']
        pair_counts[['A', 'B']] = pd.DataFrame(pair_counts['A_B_pair'].tolist(), index=pair_counts.index)
        pair_counts.drop(columns='A_B_pair', inplace=True)

        # Collect weights for each gene pair
        gene_pair_weights = {}
        for df in dfs:
            for _, row in df.iterrows():
                pair = (row['A'], row['B'])
                gene_pair_weights.setdefault(pair, []).append(row['Normalized_Weight'])

        # Compute statistical summaries
        stats_results = []
        for pair, weights in gene_pair_weights.items():
            mean_w = np.mean(weights)
            var_w = np.var(weights)
            std_w = np.std(weights)
            med_w = np.median(weights)
            cv_w = (std_w / mean_w) * 100 if mean_w else 0
            count = len(weights)
            stats_results.append({
                'A': pair[0],
                'B': pair[1],
                'Mean Normalized_Weight': mean_w,
                'Variance Normalized_Weight': var_w,
                'Std Dev Normalized_Weight': std_w,
                'Median Normalized_Weight': med_w,
                'CV Normalized_Weight': cv_w,
                'Count': count
            })

        stats_df = pd.DataFrame(stats_results)
        stats_df = stats_df.sort_values(by='Count', ascending=False).reset_index(drop=True)

        # Save the result
        stats_df.set_index('A', inplace=True)
        out_file = os.path.join(save_path, f"gcam_{cell}_res.csv")
        stats_df.to_csv(out_file)
        print(f"✅ Saved: {out_file}")