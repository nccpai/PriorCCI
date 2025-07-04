import os
import random
import numpy as np
import pandas as pd
import itertools
import scanpy as sc
import anndata as ad
from scipy.sparse import issparse
import multiprocessing as mp
from functools import partial


def process_group(i, adata_filtered, gene_list_df, cell_combinations, output_folder):
    ct1, ct2 = cell_combinations[i]
    fname = f'{output_folder}/combi-{ct1}_{ct2}_c{i}.txt'
    df = pd.read_csv(fname, header=None)

    tg_idx = [adata_filtered.var_names.get_loc(g) for g in gene_list_df['A'] if g in adata_filtered.var_names]
    og_idx = [adata_filtered.var_names.get_loc(g) for g in gene_list_df['B'] if g in adata_filtered.var_names]

    result = []
    for start in range(0, len(df), 100):
        c1 = df.iloc[start:start+100, 0]
        c2 = df.iloc[start:start+100, 1]

        x1 = np.array([adata_filtered[c, tg_idx].X.toarray().flatten() for c in c1])
        x2 = np.array([adata_filtered[c, og_idx].X.toarray().flatten() for c in c2])
        result.append(np.stack((x1, x2), axis=-1).astype(np.float32))

    save_path = f'{output_folder}/combi-{ct1}_{ct2}_c{i}.npz'
    np.savez(save_path, *result)
    print(f"✅ Saved: {save_path}")


def input_data_preprocess(adata, celltype_col,
                           gene_csv='DB/CCIdb.csv',
                           output_folder='cnn_input_data',
                           n_sample1=100,
                           n_sample2=100,
                           n_repeat=1000,
                           save_filtered_csv='DB/filtered_CCIdb.csv'):
    print("Step 1: Filtering genes with zero expression...")
    X = adata.to_df()
    X_filtered = X.loc[:, (X.sum(axis=0) != 0)]
    adata_filtered = ad.AnnData(X_filtered)
    adata_filtered.obs = adata.obs.copy()
    adata_filtered.var = adata[:, X_filtered.columns].var.copy()

    print("Step 2: QC and normalization...")
    adata_filtered.var["mt"] = adata_filtered.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata_filtered, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    sc.pp.normalize_total(adata_filtered, target_sum=1e4)
    sc.pp.regress_out(adata_filtered, ["total_counts", "pct_counts_mt"])
    sc.pp.scale(adata_filtered, max_value=10)

    print("Step 3: Loading and filtering ligand-receptor gene list...")
    gene_list_df = pd.read_csv(gene_csv, index_col=0)
    A_genes = gene_list_df['A'].str.split('+').explode().unique().tolist()
    adata_genes = adata_filtered.var.index.tolist()
    missing_genes = [g for g in A_genes if g not in adata_genes]

    for g in missing_genes:
        gene_list_df = gene_list_df[
            ~gene_list_df['A'].str.contains(g) &
            ~gene_list_df['B'].str.contains(g)
        ]
    gene_list_df.to_csv(save_filtered_csv)
    print(f"Filtered gene list saved to: {save_filtered_csv}")

    print("Step 4: Sampling cell pairs...")
    os.makedirs(output_folder, exist_ok=True)
    ct_d = {ct: adata_filtered.obs.query(f"{celltype_col} == @ct").index.tolist()
            for ct in adata_filtered.obs[celltype_col].unique()}
    cell_combinations = list(itertools.combinations(ct_d.keys(), 2))

    for idx, (ct1, ct2) in enumerate(cell_combinations):
        fname = f"{output_folder}/combi-{ct1}_{ct2}_c{idx}.txt"
        with open(fname, "w") as f:
            for _ in range(n_repeat):
                if len(ct_d[ct1]) < n_sample1 or len(ct_d[ct2]) < n_sample2:
                    continue
                s1 = random.sample(ct_d[ct1], n_sample1)
                s2 = random.sample(ct_d[ct2], n_sample2)
                random.shuffle(s1); random.shuffle(s2)
                for a, b in zip(s1, s2):
                    f.write(f"{a},{b}\n")
    print(f"Sampling files saved to: {output_folder}/combi-*.txt")

    print("Step 5: Generating npz files (this may take time)...")

    process_fn = partial(
        process_group,
        adata_filtered=adata_filtered,
        gene_list_df=gene_list_df,
        cell_combinations=cell_combinations,
        output_folder=output_folder
    )

    with mp.Pool(processes=max(1, len(cell_combinations) // 4)) as pool:
        pool.map(process_fn, range(len(cell_combinations)))

    print("All .npz files generated successfully.")

    sample_path = f'{output_folder}/combi-{cell_combinations[0][0]}_{cell_combinations[0][1]}_c0.npz'
    sample = np.load(sample_path)
    print(f"Sample file: {sample_path}")
    print(f"Shape: {sample['arr_0'].shape}")
