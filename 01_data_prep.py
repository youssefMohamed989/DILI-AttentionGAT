"""
01_data_prep.py
Section 2.1 dataset curation and scaffold splitting.

Input : data/raw/gold_standard_dili.csv  (columns: SMILES, DILI_label)
        (the 1,111-compound reconciled DILIrank/DILIst 'gold-standard' set;
        see manuscript ref. 3. Not redistributed here -- supply your own
        licensed copy, or run make_demo_data.py for a synthetic smoke test.)

Output: data/processed/{train,val,test}.csv
        data/processed/dataset_stats.json   (Supplementary S3)
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from common import (
    RAW_DATASET_PATH, PROCESSED_DATA_DIR, MW_MIN, MW_MAX, SPLIT_RATIOS, SEED, set_seed,
)
from graph_features import canonical_smiles, molecular_weight, murcko_scaffold


def curate(raw_csv_path: str, smiles_col: str = "SMILES", label_col: str = "DILI_label"):
    df = pd.read_csv(raw_csv_path)
    n_start = len(df)
    log = {"n_input": n_start}

    # (i) parse + canonicalize; drop unparsable
    df["canonical_smiles"] = df[smiles_col].apply(canonical_smiles)
    n_invalid = int(df["canonical_smiles"].isna().sum())
    df = df.dropna(subset=["canonical_smiles"]).copy()
    log["n_invalid_smiles_removed"] = n_invalid

    # (ii) duplicate canonical SMILES with conflicting labels -> flag/remove
    grp = df.groupby("canonical_smiles")[label_col].nunique()
    conflicting = grp[grp > 1].index.tolist()
    n_conflicting = int(len(conflicting))
    df = df[~df["canonical_smiles"].isin(conflicting)].copy()
    df = df.drop_duplicates(subset="canonical_smiles", keep="first")
    log["n_conflicting_duplicates_removed"] = n_conflicting

    # (iii) molecular-weight filter [100, 1200] Da
    df["mol_wt"] = df["canonical_smiles"].apply(molecular_weight)
    before = len(df)
    df = df[(df["mol_wt"] >= MW_MIN) & (df["mol_wt"] <= MW_MAX)].copy()
    log["n_mw_filtered_removed"] = int(before - len(df))

    df = df.rename(columns={label_col: "DILI_label"})
    df["DILI_label"] = df["DILI_label"].astype(int)
    df = df.reset_index(drop=True)

    log["n_curated"] = int(len(df))
    log["n_dili_positive"] = int(df["DILI_label"].sum())
    log["n_dili_negative"] = int((df["DILI_label"] == 0).sum())
    return df, log


def scaffold_split(df: pd.DataFrame, ratios=SPLIT_RATIOS, seed: int = SEED):
    """Greedy scaffold split: group by Bemis-Murcko scaffold, sort groups by
    descending size (ties broken by a fixed random seed), and greedily
    assign whole scaffold groups to train/val/test until each partition's
    target size is reached (Section 2.1)."""
    rng = np.random.RandomState(seed)
    df = df.copy()
    df["scaffold"] = df["canonical_smiles"].apply(murcko_scaffold)
    df["scaffold"] = df["scaffold"].fillna(df["canonical_smiles"])  # acyclic molecules: scaffold-of-one

    scaffold_groups = defaultdict(list)
    for idx, scaf in zip(df.index, df["scaffold"]):
        scaffold_groups[scaf].append(idx)

    groups = list(scaffold_groups.values())
    order = rng.permutation(len(groups))
    groups = [groups[i] for i in order]
    groups.sort(key=len, reverse=True)  # stable sort keeps the seeded tie order

    n_total = len(df)
    n_train_target = int(round(ratios[0] * n_total))
    n_val_target = int(round(ratios[1] * n_total))

    train_idx, val_idx, test_idx = [], [], []
    for g in groups:
        if len(train_idx) + len(g) <= n_train_target or not train_idx:
            train_idx.extend(g)
        elif len(val_idx) + len(g) <= n_val_target or not val_idx:
            val_idx.extend(g)
        else:
            test_idx.extend(g)

    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)
    return train_df, val_df, test_df, len(groups)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=RAW_DATASET_PATH)
    parser.add_argument("--smiles-col", default="SMILES")
    parser.add_argument("--label-col", default="DILI_label")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)

    if not os.path.exists(args.input):
        raise FileNotFoundError(
            f"{args.input} not found. Supply the curated gold-standard DILI CSV "
            f"(columns: SMILES, DILI_label), or run make_demo_data.py first for "
            f"a small synthetic dataset to smoke-test the pipeline."
        )

    df, log = curate(args.input, args.smiles_col, args.label_col)
    train_df, val_df, test_df, n_scaffolds = scaffold_split(df, seed=args.seed)

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "test.csv"), index=False)

    stats = {
        **log,
        "n_scaffold_groups": int(n_scaffolds),
        "seed": args.seed,
        "partitions": {
            "train": {
                "n": int(len(train_df)),
                "n_positive": int(train_df["DILI_label"].sum()),
                "positive_fraction": float(train_df["DILI_label"].mean()),
            },
            "val": {
                "n": int(len(val_df)),
                "n_positive": int(val_df["DILI_label"].sum()),
                "positive_fraction": float(val_df["DILI_label"].mean()),
            },
            "test": {
                "n": int(len(test_df)),
                "n_positive": int(test_df["DILI_label"].sum()),
                "positive_fraction": float(test_df["DILI_label"].mean()),
            },
        },
    }
    with open(os.path.join(PROCESSED_DATA_DIR, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    print(f"\nWrote train/val/test CSVs to {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    main()
