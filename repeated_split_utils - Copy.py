"""
repeated_split_utils.py
Section 2.7 helper: regenerates independent 80:10:10 scaffold splits of the
full curated dataset for the repeated-CV scripts (07, 11, 17). Infrastructure
glue shared by those numbered scripts, not itself one of the SI stages.
"""
import pandas as pd

import importlib
data_prep = importlib.import_module("01_data_prep")

from common import PROCESSED_DATA_DIR
import os


def full_curated_dataframe() -> pd.DataFrame:
    """Recombine train/val/test partitions from 01_data_prep.py back into
    the single curated dataset, so it can be re-split with new seeds."""
    parts = []
    for name in ("train", "val", "test"):
        parts.append(pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f"{name}.csv")))
    return pd.concat(parts, axis=0, ignore_index=True)


def make_split(df: pd.DataFrame, seed: int):
    return data_prep.scaffold_split(df, seed=seed)
