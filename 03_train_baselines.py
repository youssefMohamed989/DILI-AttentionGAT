"""
03_train_baselines.py
Section 2.5. Trains three descriptor-based baselines on the identical
scaffold split as the GNN:
  (i)   random forest, 500 trees, class-balanced weighting, min 2
        samples/leaf, on 2048-bit Morgan fingerprints (radius 2)
  (ii)  L2-regularized logistic regression, class-balanced weighting,
        on the same Morgan fingerprints
  (iii) random forest (matched hyperparameters) on standardized RDKit
        physicochemical descriptors (scaler fit on train partition only)

All hyperparameters are fixed a priori, not tuned on the test set.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from common import RESULTS_DIR, CHECKPOINT_DIR, SEED, set_seed
from dataset_io import load_partition
from graph_features import morgan_fingerprint, rdkit_descriptors


def featurize_partition(df: pd.DataFrame, feature_fn, smiles_col="canonical_smiles"):
    feats = []
    for smi in df[smiles_col]:
        f = feature_fn(smi)
        feats.append(f)
    return np.stack(feats, axis=0)


def build_morgan_features(name: str):
    df = load_partition(name)
    X = featurize_partition(df, lambda s: morgan_fingerprint(s))
    y = df["DILI_label"].values.astype(int)
    return X, y


def build_descriptor_features(name: str, names_ref=None):
    df = load_partition(name)
    rows, feature_names = [], names_ref
    for smi in df["canonical_smiles"]:
        vals, names = rdkit_descriptors(smi)
        if feature_names is None:
            feature_names = names
        rows.append(vals)
    X = np.nan_to_num(np.stack(rows, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    y = df["DILI_label"].values.astype(int)
    return X, y, feature_names


def main():
    set_seed(SEED)
    rf_kwargs = dict(n_estimators=500, class_weight="balanced", min_samples_leaf=2,
                      random_state=SEED, n_jobs=-1)

    results = {}

    # --- Morgan fingerprint baselines ---
    X_train, y_train = build_morgan_features("train")
    X_val, y_val = build_morgan_features("val")
    X_test, y_test = build_morgan_features("test")

    rf_fp = RandomForestClassifier(**rf_kwargs)
    rf_fp.fit(X_train, y_train)
    prob_rf_fp = rf_fp.predict_proba(X_test)[:, 1]
    results["RF_MorganFP"] = {"test_auroc": float(roc_auc_score(y_test, prob_rf_fp))}
    joblib.dump(rf_fp, os.path.join(CHECKPOINT_DIR, "rf_morgan.joblib"))
    np.save(os.path.join(RESULTS_DIR, "rf_morgan_test_probs.npy"), prob_rf_fp)

    lr_fp = LogisticRegression(penalty="l2", class_weight="balanced", max_iter=2000,
                                random_state=SEED)
    lr_fp.fit(X_train, y_train)
    prob_lr_fp = lr_fp.predict_proba(X_test)[:, 1]
    results["LR_MorganFP"] = {"test_auroc": float(roc_auc_score(y_test, prob_lr_fp))}
    joblib.dump(lr_fp, os.path.join(CHECKPOINT_DIR, "lr_morgan.joblib"))
    np.save(os.path.join(RESULTS_DIR, "lr_morgan_test_probs.npy"), prob_lr_fp)

    # --- RDKit descriptor baseline ---
    Xd_train, yd_train, feat_names = build_descriptor_features("train")
    Xd_val, yd_val, _ = build_descriptor_features("val", names_ref=feat_names)
    Xd_test, yd_test, _ = build_descriptor_features("test", names_ref=feat_names)

    scaler = StandardScaler().fit(Xd_train)  # statistics from training partition only
    Xd_train_s = scaler.transform(Xd_train)
    Xd_test_s = scaler.transform(Xd_test)

    rf_desc = RandomForestClassifier(**rf_kwargs)
    rf_desc.fit(Xd_train_s, yd_train)
    prob_rf_desc = rf_desc.predict_proba(Xd_test_s)[:, 1]
    results["RF_RDKitDescriptors"] = {"test_auroc": float(roc_auc_score(yd_test, prob_rf_desc))}
    joblib.dump({"model": rf_desc, "scaler": scaler, "feature_names": feat_names},
                os.path.join(CHECKPOINT_DIR, "rf_descriptors.joblib"))
    np.save(os.path.join(RESULTS_DIR, "rf_descriptors_test_probs.npy"), prob_rf_desc)

    with open(os.path.join(RESULTS_DIR, "baseline_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
