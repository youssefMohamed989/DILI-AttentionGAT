# Validating and Correcting Graph Neural Network Attention for Drug-Induced Liver Injury Prediction

Code accompanying the manuscript:

> Hassan, Y. M.; El-Tantawi, H.; Rabie Ali, I.; Attia, M. S. *Validating and
> Correcting Graph Neural Network Attention for Drug-Induced Liver Injury
> Prediction.* Prepared for submission to the *Journal of Chemical
> Information and Modeling* (ACS).
><img width="1386" height="1135" alt="fig 2" src="https://github.com/user-attachments/assets/9c7d5fa9-86b2-4759-9947-8a3d2a94fb35" />

> <img width="1774" height="887" alt="fig" src="https://github.com/user-attachments/assets/aa22d466-6ce0-4221-a4b0-4dcf45bf2b51" />

<img width="1699" height="926" alt="graphical abstract" src="https://github.com/user-attachments/assets/3d6b71f8-b489-42dd-a525-671a8b00c586" />

<img width="1483" height="1061" alt="fig 3" src="https://github.com/user-attachments/assets/671cd06c-e8a1-4c54-8a38-0c39b6a3943a" />

This repository implements, end to end, the curation, modeling, and
attention-validation pipeline described in the manuscript: a custom
PyTorch graph attention network (GAT) for drug-induced liver injury (DILI)
classification, descriptor-based baselines, a systematic test of whether
learned attention concentrates on literature-derived hepatotoxicity
structural alerts, and a chemistry-informed multi-task extension that
improves that alignment.

## What's implemented

| Script | Purpose (manuscript section) |
|---|---|
| `01_data_prep.py` | SMILES validation/canonicalization, deduplication, MW filtering, Bemis–Murcko scaffold split (§2.1) |
| `graph_features.py` | Atom/bond featurization, Morgan fingerprints, RDKit descriptors (§2.2) |
| `gnn_model.py` | Base 3-layer multi-head GAT + attention-pooling readout (§2.3), sparse batching |
| `02_train_gnn.py` | Single-model GAT training (§2.6) → Table 2 "GAT, single model" |
| `03_train_baselines.py` | RF/Morgan, LR/Morgan, RF/RDKit-descriptor baselines (§2.5) |
| `04_metrics_and_figures.py` | Full metric suite + ROC curves (§3.2, Table 2, Figure 1) |
| `05_interpretability.py` | Atom-level attention attribution maps (§3.8) |
| `gnn_model_v2.py` | GAT+JK architecture, shared training routine, deep-ensemble utilities (§2.3–2.4) |
| `06_train_ensemble_gnn.py` | 5-member deep ensemble (§2.4) → Table 2 "5-model deep ensemble" |
| `07_repeated_scaffold_cv.py` | Repeated (5×) scaffold-split CV, single-model GAT+JK vs. RF baseline (§2.7) |
| `08_bootstrap_analysis.py` | Bootstrap 95% CIs + paired significance tests (§2.8, Table 3) |
| `09_advanced_figures.py` | PR curves, calibration, confusion matrices, ensemble-uncertainty figure (§3.3–3.6) |
| `10_repeated_cv_figure_and_litcomparison.py` | Repeated-CV comparison figure + paired t-test |
| `11_repeated_cv_ensemble.py` | Repeated-CV **ensemble** robustness check (25 GAT trainings) |
| `12_additional_validations.py` | Y-randomization, learning curve, applicability domain, McNemar's test |
| `13_advanced_attribution_figure.py` | Composite publication-grade attribution figure (Figure 12) + graphical abstract panels |
| `14_toxicophore_attention_validation.py` | Systematic toxicophore-attention alignment test, all 1,079 compounds (§2.12, Table 7, Figure 13) |
| `gnn_model_v3.py` | Chemistry-informed multi-task GAT (GAT-CI-MT) (§2.13) |
| `15_train_chem_informed.py` | Trains GAT-CI-MT, repeats the alignment test (Table 8, Figure 14) |
| `16_mht_correction_and_aux_accuracy.py` | Bonferroni/FDR correction (Table 8) + auxiliary-task accuracy (Table 10) |
| `17_repeated_cv_chem_informed.py` | Repeated-split validation of GAT-CI-MT (Table 11, Figure 15) |

`common.py` centralizes every hyperparameter, path, and the seven
toxicophore SMARTS definitions used throughout; `dataset_io.py` and
`repeated_split_utils.py` are small shared loading/splitting helpers used
by several stages.

All models are implemented directly in plain PyTorch (no PyTorch Geometric
or DGL dependency), so the full message-passing and attention logic is
auditable from `gnn_model.py`, `gnn_model_v2.py`, and `gnn_model_v3.py`.

## Important note on the dataset

The curated gold-standard DILI dataset (reconciled DILIrank/DILIst
classifications, manuscript ref. 3) is **not redistributed in this
repository** — it is a third-party literature dataset with its own terms
of use. To reproduce the manuscript's numbers exactly:

1. Obtain the gold-standard compound list (SMILES + binary DILI label).
2. Save it as `data/raw/gold_standard_dili.csv` with columns `SMILES`,
   `DILI_label`.
3. Run the pipeline (below).

To simply verify the code runs end to end without that dataset, generate a
small synthetic placeholder set first:

```bash
python make_demo_data.py
```

This writes a tiny, clearly-synthetic SMILES/label CSV to the same path.
**Do not interpret results produced from it as DILI-predictive — it exists
only to exercise the code.**

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Requires RDKit, PyTorch, scikit-learn, scipy, statsmodels, pandas,
matplotlib. See `requirements.txt` for version floors.

## Running the pipeline

Run every stage in order:

```bash
./run_full_pipeline.sh
```

or run any stage individually, e.g.:

```bash
python 01_data_prep.py
python 02_train_gnn.py
python 14_toxicophore_attention_validation.py
```

Outputs land in `data/processed/` (curated splits + `dataset_stats.json`,
Supplementary S3), `checkpoints/` (trained model weights),
`results/` (metrics and statistical-test JSON files), and `figures/`
(all manuscript figures, regenerated from your run).

## Tests

A lightweight smoke-test suite (no real dataset required) checks
featurization, batching, and a forward/backward pass of every model
variant:

```bash
pytest tests/ -v
```

## Reproducibility

All scripts fix the governing random seed to 42 (data splitting, model
initialization; ensemble members additionally use seeds 43–46) so that,
given the same input dataset, splits and trained-model metrics match the
manuscript exactly. Repeated-CV scripts use five independent split seeds
(42–46) as described in Section 2.7.

## Citation

If you use this code, please cite the manuscript once published. A
`CITATION.cff`-style entry will be added on acceptance / DOI assignment.

## License

Code is released under the MIT License (see `LICENSE`). This does not
extend to any third-party dataset you supply.
