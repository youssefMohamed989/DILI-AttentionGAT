#!/usr/bin/env bash
# Runs the full pipeline end to end, in the order described in the
# manuscript's Supporting Information (Section S1).
#
# Usage:
#   ./run_full_pipeline.sh            # expects data/raw/gold_standard_dili.csv
#   python make_demo_data.py && ./run_full_pipeline.sh   # synthetic smoke test
set -euo pipefail

python 01_data_prep.py
python 02_train_gnn.py
python 03_train_baselines.py
python 04_metrics_and_figures.py
python 05_interpretability.py
python 06_train_ensemble_gnn.py
python 07_repeated_scaffold_cv.py
python 08_bootstrap_analysis.py
python 09_advanced_figures.py
python 10_repeated_cv_figure_and_litcomparison.py
python 11_repeated_cv_ensemble.py
python 12_additional_validations.py
python 13_advanced_attribution_figure.py
python 14_toxicophore_attention_validation.py
python 15_train_chem_informed.py
python 16_mht_correction_and_aux_accuracy.py
python 17_repeated_cv_chem_informed.py

echo "Pipeline complete. See results/ and figures/."
