# DILI-AttentionGAT
End-to-end PyTorch pipeline for predicting drug-induced liver injury (DILI) from molecular structure using a custom graph attention network (GAT), with systematic validation of whether learned attention aligns with known hepatotoxicity structural alerts. Includes baselines, deep ensembles, repeated scaffold-split CV, and bootstrap statistics.
# DILI-GAT: Validating and Correcting Graph Neural Network Attention for Drug-Induced Liver Injury Prediction
<img width="1699" height="926" alt="graphical abstract" src="https://github.com/user-attachments/assets/7573d2f8-c623-4adc-be63-0fefd2acaa45" />


Authors: Youssef M. Hassan (1) (corresponding author), Hala El-Tantawi (1), Ibrahim Rabie Ali (2), Mohamed S. Attia (3)

(1) Department of Zoology, Faculty of Science, Ain Shams University, Abbassia 11566, Cairo, Egypt
(2) Department of Immunology and Treatment Evaluation, Theodore Bilharz Research Institute, Giza 12411, Egypt
(3) Chemistry Department, College of Science, Imam Mohammad Ibn Saud Islamic University (IMSIU), Riyadh 11623, Saudi Arabia

Corresponding author: Youssef M. Hassan, ORCID 0009-0005-3615-4137, yousefmohamed_p@sci.asu.edu.eg

Code accompanying the manuscript:

> Hassan, Y. M.; El-Tantawi, H.; Rabie Ali, I.; Attia, M. S. Validating
> and Correcting Graph Neural Network Attention for Drug-Induced Liver
> Injury Prediction. Journal of Chemical Information and Modeling
> (ACS), accepted.

---

## Table of contents

1. Abstract
2. What this project does
3. Why it matters
4. Repository layout
5. Method overview
   - Dataset curation and splitting
   - Molecular graph representation
   - Model architectures
   - Baselines
   - Toxicophore-attention validation
   - Chemistry-informed correction
6. Installation
7. Getting the data
8. Running the pipeline
9. Script-by-script reference
10. Outputs
11. Tests
12. Reproducibility
13. Known limitations
14. Citation
15. License

---

## Abstract

Attention-based graph neural networks (GNNs) are increasingly used for
drug-induced liver injury (DILI) and other toxicity prediction tasks on
the claimed strength of built-in interpretability, but this claim is
almost always supported by a handful of hand-selected examples rather
than tested systematically. Using a curated, scaffold-split
gold-standard DILI dataset (1,079 compounds), we show that a graph
attention network's attention weights do not, in general, concentrate on
seven literature-derived hepatotoxicity structural alerts across the
full dataset (pooled enrichment ratio 0.35), and that the network itself
does not outperform simple descriptor-based baselines under rigorous
repeated-split evaluation (mean AUROC 0.625-0.649 vs. 0.714 for random
forest, p = 0.011). We then show both gaps can be addressed: a
chemistry-informed multi-task extension, trained with an auxiliary
structural-alert-recognition loss, improves attention alignment for two
alerts by more than 5-fold (both Bonferroni/FDR-corrected p < 10^-14),
confirmed across five independent scaffold splits, without a robust cost
to classification performance. All curated data, trained models, and
analysis code are released to support reproducibility and further
benchmarking of both predictive and interpretability claims for
molecular GNNs.

Keywords: drug-induced liver injury; graph neural networks; graph
attention networks; molecular machine learning; scaffold split;
interpretability; toxicity prediction; QSAR

## What this project does

This repository is a complete, runnable implementation of the machine
learning pipeline described in the manuscript. It:

1. Predicts DILI risk from a molecule's 2D structure (SMILES) using a
   graph attention network (GAT) built from scratch in plain PyTorch.
2. Audits the model's learned attention: rather than assuming an
   attention-based GNN's attribution maps are chemically meaningful, the
   pipeline runs a systematic statistical test of whether attention
   actually concentrates on known hepatotoxicity-associated substructures
   (structural alerts, also called toxicophores) more than on the rest of
   the molecule, across the entire dataset rather than a handful of
   examples.
3. Corrects misaligned attention where the audit finds it wanting, by
   adding an auxiliary multi-task objective (toxicophore-presence
   prediction) that shares the same molecular representation as the DILI
   classifier, and re-runs the audit to quantify the improvement.
4. Benchmarks and stress-tests all of the above against descriptor-based
   baselines (random forest and logistic regression on Morgan
   fingerprints and RDKit descriptors), a 5-member deep ensemble, repeated
   independent data splits, bootstrap confidence intervals, and several
   standard robustness checks (Y-randomization, learning curves,
   applicability domain, McNemar's test).

Every model, training loop, statistical test, and figure reported in the
manuscript is reproduced by a corresponding script in this repository.

## Why it matters

Graph neural networks with attention mechanisms are often described as
interpretable because attention weights can be visualized as atom-level
heatmaps. But a heatmap is not, by itself, evidence that the model is
attending to anything chemically meaningful, it could just as easily be
attending to a positional or connectivity artifact that happens to
correlate with the training labels. This repository treats that
interpretability claim as a hypothesis to be tested, not an assumption:

- It defines seven literature-derived structural alerts associated with
  hepatotoxic reactive-metabolite formation (anilines/arylamines,
  hydrazines/hydrazides, thiophenes, Michael acceptors, furans, aromatic
  halides, carboxylic acids).
- It asks, across the entire curated dataset, whether attention on atoms
  belonging to these alerts is systematically higher than attention
  elsewhere (Mann-Whitney U test, one-sided, per alert and pooled, with
  multiple-testing correction).
- Where the answer is no, not in general, it trains a chemistry-informed
  variant explicitly supervised to recognize these alerts as an auxiliary
  task, and re-tests whether that measurably improves
  attention-toxicophore alignment, turning a post-hoc interpretability
  critique into a concrete, reproducible model improvement.

## Repository layout

dili-gat/
├── common.py # all shared constants, paths, hyperparameters, SMARTS
├── graph_features.py # atom/bond featurization, fingerprints, descriptors
├── dataset_io.py # CSV <-> graph loading helpers (infra, not a numbered stage)
├── repeated_split_utils.py # re-splitting helper for repeated-CV scripts (infra)
├── gnn_model.py # base 3-layer GAT + attention pooling + batching
├── gnn_model_v2.py # GAT+JK architecture, shared training loop, deep ensemble
├── gnn_model_v3.py # chemistry-informed multi-task GAT (GAT-CI-MT)
├── make_demo_data.py # synthetic placeholder dataset for smoke-testing
├── 01_data_prep.py # curation + scaffold split
├── 02_train_gnn.py # single-model GAT training
├── 03_train_baselines.py # RF/LR baselines
├── 04_metrics_and_figures.py # Table 2, Figure 1
├── 05_interpretability.py # atom-level attribution maps
├── 06_train_ensemble_gnn.py # 5-member deep ensemble
├── 07_repeated_scaffold_cv.py # repeated-split single-model robustness
├── 08_bootstrap_analysis.py # Table 3, bootstrap CIs + significance
├── 09_advanced_figures.py # PR/calibration/confusion-matrix/uncertainty figures
├── 10_repeated_cv_figure_and_litcomparison.py
├── 11_repeated_cv_ensemble.py # repeated-split ensemble robustness (25 trainings)
├── 12_additional_validations.py # Y-randomization, learning curve, AD, McNemar
├── 13_advanced_attribution_figure.py # Figure 12, composite attribution panel
├── 14_toxicophore_attention_validation.py # Table 7, Figure 13 (the core audit)
├── 15_train_chem_informed.py # trains GAT-CI-MT, Table 8, Figure 14
├── 16_mht_correction_and_aux_accuracy.py # multiple-testing correction, Table 10
├── 17_repeated_cv_chem_informed.py # Table 11, Figure 15
├── run_full_pipeline.sh # runs every stage in order
├── requirements.txt
├── data/{raw,processed}/ # inputs and curated splits (gitignored)
├── checkpoints/ # trained model weights (gitignored)
├── results/ # metrics + statistical test JSON (gitignored)
├── figures/ # all rendered figures (gitignored)
├── tests/test_pipeline_smoke.py # dataset-free smoke tests
└── .github/workflows/ci.yml # CI running the smoke tests


Files starting with a number correspond directly to a pipeline stage and
are meant to be run in order; everything else is shared infrastructure
imported by those scripts.

## Method overview

### Dataset curation and splitting

01_data_prep.py takes a raw CSV of SMILES, DILI_label pairs and:

1. Canonicalizes every SMILES string with RDKit and drops anything that
   fails to parse.
2. Removes molecules whose canonical SMILES is duplicated with a
   conflicting label, then deduplicates.
3. Filters to molecular weight in [100, 1200] Da.
4. Computes each molecule's Bemis-Murcko scaffold and performs a greedy
   scaffold split: scaffold groups are sorted by size (largest first,
   ties broken by a fixed seed) and assigned whole to train/val/test
   until each partition reaches its target share of 80% / 10% / 10%. No
   scaffold appears in more than one partition, a much harder and more
   realistic generalization test than a random split.

Dataset statistics (curation counts, per-partition class balance) are
written to data/processed/dataset_stats.json.

### Molecular graph representation

graph_features.py converts each molecule into an attributed graph:

- Atom features (37-dim per atom): one-hot element identity (12 common
  elements plus "other"), degree, formal charge, one-hot hybridization,
  aromaticity, ring membership, total attached hydrogens, plus
  additional one-hot degree/valence encodings padded to a fixed width.
- Bond features (6-dim per bond): one-hot bond order (single, double,
  triple, aromatic), conjugation flag, ring-membership flag. Edges are
  represented as explicit bidirectional pairs.
- The same module also computes 2048-bit Morgan fingerprints (radius 2)
  and the full RDKit descriptor set used by the baseline models.

### Model architectures

All three architectures are implemented directly in PyTorch, with no
PyTorch Geometric or DGL dependency, so the message-passing and
attention math is fully visible in gnn_model*.py rather than hidden
inside a framework.

- DILI_GAT (gnn_model.py) is the base model. Three stacked multi-head
  graph-attention layers (4 heads, 64 hidden dim), each of which projects
  source/destination atom features and bond features into a shared
  space, computes an attention score per directed edge via a learned
  per-head vector applied to the concatenation of the destination
  representation and the edge-conditioned message, normalizes those
  scores with a softmax over each destination node's incoming edges
  (segment-softmax, implemented directly with scatter_reduce/scatter_add,
  no external graph library), and aggregates messages by
  attention-weighted sum with a residual connection and LayerNorm. A
  learned attention-pooling readout then assigns each atom a scalar
  importance score (softmax-normalized over the atoms within a
  molecule), and the resulting attention-weighted sum of atom embeddings
  is fed to a small MLP classifier head. These per-atom pooling weights
  are exactly what gets tested for toxicophore alignment.

- DILI_GAT_JK (gnn_model_v2.py) adds a jumping-knowledge (JK) connection:
  the atom representations after every layer, including the initial
  embedding, are concatenated and projected back down before pooling.
  Used for the deep ensemble and all repeated-CV / chemistry-informed
  experiments.

- DILI_GAT_CI_MT (gnn_model_v3.py) is the chemistry-informed multi-task
  extension. Identical GAT+JK backbone, but the pooled molecular
  embedding feeds two heads: the DILI classifier, and an auxiliary
  multi-label head predicting which of the seven structural alerts are
  present. Trained with combined loss L = L_DILI + lambda times
  L_toxicophore, with lambda = 0.4, fixed a priori.

Batching for variable-size molecular graphs is handled by
collate_graphs/MolGraphBatch in gnn_model.py: graphs are combined into
one block-diagonal sparse batch (PyG-style), with a graph_index vector
tracking which molecule each atom belongs to.

### Baselines

03_train_baselines.py trains, on the identical scaffold split:

- Random forest (500 trees, class_weight="balanced",
  min_samples_leaf=2) on 2048-bit Morgan fingerprints.
- L2-regularized logistic regression (class_weight="balanced") on the
  same fingerprints.
- Random forest (matched hyperparameters) on standardized RDKit
  descriptors (scaler fit on the training partition only).

All hyperparameters are fixed in advance, not tuned against the test set.

### Toxicophore-attention validation

This is the paper's central methodological contribution, implemented in
14_toxicophore_attention_validation.py:

1. Each of the seven structural alerts is defined as a SMARTS pattern in
   common.TOXICOPHORE_SMARTS.
2. For every molecule in the full curated dataset, RDKit substructure
   matching flags exactly which atoms belong to each alert.
3. The trained GAT's attention-pooling weights are extracted and
   min-max normalized within each molecule.
4. For each alert, in-alert atoms are compared to out-of-alert atoms via
   a one-sided Mann-Whitney U test (alternative hypothesis: in-alert
   attention is greater), both per-alert and pooled across all seven. An
   enrichment ratio (mean in-alert attention divided by mean
   out-of-alert attention) is reported alongside each test.
5. Two additional checks: does the count of alerts present in a molecule
   correlate with its true DILI label (point-biserial correlation), and
   with the model's predicted probability (Pearson correlation)?

16_mht_correction_and_aux_accuracy.py applies Bonferroni and
Benjamini-Hochberg FDR correction across the seven per-alert p-values,
since testing seven hypotheses at once inflates the false-positive rate
if left uncorrected.

### Chemistry-informed correction

15_train_chem_informed.py trains DILI_GAT_CI_MT and repeats the exact
same validation procedure on its attention weights, so the baseline
GAT's alignment results (Table 7) and the chemistry-informed model's
results (Table 8) are directly comparable. 16_mht_correction_and_aux_accuracy.py
additionally reports the auxiliary head's own classification accuracy
(precision, recall, F1 per alert). 17_repeated_cv_chem_informed.py
repeats the whole comparison across five independent scaffold splits to
confirm the improvement isn't an artifact of one particular split.

## Installation

git clone <this-repo-url>
cd dili-gat
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt


Core dependencies: RDKit, PyTorch (2.1 or newer), scikit-learn, scipy,
statsmodels, pandas, matplotlib, joblib. No GPU is required, everything
trains quickly on CPU given the dataset's modest size (about 1,100
compounds), though a GPU will be used automatically if
torch.cuda.is_available() returns true.

## Getting the data

The curated gold-standard DILI dataset (reconciled DILIrank/DILIst
classifications, 1,079 compounds) is not redistributed in this
repository, it is a third-party literature dataset with its own terms of
use. To reproduce the manuscript's exact numbers:

1. Obtain the gold-standard compound list (SMILES plus binary DILI
   label) yourself.
2. Save it as data/raw/gold_standard_dili.csv with columns SMILES and
   DILI_label (1 = DILI-positive, 0 = DILI-negative).
3. Proceed to Running the pipeline, below.

To simply verify the code runs end to end without that dataset, generate
a small synthetic placeholder set first:

python make_demo_data.py


This writes about 40 well-known drug-like SMILES with randomly assigned
labels to the same path, purely to exercise the pipeline mechanically.
Do not interpret any resulting metrics as evidence about real DILI
prediction performance.

## Running the pipeline

Run every stage in order:

./run_full_pipeline.sh


or run any stage individually once its inputs exist:

python 01_data_prep.py
python 02_train_gnn.py
python 14_toxicophore_attention_validation.py


Most training scripts accept --device, --seed, --max-epochs, and
--patience overrides (see each script's argparse block); defaults
reproduce the manuscript's settings.

## Script-by-script reference

| # | Script | What it produces | Manuscript reference |
|---|---|---|---|
| - | 01_data_prep.py | curated and scaffold-split train/val/test CSVs, dataset_stats.json | Section 2.1, Supplementary S3 |
| 02 | 02_train_gnn.py | trained single-model GAT checkpoint, test predictions | Section 2.6, Table 2 |
| 03 | 03_train_baselines.py | trained RF/LR baselines, test predictions | Section 2.5 |
| 04 | 04_metrics_and_figures.py | full metric suite (AUROC, AUPRC, accuracy, sensitivity, specificity, MCC, F1), ROC curves | Section 3.2, Table 2, Figure 1 |
| 05 | 05_interpretability.py | atom-level attention attribution maps for representative TP/TN/FP/FN molecules | Section 3.8 |
| 06 | 06_train_ensemble_gnn.py | 5-member deep ensemble, mean and std predictions | Section 2.4, Table 2 |
| 07 | 07_repeated_scaffold_cv.py | single-model GAT+JK vs. RF across 5 independent splits | Section 2.7, Table 6 |
| 08 | 08_bootstrap_analysis.py | 95% bootstrap CIs, paired significance tests | Section 2.8, Table 3 |
| 09 | 09_advanced_figures.py | PR curves, calibration curves, confusion matrices, ensemble-uncertainty figure | Section 3.3-3.6 |
| 10 | 10_repeated_cv_figure_and_litcomparison.py | repeated-CV comparison figure, paired t-test | Section 3.7 |
| 11 | 11_repeated_cv_ensemble.py | full ensemble robustness check across 5 splits (25 trainings) | Section 3.7 |
| 12 | 12_additional_validations.py | Y-randomization, learning curve, applicability domain, McNemar's test | Discussion |
| 13 | 13_advanced_attribution_figure.py | composite multi-panel attribution figure | Figure 12 |
| 14 | 14_toxicophore_attention_validation.py | per-alert and pooled attention-enrichment statistics | Section 2.12, Table 7, Figure 13 |
| 15 | 15_train_chem_informed.py | trained GAT-CI-MT, repeated toxicophore validation | Section 2.13, Table 8, Figure 14 |
| 16 | 16_mht_correction_and_aux_accuracy.py | Bonferroni/FDR-corrected p-values, auxiliary-task accuracy | Table 8 (corrected), Table 10 |
| 17 | 17_repeated_cv_chem_informed.py | GAT-CI-MT robustness across 5 splits | Table 11, Figure 15 |

## Outputs

Running the pipeline populates four directories (all gitignored, since
they are regenerated from your data plus these scripts):

- data/processed/: curated dataset splits and curation statistics.
- checkpoints/: trained PyTorch state dicts and joblib-serialized
  scikit-learn models.
- results/: JSON files with every reported metric, bootstrap CI,
  significance test, and per-alert enrichment statistic.
- figures/: every rendered figure (ROC curves, PR curves, calibration
  plots, confusion matrices, attribution maps, bootstrap forest plot,
  repeated-CV comparison plots, toxicophore-enrichment bar charts).

## Tests

A lightweight, dataset-free smoke-test suite checks that featurization,
batching, and a full forward and backward pass work for all three model
variants, and that every toxicophore SMARTS pattern parses and matches
at least one test molecule:

pytest tests/ -v


This also runs automatically on every push and pull request via
.github/workflows/ci.yml.

## Reproducibility

- All data-splitting and model-initialization randomness is seeded
  (default seed 42); deep-ensemble members additionally use seeds 43
  through 46.
- Repeated-CV scripts regenerate the scaffold split five times with
  independent seeds 42 through 46 (common.REPEATED_SPLIT_SEEDS).
- Every hyperparameter (learning rate, weight decay, batch size,
  gradient-clip norm, early-stopping patience, architecture dimensions,
  the chemistry-informed loss weight lambda) is centralized in
  common.py so there is a single source of truth matching the
  manuscript's Methods section.
- Given the same input dataset, splits and trained-model metrics are
  deterministic modulo standard PyTorch non-determinism on GPU (CPU runs
  are fully deterministic given a fixed seed).

## Known limitations

- The real gold-standard DILI dataset is not included (see Getting the
  data, above); results from make_demo_data.py are placeholders for a
  code smoke test only.
- Repeated-CV and ensemble scripts (06, 07, 11, 15, 17) are
  computationally more expensive than the single-model scripts, since
  they retrain from scratch multiple times.
- Model architectures are deliberately implemented without a graph
  learning framework for auditability, trading a small amount of raw
  throughput for full transparency of the message-passing and attention
  logic.

## Citation

@article{hassan2026dilignn,
author = {Hassan, Youssef M. and El-Tantawi, Hala and Rabie Ali, Ibrahim and Attia, Mohamed S.},
title = {Validating and Correcting Graph Neural Network Attention for Drug-Induced Liver Injury Prediction},
journal = {Journal of Chemical Information and Modeling},
year = {2026},
note = {Accepted}
}


A DOI-linked CITATION.cff entry will be added on formal publication.

## License

Code is released under the MIT License (see LICENSE). This does not
extend to any third-party dataset you supply to run it.
