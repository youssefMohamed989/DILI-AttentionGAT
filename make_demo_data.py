"""
make_demo_data.py
The real 1,111-compound reconciled DILIrank/DILIst 'gold-standard' dataset
(manuscript ref. 3) is not redistributed in this repository. This script
generates a small, clearly-synthetic set of drug-like SMILES with random
DILI labels so the full pipeline (01_data_prep.py onward) can be run
end-to-end as a smoke test. Do NOT use these numbers to interpret DILI
model performance -- they are placeholders for demonstrating the code
runs, not a reproduction of any result in the manuscript.

To reproduce the paper's actual numbers, obtain the curated gold-standard
DILI dataset yourself and place it at data/raw/gold_standard_dili.csv with
columns SMILES, DILI_label.
"""
import numpy as np
import pandas as pd

from common import RAW_DATASET_PATH, SEED

# A modest, diverse set of real, well-known drug-like SMILES (structures are
# public domain; DILI labels below are RANDOMLY assigned for demo purposes
# only and do not reflect real hepatotoxicity classifications).
DEMO_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",                      # aspirin
    "CC(=O)Nc1ccc(O)cc1",                          # acetaminophen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",                # caffeine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                  # ibuprofen
    "CC1=CC(=O)C=CC1=O",                           # simple quinone-like
    "c1ccc2c(c1)ccc(n2)N",                          # aminoquinoline-like
    "Nc1ccc(cc1)S(=O)(=O)N",                        # sulfanilamide
    "c1ccsc1",                                      # thiophene
    "c1ccoc1",                                      # furan
    "NNC(=O)c1ccccc1",                              # benzohydrazide
    "Clc1ccccc1",                                   # chlorobenzene
    "Brc1ccc(Br)cc1",                               # dibromobenzene
    "CC(=O)C=Cc1ccccc1",                            # enone
    "OC(=O)c1ccccc1",                               # benzoic acid
    "CCOC(=O)c1ccccc1N",                            # benzocaine
    "CC1=C(C(=O)Nc2ccccc21)C",                       # generic amide-ring
    "CN(C)CCOC(c1ccccc1)c1ccccc1",                   # diphenhydramine-like
    "Fc1ccc(cc1)C(=O)O",                             # fluorobenzoic acid
    "COc1ccc2[nH]c(nc2c1)C",                         # benzimidazole-like
    "CC(C)NCC(O)c1ccc(O)c(O)c1",                     # catecholamine-like
    "CC(C)(C)NCC(O)COc1ccccc1",                      # propranolol-like
    "CCN(CC)CCNC(=O)c1ccc(N)cc1",                     # procainamide-like
    "COc1cc2c(cc1OC)C(=O)C(CC2)Cc1ccccc1",           # generic bicyclic
    "c1ccc2c(c1)[nH]c1ccccc12",                       # carbazole
    "CCOC(=O)Nc1ccccc1",                             # phenylcarbamate
    "CC(=O)Nc1ccc(Cl)cc1",                           # chloroacetanilide
    "Nc1ccccc1",                                      # aniline
    "CC(C)Nc1ccccc1",                                 # N-alkylaniline
    "O=C(O)c1ccc(Br)cc1",                             # bromobenzoic acid
    "CCC(=O)Nc1ccc(cc1)C(=O)O",                       # generic
    "CC1=CC(=O)c2ccccc2C1=O",                          # naphthoquinone-like
    "CN1CCN(CC1)c1ccccc1",                            # phenylpiperazine
    "OC(=O)c1ccc(cc1)S(=O)(=O)N",                     # sulfa-acid
    "Clc1ccc(Cl)c(Cl)c1",                             # trichlorobenzene
    "c1ccc(cc1)Nn1cccc1",                              # phenylpyrrole-hydrazine-like
    "CC(=O)c1ccccc1",                                 # acetophenone
    "COC(=O)c1ccccc1O",                               # methyl salicylate
    "CCN(CC)C(=O)c1ccccc1",                            # benzamide
    "Nc1nc(N)nc(N)n1",                                 # melamine-like
    "CC(C)(C)c1ccc(O)cc1",                             # BHT-like phenol
    "O=C1CCC(=O)C=C1",                                 # cyclohexenedione
]


def main():
    rng = np.random.RandomState(SEED)
    labels = rng.randint(0, 2, size=len(DEMO_SMILES))
    df = pd.DataFrame({"SMILES": DEMO_SMILES, "DILI_label": labels})
    df.to_csv(RAW_DATASET_PATH, index=False)
    print(f"Wrote {len(df)} synthetic demo compounds to {RAW_DATASET_PATH}")
    print("NOTE: labels are random placeholders for a pipeline smoke test only.")


if __name__ == "__main__":
    main()
