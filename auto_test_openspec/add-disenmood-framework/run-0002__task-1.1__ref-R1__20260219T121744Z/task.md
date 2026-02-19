# TASK 1.1 (CHANGE-ID: add-disenmood-framework, REF: R1)

## Task description
Build offline attribute augmentation for missing labels using `ADMETModel.predict`, and cache results back into the dataset building flow.

## ACCEPT criteria
- Dataset building flow can call `ADMETModel.predict` for offline completion of missing attributes and cache them into the dataset.
- Minimum attribute set: `PAMPA_NCATS`, `BBB_Martins`, `logP`, `Clearance_Microsome_AZ`, `hERG`, `affinity`, `QED`, `SA`, `AMES`, `lipinski`.

## How to run
- macOS/Linux: `./run.sh`
- Windows: `run.bat`

## Machine-decidable pass/fail criteria
PASS if `outputs/augmented_labels.json` is a list, has same count as `inputs/smiles_list.txt`, and each entry contains all 10 keys: PAMPA_NCATS, BBB_Martins, logP, Clearance_Microsome_AZ, hERG, affinity, QED, SA, AMES, lipinski

