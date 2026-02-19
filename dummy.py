# from datasets import PocketLigandPairDataset_Chem

# dataset = PocketLigandPairDataset_Chem('data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final_with_all_property.lmdb')
# print(len(dataset), dataset[0])

from admet_ai import ADMETModel

model = ADMETModel()
preds = model.predict(smiles="O(c1ccc(cc1)CCOC)CC(O)CNC(C)C")