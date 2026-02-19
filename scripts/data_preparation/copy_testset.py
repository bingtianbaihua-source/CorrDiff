import os
import argparse
import random
import torch
from tqdm.auto import tqdm
import sys
import shutil
from glob import glob


sys.path.append("/home/zhoujingyuan/targetdiff/")
from torch.utils.data import Subset
from datasets.pl_pair_dataset import PocketLigandPairDataset


def get_chain_name(fn):
    return os.path.basename(fn)[:6]


def get_pdb_name(fn):
    return os.path.basename(fn)[:4]


def get_unique_pockets(dataset, raw_id, used_pdb, num_pockets):
    # only save first encountered id for unseen pdbs
    unique_id = []
    pdb_visited = set()
    for idx in tqdm(raw_id, 'Filter'):
        pdb_name = get_pdb_name(dataset[idx].ligand_filename)
        if pdb_name not in used_pdb and pdb_name not in pdb_visited:
            unique_id.append(idx)
            pdb_visited.add(pdb_name)

    print('Number of Pairs: %d' % len(unique_id))
    print('Number of PDBs:  %d' % len(pdb_visited))

    random.Random(args.seed).shuffle(unique_id)
    unique_id = unique_id[:num_pockets]
    print('Number of selected: %d' % len(unique_id))
    return unique_id, pdb_visited.union(used_pdb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/data/zhoujingyuan/kgdiff_20240314/crossdocked_v1.1_rmsd1.0_pocket10')
    parser.add_argument('--dest', type=str, default='/data/zhoujingyuan/kgdiff_20240314/crossdocked_pocket10_pose_split_chem_property.pt')
    parser.add_argument('--fixed_split', type=str, default='/data/zhoujingyuan/targetdiff_data/split_by_name.pt')
    parser.add_argument('--train', type=int, default=100000)
    parser.add_argument('--val', type=int, default=1000)
    parser.add_argument('--test', type=int, default=20000)
    parser.add_argument('--val_num_pockets', type=int, default=-1)
    parser.add_argument('--test_num_pockets', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2021)
    args = parser.parse_args()

    results_fn_list = glob(os.path.join('/data/zhoujingyuan/kgdiff_20240314/outputs/sample_mudm_sa_debug', '*result_*.pt'))
    for pt in results_fn_list:
        data = torch.load(pt)
        protein_fn = data['data']['protein_filename']
        ligand_fn = data['data']['ligand_filename']
        if not os.path.exists(os.path.join('/data/zhoujingyuan/kgdiff_20240314/test_set_chem', os.path.dirname(protein_fn))):
            os.makedirs(os.path.join('/data/zhoujingyuan/kgdiff_20240314/test_set_chem', os.path.dirname(protein_fn)))
        shutil.copy2(os.path.join(args.path, protein_fn), os.path.join('/data/zhoujingyuan/kgdiff_20240314/test_set_chem', os.path.dirname(protein_fn), os.path.basename(protein_fn)[0:10]+'.pdb'))
        shutil.copy2(os.path.join(args.path, ligand_fn), os.path.join('/data/zhoujingyuan/kgdiff_20240314/test_set_chem', os.path.dirname(protein_fn), os.path.basename(ligand_fn)))
    print(len(results_fn_list))
    print(len(os.listdir('/data/zhoujingyuan/kgdiff_20240314/test_set_chem')))
   