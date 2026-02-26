import torch
from rdkit import Chem
import sys
import os
import pickle
from tqdm import tqdm
import numpy as np
import torch
import torch.utils.tensorboard
from sklearn.metrics import roc_auc_score
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm
from rdkit.Chem.QED import qed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from utils.evaluation import scoring_func
from utils.evaluation.sascorer import compute_sa_score

lmdb_path = '/data/zhoujingyuan/kgdiff_20240314/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb'
trainset_path = '/data/zhoujingyuan/kgdiff_20240314/crossdocked_v1.1_rmsd1.0_pocket10'
testset_path = '/data/zhoujingyuan/kgdiff_20240314/test_set'
result_path = '/data/zhoujingyuan/kgdiff_20240314'
pt_path = '/data/zhoujingyuan/kgdiff_20240314/crossdocked_pocket10_pose_split.pt'

def generate_chemset_qed():
    index_path = '/data/zhoujingyuan/kgdiff_20240314/crossdocked_v1.1_rmsd1.0_pocket10/index.pkl'
    new_index_path = '/data/zhoujingyuan/kgdiff_20240314/crossdocked_v1.1_rmsd1.0_pocket10/index_with_qed.pkl'
    with open(index_path, 'rb') as f:
        index = pickle.load(f)
    
    new_index_list = []
    for i in tqdm(range(len(index))):
    # for i in tqdm(range(0, 2)):
        if index[i][0]: 
            ligand_path = os.path.join(os.path.dirname(index_path), index[i][1])
            mol = Chem.MolFromMolFile(ligand_path)
            if mol == None:
                mol = Chem.MolFromMol2File(ligand_path)
            qed_score = qed(mol)
            new_index_list.append((index[i][0], index[i][1], index[i][2], index[i][3], qed_score))
    
    with open(new_index_path, 'wb') as f:
        pickle.dump(new_index_list, f)

def generate_chemset_sa_logp():
    index_path = '/data/zhoujingyuan/kgdiff_20240314/crossdocked_v1.1_rmsd1.0_pocket10/index_with_qed.pkl'
    new_index_path = '/data/zhoujingyuan/kgdiff_20240314/crossdocked_v1.1_rmsd1.0_pocket10/index_with_chem_property.pkl'
    with open(index_path, 'rb') as f:
        index = pickle.load(f)
    
    new_index_list = []
    qed_list, sa_list, logp_list, lipinski_list = [], [], [], []
    for i in tqdm(range(len(index))):
    # for i in tqdm(range(0, 2)):
        if index[i][0]: 
            ligand_path = os.path.join(os.path.dirname(index_path), index[i][1])
            mol = Chem.MolFromMolFile(ligand_path)
            if mol == None:
                mol = Chem.MolFromMol2File(ligand_path)
            sa = compute_sa_score(mol)
            logp = Chem.Crippen.MolLogP(mol)
            lipinski = scoring_func.obey_lipinski(mol)

            qed_list.append(index[i][4])
            sa_list.append(sa)
            logp_list.append(logp)
            lipinski_list.append(lipinski)

            new_index_list.append((index[i][0], index[i][1], index[i][2], index[i][3], index[i][4], sa, logp)) 
            # pocket_fn, ligand_fn, protein_fn, rmsd, qed, sa, logp
    
    """print(len(qed_list), len(sa_list), len(logp_list))
    correlation = pearson_correlation_coefficient(qed_list, sa_list) # 0.4176
    print("Pearson correlation coefficient:", correlation)
    correlation = pearson_correlation_coefficient(qed_list, logp_list) # 0.3318
    print("Pearson correlation coefficient:", correlation)
    correlation = pearson_correlation_coefficient(logp_list, sa_list) # 0.3016
    print("Pearson correlation coefficient:", correlation)"""
    with open(new_index_path, 'wb') as f:
        pickle.dump(new_index_list, f)


def generate_chemset_lipinski():
    index_path = '/data/zhoujingyuan/kgdiff_20240314/crossdocked_v1.1_rmsd1.0_pocket10/index_with_chem_property.pkl'
    new_index_path = '/data/zhoujingyuan/kgdiff_20240314/crossdocked_v1.1_rmsd1.0_pocket10/index_with_all_property.pkl'
    with open(index_path, 'rb') as f:
        index = pickle.load(f)
    
    new_index_list = []
    min_logp = -15.230600357055664
    max_logp = 15.62600040435791
    min_lipinski = 0.0
    max_lipinski = 5.0
    for i in tqdm(range(len(index))):
    # for i in tqdm(range(0, 2)):
        if index[i][0]: 
            ligand_path = os.path.join(os.path.dirname(index_path), index[i][1])
            mol = Chem.MolFromMolFile(ligand_path)
            if mol == None:
                mol = Chem.MolFromMol2File(ligand_path)

            logp = Chem.Crippen.MolLogP(mol)
            norm_logp = (logp - min_logp) / (max_logp - min_logp) 
            lipinski = scoring_func.obey_lipinski(mol)
            norm_lipinski = (lipinski - min_lipinski) / (max_lipinski - min_lipinski) 

            new_index_list.append((index[i][0], index[i][1], index[i][2], index[i][3], index[i][4], index[i][5], norm_logp, norm_lipinski)) 
            # pocket_fn, ligand_fn, protein_fn, rmsd, qed, sa, logp, lipinski
    
    with open(new_index_path, 'wb') as f:
        pickle.dump(new_index_list, f)


def read_pkl():
    pkl_path = '/data/zhoujingyuan/kgdiff_20240314/crossdocked_v1.1_rmsd1.0_pocket10/index.pkl'
    with open(pkl_path, 'rb') as f:  # 'rb' 指定以二进制模式读取文件
        data = pickle.load(f)
    print(data[0])


def generate_chemset_new():

    train_ligand_set = []
    test_ligand_set = []

    # Load configs
    config_path = 'configs/training.yml'
    config = misc.load_config(config_path)
    misc.seed_all(config.train.seed)
   
    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ]
    if config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)

    # Datasets and loaders
    dataset, subsets = get_dataset(
        config=config.data,
        transform=transform
    )
    train_set, test_set = subsets['train'], subsets['test']
    # for i in tqdm(range(0, len(train_set))):
    #     train_ligand_set.append(train_set[i]['ligand_filename'])
    # for i in tqdm(range(0, len(test_set))):
    #     test_ligand_set.append(train_set[i]['ligand_filename'])
    # print(len(train_ligand_set))
    # print(len(test_ligand_set))
    train_chem_list = []
    test_chem_list = []
    failed_train_list = []
    failed_test_list = []
    for i in tqdm(range(0, len(train_set))):
    # for i in tqdm(range(0, 10)):
        ligand_filename = train_set[i]['ligand_filename']
        try:
            sdf = os.path.join(trainset_path, ligand_filename)
            mol = Chem.MolFromMolFile(sdf)
            chem_property = scoring_func.get_chem(mol)
            chem_property['ligand_filename'] = ligand_filename
            train_chem_list.append(chem_property)
        except:
            failed_train_list.append(ligand_filename)
            print(ligand_filename)
            continue

    for i in tqdm(range(0, len(test_set))):
    # for i in tqdm(range(0, 10)):
        ligand_filename = test_set[i]['ligand_filename']
        try:
            sdf = os.path.join(testset_path, ligand_filename)
            mol = Chem.MolFromMolFile(sdf)
            chem_property = scoring_func.get_chem(mol)
            chem_property['ligand_filename'] = ligand_filename
            test_chem_list.append(chem_property)
        except:
            failed_test_list.append(ligand_filename)
            print(ligand_filename)
            continue

    torch.save({'train': train_chem_list, 
                'test': test_chem_list}, os.path.join(result_path, 'crossdocked_property.pt'))


def generate_chemset_pdbbind2016():
    index_path = '/data/zhoujingyuan/targetdiff_data/pdbbind_v2016/refined-set/index.pkl'
    new_index_path = '/data/zhoujingyuan/targetdiff_data/pdbbind_v2016/refined-set/index_with_qed.pkl'
    with open(index_path, 'rb') as f:
        index = pickle.load(f)
    
    new_index_list = []
    for i in tqdm(range(len(index))):
    # for i in tqdm(range(0, 2)):
        ligand_path = index[i][1]
        mol = Chem.MolFromMolFile(ligand_path)
        if mol == None:
            mol = Chem.MolFromMol2File(ligand_path)
        qed_score = qed(mol)
        new_index_list.append((index[i][0], index[i][1], index[i][2], index[i][3], index[i][4], qed_score))
    
    with open(new_index_path, 'wb') as f:
        pickle.dump(new_index_list, f)

# 计算相关性
def pearson_correlation_coefficient(x, y):
    # 计算数据的均值
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # 计算协方差
    covariance = np.sum((x - mean_x) * (y - mean_y)) / len(x)
    
    # 计算标准差
    std_dev_x = np.sqrt(np.sum((x - mean_x)**2) / len(x))
    std_dev_y = np.sqrt(np.sum((y - mean_y)**2) / len(y))
    
    # 计算皮尔逊相关系数
    correlation_coefficient = covariance / (std_dev_x * std_dev_y)
    
    return correlation_coefficient


if __name__ == '__main__':
    generate_chemset_lipinski()

