import argparse
import os
import shutil
import time
import sys
sys.path.append("/data2/zhoujingyuan/MoC")

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm
from utils.data import PDBProtein
from datasets.pl_data import ProteinLigandData, torchify_dict

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D_Multi, log_sample_categorical
from utils.evaluation import atom_num

def pdb_to_pocket_data(pdb_path):
    pocket_dict = PDBProtein(pdb_path).to_dict_atom()
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict={
            'element': torch.empty([0, ], dtype=torch.long),
            'pos': torch.empty([0, 3], dtype=torch.float),
            'atom_feature': torch.empty([0, 8], dtype=torch.float),
            'bond_index': torch.empty([2, 0], dtype=torch.long),
            'bond_type': torch.empty([0, ], dtype=torch.long),
        }
    )

    return data


def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:  # step_i
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v


def sample_diffusion_ligand(model, data, num_samples, batch_size=16, device='cuda',
                            num_steps=None, center_pos_mode='protein',
                            sample_num_atoms='prior',
                            objective = ['affinity', 'qed'],
                            value_model=None, w=[1,1,1,1,1]):
    
    all_pred_pos, all_pred_v = [], []
    all_pred_property = {obj: [] for obj in objective}
    all_pred_property_traj = {obj: [] for obj in objective}
    all_pred_property_atom_traj ={obj: [] for obj in objective}
    all_pred_pos_traj, all_pred_v_traj = [], []
    all_pred_v0_traj, all_pred_vt_traj = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0
    for i in tqdm(range(num_batch)):
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)

        t1 = time.time()
        with torch.no_grad():
            batch_protein = batch.protein_element_batch
            if sample_num_atoms == 'prior':
                pocket_size = atom_num.get_space_size(batch.protein_pos.detach().cpu().numpy())
                ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)]
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'range':
                ligand_num_atoms = list(range(current_i + 1, current_i + n_data + 1))
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'ref':
                batch_ligand = batch.ligand_element_batch
                ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
            else:
                raise ValueError

            # init ligand pos
            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
            batch_center_pos = center_pos[batch_ligand]
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)

            # init ligand v
            uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
            init_ligand_v_prob = log_sample_categorical(uniform_logits)
            init_ligand_v = init_ligand_v_prob.argmax(dim=-1)

            r = model.sample_diffusion(
                objective = objective,
                value_model=value_model,
                w=w,
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch_protein,

                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v,
                batch_ligand=batch_ligand,
                num_steps=num_steps,
                center_pos_mode=center_pos_mode
            )
            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']

            # unbatch exp
            for obj in objective:
                all_pred_property[obj] += r[obj + '_traj'][-1]
                all_pred_property_traj[obj] += r[obj + '_traj']
                
            # unbatch pos
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                             range(n_data)]  # num_samples * [num_atoms_i, 3]

            all_step_pos = [[] for _ in range(n_data)]
            for p in ligand_pos_traj:  # step_i
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_step_pos = [np.stack(step_pos) for step_pos in
                            all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
            all_pred_pos_traj += [p for p in all_step_pos]

            # unbatch v
            ligand_v_array = ligand_v.cpu().numpy()
            all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

            all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
            all_pred_v_traj += [v for v in all_step_v]
            all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
            all_pred_v0_traj += [v for v in all_step_v0]
            all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
            all_pred_vt_traj += [v for v in all_step_vt]

            for obj in objective:
                all_step_property_atom = unbatch_v_traj(r[obj+'_atom_traj'], n_data, ligand_cum_atoms)
                all_pred_property_atom_traj[obj] += [v for v in all_step_property_atom]
            
            
        t2 = time.time()
        time_list.append(t2 - t1)
        current_i += n_data

    r = {
        'all_pred_pos': all_pred_pos,
        'all_pred_v': all_pred_v, 
        'all_pred_pos_traj': all_pred_pos_traj,
        'all_pred_v_traj': all_pred_v_traj,
        'all_pred_v0_traj': all_pred_v0_traj,
        'all_pred_vt_traj': all_pred_vt_traj,
        'time_list': time_list
    }
        
    for obj in objective:
        all_pred_property[obj] = torch.stack(all_pred_property[obj],dim=0).numpy()
        all_pred_property_traj[obj] = torch.stack(all_pred_property_traj[obj],dim=0).numpy()
        r.update({
            'all_pred_'+obj: all_pred_property[obj],
            'all_pred_'+obj+'_traj': all_pred_property_traj[obj],
            'all_pred_'+obj+'_atom_traj': all_pred_property_atom_traj[obj]
        })
        
    return r

def list_of_strings(arg):
    return arg.split(',')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/data2/zhoujingyuan/MoC/configs/sampling.yml')
    parser.add_argument('--pdb_path', type=str, default='/data2/zhoujingyuan/diffsbdd/testset/1BXM_bio1_ERG:A:99_pocket10.pdb')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=2) # 4
    parser.add_argument('--objective', type=list_of_strings, default='lipinski,qed,logp,affinity,sa')  # 'lipinski,qed,logp,affinity,sa'
    parser.add_argument('--w1', type=float, default=1.)
    parser.add_argument('--w2', type=float, default=1.)
    parser.add_argument('--w3', type=float, default=1.)
    parser.add_argument('--w4', type=float, default=1.)
    parser.add_argument('--w5', type=float, default=1.)
    parser.add_argument('--result_path', type=str, default='/data2/zhoujingyuan/MoC')
    args = parser.parse_args()
        
    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
    logger = misc.get_logger('sampling', log_dir=result_path)

    # Load config
    config = misc.load_config(args.config)
    logger.info(config)
    misc.seed_all(config.sample.seed)

    # Load checkpoint
    ckpt = torch.load(config.model['all_ckpt'], map_location=args.device) # qed sa affinity lipinski logp
    
    logger.info(f"Training Config: {ckpt['config']}")
    logger.info(f"args: {args}")

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
    ])

    # Load dataset
    data = pdb_to_pocket_data(args.pdb_path)
    data = transform(data)
    logger.info(f'Load {args.pdb_path}')


    # Load model
    model = ScorePosNet3D_Multi(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt['model'])

    weight = [args.w1, args.w2, args.w3, args.w4, args.w5]
    r = sample_diffusion_ligand(
        model, data, config.sample.num_samples,
        batch_size=args.batch_size, device=args.device,
        num_steps=config.sample.num_steps,
        center_pos_mode=config.sample.center_pos_mode,
        sample_num_atoms=config.sample.sample_num_atoms,
        objective = args.objective,
        w = weight
    )
    
    result = {
        'data': data,
        'protein_filename': os.path.basename(args.pdb_path),
        'pred_ligand_pos': r['all_pred_pos'],
        'pred_ligand_v': r['all_pred_v'],
        'pred_ligand_pos_traj': r['all_pred_pos_traj'],
        'pred_ligand_v_traj': r['all_pred_v_traj'],
        'time': r['time_list']
    }
    for obj in args.objective:
        result.update({
            'pred_'+obj: r['all_pred_'+obj],
            'pred_'+obj+'_traj': r['all_pred_'+obj+'_traj'],
            'pred_'+obj+'_atom_traj': r['all_pred_'+obj+'_atom_traj']
        })

    logger.info('Sample done!')

    torch.save(result, os.path.join(result_path, f'result_{os.path.basename(args.pdb_path)}.pt'))

if __name__ == '__main__':
    main()
