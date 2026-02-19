import os
import sys
import argparse
import shutil
import numpy as np
import pandas as pd
import csv
import torch
import torch.utils.tensorboard
import seaborn as sns
# sns.set_theme(style="darkgrid")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy import stats
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch.nn.parallel import DataParallel
from tqdm.auto import tqdm
import sys
sys.path.append(os.path.abspath("/data2/zhoujingyuan/MoC"))
import utils.misc as misc
import utils.train as utils_train
import utils.transforms as trans

from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D_Multi


def get_pearsonr(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return stats.pearsonr(y_true, y_pred)

def draw_pcc(predict, ground_truth):
    data={'predict': predict, 'ground truth': ground_truth}

    exp_pearsonr = get_pearsonr(data['predict'], data['ground truth'])

    plot = sns.lmplot(data=data, x='predict', y='ground truth')

    plot.figure.set_dpi(300)

    # 获取当前图形对象
    fig = plt.gcf()

    # 设置图形大小
    fig.set_size_inches(8, 6)

    # 设置横轴字体大小
    plt.xticks(fontsize=12)

    # 设置纵轴字体大小
    plt.yticks(fontsize=12)

    # 设置横轴标题字体大小
    plt.xlabel('expert network', fontsize=16)

    # 设置纵轴标题字体大小
    plt.ylabel('ground truth', fontsize=16)

    # 显示图形
    plt.savefig('/home/zhoujingyuan/KGDiff-main/img/affinity_pcc.png')
    print(exp_pearsonr)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/home/zhoujingyuan/KGDiff-main/configs/joint_training.yml')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--logdir', type=str, default='/data/zhoujingyuan/kgdiff_20240314/train_multi_ckpt')
    parser.add_argument('--ckpt', type=str, default='/data/zhoujingyuan/kgdiff_20240314/train_multi_ckpt/joint_training_2024_05_10__10_55_33/checkpoints/881000.pt')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--value_only', action='store_true')
    parser.add_argument('--train_report_iter', type=int, default=200)
    args = parser.parse_args()

    # load ckpt
    if args.ckpt:
        print(f'loading {args.ckpt}...')
        ckpt = torch.load(args.ckpt, map_location=args.device)
        config = ckpt['config']
        # config = misc.load_config(args.config)
    else:
        # Load configs
        config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    # Logging
    log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = misc.get_logger('evaluate_expert', log_dir)
    # writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

    
    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
        trans.NormalizeVina(config.data.name)
    ]
    
    if config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)

    # Datasets and loaders
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(
        config=config.data,
        transform=transform,
    )
    
    if config.data.name == 'pl' or config.data.name == 'pl_chem':
        train_set, val_set, test_set = subsets['train'], subsets['test'], []
    elif config.data.name == 'pdbbind':
        train_set, val_set, test_set = subsets['train'], subsets['val'], subsets['test']
    else:
        raise ValueError
    logger.info(f'Training: {len(train_set)} Validation: {len(val_set)} Test: {len(test_set)}')

    collate_exclude_keys = ['ligand_nbh_list']
    # train_iterator = utils_train.inf_iterator(DataLoader(
    #     train_set,
    #     batch_size=config.train.batch_size,
    #     shuffle=True,
    #     num_workers=config.train.num_workers,
    #     follow_batch=FOLLOW_BATCH,
    #     exclude_keys=collate_exclude_keys
    # ))
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                            follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys)

    # test_loader = DataLoader(test_set, config.train.batch_size, shuffle=False,
    #                         follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys)
    # Model
    logger.info('Building model...')
    model = ScorePosNet3D_Multi(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)

    
    # print(model)
    logger.info(f'protein feature dim: {protein_featurizer.feature_dim} ligand feature dim: {ligand_featurizer.feature_dim}')
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    # Optimizer and scheduler
    optimizer = utils_train.get_optimizer(config.train.optimizer, model)
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)

    start_it = 0
    if args.ckpt:
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_it = ckpt['iteration']

    
    def get_auroc(y_true, y_pred, feat_mode):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        avg_auroc = 0.
        possible_classes = set(y_true)
        for c in possible_classes:
            auroc = roc_auc_score(y_true == c, y_pred[:, c])
            avg_auroc += auroc * np.sum(y_true == c)
            mapping = {
                'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
                'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
                'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL
            }
            logger.info(f'atom: {mapping[feat_mode][c]} \t auc roc: {auroc:.4f}')
        return avg_auroc / len(y_true)


    def validate(it):
        # fix time steps
        sum_loss, sum_loss_pos, sum_loss_v, sum_loss_exp, sum_loss_sa, sum_loss_qed, sum_loss_lipinski, sum_loss_logp, sum_n = 0, 0, 0, 0, 0, 0, 0, 0, 0
        all_pred_v, all_true_v , all_pred_exp, all_true_exp, all_pred_sa, all_true_sa, all_pred_qed, all_true_qed, all_pred_lipinski, all_true_lipinski, all_pred_logp, all_true_logp = [], [], [], [], [], [], [], [], [], [], [], []
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validate'):
                batch = batch.to(args.device)
                batch_size = batch.num_graphs
                for t in np.linspace(0, model.num_timesteps - 1, 10).astype(int):
                    time_step = torch.tensor([t] * batch_size).to(args.device)
                    results = model.get_diffusion_loss(
                        protein_pos=batch.protein_pos,
                        protein_v=batch.protein_atom_feature.float(),
                        affinity=batch.affinity.float(),
                        qed = batch.qed.float(),
                        sa=batch.sa.float(),
                        lipinski = batch.lipinski.float(),
                        logp = batch.logp.float(),
                        batch_protein=batch.protein_element_batch,

                        ligand_pos=batch.ligand_pos,
                        ligand_v=batch.ligand_atom_feature_full,
                        batch_ligand=batch.ligand_element_batch,
                        time_step=time_step
                    )
                    loss, loss_pos, loss_v, loss_exp, loss_sa, loss_qed, loss_lipinski, loss_logp, pred_exp, pred_sa, pred_qed, pred_lipinski, pred_logp = results['loss'], results['loss_pos'], results['loss_v'], results['loss_exp'], results['loss_sa'], results['loss_qed'], results['loss_lipinski'], results['loss_logp'], results['pred_exp'], results['pred_sa'], results['pred_qed'], results['pred_lipinski'], results['pred_logp'] 

                    sum_loss += float(loss) * batch_size
                    sum_loss_pos += float(loss_pos) * batch_size
                    sum_loss_v += float(loss_v) * batch_size
                    sum_loss_exp += float(loss_exp) * batch_size
                    sum_loss_sa += float(loss_sa) * batch_size
                    sum_loss_qed += float(loss_qed) * batch_size
                    sum_loss_lipinski += float(loss_lipinski) * batch_size
                    sum_loss_logp += float(loss_logp) * batch_size
                    sum_n += batch_size

                    all_pred_v.append(results['ligand_v_recon'].detach().cpu().numpy())
                    all_true_v.append(batch.ligand_atom_feature_full.detach().cpu().numpy())

                    all_pred_exp.append(pred_exp.detach().cpu().numpy())
                    all_pred_sa.append(pred_sa.detach().cpu().numpy())
                    all_pred_qed.append(pred_qed.detach().cpu().numpy())
                    all_pred_lipinski.append(pred_lipinski.detach().cpu().numpy())
                    all_pred_logp.append(pred_logp.detach().cpu().numpy())

                    all_true_exp.append(batch.affinity.float().detach().cpu().numpy())
                    all_true_sa.append(batch.sa.float().detach().cpu().numpy())
                    all_true_qed.append(batch.qed.float().detach().cpu().numpy())
                    all_true_lipinski.append(batch.lipinski.float().detach().cpu().numpy())
                    all_true_logp.append(batch.logp.float().detach().cpu().numpy())


        avg_loss = sum_loss / sum_n
        avg_loss_pos = sum_loss_pos / sum_n
        avg_loss_v = sum_loss_v / sum_n
        avg_loss_exp = sum_loss_exp / sum_n
        avg_loss_sa = sum_loss_sa / sum_n
        avg_loss_qed = sum_loss_qed / sum_n
        avg_loss_lipinski = sum_loss_lipinski / sum_n
        avg_loss_logp = sum_loss_logp / sum_n
        atom_auroc = get_auroc(np.concatenate(all_true_v), np.concatenate(all_pred_v, axis=0),
                               feat_mode=config.data.transform.ligand_atom_mode)

        exp_pearsonr = get_pearsonr(np.concatenate(all_true_exp, axis=0), np.concatenate(all_pred_exp, axis=0))
        sa_pearsonr = get_pearsonr(np.concatenate(all_true_sa, axis=0), np.concatenate(all_pred_sa, axis=0))
        qed_pearsonr = get_pearsonr(np.concatenate(all_true_qed, axis=0), np.concatenate(all_pred_qed, axis=0))
        lipinski_pearsonr = get_pearsonr(np.concatenate(all_true_lipinski, axis=0), np.concatenate(all_pred_lipinski, axis=0))
        logp_pearsonr = get_pearsonr(np.concatenate(all_true_logp, axis=0), np.concatenate(all_pred_logp, axis=0))
        
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        logger.info(
            '[Validate] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f e-3 | Loss exp %.6f e-3 | Loss sa %.6f e-3 | Loss qed %.6f e-3 | Loss lipinski %.6f e-3 | Loss logp %.6f e-3 | Avg atom auroc %.6f' % (
                it, avg_loss, avg_loss_pos, avg_loss_v * 1000, avg_loss_exp * 1000, avg_loss_sa * 1000, avg_loss_qed * 1000, avg_loss_lipinski * 1000, avg_loss_logp * 1000, atom_auroc
            )
        )
        all_pred_exp = np.concatenate(all_pred_exp, axis=0)
        all_true_exp = np.concatenate(all_true_exp, axis=0)
        all_pred_sa = np.concatenate(all_pred_sa, axis=0)
        all_true_sa = np.concatenate(all_true_sa, axis=0)
        all_pred_qed = np.concatenate(all_pred_qed, axis=0)
        all_true_qed = np.concatenate(all_true_qed, axis=0)
        all_pred_logp = np.concatenate(all_pred_logp, axis=0)
        all_true_logp = np.concatenate(all_true_logp, axis=0)
        all_pred_lipinski = np.concatenate(all_pred_lipinski, axis=0)
        all_true_lipinski = np.concatenate(all_true_lipinski, axis=0)
        
        with open('/home/zhoujingyuan/KGDiff-main/img/affinity.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['predict', 'ground truth'])  # 写入表头
            for i in range(len(all_pred_exp)):
                writer.writerow([all_pred_exp[i], all_true_exp[i]])
        with open('/home/zhoujingyuan/KGDiff-main/img/sa.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['predict', 'ground truth'])  # 写入表头
            for i in range(len(all_pred_sa)):
                writer.writerow([all_pred_sa[i], all_true_sa[i]])
        with open('/home/zhoujingyuan/KGDiff-main/img/qed.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['predict', 'ground truth'])  # 写入表头
            for i in range(len(all_pred_qed)):
                writer.writerow([all_pred_qed[i], all_true_qed[i]])
        with open('/home/zhoujingyuan/KGDiff-main/img/logp.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['predict', 'ground truth'])  # 写入表头
            for i in range(len(all_pred_logp)):
                writer.writerow([all_pred_logp[i], all_true_logp[i]])
        with open('/home/zhoujingyuan/KGDiff-main/img/lipinski.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['predict', 'ground truth'])  # 写入表头
            for i in range(len(all_pred_lipinski)):
                writer.writerow([all_pred_lipinski[i], all_true_lipinski[i]])
        # fig = plt.figure(figsize=(12,12))

        
        """writer.add_figure('val/pcc_affinity_fig', sns.lmplot(data=pd.DataFrame({
                'pred': np.concatenate(all_pred_exp, axis=0),
                'true': np.concatenate(all_true_exp, axis=0)
            }), x='pred', y='true').set(title='pcc %.6f | pvalue %.6f'%(exp_pearsonr[0], exp_pearsonr[1])).fig,it)
        writer.add_figure('val/pcc_sa_fig', sns.lmplot(data=pd.DataFrame({
                'pred': np.concatenate(all_pred_sa, axis=0),
                'true': np.concatenate(all_true_sa, axis=0)
            }), x='pred', y='true').set(title='pcc %.6f | pvalue %.6f'%(sa_pearsonr[0], sa_pearsonr[1])).fig,it)
        writer.add_figure('val/pcc_qed_fig', sns.lmplot(data=pd.DataFrame({
                'pred': np.concatenate(all_pred_qed, axis=0),
                'true': np.concatenate(all_true_qed, axis=0)
            }), x='pred', y='true').set(title='pcc %.6f | pvalue %.6f'%(qed_pearsonr[0], qed_pearsonr[1])).fig,it)
        writer.add_figure('val/pcc_lipinski_fig', sns.lmplot(data=pd.DataFrame({
                'pred': np.concatenate(all_pred_lipinski, axis=0),
                'true': np.concatenate(all_true_lipinski, axis=0)
            }), x='pred', y='true').set(title='pcc %.6f | pvalue %.6f'%(lipinski_pearsonr[0], lipinski_pearsonr[1])).fig,it)
        writer.add_figure('val/pcc_logp_fig', sns.lmplot(data=pd.DataFrame({
                'pred': np.concatenate(all_pred_logp, axis=0),
                'true': np.concatenate(all_true_logp, axis=0)
            }), x='pred', y='true').set(title='pcc %.6f | pvalue %.6f'%(logp_pearsonr[0], logp_pearsonr[1])).fig,it)
        writer.flush()"""
        
        if args.value_only:
            return avg_loss_exp
        
        return avg_loss


    # try:
    #     best_loss, best_iter = None, None
    #     for it in range(start_it, config.train.max_iters):
    #         # with torch.autograd.detect_anomaly():
    #         train(it)
    #         if it % config.train.val_freq == 0 or it == config.train.max_iters:
    #             val_loss = validate(it)
    #             if config.data.name == 'pdbbind':
    #                 _ = test(it)
    #             if best_loss is None or val_loss < best_loss:
    #                 logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
    #                 best_loss, best_iter = val_loss, it
    #                 ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
    #                 torch.save({
    #                     'config': config,
    #                     'model': model.state_dict(),
    #                     'optimizer': optimizer.state_dict(),
    #                     'scheduler': scheduler.state_dict(),
    #                     'iteration': it,
    #                 }, ckpt_path)
    #             else:
    #                 logger.info(f'[Validate] Val loss is not improved. '
    #                             f'Best val loss: {best_loss:.6f} at iter {best_iter}')
    # except KeyboardInterrupt:
    #     logger.info('Terminating...')
        
    val_loss = validate(start_it)
        
if __name__ == '__main__':
    main()
