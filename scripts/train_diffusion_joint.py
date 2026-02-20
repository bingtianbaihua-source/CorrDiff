import os
import sys
import argparse
import shutil
import json
import numpy as np
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_

sys.path.append(os.path.abspath("/data2/zhoujingyuan/MoC"))
import utils.misc as misc
from models.guide_model import DisentangledVAE
from models.molopt_score_model import BranchDiffusion


def get_pearsonr(y_true, y_pred):
    from scipy import stats
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return stats.pearsonr(y_true, y_pred)

def _make_toy_disenmood_batch(
    *,
    batch_size: int,
    n_protein_atoms: int,
    n_ligand_atoms: int,
    protein_feat_dim: int,
    ligand_feat_dim: int,
    device,
) -> dict:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    protein_pos = torch.randn(batch_size * n_protein_atoms, 3, device=device)
    ligand_pos = torch.randn(batch_size * n_ligand_atoms, 3, device=device)
    protein_feat = torch.randn(batch_size * n_protein_atoms, protein_feat_dim, device=device)
    ligand_feat = torch.randn(batch_size * n_ligand_atoms, ligand_feat_dim, device=device)

    batch_protein = (
        torch.arange(batch_size, device=device).repeat_interleave(n_protein_atoms).long()
    )
    batch_ligand = torch.arange(batch_size, device=device).repeat_interleave(n_ligand_atoms).long()

    return {
        "protein_pos": protein_pos,
        "protein_feat": protein_feat,
        "ligand_pos": ligand_pos,
        "ligand_feat": ligand_feat,
        "batch_protein": batch_protein,
        "batch_ligand": batch_ligand,
    }


def _latent_denoising_loss(*, score_net, x0: torch.Tensor, t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    noise = torch.randn_like(x0)
    x_t = x0 + sigma * noise
    pred = score_net(x_t, t)
    target = -(noise / sigma)
    return (pred - target).pow(2).mean()


def _run_disenmood_one_step(*, config, args, logger) -> dict:
    from utils.latent_cache import encode_batch_latents

    device = torch.device(args.device)
    seed = int(getattr(config.train, "seed", 0))
    misc.seed_all(seed)

    disenmood_cfg = getattr(getattr(config, "model", {}), "disenmood", {})
    vae_cfg = getattr(disenmood_cfg, "vae", {})
    bd_cfg = getattr(disenmood_cfg, "branch_diffusion", {})

    property_names = list(getattr(vae_cfg, "property_names", ["affinity", "qed", "sa"]))

    toy_cfg = getattr(config, "toy", {})
    protein_atom_feature_dim = int(getattr(toy_cfg, "protein_atom_feature_dim", 8))
    ligand_atom_feature_dim = int(getattr(toy_cfg, "ligand_atom_feature_dim", 8))
    n_protein_atoms = int(getattr(toy_cfg, "n_protein_atoms", 4))
    n_ligand_atoms = int(getattr(toy_cfg, "n_ligand_atoms", 5))
    batch_size = int(getattr(config.train, "batch_size", 2))

    vae = DisentangledVAE(
        vae_cfg,
        protein_atom_feature_dim=protein_atom_feature_dim,
        ligand_atom_feature_dim=ligand_atom_feature_dim,
    ).to(device)
    vae.eval()

    bd = BranchDiffusion(
        z_shared_dim=int(getattr(vae, "z_shared_dim", 32)),
        z_pi_dim=int(getattr(vae, "z_pi_dim", 16)),
        property_names=property_names,
        num_steps=int(getattr(bd_cfg, "num_steps", 50)),
        time_emb_dim=int(getattr(bd_cfg, "time_emb_dim", 64)),
        hidden_dim=int(getattr(bd_cfg, "hidden_dim", 128)),
        num_layers=int(getattr(bd_cfg, "num_layers", 2)),
        sigma_min=float(getattr(bd_cfg, "sigma_min", 0.01)),
        sigma_max=float(getattr(bd_cfg, "sigma_max", 1.0)),
        device=device,
    )

    optimizer = torch.optim.Adam(bd.parameters(), lr=float(getattr(config.train, "lr", 1e-3)))
    optimizer.zero_grad(set_to_none=True)

    batch = _make_toy_disenmood_batch(
        batch_size=batch_size,
        n_protein_atoms=n_protein_atoms,
        n_ligand_atoms=n_ligand_atoms,
        protein_feat_dim=protein_atom_feature_dim,
        ligand_feat_dim=ligand_atom_feature_dim,
        device=device,
    )

    with torch.no_grad():
        z_shared, z_pi_list = encode_batch_latents(
            vae=vae, batch=batch, property_names=property_names, device=device
        )

    num_steps = int(bd.num_steps)
    t = torch.randint(low=0, high=num_steps, size=(batch_size,), device=device, dtype=torch.long)
    sigma = bd.sigmas[t].view(-1, 1).to(device)

    loss_shared = _latent_denoising_loss(score_net=bd.backbone, x0=z_shared, t=t, sigma=sigma)
    loss_pi = []
    for name, zpi in zip(property_names, z_pi_list):
        loss_pi.append(_latent_denoising_loss(score_net=bd.branches[name], x0=zpi, t=t, sigma=sigma))
    if len(loss_pi) > 0:
        loss_pi = torch.stack(loss_pi).mean()
    else:
        loss_pi = loss_shared.new_zeros(())

    latent_diffusion_loss = loss_shared + loss_pi
    latent_diffusion_loss.backward()
    optimizer.step()

    logger.info("Skipping 3D coordinate diffusion (DisenMoOD mode)")
    logger.info(
        "DisenMoOD mode: using BranchDiffusion for latent diffusion, skipping 3D coordinate noise"
    )
    print("Skipping 3D coordinate diffusion (DisenMoOD mode)")
    print("DisenMoOD mode: using BranchDiffusion for latent diffusion, skipping 3D coordinate noise")

    metrics = {
        "mode": "disenmood",
        "latent_diffusion_loss": float(latent_diffusion_loss.detach().cpu().item()),
    }

    metrics_path = getattr(config.train, "step_metrics_path", "outputs/train_step_metrics.json")
    metrics_path = os.path.abspath(metrics_path)
    os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    logger.info("Wrote step metrics JSON: %s", metrics_path)
    return metrics


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/data2/zhoujingyuan/MoC/configs/joint_training.yml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--logdir', type=str, default='/data2/zhoujingyuan/MoC/logs/train_multi_ckpt')
    parser.add_argument('--ckpt', type=str, default='')
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
    disenmood_mode = bool(getattr(config, "disenmood_mode", False))

    if disenmood_mode:
        logger = misc.get_logger("train_disenmood", log_dir=None)
        _run_disenmood_one_step(config=config, args=args, logger=logger)
        return

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score
    from torch_geometric.loader import DataLoader
    from torch_geometric.transforms import Compose
    from tqdm.auto import tqdm

    import utils.train as utils_train
    import utils.transforms as trans

    from datasets import get_dataset
    from datasets.pl_data import FOLLOW_BATCH
    from models.molopt_score_model import MINIMAL_PROPERTY_SET, ScorePosNet3D_Multi, compute_r2

    def _compute_expert_r2_from_batch(*, batch, results) -> dict:
        r2: dict = {k: "N/A" for k in MINIMAL_PROPERTY_SET}

        mapping = {
            "affinity": ("affinity", "pred_exp"),
            "QED": ("qed", "pred_qed"),
            "SA": ("sa", "pred_sa"),
            "lipinski": ("lipinski", "pred_lipinski"),
            "logP": ("logp", "pred_logp"),
        }
        for prop, (true_attr, pred_key) in mapping.items():
            if not hasattr(batch, true_attr):
                continue
            if pred_key not in results:
                continue
            y_true = getattr(batch, true_attr)
            y_pred = results[pred_key]
            try:
                v = compute_r2(y_true, y_pred)
            except Exception:
                v = None
            r2[prop] = "N/A" if v is None else float(v)

        return r2

    def _format_expert_r2(r2: dict) -> str:
        parts = []
        for k in MINIMAL_PROPERTY_SET:
            v = r2.get(k, "N/A")
            if isinstance(v, (float, int, np.floating, np.integer)):
                parts.append(f"{k}={float(v):.4f}")
            else:
                parts.append(f"{k}=N/A")
        return "{" + ", ".join(parts) + "}"

    misc.seed_all(config.train.seed)

    # Logging
    log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = misc.get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info('Objective: affinity, sa, qed')
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
    train_iterator = utils_train.inf_iterator(DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    ))
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                            follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys)

    test_loader = DataLoader(test_set, config.train.batch_size, shuffle=False,
                            follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys)
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
        # optimizer.load_state_dict(ckpt['optimizer'])
        # scheduler.load_state_dict(ckpt['scheduler'])
        # start_it = ckpt['iteration']

    
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


    def train(it):
        model.train()
        optimizer.zero_grad()
        last_batch = None
        last_results = None
        for _ in range(config.train.n_acc_batch):
            batch = next(train_iterator).to(args.device)
            last_batch = batch

            protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std
            gt_protein_pos = batch.protein_pos + protein_noise

            results = model.get_diffusion_loss(
                protein_pos=gt_protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                affinity=batch.affinity.float(),
                qed = batch.qed.float(),
                sa = batch.sa.float(),
                lipinski = batch.lipinski.float(),
                logp = batch.logp.float(),
                batch_protein=batch.protein_element_batch,

                ligand_pos=batch.ligand_pos,
                ligand_v=batch.ligand_atom_feature_full,
                batch_ligand=batch.ligand_element_batch
            )
            last_results = results
            if args.value_only:
                results['loss'] = results['loss_exp']
                
            loss, loss_pos, loss_v, loss_exp, loss_sa, loss_qed, loss_lipinski, loss_logp = results['loss'], results['loss_pos'], results['loss_v'], results['loss_exp'], results['loss_sa'], results['loss_qed'], results['loss_lipinski'], results['loss_logp'] 
            loss = loss / config.train.n_acc_batch
            loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()

        if it % args.train_report_iter == 0:
            expert_r2 = (
                _compute_expert_r2_from_batch(batch=last_batch, results=last_results)
                if last_batch is not None and last_results is not None
                else {k: "N/A" for k in MINIMAL_PROPERTY_SET}
            )
            logger.info(
                '[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f | exp %.6f | sa %.6f | qed %.6f | lipinski %.6f | logp %.6f) | Lr: %.6f | Grad Norm: %.6f' % (
                    it, loss, loss_pos, loss_v, loss_exp, loss_sa, loss_qed, loss_lipinski, loss_logp, optimizer.param_groups[0]['lr'], orig_grad_norm
                )
            )
            logger.info('[Train] Expert R2 %s' % _format_expert_r2(expert_r2))
            for k, v in results.items():
                if torch.is_tensor(v) and v.squeeze().ndim == 0:
                    writer.add_scalar(f'train/{k}', v, it)
            for prop, v in expert_r2.items():
                if isinstance(v, (float, int, np.floating, np.integer)):
                    writer.add_scalar(f"train/r2_{prop}", float(v), it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            writer.add_scalar('train/grad', orig_grad_norm, it)
            writer.flush()


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

        expert_r2 = {k: "N/A" for k in MINIMAL_PROPERTY_SET}
        if len(all_true_exp) > 0:
            v = compute_r2(np.concatenate(all_true_exp, axis=0), np.concatenate(all_pred_exp, axis=0))
            expert_r2["affinity"] = "N/A" if v is None else float(v)
        if len(all_true_sa) > 0:
            v = compute_r2(np.concatenate(all_true_sa, axis=0), np.concatenate(all_pred_sa, axis=0))
            expert_r2["SA"] = "N/A" if v is None else float(v)
        if len(all_true_qed) > 0:
            v = compute_r2(np.concatenate(all_true_qed, axis=0), np.concatenate(all_pred_qed, axis=0))
            expert_r2["QED"] = "N/A" if v is None else float(v)
        if len(all_true_lipinski) > 0:
            v = compute_r2(np.concatenate(all_true_lipinski, axis=0), np.concatenate(all_pred_lipinski, axis=0))
            expert_r2["lipinski"] = "N/A" if v is None else float(v)
        if len(all_true_logp) > 0:
            v = compute_r2(np.concatenate(all_true_logp, axis=0), np.concatenate(all_pred_logp, axis=0))
            expert_r2["logP"] = "N/A" if v is None else float(v)
        
        
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
        logger.info('[Validate] Expert R2 %s' % _format_expert_r2(expert_r2))
        writer.add_scalar('val/loss', avg_loss, it)
        writer.add_scalar('val/loss_pos', avg_loss_pos, it)
        writer.add_scalar('val/loss_v', avg_loss_v, it)
        writer.add_scalar('val/loss_exp', avg_loss_exp, it)
        writer.add_scalar('val/loss_sa', avg_loss_sa, it)
        writer.add_scalar('val/loss_qed', avg_loss_qed, it)
        writer.add_scalar('val/loss_lipinski', avg_loss_sa, it)
        writer.add_scalar('val/loss_logp', avg_loss_qed, it)

        writer.add_scalar('val/atom_auroc', atom_auroc, it)
        writer.add_scalar('val/pcc_affinity', exp_pearsonr[0], it)
        writer.add_scalar('val/pvalue_affinity', exp_pearsonr[1], it)
        writer.add_scalar('val/pcc_sa', sa_pearsonr[0], it)
        writer.add_scalar('val/pvalue_sa', sa_pearsonr[1], it)
        writer.add_scalar('val/pcc_qed', qed_pearsonr[0], it)
        writer.add_scalar('val/pvalue_qed', qed_pearsonr[1], it)
        writer.add_scalar('val/pcc_lipinski', lipinski_pearsonr[0], it)
        writer.add_scalar('val/pvalue_lipinski', lipinski_pearsonr[1], it)
        writer.add_scalar('val/pcc_logp', logp_pearsonr[0], it)
        writer.add_scalar('val/pvalue_logp', logp_pearsonr[1], it)
        for prop, v in expert_r2.items():
            if isinstance(v, (float, int, np.floating, np.integer)):
                writer.add_scalar(f"val/r2_{prop}", float(v), it)
        # fig = plt.figure(figsize=(12,12))
        
        writer.add_figure('val/pcc_affinity_fig', sns.lmplot(data=pd.DataFrame({
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
        writer.flush()
        
        if args.value_only:
            return avg_loss_exp
        
        return avg_loss

    def test(it):
        # fix time steps
        sum_loss, sum_loss_pos, sum_loss_v, sum_loss_exp, sum_loss_sa, sum_n = 0, 0, 0, 0, 0, 0
        all_pred_v, all_true_v , all_pred_exp, all_pred_sa, all_true_exp, all_true_sa= [], [], [], [], [], []
        with torch.no_grad():
            model.eval()
            for batch in tqdm(test_loader, desc='Test'):
                batch = batch.to(args.device)
                batch_size = batch.num_graphs
                for t in np.linspace(0, model.num_timesteps - 1, 10).astype(int):
                    time_step = torch.tensor([t] * batch_size).to(args.device)
                    results = model.get_diffusion_loss(
                        protein_pos=batch.protein_pos,
                        protein_v=batch.protein_atom_feature.float(),
                        affinity=batch.affinity.float(),
                        sa = batch.sa.float(),
                        batch_protein=batch.protein_element_batch,

                        ligand_pos=batch.ligand_pos,
                        ligand_v=batch.ligand_atom_feature_full,
                        batch_ligand=batch.ligand_element_batch,
                        time_step=time_step
                    )
                    loss, loss_pos, loss_v, loss_exp, loss_sa, pred_exp, pred_sa = results['loss'], results['loss_pos'], results['loss_v'], results['loss_exp'], results['loss_sa'], results['pred_exp'], results['pred_sa']

                    sum_loss += float(loss) * batch_size
                    sum_loss_pos += float(loss_pos) * batch_size
                    sum_loss_v += float(loss_v) * batch_size
                    sum_loss_exp += float(loss_exp) * batch_size
                    sum_loss_sa += float(loss_sa) * batch_size
                    sum_n += batch_size
                    all_pred_v.append(results['ligand_v_recon'].detach().cpu().numpy())
                    all_true_v.append(batch.ligand_atom_feature_full.detach().cpu().numpy())
                    all_pred_exp.append(pred_exp.detach().cpu().numpy())
                    all_true_exp.append(batch.affinity.float().detach().cpu().numpy())
                    all_pred_sa.append(pred_sa.detach().cpu().numpy())
                    all_true_sa.append(batch.sa.float().detach().cpu().numpy())

        avg_loss = sum_loss / sum_n
        avg_loss_pos = sum_loss_pos / sum_n
        avg_loss_v = sum_loss_v / sum_n
        avg_loss_exp = sum_loss_exp / sum_n
        avg_loss_sa = sum_loss_sa / sum_n
        atom_auroc = get_auroc(np.concatenate(all_true_v), np.concatenate(all_pred_v, axis=0),
                               feat_mode=config.data.transform.ligand_atom_mode)

        exp_pearsonr = get_pearsonr(np.concatenate(all_true_exp, axis=0), np.concatenate(all_pred_exp, axis=0))
        sa_pearsonr = get_pearsonr(np.concatenate(all_true_sa, axis=0), np.concatenate(all_pred_sa, axis=0))
        
        logger.info(
            '[Test] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f e-3 | Loss exp %.6f e-3 | Loss sa %.6f e-3 | Avg atom auroc %.6f' % (
                it, avg_loss, avg_loss_pos, avg_loss_v * 1000, avg_loss_exp * 1000, avg_loss_sa * 1000, atom_auroc
            )
        )
        writer.add_scalar('test/loss', avg_loss, it)
        writer.add_scalar('test/loss_pos', avg_loss_pos, it)
        writer.add_scalar('test/loss_v', avg_loss_v, it)
        writer.add_scalar('test/loss_exp', avg_loss_exp, it)
        writer.add_scalar('test/loss_sa', avg_loss_sa, it)
        writer.add_scalar('test/atom_auroc', atom_auroc, it)
        writer.add_scalar('test/pcc_affinity', exp_pearsonr[0], it)
        writer.add_scalar('test/pvalue_affinity', exp_pearsonr[1], it)
        writer.add_scalar('test/pcc_sa', sa_pearsonr[0], it)
        writer.add_scalar('test/pvalue_sa', sa_pearsonr[1], it)
        # fig = plt.figure(figsize=(12,12))
        
        writer.add_figure('test/pcc_affinity_fig', sns.lmplot(data=pd.DataFrame({
                'pred': np.concatenate(all_pred_exp, axis=0),
                'true': np.concatenate(all_true_exp, axis=0)
            }), x='pred', y='true').set(title='pcc %.6f | pvalue %.6f'%(exp_pearsonr[0], exp_pearsonr[1])).fig,it)
        writer.add_figure('test/pcc_affinity_fig', sns.lmplot(data=pd.DataFrame({
                'pred': np.concatenate(all_pred_sa, axis=0),
                'true': np.concatenate(all_true_sa, axis=0)
            }), x='pred', y='true').set(title='pcc %.6f | pvalue %.6f'%(sa_pearsonr[0], sa_pearsonr[1])).fig,it)
        writer.flush()
        
        if args.value_only:
            return avg_loss_exp
        
        return avg_loss
    
    try:
        best_loss, best_iter = None, None
        for it in range(start_it, config.train.max_iters):
            # with torch.autograd.detect_anomaly():
            train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                val_loss = validate(it)
                if config.data.name == 'pdbbind':
                    _ = test(it)
                if best_loss is None or val_loss < best_loss:
                    logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
                    best_loss, best_iter = val_loss, it
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                    }, ckpt_path)
                else:
                    logger.info(f'[Validate] Val loss is not improved. '
                                f'Best val loss: {best_loss:.6f} at iter {best_iter}')
    except KeyboardInterrupt:
        logger.info('Terminating...')
        
        
if __name__ == '__main__':
    main()
