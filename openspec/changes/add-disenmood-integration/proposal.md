# Change: Integrate DisenMoOD Latent Pipeline Into Main Training

## Why
现有 DisenMoOD 模块已实现但未接入主训练流程，导致潜空间扩散无法端到端训练与采样。本变更聚焦于把 VAE 预训练、潜码生成、BranchDiffusion 训练与解码回 3D 的关键链路打通。

## What Changes
- 新增 VAE 预训练脚本与配置入口
- 增加潜码提取/缓存或在线编码路径（离线为默认，在线为可选），支持 BranchDiffusion 训练
- 改造主训练循环，使 BranchDiffusion 作为潜空间扩散主流程
- 补齐 DisentangledVAE 解码到 3D 坐标与 SMILES 的输出能力
- 明确两阶段训练方案（先 VAE 后潜扩散）与运行入口
- 弃用旧 3D 直接扩散路径以避免混淆（DisenMoOD 模式下不保留回退）

## Impact
- Affected specs: disenmood-integration
- Affected code: scripts/train_diffusion_joint.py, scripts/train_vae.py (new), models/egnn.py, models/guide_model.py, models/molopt_score_model.py, datasets/pl_data.py (optional latent cache)
- External dependencies: none (优先复用现有依赖)
