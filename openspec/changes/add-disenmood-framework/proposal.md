# Change: Add DisenMOOD Disentangled Latent Diffusion With Training-Free Multi-Objective Guidance

## Why
当前扩散分子生成难以在不重新训练的情况下实现多目标优化，且多属性间存在冲突与相关性难以显式建模。本变更提出 DisenMOOD 框架以提升可控性与研究可验证性。

## What Changes
- 新增解耦潜在空间（z_shared + {z_pi}）与相应 VAE 训练目标
- 在解耦潜在空间上引入“主干-分支”扩散结构以独立生成共享与属性专属变量
- 增加属性相关性矩阵 C 的显式学习与分支交互掩码机制（分子图/口袋图输入）
- 引入训练自由的能量引导采样，支持多目标且可冻结约束属性
- 增加相关性感知的多目标引导与帕累托候选输出
- 增加评估流程（HV、Sparsity、相关性误差、解耦干预、对接评分）
- 数据集增强：使用 ADMETModel 离线补齐属性标签并缓存
- 训练监控：增加专家网络属性预测性能评估（R2）

## Impact
- Affected specs: disentangled-latent, branch-diffusion, correlation-modeling, training-free-guidance, disenmood-eval
- Affected code: datasets/pl_data.py, models/molopt_score_model.py, models/guide_model.py, models/egnn.py, scripts/sample_diffusion_multi_for_pocket.py, scripts/train_diffusion_joint.py
- External dependencies: admet_ai (ADMETModel)
