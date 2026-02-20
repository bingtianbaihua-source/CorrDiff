## Context
DisenMoOD 模块已实现（VAE/BranchDiffusion/Guidance），但未接入主训练与采样流程，导致潜空间扩散无法端到端运行。

## Goals / Non-Goals
- Goals:
  - 打通 VAE 预训练 → 潜空间扩散训练 → 解码采样的端到端流程
  - 保持配置可切换（DisenMoOD vs 传统 3D 扩散）
- Non-Goals:
  - 重新设计现有分子表示或替换现有数据管线
  - 引入新的第三方依赖

## Decisions
- Decision: 采用两阶段训练（先 VAE 后 BranchDiffusion）
- Decision: 在训练循环内以潜空间扩散为唯一主路径，3D 扩散路径在 DisenMoOD 模式下禁用
- Decision: 解码输出合同写入 `openspec/changes/add-disenmood-integration/design.md`，并以 `utils/reconstruct.reconstruct_from_generated` 为重建入口
- Alternatives considered: 端到端联合训练（因工程风险与不确定性，暂不作为默认路径）

## Risks / Trade-offs
- 风险: VAE 解码到 3D 可能不稳定 → 通过最小可用解码与评估脚本校验
- 取舍: 直接弃用旧 3D 扩散路径，减少维护成本但需要更完善的潜空间验证

## Decoder Output Contract
- Required:
  - `xyz`: float tensor, shape (N_atoms, 3)
  - `atomic_nums`: int tensor, shape (N_atoms,)
- Optional:
  - `aromatic`: bool/int tensor, shape (N_atoms,)
  - `atom_affinity`: float tensor, shape (N_atoms,)
- Reconstruction entrypoint: `utils.reconstruct.reconstruct_from_generated(xyz, atomic_nums, aromatic=None, atom_affinity=[])`

## Migration Plan
1. 增加 VAE 预训练入口与配置
2. 训练循环接入 BranchDiffusion，并新增 latent 编码路径
3. 补齐解码与采样输出
4. 文档化两阶段流程与切换开关

## Open Questions
- 3D 解码路径是否需要单独的坐标正则/几何约束？
- SMILES 导出是否通过现有 RDKit 依赖或已有转换工具链？
