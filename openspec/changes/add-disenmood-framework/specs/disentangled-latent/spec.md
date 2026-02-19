## ADDED Requirements
### Requirement: Disentangled Latent Representation
系统 MUST 训练解耦 VAE，将分子编码为共享潜变量 z_shared 与属性专属潜变量集合 {z_pi}，并使用总相关或互信息相关损失促进解耦。

#### Scenario: VAE training produces disentangled latents
- **WHEN** 给定 ProteinLigandData 批量输入进行前向与反向传播
- **THEN** 输出包含 z_shared 与每个 z_pi，并计算包含解耦正则项的损失

### Requirement: Latent Intervention Validation
系统 MUST 提供解耦干预验证流程，通过单独调整 z_pi 检查对应属性变化而其他属性保持稳定。

#### Scenario: Single-factor intervention
- **WHEN** 固定 z_shared 与其他 z_pj，仅对 z_pi 进行干预
- **THEN** 生成分子的目标属性发生变化，非目标属性变化在阈值内
