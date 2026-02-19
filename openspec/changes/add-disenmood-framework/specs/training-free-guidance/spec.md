## ADDED Requirements
### Requirement: Training-Free Energy Guidance Per Attribute
系统 MUST 支持训练自由的能量引导采样，并将目标属性的梯度仅作用于对应 z_pi 分支；约束属性允许冻结分支采样。

#### Scenario: Guidance targets only selected attribute branches
- **WHEN** 用户指定优化属性集合与冻结属性集合
- **THEN** 采样过程仅对目标属性分支应用梯度，并冻结约束分支

### Requirement: Correlation-Aware Multi-Objective Guidance
系统 MUST 基于 C 对多属性引导进行合并或分离，并在冲突情况下输出帕累托候选集合。

#### Scenario: Conflicting objectives remain decoupled
- **WHEN** C_ij 低或为负且两个目标被同时优化
- **THEN** 引导过程保持分离，并输出帕累托候选集合
