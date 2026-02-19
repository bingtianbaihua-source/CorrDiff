## ADDED Requirements
### Requirement: Multi-Objective Evaluation Metrics
系统 MUST 计算并报告多目标性能指标，包括 Hypervolume (HV) 与 Sparsity。

#### Scenario: Report multi-objective metrics
- **WHEN** 对生成分子集合执行评估
- **THEN** 输出 HV 与 Sparsity 指标

### Requirement: Correlation Modeling Accuracy
系统 MUST 计算预测相关性矩阵 C 与数据统计相关性的误差指标。

#### Scenario: Compare predicted vs empirical correlations
- **WHEN** 评估相关性建模模块
- **THEN** 输出相关性误差指标

### Requirement: Disentanglement Intervention Evaluation
系统 MUST 通过单变量干预验证解耦有效性，并报告对应的属性变化结果。

#### Scenario: Intervention evaluation report
- **WHEN** 对 z_pi 进行干预并生成样本
- **THEN** 输出干预实验结果与统计摘要

### Requirement: Docking-Based Affinity Evaluation
系统 MUST 对生成分子进行对接评分评估，并输出与目标蛋白的结合评分。

#### Scenario: Docking scores are reported
- **WHEN** 对生成分子运行对接评估
- **THEN** 输出对接评分结果
