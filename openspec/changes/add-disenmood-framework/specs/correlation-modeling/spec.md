## ADDED Requirements
### Requirement: Correlation Matrix Modeling
系统 MUST 以分子图/口袋图为输入预测属性相关性矩阵 C。

#### Scenario: Predict correlation matrix from graphs
- **WHEN** 给定分子图与口袋图输入
- **THEN** 输出属性相关性矩阵 C

### Requirement: Gated Branch Interaction
系统 MUST 使用 C 作为掩码调节分支扩散的信息流，支持高相关属性的受控共享与低相关属性的隔离。

#### Scenario: Correlated attributes allow sharing
- **WHEN** C_ij 被预测为高相关
- **THEN** 分支 i 与 j 允许受控信息交换

#### Scenario: Conflicting attributes remain isolated
- **WHEN** C_ij 低或为负
- **THEN** 分支 i 与 j 的信息流被抑制或隔离
