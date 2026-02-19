## ADDED Requirements
### Requirement: Backbone-Branch Latent Diffusion
系统 MUST 在解耦潜在空间训练主干-分支扩散模型，主干生成 z_shared，分支分别生成各 z_pi。

#### Scenario: Sampling generates shared and attribute latents
- **WHEN** 进行一次完整的逆扩散采样
- **THEN** 生成一组 z_shared 与 {z_pi}，可被解码为分子

### Requirement: Sampling Interface Compatibility
系统 MUST 保持采样入口与 `scripts/sample_diffusion_multi_for_pocket.py` 一致，支持指定口袋、目标属性与权重。

#### Scenario: Sampler consumes pocket inputs
- **WHEN** 传入 pocket 信息与目标属性配置
- **THEN** 采样过程可执行并返回对应潜变量或分子输出
