## 1. Implementation
- [x] 1.1 Build offline attribute augmentation using ADMETModel for missing labels (datasets/pl_data.py or dataset builder) [#R1]
  - ACCEPT: 数据集构建流程（`datasets/pl_data.py` 或对应数据构建脚本）可对缺失属性调用 `ADMETModel.predict` 进行离线补齐，并缓存到数据集中。
  - TEST: SCOPE: CLI
    - When done, generate validation bundle under:
      auto_test_openspec/add-disenmood-framework/<run-folder>/
    - Run: auto_test_openspec/add-disenmood-framework/<run-folder>/run.sh (macOS/Linux) or run.bat (Windows)
    - Inputs: inputs/smiles_list.txt
      Outputs: outputs/augmented_labels.json
    - Verify: 输出包含最小属性集合中缺失项（PAMPA_NCATS, BBB_Martins, logP, Clearance_Microsome_AZ, hERG, affinity, QED, SA, AMES, lipinski）且与输入 SMILES 数量匹配
  BUNDLE (RUN #1): CODEX_CMD=codex exec --full-auto --skip-git-repo-check --model gpt-5.2 -c model_reasoning_effort=medium | SCOPE: CLI | VALIDATION_BUNDLE: auto_test_openspec/add-disenmood-framework/run-0001__task-1.1__ref-R1__20260219T115944Z | HOW_TO_RUN: run.sh/run.bat
  EVIDENCE (RUN #1): CODEX_CMD=codex exec --full-auto --skip-git-repo-check --model gpt-5.2 -c model_reasoning_effort=medium | SCOPE: CLI | VALIDATION_BUNDLE: auto_test_openspec/add-disenmood-framework/run-0001__task-1.1__ref-R1__20260219T115944Z | WORKER_STARTUP_LOG: auto_test_openspec/add-disenmood-framework/run-0001__task-1.1__ref-R1__20260219T115944Z/logs/worker_startup.txt | VALIDATED_CLI: bash auto_test_openspec/add-disenmood-framework/run-0001__task-1.1__ref-R1__20260219T115944Z/run.sh | EXIT_CODE: 1 | RESULT: FAIL
  REVIEW (RUN #1, Attempt #1): run.sh exits code 1 — inline Python heredoc uses Path(__file__).resolve().parent which resolves to CWD/project-root in stdin context, not the run-folder; FileNotFoundError: /Users/mac/Downloads/code/project/MoC/inputs/smiles_list.txt | EVIDENCE_PATH: auto_test_openspec/add-disenmood-framework/run-0001__task-1.1__ref-R1__20260219T115944Z/ | CMD: bash run.sh | EXIT_CODE: 1
  UNBLOCK GUIDANCE (RUN #1, Attempt #1): Root cause: `Path(__file__).resolve().parent` in a heredoc/stdin Python context resolves to CWD instead of the run-folder. Fix for Attempt #2: In run.sh, export `export BUNDLE_DIR="${HERE}"` before the heredoc block, then replace `root = Path(__file__).resolve().parent` with `root = Path(os.environ["BUNDLE_DIR"])` (add `import os` if missing). The augment_admet_labels.py implementation and actual feature code are correct; only the run.sh assertion path resolution needs fixing. Use a BRAND-NEW run-folder (run-0002__task-1.1__ref-R1__<timestamp>).
  BUNDLE (RUN #2): CODEX_CMD=codex exec --full-auto --skip-git-repo-check --model gpt-5.2 -c model_reasoning_effort=medium | SCOPE: CLI | VALIDATION_BUNDLE: auto_test_openspec/add-disenmood-framework/run-0002__task-1.1__ref-R1__20260219T121744Z | HOW_TO_RUN: run.sh/run.bat
  EVIDENCE (RUN #2): CODEX_CMD=codex exec --full-auto --skip-git-repo-check --model gpt-5.2 -c model_reasoning_effort=medium | SCOPE: CLI | VALIDATION_BUNDLE: auto_test_openspec/add-disenmood-framework/run-0002__task-1.1__ref-R1__20260219T121744Z | WORKER_STARTUP_LOG: auto_test_openspec/add-disenmood-framework/run-0002__task-1.1__ref-R1__20260219T121744Z/logs/worker_startup.txt | VALIDATED_CLI: bash auto_test_openspec/add-disenmood-framework/run-0002__task-1.1__ref-R1__20260219T121744Z/run.sh | EXIT_CODE: 0 | RESULT: PASS | GIT_COMMIT: (pending) | COMMIT_MSG: (pending) | DIFFSTAT: (pending)

- [ ] 1.2 Implement disentangled VAE with z_shared and {z_pi} losses (models/egnn.py, models/guide_model.py) [#R2]
  - ACCEPT: 训练前向输出包含 z_shared 与每个 z_pi（如 `models/egnn.py` / `models/guide_model.py` 中的编码输出），损失包含重建项与解耦正则项。
  - TEST: SCOPE: CLI
    - When done, generate validation bundle under:
      auto_test_openspec/add-disenmood-framework/<run-folder>/
    - Run: auto_test_openspec/add-disenmood-framework/<run-folder>/run.sh (macOS/Linux) or run.bat (Windows)
    - Inputs: inputs/mini_batch.pt
      Outputs: outputs/vae_losses.json
    - Verify: JSON 包含 recon_loss、tc_loss、mi_loss 字段且 z_shared/z_pi 形状符合配置

- [ ] 1.3 Implement backbone-branch latent diffusion for z_shared and z_pi (models/molopt_score_model.py) [#R3]
  - ACCEPT: 主干与各分支可独立前向/反向扩散（`models/molopt_score_model.py`），采样输出完整潜在集合。
  - TEST: SCOPE: CLI
    - When done, generate validation bundle under:
      auto_test_openspec/add-disenmood-framework/<run-folder>/
    - Run: auto_test_openspec/add-disenmood-framework/<run-folder>/run.sh (macOS/Linux) or run.bat (Windows)
    - Inputs: inputs/diffusion_config.json
      Outputs: outputs/sample_latents.npz
    - Verify: 输出包含 z_shared 与所有 z_pi，且维度与配置一致

- [ ] 1.4 Implement correlation matrix module from molecule/pocket graphs with gated branch interaction (models/egnn.py, models/guide_model.py) [#R4]
  - ACCEPT: 相关性矩阵 C 由分子图/口袋图预测（`models/egnn.py` / `models/guide_model.py`），并用于分支交互掩码控制。
  - TEST: SCOPE: CLI
    - When done, generate validation bundle under:
      auto_test_openspec/add-disenmood-framework/<run-folder>/
    - Run: auto_test_openspec/add-disenmood-framework/<run-folder>/run.sh (macOS/Linux) or run.bat (Windows)
    - Inputs: inputs/mini_batch.pt
    - Outputs: outputs/correlation_matrix.npy
    - Verify: C 为对称矩阵且对角为 1，掩码影响分支信息流（通过日志/统计验证）

- [ ] 1.5 Implement training-free energy guidance with branch-local gradients and freezing (models/guide_model.py, models/molopt_score_model.py) [#R5]
  - ACCEPT: 指定目标属性时仅对应分支被引导（`models/guide_model.py`/`models/molopt_score_model.py`），冻结属性不被更新。
  - TEST: SCOPE: CLI
    - When done, generate validation bundle under:
      auto_test_openspec/add-disenmood-framework/<run-folder>/
    - Run: auto_test_openspec/add-disenmood-framework/<run-folder>/run.sh (macOS/Linux) or run.bat (Windows)
    - Inputs: inputs/guidance_targets.json
      Outputs: outputs/guided_latents.npz
    - Verify: 目标分支变化显著，冻结分支在采样步内保持不变（容差内）

- [ ] 1.6 Implement correlation-aware multi-objective guidance and Pareto selection (models/molopt_score_model.py) [#R6]
  - ACCEPT: 高相关目标合并引导，低相关/冲突目标分离引导并输出帕累托候选（`models/molopt_score_model.py`）。
  - TEST: SCOPE: CLI
    - When done, generate validation bundle under:
      auto_test_openspec/add-disenmood-framework/<run-folder>/
    - Run: auto_test_openspec/add-disenmood-framework/<run-folder>/run.sh (macOS/Linux) or run.bat (Windows)
    - Inputs: inputs/multi_objective_case.json
      Outputs: outputs/pareto_set.json
    - Verify: 输出包含多解且满足帕累托非支配关系

- [ ] 1.7 Add expert predictor evaluation (R2) during training (scripts/train_diffusion_joint.py, models/molopt_score_model.py) [#R7]
  - ACCEPT: 训练过程中除 loss 外输出各属性专家网络的 R2 指标（记录在 `scripts/train_diffusion_joint.py` 的训练/验证日志中）。
  - TEST: SCOPE: CLI
    - When done, generate validation bundle under:
      auto_test_openspec/add-disenmood-framework/<run-folder>/
    - Run: auto_test_openspec/add-disenmood-framework/<run-folder>/run.sh (macOS/Linux) or run.bat (Windows)
    - Inputs: inputs/mini_batch.pt
      Outputs: outputs/expert_r2.json
    - Verify: 输出包含最小属性集合的 R2，缺失属性标记为 N/A

- [ ] 1.8 Add evaluation pipeline for HV/Sparsity, correlation error, intervention, docking (scripts/sample_diffusion_multi_for_pocket.py or new eval script) [#R8]
  - ACCEPT: 评估脚本可在 CrossDocked2020（或其 toy 子集）上输出 HV、Sparsity、相关性误差、解耦干预与对接评分。
  - TEST: SCOPE: CLI
    - When done, generate validation bundle under:
      auto_test_openspec/add-disenmood-framework/<run-folder>/
    - Run: auto_test_openspec/add-disenmood-framework/<run-folder>/run.sh (macOS/Linux) or run.bat (Windows)
    - Inputs: inputs/eval_config.json
      Outputs: outputs/eval_report.json
    - Verify: 报告包含 HV、Sparsity、相关性误差、干预实验结果与对接评分字段
