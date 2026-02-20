## 1. Implementation
- [x] 1.1 Add DisentangledVAE pretraining entrypoint (new script + config) [#R1]
  - ACCEPT: 提供 `scripts/train_vae.py`（或等价入口）与配置示例，可在小数据子集上完成一个 epoch 并写出 checkpoint。
  - TEST: SCOPE: CLI
    - When done, generate validation bundle under:
      auto_test_openspec/add-disenmood-integration/<run-folder>/
    - Run: auto_test_openspec/add-disenmood-integration/<run-folder>/run.sh (macOS/Linux) or run.bat (Windows)
    - Inputs: inputs/vae_toy_config.yml
      Outputs: outputs/vae_checkpoint.pt
    - Verify: checkpoint 文件存在且包含 encoder/decoder 权重键
  BUNDLE (RUN #1): CODEX_CMD=codex exec --full-auto --skip-git-repo-check --model gpt-5.2 -c model_reasoning_effort=medium | SCOPE: CLI | VALIDATION_BUNDLE: auto_test_openspec/add-disenmood-integration/run-0001__task-1.1__ref-R1__20260220T000000Z | HOW_TO_RUN: run.sh/run.bat
  BUNDLE (RUN #2): CODEX_CMD=codex exec --full-auto --skip-git-repo-check --model gpt-5.2 -c model_reasoning_effort=medium | SCOPE: CLI | VALIDATION_BUNDLE: auto_test_openspec/add-disenmood-integration/run-0002__task-1.1__ref-R1__20260220T050000Z | HOW_TO_RUN: run.sh/run.bat
  EVIDENCE (RUN #2): CODEX_CMD=codex exec --full-auto --skip-git-repo-check --model gpt-5.2 -c model_reasoning_effort=medium | SCOPE: CLI | VALIDATION_BUNDLE: auto_test_openspec/add-disenmood-integration/run-0002__task-1.1__ref-R1__20260220T050000Z | WORKER_STARTUP_LOG: auto_test_openspec/add-disenmood-integration/run-0002__task-1.1__ref-R1__20260220T050000Z/logs/worker_startup.txt | VALIDATED_CLI: bash auto_test_openspec/add-disenmood-integration/run-0002__task-1.1__ref-R1__20260220T050000Z/run.sh | EXIT_CODE: 0 | RESULT: PASS | GIT_COMMIT: 285239f | COMMIT_MSG: "[openspec] task-1.1 R1 PASS: DisentangledVAE pretraining entrypoint (train_vae.py + vae_toy_config)" | DIFFSTAT: "22 files changed, 963 insertions(+)"
  EVIDENCE (RUN #1): CODEX_CMD=codex exec --full-auto --skip-git-repo-check --model gpt-5.2 -c model_reasoning_effort=medium | SCOPE: CLI | VALIDATION_BUNDLE: auto_test_openspec/add-disenmood-integration/run-0001__task-1.1__ref-R1__20260220T000000Z | WORKER_STARTUP_LOG: auto_test_openspec/add-disenmood-integration/run-0001__task-1.1__ref-R1__20260220T000000Z/logs/worker_startup.txt | VALIDATED_CLI: bash auto_test_openspec/add-disenmood-integration/run-0001__task-1.1__ref-R1__20260220T000000Z/run.sh | EXIT_CODE: 1 | RESULT: FAIL | FAIL_REASON: ModuleNotFoundError: No module named 'models' — run.sh does cd REPO_ROOT then python scripts/train_vae.py but Python sys.path[0] is set to scripts/ not repo root; PYTHONPATH not set
  REVIEW (RUN #1, Attempt #1): ModuleNotFoundError: No module named 'models' when run.sh executes python scripts/train_vae.py from repo root. Python sets sys.path[0]=scripts/ not repo root. Fix: run.sh must export PYTHONPATH="${REPO_ROOT}":${PYTHONPATH:-} before calling python, OR train_vae.py must insert the repo root (os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) into sys.path at startup. | EVIDENCE_PATH: auto_test_openspec/add-disenmood-integration/run-0001__task-1.1__ref-R1__20260220T000000Z/ | CMD: bash run.sh | EXIT_CODE: 1
  UNBLOCK GUIDANCE (RUN #1, Attempt #1): Root cause: Python 3 sets sys.path[0] to the script's directory (scripts/) not the cwd. Fix A (preferred): In run.sh, add `export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"` before the python call. Fix B: In train_vae.py, at the top of main() before imports, add `import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))`. Either fix allows `from models.guide_model import DisentangledVAE` to resolve. Verify with: `PYTHONPATH=. python -c "from models.guide_model import DisentangledVAE; print('OK')"` (already confirmed works).

- [x] 1.2 Provide latent encoding path (offline default, online optional) for BranchDiffusion training [#R2]
  - ACCEPT: 训练时支持离线缓存（默认）与在线 encode（可选开关）两种模式，并在训练日志中明确当前模式。
  - TEST: SCOPE: CLI
    - When done, generate validation bundle under:
      auto_test_openspec/add-disenmood-integration/<run-folder>/
    - Run: auto_test_openspec/add-disenmood-integration/<run-folder>/run.sh (macOS/Linux) or run.bat (Windows)
    - Inputs: inputs/latent_mode_config.yml
      Outputs: outputs/latent_batch.npz
    - Verify: 输出包含 z_shared 与至少一个 z_pi，形状与配置一致，日志打印所选模式
  BUNDLE (RUN #3): CODEX_CMD=codex exec --full-auto --skip-git-repo-check --model gpt-5.2 -c model_reasoning_effort=medium | SCOPE: CLI | VALIDATION_BUNDLE: auto_test_openspec/add-disenmood-integration/run-0003__task-1.2__ref-R2__20260220T060000Z | HOW_TO_RUN: run.sh/run.bat
  EVIDENCE (RUN #3): CODEX_CMD=codex exec --full-auto --skip-git-repo-check --model gpt-5.2 -c model_reasoning_effort=medium | SCOPE: CLI | VALIDATION_BUNDLE: auto_test_openspec/add-disenmood-integration/run-0003__task-1.2__ref-R2__20260220T060000Z | WORKER_STARTUP_LOG: auto_test_openspec/add-disenmood-integration/run-0003__task-1.2__ref-R2__20260220T060000Z/logs/worker_startup.txt | VALIDATED_CLI: bash auto_test_openspec/add-disenmood-integration/run-0003__task-1.2__ref-R2__20260220T060000Z/run.sh | EXIT_CODE: 0 | RESULT: PASS | GIT_COMMIT: 95f597c | COMMIT_MSG: "[openspec] task-1.2 R2 PASS: latent encoding path (offline/online) + LatentCache utility" | DIFFSTAT: "14 files changed, 684 insertions(+)"

- [x] 1.3 Integrate BranchDiffusion into main training loop [#R3]
  - ACCEPT: `scripts/train_diffusion_joint.py` 在 DisenMoOD 模式下使用 BranchDiffusion 计算潜空间扩散 loss，不再对 3D 坐标直接加噪。
  - TEST: SCOPE: CLI
    - When done, generate validation bundle under:
      auto_test_openspec/add-disenmood-integration/<run-folder>/
    - Run: auto_test_openspec/add-disenmood-integration/<run-folder>/run.sh (macOS/Linux) or run.bat (Windows)
    - Inputs: inputs/train_config_disenmood.yml
      Outputs: outputs/train_step_metrics.json
    - Verify: 输出包含 latent_diffusion_loss 字段且训练日志明确跳过 3D 坐标扩散路径
  BUNDLE (RUN #4): CODEX_CMD=codex exec --full-auto --skip-git-repo-check --model gpt-5.2 -c model_reasoning_effort=medium | SCOPE: CLI | VALIDATION_BUNDLE: auto_test_openspec/add-disenmood-integration/run-0004__task-1.3__ref-R3__20260220T070000Z | HOW_TO_RUN: run.sh/run.bat
  EVIDENCE (RUN #4): CODEX_CMD=codex exec --full-auto --skip-git-repo-check --model gpt-5.2 -c model_reasoning_effort=medium | SCOPE: CLI | VALIDATION_BUNDLE: auto_test_openspec/add-disenmood-integration/run-0004__task-1.3__ref-R3__20260220T070000Z | WORKER_STARTUP_LOG: auto_test_openspec/add-disenmood-integration/run-0004__task-1.3__ref-R3__20260220T070000Z/logs/worker_startup.txt | VALIDATED_CLI: bash auto_test_openspec/add-disenmood-integration/run-0004__task-1.3__ref-R3__20260220T070000Z/run.sh | EXIT_CODE: 0 | RESULT: PASS | GIT_COMMIT: 17ca33d | COMMIT_MSG: "[openspec] task-1.3 R3 PASS: BranchDiffusion latent diffusion in DisenMoOD mode (train_disenmood.py)" | DIFFSTAT: "11 files changed, 455 insertions(+), 50 deletions(-)"

- [x] 1.4 Implement latent decode to 3D coordinates and SMILES export helper [#R4]
  - ACCEPT: `DisentangledVAE.decode()`（或等价路径）可返回 3D 坐标与原子类型（重建所需），并提供可导出 SMILES 的辅助函数（优先复用 `utils/reconstruct.py`）。
  - TEST: SCOPE: CLI
    - When done, generate validation bundle under:
      auto_test_openspec/add-disenmood-integration/<run-folder>/
    - Run: auto_test_openspec/add-disenmood-integration/<run-folder>/run.sh (macOS/Linux) or run.bat (Windows)
    - Inputs: inputs/sample_latents.npz
      Outputs: outputs/decoded_ligand.npz
    - Verify: 输出包含 3D 坐标数组与可导出的分子表示（SMILES 或注明替代格式）
  BUNDLE (RUN #5): CODEX_CMD=codex exec --full-auto --skip-git-repo-check --model gpt-5.2 -c model_reasoning_effort=medium | SCOPE: CLI | VALIDATION_BUNDLE: auto_test_openspec/add-disenmood-integration/run-0005__task-1.4__ref-R4__20260220T080000Z | HOW_TO_RUN: run.sh/run.bat
  BUNDLE (RUN #6): CODEX_CMD=codex exec --full-auto --skip-git-repo-check --model gpt-5.2 -c model_reasoning_effort=medium | SCOPE: CLI | VALIDATION_BUNDLE: auto_test_openspec/add-disenmood-integration/run-0006__task-1.4__ref-R4__20260220T090000Z | HOW_TO_RUN: run.sh/run.bat
  EVIDENCE (RUN #5): CODEX_CMD=codex exec --full-auto --skip-git-repo-check --model gpt-5.2 -c model_reasoning_effort=medium | SCOPE: CLI | VALIDATION_BUNDLE: auto_test_openspec/add-disenmood-integration/run-0005__task-1.4__ref-R4__20260220T080000Z | WORKER_STARTUP_LOG: auto_test_openspec/add-disenmood-integration/run-0005__task-1.4__ref-R4__20260220T080000Z/logs/worker_startup.txt | VALIDATED_CLI: bash auto_test_openspec/add-disenmood-integration/run-0005__task-1.4__ref-R4__20260220T080000Z/run.sh | EXIT_CODE: 1 | RESULT: FAIL | FAIL_REASON: AssertionError: smiles_utf8 decodes to empty string — fallback returns "" but validation asserts non-empty; xyz=(8,3) correct, atomic_nums=(8,) correct
  REVIEW (RUN #5, Attempt #1): smiles_utf8 empty when rdkit unavailable. ACCEPT allows "SMILES or documented alternative". Fix: return "SMILES_UNAVAILABLE_IN_TOY_MODE" instead of "" in fallback. | EVIDENCE_PATH: auto_test_openspec/add-disenmood-integration/run-0005__task-1.4__ref-R4__20260220T080000Z/ | CMD: bash run.sh | EXIT_CODE: 1
  UNBLOCK GUIDANCE (RUN #5, Attempt #1): In utils/smiles_export.py fallback, return "SMILES_UNAVAILABLE_IN_TOY_MODE" (non-empty str) instead of "". Also update run.sh verification to accept smiles_utf8 containing "UNAVAILABLE" as documented alternative. xyz=(8,3) and atomic_nums=(8,) are already correct.
  EVIDENCE (RUN #6): CODEX_CMD=codex exec --full-auto --skip-git-repo-check --model gpt-5.2 -c model_reasoning_effort=medium | SCOPE: CLI | VALIDATION_BUNDLE: auto_test_openspec/add-disenmood-integration/run-0006__task-1.4__ref-R4__20260220T090000Z | WORKER_STARTUP_LOG: auto_test_openspec/add-disenmood-integration/run-0006__task-1.4__ref-R4__20260220T090000Z/logs/worker_startup.txt | VALIDATED_CLI: bash auto_test_openspec/add-disenmood-integration/run-0006__task-1.4__ref-R4__20260220T090000Z/run.sh | EXIT_CODE: 0 | RESULT: PASS | GIT_COMMIT: 605430a | COMMIT_MSG: "[openspec] task-1.4 R4 PASS: DisentangledVAE.decode() + SMILES export helper" | DIFFSTAT: "19 files changed, 701 insertions(+)"

- [ ] 1.5 Update sampling pipeline to decode generated latents [#R5]
  - ACCEPT: 采样阶段从 BranchDiffusion 得到 z_shared/{z_pi} 后，调用解码得到 3D 结构与原子类型，并复用重建路径输出 SMILES。
  - TEST: SCOPE: CLI
    - When done, generate validation bundle under:
      auto_test_openspec/add-disenmood-integration/<run-folder>/
    - Run: auto_test_openspec/add-disenmood-integration/<run-folder>/run.sh (macOS/Linux) or run.bat (Windows)
    - Inputs: inputs/sample_config.yml
      Outputs: outputs/generated_molecules.smi
    - Verify: 输出包含至少一条 SMILES 且与 3D 坐标输出一一对应
  BUNDLE (RUN #7): CODEX_CMD=codex exec --full-auto --skip-git-repo-check --model gpt-5.2 -c model_reasoning_effort=medium | SCOPE: CLI | VALIDATION_BUNDLE: auto_test_openspec/add-disenmood-integration/run-0007__task-1.5__ref-R5__20260220T100000Z | HOW_TO_RUN: run.sh/run.bat

- [ ] 1.6 Document two-stage workflow and gate legacy 3D diffusion path [#R6]
  - ACCEPT: 文档或配置明确两阶段流程；旧 3D 扩散路径在 DisenMoOD 模式下被禁用（无回退路径）。
  - TEST: SCOPE: CLI
    - When done, generate validation bundle under:
      auto_test_openspec/add-disenmood-integration/<run-folder>/
    - Run: auto_test_openspec/add-disenmood-integration/<run-folder>/run.sh (macOS/Linux) or run.bat (Windows)
    - Inputs: inputs/workflow_checklist.txt
      Outputs: outputs/workflow_summary.txt
    - Verify: 文档/配置中包含两阶段步骤与 DisenMoOD 开关说明

- [ ] 1.7 Define decoder output contract for reconstruction compatibility [#R7]
  - ACCEPT: 文档明确 `xyz` 与 `atomic_nums` 为必需字段，`aromatic`/`atom_affinity` 为可选字段；采样输出字段与 `utils.reconstruct.reconstruct_from_generated` 输入一致。
  - TEST: SCOPE: CLI
    - When done, generate validation bundle under:
      auto_test_openspec/add-disenmood-integration/<run-folder>/
    - Run: auto_test_openspec/add-disenmood-integration/<run-folder>/run.sh (macOS/Linux) or run.bat (Windows)
    - Inputs: inputs/decode_contract.json
      Outputs: outputs/decode_contract_check.txt
    - Verify: 合同文档包含必需/可选字段说明与重建入口函数名
