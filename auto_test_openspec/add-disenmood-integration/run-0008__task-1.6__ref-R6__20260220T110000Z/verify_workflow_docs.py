from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


@dataclass(frozen=True)
class Paths:
    bundle_dir: Path
    repo_root: Path
    checklist_path: Path
    doc_path: Path
    summary_path: Path


def _resolve_paths() -> Paths:
    bundle_dir = Path(__file__).resolve().parent
    repo_root = (bundle_dir / "../../..").resolve()
    return Paths(
        bundle_dir=bundle_dir,
        repo_root=repo_root,
        checklist_path=bundle_dir / "inputs" / "workflow_checklist.txt",
        doc_path=repo_root / "docs" / "disenmood_workflow.md",
        summary_path=bundle_dir / "outputs" / "workflow_summary.txt",
    )


def _read_checklist(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing checklist file: {path}")
    items: list[str] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        items.append(s)
    if not items:
        raise ValueError(f"Checklist is empty: {path}")
    return items


def _validate_doc_contains(*, doc_text: str, items: list[str]) -> None:
    missing: list[str] = []
    for item in items:
        if item not in doc_text:
            missing.append(item)

    requirements: dict[str, list[str]] = {
        "stage_1_vae_pretraining": [
            "Stage 1",
            "scripts/train_vae.py",
            "configs/vae_toy_config.yml",
        ],
        "stage_2_latent_diffusion": [
            "Stage 2",
            "BranchDiffusion",
            "scripts/train_diffusion_joint.py",
        ],
        "disenmood_mode_switch": [
            "disenmood_mode: true",
        ],
        "legacy_3d_path_disabled": [
            "Legacy 3D diffusion path status in DisenMoOD mode: DISABLED",
            "no fallback",
        ],
    }

    missing_detail: list[str] = []
    for key in items:
        for needle in requirements.get(key, []):
            if needle not in doc_text:
                missing_detail.append(f"{key} => missing {needle!r}")

    if missing or missing_detail:
        lines = ["Documentation check failed."]
        if missing:
            lines.append("Missing checklist keys in docs/disenmood_workflow.md:")
            lines.extend([f"- {m}" for m in missing])
        if missing_detail:
            lines.append("Missing required phrases per key:")
            lines.extend([f"- {m}" for m in missing_detail])
        raise AssertionError("\n".join(lines))


def _write_summary(*, path: Path, items: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        [
            "DisenMoOD workflow summary (task 1.6 / R6)",
            "",
            "Stage 1: VAE pretraining",
            "- Script: scripts/train_vae.py",
            "- Config: configs/vae_toy_config.yml",
            "- Output: VAE checkpoint (.pt) with encoder/decoder weights",
            "",
            "Stage 2: BranchDiffusion latent diffusion training",
            "- Script: scripts/train_diffusion_joint.py",
            "- Config: configs/train_config_disenmood.yml",
            "- Uses latents from the DisentangledVAE encoder (z_shared + z_pi_*)",
            "",
            "DisenMoOD switch",
            "- Set disenmood_mode: true in config to enable the two-stage pipeline",
            "",
            "Legacy 3D diffusion path (coordinate diffusion)",
            "- Status when disenmood_mode: true => DISABLED (no fallback route)",
            "",
            "Checklist keys verified in docs/disenmood_workflow.md:",
            *[f"- {k}" for k in items],
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def main() -> int:
    p = _resolve_paths()
    items = _read_checklist(p.checklist_path)

    if not p.doc_path.exists():
        print(f"Missing documentation file: {p.doc_path}", file=sys.stderr)
        return 2
    doc_text = p.doc_path.read_text(encoding="utf-8")

    try:
        _validate_doc_contains(doc_text=doc_text, items=items)
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1

    _write_summary(path=p.summary_path, items=items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

