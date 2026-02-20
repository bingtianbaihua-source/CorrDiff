#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "models").is_dir() and (p / "utils").is_dir() and (p / "openspec").is_dir():
            return p
    raise RuntimeError(f"Could not find repo root from: {start}")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _base_name(expr: ast.AST) -> Optional[str]:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Subscript):
        return _base_name(expr.value)
    if isinstance(expr, ast.Attribute):
        return _base_name(expr.value)
    return None


def _collect_return_names(func: ast.FunctionDef) -> list[set[str]]:
    returns: list[set[str]] = []
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and node.value is not None:
            names: set[str] = set()
            value = node.value
            if isinstance(value, (ast.Tuple, ast.List)):
                elts = value.elts
            else:
                elts = [value]
            for e in elts:
                n = _base_name(e)
                if n is not None:
                    names.add(n)
            returns.append(names)
    return returns


def _find_method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(_read_text(path), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise RuntimeError(f"Could not find {class_name}.{method_name} in {path}")


@dataclass(frozen=True)
class FnSig:
    name: str
    args: list[str]
    defaults_count: int


def _find_function_signature(path: Path, fn_name: str) -> FnSig:
    tree = ast.parse(_read_text(path), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            args = [a.arg for a in node.args.args]
            defaults_count = len(node.args.defaults or [])
            return FnSig(name=node.name, args=args, defaults_count=defaults_count)
    raise RuntimeError(f"Could not find function def {fn_name}(...) in {path}")


def _fmt_list(items: Iterable[str]) -> str:
    return ", ".join(items)


def main() -> int:
    here = Path(__file__).resolve()
    run_dir = here.parent
    repo_root = _find_repo_root(run_dir)

    in_path = run_dir / "inputs" / "decode_contract.json"
    out_path = run_dir / "outputs" / "decode_contract_check.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    schema = json.loads(in_path.read_text(encoding="utf-8"))
    required = list(schema.get("required", []))
    optional = list(schema.get("optional", []))
    reconstruction_fn = str(schema.get("reconstruction_fn", "")).strip()
    if not required or not reconstruction_fn:
        raise RuntimeError("decode_contract.json must include non-empty 'required' and 'reconstruction_fn'")

    # 1) Confirm DisentangledVAE.decode returns xyz + atomic_nums (static analysis).
    guide_model_py = repo_root / "models" / "guide_model.py"
    decode_fn = _find_method(guide_model_py, class_name="DisentangledVAE", method_name="decode")
    return_sets = _collect_return_names(decode_fn)
    missing_required: list[str] = []
    for field in required:
        if not any(field in s for s in return_sets):
            missing_required.append(field)

    # 2) Confirm reconstruction entry function exists and accepts required/optional args (static analysis).
    if reconstruction_fn != "utils.reconstruct.reconstruct_from_generated":
        raise RuntimeError(f"Unexpected reconstruction_fn in schema: {reconstruction_fn}")
    reconstruct_py = repo_root / "utils" / "reconstruct.py"
    recon_sig = _find_function_signature(reconstruct_py, fn_name="reconstruct_from_generated")

    recon_args = recon_sig.args
    recon_required_ok = recon_args[:2] == ["xyz", "atomic_nums"]
    recon_optional_present = all(name in recon_args for name in optional)

    lines: list[str] = []
    lines.append("DECODE_OUTPUT_CONTRACT_CHECK")
    lines.append(f"REPO_ROOT: {repo_root}")
    lines.append("")
    lines.append("REQUIRED:")
    for field in required:
        if field in missing_required:
            lines.append(f"- {field}: MISSING (decode return)")
        else:
            lines.append(f"- {field}: CONFIRMED (decode return)")
    lines.append("")
    lines.append("OPTIONAL:")
    for field in optional:
        lines.append(f"- {field}: OPTIONAL (contract)")
    lines.append("")
    lines.append(f"RECONSTRUCTION_FN: {reconstruction_fn}")
    lines.append(f"RECONSTRUCTION_SIG: {recon_sig.name}({_fmt_list(recon_args)})")
    lines.append(f"RECONSTRUCTION_REQUIRED_ARGS_OK: {recon_required_ok}")
    lines.append(f"RECONSTRUCTION_OPTIONAL_ARGS_PRESENT: {recon_optional_present}")
    lines.append("")

    compatible = (len(missing_required) == 0) and recon_required_ok
    if compatible:
        lines.append("COMPATIBILITY: PASS")
    else:
        lines.append("COMPATIBILITY: FAIL")
        if missing_required:
            lines.append(f"MISSING_REQUIRED_FIELDS: {_fmt_list(missing_required)}")
        if not recon_required_ok:
            lines.append("RECONSTRUCTION_REQUIRED_ARGS_MISMATCH: expected xyz, atomic_nums as first two args")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if not compatible:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

