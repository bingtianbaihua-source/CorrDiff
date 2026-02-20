#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _as_float(x: Any, *, name: str) -> float:
    try:
        return float(x)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"{name} must be a number, got {type(x).__name__}") from e


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Config not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config: {path}") from e
    if not isinstance(data, dict):
        raise ValueError("Config root must be an object/dict")
    return data


def _pareto_front_max(points: Sequence[Sequence[float]]) -> List[List[float]]:
    pts = [list(p) for p in points]
    if not pts:
        return []

    front: List[List[float]] = []
    for i, p in enumerate(pts):
        dominated = False
        for j, q in enumerate(pts):
            if i == j:
                continue
            if all(qk >= pk for qk, pk in zip(q, p)) and any(qk > pk for qk, pk in zip(q, p)):
                dominated = True
                break
        if not dominated:
            front.append(p)
    return front


def _hypervolume_max(points: Sequence[Sequence[float]], reference: Sequence[float]) -> float:
    if len(reference) == 0:
        return 0.0

    dim = len(reference)
    if any(len(p) != dim for p in points):
        raise ValueError("All HV points must have the same dimensionality as reference")

    ref = [float(r) for r in reference]
    valid = []
    for p in points:
        pp = [float(x) for x in p]
        if any(pp[k] <= ref[k] for k in range(dim)):
            continue
        valid.append(pp)

    if not valid:
        return 0.0

    valid = _pareto_front_max(valid)

    if dim == 1:
        return max(p[0] - ref[0] for p in valid)

    coords = sorted(set([ref[0]] + [p[0] for p in valid if p[0] > ref[0]]))
    hv = 0.0
    for i in range(len(coords) - 1):
        x0 = coords[i]
        x1 = coords[i + 1]
        width = x1 - x0
        if width <= 0:
            continue

        projected = [p[1:] for p in valid if p[0] >= x1]
        if not projected:
            continue
        projected = _pareto_front_max(projected)
        hv += width * _hypervolume_max(projected, ref[1:])
    return hv


def _mean_pairwise_distance(points: Sequence[Sequence[float]]) -> float:
    pts = [list(map(float, p)) for p in points]
    n = len(pts)
    if n < 2:
        return 0.0

    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += math.sqrt(sum((pts[i][k] - pts[j][k]) ** 2 for k in range(len(pts[i]))))
            count += 1
    return total / float(count)


def _frobenius_norm_diff(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    if len(a) != len(b):
        raise ValueError("C_pred and C_true must have the same shape")
    if not a:
        return 0.0
    if any(len(row_a) != len(row_b) for row_a, row_b in zip(a, b)):
        raise ValueError("C_pred and C_true must have the same shape")
    s = 0.0
    for row_a, row_b in zip(a, b):
        if len(row_a) != len(row_b):
            raise ValueError("C_pred and C_true must have the same shape")
        for xa, xb in zip(row_a, row_b):
            da = float(xa) - float(xb)
            s += da * da
    return math.sqrt(s)


def _get_candidates(config: Dict[str, Any]) -> Tuple[List[Dict[str, float]], List[List[float]], List[List[float]]]:
    candidates_raw = config.get("candidates")
    if not isinstance(candidates_raw, list) or not candidates_raw:
        raise ValueError("config.candidates must be a non-empty list")

    candidates: List[Dict[str, float]] = []
    raw_vectors: List[List[float]] = []
    hv_vectors: List[List[float]] = []
    for idx, c in enumerate(candidates_raw):
        if not isinstance(c, dict):
            raise ValueError(f"candidates[{idx}] must be an object/dict")

        qed = _as_float(c.get("qed"), name=f"candidates[{idx}].qed")
        sa = _as_float(c.get("sa"), name=f"candidates[{idx}].sa")
        logp = _as_float(c.get("logp"), name=f"candidates[{idx}].logp")
        affinity = _as_float(c.get("affinity"), name=f"candidates[{idx}].affinity")

        candidates.append({"qed": qed, "sa": sa, "logp": logp, "affinity": affinity})
        raw_vectors.append([qed, sa, logp, affinity])
        hv_vectors.append([qed, sa, logp, -affinity])

    return candidates, raw_vectors, hv_vectors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Toy evaluation for DisenMoOD metrics (HV, sparsity, correlation error).")
    parser.add_argument("--config", required=True, type=str, help="Path to eval_config.json")
    parser.add_argument("--output", required=True, type=str, help="Path to write eval_report.json")
    args = parser.parse_args(argv)

    config_path = Path(args.config).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config = _load_json(config_path)
    _, raw_vectors, hv_vectors = _get_candidates(config)

    ref = config.get("hv_reference")
    if not isinstance(ref, list) or len(ref) != 4:
        raise ValueError("config.hv_reference must be a list of 4 numbers (qed, sa, logp, -affinity)")
    hv_reference = [float(x) for x in ref]

    hv = _hypervolume_max(hv_vectors, hv_reference)
    sparsity = _mean_pairwise_distance(raw_vectors)

    c_pred = config.get("C_pred")
    c_true = config.get("C_true")
    if not isinstance(c_pred, list) or not isinstance(c_true, list):
        raise ValueError("config.C_pred and config.C_true must be matrices (lists of lists)")
    correlation_error = _frobenius_norm_diff(c_pred, c_true)

    intervention = config.get("intervention")
    if not isinstance(intervention, dict):
        raise ValueError("config.intervention must be an object/dict")
    property_varied = intervention.get("property_varied")
    if not isinstance(property_varied, str) or not property_varied:
        raise ValueError("intervention.property_varied must be a non-empty string")
    delta_target = _as_float(intervention.get("delta_target"), name="intervention.delta_target")
    delta_others_max = _as_float(intervention.get("delta_others_max"), name="intervention.delta_others_max")
    intervention_result = {
        "property_varied": property_varied,
        "delta_target": float(delta_target),
        "delta_others_max": float(delta_others_max),
    }

    docking_scores = config.get("docking_scores")
    if not isinstance(docking_scores, list) or not docking_scores:
        raise ValueError("config.docking_scores must be a non-empty list")
    docking_scores_out = [float(x) for x in docking_scores]

    report = {
        "HV": float(hv),
        "Sparsity": float(sparsity),
        "correlation_error": float(correlation_error),
        "intervention_result": intervention_result,
        "docking_scores": docking_scores_out,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

