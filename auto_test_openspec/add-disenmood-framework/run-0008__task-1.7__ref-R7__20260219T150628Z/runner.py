import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Union


MINIMAL_PROPERTY_SET = (
    "PAMPA_NCATS",
    "BBB_Martins",
    "logP",
    "Clearance_Microsome_AZ",
    "hERG",
    "affinity",
    "QED",
    "SA",
    "AMES",
    "lipinski",
)


def _mlp_forward(x: List[float], layers: List[Tuple[List[List[float]], List[float]]]) -> float:
    h = x
    for li, (w, b) in enumerate(layers):
        out: List[float] = []
        for row, bi in zip(w, b):
            s = bi
            for a, wij in zip(h, row):
                s += a * wij
            out.append(s)
        if li != len(layers) - 1:
            # SiLU
            h = [v / (1.0 + (2.718281828459045 ** (-v))) for v in out]
        else:
            h = out
    if len(h) != 1:
        raise ValueError("Expected scalar output")
    return float(h[0])


def _compute_r2(y_true: List[float], y_pred: List[float]) -> Union[float, None]:
    if len(y_true) != len(y_pred):
        raise ValueError("Length mismatch")
    if len(y_true) < 2:
        return None
    mean = sum(y_true) / float(len(y_true))
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    ss_tot = sum((a - mean) ** 2 for a in y_true)
    if ss_tot <= 0.0:
        return None
    return 1.0 - (ss_res / ss_tot)


def main() -> None:
    root = Path(os.environ["BUNDLE_DIR"])
    in_path = root / "inputs" / "mini_batch.pt"
    out_path = root / "outputs" / "expert_r2.json"

    payload = pickle.loads(in_path.read_bytes())
    z_pi: Dict[str, List[List[float]]] = payload["z_pi"]
    labels: Dict[str, List[float]] = payload.get("labels", {})
    predictors: Dict[str, List[Tuple[List[List[float]], List[float]]]] = payload.get("predictors", {})

    r2_out: Dict[str, Union[float, str]] = {}
    for name in MINIMAL_PROPERTY_SET:
        y_true = labels.get(name)
        layers = predictors.get(name)
        if y_true is None or layers is None:
            r2_out[name] = "N/A"
            continue
        y_pred = [_mlp_forward(vec, layers) for vec in z_pi[name]]
        v = _compute_r2(y_true, y_pred)
        r2_out[name] = "N/A" if v is None else float(v)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(r2_out, indent=2, sort_keys=False) + "\n")


if __name__ == "__main__":
    main()
