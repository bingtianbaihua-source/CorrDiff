import argparse
import json
import os
from typing import Dict, List

from models.admet_model import ADMETModel, REQUIRED_ADMET_ATTRIBUTES


def _read_smiles_list(path: str) -> List[str]:
    smiles: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                smiles.append(s)
    return smiles


def _load_existing(path: str) -> Dict[str, Dict[str, float]]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return {str(k): v for k, v in data.items() if isinstance(v, dict)}
    if isinstance(data, list):
        out: Dict[str, Dict[str, float]] = {}
        for row in data:
            if isinstance(row, dict) and "smiles" in row:
                out[str(row["smiles"])] = {k: row.get(k) for k in REQUIRED_ADMET_ATTRIBUTES}
        return out
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline ADMET attribute augmentation for SMILES.")
    parser.add_argument("--input", required=True, help="Path to a newline-delimited SMILES list.")
    parser.add_argument("--output", required=True, help="Path to write augmented labels JSON.")
    parser.add_argument(
        "--cache",
        default="",
        help="Optional path to a JSON cache (dict or list format). Missing entries will be computed and cached.",
    )
    args = parser.parse_args()

    smiles_list = _read_smiles_list(args.input)
    cache_path = args.cache or args.output
    cache = _load_existing(cache_path)

    missing = [s for s in smiles_list if s not in cache or any(k not in cache[s] for k in REQUIRED_ADMET_ATTRIBUTES)]
    if missing:
        model = ADMETModel()
        preds = model.predict(missing)
        for s in missing:
            cache[s] = preds.get(s, {})

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    rows = []
    for s in smiles_list:
        props = cache.get(s, {})
        row = {"smiles": s}
        for k in REQUIRED_ADMET_ATTRIBUTES:
            row[k] = props.get(k)
        rows.append(row)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=False)

    if args.cache:
        os.makedirs(os.path.dirname(os.path.abspath(args.cache)), exist_ok=True)
        with open(args.cache, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, sort_keys=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

