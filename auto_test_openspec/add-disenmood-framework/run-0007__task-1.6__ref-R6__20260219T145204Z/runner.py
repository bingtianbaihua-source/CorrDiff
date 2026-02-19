import json
import os
from pathlib import Path

# Ensure OpenMP shared-memory is disabled (runner is pure-Python, but keep parity with bundle contracts).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("KMP_DISABLE_SHM", "1")

import math
import random


def is_pareto_optimal(costs: list[list[float]]) -> list[bool]:
    # Minimization: i is dominated if ∃j: costs[j] <= costs[i] (all) and < (any).
    n = len(costs)
    mask = [True] * n
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            cj = costs[j]
            ci = costs[i]
            if all(a <= b for a, b in zip(cj, ci)) and any(a < b for a, b in zip(cj, ci)):
                mask[i] = False
                break
    return mask


def main() -> None:
    root = Path(os.environ["BUNDLE_DIR"])
    in_path = root / "inputs" / "multi_objective_case.json"
    out_path = root / "outputs" / "pareto_set.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.loads(in_path.read_text())
    objectives = list(payload["objectives"])
    n_props = int(payload["n_props"])
    n_candidates = int(payload["n_candidates"])
    corr_threshold = float(payload["corr_threshold"])

    if len(objectives) != n_props:
        raise ValueError(f"objectives length must equal n_props, got {len(objectives)} vs {n_props}")

    # Synthetic correlation matrix (embedded as constants):
    # - (qed, sa): high correlation => merged guidance
    # - others: low-correlation => separate guidance
    corr = [[0.1 for _ in range(n_props)] for _ in range(n_props)]
    for i in range(n_props):
        corr[i][i] = 1.0
    corr[0][1] = corr[1][0] = 0.8
    corr[2][3] = corr[3][2] = 0.2

    def make_groups() -> list[list[int]]:
        parent = list(range(n_props))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n_props):
            for j in range(i + 1, n_props):
                if float(corr[i][j]) > corr_threshold:
                    union(i, j)

        groups: dict[int, list[int]] = {}
        for i in range(n_props):
            groups.setdefault(find(i), []).append(i)
        return [sorted(v) for _, v in sorted(groups.items(), key=lambda kv: min(kv[1]))]

    groups = make_groups()

    latent_dim = 8
    # Targets are chosen to create a trade-off: logp wants -1 while others want +1.
    targets: dict[str, list[float]] = {
        objectives[0]: [1.0] * latent_dim,    # qed
        objectives[1]: [1.05] * latent_dim,   # sa (correlated with qed)
        objectives[2]: [-1.0] * latent_dim,   # logp (conflicts with the others)
        objectives[3]: [1.0] * latent_dim,    # affinity (aligns with qed/sa)
    }

    def energy(name: str, z: list[float]) -> float:
        t = targets[name]
        return sum((zi - ti) ** 2 for zi, ti in zip(z, t)) / float(len(z))

    def grad(name: str, z: list[float]) -> list[float]:
        t = targets[name]
        scale = 2.0 / float(len(z))
        return [scale * (zi - ti) for zi, ti in zip(z, t)]

    def make_weight_vectors(seed: int) -> list[list[float]]:
        w = []
        for i in range(min(n_candidates, n_props)):
            vec = [0.0] * n_props
            vec[i] = 1.0
            w.append(vec)
        rng = random.Random(seed)
        while len(w) < n_candidates:
            # Dirichlet-ish weights via gamma.
            g = [rng.gammavariate(1.0, 1.0) for _ in range(n_props)]
            s = sum(g) or 1.0
            w.append([x / s for x in g])
        return w

    weight_vectors = make_weight_vectors(seed=123)

    def guided_sample(seed: int, weights: list[float]) -> list[float]:
        rng = random.Random(seed)
        z = [rng.gauss(0.0, 1.0) for _ in range(latent_dim)]
        step_size = 0.15
        n_steps = 60

        for _ in range(n_steps):
            for group in groups:
                # Merge correlated objectives: sum weighted gradients, then apply one joint update.
                merged_g = [0.0] * latent_dim
                for idx in group:
                    wi = float(weights[idx])
                    if wi <= 0.0:
                        continue
                    name = objectives[idx]
                    gi = grad(name, z)
                    merged_g = [mg + wi * gk for mg, gk in zip(merged_g, gi)]
                z = [zi - step_size * gi for zi, gi in zip(z, merged_g)]
        return z

    candidates: list[dict] = []
    all_costs: list[list[float]] = []
    for i in range(n_candidates):
        z = guided_sample(seed=123 + i, weights=weight_vectors[i])
        costs = [energy(name, z) for name in objectives]
        all_costs.append(costs)
        candidates.append({"z": z, "costs": costs})

    pareto_mask = is_pareto_optimal(all_costs)
    pareto = [c for c, m in zip(candidates, pareto_mask) if m]

    if len(pareto) < 2:
        raise AssertionError(f"Expected >= 2 Pareto solutions, got {len(pareto)}")

    # Serialize Pareto set as objective scores (higher is better): score = -cost.
    records = []
    pareto_costs = []
    for i, cand in enumerate(pareto):
        costs = [float(x) for x in cand["costs"]]
        pareto_costs.append(costs)
        scores = [-c for c in costs]
        rec = {"candidate_id": i}
        for name, s in zip(objectives, scores):
            rec[name] = float(s)
        records.append(rec)

    # Verify: all returned are Pareto non-dominated.
    check = is_pareto_optimal(pareto_costs)
    if not all(check):
        raise AssertionError("pareto_set.json contains a dominated solution")

    out_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
