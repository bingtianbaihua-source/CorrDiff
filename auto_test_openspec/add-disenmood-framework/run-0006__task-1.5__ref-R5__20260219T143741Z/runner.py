import json
import os
import struct
import zipfile
from pathlib import Path


def main() -> None:
    root = Path(os.environ["BUNDLE_DIR"])
    in_path = root / "inputs" / "guidance_targets.json"
    out_npz = root / "outputs" / "guided_latents.npz"
    out_stats = root / "outputs" / "guidance_stats.json"

    cfg = json.loads(in_path.read_text())
    target = [str(x) for x in cfg["target"]]
    frozen = [str(x) for x in cfg["frozen"]]
    num_steps = int(cfg["num_steps"])
    step_size = float(cfg["step_size"])
    z_pi_dim = int(cfg["z_pi_dim"])
    batch_size = int(cfg["batch_size"])

    rng = __import__("random").Random(0)
    z_pi_dict: dict[str, list[list[float]]] = {}
    for name in sorted(set(target + frozen)):
        rows = []
        for _ in range(batch_size):
            rows.append([rng.gauss(0.0, 1.0) for _ in range(z_pi_dim)])
        z_pi_dict[name] = rows

    z_init = {k: v.copy() for k, v in z_pi_dict.items()}
    z_guided = {k: v.copy() for k, v in z_pi_dict.items()}

    def energy_and_grad(z: list[list[float]]) -> tuple[float, list[list[float]]]:
        energy = 0.0
        grad = []
        for row in z:
            grad_row = []
            for v in row:
                diff = v - 1.0
                energy += diff * diff
                grad_row.append(2.0 * diff)
            grad.append(grad_row)
        return energy, grad

    frozen_set = set(frozen)
    target_set = set(target)

    for _step_idx in range(num_steps):
        for name in sorted(z_guided.keys()):
            if name in frozen_set:
                continue
            if name not in target_set:
                continue
            _energy, grad = energy_and_grad(z_guided[name])
            next_z = []
            for row, grad_row in zip(z_guided[name], grad):
                next_z.append([v - step_size * g for v, g in zip(row, grad_row)])
            z_guided[name] = next_z

    def to_float32_le_bytes(z: list[list[float]]) -> bytes:
        buf = bytearray()
        for row in z:
            for v in row:
                buf.extend(struct.pack("<f", float(v)))
        return bytes(buf)

    def npy_bytes_float32_2d(z: list[list[float]]) -> bytes:
        rows = len(z)
        cols = len(z[0]) if rows else 0
        for row in z:
            if len(row) != cols:
                raise ValueError("ragged array")

        header = {
            "descr": "<f4",
            "fortran_order": False,
            "shape": (rows, cols),
        }
        header_str = str(header)
        header_bytes = (header_str + "\n").encode("latin1")

        magic = b"\x93NUMPY"
        version = b"\x01\x00"
        header_len = len(header_bytes)
        pad_len = (16 - ((len(magic) + len(version) + 2 + header_len) % 16)) % 16
        header_bytes = header_bytes[:-1] + (b" " * pad_len) + b"\n"
        header_len = len(header_bytes)

        out = bytearray()
        out.extend(magic)
        out.extend(version)
        out.extend(struct.pack("<H", header_len))
        out.extend(header_bytes)
        out.extend(to_float32_le_bytes(z))
        return bytes(out)

    with zipfile.ZipFile(out_npz, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for k in sorted(z_init.keys()):
            zf.writestr(f"z_pi_initial_{k}.npy", npy_bytes_float32_2d(z_init[k]))
            zf.writestr(f"z_pi_guided_{k}.npy", npy_bytes_float32_2d(z_guided[k]))

    frozen_deltas = []
    for k in frozen:
        max_abs = 0.0
        for row_g, row_i in zip(z_guided[k], z_init[k]):
            for vg, vi in zip(row_g, row_i):
                max_abs = max(max_abs, abs(vg - vi))
        d = float(max_abs)
        frozen_deltas.append(d)
    max_frozen_delta = float(max(frozen_deltas) if frozen_deltas else 0.0)

    target_deltas = []
    for k in target:
        total = 0.0
        count = 0
        for row_g, row_i in zip(z_guided[k], z_init[k]):
            for vg, vi in zip(row_g, row_i):
                total += abs(vg - vi)
                count += 1
        d = float(total / max(count, 1))
        target_deltas.append(d)
    min_target_delta = float(min(target_deltas) if target_deltas else 0.0)

    tol = 1e-8
    stats = {
        "target_changed": bool(min_target_delta > 1e-3),
        "frozen_unchanged": bool(max_frozen_delta <= tol),
        "max_frozen_delta": max_frozen_delta,
        "min_target_delta": min_target_delta,
    }
    out_stats.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
