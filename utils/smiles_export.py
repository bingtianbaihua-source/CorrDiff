from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SmilesExportResult:
    smiles: str
    note: str | None = None


def reconstruct_smiles(xyz, atomic_nums, *, basic_mode: bool = True) -> SmilesExportResult:
    """
    Best-effort SMILES export from generated coordinates and atom types.

    This prefers the project's existing reconstruction path (`utils.reconstruct`), but
    falls back to a placeholder string when optional dependencies (rdkit/openbabel/scipy)
    are unavailable in the runtime.
    """
    try:
        from rdkit import Chem as _Chem
        from utils.reconstruct import MolReconsError, reconstruct_from_generated
    except Exception as e:  # pragma: no cover - depends on optional env deps
        return SmilesExportResult(smiles="SMILES_UNAVAILABLE_IN_TOY_MODE", note=str(e))

    try:
        mol = reconstruct_from_generated(xyz, atomic_nums, basic_mode=basic_mode)
        smiles = _Chem.MolToSmiles(mol)
        if not smiles:
            return SmilesExportResult(
                smiles="SMILES_UNAVAILABLE_IN_TOY_MODE", note="RDKit returned empty SMILES"
            )
        return SmilesExportResult(smiles=smiles, note=None)
    except MolReconsError as e:  # pragma: no cover - reconstruction may fail on random toy outputs
        return SmilesExportResult(smiles="SMILES_UNAVAILABLE_IN_TOY_MODE", note=str(e))
    except Exception as e:  # pragma: no cover
        return SmilesExportResult(smiles="SMILES_UNAVAILABLE_IN_TOY_MODE", note=str(e))
