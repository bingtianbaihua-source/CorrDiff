from __future__ import annotations

import ast
import io
import os
import struct
import sys
import zipfile
from array import array
from dataclasses import dataclass


@dataclass(frozen=True)
class NpyArray:
    descr: str
    shape: tuple[int, ...]
    data: list


_TYPECODE_BY_DESCR = {
    "<f4": "f",
    "<i8": "q",
    "|u1": "B",
}

_ITEMSIZE_BY_DESCR = {
    "<f4": 4,
    "<i8": 8,
    "|u1": 1,
}


def _flatten_c_order(data: list) -> list:
    if not isinstance(data, list):
        raise TypeError("data must be a (nested) list")
    if len(data) == 0:
        return []
    if isinstance(data[0], list):
        flat: list = []
        for row in data:
            if not isinstance(row, list):
                raise TypeError("data must be rectangular nested lists for nd>1")
            flat.extend(row)
        return flat
    return list(data)


def _reshape_2d(flat: list, *, rows: int, cols: int) -> list[list]:
    out = []
    idx = 0
    for _ in range(rows):
        out.append(flat[idx : idx + cols])
        idx += cols
    return out


def _encode_npy(*, descr: str, shape: tuple[int, ...], data: list) -> bytes:
    if descr not in _TYPECODE_BY_DESCR:
        raise ValueError(f"Unsupported dtype descr: {descr!r}")
    if any(int(d) < 0 for d in shape):
        raise ValueError(f"Invalid shape: {shape!r}")

    typecode = _TYPECODE_BY_DESCR[descr]
    flat = _flatten_c_order(data)

    header_dict = {"descr": descr, "fortran_order": False, "shape": shape}
    header = (repr(header_dict) + "\n").encode("latin1")

    magic = b"\x93NUMPY"
    version = b"\x01\x00"

    # Pad so that the entire header ends on a 16-byte boundary.
    prelude_len = len(magic) + len(version) + 2
    pad_len = (-((prelude_len + len(header)) % 16)) % 16
    header = header[:-1] + (b" " * pad_len) + b"\n"
    if len(header) >= 65536:
        raise ValueError("Header too long for .npy v1.0")

    header_len = struct.pack("<H", len(header))

    payload = array(typecode, flat)
    if sys.byteorder != "little" and descr.startswith("<"):
        payload.byteswap()
    data_bytes = payload.tobytes()

    return magic + version + header_len + header + data_bytes


def _decode_npy(buf: bytes) -> NpyArray:
    bio = io.BytesIO(buf)
    magic = bio.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError("Not a .npy file")
    version = bio.read(2)
    if version != b"\x01\x00":
        raise ValueError(f"Unsupported .npy version: {version!r}")
    (header_len,) = struct.unpack("<H", bio.read(2))
    header = bio.read(header_len).decode("latin1")
    header_dict = ast.literal_eval(header)
    descr = header_dict["descr"]
    shape = tuple(int(x) for x in header_dict["shape"])
    if descr not in _TYPECODE_BY_DESCR:
        raise ValueError(f"Unsupported dtype descr: {descr!r}")

    typecode = _TYPECODE_BY_DESCR[descr]
    itemsize = _ITEMSIZE_BY_DESCR[descr]
    count = 1
    for d in shape:
        count *= int(d)
    expected = count * itemsize
    payload = bio.read()
    if len(payload) != expected:
        raise ValueError(f"Truncated .npy payload: expected {expected} bytes, got {len(payload)}")

    arr = array(typecode)
    arr.frombytes(payload)
    if sys.byteorder != "little" and descr.startswith("<"):
        arr.byteswap()
    flat = list(arr)

    if len(shape) == 1:
        data = flat
    elif len(shape) == 2:
        data = _reshape_2d(flat, rows=shape[0], cols=shape[1])
    else:
        raise ValueError(f"Only 1D/2D arrays supported, got shape={shape!r}")
    return NpyArray(descr=descr, shape=shape, data=data)


def save_npz(path: str, arrays: dict[str, NpyArray]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key, arr in arrays.items():
            if not isinstance(arr, NpyArray):
                raise TypeError(f"arrays[{key!r}] must be NpyArray")
            npy = _encode_npy(descr=arr.descr, shape=arr.shape, data=arr.data)
            zf.writestr(f"{key}.npy", npy)


def load_npz(path: str) -> dict[str, NpyArray]:
    out: dict[str, NpyArray] = {}
    with zipfile.ZipFile(path, "r") as zf:
        for name in zf.namelist():
            if not name.endswith(".npy"):
                continue
            key = name[: -len(".npy")]
            out[key] = _decode_npy(zf.read(name))
    return out

