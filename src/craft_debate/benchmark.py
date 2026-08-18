"""Loader/validator for the CRAFT structures dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .domain import ALL_COORDS


def load_structures(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Benchmark file must be a non-empty JSON list: {path}")

    structures = []
    for item in data:
        structure = dict(item)
        structure["spans"] = {int(k): v for k, v in item.get("spans", {}).items()}
        structure["structure"] = {str(k): list(v) for k, v in item.get("structure", {}).items()}
        for coord in structure["structure"]:
            if coord not in ALL_COORDS:
                raise ValueError(
                    f"{structure.get('id', '?')}: invalid coordinate {coord!r} in structure"
                )
        if not structure.get("id") or not structure.get("complexity"):
            raise ValueError(f"Structure missing id/complexity in {path}")
        structures.append(structure)
    return structures
