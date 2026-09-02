"""Experiment naming and JSON I/O for the paper reproduction."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def experiment_name(timestamp: datetime, model: str) -> str:
    """Return a filesystem-safe timestamped experiment name."""
    slug = re.sub(r"[^A-Za-z0-9._+-]+", "-", model).strip("-").lower()
    return f"{timestamp:%Y%m%d%H%M}-{slug}"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
