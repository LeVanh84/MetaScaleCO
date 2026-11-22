"""Paths and configuration.

Environment variables:
    METASCALE_CONFIG  path to an alternative config json (default: config.json)
    METASCALE_TAG     suffix for output folders, e.g. "test" -> artifacts_test/, ...
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = os.environ.get("METASCALE_TAG", "")
_SUF = f"_{TAG}" if TAG else ""
ART = ROOT / f"artifacts{_SUF}"
RES = ROOT / f"results{_SUF}"
FIG = ROOT / f"figures{_SUF}"
for p in (ART, RES, FIG, ART / "data", ART / "models"):
    p.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    path = Path(os.environ.get("METASCALE_CONFIG", ROOT / "config.json"))
    cfg = json.loads(path.read_text())
    cfg.setdefault("workers", 8)
    return cfg


def read_json(name: str):
    return json.loads((ART / name).read_text())


def write_json(name: str, obj):
    (ART / name).write_text(json.dumps(obj, indent=2))
