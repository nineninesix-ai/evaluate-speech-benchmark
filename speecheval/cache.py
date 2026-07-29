"""
Per-stage cache, which is also the resume mechanism.

Every expensive unit of work — one subset read by one recogniser, one subset
embedded by one encoder — writes a per-utterance parquet and a JSON summary. A
rerun skips any unit whose artefacts exist and whose recorded parameters still
match. Nothing is ever silently reused after the configuration changed: the key
carries the parameters that decide the numbers, so editing one of them
invalidates exactly the units it affects and leaves the rest.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def fingerprint(payload: Any) -> str:
    """Short stable digest of anything JSON-serialisable."""
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class StageUnit:
    """One cacheable piece of work."""

    stage: str
    name: str
    parameters: dict

    @property
    def key(self) -> str:
        return fingerprint(self.parameters)


class StageCache:
    def __init__(self, root: Path, run_name: str, enabled: bool = True):
        self.root = Path(root) / run_name
        self.enabled = enabled

    def _paths(self, unit: StageUnit) -> tuple[Path, Path]:
        directory = self.root / unit.stage
        return directory / f"{unit.name}.parquet", directory / f"{unit.name}.json"

    def load(self, unit: StageUnit) -> Optional[tuple[pd.DataFrame, dict]]:
        """Return cached results when they exist and were produced by these parameters."""
        if not self.enabled:
            return None
        data_path, meta_path = self._paths(unit)
        if not (data_path.is_file() and meta_path.is_file()):
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("%s: unreadable cache summary, recomputing", meta_path)
            return None
        if meta.get("_key") != unit.key:
            logger.info("%s/%s: parameters changed, recomputing", unit.stage, unit.name)
            return None
        return pd.read_parquet(data_path), meta

    def save(self, unit: StageUnit, frame: pd.DataFrame, summary: dict) -> Path:
        data_path, meta_path = self._paths(unit)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(data_path, index=False)
        payload = {**summary, "_key": unit.key, "_parameters": unit.parameters}
        meta_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return data_path

    def iter_saved(self, stage: str):
        """Every cached frame of a stage, for the report step."""
        directory = self.root / stage
        if not directory.is_dir():
            return
        for meta_path in sorted(directory.glob("*.json")):
            data_path = meta_path.with_suffix(".parquet")
            if not data_path.is_file():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            yield pd.read_parquet(data_path), meta
