"""
Data sources: the benchmark (reference side) and the synthesis (system side).

Both sides are plain parquet on disk. They are read directly with pyarrow rather
than through `datasets`, so audio stays as encoded bytes until something actually
needs a waveform — a full subset of decoded 16 kHz float32 would otherwise sit in
RAM for the whole stage.

The two sides join on `utt`, which is stable across benchmark versions.
"""
from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from .config import EvalConfig, VoiceConfig

logger = logging.getLogger(__name__)

SUBSET_PATTERN = re.compile(r"^(?P<language>[^_]+(?:-[^_]+)?)__(?P<voice>.+)$")


class DataError(RuntimeError):
    """Raised when the data on disk does not match what the config promises."""


def decode_audio(blob: dict | None) -> tuple[np.ndarray, int]:
    """Decode a HuggingFace audio struct ({'bytes': ..., 'path': ...}) to mono float32."""
    if not blob:
        raise DataError("empty audio cell")
    payload = blob.get("bytes")
    if payload is None:
        path = blob.get("path")
        if not path:
            raise DataError("audio cell carries neither bytes nor path")
        samples, sr = sf.read(path, dtype="float32", always_2d=False)
    else:
        samples, sr = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32), int(sr)


def read_audio_file(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a local audio file to mono float32."""
    samples, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32), int(sr)


# ---------------------------------------------------------------------------
# reference side
# ---------------------------------------------------------------------------

# Columns pulled from the benchmark for every join. Audio columns are added on
# demand, because they dominate the read.
BENCHMARK_META_COLUMNS = (
    "utt", "lang", "text", "text_norm", "n_words", "len_bin",
    "speaker_id", "speaker_gender", "prompt_dur", "prompt_dur_bin",
    "has_gt", "gt_dur", "gt_speaker_id",
    "has_sim_ref", "sim_ref_dur", "sim_ref_dur_bin",
    # the human anchors, carried per row
    "anchor_asr", "anchor_wer", "anchor_cer", "anchor_hyp",
    "anchor_subs", "anchor_dels", "anchor_ins", "anchor_n_ref_words",
    "anchor_cer_err", "anchor_n_ref_chars",
    "anchor_wer_mms", "anchor_cer_mms",
    "anchor_subs_mms", "anchor_dels_mms", "anchor_ins_mms", "anchor_cer_err_mms",
    "anchor_wer_scribe", "anchor_cer_scribe", "anchor_hyp_scribe",
    "anchor_subs_scribe", "anchor_dels_scribe", "anchor_ins_scribe",
    "anchor_cer_err_scribe",
    "anchor_sim_wavlm_sv", "anchor_sim_wavlm_ft", "anchor_sim_ecapa",
)


class BenchmarkSource:
    """Reads the benchmark: target texts, prompt audio, and the human anchors."""

    def __init__(self, config: EvalConfig):
        self._config = config
        self._root = config.benchmark.path
        self._split = config.benchmark.split

    def parquet_path(self, language: str) -> Path:
        directory = self._root / language
        if not directory.is_dir():
            raise DataError(f"benchmark subset '{language}' not found under {self._root}")
        files = sorted(directory.glob(f"{self._split}-*.parquet"))
        if not files:
            files = sorted(directory.glob("*.parquet"))
        if not files:
            raise DataError(f"no parquet files in {directory}")
        if len(files) > 1:
            raise DataError(
                f"{directory}: expected a single parquet shard for split "
                f"'{self._split}', found {len(files)}"
            )
        return files[0]

    def available_columns(self, language: str) -> list[str]:
        return list(pq.ParquetFile(self.parquet_path(language)).schema_arrow.names)

    @lru_cache(maxsize=8)
    def _meta(self, language: str):
        path = self.parquet_path(language)
        present = set(pq.ParquetFile(path).schema_arrow.names)
        columns = [c for c in BENCHMARK_META_COLUMNS if c in present]
        missing = [c for c in BENCHMARK_META_COLUMNS if c not in present]
        if missing:
            logger.debug("%s: benchmark columns absent: %s", language, ", ".join(missing))
        return pq.read_table(path, columns=columns).to_pandas()

    def metadata(self, language: str):
        """Per-row metadata and anchors, indexed by `utt`."""
        return self._meta(language)

    def audio_column(self, language: str, column: str):
        """Read one audio column as an arrow ChunkedArray, aligned with `metadata`."""
        path = self.parquet_path(language)
        if column not in pq.ParquetFile(path).schema_arrow.names:
            raise DataError(f"benchmark subset '{language}' has no column '{column}'")
        return pq.read_table(path, columns=["utt", column])

    def row_count(self, language: str) -> int:
        return pq.ParquetFile(self.parquet_path(language)).metadata.num_rows


# ---------------------------------------------------------------------------
# system side
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SynthesisSubset:
    """One directory of generated audio: a single language read in a single voice."""

    name: str
    language: str
    voice: VoiceConfig
    path: Path
    n_rows: int

    def __str__(self) -> str:
        return self.name


class SynthesisSource:
    """Discovers and reads the generated audio produced by the system under test."""

    def __init__(self, config: EvalConfig):
        self._config = config
        self._root = config.synthesis.path
        self._split = config.synthesis.split
        self._columns = config.synthesis.columns

    def _parquet_path(self, directory: Path) -> Optional[Path]:
        files = sorted(directory.glob(f"{self._split}-*.parquet"))
        if not files:
            files = sorted(directory.glob("*.parquet"))
        return files[0] if files else None

    def _make_subset(self, name: str, language: str, voice_name: str,
                     parquet: Path) -> SynthesisSubset:
        return SynthesisSubset(
            name=name,
            language=language,
            voice=self._config.voice(voice_name),
            path=parquet,
            n_rows=pq.ParquetFile(parquet).metadata.num_rows,
        )

    def discover(self) -> list[SynthesisSubset]:
        """
        Find every synthesis subset the config admits, in either layout.

        A subset is one language read in one voice, and its name is always
        `<language>__<voice>`. Two directory shapes produce it:

        * nested (the shape the generator now writes, mirroring a HF dataset):
          a directory named for a benchmark *language*, holding one
          `<voice>.parquet` per voice;
        * flat (the original shape): a `<language>__<voice>` directory holding a
          single parquet.

        Both are accepted, so a results tree in either shape joins the same way.
        """
        if not self._root.is_dir():
            raise DataError(f"synthesis.path is not a directory: {self._root}")

        include = self._config.synthesis.include
        exclude = set(self._config.synthesis.exclude)
        languages = set(self._config.benchmark.languages)

        subsets: list[SynthesisSubset] = []
        skipped: list[str] = []

        def admit(subset_name: str) -> bool:
            if include is not None and subset_name not in include:
                return False
            if subset_name in exclude:
                skipped.append(f"{subset_name} (excluded)")
                return False
            return True

        for directory in sorted(p for p in self._root.iterdir() if p.is_dir()):
            name = directory.name
            if name.startswith("."):
                continue

            # Nested layout: a language directory of <voice>.parquet files.
            if name in languages:
                for parquet in sorted(directory.glob("*.parquet")):
                    voice_name = parquet.stem
                    subset_name = f"{name}__{voice_name}"
                    if admit(subset_name):
                        subsets.append(self._make_subset(subset_name, name, voice_name, parquet))
                continue

            # Flat layout: a <language>__<voice> directory with one parquet.
            if not admit(name):
                continue

            match = SUBSET_PATTERN.match(name)
            if not match:
                skipped.append(f"{name} (name is not <language>__<voice>)")
                continue

            language = match.group("language")
            voice_name = match.group("voice")
            if language not in languages:
                skipped.append(f"{name} (language '{language}' not in benchmark.languages)")
                continue

            parquet = self._parquet_path(directory)
            if parquet is None:
                skipped.append(f"{name} (no parquet for split '{self._split}')")
                continue

            subsets.append(self._make_subset(name, language, voice_name, parquet))

        for message in skipped:
            logger.warning("skipping %s", message)
        if not subsets:
            raise DataError(f"no usable synthesis subsets found under {self._root}")
        return subsets

    def metadata(self, subset: SynthesisSubset):
        """Everything except the audio, as a pandas frame."""
        columns = [
            self._columns.key, self._columns.text, self._columns.duration,
            self._columns.latency, self._columns.model, self._columns.voice,
        ]
        present = set(pq.ParquetFile(subset.path).schema_arrow.names)
        return pq.read_table(subset.path, columns=[c for c in columns if c in present]).to_pandas()

    def iter_rows(self, subset: SynthesisSubset, columns: Optional[list[str]] = None,
                  batch_size: int = 32) -> Iterator[dict]:
        """
        Stream rows, audio included but still encoded.

        The audio cell is handed on as it came out of parquet so it can go
        straight into the engine's own decoder, which is what applies the
        preprocessing the anchors were measured with. A whole subset of decoded
        16 kHz float32 would otherwise sit in RAM for the length of the stage.
        """
        available = set(pq.ParquetFile(subset.path).schema_arrow.names)
        wanted = columns or [self._columns.key, self._columns.audio,
                             self._columns.duration, self._columns.latency]
        take = [c for c in wanted if c in available]
        reader = pq.ParquetFile(subset.path)
        for batch in reader.iter_batches(batch_size=batch_size, columns=take):
            yield from batch.to_pylist()


# ---------------------------------------------------------------------------
# joining
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JoinReport:
    """What lined up between the two sides, and what did not."""

    subset: str
    n_synthesis: int
    n_benchmark: int
    n_joined: int
    missing_in_benchmark: tuple[str, ...]
    missing_in_synthesis: tuple[str, ...]
    text_mismatches: tuple[str, ...]

    @property
    def is_clean(self) -> bool:
        return not (self.missing_in_benchmark or self.missing_in_synthesis or self.text_mismatches)


def join(benchmark: BenchmarkSource, synthesis: SynthesisSource,
         subset: SynthesisSubset, key_column: str = "utt",
         text_column: str = "text") -> tuple["object", JoinReport]:
    """
    Inner-join one synthesis subset onto its benchmark language.

    Returns the joined frame and a report of everything that did not line up:
    keys present on one side only, and rows whose synthesis text differs from the
    benchmark text (which would mean the system was fed something else).
    """
    import pandas as pd

    left = synthesis.metadata(subset)
    right = benchmark.metadata(subset.language)

    left_keys = set(left[key_column])
    right_keys = set(right[key_column])

    merged = left.merge(right, on=key_column, how="inner", suffixes=("_synth", ""))

    mismatches: list[str] = []
    synth_text = f"{text_column}_synth"
    if synth_text in merged.columns and text_column in merged.columns:
        differs = merged[synth_text].fillna("") .str.strip() != merged[text_column].fillna("").str.strip()
        mismatches = merged.loc[differs, key_column].tolist()

    report = JoinReport(
        subset=subset.name,
        n_synthesis=len(left),
        n_benchmark=len(right),
        n_joined=len(merged),
        missing_in_benchmark=tuple(sorted(left_keys - right_keys)),
        missing_in_synthesis=tuple(sorted(right_keys - left_keys)),
        text_mismatches=tuple(mismatches),
    )
    return merged, report
