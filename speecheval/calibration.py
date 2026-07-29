"""
Turning a cosine into something readable.

An absolute SIM value carries almost no information on its own. Whether 0.87 is
good depends entirely on what two *different* speakers score, and that floor
varies by language and by encoder — under `wavlm_sv` the impostor p95 reaches
0.87-0.93, and on Kyrgyz it is *above* the human anchor. So every similarity is
reported on a normalised scale as well:

    sim_norm = (sim - floor) / (anchor - floor)

0 means indistinguishable from an impostor, 1 means as close as two recordings of
the same person.

Where the two ends come from depends on what the reference voice is.

**A benchmark prompt speaker** (`cv_voice`). Both ends are published in the
benchmark's own reports, measured on the same audio and the same encoders, so
they are read from there rather than recomputed.

**One of our own voices** (`nurisa_en`, `ulan_emo`, `audio_ru`, `audio_en`). The
published coefficients do not apply — the voice is not in the corpus. Both ends
are therefore computed here:

*floor* — cosine between our reference and the prompt clip of every speaker in
that language. This is a better floor than the published one for this purpose:
it is the impostor distribution *of this specific voice* against that specific
population, not a generic cross-speaker average.

*anchor* — cosine between chunks of the reference recording itself. This is
**optimistic and is labelled as such everywhere it appears**: two chunks of one
recording share a microphone, a room and a session, which two recordings of a
person made on different days do not. It bounds the scale rather than describing
a human ceiling, and a `sim_norm` computed against it should be read as "at
least this far", never as a ceiling-relative score.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Calibration:
    """The two ends of the readable scale, and where they came from."""

    language: str
    encoder: str
    voice: str
    anchor: float
    floor_mean: float
    floor_p95: float
    source: str                      # published | measured
    anchor_kind: str                 # human | intra_session
    n_floor_pairs: int
    note: str = ""

    @property
    def usable_range(self) -> float:
        return self.anchor - self.floor_mean

    def normalise(self, similarity):
        span = self.usable_range
        if not np.isfinite(span) or span <= 0:
            return np.nan if np.isscalar(similarity) else np.full(len(similarity), np.nan)
        return (similarity - self.floor_mean) / span

    def to_dict(self) -> dict:
        return {
            "lang": self.language,
            "encoder": self.encoder,
            "voice": self.voice,
            "anchor": self.anchor,
            "anchor_kind": self.anchor_kind,
            "floor_mean": self.floor_mean,
            "floor_p95": self.floor_p95,
            "usable_range": self.usable_range,
            "calibration_source": self.source,
            "n_floor_pairs": self.n_floor_pairs,
            "note": self.note,
        }


class PublishedCalibration:
    """The anchor and impostor floor the benchmark ships, per language and encoder."""

    def __init__(self, reports_dir: Path, csv_relative: str = "csv/sim.csv"):
        self.path = Path(reports_dir) / csv_relative
        self._table: Optional[pd.DataFrame] = None

    def _load(self) -> pd.DataFrame:
        if self._table is None:
            if not self.path.is_file():
                raise FileNotFoundError(
                    f"published SIM calibration not found at {self.path}; without it "
                    "a cosine cannot be put on a readable scale"
                )
            self._table = pd.read_csv(self.path)
        return self._table

    def get(self, language: str, encoder: str, voice: str) -> Calibration:
        table = self._load()
        row = table[(table["lang"] == language) & (table["encoder"] == encoder)]
        if row.empty:
            raise KeyError(
                f"{self.path}: no published calibration for {language}/{encoder}"
            )
        row = row.iloc[0]
        return Calibration(
            language=language,
            encoder=encoder,
            voice=voice,
            anchor=float(row["anchor"]),
            floor_mean=float(row["floor mean"]),
            floor_p95=float(row["floor p95"]),
            source="published",
            anchor_kind="human",
            n_floor_pairs=int(row.get("floor pairs", 0) or 0),
            note="anchor and floor as published by the benchmark",
        )


class VoiceCalibrator:
    """Measures both ends of the scale for a reference voice that is not in the corpus."""

    def __init__(self, config):
        self.config = config
        self._chunk = config.sim.calibration.anchor

    def floor(self, encoder, reference_embedding: np.ndarray,
              prompt_embeddings: dict[str, np.ndarray]) -> tuple[float, float, int]:
        """Cosine of our voice against every prompt speaker of the language."""
        if not prompt_embeddings:
            return float("nan"), float("nan"), 0
        matrix = np.stack(list(prompt_embeddings.values()))
        similarities = matrix @ reference_embedding
        return (float(similarities.mean()), float(np.quantile(similarities, 0.95)),
                len(similarities))

    def anchor(self, encoder, waveform: np.ndarray, sample_rate: int = 16000) -> float:
        """
        Mean cosine between chunks of the reference recording.

        Optimistic by construction — see the module docstring. Returns NaN when
        the recording is too short to yield two chunks, in which case the voice
        simply has no anchor and only the floor is reported.
        """
        chunk = int(self._chunk.chunk_sec * sample_rate)
        hop = int(self._chunk.hop_sec * sample_rate)
        if len(waveform) < 2 * chunk:
            logger.warning(
                "reference is %.1f s, too short for two %.1f s chunks — no anchor",
                len(waveform) / sample_rate, self._chunk.chunk_sec,
            )
            return float("nan")

        pieces = []
        start = 0
        while start + chunk <= len(waveform) and len(pieces) < self._chunk.max_chunks:
            pieces.append(waveform[start:start + chunk])
            start += hop

        embeddings = np.stack([encoder.embed([p])[0] for p in pieces])
        gram = embeddings @ embeddings.T
        upper = gram[np.triu_indices(len(pieces), k=1)]
        return float(upper.mean())

    def calibrate(self, encoder, language: str, voice_name: str,
                  reference_waveform: np.ndarray, reference_embedding: np.ndarray,
                  prompt_embeddings: dict[str, np.ndarray]) -> Calibration:
        floor_mean, floor_p95, n_pairs = self.floor(
            encoder, reference_embedding, prompt_embeddings)
        anchor = self.anchor(encoder, reference_waveform)
        return Calibration(
            language=language,
            encoder=encoder.name,
            voice=voice_name,
            anchor=anchor,
            floor_mean=floor_mean,
            floor_p95=floor_p95,
            source="measured",
            anchor_kind="intra_session",
            n_floor_pairs=n_pairs,
            note=("floor: this voice against every prompt speaker of the language; "
                  "anchor: between chunks of one recording, so it shares a channel "
                  "and a session and is optimistic"),
        )
