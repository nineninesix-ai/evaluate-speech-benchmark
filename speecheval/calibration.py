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

    # A chunk shorter than this is too little speech for a speaker embedding to
    # mean anything, and short of the encoders' own minimum input.
    MIN_CHUNK_SEC = 2.0

    def anchor(self, encoder, waveform: np.ndarray,
               sample_rate: int = 16000) -> tuple[float, float, int]:
        """
        Mean cosine between chunks of the reference recording.

        Optimistic by construction — see the module docstring. The chunk length
        shrinks to fit a short recording rather than giving up on it: a 7 s
        reference is cut into two 3.5 s halves instead of yielding no anchor at
        all, and overlapping chunks are avoided wherever the length allows,
        because overlap inflates a self-similarity that is already optimistic.

        Returns (anchor, chunk_sec_used, n_chunks); the anchor is NaN when the
        recording cannot yield two usable chunks, and the voice is then reported
        with a floor and no anchor.
        """
        duration = len(waveform) / sample_rate
        chunk_sec = min(self._chunk.chunk_sec, duration / 2)
        if chunk_sec < self.MIN_CHUNK_SEC:
            logger.warning(
                "reference is %.1f s — cannot cut two chunks of at least %.1f s, "
                "so this voice gets a floor but no anchor",
                duration, self.MIN_CHUNK_SEC,
            )
            return float("nan"), float("nan"), 0

        chunk = int(chunk_sec * sample_rate)
        hop = max(int(self._chunk.hop_sec * sample_rate), 1)
        if hop < chunk and duration >= 2 * chunk_sec:
            hop = chunk        # no overlap when the recording is long enough

        pieces = []
        start = 0
        while start + chunk <= len(waveform) and len(pieces) < self._chunk.max_chunks:
            pieces.append(waveform[start:start + chunk])
            start += hop
        if len(pieces) < 2:
            return float("nan"), chunk_sec, len(pieces)

        embeddings = np.stack([encoder.embed([p])[0] for p in pieces])
        gram = embeddings @ embeddings.T
        upper = gram[np.triu_indices(len(pieces), k=1)]
        return float(upper.mean()), chunk_sec, len(pieces)

    def calibrate(self, encoder, language: str, voice_name: str,
                  reference_waveform: np.ndarray, reference_embedding: np.ndarray,
                  prompt_embeddings: dict[str, np.ndarray]) -> Calibration:
        floor_mean, floor_p95, n_pairs = self.floor(
            encoder, reference_embedding, prompt_embeddings)
        anchor, chunk_sec, n_chunks = self.anchor(encoder, reference_waveform)
        note = ("floor: this voice against every prompt speaker of the language; "
                "anchor: between chunks of one recording, so it shares a channel "
                "and a session and is optimistic")
        if np.isfinite(anchor):
            note += f" ({n_chunks} chunks of {chunk_sec:.1f} s)"
        else:
            note += " — reference too short to anchor, floor only"
        return Calibration(
            language=language,
            encoder=encoder.name,
            voice=voice_name,
            anchor=anchor,
            floor_mean=floor_mean,
            floor_p95=floor_p95,
            source="measured",
            anchor_kind="intra_session" if np.isfinite(anchor) else "none",
            n_floor_pairs=n_pairs,
            note=note,
        )
