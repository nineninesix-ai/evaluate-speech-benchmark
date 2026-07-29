"""
Speaker similarity: cos(synthesis, the reference the system was conditioned on).

Three encoders, because one is not enough to read. `wavlm_ft` is the scale the
seed-tts-eval literature uses; `wavlm_sv` is what the benchmark's v1 numbers were
published on; `ecapa` shares neither architecture nor training data with either,
which is what makes the benchmark's own selection bias visible — prompt QC
rejected clips by their distance from the speaker centroid *in WavLM space*, so a
WavLM anchor is partly measuring that selection.

Reference embeddings are cached per speaker: a prompt clip serves up to seven
rows, and embedding it seven times is the same number computed seven times.

Every similarity is reported twice — as a raw cosine, and normalised against the
floor and anchor for its language, encoder and voice (see `calibration`).
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..cache import StageUnit
from ..calibration import Calibration, PublishedCalibration, VoiceCalibrator
from ..engine import build_encoder, describe
from ..sources import SynthesisSubset, read_audio_file

logger = logging.getLogger(__name__)

STAGE = "sim"

# Bump when the calibration logic changes; it is part of the cache key for the
# voices that logic applies to.
CALIBRATION_LOGIC = 2

# A speaker encoder cannot embed an arbitrarily short clip. WavLM-SV downsamples
# by 320 and then runs a TDNN stack whose dilated kernels need ~15 frames, so
# anything under about 0.3 s reaches conv1d with fewer frames than the kernel and
# raises. 0.4 s leaves margin for all three encoders.
#
# This is not a hypothetical: the system under test emitted 89 clips shorter than
# 0.3 s, 74 of them in one subset and most of them exactly 46 ms — silence where a
# sentence should be. Those are failures of the system, so they are counted and
# reported, not allowed to kill a four-hour run.
MIN_EMBED_SAMPLES = 6400   # 0.4 s at 16 kHz

CARRY = ["speaker_id", "speaker_gender", "len_bin", "prompt_dur", "prompt_dur_bin",
         "sim_ref_dur", "sim_ref_dur_bin", "has_sim_ref", "has_gt",
         "gt_same_speaker", "anchor_sim_wavlm_sv", "anchor_sim_wavlm_ft",
         "anchor_sim_ecapa"]


class SimStage:
    def __init__(self, config, benchmark, synthesis, cache):
        self.config = config
        self.benchmark = benchmark
        self.synthesis = synthesis
        self.cache = cache
        self.row_limit: Optional[int] = None
        self.published = PublishedCalibration(
            config.benchmark.reports_dir, config.sim.calibration.per_row_csv)
        self.calibrator = VoiceCalibrator(config)
        self.calibrations: list[Calibration] = []

    # -- planning -----------------------------------------------------------

    def _unit(self, encoder_name: str, subset: SynthesisSubset) -> StageUnit:
        return StageUnit(
            stage=STAGE,
            name=f"{subset.name}__{encoder_name}",
            parameters={
                "encoder": encoder_name,
                "batch": self.config.sim.batch,
                "subset": subset.name,
                "language": subset.language,
                "voice": subset.voice.name,
                "voice_type": subset.voice.type,
                "audio": {
                    "vad": self.config.audio.vad.enabled,
                    "peak_dbfs": self.config.audio.peak_dbfs,
                },
                # Which clips are short enough to skip decides which rows are
                # scored, so it belongs in the key for every unit.
                "min_embed_samples": MIN_EMBED_SAMPLES,
                "calibration": {
                    "floor_pairs": self.config.sim.calibration.floor.pairs,
                    "anchor_chunk_sec": self.config.sim.calibration.anchor.chunk_sec,
                    "anchor_hop_sec": self.config.sim.calibration.anchor.hop_sec,
                    "anchor_max_chunks": self.config.sim.calibration.anchor.max_chunks,
                    # Only the voices that are actually calibrated here care how
                    # the calibration is computed; per-row voices read published
                    # coefficients and are unaffected by changes to it.
                    **({} if subset.voice.is_per_row
                       else {"logic": CALIBRATION_LOGIC}),
                },
                "engine_build": describe(),
                "row_limit": self.row_limit,
            },
        )

    # -- reference embeddings ------------------------------------------------

    def _prompt_embeddings(self, encoder, language: str) -> dict[str, np.ndarray]:
        """One embedding per prompt speaker, from the benchmark's stored audio."""
        from msbench.audio import read_cell

        table = self.benchmark.audio_column(language, "prompt_audio").to_pydict()
        metadata = self.benchmark.metadata(language)
        speakers = dict(zip(metadata["utt"], metadata["speaker_id"]))

        first: dict[str, np.ndarray] = {}
        for utt, cell in tqdm(list(zip(table["utt"], table["prompt_audio"])),
                              desc=f"{language} · prompts · {encoder.name}",
                              unit="clip", leave=False):
            speaker = speakers.get(utt)
            if speaker is None or speaker in first or cell is None:
                continue
            # Stored benchmark audio is already canonical; re-trimming it with a
            # different VAD build than the one that produced the pack would move
            # the anchor. Preprocess synthesis, leave the pack alone.
            wave = read_cell(cell, stored=True)
            if wave is None or len(wave) < MIN_EMBED_SAMPLES:
                continue
            first[speaker] = encoder.embed([wave])[0]
        logger.info("%s: %d prompt speaker embeddings (%s)", language, len(first),
                    encoder.name)
        return first

    def _fixed_reference(self, encoder, subset: SynthesisSubset):
        """Embed a reference voice that lives in a file rather than in the corpus."""
        from msbench.audio import prepare

        waveform, sample_rate = read_audio_file(subset.voice.path)
        prepared = prepare(waveform, sample_rate,
                           trim=self.config.audio.vad.enabled, peak=True)
        return prepared, encoder.embed([prepared])[0]

    # -- execution ----------------------------------------------------------

    def run(self, subsets: list[SynthesisSubset]) -> dict[str, pd.DataFrame]:
        results: dict[str, pd.DataFrame] = {}

        for encoder_name in self.config.sim.encoders:
            pending = [s for s in subsets
                       if self.cache.load(self._unit(encoder_name, s)) is None]
            for subset in subsets:
                unit = self._unit(encoder_name, subset)
                cached = self.cache.load(unit)
                if cached is not None:
                    frame, summary = cached
                    results[unit.name] = frame
                    self._remember(summary)
                    logger.info("%s: cached (SIM %.4f)", unit.name,
                                summary.get("mean", float("nan")))
            if not pending:
                continue

            encoder = build_encoder(encoder_name, self.config)
            prompt_cache: dict[str, dict[str, np.ndarray]] = {}

            for subset in pending:
                if subset.language not in prompt_cache:
                    prompt_cache[subset.language] = self._prompt_embeddings(
                        encoder, subset.language)
                unit = self._unit(encoder_name, subset)
                frame, summary = self._score(encoder, subset,
                                             prompt_cache[subset.language])
                self.cache.save(unit, frame, summary)
                results[unit.name] = frame

        return results

    def _score(self, encoder, subset: SynthesisSubset,
               prompt_embeddings: dict[str, np.ndarray]):
        from msbench.audio import read_cell

        columns = self.config.synthesis.columns
        language = subset.language
        started = time.time()

        metadata = self.benchmark.metadata(language).set_index("utt")
        speaker_of = metadata["speaker_id"].to_dict()

        if subset.voice.is_per_row:
            reference_waveform, reference_embedding = None, None
            calibration = self.published.get(language, encoder.name, subset.voice.name)
        else:
            reference_waveform, reference_embedding = self._fixed_reference(
                encoder, subset)
            calibration = self.calibrator.calibrate(
                encoder, language, subset.voice.name,
                reference_waveform, reference_embedding, prompt_embeddings)

        rows: list[dict] = []
        n_missing = 0
        n_too_short = 0
        n_failed = 0
        seen = 0
        progress = tqdm(total=subset.n_rows, desc=f"{subset.name} · {encoder.name}",
                        unit="clip", leave=False)

        for row in self.synthesis.iter_rows(
                subset, columns=[columns.key, columns.audio]):
            if self.row_limit and seen >= self.row_limit:
                break
            seen += 1
            progress.update(1)

            utt = row[columns.key]
            if subset.voice.is_per_row:
                reference = prompt_embeddings.get(speaker_of.get(utt))
            else:
                reference = reference_embedding
            if reference is None:
                n_missing += 1
                continue

            wave = read_cell(row[columns.audio], stored=False,
                             trim=self.config.audio.vad.enabled, peak=True)
            if wave is None:
                n_missing += 1
                continue
            if len(wave) < MIN_EMBED_SAMPLES:
                # Too short for the encoder to embed at all — a system failure,
                # recorded as one rather than crashing the stage.
                n_too_short += 1
                continue

            try:
                embedding = encoder.embed([wave])[0]
            except Exception as exc:  # noqa: BLE001 — one clip must not cost a subset
                n_failed += 1
                logger.warning("%s: %s could not embed %s (%.3f s): %s",
                               subset.name, encoder.name, utt,
                               len(wave) / self.config.audio.target_sample_rate,
                               str(exc)[:160])
                continue
            rows.append({"utt": utt, "sim": float(reference @ embedding)})
        progress.close()

        if n_too_short or n_failed:
            logger.warning("%s · %s: %d clip(s) too short to embed, %d failed",
                           subset.name, encoder.name, n_too_short, n_failed)

        frame = pd.DataFrame(rows)
        if frame.empty:
            raise RuntimeError(f"{subset.name}/{encoder.name}: nothing was scored")

        for column in CARRY:
            if column in metadata.columns:
                frame[column] = frame.utt.map(metadata[column])

        frame["sim_norm"] = calibration.normalise(frame["sim"].to_numpy())
        frame["encoder"] = encoder.name
        frame["lang"] = language
        frame["voice"] = subset.voice.name
        frame["subset"] = subset.name
        frame["model"] = self.config.run.name

        summary = self._summarise(frame, calibration, subset, encoder,
                                  n_missing, started)
        summary["n_too_short_to_embed"] = n_too_short
        summary["n_embed_failures"] = n_failed
        self._remember(summary)
        self._log(summary, subset, encoder, calibration)
        return frame, summary

    def _summarise(self, frame, calibration: Calibration, subset, encoder,
                   n_missing, started) -> dict:
        import msbench.stats as stats

        cluster = self.config.stats.bootstrap.cluster_by
        similarity = frame["sim"]
        summary = {
            "subset": subset.name,
            "lang": subset.language,
            "voice": subset.voice.name,
            "voice_type": subset.voice.type,
            "encoder": encoder.name,
            "encoder_scale": getattr(encoder, "scale", ""),
            "model": self.config.run.name,
            "n_rows": len(frame),
            "n_missing": n_missing,
            "n_speakers": int(frame[cluster].nunique()) if cluster in frame else 0,
            "mean": float(similarity.mean()),
            "median": float(similarity.median()),
            "p05": float(similarity.quantile(0.05)),
            "p95": float(similarity.quantile(0.95)),
            "std": float(similarity.std()),
            "sim_norm_mean": float(frame["sim_norm"].mean()),
            # The share of clips an impostor-level score would not distinguish
            # from a stranger: below the floor's p95 is a false-accept rate of
            # one in twenty by construction.
            "below_floor_p95": float((similarity < calibration.floor_p95).mean()),
            **calibration.to_dict(),
            "seconds": round(time.time() - started, 1),
        }
        if cluster in frame and len(frame) > 1:
            _, lo, hi = stats.cluster_bootstrap(
                frame, stats.macro("sim"), cluster_col=cluster,
                n_boot=self.config.stats.bootstrap.replicates,
                seed=self.config.run.seed,
                alpha=1 - self.config.stats.bootstrap.confidence,
            )
            summary["mean_ci"] = [round(lo, 6), round(hi, 6)]
        return summary

    def _remember(self, summary: dict) -> None:
        """Keep every calibration used, so the report can show the scales."""
        self.calibrations.append(summary)

    @staticmethod
    def _log(summary, subset, encoder, calibration: Calibration) -> None:
        ci = summary.get("mean_ci")
        interval = f" 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""
        logger.info(
            "%s · %s: SIM %.4f%s | norm %.3f | anchor %.4f (%s) | floor %.4f "
            "| below floor p95 %.1f%% | n=%d (%.0fs)",
            subset.name, encoder.name, summary["mean"], interval,
            summary["sim_norm_mean"], calibration.anchor, calibration.anchor_kind,
            calibration.floor_mean, 100 * summary["below_floor_p95"],
            summary["n_rows"], summary["seconds"],
        )
