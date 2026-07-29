"""
Intelligibility: every subset read by every recogniser that covers its language.

One recogniser cannot separate what the system got wrong from what that
recogniser is bad at. Whisper's internal language model repairs half-swallowed
synthesis into the word the sentence implies, and credits the system with
intelligibility it did not produce; MMS is CTC without a language model and
emits what it heard; Scribe is a second strong-LM read from another vendor. The
gaps between them are a measurement, not noise.

Decoding, preprocessing, normalisation and aggregation all come from the
benchmark's own code, so these numbers share a ruler with the published human
anchors. What this module adds is the loop over our data layout, the per-row
provenance needed to slice the result, and two diagnostics the protocol has no
opinion about: how long the synthesis actually is, and how often a recogniser
emitted a typographic apostrophe.
"""
from __future__ import annotations

import logging
import time
from typing import Iterator, Optional

import pandas as pd
from tqdm import tqdm

from ..cache import StageUnit
from ..config import AsrEngineConfig
from ..engine import build_asr, describe
from ..sources import SynthesisSubset

logger = logging.getLogger(__name__)

STAGE = "asr"

# Shorter than 10 ms of audio is not a clip; msbench's own driver uses this bound.
MIN_SAMPLES = 160

TYPOGRAPHIC_APOSTROPHES = "’ʼ՚＇"

# Bump when the per-utterance schema changes: it is part of the cache key, so a
# frame written by older code is recomputed instead of silently missing columns.
SCHEMA_VERSION = 2

# Benchmark columns carried into the per-utterance output. The cluster column is
# what makes a bootstrap honest, and the anchor counts are what make the paired
# comparison exact — every recogniser is paired against the *same* recogniser's
# reading of a human recording, which is why all three variants are carried.
CARRY = [
    "speaker_id", "speaker_gender", "len_bin", "n_words", "prompt_dur",
    "prompt_dur_bin", "gt_dur", "has_gt", "gt_speaker_id",
    # primary recogniser (whisper, or gigaam for ky)
    "anchor_asr", "anchor_wer", "anchor_cer", "anchor_subs", "anchor_dels",
    "anchor_ins", "anchor_n_ref_words", "anchor_cer_err", "anchor_n_ref_chars",
    # MMS
    "anchor_wer_mms", "anchor_cer_mms", "anchor_subs_mms", "anchor_dels_mms",
    "anchor_ins_mms", "anchor_cer_err_mms",
    # ElevenLabs Scribe
    "anchor_wer_scribe", "anchor_cer_scribe", "anchor_subs_scribe",
    "anchor_dels_scribe", "anchor_ins_scribe", "anchor_cer_err_scribe",
]


class AsrStage:
    """Transcribes synthesis and scores it against the benchmark text."""

    def __init__(self, config, benchmark, synthesis, cache):
        self.config = config
        self.benchmark = benchmark
        self.synthesis = synthesis
        self.cache = cache
        # Smoke-test knob. It is part of the cache key: a truncated run must
        # never satisfy a later full one.
        self.row_limit: Optional[int] = None

    # -- planning -----------------------------------------------------------

    def units(self, subsets: list[SynthesisSubset]) -> list[tuple[AsrEngineConfig, SynthesisSubset]]:
        """Engine-major order: a recogniser is loaded once and reused."""
        plan = []
        for engine in self.config.asr.enabled_engines():
            for subset in subsets:
                if engine.covers(subset.language):
                    plan.append((engine, subset))
        return plan

    def _unit(self, engine: AsrEngineConfig, subset: SynthesisSubset) -> StageUnit:
        return StageUnit(
            stage=STAGE,
            name=f"{subset.name}__{engine.name}",
            parameters={
                "engine": engine.name,
                "options": engine.options,
                "subset": subset.name,
                "language": subset.language,
                "audio": {
                    "target_sample_rate": self.config.audio.target_sample_rate,
                    "resampler": self.config.audio.resampler,
                    "vad": self.config.audio.vad.enabled,
                    "pad_ms": self.config.audio.vad.pad_ms,
                    "peak_dbfs": self.config.audio.peak_dbfs,
                },
                "text": {
                    "fold_typographic_apostrophes":
                        self.config.text.fold_typographic_apostrophes,
                },
                "engine_build": describe(),
                "row_limit": self.row_limit,
                "schema": SCHEMA_VERSION,
            },
        )

    # -- execution ----------------------------------------------------------

    def run(self, subsets: list[SynthesisSubset]) -> dict[str, pd.DataFrame]:
        plan = self.units(subsets)
        results: dict[str, pd.DataFrame] = {}
        backend = None
        loaded: Optional[str] = None

        for engine, subset in plan:
            unit = self._unit(engine, subset)
            cached = self.cache.load(unit)
            if cached is not None:
                frame, summary = cached
                results[unit.name] = frame
                logger.info("%s: cached (WER corpus %.4f)", unit.name,
                            summary.get("wer_corpus", float("nan")))
                continue

            if loaded != engine.name:
                backend = build_asr(engine)
                loaded = engine.name

            frame, summary = self._score(backend, engine, subset)
            self.cache.save(unit, frame, summary)
            results[unit.name] = frame

        return results

    def _batches(self, subset: SynthesisSubset, batch_size: int) -> Iterator[list[dict]]:
        columns = [
            self.config.synthesis.columns.key,
            self.config.synthesis.columns.audio,
            self.config.synthesis.columns.duration,
            self.config.synthesis.columns.latency,
        ]
        batch: list[dict] = []
        seen = 0
        for row in self.synthesis.iter_rows(subset, columns=columns):
            if self.row_limit and seen >= self.row_limit:
                break
            seen += 1
            batch.append(row)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    @staticmethod
    def _transcribe(backend, waves, language) -> tuple[list[Optional[str]], list[Optional[str]]]:
        """
        Transcribe a batch, degrading to one clip at a time if the batch fails.

        A single unreadable clip must not cost the whole subset. This happens for
        real: Scribe rejects audio below its minimum length with HTTP 400, and a
        synthesis so short the recogniser refuses it is a failure of the system
        under test — it belongs in `n_missing`, counted and visible, not in a
        traceback that discards 1,499 good rows with it.
        """
        try:
            return list(backend.transcribe(waves, language)), [None] * len(waves)
        except Exception as batch_error:  # noqa: BLE001 — backends raise anything
            logger.warning("batch of %d failed (%s); retrying one at a time",
                           len(waves), str(batch_error)[:160])

        hypotheses: list[Optional[str]] = []
        failures: list[Optional[str]] = []
        for wave in waves:
            try:
                hypotheses.append(backend.transcribe([wave], language)[0])
                failures.append(None)
            except Exception as one_error:  # noqa: BLE001
                hypotheses.append(None)
                failures.append(f"{type(one_error).__name__}: {str(one_error)[:200]}")
        n_failed = sum(f is not None for f in failures)
        if n_failed:
            logger.warning("%d/%d clip(s) could not be transcribed at all",
                           n_failed, len(waves))
        return hypotheses, failures

    def _score(self, backend, engine: AsrEngineConfig, subset: SynthesisSubset):
        from msbench.audio import read_cell
        from msbench.metrics import score_pairs
        from msbench.normalize import normalize

        columns = self.config.synthesis.columns
        language = subset.language
        batch_size = int(engine.option("batch", 16) or 16)
        started = time.time()

        rows: list[dict] = []
        n_missing = 0
        progress = tqdm(total=subset.n_rows, desc=f"{subset.name} · {engine.name}",
                        unit="clip", leave=False)

        for batch in self._batches(subset, batch_size):
            waves, metas = [], []
            for row in batch:
                # stored=False routes synthesis through the full parity pipeline;
                # the stored benchmark audio must never take this path.
                wav = read_cell(row[columns.audio], stored=False,
                                trim=self.config.audio.vad.enabled,
                                peak=True)
                meta = {
                    "utt": row[columns.key],
                    "gen_dur": row.get(columns.duration),
                    "latency_s": row.get(columns.latency),
                }
                if wav is None or len(wav) < MIN_SAMPLES:
                    n_missing += 1
                    rows.append({**meta, "hyp_raw": None, "missing": True,
                                 "synth_dur": 0.0})
                    continue
                meta["synth_dur"] = len(wav) / self.config.audio.target_sample_rate
                waves.append(wav)
                metas.append(meta)

            if waves:
                hypotheses, failures = self._transcribe(backend, waves, language)
                for meta, hypothesis, failure in zip(metas, hypotheses, failures):
                    if hypothesis is None:
                        n_missing += 1
                        rows.append({**meta, "hyp_raw": None, "missing": True,
                                     "failure": failure})
                    else:
                        rows.append({**meta, "hyp_raw": hypothesis, "missing": False,
                                     "failure": None})
            progress.update(len(batch))
        progress.close()

        frame = pd.DataFrame(rows)
        scored = frame[~frame.missing].copy()

        reference = self.benchmark.metadata(language).set_index("utt")
        scored["text"] = scored.utt.map(reference["text"])
        scored["ref_norm"] = [normalize(t, language) for t in scored.text]
        scored["hyp_norm"] = [normalize(t, language) for t in scored.hyp_raw]

        aggregate = score_pairs(
            list(zip(scored.ref_norm, scored.hyp_norm)),
            n_missing=n_missing,
            utts=list(scored.utt),
        )
        per_utterance = pd.DataFrame(aggregate.per_utt)
        per_utterance = per_utterance.merge(
            scored.drop(columns=["ref_norm", "hyp_norm", "text"]), on="utt", how="left"
        )
        for column in CARRY:
            if column in reference.columns:
                per_utterance[column] = per_utterance.utt.map(reference[column])

        per_utterance["asr"] = engine.name
        per_utterance["lang"] = language
        per_utterance["voice"] = subset.voice.name
        per_utterance["subset"] = subset.name
        per_utterance["model"] = self.config.run.name

        summary = self._summarise(aggregate, per_utterance, engine, subset,
                                  scored, started, backend)
        summary["n_transcription_failures"] = (
            int(frame["failure"].notna().sum()) if "failure" in frame else 0
        )
        summary["n_below_min_samples"] = int(
            (frame.missing & frame.get("failure", pd.Series(dtype=object)).isna()).sum()
        )
        self._log(summary, subset, engine)
        return per_utterance, summary

    def _summarise(self, aggregate, per_utterance, engine, subset, scored, started,
                   backend):
        import msbench.stats as stats

        cluster = self.config.stats.bootstrap.cluster_by
        summary = aggregate.summary()
        summary.update(
            subset=subset.name,
            lang=subset.language,
            voice=subset.voice.name,
            voice_type=subset.voice.type,
            asr=engine.name,
            model=self.config.run.name,
            # Every decoding parameter that determined these numbers, as the
            # backend itself reports them.
            config=backend.config() if hasattr(backend, "config") else {},
            cluster_col=cluster,
            n_clusters=int(per_utterance[cluster].nunique())
            if cluster in per_utterance else 0,
            # How often a recogniser wrote ’ where the reference has '. The
            # protocol turns both into a space, so this is a sensitivity note
            # rather than a correction — but an unread one would be a silent bias.
            n_hyp_typographic_apostrophe=int(
                scored.hyp_raw.fillna("").str.contains(
                    f"[{TYPOGRAPHIC_APOSTROPHES}]", regex=True).sum()
            ),
            seconds=round(time.time() - started, 1),
        )

        if cluster in per_utterance and len(per_utterance) > 1:
            for label, statistic in (
                ("wer_corpus", stats.wer_corpus),
                ("cer_corpus", stats.cer_corpus),
                ("wer_macro", stats.macro("wer")),
                ("catastrophic_rate",
                 stats.rate_above("wer", self.config.failure.catastrophic_wer)),
            ):
                _, lo, hi = stats.cluster_bootstrap(
                    per_utterance, statistic, cluster_col=cluster,
                    n_boot=self.config.stats.bootstrap.replicates,
                    seed=self.config.run.seed,
                    alpha=1 - self.config.stats.bootstrap.confidence,
                )
                summary[f"{label}_ci"] = [round(lo, 6), round(hi, 6)]
        return summary

    @staticmethod
    def _log(summary: dict, subset: SynthesisSubset, engine: AsrEngineConfig) -> None:
        ci = summary.get("wer_corpus_ci")
        interval = f"  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""
        logger.info(
            "%s · %s: WER corpus %.4f%s | macro %.4f | CER %.4f | "
            "exact %.1f%% | catastrophic %.1f%% | n=%d missing=%d (%.0fs)",
            subset.name, engine.name, summary["wer_corpus"], interval,
            summary["wer_macro"], summary["cer_corpus"],
            100 * summary["exact_match"], 100 * summary["catastrophic_rate"],
            summary["n_scored"], summary["n_missing"], summary["seconds"],
        )
