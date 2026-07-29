"""
Reporting: our rows put next to the published human anchors.

Layout mirrors the benchmark's own `reports/`, so a table here can be read
beside the corresponding table there without translating anything:

    results.md              every headline with its interval
    csv/*.csv               the same tables, machine-readable
    per_utterance/*.parquet S/D/I, transcripts and similarities per utt
    summary.json            one object, every number and the config that made it

Three things this adds to a plain dump of the numbers.

**The anchor comparison is paired.** Comparing two independent confidence
intervals throws away the fact that both were measured on the same sentences.
The benchmark ships the anchor's edit counts per row, so the delta against a
human reading of the same utterance is computed exactly, resampling the shared
speakers once per replicate and applying that resample to both sides.

**Similarity is reported on a readable scale.** A bare cosine cannot be read: the
same 0.90 is excellent under one encoder and indistinguishable from a stranger
under another. Every SIM carries its floor, its anchor and the share of clips
below the impostor p95.

**Naturalness is quarantined.** It has no anchor and is not part of the protocol,
so it lives in its own section and never appears in a WER or SIM table.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import console
from .config import EvalConfig
from .engine import describe

logger = logging.getLogger(__name__)

# Our recogniser -> the benchmark's anchor columns for the same recogniser.
ANCHOR_COLUMNS = {
    "whisper": ("anchor_subs", "anchor_dels", "anchor_ins", "anchor_n_ref_words",
                "anchor_cer_err", "anchor_n_ref_chars", "anchor_wer"),
    "gigaam": ("anchor_subs", "anchor_dels", "anchor_ins", "anchor_n_ref_words",
               "anchor_cer_err", "anchor_n_ref_chars", "anchor_wer"),
    "mms": ("anchor_subs_mms", "anchor_dels_mms", "anchor_ins_mms",
            "anchor_n_ref_words", "anchor_cer_err_mms", "anchor_n_ref_chars",
            "anchor_wer_mms"),
    "elevenlabs": ("anchor_subs_scribe", "anchor_dels_scribe", "anchor_ins_scribe",
                   "anchor_n_ref_words", "anchor_cer_err_scribe",
                   "anchor_n_ref_chars", "anchor_wer_scribe"),
}


def _ci(lo: float, hi: float, digits: int = 4) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "—"
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


class Report:
    def __init__(self, config: EvalConfig, cache):
        self.config = config
        self.cache = cache
        self.output = Path(config.run.output_dir)

    # -- gathering ----------------------------------------------------------

    def _collect(self, stage: str) -> tuple[pd.DataFrame, list[dict]]:
        frames, summaries = [], []
        for frame, summary in self.cache.iter_saved(stage):
            frames.append(frame)
            summaries.append({k: v for k, v in summary.items()
                              if not k.startswith("_")})
        if not frames:
            return pd.DataFrame(), []
        return pd.concat(frames, ignore_index=True), summaries

    # -- tables -------------------------------------------------------------

    def intelligibility(self, per_utterance: pd.DataFrame,
                        summaries: list[dict]) -> pd.DataFrame:
        """WER and CER per subset and recogniser, against the human anchor."""
        import msbench.stats as stats

        cluster = self.config.stats.bootstrap.cluster_by
        rows = []

        for summary in summaries:
            subset, asr = summary["subset"], summary["asr"]
            ours = per_utterance[(per_utterance.subset == subset)
                                 & (per_utterance.asr == asr)]
            anchor_wer, delta, delta_ci, p_better = self._against_anchor(ours, asr, cluster)

            wer_ci = summary.get("wer_corpus_ci") or [np.nan, np.nan]
            rows.append({
                "subset": subset,
                "lang": summary["lang"],
                "voice": summary["voice"],
                "asr": asr,
                "n": summary["n_scored"],
                "speakers": summary.get("n_clusters", 0),
                "WER corpus": round(summary["wer_corpus"], 4),
                "95% CI": _ci(*wer_ci),
                "WER macro": round(summary["wer_macro"], 4),
                "CER corpus": round(summary["cer_corpus"], 4),
                "exact": f"{100 * summary['exact_match']:.1f}%",
                "catastrophic": f"{100 * summary['catastrophic_rate']:.1f}%",
                "human anchor": round(anchor_wer, 4) if np.isfinite(anchor_wer) else "—",
                "delta vs anchor": round(delta, 4) if np.isfinite(delta) else "—",
                "delta 95% CI": delta_ci,
                "P(better than human)": (f"{p_better:.2f}"
                                         if np.isfinite(p_better) else "—"),
                "missing": summary["n_missing"],
                "empty ref": summary["n_empty_ref"],
            })
        return pd.DataFrame(rows).sort_values(["lang", "voice", "asr"])

    def _against_anchor(self, ours: pd.DataFrame, asr: str, cluster: str):
        """
        Paired delta between our synthesis and a human recording of the same text.

        The benchmark carries the anchor's own S/D/I per row, so the human side of
        the pair needs no recomputation — and because both sides are the same
        utterances read by the same recogniser, the difference isolates the
        system rather than the sentences.
        """
        import msbench.stats as stats

        columns = ANCHOR_COLUMNS.get(asr)
        if not columns or ours.empty or columns[0] not in ours.columns:
            return np.nan, np.nan, "—", np.nan

        subs, dels, ins, n_words, cer_err, n_chars, _ = columns
        anchor = ours[["utt", cluster, subs, dels, ins, n_words]].copy()
        anchor = anchor.rename(columns={subs: "subs", dels: "dels", ins: "ins",
                                        n_words: "n_ref_words"})
        anchor = anchor.dropna(subset=["subs", "dels", "ins", "n_ref_words"])
        if anchor.empty:
            return np.nan, np.nan, "—", np.nan

        anchor_wer = float(
            (anchor.subs + anchor.dels + anchor.ins).sum() / anchor.n_ref_words.sum())

        mine = ours.dropna(subset=["subs", "dels", "ins", "n_ref_words"])
        try:
            result = stats.paired_bootstrap(
                mine, anchor, stats.wer_corpus, key="utt", cluster_col=cluster,
                n_boot=self.config.stats.bootstrap.replicates,
                seed=self.config.run.seed,
                alpha=1 - self.config.stats.bootstrap.confidence,
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("paired comparison failed for %s: %s", asr, exc)
            return anchor_wer, np.nan, "—", np.nan

        return (anchor_wer, result["delta"], _ci(result["lo"], result["hi"]),
                result["p_a_better"])

    def disagreement(self, per_utterance: pd.DataFrame) -> pd.DataFrame:
        """Where the recognisers disagree — acoustics versus expectation."""
        import msbench.stats as stats

        cluster = self.config.stats.bootstrap.cluster_by
        rows = []
        for subset, group in per_utterance.groupby("subset"):
            engines = sorted(group.asr.unique())
            for i, a in enumerate(engines):
                for b in engines[i + 1:]:
                    left = group[group.asr == a]
                    right = group[group.asr == b]
                    shared = set(left.utt) & set(right.utt)
                    if len(shared) < 2:
                        continue
                    left = left[left.utt.isin(shared)]
                    right = right[right.utt.isin(shared)]
                    result = stats.paired_bootstrap(
                        left, right, stats.wer_corpus, key="utt",
                        cluster_col=cluster,
                        n_boot=self.config.stats.bootstrap.replicates,
                        seed=self.config.run.seed,
                        alpha=1 - self.config.stats.bootstrap.confidence,
                    )
                    identical = left.set_index("utt").hyp_norm.reindex(sorted(shared)) \
                        .eq(right.set_index("utt").hyp_norm.reindex(sorted(shared)))
                    rows.append({
                        "subset": subset,
                        "A": a, "B": b,
                        "WER A": round(float(stats.wer_corpus(left)), 4),
                        "WER B": round(float(stats.wer_corpus(right)), 4),
                        "delta (A-B)": round(result["delta"], 4),
                        "95% CI": _ci(result["lo"], result["hi"]),
                        "identical transcripts": f"{100 * identical.mean():.1f}%",
                        "n shared": len(shared),
                    })
        return pd.DataFrame(rows)

    def similarity(self, summaries: list[dict]) -> pd.DataFrame:
        rows = []
        for summary in summaries:
            ci = summary.get("mean_ci") or [np.nan, np.nan]
            rows.append({
                "subset": summary["subset"],
                "lang": summary["lang"],
                "voice": summary["voice"],
                "encoder": summary["encoder"],
                "n": summary["n_rows"],
                "speakers": summary.get("n_speakers", 0),
                "SIM": round(summary["mean"], 4),
                "95% CI": _ci(*ci),
                "sim_norm": round(summary["sim_norm_mean"], 3),
                "anchor": round(summary["anchor"], 4),
                "anchor kind": summary["anchor_kind"],
                "floor": round(summary["floor_mean"], 4),
                "floor p95": round(summary["floor_p95"], 4),
                "usable range": round(summary["usable_range"], 4),
                "below floor p95": f"{100 * summary['below_floor_p95']:.1f}%",
                "calibration": summary["calibration_source"],
            })
        return pd.DataFrame(rows).sort_values(["lang", "voice", "encoder"])

    def failures(self, per_utterance: pd.DataFrame,
                 summaries: list[dict]) -> pd.DataFrame:
        """What a mean WER hides: looping, truncation, clips nothing could read."""
        rows = []
        runaway = self.config.failure.runaway_duration_ratio
        truncated = self.config.failure.min_duration_ratio

        for summary in summaries:
            subset, asr = summary["subset"], summary["asr"]
            if asr != self.config.asr.primary_engine(summary["lang"]):
                continue
            group = per_utterance[(per_utterance.subset == subset)
                                  & (per_utterance.asr == asr)]
            ratio = pd.Series(dtype=float)
            if {"synth_dur", "gt_dur"}.issubset(group.columns):
                valid = group.gt_dur.gt(0)
                ratio = (group.synth_dur[valid] / group.gt_dur[valid]).dropna()
            rows.append({
                "subset": subset,
                "lang": summary["lang"],
                "voice": summary["voice"],
                "asr": asr,
                "catastrophic (WER>0.5)": f"{100 * summary['catastrophic_rate']:.1f}%",
                "runaway (dur>%.1fx)" % runaway:
                    f"{100 * float((ratio > runaway).mean()):.1f}%" if len(ratio) else "—",
                "truncated (dur<%.1fx)" % truncated:
                    f"{100 * float((ratio < truncated).mean()):.1f}%" if len(ratio) else "—",
                "median dur ratio": round(float(ratio.median()), 3) if len(ratio) else "—",
                "max dur ratio": round(float(ratio.max()), 1) if len(ratio) else "—",
                "unreadable clips": summary.get("n_transcription_failures", 0)
                + summary.get("n_below_min_samples", 0),
                "hyp with ’": summary.get("n_hyp_typographic_apostrophe", 0),
            })
        return pd.DataFrame(rows).sort_values(["lang", "voice"])

    def naturalness(self, summaries: list[dict]) -> pd.DataFrame:
        rows = []
        for summary in summaries:
            row = {"subset": summary["subset"], "lang": summary["lang"],
                   "voice": summary["voice"], "n": summary["n_rows"]}
            for key, value in summary.items():
                if key.endswith("_mean") and isinstance(value, (int, float)):
                    row[key[:-5]] = round(float(value), 3)
            rows.append(row)
        frame = pd.DataFrame(rows)
        return frame.sort_values(["lang", "voice"]) if not frame.empty else frame

    # -- rendering ----------------------------------------------------------

    def build(self) -> dict:
        asr_frame, asr_summaries = self._collect("asr")
        sim_frame, sim_summaries = self._collect("sim")
        quality_frame, quality_summaries = self._collect("quality")

        if asr_frame.empty and sim_frame.empty:
            raise SystemExit(
                "nothing to report — run `make evaluate` first, or point "
                "run.cache_dir at a completed run"
            )

        tables: dict[str, pd.DataFrame] = {}
        if not asr_frame.empty:
            tables["intelligibility"] = self.intelligibility(asr_frame, asr_summaries)
            tables["disagreement"] = self.disagreement(asr_frame)
            tables["failures"] = self.failures(asr_frame, asr_summaries)
        if sim_summaries:
            tables["similarity"] = self.similarity(sim_summaries)
        if quality_summaries:
            tables["naturalness"] = self.naturalness(quality_summaries)

        self._write(tables, asr_frame, sim_frame, quality_frame,
                    asr_summaries, sim_summaries, quality_summaries)
        return tables

    def _write(self, tables, asr_frame, sim_frame, quality_frame,
               asr_summaries, sim_summaries, quality_summaries) -> None:
        self.output.mkdir(parents=True, exist_ok=True)

        if self.config.report.csv:
            directory = self.output / "csv"
            directory.mkdir(exist_ok=True)
            for name, frame in tables.items():
                frame.to_csv(directory / f"{name}.csv", index=False)

        if self.config.report.per_utterance_parquet:
            directory = self.output / "per_utterance"
            directory.mkdir(exist_ok=True)
            for name, frame in (("asr", asr_frame), ("sim", sim_frame),
                                ("quality", quality_frame)):
                if not frame.empty:
                    frame.to_parquet(directory / f"{name}.parquet", index=False)

        if self.config.report.json_summary:
            payload = {
                "run": self.config.run.name,
                "config": str(self.config.path),
                "engine": describe(),
                "asr": asr_summaries,
                "sim": sim_summaries,
                "quality": quality_summaries,
            }
            (self.output / "summary.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8")

        if self.config.report.markdown:
            (self.output / "results.md").write_text(self._markdown(tables),
                                                    encoding="utf-8")
        console.ok(f"reports written to {self.output}")

    def _markdown(self, tables: dict[str, pd.DataFrame]) -> str:
        engine = describe()
        parts = [
            f"# {self.config.run.name} — evaluation against the "
            "Multilingual Speech Benchmark v2.0",
            "",
            "Measured with the benchmark's own code, so every number below shares a "
            "ruler with the published human anchors: preprocessing (mono, 16 kHz "
            "soxr HQ, Silero-VAD trimmed with a 50 ms pad, peak −1 dBFS), text "
            "normalisation, corpus-level aggregation Σ(S+D+I)/Σ N_ref, and cluster "
            "bootstraps resampling speakers rather than rows.",
            "",
            f"Engine: `{engine.get('msbench_path')}`.",
            "",
        ]

        sections = [
            ("intelligibility",
             "## Intelligibility",
             "`delta vs anchor` is a **paired** comparison: the same utterances, "
             "the same recogniser, our synthesis against a human recording of the "
             "same text, resampling the shared speakers once per replicate. "
             "Negative means we are more intelligible than the human anchor. "
             "`catastrophic` is the share above 50 % WER — the indicator that "
             "catches looping and dropped clauses long before it moves the mean."),
            ("disagreement",
             "## Where the recognisers disagree",
             "MMS decodes with CTC and no language model; Whisper and Scribe repair "
             "what they half-hear. A large gap means the intelligible reading "
             "depends on the listener's expectations rather than on the acoustics."),
            ("similarity",
             "## Speaker similarity",
             "`sim_norm` places the cosine on a readable scale: 0 is "
             "indistinguishable from an impostor, 1 is as close as two recordings "
             "of the same person. `below floor p95` is the share of clips a "
             "stranger would beat one time in twenty. For our own reference voices "
             "the anchor is `intra_session` — measured between chunks of one "
             "recording, sharing a microphone and a room, so it is optimistic and "
             "bounds the scale rather than describing a human ceiling."),
            ("failures",
             "## Failure modes",
             "Duration ratio is synthesis length over the length of the human "
             "recording of the same sentence. `hyp with ’` counts rows where a "
             "recogniser wrote a typographic apostrophe; the protocol turns it into "
             "a space, so this is a sensitivity note, not a correction."),
            ("naturalness",
             "## Naturalness — outside the benchmark protocol",
             "The benchmark measures no naturalness axis and publishes no anchor "
             "for one. These numbers have no reference point in this dataset and "
             "must not be read against the tables above. DNSMOS is computed on the "
             "canonical 16 kHz audio, so it is comparable with the benchmark's own "
             "`qc_dnsmos_*` columns for prompt and ground-truth recordings; NISQA "
             "runs on the delivered audio at 48 kHz."),
        ]

        for key, heading, blurb in sections:
            frame = tables.get(key)
            if frame is None or frame.empty:
                continue
            parts += [heading, "", blurb, "", frame.to_markdown(index=False), ""]

        return "\n".join(parts)
