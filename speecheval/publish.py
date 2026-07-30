"""
Publishing a run to the Hub as a research pool.

The destination repository already holds the synthesised audio in twenty configs.
Metrics are added beside it under one dated folder, which buys three things:

* **the audio cannot be orphaned.** The card's `dataset_info` is what makes those
  twenty configs loadable. Declaring `configs:` for our tables would take over
  data-file resolution for the whole dataset, so the card is left untouched and
  these tables are loaded by explicit path instead;
* **runs accumulate.** A second model, or the same model re-measured, lands in its
  own folder. Nothing is overwritten and two runs can be compared directly;
* **a run is self-contained.** The pools, the aggregates, the SIM calibration, the
  provenance and the report ship together, so a number found here a year from now
  can be traced to the code and parameters that produced it.

What is deliberately not uploaded: the audio, which is already in the repository,
and the benchmark's own columns beyond the join keys and the anchor counts — those
live in the benchmark repository, and a copy would drift from it.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from . import console
from .config import EvalConfig
from .engine import describe

logger = logging.getLogger(__name__)

# Aggregate tables, as produced by the report step.
SUMMARY_TABLES = ("intelligibility", "disagreement", "similarity", "failures",
                  "naturalness")


@dataclass
class Manifest:
    """What was assembled, for the console and for the folder's own README."""

    folder: str
    files: list[tuple[str, int, str]]     # (path in repo, bytes, description)

    @property
    def total_bytes(self) -> int:
        return sum(size for _, size, _ in self.files)


class Publisher:
    def __init__(self, config: EvalConfig, cache, date: str):
        self.config = config
        self.cache = cache
        self.push = config.report.push
        self.date = date
        self.folder = self.push.folder_for(date)
        self.source = Path(config.run.output_dir)
        self.staging = self.source / "publish" / self.folder

    # -- derived tables -----------------------------------------------------

    def _calibration(self, sim_summaries: list[dict]) -> pd.DataFrame:
        """
        The two ends of every SIM scale actually used.

        Without this table `sim_norm` cannot be re-derived or challenged — and it
        needs challenging, because for our own reference voices the anchor is
        measured between chunks of a single recording and is optimistic by
        construction.
        """
        seen, rows = set(), []
        for summary in sim_summaries:
            key = (summary["lang"], summary["voice"], summary["encoder"])
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "model": summary["model"],
                "lang": summary["lang"],
                "voice": summary["voice"],
                "voice_type": summary["voice_type"],
                "encoder": summary["encoder"],
                "encoder_scale": summary.get("encoder_scale", ""),
                "anchor": summary["anchor"],
                "anchor_kind": summary["anchor_kind"],
                "floor_mean": summary["floor_mean"],
                "floor_p95": summary["floor_p95"],
                "usable_range": summary["usable_range"],
                "n_floor_pairs": summary["n_floor_pairs"],
                "calibration_source": summary["calibration_source"],
                "sim_norm_formula": "(sim - floor_mean) / (anchor - floor_mean)",
                "note": summary.get("note", ""),
            })
        return pd.DataFrame(rows).sort_values(["lang", "voice", "encoder"])

    def _provenance(self, summaries: dict[str, list[dict]]) -> pd.DataFrame:
        """One row per unit of work: what ran, with which parameters, for how long."""
        engine = describe()
        bootstrap = self.config.stats.bootstrap
        rows = []
        for stage, entries in summaries.items():
            for summary in entries:
                rows.append({
                    "model": summary.get("model", self.config.run.name),
                    "run_date": self.date,
                    "stage": stage,
                    "subset": summary.get("subset"),
                    "lang": summary.get("lang"),
                    "voice": summary.get("voice"),
                    "engine": summary.get("asr") or summary.get("encoder"),
                    "engine_config": json.dumps(summary.get("config", {}),
                                                ensure_ascii=False, default=str),
                    "n_rows": summary.get("n_scored", summary.get("n_rows")),
                    "n_missing": summary.get("n_missing", 0),
                    "seconds": summary.get("seconds"),
                    "msbench_version": engine.get("msbench_version"),
                    "msbench_commit": engine.get("msbench_commit"),
                    "seed": self.config.run.seed,
                    "bootstrap_replicates": bootstrap.replicates,
                    "bootstrap_cluster": bootstrap.cluster_by,
                    "confidence": bootstrap.confidence,
                    "aggregation": self.config.stats.aggregation,
                    "vad": self.config.audio.vad.enabled,
                    "vad_pad_ms": self.config.audio.vad.pad_ms,
                    "peak_dbfs": self.config.audio.peak_dbfs,
                    "target_sample_rate": self.config.audio.target_sample_rate,
                    "resampler": self.config.audio.resampler,
                })
        return pd.DataFrame(rows).sort_values(["stage", "subset", "engine"])

    # -- assembly -----------------------------------------------------------

    def build(self) -> Manifest:
        if self.staging.exists():
            shutil.rmtree(self.staging)
        self.staging.mkdir(parents=True)

        files: list[tuple[str, int, str]] = []

        def record(path: Path, description: str) -> None:
            files.append((str(path.relative_to(self.staging)),
                          path.stat().st_size, description))

        summaries = {stage: [s for _, s in self.cache.iter_saved(stage)]
                     for stage in ("asr", "sim", "quality")}
        summaries = {k: [{kk: vv for kk, vv in s.items() if not kk.startswith("_")}
                         for s in v] for k, v in summaries.items()}

        if self.push.wants("per_utterance"):
            target = self.staging / "per_utterance"
            target.mkdir()
            for name in ("asr", "sim", "quality"):
                source = self.source / "per_utterance" / f"{name}.parquet"
                if not source.is_file():
                    continue
                destination = target / f"{name}.parquet"
                shutil.copy2(source, destination)
                rows = len(pd.read_parquet(destination, columns=["utt"]))
                record(destination, f"{rows:,} rows — one per "
                       + {"asr": "utterance x recogniser",
                          "sim": "utterance x encoder",
                          "quality": "utterance"}[name])

        if self.push.wants("summaries"):
            target = self.staging / "summary"
            target.mkdir(exist_ok=True)
            for name in SUMMARY_TABLES:
                source = self.source / "csv" / f"{name}.csv"
                if not source.is_file():
                    continue
                frame = pd.read_csv(source)
                destination = target / f"{name}.parquet"
                frame.to_parquet(destination, index=False)
                record(destination, f"{len(frame)} rows — aggregate table")

        if self.push.wants("calibration") and summaries["sim"]:
            target = self.staging / "summary"
            target.mkdir(exist_ok=True)
            destination = target / "calibration.parquet"
            self._calibration(summaries["sim"]).to_parquet(destination, index=False)
            record(destination, "the anchor and impostor floor behind every sim_norm")

        if self.push.wants("provenance"):
            target = self.staging / "summary"
            target.mkdir(exist_ok=True)
            destination = target / "provenance.parquet"
            self._provenance(summaries).to_parquet(destination, index=False)
            record(destination, "what ran, with which parameters, for how long")

        if self.push.wants("raw_summaries"):
            target = self.staging / "raw"
            target.mkdir(exist_ok=True)
            source = self.source / "summary.json"
            if source.is_file():
                destination = target / "summary.json"
                shutil.copy2(source, destination)
                record(destination, "per-unit summaries exactly as measured")
            destination = target / "eval.yaml"
            shutil.copy2(self.config.path, destination)
            record(destination, "the configuration this run used, verbatim")

        if self.push.wants("report_markdown"):
            source = self.source / "results.md"
            if source.is_file():
                destination = self.staging / "results.md"
                shutil.copy2(source, destination)
                record(destination, "the human-readable report")

        if self.push.wants("log"):
            source = self.source / "run.log"
            if source.is_file():
                destination = self.staging / "run.log"
                shutil.copy2(source, destination)
                record(destination, "the run log")

        manifest = Manifest(folder=self.folder, files=files)
        readme = self.staging / "README.md"
        readme.write_text(self._readme(manifest, summaries), encoding="utf-8")
        manifest.files.insert(0, ("README.md", readme.stat().st_size,
                                  "what this folder is, and how to read it"))
        return manifest

    # -- the folder's own card ----------------------------------------------

    def _readme(self, manifest: Manifest, summaries: dict[str, list[dict]]) -> str:
        repo = self.push.repo
        engine = describe()
        model = self.config.run.name
        languages = ", ".join(self.config.benchmark.languages)
        voices = ", ".join(sorted(self.config.voices))
        recognisers = ", ".join(sorted({s["asr"] for s in summaries["asr"]}))
        encoders = ", ".join(self.config.sim.encoders)

        # Defects worth naming, measured rather than assumed.
        empty = self._count_empty_output()

        return f"""# {model} — evaluation metrics, {self.date}

Measurements of `{model}` against the Multilingual Speech Benchmark v2.0. The
synthesised audio these numbers describe lives in the twenty audio configs of
this same repository; this folder holds only metrics.

Produced by [nineninesix-ai/evaluate-speech-benchmark](https://github.com/nineninesix-ai/evaluate-speech-benchmark).
Methodology and its reference implementation:
[nineninesix-ai/make-speech-benchmark](https://github.com/nineninesix-ai/make-speech-benchmark)
`v{engine.get('msbench_version')}`, commit `{engine.get('msbench_commit')}` —
preprocessing, text normalisation, error-rate aggregation, the bootstraps, the
recognisers and the speaker encoders are all delegated to it, so these numbers
share a ruler with the benchmark's published human anchors. Verified against that
ruler on this run: the `ky` SIM anchors recompute to 0.5593 / 0.6069 / 0.9214
(`ecapa` / `wavlm_ft` / `wavlm_sv`), the published values to four decimals, and
all 8,200 benchmark texts normalise byte-for-byte to the shipped `text_norm`.

- languages: {languages}
- reference voices: {voices}
- recognisers: {recognisers}
- speaker encoders: {encoders}

## Loading

The root card is intentionally not modified — it is what makes the audio configs
resolvable — so these tables load by path:

```python
from datasets import load_dataset

wer = load_dataset("{repo}", data_files="{self.folder}/per_utterance/asr.parquet")
sim = load_dataset("{repo}", data_files="{self.folder}/per_utterance/sim.parquet")
```

or, for analysis, straight into pandas:

```python
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download("{repo}", "{self.folder}/per_utterance/asr.parquet",
                       repo_type="dataset")
wer = pd.read_parquet(path)
```

## Contents

| file | what it is |
|---|---|
{chr(10).join(f'| `{p}` | {d} |' for p, _, d in manifest.files)}

## Grain and join keys

`utt` is the benchmark's stable identifier and joins three ways: to the benchmark
itself (`nineninesix/multilingual-speech-benchmark`, config = language, split
`main`), to the audio configs in this repository, and between the tables here.

| table | one row per |
|---|---|
| `per_utterance/asr.parquet` | utterance x recogniser |
| `per_utterance/sim.parquet` | utterance x speaker encoder |
| `per_utterance/quality.parquet` | utterance |

`subset` equals `<language>__<voice>`, which is also the name of the audio config,
so metrics and audio line up by construction. A measurement is identified by
`(model, subset, engine, utt)` — `model` is carried everywhere so future runs and
other systems accumulate in this repository rather than replacing each other.

The pools carry the numerators, not only the rates: `subs`, `dels`, `ins`,
`n_ref_words`, `cer_err`, `n_ref_chars`, both normalised strings and the raw
transcript. Any aggregation — corpus, macro, or an arbitrary slice — is
re-derivable without a GPU and without re-running a recogniser. The human
anchor's own edit counts travel in the same rows, so a paired test against a
human reading of the same sentence needs nothing else.

## How to read these numbers

**Compare paired, not side by side.** Two independent confidence intervals throw
away the fact that both systems read the same sentences. `summary/intelligibility`
carries the paired delta against the human anchor, resampling shared speakers once
per replicate and applying that resample to both sides. Intervals everywhere are
95 % cluster bootstraps over **speakers**, not rows: one prompt serves up to seven
utterances and the SIM anchor is constant within a speaker, so a row bootstrap
would understate every interval.

**A bare cosine is meaningless without its scale.** Always join
`summary/calibration` before reading a SIM. Under `wavlm_sv` a similarity of 0.90
can still sit below the impostor p95 — a stranger would score higher one time in
twenty. `wavlm_ft` is the seed-tts-eval scale; `ecapa` comes from outside the
WavLM family and is the one that sees past the benchmark's own selection bias.

**Half the anchors here are optimistic, and say so.** For voices that are
benchmark speakers (`cv_voice`) the anchor and floor are the published ones,
measured between two recordings of one person. For our own reference voices there
is no such pair, so the anchor is measured between chunks of a single recording —
one microphone, one room, one session. Those rows are marked
`anchor_kind = intra_session`, and a `sim_norm` computed against them bounds the
scale rather than describing a human ceiling.

**Cross-language comparison is not supported.** Anchor-normalised comparison is
valid within a language only; differences between anchors are a property of the
recogniser, and for `ky` it is a different model entirely.

## Two measurement traps

Both were hit while building this, and both silently produce plausible numbers on
the wrong scale.

**Speaker embeddings must not be batched.** Padding reaches the pooling layer: the
same 2 s clip scores cos 0.295 (`wavlm_ft`), 0.339 (`wavlm_sv`) and 0.997
(`ecapa`) against itself embedded alone. Everything here was embedded one clip at
a time.

**Punctuation normalises to a space, not to nothing.** `pre-instalado` is two
words, not one, and the difference lands in `n_ref_words` — the denominator of
every corpus-level rate. Verified: all 8,200 benchmark texts normalise
byte-for-byte to the shipped `text_norm`.

## Known defects in this data

- **{empty} clips shorter than 0.3 s**, most of them 46 ms of silence where a
  sentence should be, and {self._empty_worst()}. WER records these as "every word
  wrong", which is true but hides that nothing was produced; `summary/failures`
  counts them separately as `empty output`. Clips too short for a speaker encoder
  to embed are absent from `per_utterance/sim.parquet` and counted in the run
  summaries.
- **No third recogniser.** ElevenLabs Scribe was disabled for this run, so rows
  with `asr = "elevenlabs"` are absent. The benchmark's own `anchor_*_scribe`
  columns are still present for reference. Kyrgyz therefore rests on GigaAM
  cross-checked by one language-model-free CTC baseline.
- **The encoders disagree sharply on two voices.** For `audio_ru` and `audio_en`,
  `wavlm_ft` places most clips below the impostor p95 while `ecapa` places almost
  none there. The ranking agrees, the magnitude differs by an order of magnitude.
  Do not quote one encoder alone for those voices.

## What is not measured here

No human MOS or CMOS. The naturalness tables are model predictions (DNSMOS,
UTMOSv2, NISQA) with no anchor in this dataset, which is why they are kept in
their own table and never mixed into the WER or SIM ones. The benchmark texts
contain no digits, no abbreviations and no long-form passages, so text
normalisation and long-context prosody are untested.

Private, for internal use.
"""

    def _count_empty_output(self) -> int:
        path = self.source / "per_utterance" / "asr.parquet"
        if not path.is_file():
            return 0
        frame = pd.read_parquet(path, columns=["utt", "subset", "synth_dur", "asr"])
        primary = frame.drop_duplicates(["subset", "utt"])
        return int((primary.synth_dur < 0.3).sum())

    def _empty_worst(self) -> str:
        path = self.source / "per_utterance" / "asr.parquet"
        if not path.is_file():
            return "spread across subsets"
        frame = pd.read_parquet(path, columns=["utt", "subset", "synth_dur"])
        primary = frame.drop_duplicates(["subset", "utt"])
        counts = primary[primary.synth_dur < 0.3].subset.value_counts()
        if counts.empty:
            return "spread across subsets"
        return f"{int(counts.iloc[0])} of them in `{counts.index[0]}`"

    # -- upload -------------------------------------------------------------

    def push_to_hub(self, manifest: Manifest) -> str:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(self.push.repo, repo_type="dataset",
                        private=self.push.private, exist_ok=True)

        existing = self._root_card_configs(api)
        api.upload_folder(
            repo_id=self.push.repo,
            repo_type="dataset",
            folder_path=str(self.staging),
            path_in_repo=manifest.folder,
            commit_message=f"metrics: {self.config.run.name} ({self.date})",
        )
        after = self._root_card_configs(api)
        if existing and after != existing:
            console.warn(
                f"the root card's config list changed ({len(existing)} -> {len(after)}) "
                "— check that the audio configs still load"
            )
        else:
            console.ok(f"root card untouched ({len(existing)} config entries intact)")

        return f"https://huggingface.co/datasets/{self.push.repo}/tree/main/{manifest.folder}"

    def _root_card_configs(self, api) -> list[str]:
        """Config names declared in the repository's card, to prove we did not break them."""
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(self.push.repo, "README.md", repo_type="dataset")
            text = Path(path).read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 — a check, not a step
            logger.debug("could not read the root card: %s", exc)
            return []
        return [line.split("config_name:", 1)[1].strip()
                for line in text.splitlines() if "config_name:" in line]
