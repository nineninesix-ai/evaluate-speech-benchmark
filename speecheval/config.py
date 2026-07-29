"""
Typed configuration for the evaluation pipeline.

Everything the pipeline does is decided here. The config is parsed into frozen
dataclasses and validated up front, so a run either starts with every path,
model and language code resolved, or fails immediately with a message that says
which key is wrong — never halfway through a four-hour GPU job.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


class ConfigError(ValueError):
    """Raised when the configuration is malformed or internally inconsistent."""


def _require(mapping: dict, key: str, context: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"{context}: missing required key '{key}'")
    return mapping[key]


def _section(mapping: dict, key: str, context: str) -> dict:
    value = mapping.get(key)
    if value is None:
        raise ConfigError(f"{context}: missing required section '{key}'")
    if not isinstance(value, dict):
        raise ConfigError(f"{context}.{key}: expected a mapping, got {type(value).__name__}")
    return value


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: Path
    cache_dir: Path
    device: str
    seed: int
    resume: bool

    @classmethod
    def parse(cls, d: dict) -> "RunConfig":
        device = d.get("device", "auto")
        if device not in ("auto", "cuda", "cpu"):
            raise ConfigError(f"run.device: expected auto|cuda|cpu, got '{device}'")
        return cls(
            name=str(_require(d, "name", "run")),
            output_dir=Path(_require(d, "output_dir", "run")),
            cache_dir=Path(d.get("cache_dir", ".cache/speecheval")),
            device=device,
            seed=int(d.get("seed", 20260725)),
            resume=bool(d.get("resume", True)),
        )


@dataclass(frozen=True)
class BenchmarkConfig:
    path: Path
    split: str
    languages: tuple[str, ...]
    reports_dir: Path

    @classmethod
    def parse(cls, d: dict) -> "BenchmarkConfig":
        languages = _require(d, "languages", "benchmark")
        if not isinstance(languages, list) or not languages:
            raise ConfigError("benchmark.languages: expected a non-empty list")
        return cls(
            path=Path(_require(d, "path", "benchmark")),
            split=str(d.get("split", "main")),
            languages=tuple(str(x) for x in languages),
            reports_dir=Path(d.get("reports_dir", "./benchmark/reports")),
        )


@dataclass(frozen=True)
class SynthesisColumns:
    key: str
    audio: str
    text: str
    duration: str
    latency: str
    model: str
    voice: str

    @classmethod
    def parse(cls, d: dict) -> "SynthesisColumns":
        return cls(
            key=str(d.get("key", "utt")),
            audio=str(d.get("audio", "audio")),
            text=str(d.get("text", "text")),
            duration=str(d.get("duration", "gen_dur")),
            latency=str(d.get("latency", "latency_s")),
            model=str(d.get("model", "model_id")),
            voice=str(d.get("voice", "ref_voice")),
        )


@dataclass(frozen=True)
class SynthesisConfig:
    path: Path
    split: str
    include: Optional[tuple[str, ...]]
    exclude: tuple[str, ...]
    columns: SynthesisColumns

    @classmethod
    def parse(cls, d: dict) -> "SynthesisConfig":
        include = d.get("include")
        return cls(
            path=Path(_require(d, "path", "synthesis")),
            split=str(d.get("split", "train")),
            include=tuple(str(x) for x in include) if include else None,
            exclude=tuple(str(x) for x in (d.get("exclude") or [])),
            columns=SynthesisColumns.parse(d.get("columns") or {}),
        )


@dataclass(frozen=True)
class VoiceConfig:
    """How the reference voice of one synthesis subset is obtained."""

    name: str
    type: str                       # per_row | fixed_file
    column: Optional[str] = None    # per_row: benchmark audio column
    path: Optional[Path] = None     # fixed_file: local audio file
    native_language: Optional[str] = None
    note: Optional[str] = None

    @property
    def is_per_row(self) -> bool:
        return self.type == "per_row"

    @classmethod
    def parse(cls, name: str, d: dict) -> "VoiceConfig":
        ctx = f"voices.{name}"
        vtype = str(_require(d, "type", ctx))
        if vtype not in ("per_row", "fixed_file"):
            raise ConfigError(f"{ctx}.type: expected per_row|fixed_file, got '{vtype}'")
        if vtype == "per_row" and not d.get("column"):
            raise ConfigError(f"{ctx}: type 'per_row' requires 'column'")
        if vtype == "fixed_file" and not d.get("path"):
            raise ConfigError(f"{ctx}: type 'fixed_file' requires 'path'")
        return cls(
            name=name,
            type=vtype,
            column=d.get("column"),
            path=Path(d["path"]) if d.get("path") else None,
            native_language=d.get("native_language"),
            note=d.get("note"),
        )


@dataclass(frozen=True)
class VadConfig:
    enabled: bool
    pad_ms: int
    threshold: float


@dataclass(frozen=True)
class AudioConfig:
    target_sample_rate: int
    resampler: str
    mono: bool
    vad: VadConfig
    peak_dbfs: float

    @classmethod
    def parse(cls, d: dict) -> "AudioConfig":
        vad = d.get("vad") or {}
        resampler = str(d.get("resampler", "soxr_hq"))
        if not resampler.startswith("soxr"):
            raise ConfigError(
                f"audio.resampler: '{resampler}' is not a soxr mode. The anchors were "
                "measured with soxr HQ; anything else folds aliasing into the speech band."
            )
        return cls(
            target_sample_rate=int(d.get("target_sample_rate", 16000)),
            resampler=resampler,
            mono=bool(d.get("mono", True)),
            vad=VadConfig(
                enabled=bool(vad.get("enabled", True)),
                pad_ms=int(vad.get("pad_ms", 50)),
                threshold=float(vad.get("threshold", 0.5)),
            ),
            peak_dbfs=float(d.get("peak_dbfs", -1.0)),
        )


@dataclass(frozen=True)
class TextNormalizationConfig:
    unicode_form: str
    lowercase: bool
    expand_digits: bool
    expand_digits_skip_languages: tuple[str, ...]
    keep_apostrophes: bool
    keep_diacritics: bool
    fold_typographic_apostrophes: bool
    punctuation: str

    @classmethod
    def parse(cls, d: dict) -> "TextNormalizationConfig":
        form = str(d.get("unicode_form", "NFC"))
        if form not in ("NFC", "NFD", "NFKC", "NFKD"):
            raise ConfigError(f"text_normalization.unicode_form: unknown form '{form}'")
        return cls(
            unicode_form=form,
            lowercase=bool(d.get("lowercase", True)),
            expand_digits=bool(d.get("expand_digits", True)),
            expand_digits_skip_languages=tuple(d.get("expand_digits_skip_languages") or []),
            keep_apostrophes=bool(d.get("keep_apostrophes", True)),
            keep_diacritics=bool(d.get("keep_diacritics", True)),
            fold_typographic_apostrophes=bool(d.get("fold_typographic_apostrophes", True)),
            punctuation=str(_require(d, "punctuation", "text_normalization")),
        )


@dataclass(frozen=True)
class AsrEngineConfig:
    """One msbench ASR backend. The name is its registry name.

    Model identity, decoding defaults and language coverage belong to msbench;
    anything here is an explicit override, so the two cannot drift apart.
    """

    name: str
    enabled: bool
    options: dict[str, Any]

    def option(self, key: str, default: Any = None) -> Any:
        return self.options.get(key, default)

    def covers(self, language: str) -> bool:
        from .engine import asr_coverage
        return language in asr_coverage(self.name)

    @classmethod
    def parse(cls, name: str, d: dict) -> "AsrEngineConfig":
        if "languages" in d:
            raise ConfigError(
                f"asr.engines.{name}.languages: language coverage comes from the "
                "backend itself — remove this key rather than risk it disagreeing"
            )
        options = {k: v for k, v in d.items() if k != "enabled"}
        return cls(name=name, enabled=bool(d.get("enabled", True)), options=options)


@dataclass(frozen=True)
class AsrConfig:
    primary: dict[str, str]
    engines: dict[str, AsrEngineConfig]

    def enabled_engines(self) -> list[AsrEngineConfig]:
        return [e for e in self.engines.values() if e.enabled]

    def engines_for(self, language: str) -> list[AsrEngineConfig]:
        """Every enabled engine that covers this language."""
        return [e for e in self.enabled_engines() if e.covers(language)]

    def primary_engine(self, language: str) -> str:
        if language not in self.primary:
            raise ConfigError(f"asr.primary: no primary recogniser declared for '{language}'")
        return self.primary[language]

    @classmethod
    def parse(cls, d: dict) -> "AsrConfig":
        primary = _section(d, "primary", "asr")
        engines_raw = _section(d, "engines", "asr")
        engines = {name: AsrEngineConfig.parse(name, cfg or {}) for name, cfg in engines_raw.items()}
        return cls(primary={str(k): str(v) for k, v in primary.items()}, engines=engines)


@dataclass(frozen=True)
class SimFloorConfig:
    against: str
    pairs: int


@dataclass(frozen=True)
class SimAnchorConfig:
    method: str
    chunk_sec: float
    hop_sec: float
    max_chunks: int


@dataclass(frozen=True)
class SimCalibrationConfig:
    per_row_source: str
    per_row_csv: str
    floor: SimFloorConfig
    anchor: SimAnchorConfig

    @classmethod
    def parse(cls, d: dict) -> "SimCalibrationConfig":
        per_row = d.get("per_row_voices") or {}
        fixed = d.get("fixed_file_voices") or {}
        floor = fixed.get("floor") or {}
        anchor = fixed.get("anchor") or {}
        method = str(anchor.get("method", "intra_session"))
        if method != "intra_session":
            raise ConfigError(
                f"sim.calibration.fixed_file_voices.anchor.method: unsupported '{method}'"
            )
        return cls(
            per_row_source=str(per_row.get("source", "benchmark_reports")),
            per_row_csv=str(per_row.get("csv", "csv/sim.csv")),
            floor=SimFloorConfig(
                against=str(floor.get("against", "prompt_audio")),
                pairs=int(floor.get("pairs", 5000)),
            ),
            anchor=SimAnchorConfig(
                method=method,
                chunk_sec=float(anchor.get("chunk_sec", 4.0)),
                hop_sec=float(anchor.get("hop_sec", 2.0)),
                max_chunks=int(anchor.get("max_chunks", 8)),
            ),
        )


@dataclass(frozen=True)
class SimConfig:
    encoders: tuple[str, ...]
    batch: int
    calibration: SimCalibrationConfig

    @classmethod
    def parse(cls, d: dict) -> "SimConfig":
        if "max_duration_sec" in d:
            raise ConfigError(
                "sim.max_duration_sec: clips are embedded whole by the protocol; "
                "truncating would change the value and break parity with the anchors"
            )
        batch = int(d.get("batch", 1))
        if batch != 1:
            raise ConfigError(
                f"sim.batch: must be 1, got {batch}. Batched embedding pads the "
                "shorter clips and the padding reaches the pooling layer: the same "
                "2 s clip scores cos 0.295 (wavlm_ft) / 0.339 (wavlm_sv) against "
                "itself embedded alone. Every SIM in the run would be wrong."
            )
        return cls(
            encoders=tuple(str(x) for x in (d.get("encoders") or [])),
            batch=batch,
            calibration=SimCalibrationConfig.parse(d.get("calibration") or {}),
        )


@dataclass(frozen=True)
class QualityConfig:
    enabled: bool
    metrics: dict[str, dict]

    def is_on(self, metric: str) -> bool:
        return self.enabled and bool((self.metrics.get(metric) or {}).get("enabled", False))

    def options(self, metric: str) -> dict:
        return self.metrics.get(metric) or {}

    @classmethod
    def parse(cls, d: dict) -> "QualityConfig":
        metrics = {k: v for k, v in d.items() if k != "enabled" and isinstance(v, dict)}
        return cls(enabled=bool(d.get("enabled", True)), metrics=metrics)


@dataclass(frozen=True)
class FailureConfig:
    catastrophic_wer: float
    runaway_duration_ratio: float
    min_duration_ratio: float

    @classmethod
    def parse(cls, d: dict) -> "FailureConfig":
        return cls(
            catastrophic_wer=float(d.get("catastrophic_wer", 0.5)),
            runaway_duration_ratio=float(d.get("runaway_duration_ratio", 2.0)),
            min_duration_ratio=float(d.get("min_duration_ratio", 0.5)),
        )


@dataclass(frozen=True)
class BootstrapConfig:
    replicates: int
    confidence: float
    cluster_by: str


@dataclass(frozen=True)
class StatsConfig:
    aggregation: str
    also_report_macro: bool
    bootstrap: BootstrapConfig
    paired_enabled: bool
    paired_against: tuple[str, ...]

    @classmethod
    def parse(cls, d: dict) -> "StatsConfig":
        aggregation = str(d.get("aggregation", "corpus"))
        if aggregation != "corpus":
            raise ConfigError(
                "stats.aggregation: only 'corpus' is a valid headline "
                "(sum(S+D+I) / sum(N_ref)); macro is reported alongside it"
            )
        boot = d.get("bootstrap") or {}
        paired = d.get("paired") or {}
        return cls(
            aggregation=aggregation,
            also_report_macro=bool(d.get("also_report_macro", True)),
            bootstrap=BootstrapConfig(
                replicates=int(boot.get("replicates", 2000)),
                confidence=float(boot.get("confidence", 0.95)),
                cluster_by=str(boot.get("cluster_by", "speaker_id")),
            ),
            paired_enabled=bool(paired.get("enabled", True)),
            paired_against=tuple(str(x) for x in (paired.get("against") or [])),
        )


@dataclass(frozen=True)
class PushConfig:
    enabled: bool
    repo: Optional[str]
    private: bool
    include_audio: bool


@dataclass(frozen=True)
class ReportConfig:
    markdown: bool
    csv: bool
    per_utterance_parquet: bool
    json_summary: bool
    push: PushConfig

    @classmethod
    def parse(cls, d: dict) -> "ReportConfig":
        push = d.get("push") or {}
        cfg = cls(
            markdown=bool(d.get("markdown", True)),
            csv=bool(d.get("csv", True)),
            per_utterance_parquet=bool(d.get("per_utterance_parquet", True)),
            json_summary=bool(d.get("json_summary", True)),
            push=PushConfig(
                enabled=bool(push.get("enabled", False)),
                repo=push.get("repo"),
                private=bool(push.get("private", True)),
                include_audio=bool(push.get("include_audio", False)),
            ),
        )
        if cfg.push.enabled and not cfg.push.repo:
            raise ConfigError("report.push: enabled but no 'repo' given")
        if not any([cfg.markdown, cfg.csv, cfg.per_utterance_parquet, cfg.json_summary,
                    cfg.push.enabled]):
            raise ConfigError("report: every output is disabled — the run would produce nothing")
        return cfg


# ---------------------------------------------------------------------------
# root
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalConfig:
    path: Path
    run: RunConfig
    benchmark: BenchmarkConfig
    synthesis: SynthesisConfig
    voices: dict[str, VoiceConfig]
    audio: AudioConfig
    text: TextNormalizationConfig
    asr: AsrConfig
    sim: SimConfig
    quality: QualityConfig
    failure: FailureConfig
    stats: StatsConfig
    report: ReportConfig
    raw: dict = field(repr=False, default_factory=dict)

    # -- loading ------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "EvalConfig":
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        with path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: top level must be a mapping")

        voices_raw = _section(raw, "voices", "config")
        cfg = cls(
            path=path,
            run=RunConfig.parse(_section(raw, "run", "config")),
            benchmark=BenchmarkConfig.parse(_section(raw, "benchmark", "config")),
            synthesis=SynthesisConfig.parse(_section(raw, "synthesis", "config")),
            voices={name: VoiceConfig.parse(name, d or {}) for name, d in voices_raw.items()},
            audio=AudioConfig.parse(_section(raw, "audio", "config")),
            text=TextNormalizationConfig.parse(_section(raw, "text_normalization", "config")),
            asr=AsrConfig.parse(_section(raw, "asr", "config")),
            sim=SimConfig.parse(_section(raw, "sim", "config")),
            quality=QualityConfig.parse(raw.get("quality") or {}),
            failure=FailureConfig.parse(raw.get("failure") or {}),
            stats=StatsConfig.parse(raw.get("stats") or {}),
            report=ReportConfig.parse(_section(raw, "report", "config")),
            raw=raw,
        )
        cfg.validate()
        return cfg

    # -- validation ---------------------------------------------------------

    def validate(self) -> None:
        """Fail fast on anything that would only surface hours into a run."""
        problems: list[str] = []

        if not self.benchmark.path.exists():
            problems.append(f"benchmark.path does not exist: {self.benchmark.path}")
        if not self.synthesis.path.exists():
            problems.append(f"synthesis.path does not exist: {self.synthesis.path}")
        if not self.benchmark.reports_dir.exists():
            problems.append(
                f"benchmark.reports_dir does not exist: {self.benchmark.reports_dir} "
                "(it carries the anchor and impostor-floor coefficients)"
            )

        for voice in self.voices.values():
            if voice.type == "fixed_file" and voice.path and not voice.path.is_file():
                problems.append(f"voices.{voice.name}.path does not exist: {voice.path}")

        from .engine import asr_engine_names, sim_encoder_names

        known_engines = set(asr_engine_names())
        for name in self.asr.engines:
            if name not in known_engines:
                problems.append(
                    f"asr.engines.{name}: not an msbench backend "
                    f"(have: {', '.join(sorted(known_engines))})"
                )

        known_encoders = set(sim_encoder_names())
        for encoder in self.sim.encoders:
            if encoder not in known_encoders:
                problems.append(
                    f"sim.encoders: unknown encoder '{encoder}' "
                    f"(have: {', '.join(sorted(known_encoders))})"
                )

        for language in self.benchmark.languages:
            engine_name = self.asr.primary.get(language)
            if not engine_name:
                problems.append(f"asr.primary: no recogniser declared for '{language}'")
                continue
            engine = self.asr.engines.get(engine_name)
            if engine is None:
                problems.append(
                    f"asr.primary.{language}: engine '{engine_name}' is not defined "
                    f"in asr.engines"
                )
            elif not engine.enabled:
                problems.append(
                    f"asr.primary.{language}: engine '{engine_name}' is the headline "
                    "recogniser but is disabled"
                )
            elif engine_name in known_engines and not engine.covers(language):
                problems.append(
                    f"asr.primary.{language}: '{engine_name}' does not cover this "
                    "language, so it cannot be its headline recogniser"
                )

        for engine in self.asr.enabled_engines():
            key_env = engine.option("api_key_env")
            if key_env and not os.environ.get(key_env):
                problems.append(
                    f"asr.engines.{engine.name}: ${key_env} is not set "
                    f"(put it in .env, or set enabled: false)"
                )

        if not self.sim.encoders:
            problems.append("sim.encoders: no encoder selected")

        if problems:
            raise ConfigError(
                "configuration is not usable:\n  - " + "\n  - ".join(problems)
            )

    # -- helpers ------------------------------------------------------------

    def voice(self, name: str) -> VoiceConfig:
        if name not in self.voices:
            raise ConfigError(
                f"synthesis subset references voice '{name}', which is not described "
                f"in voices (known: {', '.join(sorted(self.voices))})"
            )
        return self.voices[name]

    @property
    def output_dir(self) -> Path:
        return self.run.output_dir

    @property
    def cache_dir(self) -> Path:
        return self.run.cache_dir


def load_dotenv(path: str | Path = ".env") -> list[str]:
    """Load KEY=VALUE lines into the environment. Returns the keys that were set."""
    path = Path(path)
    if not path.is_file():
        return []
    loaded = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)
            loaded.append(key)
    return loaded
