"""Command line entry point."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__, console
from .config import ConfigError, EvalConfig, load_dotenv

DEFAULT_CONFIG = "config/eval.yaml"


def _load(args) -> EvalConfig:
    keys = load_dotenv(args.env)
    if keys:
        logging.getLogger(__name__).debug("loaded %d key(s) from %s", len(keys), args.env)
    return EvalConfig.load(args.config)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_validate(args) -> int:
    console.banner("Configuration check")
    config = _load(args)
    console.ok(f"{config.path} parsed and validated")

    console.step("System under test")
    console.note(f"name        {config.run.name}")
    console.note(f"output      {config.run.output_dir}")
    console.note(f"cache       {config.run.cache_dir}")

    from .engine import asr_coverage, describe

    console.step("Measurement engine")
    for key, value in describe().items():
        console.note(f"{key:<16} {value}")

    console.step("Recognisers")
    for engine in config.asr.enabled_engines():
        languages = ", ".join(sorted(asr_coverage(engine.name)))
        primary_for = sorted(l for l, e in config.asr.primary.items() if e == engine.name)
        role = f"primary for {', '.join(primary_for)}" if primary_for else "second opinion"
        console.note(f"{engine.name:<12} {role}")
        console.note(f"{'':<12} covers: {languages}")

    console.step("Speaker encoders")
    console.note(", ".join(config.sim.encoders))

    console.step("Naturalness (outside the benchmark protocol)")
    for metric in ("dnsmos", "utmos", "nisqa"):
        state = "on" if config.quality.is_on(metric) else "off"
        console.note(f"{metric:<10} {state}")

    console.ok("configuration is usable")
    return 0


def cmd_discover(args) -> int:
    from .sources import BenchmarkSource, SynthesisSource, join

    console.banner("Discovery and join check")
    config = _load(args)

    benchmark = BenchmarkSource(config)
    synthesis = SynthesisSource(config)
    subsets = synthesis.discover()

    console.ok(f"{len(subsets)} synthesis subset(s) under {config.synthesis.path}")
    print()

    rows = []
    total_rows = 0
    problems = 0

    for subset in subsets:
        merged, report = join(
            benchmark, synthesis, subset,
            key_column=config.synthesis.columns.key,
            text_column=config.synthesis.columns.text,
        )
        total_rows += report.n_joined

        issues = []
        if report.missing_in_benchmark:
            issues.append(f"{len(report.missing_in_benchmark)} unknown utt")
        if report.missing_in_synthesis:
            issues.append(f"{len(report.missing_in_synthesis)} not synthesised")
        if report.text_mismatches:
            issues.append(f"{len(report.text_mismatches)} text mismatch")
        if issues:
            problems += 1

        primary = config.asr.primary_engine(subset.language)
        engines = ", ".join(e.name for e in config.asr.engines_for(subset.language))
        rows.append([
            subset.name,
            subset.voice.type,
            str(report.n_joined),
            primary,
            engines,
            "ok" if not issues else "; ".join(issues),
        ])

    console.table(
        ["subset", "voice", "joined", "primary ASR", "recognisers", "status"],
        rows,
    )
    print()

    console.step("Work implied by this configuration")
    n_asr_calls = 0
    for subset in subsets:
        n_asr_calls += subset.n_rows * len(config.asr.engines_for(subset.language))
    console.note(f"clips to prepare      {total_rows:,}")
    console.note(f"recogniser passes     {n_asr_calls:,}")
    console.note(f"similarity passes     {total_rows * len(config.sim.encoders):,}")

    if problems:
        console.warn(f"{problems} subset(s) did not join cleanly — see the status column")
        return 1
    console.ok("every subset joins cleanly onto the benchmark")
    return 0


def cmd_run(args) -> int:
    from .pipeline import STAGES, Pipeline

    console.banner("speecheval")
    config = _load(args)
    console.setup_logging(
        logging.DEBUG if args.verbose else logging.INFO,
        log_file=config.run.output_dir / "run.log",
    )

    stages = tuple(args.stages) if args.stages else STAGES
    unknown = set(stages) - set(STAGES)
    if unknown:
        console.fail(f"unknown stage(s): {', '.join(sorted(unknown))}")
        return 2

    pipeline = Pipeline(config, limit=args.limit, only=args.subset or None)
    pipeline.run(stages)
    return 0


def cmd_report(args) -> int:
    """Rebuild every report from cached metrics — no models, no GPU."""
    from .cache import StageCache
    from .report import Report

    console.banner("Reports")
    config = _load(args)
    cache = StageCache(config.cache_dir, config.run.name, enabled=True)
    tables = Report(config, cache).build()
    for name, frame in tables.items():
        console.note(f"{name:<18} {len(frame)} row(s)")
    return 0


def cmd_normalize(args) -> int:
    """Show what the normaliser does to a language's texts — protocol parity check."""
    from .sources import BenchmarkSource
    from .text import TextNormalizer

    console.banner("Text normalisation check")
    config = _load(args)
    benchmark = BenchmarkSource(config)

    for language in config.benchmark.languages:
        normalizer = TextNormalizer(config.text, language)
        frame = benchmark.metadata(language)
        if "text_norm" not in frame.columns:
            console.warn(f"{language}: benchmark carries no text_norm to compare against")
            continue

        ours = frame["text"].map(normalizer)
        theirs = frame["text_norm"].fillna("")
        differs = ours != theirs
        n_diff = int(differs.sum())

        if n_diff == 0:
            console.ok(f"{language}: {len(frame)} texts, identical to the benchmark's text_norm")
        else:
            console.warn(f"{language}: {n_diff}/{len(frame)} texts differ from text_norm")
            for utt, mine, ref in list(zip(
                frame.loc[differs, "utt"], ours[differs], theirs[differs]
            ))[:args.examples]:
                console.note(f"{utt}")
                console.note(f"  ours  {mine!r}")
                console.note(f"  card  {ref!r}")
    return 0


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="speecheval",
        description="Evaluate zero-shot TTS against the Multilingual Speech Benchmark v2.0",
    )
    parser.add_argument("--version", action="version", version=f"speecheval {__version__}")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config", default=DEFAULT_CONFIG, help="path to the config YAML")
    common.add_argument("--env", default=".env", help="file with API keys")
    common.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", parents=[common], help="parse and check the configuration")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("discover", parents=[common],
                       help="list synthesis subsets and verify they join onto the benchmark")
    p.set_defaults(func=cmd_discover)

    p = sub.add_parser("run", parents=[common], help="run the evaluation pipeline")
    p.add_argument("--stages", nargs="+", metavar="STAGE",
                   help="subset of stages to run (default: all)")
    p.add_argument("--subset", nargs="+", metavar="NAME",
                   help="restrict to these synthesis subsets")
    p.add_argument("--limit", type=int, default=None,
                   help="rows per subset — a smoke test, never a result")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report", parents=[common],
                       help="rebuild the reports from cached metrics, without a GPU")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("normalize", parents=[common],
                       help="compare our text normalisation against the benchmark's text_norm")
    p.add_argument("--examples", type=int, default=5, help="differing rows to print per language")
    p.set_defaults(func=cmd_normalize)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console.setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    try:
        return args.func(args)
    except ConfigError as exc:
        console.fail(str(exc))
        return 2
    except KeyboardInterrupt:
        console.warn("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
