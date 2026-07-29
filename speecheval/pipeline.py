"""
Stage orchestration.

Each stage is independent and cached per unit of work, so a run can be stopped,
resumed, or re-entered after a configuration change and only the affected units
recompute. Stages are ordered by cost: the cheap structural checks run first and
fail the run before a model is downloaded.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from . import console
from .cache import StageCache
from .config import EvalConfig
from .sources import BenchmarkSource, SynthesisSource, SynthesisSubset, join

logger = logging.getLogger(__name__)

STAGES = ("asr", "sim", "quality", "report")


@dataclass
class RunState:
    subsets: list[SynthesisSubset] = field(default_factory=list)
    asr: dict = field(default_factory=dict)
    sim: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)


class Pipeline:
    def __init__(self, config: EvalConfig, limit: Optional[int] = None,
                 only: Optional[list[str]] = None):
        self.config = config
        self.limit = limit
        self.only = only
        self.benchmark = BenchmarkSource(config)
        self.synthesis = SynthesisSource(config)
        self.cache = StageCache(config.cache_dir, config.run.name,
                                enabled=config.run.resume)
        self.state = RunState()

    # -- discovery ----------------------------------------------------------

    def discover(self) -> list[SynthesisSubset]:
        subsets = self.synthesis.discover()
        if self.only:
            subsets = [s for s in subsets if s.name in set(self.only)]
            if not subsets:
                raise SystemExit(f"no subset matches {self.only}")

        unclean = 0
        for subset in subsets:
            _, report = join(self.benchmark, self.synthesis, subset,
                             key_column=self.config.synthesis.columns.key,
                             text_column=self.config.synthesis.columns.text)
            if not report.is_clean:
                unclean += 1
                logger.warning(
                    "%s: %d unknown utt, %d not synthesised, %d text mismatches",
                    subset.name, len(report.missing_in_benchmark),
                    len(report.missing_in_synthesis), len(report.text_mismatches),
                )
        if unclean:
            console.warn(f"{unclean} subset(s) do not join cleanly — see the log")

        if self.limit:
            subsets = [
                SynthesisSubset(s.name, s.language, s.voice, s.path,
                                min(s.n_rows, self.limit))
                for s in subsets
            ]
            console.warn(f"limit={self.limit}: this run is a smoke test, not a result")

        self.state.subsets = subsets
        return subsets

    # -- stages -------------------------------------------------------------

    def run(self, stages: tuple[str, ...] = STAGES) -> RunState:
        started = time.time()
        subsets = self.discover()
        console.ok(f"{len(subsets)} subset(s), {sum(s.n_rows for s in subsets):,} clips")

        if "asr" in stages:
            from .stages.asr import AsrStage
            console.banner("Intelligibility")
            stage = AsrStage(self.config, self.benchmark, self.synthesis, self.cache)
            if self.limit:
                stage.row_limit = self.limit
            self.state.asr = stage.run(subsets)

        if "sim" in stages:
            from .stages.sim import SimStage
            console.banner("Speaker similarity")
            stage = SimStage(self.config, self.benchmark, self.synthesis, self.cache)
            if self.limit:
                stage.row_limit = self.limit
            self.state.sim = stage.run(subsets)

        if "quality" in stages and self.config.quality.enabled:
            from .stages.quality import QualityStage
            console.banner("Naturalness (outside the protocol)")
            stage = QualityStage(self.config, self.benchmark, self.synthesis, self.cache)
            if self.limit:
                stage.row_limit = self.limit
            self.state.quality = stage.run(subsets)

        if "report" in stages:
            from .report import Report
            console.banner("Reports")
            Report(self.config, self.cache).build()

        console.ok(f"finished in {time.time() - started:.0f}s")
        return self.state
