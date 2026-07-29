"""
Adapter onto `msbench` — the benchmark's own measurement code.

The published human anchors were produced by that package, so measuring our
synthesis with a reimplementation would put the two numbers on rulers that only
look identical. Everything that decides a value is therefore delegated:

    msbench.audio.prepare       preprocessing parity (soxr HQ, VAD, peak)
    msbench.normalize.normalize text normalisation
    msbench.metrics.score_pairs S/D/I, corpus and macro aggregation
    msbench.stats               cluster and paired bootstraps
    msbench.asr / msbench.sim   the recognisers and speaker encoders

What lives in speecheval is everything msbench has no opinion about: our data
layout, reference voices that are not benchmark speakers, the naturalness axis,
and the reporting that puts our rows next to the published anchors.

Verified in this environment: the ky SIM anchors recompute to 0.5593 (ecapa),
0.6069 (wavlm_ft) and 0.9214 (wavlm_sv) — the published values to four decimals.
"""
from __future__ import annotations

import importlib
import logging
import os
from typing import Any

from .config import AsrEngineConfig, EvalConfig

logger = logging.getLogger(__name__)

# Where each backend module keeps its subset-code -> language-code map. GigaAM
# has none: it is Kyrgyz-only, and its `supports()` is a literal comparison.
_ASR_COVERAGE_ATTR = {
    "whisper": "WHISPER_LANG",
    "mms": "ADAPTER",
    "elevenlabs": "LANG",
}
_ASR_FIXED_COVERAGE = {"gigaam": frozenset({"ky"})}

# Keys speecheval interprets itself and must not forward to msbench's builders.
_LOCAL_ONLY_KEYS = frozenset({"api_key_env"})

# msbench's ElevenLabs backend reads the key from one of these names.
_ELEVENLABS_ENV = "ELEVENLABS_API_KEY"


def asr_coverage(engine: str) -> frozenset[str]:
    """Which benchmark languages an ASR backend covers, per msbench itself."""
    if engine in _ASR_FIXED_COVERAGE:
        return _ASR_FIXED_COVERAGE[engine]
    attribute = _ASR_COVERAGE_ATTR.get(engine)
    if attribute is None:
        raise KeyError(f"unknown ASR engine {engine!r}")
    module = importlib.import_module(f"msbench.asr.{engine}")
    return frozenset(getattr(module, attribute))


def asr_engine_names() -> list[str]:
    import msbench.asr as registry
    return registry.available()


def sim_encoder_names() -> list[str]:
    import msbench.sim as registry
    return list(registry.ENCODERS)


def build_asr(config: AsrEngineConfig):
    """Instantiate an msbench ASR backend from our configuration."""
    import msbench.asr as registry

    key_env = config.option("api_key_env")
    if key_env:
        # msbench reads ELLBS_TOKEN / ELEVENLABS_API_KEY; the user's .env is free
        # to call it something else.
        value = os.environ.get(key_env)
        if value and not os.environ.get(_ELEVENLABS_ENV):
            os.environ[_ELEVENLABS_ENV] = value

    kwargs: dict[str, Any] = {
        k: v for k, v in config.options.items() if k not in _LOCAL_ONLY_KEYS
    }
    logger.info("building ASR backend %s with %s", config.name, kwargs)
    return registry.build(config.name, **kwargs)


def build_encoder(name: str, config: EvalConfig):
    """Instantiate an msbench speaker encoder."""
    import msbench.sim as registry

    logger.info("building speaker encoder %s (batch=%d)", name, config.sim.batch)
    return registry.build(name, batch=config.sim.batch)


def describe() -> dict[str, str]:
    """Provenance of the measurement code, recorded in every report."""
    import msbench

    return {
        "msbench_version": getattr(msbench, "__version__", "unknown"),
        "msbench_path": getattr(msbench, "__file__", "unknown"),
    }
