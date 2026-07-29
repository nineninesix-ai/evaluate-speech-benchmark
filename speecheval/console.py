"""Terminal output: colours, banners and a logging setup shared by every stage."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

BLUE = "\033[1;34m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
RED = "\033[1;31m"
DIM = "\033[2m"
RESET = "\033[0m"

_WIDTH = 74


def _supports_colour() -> bool:
    return sys.stdout.isatty()


def paint(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}" if _supports_colour() else text


def banner(title: str, colour: str = BLUE) -> None:
    line = "═" * (_WIDTH - 2)
    padded = title.center(_WIDTH - 2)
    print(paint(f"╔{line}╗", colour))
    print(paint(f"║{padded}║", colour))
    print(paint(f"╚{line}╝", colour))


def step(message: str) -> None:
    print(paint(f"→ {message}", YELLOW))


def ok(message: str) -> None:
    print(paint(f"✓ {message}", GREEN))


def warn(message: str) -> None:
    print(paint(f"! {message}", YELLOW))


def fail(message: str) -> None:
    print(paint(f"✗ {message}", RED))


def note(message: str) -> None:
    print(paint(f"  {message}", DIM))


def table(headers: list[str], rows: list[list[str]]) -> None:
    """A minimal fixed-width table; report tables are written as markdown instead."""
    if not rows:
        return
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    header = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print(paint(header, BLUE))
    print(paint("  ".join("─" * w for w in widths), DIM))
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))


def setup_logging(level: int = logging.INFO, log_file: Optional[Path] = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    # These are chatty at INFO and say nothing useful during a run.
    for noisy in ("urllib3", "filelock", "huggingface_hub", "datasets", "numba",
                  "matplotlib", "httpx", "httpcore", "speechbrain", "s3prl",
                  "pyannote", "transformers", "torio", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
