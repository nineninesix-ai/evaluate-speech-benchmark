"""
speecheval — evaluation of zero-shot TTS against the Multilingual Speech
Benchmark v2.0 protocol.

Two axes are measured the way the benchmark defines them — intelligibility
(WER/CER, corpus-level, several recognisers) and speaker similarity (SIM,
several encoders, normalised against a published anchor and impostor floor) —
plus a naturalness section that the benchmark deliberately does not cover and
that is therefore reported separately.
"""

__version__ = "2.0.0"
