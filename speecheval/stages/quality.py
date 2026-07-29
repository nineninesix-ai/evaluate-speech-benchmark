"""
Naturalness — deliberately outside the benchmark protocol.

The benchmark measures no naturalness axis at all, by design: a system can score
3 % WER and 0.95 SIM and still sound robotic, and nothing in the protocol would
notice. These numbers fill that gap and are reported in their own section, never
mixed into a WER or SIM table, because they have no human anchor to be read
against.

Two audio states are used, on purpose:

* **DNSMOS and UTMOSv2** run on the canonical 16 kHz audio — the same state the
  benchmark's own `qc_dnsmos_*` columns were computed in, so our synthesis can be
  put directly beside the prompt and ground-truth recordings on one scale.
* **NISQA** runs on the delivered audio resampled to its native 48 kHz. Feeding it
  16 kHz audio upsampled to 48 would throw away everything above 8 kHz and its
  coloration dimension would read that loss as a defect of the system.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..cache import StageUnit
from ..sources import SynthesisSubset, decode_audio

logger = logging.getLogger(__name__)

STAGE = "quality"
CHUNK = 128            # clips held decoded at once
NISQA_SR = 48000
CANONICAL_SR = 16000

# Below this there is no speech to judge — the system emitted silence, which the
# failure table already records. Feeding it to the predictors would either crash
# them or return a confident score for nothing, and one bad clip would take the
# whole chunk's batch with it.
MIN_QUALITY_SEC = 0.3


class QualityStage:
    def __init__(self, config, benchmark, synthesis, cache):
        self.config = config
        self.benchmark = benchmark
        self.synthesis = synthesis
        self.cache = cache
        self.row_limit: Optional[int] = None
        self._dnsmos = None
        self._utmos = None
        self._nisqa = None

    # -- models -------------------------------------------------------------

    def _load_dnsmos(self):
        if self._dnsmos is None:
            from speechmos import dnsmos
            self._dnsmos = dnsmos
        return self._dnsmos

    def _load_utmos(self):
        if self._utmos is None:
            import torch
            from utmosv2 import create_model

            device = self.config.run.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            model = create_model(device=device).to(device)
            self._patch_utmos_melspec(device)
            self._utmos = (model, device)
        return self._utmos

    @staticmethod
    def _patch_utmos_melspec(device: str) -> None:
        """
        Move UTMOSv2's mel-spectrogram to torchaudio on the GPU.

        Its fusion_stage3 config uses hop_length=32, which is ~700 STFT frames per
        1.4 s window; librosa spends ~150 ms per window in Python where torchaudio
        spends 3-5 ms in CUDA. Same algorithm, 30-50x the throughput, and the
        patched function still hands back a CPU array so nothing downstream
        changes.
        """
        try:
            import torch
            import torchaudio.transforms as transforms
            from utmosv2.dataset import multi_spec

            cache: dict = {}

            def melspec(cfg, spec_cfg, y: np.ndarray) -> np.ndarray:
                key = (spec_cfg.n_fft, spec_cfg.hop_length,
                       getattr(spec_cfg, "win_length", None), spec_cfg.n_mels)
                if key not in cache:
                    cache[key] = transforms.MelSpectrogram(
                        sample_rate=cfg.sr, n_fft=spec_cfg.n_fft,
                        hop_length=spec_cfg.hop_length,
                        win_length=getattr(spec_cfg, "win_length", None),
                        n_mels=spec_cfg.n_mels, power=2.0,
                    ).to(device)
                with torch.no_grad():
                    spectrum = cache[key](
                        torch.from_numpy(y.astype(np.float32)).to(device)).cpu().numpy()
                # Replicates librosa.power_to_db(spec, ref=np.max).
                decibels = 10.0 * np.log10(np.maximum(spectrum, 1e-10))
                decibels -= decibels.max()
                if spec_cfg.norm is not None:
                    decibels = (decibels + spec_cfg.norm) / spec_cfg.norm
                return decibels

            multi_spec._make_melspec = melspec
            logger.debug("UTMOSv2 mel-spectrogram patched onto %s", device)
        except Exception as exc:  # noqa: BLE001 — a speed-up, never a correctness step
            logger.warning("could not patch UTMOSv2 mel-spec (%s); falling back", exc)

    def _load_nisqa(self):
        if self._nisqa is None:
            from nisqa import NisqaModel
            self._nisqa = NisqaModel()
        return self._nisqa

    # -- planning -----------------------------------------------------------

    def _unit(self, subset: SynthesisSubset) -> StageUnit:
        return StageUnit(
            stage=STAGE,
            name=subset.name,
            parameters={
                "subset": subset.name,
                "metrics": {m: self.config.quality.is_on(m)
                            for m in ("dnsmos", "utmos", "nisqa")},
                "audio": {
                    "canonical_sr": CANONICAL_SR,
                    "nisqa_sr": NISQA_SR,
                    "vad": self.config.audio.vad.enabled,
                    "min_sec": MIN_QUALITY_SEC,
                },
                "row_limit": self.row_limit,
            },
        )

    # -- execution ----------------------------------------------------------

    def run(self, subsets: list[SynthesisSubset]) -> dict[str, pd.DataFrame]:
        results: dict[str, pd.DataFrame] = {}
        for subset in subsets:
            unit = self._unit(subset)
            cached = self.cache.load(unit)
            if cached is not None:
                frame, summary = cached
                results[unit.name] = frame
                logger.info("%s: cached (%s)", unit.name,
                            ", ".join(f"{k} {v:.3f}" for k, v in summary.items()
                                      if k.endswith("_mean")))
                continue
            frame, summary = self._measure(subset)
            self.cache.save(unit, frame, summary)
            results[unit.name] = frame
        return results

    def _measure(self, subset: SynthesisSubset):
        import soxr
        from msbench.audio import prepare

        columns = self.config.synthesis.columns
        started = time.time()
        rows: list[dict] = []

        canonical_buffer: list[np.ndarray] = []
        native_buffer: list[np.ndarray] = []
        keys: list[str] = []
        seen = 0
        n_too_short = 0

        progress = tqdm(total=subset.n_rows, desc=f"{subset.name} · naturalness",
                        unit="clip", leave=False)

        def flush():
            if not keys:
                return
            rows.extend(self._score_chunk(keys, canonical_buffer, native_buffer))
            keys.clear()
            canonical_buffer.clear()
            native_buffer.clear()

        for row in self.synthesis.iter_rows(subset, columns=[columns.key, columns.audio]):
            if self.row_limit and seen >= self.row_limit:
                break
            seen += 1
            progress.update(1)
            try:
                samples, sample_rate = decode_audio(row[columns.audio])
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s: undecodable audio (%s)", row[columns.key], exc)
                continue
            if samples.size == 0:
                continue

            canonical = prepare(samples, sample_rate,
                                trim=self.config.audio.vad.enabled, peak=True)
            if len(canonical) < MIN_QUALITY_SEC * CANONICAL_SR:
                n_too_short += 1
                continue

            keys.append(row[columns.key])
            canonical_buffer.append(canonical)
            native_buffer.append(
                soxr.resample(samples, sample_rate, NISQA_SR, quality="HQ")
                .astype(np.float32)
            )
            if len(keys) >= CHUNK:
                flush()
        flush()
        progress.close()

        frame = pd.DataFrame(rows)
        if frame.empty:
            raise RuntimeError(f"{subset.name}: no clip could be measured")

        metadata = self.benchmark.metadata(subset.language).set_index("utt")
        for column in ("speaker_id", "len_bin", "n_words",
                       "gt_qc_dnsmos_ovrl", "qc_dnsmos_ovrl"):
            if column in metadata.columns:
                frame[column] = frame.utt.map(metadata[column])

        frame["lang"] = subset.language
        frame["voice"] = subset.voice.name
        frame["subset"] = subset.name
        frame["model"] = self.config.run.name

        summary = {
            "subset": subset.name,
            "lang": subset.language,
            "voice": subset.voice.name,
            "model": self.config.run.name,
            "n_rows": len(frame),
            "n_too_short_to_judge": n_too_short,
            "seconds": round(time.time() - started, 1),
        }
        for column in frame.columns:
            if frame[column].dtype.kind == "f" and column not in ("n_words",):
                summary[f"{column}_mean"] = float(frame[column].mean())

        logger.info(
            "%s: %s (n=%d, %.0fs)", subset.name,
            " | ".join(f"{k[:-5]} {v:.3f}" for k, v in summary.items()
                       if k.endswith("_mean")),
            len(frame), summary["seconds"],
        )
        return frame, summary

    def _score_chunk(self, keys: list[str], canonical: list[np.ndarray],
                     native: list[np.ndarray]) -> list[dict]:
        scores: list[dict] = [{"utt": k} for k in keys]

        if self.config.quality.is_on("dnsmos"):
            dnsmos = self._load_dnsmos()
            for record, wave in zip(scores, canonical):
                try:
                    out = dnsmos.run(wave, CANONICAL_SR)
                    record.update(
                        dnsmos_ovrl=float(out["ovrl_mos"]),
                        dnsmos_sig=float(out["sig_mos"]),
                        dnsmos_bak=float(out["bak_mos"]),
                        dnsmos_p808=float(out["p808_mos"]),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("DNSMOS failed on %s: %s", record["utt"], exc)

        if self.config.quality.is_on("utmos"):
            model, device = self._load_utmos()
            batch = int(self.config.quality.options("utmos").get("batch_size", 32))
            # 3 s is the SSL branch's receptive field; longer input costs RAM and
            # changes nothing.
            limit = 3 * CANONICAL_SR
            lengths = [min(len(w), limit) for w in canonical]
            packed = np.zeros((len(canonical), max(lengths)), dtype=np.float32)
            for i, (wave, n) in enumerate(zip(canonical, lengths)):
                packed[i, :n] = wave[:n]
            try:
                predictions = model.predict(data=packed, verbose=False, device=device,
                                            num_workers=0, batch_size=batch)
                for record, value in zip(scores, np.atleast_1d(predictions)):
                    record["utmos"] = float(value)
            except Exception as exc:  # noqa: BLE001
                logger.warning("UTMOSv2 batch failed (%s)", exc)

        if self.config.quality.is_on("nisqa"):
            nisqa = self._load_nisqa()
            batch = int(self.config.quality.options("nisqa").get("batch_size", 32))
            try:
                predictions = nisqa.predict_batch(native, sr=NISQA_SR) \
                    if hasattr(nisqa, "predict_batch") else None
                if predictions is None:
                    predictions = [nisqa(waveform=w, sr=NISQA_SR) for w in native]
                for record, out in zip(scores, predictions):
                    record.update(
                        nisqa_mos=float(out["mos_pred"]),
                        nisqa_noi=float(out["noi_pred"]),
                        nisqa_dis=float(out["dis_pred"]),
                        nisqa_col=float(out["col_pred"]),
                        nisqa_loud=float(out["loud_pred"]),
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("NISQA batch failed (%s)", exc)
        return scores
