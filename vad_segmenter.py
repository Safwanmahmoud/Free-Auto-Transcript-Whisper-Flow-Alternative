"""Silero VAD: buffer PCM and cut complete speech segments for Whisper."""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)


class VadSegmenter:
    def __init__(
        self,
        sample_rate: int = 16000,
        vad_threshold: float = 0.5,
        min_speech_seconds: float = 0.35,
        silence_end_seconds: float = 0.35,
        max_chunk_seconds: float = 2.5,
    ) -> None:
        self.sample_rate = sample_rate
        self.vad_threshold = vad_threshold
        self.min_speech_seconds = min_speech_seconds
        self.silence_end_seconds = silence_end_seconds
        self.max_chunk_seconds = max_chunk_seconds
        self._model = None
        self._get_timestamps: Optional[Callable] = None
        self._buf = np.zeros(0, dtype=np.float32)

    def load(self) -> None:
        if self._model is not None:
            return
        log.info("Loading Silero VAD (torch.hub)...")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        self._get_timestamps = utils[0]
        self._model = model
        self._model.eval()
        log.info("Silero VAD loaded.")

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)

    def add_samples(self, chunk: np.ndarray) -> None:
        flat = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if flat.size == 0:
            return
        self._buf = np.concatenate([self._buf, flat])

    def _timestamps(self, wav: torch.Tensor) -> List[dict]:
        if self._model is None or self._get_timestamps is None:
            self.load()
        assert self._get_timestamps is not None
        min_ms = int(self.min_speech_seconds * 1000)
        sil_ms = int(self.silence_end_seconds * 1000)
        return self._get_timestamps(
            wav,
            self._model,
            threshold=self.vad_threshold,
            sampling_rate=self.sample_rate,
            min_speech_duration_ms=max(100, min_ms),
            min_silence_duration_ms=max(50, sil_ms),
            speech_pad_ms=60,
        )

    def pop_complete_segments(self, force_final: bool = False) -> List[np.ndarray]:
        """
        Return completed speech segments (float32 mono) and remove consumed audio from the buffer.
        If force_final, emit trailing speech even without trailing silence (hotkey released).
        """
        min_samp = int(self.min_speech_seconds * self.sample_rate)
        if self._buf.size < min_samp:
            if force_final and self._buf.size > 0:
                seg = self._buf.copy()
                self._buf = np.zeros(0, dtype=np.float32)
                return [seg]
            return []

        wav = torch.from_numpy(self._buf)
        try:
            ts = self._timestamps(wav)
        except Exception:
            log.exception("VAD failed")
            return []

        buf_len = self._buf.size
        max_samples = int(self.max_chunk_seconds * self.sample_rate)
        silence_samp = int(self.silence_end_seconds * self.sample_rate)
        spans: List[Tuple[int, int]] = []

        if not ts:
            if force_final:
                spans.append((0, buf_len))
            elif self._buf.size >= max_samples:
                spans.append((0, max_samples))
        else:
            for i, seg in enumerate(ts):
                start = int(seg["start"])
                end = int(seg["end"])
                is_last = i == len(ts) - 1
                tail_silence = buf_len - end
                long_enough = (end - start) >= min_samp
                chunk_too_long = (end - start) >= max_samples

                if not is_last:
                    if long_enough:
                        spans.append((start, end))
                    continue

                if force_final and end > start:
                    spans.append((start, end))
                elif tail_silence >= silence_samp and long_enough:
                    spans.append((start, end))
                elif chunk_too_long and long_enough:
                    spans.append((start, end))

        if not spans:
            if force_final and self._buf.size > 0:
                spans.append((0, buf_len))
            elif self._buf.size >= max_samples:
                spans.append((0, max_samples))

        if not spans:
            return []

        out = [self._buf[s:e].copy() for s, e in spans if e > s]
        cut = spans[-1][1]
        self._buf = self._buf[cut:].copy()
        return out

    def merge_overlap_text(self, prev: str, new: str) -> str:
        """If Whisper overlaps repeated phrases across chunks, strip duplicate prefix."""
        if not new:
            return ""
        if not prev:
            return new
        prev_stripped = prev.rstrip()
        new_stripped = new.lstrip()
        max_k = min(len(prev_stripped), len(new_stripped))
        for k in range(max_k, 0, -1):
            if prev_stripped[-k:] == new_stripped[:k]:
                return new[k:].lstrip() if k < len(new) else ""
        return new
