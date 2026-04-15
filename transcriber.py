"""faster-whisper wrapper — load once, transcribe float32 mono PCM."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def _is_cuda_runtime_failure(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "cublas" in msg or "cudnn" in msg or "cudart" in msg:
        return True
    if "cuda" in msg and ("load" in msg or "found" in msg or "dll" in msg):
        return True
    if "nvrtc" in msg or "nvidia" in msg and "dll" in msg:
        return True
    return False


class Transcriber:
    uses_local_whisper = True

    def __init__(
        self,
        model_size: str = "turbo",
        device: str = "cuda",
        compute_type: str = "float16",
        language: Optional[str] = "en",
        use_whisper_vad_filter: bool = False,
        strip_fillers: bool = True,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._use_whisper_vad = use_whisper_vad_filter
        self._strip_fillers = strip_fillers
        self._model = None
        self._fell_back_to_cpu = False

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        log.info(
            "Loading Whisper model=%s device=%s compute_type=%s",
            self._model_size,
            self._device,
            self._compute_type,
        )
        self._model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
        )
        log.info("Whisper model loaded.")

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _fallback_to_cpu(self, reason: str) -> None:
        log.warning(
            "GPU inference unavailable (%s). Reloading Whisper on CPU (int8). "
            "Install CUDA 12 + cuBLAS, or set device: cpu in config.yaml.",
            reason,
        )
        self._model = None
        self._device = "cpu"
        self._compute_type = "int8"
        self._fell_back_to_cpu = True
        self.load()

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        initial_prompt: Optional[str] = None,
    ) -> str:
        if self._model is None:
            self.load()
        if audio.size == 0:
            return ""
        pcm = np.asarray(audio, dtype=np.float32).flatten()
        if pcm.max() > 1.0 or pcm.min() < -1.0:
            pcm = np.clip(pcm, -1.0, 1.0)

        try:
            return self._transcribe_pcm(pcm, initial_prompt)
        except (RuntimeError, OSError) as e:
            if (
                self._device == "cuda"
                and not self._fell_back_to_cpu
                and _is_cuda_runtime_failure(e)
            ):
                self._fallback_to_cpu(str(e))
                return self._transcribe_pcm(pcm, initial_prompt)
            raise

    def _whisper_initial_prompt(self, initial_prompt: Optional[str]) -> Optional[str]:
        if not self._strip_fillers:
            return initial_prompt or None
        hint = (
            "Clean transcript without fillers: omit uh, um, ah, er, hmm, false starts, "
            "and hesitant repetitions; keep meaning and punctuation."
        )
        if initial_prompt and initial_prompt.strip():
            return f"{hint} Prior text: {initial_prompt.strip()[-350:]}"
        return hint

    def _transcribe_pcm(
        self,
        pcm: np.ndarray,
        initial_prompt: Optional[str],
    ) -> str:
        assert self._model is not None
        segments, _info = self._model.transcribe(
            pcm,
            language=self._language,
            task="transcribe",
            vad_filter=self._use_whisper_vad,
            initial_prompt=self._whisper_initial_prompt(initial_prompt),
            beam_size=5,
        )
        parts = []
        for seg in segments:
            t = (seg.text or "").strip()
            if t:
                parts.append(t)
        return " ".join(parts).strip()
