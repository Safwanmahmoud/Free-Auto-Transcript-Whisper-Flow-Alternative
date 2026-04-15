"""Transcribe short audio clips with Google Gemini (Generative Language API)."""

from __future__ import annotations

import io
import logging
import wave
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def _pcm_f32_to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    pcm = np.asarray(pcm, dtype=np.float32).flatten()
    pcm = np.clip(pcm, -1.0, 1.0)
    int16 = (pcm * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16.tobytes())
    return buf.getvalue()


class GoogleGeminiTranscriber:
    """Uses GOOGLE_API_KEY / GEMINI_API_KEY; audio is sent to Google's servers."""

    uses_local_whisper = False

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        language: Optional[str] = "en",
        strip_fillers: bool = True,
    ) -> None:
        self._api_key = api_key.strip()
        self._model_name = model_name.strip()
        self._language = language
        self._strip_fillers = strip_fillers
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        import google.generativeai as genai

        genai.configure(api_key=self._api_key)
        self._model = genai.GenerativeModel(self._model_name)
        log.info("Google Gemini model ready (%s).", self._model_name)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _instruction(self, initial_prompt: Optional[str]) -> str:
        lang = (self._language or "").strip().lower()
        lang_hint = (
            f"The spoken language is {lang}. "
            if lang and lang != "auto"
            else "Detect the spoken language. "
        )
        clean = (
            "Omit verbal fillers and disfluencies such as uh, um, ah, er, hmm, "
            "and hesitant repetitions or false starts; keep the intended meaning "
            "and normal punctuation. "
            if self._strip_fillers
            else ""
        )
        base = (
            f"{lang_hint}"
            f"{clean}"
            "Transcribe the audio: output only the cleaned speech as readable text. "
            "Do not add labels, quotes, or commentary."
        )
        if initial_prompt and initial_prompt.strip():
            base += (
                " This clip may continue a phrase; prior context (may be incomplete): "
                + initial_prompt.strip()[:500]
            )
        return base

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

        from google.generativeai.types import HarmBlockThreshold, HarmCategory

        wav_bytes = _pcm_f32_to_wav_bytes(audio, sample_rate)
        if len(wav_bytes) < 1000:
            log.debug("Audio very short for API (%s bytes)", len(wav_bytes))

        safety = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        parts = [
            self._instruction(initial_prompt),
            {"mime_type": "audio/wav", "data": wav_bytes},
        ]

        gen_cfg = {"temperature": 0.0, "max_output_tokens": 2048}
        try:
            response = self._model.generate_content(
                parts,
                generation_config=gen_cfg,
                safety_settings=safety,
            )
        except TypeError:
            response = self._model.generate_content(parts, generation_config=gen_cfg)

        try:
            text = (response.text or "").strip()
        except ValueError:
            log.warning(
                "Gemini returned no text (blocked or empty). feedback=%s",
                getattr(response, "prompt_feedback", None),
            )
            return ""

        return text
