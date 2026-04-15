"""Microphone capture via sounddevice into a thread-safe queue."""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Optional

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)


class AudioCapture:
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device: Optional[Any] = None,
        blocksize: int = 1024,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.blocksize = blocksize
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()

    def set_device(self, device: Optional[Any]) -> None:
        """Update input device for the next capture session (not while stream is running)."""
        with self._lock:
            if self._stream is not None:
                raise RuntimeError("Cannot change microphone while recording is active")
            self.device = device

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            log.debug("Audio status: %s", status)
        mono = indata.copy()
        if mono.shape[1] > 1:
            mono = mono.mean(axis=1, keepdims=True)
        self._q.put(mono.astype(np.float32, copy=False))

    def start(self) -> None:
        with self._lock:
            if self._stream is not None:
                return
            self._stream = sd.InputStream(
                device=self.device,
                channels=self.channels,
                samplerate=self.sample_rate,
                dtype=np.float32,
                blocksize=self.blocksize,
                callback=self._callback,
            )
            self._stream.start()
            log.info(
                "Audio capture started sr=%s device=%s",
                self.sample_rate,
                self.device,
            )

    def stop(self) -> None:
        with self._lock:
            if self._stream is None:
                return
            self._stream.stop()
            self._stream.close()
            self._stream = None
            log.info("Audio capture stopped.")

    def drain_queue(self) -> None:
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def get_chunk(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_nowait(self) -> Optional[np.ndarray]:
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None
