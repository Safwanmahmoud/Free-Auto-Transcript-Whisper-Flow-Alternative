"""
Local Windows dictation: hold Ctrl+Win (configurable), speak, text is typed at the caret.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import queue
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pystray
import tkinter as tk
import yaml
from dotenv import load_dotenv
from PIL import Image, ImageDraw

from audio_capture import AudioCapture
from hotkeys import PushToTalkHotkey
from text_injection import type_unicode
from transcriber import Transcriber
from vad_segmenter import VadSegmenter

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"


def load_config(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config_dict(cfg: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            cfg,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )


def build_transcriber(cfg: Dict[str, Any]) -> Any:
    provider = str(cfg.get("transcription_provider") or "local").strip().lower()
    lang = cfg.get("language")
    lang_opt = lang if lang else None
    strip_fillers = bool(cfg.get("strip_filler_words", True))
    if provider == "google":
        api_key = (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()
        if not api_key:
            log.error(
                "transcription_provider is 'google' but GOOGLE_API_KEY is missing. "
                "Add GOOGLE_API_KEY=... to .env in %s",
                BASE_DIR,
            )
            sys.exit(1)
        from google_transcriber import GoogleGeminiTranscriber

        return GoogleGeminiTranscriber(
            api_key=api_key,
            model_name=str(cfg.get("google_model") or "gemini-2.5-flash"),
            language=lang_opt,
            strip_fillers=strip_fillers,
        )
    return Transcriber(
        model_size=str(cfg.get("model_size", "turbo")),
        device=str(cfg.get("device", "cuda")),
        compute_type=str(cfg.get("compute_type", "float16")),
        language=lang_opt,
        strip_fillers=strip_fillers,
    )


def make_tray_image(armed: bool) -> Image.Image:
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    fill = (80, 200, 120, 255) if armed else (120, 120, 130, 255)
    outline = (40, 40, 45, 255)
    draw.ellipse((6, 6, size - 6, size - 6), fill=fill, outline=outline, width=2)
    return img


class DictationApp:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.require_arm = bool(cfg.get("require_arm", False))
        self._armed = not self.require_arm
        self.transcript_dir: Optional[Path] = None
        td = cfg.get("transcript_log_dir")
        if td:
            self.transcript_dir = Path(td).expanduser()
            self.transcript_dir.mkdir(parents=True, exist_ok=True)

        self._recording = threading.Event()
        self._shutdown = threading.Event()
        self._capture_worker: Optional[threading.Thread] = None
        self._inference_worker: Optional[threading.Thread] = None

        self._segment_queue: queue.Queue[Tuple[int, Any]] = queue.Queue()
        self._utterance_id = 0

        self._context_tail = ""
        self._last_segment_text = ""
        self._typed_any = False
        self._text_lock = threading.Lock()

        _mode = str(cfg.get("transcription_audio_mode") or "single").strip().lower()
        self._single_clip_audio = _mode in ("single", "one", "full")

        device = cfg.get("audio_device")
        self.audio = AudioCapture(
            sample_rate=int(cfg.get("sample_rate", 16000)),
            channels=1,
            device=device,
        )
        self.vad = VadSegmenter(
            sample_rate=int(cfg.get("sample_rate", 16000)),
            vad_threshold=float(cfg.get("vad_threshold", 0.5)),
            min_speech_seconds=float(cfg.get("min_speech_seconds", 0.35)),
            silence_end_seconds=float(cfg.get("silence_end_seconds", 0.35)),
            max_chunk_seconds=float(cfg.get("max_chunk_seconds", 2.5)),
        )
        self.transcriber = build_transcriber(cfg)

        mods = cfg.get("hotkey_modifiers") or ["ctrl", "cmd"]
        self.hotkey = PushToTalkHotkey(
            list(mods),
            on_start=self._on_ptt_start,
            on_stop=self._on_ptt_stop,
        )

        self.icon: Optional[pystray.Icon] = None
        self._icon_armed_state = self._armed
        self._tk_root: Optional[tk.Tk] = None
        self._settings_shell: Any = None

    def is_recording(self) -> bool:
        return self._recording.is_set()

    def save_config(self) -> None:
        save_config_dict(self.cfg, CONFIG_PATH)

    def _append_transcript_log(self, text: str) -> None:
        if not self.transcript_dir or not text.strip():
            return
        day = dt.datetime.now().strftime("%Y-%m-%d")
        path = self.transcript_dir / f"dictation_{day}.log"
        line = f"{dt.datetime.now().isoformat(timespec='seconds')}\t{text.strip()}\n"
        with path.open("a", encoding="utf-8") as f:
            f.write(line)

    def _reset_session_text(self) -> None:
        with self._text_lock:
            self._last_segment_text = ""
            self._typed_any = False
            self._context_tail = ""

    def _emit_text(self, raw: str) -> None:
        raw = (raw or "").strip()
        if not raw:
            return
        with self._text_lock:
            suffix = self.vad.merge_overlap_text(self._last_segment_text, raw)
            if not suffix:
                self._last_segment_text = ((self._last_segment_text + " " + raw).strip())[-600:]
                return
            needs_space = self._typed_any and suffix[0] not in ".,!?;:'\")]}\n"
            to_type = ((" " if needs_space else "") + suffix).strip()
            if not to_type:
                return
            self._typed_any = True
            self._last_segment_text = ((self._last_segment_text + " " + raw).strip())[-600:]
            self._context_tail = (self._context_tail + " " + to_type)[-500:]
        try:
            type_unicode(to_type)
        except OSError:
            log.exception("SendInput failed (elevated target window?)")
            return
        self._append_transcript_log(to_type)

    def _capture_loop(self) -> None:
        """Capture mic; VAD chunking or one full clip. Never block on transcription."""
        while not self._shutdown.is_set():
            if not self._recording.wait(timeout=0.5):
                if self._shutdown.is_set():
                    break
                continue
            if self._shutdown.is_set():
                break
            if not self._recording.is_set():
                continue
            uid = self._utterance_id
            try:
                if self._single_clip_audio:
                    pieces: List[np.ndarray] = []
                    while self._recording.is_set() and not self._shutdown.is_set():
                        chunk = self.audio.get_chunk(0.05)
                        if chunk is not None:
                            pieces.append(
                                np.asarray(chunk, dtype=np.float32).reshape(-1)
                            )
                    while True:
                        extra = self.audio.get_nowait()
                        if extra is None:
                            break
                        pieces.append(
                            np.asarray(extra, dtype=np.float32).reshape(-1)
                        )
                    if pieces:
                        full = np.concatenate(pieces)
                        if full.size > 0:
                            self._segment_queue.put((uid, full))
                else:
                    while self._recording.is_set() and not self._shutdown.is_set():
                        chunk = self.audio.get_chunk(0.05)
                        if chunk is not None:
                            self.vad.add_samples(chunk)
                            for seg in self.vad.pop_complete_segments(False):
                                self._segment_queue.put((uid, seg))
                    while True:
                        extra = self.audio.get_nowait()
                        if extra is None:
                            break
                        self.vad.add_samples(extra)
                    for seg in self.vad.pop_complete_segments(True):
                        self._segment_queue.put((uid, seg))
            finally:
                self.audio.stop()

    def _inference_loop(self) -> None:
        """Consume segment queue; GPU work never blocks microphone / VAD capture."""
        sr = int(self.cfg.get("sample_rate", 16000))
        while not self._shutdown.is_set():
            try:
                uid, pcm = self._segment_queue.get(timeout=0.35)
            except queue.Empty:
                continue
            if self._shutdown.is_set():
                break
            if uid != self._utterance_id:
                continue
            with self._text_lock:
                prompt = self._context_tail[-220:] if self._context_tail else None
            try:
                text = self.transcriber.transcribe(
                    pcm,
                    sample_rate=sr,
                    initial_prompt=prompt,
                )
            except Exception:
                log.exception("Transcription failed")
                continue
            if uid != self._utterance_id:
                continue
            if text:
                self._emit_text(text)

    def _on_ptt_start(self) -> None:
        if self.require_arm and not self._armed:
            log.debug("PTT ignored (not armed)")
            return
        self._utterance_id += 1
        self._reset_session_text()
        self.audio.drain_queue()
        if not self._single_clip_audio:
            self.vad.reset()
        try:
            self.audio.start()
        except Exception:
            log.exception("Failed to start microphone")
            return
        self._recording.set()

    def _on_ptt_stop(self) -> None:
        self._recording.clear()

    def _update_icon_image(self) -> None:
        if self.icon is None:
            return
        armed_visual = self._armed or not self.require_arm
        self.icon.icon = make_tray_image(armed_visual)

    def _toggle_arm(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        self._armed = not self._armed
        self._update_icon_image()
        if self.icon is not None:
            self.icon.menu = self._build_menu()

    def _open_log_dir(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        if self.transcript_dir and self.transcript_dir.is_dir():
            import subprocess

            subprocess.Popen(
                f'explorer "{self.transcript_dir}"',
                shell=True,
            )
        else:
            log.warning("No transcript_log_dir configured or folder missing")

    def _open_app_dir(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        import subprocess

        subprocess.Popen(f'explorer "{BASE_DIR}"', shell=True)

    def _menu_open_settings(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        if self._tk_root is not None:
            self._tk_root.after(0, self._open_settings_ui)

    def _open_settings_ui(self) -> None:
        from settings_ui import SettingsWindow

        if self._settings_shell is None:
            self._settings_shell = SettingsWindow(self._tk_root, self)
        self._settings_shell.show()

    def _quit(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        self._shutdown.set()
        self._recording.set()
        self.hotkey.stop()
        self.audio.stop()
        icon.stop()
        if self._tk_root is not None:
            self._tk_root.after(0, self._tk_root.quit)

    def _build_menu(self) -> pystray.Menu:
        items: List[pystray.MenuItem] = []
        if self.require_arm:
            items.append(
                pystray.MenuItem(
                    lambda t: f"Arm dictation: {'on' if self._armed else 'off'}",
                    self._toggle_arm,
                )
            )
            items.append(pystray.Menu.SEPARATOR)
        items.append(pystray.MenuItem("Settings...", self._menu_open_settings))
        items.append(pystray.Menu.SEPARATOR)
        items.append(pystray.MenuItem("Open app folder", self._open_app_dir))
        if self.transcript_dir:
            items.append(pystray.MenuItem("Open transcript folder", self._open_log_dir))
        items.append(pystray.Menu.SEPARATOR)
        items.append(pystray.MenuItem("Quit", self._quit))
        return pystray.Menu(*items)

    def _preload_models(self) -> None:
        try:
            if not self._single_clip_audio:
                self.vad.load()
            self.transcriber.load()
            log.info("Models ready.")
        except Exception:
            log.exception("Model preload failed — will retry on first use")

    def _run_tray_icon(self) -> None:
        if self.icon is None:
            return
        self.icon.run()

    def run(self) -> None:
        self._tk_root = tk.Tk()
        self._tk_root.withdraw()
        self._tk_root.title("Dictation")
        self._tk_root.protocol("WM_DELETE_WINDOW", lambda: None)

        self._capture_worker = threading.Thread(target=self._capture_loop, daemon=True)
        self._inference_worker = threading.Thread(target=self._inference_loop, daemon=True)
        self._capture_worker.start()
        self._inference_worker.start()

        threading.Thread(target=self._preload_models, daemon=True).start()

        self.hotkey.start()

        initial = make_tray_image(self._armed or not self.require_arm)
        self.icon = pystray.Icon(
            "local_dictation",
            initial,
            "Dictation",
            menu=self._build_menu(),
        )
        threading.Thread(target=self._run_tray_icon, daemon=True).start()
        self._tk_root.mainloop()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv(BASE_DIR / ".env")
    if not CONFIG_PATH.is_file():
        log.error("Missing %s", CONFIG_PATH)
        sys.exit(1)
    cfg = load_config(CONFIG_PATH)
    app = DictationApp(cfg)
    app.run()


if __name__ == "__main__":
    main()
