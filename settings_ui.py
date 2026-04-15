"""Tkinter settings window (microphone, etc.); close hides to tray."""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Optional

from audio_devices import list_input_device_choices

log = logging.getLogger(__name__)


class SettingsWindow:
    def __init__(self, master: tk.Tk, app: Any) -> None:
        self._app = app
        self._win = tk.Toplevel(master)
        self._win.title("Dictation settings")
        self._win.minsize(420, 200)
        self._win.resizable(True, False)

        self._labels, self._ids = list_input_device_choices()

        main = ttk.Frame(self._win, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="Microphone").grid(row=0, column=0, sticky=tk.W, pady=(0, 4))
        row_mic = ttk.Frame(main)
        row_mic.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(0, 12))
        self._mic_var = tk.StringVar()
        self._combo = ttk.Combobox(
            row_mic,
            textvariable=self._mic_var,
            values=self._labels,
            state="readonly",
            width=52,
        )
        self._combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row_mic, text="Refresh list", command=self._refresh_mics).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        self._sync_mic_selection()

        hint = ttk.Label(
            main,
            text="Close this window to keep dictation running in the tray.",
            wraplength=400,
        )
        hint.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(0, 12))

        btn_row = ttk.Frame(main)
        btn_row.grid(row=3, column=0, columnspan=2, sticky=tk.E)
        ttk.Button(btn_row, text="Apply", command=self._apply).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(btn_row, text="Close to tray", command=self._hide).pack(side=tk.RIGHT)

        main.columnconfigure(0, weight=1)
        self._win.protocol("WM_DELETE_WINDOW", self._hide)
        self._win.bind("<Escape>", lambda e: self._hide())
        self._win.withdraw()

    def _index_for_cfg_device(self) -> int:
        current = self._app.cfg.get("audio_device")
        try:
            idx = self._ids.index(current if current is not None else None)
            return idx
        except ValueError:
            return 0

    def _sync_mic_selection(self) -> None:
        self._labels, self._ids = list_input_device_choices()
        self._combo.configure(values=self._labels)
        i = self._index_for_cfg_device()
        i = min(i, len(self._labels) - 1)
        if self._labels:
            self._mic_var.set(self._labels[i])
            self._combo.current(i)

    def _refresh_mics(self) -> None:
        self._sync_mic_selection()
        messagebox.showinfo("Microphones", "Device list refreshed.", parent=self._win)

    def _selected_device_id(self) -> Optional[int]:
        try:
            idx = self._combo.current()
        except tk.TclError:
            return None
        if idx < 0 or idx >= len(self._ids):
            return None
        return self._ids[idx]

    def _apply(self) -> None:
        if self._app.is_recording():
            messagebox.showwarning(
                "Busy",
                "Stop dictating (release the hotkey) before changing the microphone.",
                parent=self._win,
            )
            return
        dev = self._selected_device_id()
        if dev is not None:
            dev = int(dev)
        self._app.cfg["audio_device"] = dev
        try:
            self._app.audio.set_device(dev)
        except RuntimeError as e:
            messagebox.showerror("Microphone", str(e), parent=self._win)
            return
        try:
            self._app.save_config()
        except OSError as e:
            log.exception("Save config failed")
            messagebox.showerror("Save failed", str(e), parent=self._win)
            return
        messagebox.showinfo("Saved", "Microphone selection saved to config.yaml.", parent=self._win)

    def _hide(self) -> None:
        self._win.withdraw()

    def show(self) -> None:
        self._sync_mic_selection()
        self._win.deiconify()
        self._win.lift()
        self._win.focus_force()
