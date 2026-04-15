"""Global hold-to-talk: all configured modifiers must be pressed."""

from __future__ import annotations

import logging
import threading
from typing import Callable, Iterable, List, Optional, Set

from pynput import keyboard
from pynput.keyboard import Key

log = logging.getLogger(__name__)

_NAME_TO_KEYS = {
    "ctrl": {Key.ctrl, Key.ctrl_l, Key.ctrl_r},
    "shift": {Key.shift, Key.shift_l, Key.shift_r},
    "alt": {Key.alt, Key.alt_l, Key.alt_r},
    "cmd": {Key.cmd, Key.cmd_l, Key.cmd_r},
}


def _normalize_modifiers(names: Iterable[str]) -> List[Set[Key]]:
    groups: List[Set[Key]] = []
    for n in names:
        key = n.strip().lower()
        if key not in _NAME_TO_KEYS:
            raise ValueError(f"Unknown hotkey modifier {n!r}; use ctrl, shift, alt, cmd")
        groups.append(_NAME_TO_KEYS[key])
    return groups


def _key_in_group(key: Key, group: Set[Key]) -> bool:
    if key in group:
        return True
    try:
        vk = key.vk
    except AttributeError:
        return False
    for g in group:
        try:
            if g.vk == vk:
                return True
        except AttributeError:
            continue
    return False


class PushToTalkHotkey:
    def __init__(
        self,
        modifier_names: List[str],
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
    ) -> None:
        self._groups = _normalize_modifiers(modifier_names)
        self._on_start = on_start
        self._on_stop = on_stop
        self._pressed: Set[Key] = set()
        self._active = False
        self._listener: Optional[keyboard.Listener] = None
        self._lock = threading.Lock()

    def _update_active(self) -> None:
        want = all(
            any(_key_in_group(k, g) for k in self._pressed) for g in self._groups
        )
        if want and not self._active:
            self._active = True
            log.debug("Push-to-talk start")
            self._on_start()
        elif not want and self._active:
            self._active = False
            log.debug("Push-to-talk stop")
            self._on_stop()

    def _on_press(self, key) -> None:
        try:
            with self._lock:
                self._pressed.add(key)
                self._update_active()
        except Exception:
            log.exception("on_press")

    def _on_release(self, key) -> None:
        try:
            with self._lock:
                self._pressed.discard(key)
                self._update_active()
        except Exception:
            log.exception("on_release")

    def start(self) -> None:
        if self._listener is not None:
            return
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()
        log.info("Hotkey listener started modifiers=%s", self._groups)

    def stop(self) -> None:
        if self._listener is None:
            return
        self._listener.stop()
        self._listener = None
        with self._lock:
            self._pressed.clear()
            self._active = False
        log.info("Hotkey listener stopped.")
