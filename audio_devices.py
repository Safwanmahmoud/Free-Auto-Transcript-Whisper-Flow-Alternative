"""List PortAudio input devices for the settings UI."""

from __future__ import annotations

from typing import List, Optional, Tuple

import sounddevice as sd


def list_input_device_choices() -> Tuple[List[str], List[Optional[int]]]:
    """
    Return (labels, device_ids) aligned by index.
    device_ids[0] is always None = system default.
    """
    labels: List[str] = ["Default (system input)"]
    ids: List[Optional[int]] = [None]
    try:
        devices = sd.query_devices()
    except OSError:
        return labels, ids
    for i, d in enumerate(devices):
        if int(d.get("max_input_channels") or 0) < 1:
            continue
        name = str(d.get("name", f"device {i}"))
        if len(name) > 72:
            name = name[:69] + "..."
        labels.append(f"[{i}] {name}")
        ids.append(i)
    return labels, ids
