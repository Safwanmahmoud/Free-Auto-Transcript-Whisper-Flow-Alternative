# Local Windows dictation (Python)

Tray app: **hold the configured modifiers** (default **Ctrl + Windows**) and speak; recognized text is typed at the text cursor via `SendInput`. **Silero VAD** splits speech locally; transcription is either **Google Gemini** (cloud) or **faster-whisper** (local GPU), depending on [`config.yaml`](config.yaml).

## Prerequisites

- Windows 10 or 11, 64-bit
- Python 3.10+ (3.11 recommended)
- NVIDIA driver installed (for CUDA)
- Microphone

## Setup

1. Create a virtual environment in this folder:

   ```bat
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **GPU (RTX 3050 and similar):** `faster-whisper` pulls `ctranslate2` with CUDA support on Windows when you install the CUDA-enabled wheel. If transcribe fails on GPU, install the matching package from the [CTranslate2 docs](https://opennmt.net/CTranslate2/installation.html) or set `device: cpu` in `config.yaml` (slower).

3. **PyTorch (CPU)** is used only to run Silero VAD from `torch.hub`. The default `pip install torch` CPU wheel is enough.

4. Edit [`config.yaml`](config.yaml):
   - **`transcription_provider: google`** (default in repo): create `.env` next to `main.py` with `GOOGLE_API_KEY=` from [Google AI Studio](https://aistudio.google.com/apikey). Optional `google_model:` e.g. `gemini-2.5-flash` (`gemini-2.0-flash` is deprecated for new API keys). Audio is sent to Google’s API (not private on-device).
   - **`transcription_audio_mode`:** `single` (default) sends **one** clip per hotkey press (when you release). `chunked` splits on pauses and transcribes while you hold.
   - **`strip_filler_words`:** `true` (default) tells the model to drop fillers like “uh”/“um” and obvious false starts; set `false` for a more verbatim transcript.
   - **`transcription_provider: local`**: uses faster-whisper (`model_size` e.g. `turbo`, `distil-large-v3`), plus `device` / `compute_type` as before.
   - Also: optional `transcript_log_dir`, `require_arm`, etc.

   Copy [`.env.example`](.env.example) to `.env` for the key; **do not commit `.env`** (it is gitignored).

5. Run:

   ```bat
   run.bat
   ```

   Or: `python main.py`

6. **Tray + settings window:** The app stays in the **system tray** after you start it. Use **Settings...** in the tray menu to pick a **microphone**, then **Apply** (saves `config.yaml`). **Close to tray** (or the window X) hides the UI; dictation keeps running until **Quit**.

## Start with Windows (background + tray)

The app is **already a background tray app** once it is running: there is no extra “background mode” to turn on. You only need Windows to **start it automatically at sign-in**.

### Option A — Startup folder (simplest)

1. **Keep the project in a stable folder** (any path is fine, including OneDrive; avoid renaming the folder later or you must fix the shortcut).
2. **Test the silent launcher once** by double‑clicking [`run_startup.bat`](run_startup.bat). You should see **no console window**, only the **tray icon**. If that works, continue.
3. **Create a shortcut** to `run_startup.bat`:
   - Right‑click `run_startup.bat` → **Show more options** → **Create shortcut** (or send to Desktop).
4. Open the Startup folder: press **Win + R**, type `shell:startup`, Enter.
5. **Move or copy** that shortcut into the Startup folder.
6. **Sign out and sign back in** (or reboot) and confirm the tray icon appears.

**Optional (no `.bat` flash at all):** delete the shortcut to the `.bat` and instead create a shortcut whose **Target** is your venv’s `pythonw.exe`, **Arguments** are `main.py`, and **Start in** is this project folder, for example:

`"D:\OneDrive\Desktop\New folder\.venv\Scripts\pythonw.exe"`  
Arguments: `"D:\OneDrive\Desktop\New folder\main.py"`  
Start in: `D:\OneDrive\Desktop\New folder`  

(Adjust the drive and folder to match your PC.)

### Option B — Task Scheduler

1. Open **Task Scheduler** → **Create Task…** (not “Create Basic Task”).
2. **General:** name e.g. `Dictation`; choose **Run only when user is logged on** (so the tray UI works).
3. **Triggers:** **New…** → **At log on** → your user account.
4. **Actions:** **New…** → **Start a program**:
   - Program: full path to `.venv\Scripts\pythonw.exe` (or `venv\Scripts\pythonw.exe`).
   - Add arguments: full path to `main.py` in quotes.
   - Start in: full path to this project folder.
5. **Conditions:** optionally uncheck **Start only if on AC power** for a laptop.
6. OK, then test with **Run** in Task Scheduler.

### Daily use

- **`run.bat`** or **`python main.py`**: console window stays open (good for seeing errors).
- **`run_startup.bat`** or **`pythonw main.py`**: **no** console; tray only — best for Startup.

## PyInstaller (optional)

From the venv, after `pip install pyinstaller`:

```bat
pyinstaller --noconfirm --onedir --windowed --name LocalDictation ^
  --add-data "config.yaml;." main.py
```

Copy `config.yaml` next to the built executable if it is not bundled as expected, then add that folder to Startup as above.

## Usage

- Hold **Ctrl + Win** (or the modifiers in `config.yaml`) and speak; release to flush the last words.
- If `require_arm: true`, use the tray menu **Arm dictation** before the hotkey works.
- Tray: **Open app folder**, optional **Open transcript folder**, **Quit**.

## Troubleshooting

- **`RuntimeError: Library cublas64_12.dll is not found`:** The CUDA build of CTranslate2 expects **CUDA 12** cuBLAS on your `PATH` (often from the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)). The app will **automatically fall back to CPU** (slower) after the first failed GPU encode. To force CPU only, set `device: cpu` and `compute_type: int8` in `config.yaml`.

## Notes

- **Ctrl + Win** can conflict with some Windows shortcuts; if it misbehaves, try e.g. `hotkey_modifiers: [ctrl, alt]` and hold **Ctrl + Alt** (you can document your own combo — any set listed under `hotkeys.py` names: `ctrl`, `cmd`, `alt`, `shift`).
- Typing into **elevated (Run as administrator)** windows from a normal app may not work; run this app elevated only if you need that.
- First run downloads **Silero VAD** and **Whisper** weights; keep internet once, or download models ahead of time per faster-whisper docs.
