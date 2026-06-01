# FORMA — FFT-based Orbit Response Matrix Analyzer

FORMA measures the orbit response matrix (ORM) of a beamline by driving each
corrector with a sinusoidal excitation and extracting the response at every beam
position monitor (BPM) using an FFT. Working in the frequency domain lets the
response be separated from drift and broadband noise, which yields a cleaner ORM
than a step-and-settle scan.

The application is built around an interactive scan workflow: configure the
correctors and BPMs, run a sinusoidal scan, monitor beam loss and orbit
deviation for safety, and compute and visualize the resulting response matrix.

## Requirements

- Python 3.10+
- The Python packages listed in `requirements.txt`
- `acsys` — Fermilab's ACNET data-acquisition library, installed separately from
  Fermilab sources. The application talks to the control system through it.
- A valid Kerberos ticket in the `FNAL.GOV` realm. FORMA manages tickets through
  the system `kinit` / `klist` / `kdestroy` tools.

`tkinter` ships with most Python installations and is used for the main GUI.
`PyQt5` is required only for the advanced analysis viewer.

## Installation

```
pip install -r requirements.txt
```

Install `acsys` separately according to your local Fermilab controls
environment.

## Running

From the repository root:

```
python forma.py
```

On Windows you can also double-click `RUN_FORMA.bat`, which launches the
application and keeps the console open if it exits with an error.

## Companion tools

- **Advanced analysis viewer** (`analysis_viewer.py`) — a PyQt5 viewer for
  inspecting multi-plane response matrices. It can be opened from within FORMA
  ("Open Advanced Analysis") or run directly on a saved CSV.
- **ACSys connection monitor** (`acsys_connection_monitor.py`) — a read-only
  diagnostic that opens a single persistent ACSys stream (default `G:AMANDA` at
  5 Hz) using FORMA's own backend and reports stalls, drops, recoveries, and an
  end-of-session reliability summary. Useful for answering "how often does FORMA
  lose ACSys over a long run?"

## Repository layout

```
forma.py                     Main application and GUI
analysis_viewer.py           PyQt5 advanced analysis viewer (also imported by forma.py)
acsys_connection_monitor.py  Read-only ACSys connectivity diagnostic
RUN_FORMA.bat                Windows launcher

backend/                     Control-system communication
  acnet_scanner.py             ACSys / DPM scanning and data streaming
  beam_interlock.py            Beam-loss interlock monitoring
config/
  settings.py                  Configuration constants (UI, ACNET, scan, safety)
ui/
  dialogs.py                   Login, device-selection, and confirmation dialogs
utils/
  safety_monitor.py            Orbit-deviation safety monitoring
  bounded_data.py              Bounded buffers for live data
  file_operations.py           Scan-data file I/O
  text_redirector.py           Console/log redirection helper
models/
  scan_config.py               Scan configuration data models
  safety_config.py             Safety thresholds and baseline models
  beam_interlock.py            Beam status and trip-event models
auth/
  kerberos_manager.py          Kerberos ticket management

docs/
  FORMA_Manual.pdf             User manual
  FORMA_Manual.tex             User manual source
```

## Documentation

See `docs/FORMA_Manual.pdf` for the full user manual.
