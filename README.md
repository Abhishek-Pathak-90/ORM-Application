<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f172a,45:0ea5e9,100:38bdf8&height=210&section=header&text=FORMA&fontSize=80&fontColor=f8fafc&fontAlignY=38&desc=FFT-based%20Orbit%20Response%20Matrix%20Analyzer&descSize=20&descColor=e2e8f0&descAlignY=62" alt="FORMA — FFT-based Orbit Response Matrix Analyzer" width="100%"/>

<br/>

**Measure a beamline's orbit response matrix in the frequency domain — clean, drift-immune, and safe.**

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![GUI](https://img.shields.io/badge/GUI-Tkinter-FF6F00?style=for-the-badge)](#)
[![Analysis](https://img.shields.io/badge/Analysis-PyQt5-41CD52?style=for-the-badge&logo=qt&logoColor=white)](#)
[![Control System](https://img.shields.io/badge/Control_System-ACNET%20%2F%20ACSys-8B5CF6?style=for-the-badge)](#)
[![Auth](https://img.shields.io/badge/Auth-Kerberos-EF4444?style=for-the-badge)](#)

[![Status](https://img.shields.io/badge/status-prototype-FBBF24?style=for-the-badge)](#)
[![License](https://img.shields.io/badge/license-Internal-64748B?style=for-the-badge)](#)
[![Fermilab](https://img.shields.io/badge/Fermilab-PIP--II%20HLA-005DAA?style=for-the-badge)](https://ad.fnal.gov/)

</div>

---

## 🎯 What is FORMA?

**FORMA** drives each corrector magnet with a **sinusoidal excitation** and extracts the
beam's response at every **beam position monitor (BPM)** using an **FFT**. Because the
response lives at a single known frequency, it separates cleanly from slow orbit drift and
broadband noise — yielding a far cleaner **orbit response matrix (ORM)** than a
conventional step-and-settle scan.

Everything happens inside one interactive workflow: pick your correctors and BPMs, run the
scan, watch beam loss and orbit deviation in real time, and compute and visualize the
resulting matrix.

---

## ✨ Highlights

| | Feature | What it does |
|:--:|:--|:--|
| 🌊 | **FFT-based ORM** | Sinusoidal corrector drive + single-frequency FFT readout for a drift-immune response matrix |
| 📡 | **Live ACNET / ACSys streaming** | Streams BPM and device data through a persistent DPM context — the same path a real scan uses |
| 🛡️ | **Orbit-deviation safety monitor** | Watches BPM orbit against a baseline and aborts on excursion |
| 🚨 | **Beam-loss interlock** | Monitors loss devices during the scan and trips beam off on threshold breach |
| 🔐 | **Kerberos authentication** | Manages `FNAL.GOV` tickets via the system `kinit` / `klist` / `kdestroy` |
| 📈 | **Live plots & CSV export** | Real-time scan visualization; results saved with readback **and** `.SETTING` columns |
| 🔬 | **Advanced analysis viewer** | Standalone PyQt5 tool for inspecting multi-plane response matrices |
| 🩺 | **Connection monitor** | Read-only diagnostic that quantifies ACSys reliability over long runs |

---

## 🧭 How it works

```text
   ┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
   │  Configure  │ ──▶ │  Sinusoidal  │ ──▶ │   FFT at    │ ──▶ │   Orbit      │
   │ correctors  │     │   corrector  │     │  the drive  │     │  Response    │
   │  & BPMs     │     │  excitation  │     │  frequency  │     │  Matrix      │
   └─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
          │                    │                                        │
          │                    └──── safety monitor + beam interlock ───┘
          ▼
   live BPM / device stream  ◀── ACNET / ACSys (DPM)
```

---

## 📦 Requirements

<div align="center">

[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](#)
[![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)](#)
[![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white)](#)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)](#)
[![PyQt5](https://img.shields.io/badge/PyQt5-41CD52?style=flat-square&logo=qt&logoColor=white)](#)

</div>

- **Python 3.10+**
- The packages in [`requirements.txt`](requirements.txt)
- **`acsys`** — Fermilab's ACNET data-acquisition library, installed separately from Fermilab sources
- A valid **Kerberos ticket** in the `FNAL.GOV` realm

> `tkinter` ships with most Python installs and powers the main GUI. `PyQt5` is needed only for the advanced analysis viewer.

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

Install `acsys` separately according to your local Fermilab controls environment.

---

## ▶️ Running

From the repository root:

```bash
python forma.py
```

On Windows, double-click <kbd>RUN_FORMA.bat</kbd> — it launches the app and keeps the
console open if it exits with an error.

---

## 🧰 Companion tools

<details open>
<summary><b>🔬 Advanced analysis viewer — <code>analysis_viewer.py</code></b></summary>

<br/>

A PyQt5 viewer for inspecting multi-plane response matrices. Open it from inside FORMA via
**"Open Advanced Analysis"**, or run it directly on a saved CSV.

</details>

<details>
<summary><b>🩺 ACSys connection monitor — <code>acsys_connection_monitor.py</code></b></summary>

<br/>

A read-only diagnostic that opens a single persistent ACSys stream (default `G:AMANDA` at
5 Hz) using FORMA's own backend and reports **stalls**, **drops**, **recoveries**, and an
end-of-session reliability summary. It answers: *how often does FORMA lose ACSys over a
long run?*

</details>

---

## 🗂️ Repository layout

```text
forma.py                     ▸ Main application and GUI
analysis_viewer.py           ▸ PyQt5 advanced analysis viewer (also imported by forma.py)
acsys_connection_monitor.py  ▸ Read-only ACSys connectivity diagnostic
RUN_FORMA.bat                ▸ Windows launcher

backend/                     ▸ Control-system communication
  ├─ acnet_scanner.py          ACSys / DPM scanning and data streaming
  └─ beam_interlock.py         Beam-loss interlock monitoring
config/
  └─ settings.py               Configuration constants (UI, ACNET, scan, safety)
ui/
  └─ dialogs.py                Login, device-selection, and confirmation dialogs
utils/
  ├─ safety_monitor.py         Orbit-deviation safety monitoring
  ├─ bounded_data.py           Bounded buffers for live data
  ├─ file_operations.py        Scan-data file I/O
  └─ text_redirector.py        Console/log redirection helper
models/
  ├─ scan_config.py            Scan configuration data models
  ├─ safety_config.py          Safety thresholds and baseline models
  └─ beam_interlock.py         Beam status and trip-event models
auth/
  └─ kerberos_manager.py       Kerberos ticket management

docs/
  ├─ FORMA_Manual.pdf          User manual
  └─ FORMA_Manual.tex          User manual source
```

---

## 📖 Documentation

The full user manual lives in **[`docs/FORMA_Manual.pdf`](docs/FORMA_Manual.pdf)**.

<div align="center">

<br/>

<sub>Part of the <b>PIP-II High-Level Applications</b> prototypes · Fermilab Accelerator Directorate</sub>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:38bdf8,55:0ea5e9,100:0f172a&height=120&section=footer" width="100%"/>

</div>
