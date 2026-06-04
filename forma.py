"""
FORMA - FFT-based Orbit Response Matrix Analyzer

ORM data convention (carried over from v4):
    The ORM is computed from actual READBACKS for setting devices (correctors),
    NOT the .SETTING property (commanded setpoints). This makes the matrix account
    for discrepancies between commanded settings and actual corrector output
    (power-supply limits, saturation, calibration error, slew-rate limiting).
    .SETTING (commanded) values are recorded alongside the readbacks in the scan
    CSV for comparison (columns with a .SETTING suffix, e.g. H:C1 for readback,
    H:C1.SETTING for setpoint) but are EXCLUDED from the ORM analysis. Loading v4
    files that already contain .SETTING columns is still supported.

CHANGES IN v5 (hardening + safety/observability pass, #1-#8):
    #1 Loss-monitor trip restores correctors to nominal BEFORE disabling the beam,
       and tripping requires a positive read-back confirmation that the beam is
       actually OFF (BeamInterlockMonitor.trip_beam) before continuing; an
       unconfirmed trip escalates to a modal instead of silently proceeding.
    #2 ACNET SET/disable replies are now status-validated: apply_settings_once and
       disable_beam return a confirmed/rejected bool instead of treating any reply
       as success. Rejected corrector SETs are recorded to Device Health.
    #3 Read paths log error/status replies and leave the slot None instead of
       mistaking a status for a reading.
    #4/#5 New "Device Health" tab renders per-device missed / non-finite /
       set-rejected counts and last-seen after each scan; beam ON/OFF/Refresh
       buttons and the interlock-enable toggle are disabled while a scan runs.
    #5 Beam button handlers and pre-scan beam/loss reads run on a short worker
       thread (still through the single shared ACNET executor) and marshal results
       back to Tk via self.after(0, ...), so they no longer block the Tk main loop.
    #6 NaN/inf readings are guarded in the safety monitor (not buffered, never
       poison the mean) and reported to Device Health without aborting.
    #7 Scan CSV / response-matrix CSV / JSON meta / safety-log writes go through
       utils.file_operations atomic helpers (temp file + os.replace, fsync,
       encoding='utf-8'); the *_meta.json filename is derived from the path's real
       extension instead of str.replace('.csv', ...) (which replaced every match).
    #8 ORM error propagation includes the manual's 1/2 factor on BOTH terms
       (eq:errorprop).

    A pinned-acsys compatibility check (utils.acsys_compat, EXPECTED_ACSYS_VERSION)
    runs at acnet_scanner import time so an incompatible DPM reply API fails loud.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import font as tkfont
import pandas as pd
import numpy as np
import os
import threading
import time
import subprocess
from datetime import datetime
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm
from functools import reduce
import getpass
import asyncio
import csv
import io
import math
import queue
import warnings
from scipy.fft import fft, fftfreq
import copy
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional

# Import from modular structure
from backend.acnet_scanner import AcnetScanner, CredentialExpiredError
from auth.kerberos_manager import KerberosManager
from models.scan_config import ScanDeviceConfig, ScanRunConfig
from models.safety_config import SafetyConfiguration, NominalSettings, SafetyThresholdType, DeviceBaseline, ViolationType, MIN_BUFFER_SIZE
from models.beam_interlock import BeamInterlockConfig, BeamStatus, BeamTripEvent
from backend.beam_interlock import BeamInterlockMonitor
from ui.dialogs import KerberosLoginDialog, DeviceSelectionDialog, ConfirmationDialog
from utils.safety_monitor import SafetyMonitor
from utils.file_operations import atomic_write_text, atomic_to_csv

# Import configuration
from config.settings import (
    PALETTE, DEFAULT_FONT_FAMILY, DEFAULT_FONT_SIZE, HEADING_FONT_SIZE,
    SAFETY_ENABLED_BY_DEFAULT, SAFETY_LOG_DIR, SAFETY_LOG_FILE,
    DEFAULT_PER_DEVICE_WARNING_THRESHOLD, DEFAULT_PER_DEVICE_ABORT_THRESHOLD,
    DEFAULT_OVERALL_WARNING_THRESHOLD, DEFAULT_OVERALL_ABORT_THRESHOLD,
    DEFAULT_BASELINE_SAMPLES, DEFAULT_SAFETY_BUFFER_SIZE, SAFETY_ACTIVE_COLOR,
    SAFETY_WARNING_COLOR, SAFETY_ABORT_COLOR, SAFETY_DISABLED_COLOR,
    ACNET_MAX_CONSECUTIVE_TIMEOUTS, ACNET_RESTORE_VERIFY_TOLERANCE,
    BEAM_INTERLOCK_ENABLED_BY_DEFAULT, DEFAULT_LOSS_MONITORS,
    BEAM_CONTROL_DRF, LOSS_MONITOR_MAX_CONSECUTIVE_MISSES,
)

# ############################################################################
#
# Helper Classes
#
# ############################################################################

class TextRedirector(object):
    """A class to redirect stdout/stderr to a tkinter Text widget (thread-safe)."""
    def __init__(self, widget, tag="INFO"):
        self.widget = widget
        self.tag = tag

    def write(self, str_in):
        tag = self.tag
        if str_in.strip().startswith("[SUCCESS]"):
            tag = "SUCCESS"
        elif str_in.strip().startswith("[WARNING]"):
            tag = "WARNING"
        elif str_in.strip().startswith("[ERROR]"):
            tag = "ERROR"
        try:
            self.widget.after(0, self._do_insert, str_in, tag)
        except RuntimeError:
            pass  # Widget destroyed

    def _do_insert(self, str_in, tag):
        try:
            self.widget.configure(state="normal")
            self.widget.insert("end", str_in, (tag,))
            self.widget.see("end")
            self.widget.configure(state="disabled")
        except tk.TclError:
            pass  # Widget destroyed

    def flush(self):
        pass


class DeviceControlApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FORMA - FFT-based Orbit Response Matrix Analyzer")

        # Adaptive window sizing for different screen sizes
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Use 90% of screen height, but max 900px width
        window_width = min(900, int(screen_width * 0.8))
        window_height = min(int(screen_height * 0.9), screen_height - 100)  # Leave space for taskbar

        # Center the window
        x_position = (screen_width - window_width) // 2
        y_position = max(0, (screen_height - window_height) // 2 - 30)  # Slight offset up

        self.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
        self.minsize(800, 600)  # Minimum usable size

        self._configure_theme()

        self.scanner = AcnetScanner()
        self.setting_devices = []
        self.reading_devices = []
        # BPMs found dead (all-zero) during baseline measurement — excluded
        # from scan readback, CSV, and the response matrix.
        self.dead_bpms = set()
        self.device_setting_entries = {}
        self.is_reading = False
        self.is_scanning = False
        self.plot_data = {}
        self.device_stats = {}
        # Per-device health accounting populated by the scan worker (missed /
        # non-finite / rejected-SET counts + last_seen). Initialized empty so the
        # UI can render it before any scan runs; reset to {} at each scan start.
        self.device_health = {}
        self.last_analysis_source = None
        self._analysis_processes = []
        self.data_queue = queue.Queue()
        self._closing = False
        self._scan_thread = None
        self.stop_scan_flag = threading.Event()
        self.fft_data = None
        self._aligned_fft_data = None
        self.fft_scan_params = None
        self.scan_progressbar = None
        self.rms_progressbar = None
        self.response_matrix_data = None
        self.response_reading_devices = []
        self.response_setting_devices = []
        self.response_error_matrix = None
        self.response_bpm_amplitudes = None
        self.response_corrector_amplitudes = None
        self.response_corrector_noise = {}
        self.response_bpm_noise = {}
        self.response_frequency_map = {}
        self._setting_device_aliases = {}
        self.response_setting_labels = []
        self.response_reading_labels = []
        self._horizontal_indices = []
        self._vertical_indices = []
        self.response_colorbar_h = None
        self.response_colorbar_v = None
        self._auth_status_value = False
        self._last_run_config: Optional[ScanRunConfig] = None
        self._scan_preview_window = None
        self.preview_scan_button = None
        
        # Default setting/reading values
        self.setting_role = tk.StringVar(value="OPERATOR")
        self.acnet_event = tk.StringVar(value="@p,1000")
        self.dpm_node = tk.StringVar(value="")  # empty = auto-discover
        self.plotted_device = tk.StringVar()
        self.scan_data_path = tk.StringVar(value=os.getcwd())
        self.scan_mode = tk.StringVar(value="Simultaneous")
        # Opt-in: drive a Simultaneous scan with the persistent advance-on-event
        # engine (machine-validated 2026-06-04, ~6x faster). Default OFF keeps the
        # existing serial _synchronous_scan_loop as the path.
        self.use_persistent_scan = tk.BooleanVar(value=False)
        self.auto_calc_orm = tk.BooleanVar(value=False)
        
        # Scan parameters
        self.scan_points_per_superperiod = tk.IntVar(value=100)
        self.scan_num_superperiods = tk.IntVar(value=1)
        self.rms_samples = tk.IntVar(value=50)

        # Safety system
        self.safety_config = SafetyConfiguration(
            enabled=SAFETY_ENABLED_BY_DEFAULT,
            per_device_warning_threshold=DEFAULT_PER_DEVICE_WARNING_THRESHOLD,
            per_device_abort_threshold=DEFAULT_PER_DEVICE_ABORT_THRESHOLD,
            overall_warning_threshold=DEFAULT_OVERALL_WARNING_THRESHOLD,
            overall_abort_threshold=DEFAULT_OVERALL_ABORT_THRESHOLD,
            buffer_size=DEFAULT_SAFETY_BUFFER_SIZE
        )
        self.safety_monitor = SafetyMonitor(
            self.safety_config,
            violation_callback=self._handle_safety_violation,
            warning_callback=self._handle_safety_warning
        )
        self.nominal_settings = NominalSettings()
        self.safety_baseline_samples = tk.IntVar(value=DEFAULT_BASELINE_SAMPLES)
        self.safety_enabled = tk.BooleanVar(value=SAFETY_ENABLED_BY_DEFAULT)
        self.safety_threshold_type = tk.StringVar(value="per_device")
        self.safety_per_device_warning = tk.DoubleVar(value=DEFAULT_PER_DEVICE_WARNING_THRESHOLD)
        self.safety_per_device_abort = tk.DoubleVar(value=DEFAULT_PER_DEVICE_ABORT_THRESHOLD)
        self.safety_overall_warning = tk.DoubleVar(value=DEFAULT_OVERALL_WARNING_THRESHOLD)
        self.safety_overall_abort = tk.DoubleVar(value=DEFAULT_OVERALL_ABORT_THRESHOLD)
        self.safety_baselines_measured = False
        self.safety_status_label = None
        self.safety_baseline_tree = None

        # Beam interlock system
        self.beam_interlock_config = BeamInterlockConfig(
            enabled=BEAM_INTERLOCK_ENABLED_BY_DEFAULT,
            loss_monitors=dict(DEFAULT_LOSS_MONITORS),
        )
        self.beam_interlock = BeamInterlockMonitor(
            self.beam_interlock_config, self.scanner)

        # Main frame to hold tabs and log
        main_frame = ttk.Frame(self, style="Main.TFrame")
        main_frame.pack(fill="both", expand=True)

        # Create Tab Control
        self.tabControl = ttk.Notebook(main_frame)
        self.settings_tab = ttk.Frame(self.tabControl, style="Main.TFrame")
        self.scan_tab = ttk.Frame(self.tabControl, style="Main.TFrame")
        self.safety_tab = ttk.Frame(self.tabControl, style="Main.TFrame")
        self.rms_tab = ttk.Frame(self.tabControl, style="Main.TFrame")
        self.reading_tab = ttk.Frame(self.tabControl, style="Main.TFrame")
        self.stats_tab = ttk.Frame(self.tabControl, style="Main.TFrame")
        self.fft_tab = ttk.Frame(self.tabControl, style="Main.TFrame")
        self.response_tab = ttk.Frame(self.tabControl, style="Main.TFrame")
        self.device_health_tab = ttk.Frame(self.tabControl, style="Main.TFrame")
        self.log_tab = ttk.Frame(self.tabControl, style="Main.TFrame")
        self.tabControl.add(self.settings_tab, text='Device Settings & Selection')
        self.tabControl.add(self.scan_tab, text='Scan Settings')
        self.tabControl.add(self.safety_tab, text='Safety Configuration')
        self.tabControl.add(self.rms_tab, text='Baseline RMS')
        self.tabControl.add(self.reading_tab, text='Data Reading and Plots')
        self.tabControl.add(self.stats_tab, text='Statistics')
        self.tabControl.add(self.fft_tab, text='FFT Analysis')
        self.tabControl.add(self.response_tab, text='Response Matrix')
        self.tabControl.add(self.device_health_tab, text='Device Health')
        self.tabControl.add(self.log_tab, text='Activity Log')
        self.tabControl.pack(expand=1, fill="both", padx=10, pady=10)

        # Populate tabs
        self._create_settings_widgets()
        self._create_scan_widgets()
        self._create_safety_widgets()
        self._create_reading_widgets()
        self._create_stats_widgets()
        self._create_fft_widgets()
        self._create_rms_widgets()
        self._create_response_matrix_widgets()
        self._create_device_health_widgets()
        self._create_log_viewer()

        # Add a menu for login
        self._create_menu()
        
        # Start the GUI polling loop
        self._poll_data_queue()


    def _configure_theme(self):
        """Set up a modern dark theme for the application."""
        self.configure(bg=PALETTE['background'])
        style = ttk.Style()
        self.style = style
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass

        base_fonts = ['TkDefaultFont', 'TkTextFont', 'TkMenuFont']
        for font_name in base_fonts:
            try:
                tkfont.nametofont(font_name).configure(family=DEFAULT_FONT_FAMILY, size=DEFAULT_FONT_SIZE)
            except tk.TclError:
                pass
        try:
            tkfont.nametofont('TkHeadingFont').configure(family=DEFAULT_FONT_FAMILY, size=HEADING_FONT_SIZE, weight='bold')
        except tk.TclError:
            pass

        style.configure('TFrame', background=PALETTE['background'])
        style.configure('Main.TFrame', background=PALETTE['background'])
        style.configure('Card.TLabelframe', background=PALETTE['surface'], borderwidth=1, relief='solid')
        style.configure('Card.TLabelframe.Label', background=PALETTE['surface'], foreground=PALETTE['text'], font=(DEFAULT_FONT_FAMILY, HEADING_FONT_SIZE, 'bold'))
        style.configure('Section.TLabelframe', background=PALETTE['card'], borderwidth=0)
        style.configure('Section.TLabelframe.Label', background=PALETTE['card'], foreground=PALETTE['text'], font=(DEFAULT_FONT_FAMILY, HEADING_FONT_SIZE, 'bold'))
        style.configure('TLabelFrame', background=PALETTE['surface'], foreground=PALETTE['text'])
        style.configure('TLabelframe.Label', background=PALETTE['surface'], foreground=PALETTE['text'])
        style.configure('TLabel', background=PALETTE['background'], foreground=PALETTE['text'])
        style.configure('Info.TLabel', foreground=PALETTE['muted_text'])

        style.configure('TNotebook', background=PALETTE['background'], borderwidth=0)
        style.configure('TNotebook.Tab', background=PALETTE['surface'], foreground=PALETTE['muted_text'], padding=(16, 8), borderwidth=0)
        style.map('TNotebook.Tab', background=[('selected', PALETTE['accent'])], foreground=[('selected', PALETTE['background'])])

        style.configure('TButton', background=PALETTE['surface'], foreground=PALETTE['text'], padding=(12, 6), borderwidth=0)
        style.map('TButton', background=[('active', PALETTE['card'])], relief=[('pressed', 'sunken')])
        style.configure('Accent.TButton', background=PALETTE['accent'], foreground=PALETTE['background'], padding=(14, 6), borderwidth=0)
        style.map('Accent.TButton', background=[('active', PALETTE['accent_hover'])], foreground=[('disabled', PALETTE['muted_text'])])

        style.configure('Danger.TButton', background=PALETTE['error'], foreground=PALETTE['background'], padding=(14, 6), borderwidth=0)
        style.map('Danger.TButton', background=[('active', '#fb7185')], foreground=[('disabled', PALETTE['muted_text'])])

        style.configure('Treeview', background=PALETTE['surface'], foreground=PALETTE['text'], fieldbackground=PALETTE['surface'])
        style.configure('Treeview.Heading', background=PALETTE['card'], foreground=PALETTE['text'], font=(DEFAULT_FONT_FAMILY, HEADING_FONT_SIZE, 'bold'))
        style.map('Treeview', background=[('selected', PALETTE['accent'])], foreground=[('selected', PALETTE['background'])])

        style.configure('Horizontal.TProgressbar', background=PALETTE['accent'], troughcolor=PALETTE['card'])
        style.configure('TCombobox', fieldbackground=PALETTE['surface'], background=PALETTE['surface'], foreground=PALETTE['text'])

        self.option_add('*TCombobox*Listbox.background', PALETTE['surface'])
        self.option_add('*TCombobox*Listbox.foreground', PALETTE['text'])
        self.option_add('*TCombobox*Listbox.selectBackground', PALETTE['accent'])
        self.option_add('*TCombobox*Listbox.selectForeground', PALETTE['background'])

    def _create_log_viewer(self):
        """Create the Activity Log tab."""
        # Main log frame
        log_frame = ttk.LabelFrame(self.log_tab, text="Console Output", padding=10, style="Card.TLabelframe")
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Info label
        info_frame = ttk.Frame(log_frame, style="Main.TFrame")
        info_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(
            info_frame,
            text="All application messages, scan progress, and safety alerts appear here",
            style="Info.TLabel"
        ).pack(side="left")

        # Clear button
        ttk.Button(
            info_frame,
            text="Clear Log",
            command=self._clear_log,
            style="TButton"
        ).pack(side="right", padx=5)

        # Text widget with scrollbar
        text_frame = ttk.Frame(log_frame, style="Main.TFrame")
        text_frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(
            text_frame,
            state="disabled",
            wrap="word",
            bg=PALETTE['card'],
            fg=PALETTE['text'],
            insertbackground=PALETTE['text'],
            relief='flat',
            highlightthickness=0,
            font=('Consolas', 9)
        )
        scrollbar = ttk.Scrollbar(text_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=scrollbar.set)

        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Configure text tags for colored output
        self.log_text.tag_config("INFO", foreground=PALETTE['text'])
        self.log_text.tag_config("SUCCESS", foreground=PALETTE['success'])
        self.log_text.tag_config("WARNING", foreground=PALETTE['warning'])
        self.log_text.tag_config("ERROR", foreground=PALETTE['error'])

        # Redirect stdout and stderr
        sys.stdout = TextRedirector(self.log_text, "INFO")
        sys.stderr = TextRedirector(self.log_text, "ERROR")

    def _clear_log(self):
        """Clear the activity log."""
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state="disabled")
        print("[INFO] Log cleared")

    def _create_menu(self):
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        auth_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Authentication", menu=auth_menu)
        auth_menu.add_command(label="Kerberos Login", command=self._kerberos_login)

    def _kerberos_login(self):
        """Open the Kerberos dialog and refresh auth banner upon completion."""
        dialog = KerberosLoginDialog(self, status_callback=self._handle_auth_status_change)
        self.wait_window(dialog)

    def _renew_kerberos_ticket(self) -> bool:
        """Attempt to silently renew the Kerberos ticket using kinit -R.

        Returns True if renewal succeeded, False otherwise.
        """
        return KerberosManager().renew_ticket()

    def _sync_dpm_node(self):
        """Push the UI DPM node selection to the scanner."""
        node = self.dpm_node.get().strip() or None
        self.scanner.dpm_node = node

    def _discover_dpms(self):
        """Discover available DPM nodes and populate the combo box."""
        self._sync_dpm_node()
        print("[INFO] Discovering DPM nodes...")
        threading.Thread(target=self._discover_dpms_thread, daemon=True).start()

    def _discover_dpms_thread(self):
        nodes = self.scanner.discover_dpms()
        if nodes:
            print(f"[SUCCESS] Found DPM nodes: {', '.join(nodes)}")
            self.after(0, self._update_dpm_combo, nodes)
        else:
            print("[WARNING] No DPM nodes found.")

    def _update_dpm_combo(self, nodes):
        self.dpm_combo['values'] = [''] + nodes

    def _acnet_call_with_renewal(self, fn, *args, _renewed=None, **kwargs):
        """Call an ACNET function, attempting credential renewal once on expiry.

        Args:
            fn: The scanner method to call.
            *args, **kwargs: Passed to fn.
            _renewed: Mutable list acting as a one-shot flag across calls
                      within a scan. Pass the same list to share the flag.
                      If None, renewal is always attempted.
        Returns:
            The return value of fn(*args, **kwargs).
        """
        try:
            return fn(*args, **kwargs)
        except CredentialExpiredError:
            if _renewed is None or not _renewed[0]:
                if self._renew_kerberos_ticket():
                    if _renewed is not None:
                        _renewed[0] = True
                    return fn(*args, **kwargs)
            raise

    def _create_settings_widgets(self):
        """Widgets for the Device Settings tab."""
        status_banner = ttk.Frame(self.settings_tab, style="Main.TFrame")
        status_banner.pack(side="top", fill="x", padx=10, pady=(5, 0))

        ttk.Label(status_banner, text="Authentication Status:", font=("Arial", 10, "bold"), foreground=PALETTE['text']).pack(side="left")
        self.auth_status_label = ttk.Label(status_banner, text="Not Authenticated", foreground=PALETTE['error'])
        self.auth_status_label.pack(side="left", padx=6)
        ttk.Button(status_banner, text="Kerberos Login", command=self._kerberos_login, style="Accent.TButton").pack(side="right", padx=5)
        ttk.Separator(self.settings_tab, orient="horizontal").pack(fill="x", padx=10, pady=6)

        selection_frame = ttk.LabelFrame(self.settings_tab, text="Device Selection", padding=12, style="Card.TLabelframe")
        selection_frame.pack(side="top", fill="x", padx=10, pady=5)

        manual_entry_frame = ttk.LabelFrame(selection_frame, text="Manual Device Entry", padding=8, style="Section.TLabelframe")
        manual_entry_frame.pack(fill='x', pady=6)

        ttk.Label(manual_entry_frame, text="Device Name:").pack(side="left", padx=(0, 5))
        self.manual_device_entry = ttk.Entry(manual_entry_frame, width=30)
        self.manual_device_entry.pack(side="left", padx=5)
        ttk.Button(manual_entry_frame, text="Add to Setting List", command=lambda: self.add_devices_to_list([self.manual_device_entry.get()], 'setting')).pack(side="left", padx=5)
        ttk.Button(manual_entry_frame, text="Add to Reading List", command=lambda: self.add_devices_to_list([self.manual_device_entry.get()], 'reading')).pack(side="left", padx=5)

        file_entry_frame = ttk.LabelFrame(selection_frame, text="Load Devices From File", padding=8, style="Section.TLabelframe")
        file_entry_frame.pack(fill='x', pady=5)
        ttk.Button(file_entry_frame, text="Open Device Selection Tool...",
                  command=self._open_device_selector,
                  style="Accent.TButton").pack(pady=5, fill='x', padx=10)

        status_frame = ttk.Frame(selection_frame, style="Main.TFrame")
        status_frame.pack(fill='x', pady=5)
        self.setting_file_label = ttk.Label(status_frame, text="0 devices in Setting List.", style="Info.TLabel")
        self.setting_file_label.pack(side="left", padx=5)
        ttk.Button(status_frame, text="Clear Setting List", command=lambda: self._clear_device_list('setting')).pack(side="left", padx=5)
        self.reading_file_label = ttk.Label(status_frame, text="0 devices in Reading List.", style="Info.TLabel")
        self.reading_file_label.pack(side="right", padx=5)
        ttk.Button(status_frame, text="Clear Reading List", command=lambda: self._clear_device_list('reading')).pack(side="right", padx=5)

        settings_grid_frame = ttk.LabelFrame(self.settings_tab, text="Current Setting Devices", padding=12, style="Card.TLabelframe")
        settings_grid_frame.pack(side="top", fill="both", expand=True, padx=10, pady=5)

        self.settings_canvas = tk.Canvas(settings_grid_frame, bg=PALETTE['surface'], highlightthickness=0)
        self.settings_scrollbar = ttk.Scrollbar(settings_grid_frame, orient="vertical", command=self.settings_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.settings_canvas, style="Main.TFrame")

        self.scrollable_frame.bind("<Configure>", lambda e: self.settings_canvas.configure(scrollregion=self.settings_canvas.bbox("all")))
        self.settings_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.settings_canvas.configure(yscrollcommand=self.settings_scrollbar.set)

        self.settings_canvas.pack(side="left", fill="both", expand=True)
        self.settings_scrollbar.pack(side="right", fill="y")

        ttk.Label(self.scrollable_frame, text="Device", font=('Helvetica', 10, 'bold')).grid(row=0, column=0, padx=5, pady=2, sticky='w')
        ttk.Label(self.scrollable_frame, text="New Value", font=('Helvetica', 10, 'bold')).grid(row=0, column=1, padx=5, pady=2, sticky='w')

        control_frame = ttk.Frame(self.settings_tab, padding=12, style="Main.TFrame")
        control_frame.pack(side="bottom", fill="x", padx=10, pady=5)

        roles = ["OPERATOR", "testing", "linac_trims", "linac_quads"]
        ttk.Label(control_frame, text="ACNET Role:").pack(side="left", padx=5)
        ttk.Combobox(control_frame, textvariable=self.setting_role, values=roles).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Fetch Nominals", command=self._fetch_nominals, style="Accent.TButton").pack(side="right", padx=5)
        ttk.Button(control_frame, text="Apply Settings", command=self._apply_settings_with_confirmation, style="Accent.TButton").pack(side="right", padx=5)

    def _refresh_auth_status(self):
        """Check Kerberos status and update the banner."""
        try:
            manager = KerberosManager()
            has_ticket = manager.check_existing_ticket()
        except Exception as exc:
            print(f"[WARNING] Unable to verify Kerberos ticket: {exc}")
            has_ticket = False
        self._update_auth_status_label(has_ticket)

    def _update_auth_status_label(self, is_authenticated, message=None):
        if not hasattr(self, 'auth_status_label') or self.auth_status_label is None:
            return
        if is_authenticated:
            self.auth_status_label.config(text="Authenticated", foreground=PALETTE['success'])
            self._auth_status_value = True
        else:
            self.auth_status_label.config(text="Not Authenticated", foreground=PALETTE['error'])
            self._auth_status_value = False
        if message:
            print(message)

    def _handle_auth_status_change(self, success, message=None):
        self._update_auth_status_label(success, message)

    def _create_scan_widgets(self):
        """Widgets for the Scan Settings tab."""
        main_frame = ttk.LabelFrame(self.scan_tab, text="Scan Parameters", padding=12, style="Card.TLabelframe")
        main_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Frame for global scan settings
        global_scan_frame = ttk.Frame(main_frame, style="Main.TFrame")
        global_scan_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(global_scan_frame, text="Points per Superperiod:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(global_scan_frame, textvariable=self.scan_points_per_superperiod, width=10).grid(row=0, column=1, sticky='w', padx=5, pady=2)
        
        ttk.Label(global_scan_frame, text="Number of Superperiods:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(global_scan_frame, textvariable=self.scan_num_superperiods, width=10).grid(row=1, column=1, sticky='w', padx=5, pady=2)

        # Frame for scan mode
        scan_mode_frame = ttk.LabelFrame(main_frame, text="Scan Mode", padding=8, style="Section.TLabelframe")
        scan_mode_frame.pack(fill='x', pady=5)
        ttk.Radiobutton(scan_mode_frame, text="Simultaneous", variable=self.scan_mode, value="Simultaneous").pack(side='left', padx=10)
        ttk.Radiobutton(scan_mode_frame, text="Sequential", variable=self.scan_mode, value="Sequential").pack(side='left', padx=10)
        # Opt-in persistent engine (Simultaneous only for now; Sequential keeps the
        # serial path until the persistent path is machine-validated end to end).
        ttk.Checkbutton(scan_mode_frame, text="Persistent (advance-on-event, ~6x — Simultaneous only)",
                        variable=self.use_persistent_scan).pack(side='left', padx=20)

        # Frame for save location
        save_loc_frame = ttk.LabelFrame(main_frame, text="Scan Data Save Location", padding=8, style="Section.TLabelframe")
        save_loc_frame.pack(fill='x', pady=5)
        
        save_path_entry = ttk.Entry(save_loc_frame, textvariable=self.scan_data_path, state='readonly')
        save_path_entry.pack(side='left', fill='x', expand=True, padx=5)
        ttk.Button(save_loc_frame, text="Browse...", command=self._select_scan_data_directory).pack(side='right', padx=5)

        # Treeview for per-device settings
        tree_frame = ttk.LabelFrame(main_frame, text="Per-Device Settings (Double-click to edit)", padding=8, style="Card.TLabelframe")
        tree_frame.pack(fill='both', expand=True, pady=10)
        cols = ('Device', 'Amplitude', 'Number of Periods')
        self.scan_tree = ttk.Treeview(tree_frame, columns=cols, show='headings')
        for col in cols:
            self.scan_tree.heading(col, text=col)
            self.scan_tree.column(col, width=180, anchor='center')
        self.scan_tree.pack(fill="both", expand=True)
        self.scan_tree.bind('<Double-1>', self._on_treeview_edit)

        # Quick-set toolbar for bulk amplitude/period assignment
        quickset_frame = ttk.Frame(tree_frame, style="Main.TFrame")
        quickset_frame.pack(fill='x', pady=(5, 0))

        ttk.Label(quickset_frame, text="Amplitude:").pack(side='left', padx=(0, 3))
        self._quickset_amplitude = tk.DoubleVar(value=1.0)
        ttk.Entry(quickset_frame, textvariable=self._quickset_amplitude, width=8).pack(side='left')
        ttk.Button(quickset_frame, text="Apply to All", command=self._apply_amplitude_to_all).pack(side='left', padx=(3, 15))

        ttk.Label(quickset_frame, text="Periods start:").pack(side='left', padx=(0, 3))
        self._quickset_periods_start = tk.IntVar(value=10)
        ttk.Entry(quickset_frame, textvariable=self._quickset_periods_start, width=6).pack(side='left')
        ttk.Button(quickset_frame, text="Apply Sequential", command=self._apply_sequential_periods).pack(side='left', padx=3)

        # Frame for scan controls
        scan_control_frame = ttk.Frame(self.scan_tab, padding=12, style="Main.TFrame")
        scan_control_frame.pack(fill="x")

        self.start_scan_button = ttk.Button(scan_control_frame, text="Start Scan", command=self._start_scan, style="Accent.TButton")
        self.start_scan_button.pack(side="left", padx=5)
        self.stop_scan_button = ttk.Button(scan_control_frame, text="Stop Scan", command=self._stop_scan, state="disabled", style="Danger.TButton")
        self.stop_scan_button.pack(side="left", padx=5)
        self.preview_scan_button = ttk.Button(scan_control_frame, text="Preview Waveform", command=self._preview_scan_waveform)
        self.preview_scan_button.pack(side="left", padx=5)

        config_button_frame = ttk.Frame(scan_control_frame, style="Main.TFrame")
        config_button_frame.pack(side="right", padx=5)
        ttk.Button(config_button_frame, text="Load Scan Setup", command=self._load_scan_setup, style="TButton").pack(side="right", padx=5)
        ttk.Button(config_button_frame, text="Save Scan Setup", command=self._save_scan_setup, style="TButton").pack(side="right")
        
        self.scan_progressbar = ttk.Progressbar(self.scan_tab, orient='horizontal', mode='determinate', style="Horizontal.TProgressbar")
        self.scan_progressbar.pack(fill='x', padx=10, pady=(5, 10))


    def _create_rms_widgets(self):
        """Widgets for the Baseline RMS tab."""
        main_frame = ttk.LabelFrame(self.rms_tab, text="Baseline Fluctuation (RMS)", padding=12, style="Card.TLabelframe")
        main_frame.pack(padx=10, pady=10, fill="both", expand=True)

        control_frame = ttk.Frame(main_frame, style="Main.TFrame")
        control_frame.pack(fill='x', pady=(0, 10))
        ttk.Label(control_frame, text="Number of Samples:").pack(side='left', padx=5)
        ttk.Entry(control_frame, textvariable=self.rms_samples, width=10).pack(side='left', padx=5)
        self.rms_button = ttk.Button(control_frame, text="Measure Baseline RMS", command=self._start_rms_measurement, style="Accent.TButton")
        self.rms_button.pack(side='left', padx=10)
        self.rms_status_label = ttk.Label(control_frame, text="", style="Info.TLabel")
        self.rms_status_label.pack(side='left', padx=5)

        self.rms_progressbar = ttk.Progressbar(main_frame, orient='horizontal', mode='determinate', style="Horizontal.TProgressbar")
        self.rms_progressbar.pack(fill='x', pady=5, padx=5)

        cols = ('Device', 'RMS Fluctuation')
        self.rms_tree = ttk.Treeview(main_frame, columns=cols, show='headings')
        for col in cols:
            self.rms_tree.heading(col, text=col)
            self.rms_tree.column(col, width=200, anchor='center')
        self.rms_tree.pack(fill="both", expand=True, pady=(5,0))

        # Export button
        export_frame = ttk.Frame(main_frame, style="Main.TFrame")
        export_frame.pack(fill='x', pady=(10, 0))
        ttk.Button(export_frame, text="Save RMS Data to CSV", command=self._export_rms_data, style="Accent.TButton").pack(side='left')


    def _create_safety_widgets(self):
        """Widgets for the Safety Configuration tab."""
        # Main container
        main_frame = ttk.LabelFrame(self.safety_tab, text="Safety Monitoring Configuration", padding=12, style="Card.TLabelframe")
        main_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Safety Status Banner
        status_frame = ttk.Frame(main_frame, style="Main.TFrame")
        status_frame.pack(fill='x', pady=(0, 15))

        self.safety_status_label = tk.Label(
            status_frame,
            text="⚠ SAFETY: NOT CONFIGURED",
            font=('Arial', 11, 'bold'),
            bg=SAFETY_DISABLED_COLOR,
            fg=PALETTE['background'],
            padx=15,
            pady=8
        )
        self.safety_status_label.pack(fill='x')

        # Enable/Disable Safety
        enable_frame = ttk.LabelFrame(main_frame, text="Safety System Control", padding=10, style="Section.TLabelframe")
        enable_frame.pack(fill='x', pady=(0, 10))

        ttk.Checkbutton(
            enable_frame,
            text="Enable Safety Monitoring (stops scan if thresholds exceeded)",
            variable=self.safety_enabled,
            command=self._toggle_safety
        ).pack(anchor='w', pady=5)

        # Baseline Measurement Section
        baseline_frame = ttk.LabelFrame(main_frame, text="1. Measure Baseline RMS", padding=10, style="Section.TLabelframe")
        baseline_frame.pack(fill='x', pady=(0, 10))

        baseline_info = ttk.Label(
            baseline_frame,
            text="First, measure baseline RMS values for all reading devices (BPMs). This establishes the nominal noise floor.",
            style="Info.TLabel",
            wraplength=800
        )
        baseline_info.pack(anchor='w', pady=(0, 8))

        baseline_control = ttk.Frame(baseline_frame, style="Main.TFrame")
        baseline_control.pack(fill='x')

        ttk.Label(baseline_control, text="Number of Samples:").pack(side='left', padx=(0, 5))
        ttk.Entry(baseline_control, textvariable=self.safety_baseline_samples, width=10).pack(side='left', padx=5)

        self.safety_measure_button = ttk.Button(
            baseline_control,
            text="Measure Safety Baseline",
            command=self._measure_safety_baseline,
            style="Accent.TButton"
        )
        self.safety_measure_button.pack(side='left', padx=10)

        self.safety_baseline_status = ttk.Label(baseline_control, text="Status: Not measured", style="Info.TLabel")
        self.safety_baseline_status.pack(side='left', padx=10)

        # Baseline data display
        baseline_data_frame = ttk.Frame(baseline_frame, style="Main.TFrame")
        baseline_data_frame.pack(fill='both', expand=True, pady=(10, 0))

        cols = ('Device', 'Mean', 'RMS', 'Std Dev', 'Min', 'Max', 'Samples')
        self.safety_baseline_tree = ttk.Treeview(baseline_data_frame, columns=cols, show='headings', height=6)
        for col in cols:
            self.safety_baseline_tree.heading(col, text=col)
            width = 150 if col == 'Device' else 100
            self.safety_baseline_tree.column(col, width=width, anchor='center')

        scrollbar = ttk.Scrollbar(baseline_data_frame, command=self.safety_baseline_tree.yview)
        self.safety_baseline_tree.config(yscrollcommand=scrollbar.set)

        self.safety_baseline_tree.pack(side='left', fill="both", expand=True)
        scrollbar.pack(side='right', fill='y')

        # Threshold Configuration Section
        threshold_frame = ttk.LabelFrame(main_frame, text="2. Configure Thresholds", padding=10, style="Section.TLabelframe")
        threshold_frame.pack(fill='x', pady=(0, 10))

        threshold_info = ttk.Label(
            threshold_frame,
            text="Set thresholds as sigma (standard deviation) units for mean position shift from baseline. WARNING threshold logs a message, ABORT threshold stops the scan.",
            style="Info.TLabel",
            wraplength=800
        )
        threshold_info.pack(anchor='w', pady=(0, 8))

        # Buffer size configuration
        buffer_frame = ttk.Frame(threshold_frame, style="Main.TFrame")
        buffer_frame.pack(fill='x', pady=(0, 10))

        ttk.Label(buffer_frame, text="Averaging Window:", font=('Arial', 10, 'bold')).pack(side='left', padx=(0, 10))
        ttk.Label(buffer_frame, text="Number of samples to average:").pack(side='left', padx=(0, 5))
        self.safety_buffer_size = tk.IntVar(value=self.safety_config.buffer_size)
        ttk.Entry(buffer_frame, textvariable=self.safety_buffer_size, width=10).pack(side='left', padx=5)
        ttk.Label(buffer_frame, text="(1 = check each reading individually, >1 = average over N samples)", style="Info.TLabel").pack(side='left', padx=10)

        # Threshold type selection
        type_frame = ttk.Frame(threshold_frame, style="Main.TFrame")
        type_frame.pack(fill='x', pady=(0, 10))

        ttk.Label(type_frame, text="Monitoring Mode:", font=('Arial', 10, 'bold')).pack(side='left', padx=(0, 10))
        ttk.Radiobutton(
            type_frame,
            text="Per-Device Mean (monitor each BPM position shift individually)",
            variable=self.safety_threshold_type,
            value="per_device",
            command=self._update_safety_config
        ).pack(side='left', padx=10)
        ttk.Radiobutton(
            type_frame,
            text="Overall Mean (monitor combined mean position across all BPMs)",
            variable=self.safety_threshold_type,
            value="overall",
            command=self._update_safety_config
        ).pack(side='left', padx=10)

        # Per-device thresholds
        per_device_frame = ttk.Frame(threshold_frame, style="Main.TFrame")
        per_device_frame.pack(fill='x', pady=5)

        ttk.Label(per_device_frame, text="Per-Device Thresholds:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky='w', pady=(0, 5), columnspan=4)

        ttk.Label(per_device_frame, text="  Warning Threshold (σ):").grid(row=1, column=0, sticky='w', padx=(20, 5))
        ttk.Entry(per_device_frame, textvariable=self.safety_per_device_warning, width=10).grid(row=1, column=1, padx=5)
        ttk.Label(per_device_frame, text="sigma (std dev) units", style="Info.TLabel").grid(row=1, column=2, sticky='w', padx=5)

        ttk.Label(per_device_frame, text="  Abort Threshold (σ):").grid(row=2, column=0, sticky='w', padx=(20, 5), pady=(5, 0))
        ttk.Entry(per_device_frame, textvariable=self.safety_per_device_abort, width=10).grid(row=2, column=1, padx=5, pady=(5, 0))
        ttk.Label(per_device_frame, text="sigma (std dev) units", style="Info.TLabel").grid(row=2, column=2, sticky='w', padx=5, pady=(5, 0))

        # Overall thresholds
        overall_frame = ttk.Frame(threshold_frame, style="Main.TFrame")
        overall_frame.pack(fill='x', pady=5)

        ttk.Label(overall_frame, text="Overall Mean Thresholds:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky='w', pady=(5, 5), columnspan=4)

        ttk.Label(overall_frame, text="  Warning Threshold (σ):").grid(row=1, column=0, sticky='w', padx=(20, 5))
        ttk.Entry(overall_frame, textvariable=self.safety_overall_warning, width=10).grid(row=1, column=1, padx=5)
        ttk.Label(overall_frame, text="sigma (std dev) units", style="Info.TLabel").grid(row=1, column=2, sticky='w', padx=5)

        ttk.Label(overall_frame, text="  Abort Threshold (σ):").grid(row=2, column=0, sticky='w', padx=(20, 5), pady=(5, 0))
        ttk.Entry(overall_frame, textvariable=self.safety_overall_abort, width=10).grid(row=2, column=1, padx=5, pady=(5, 0))
        ttk.Label(overall_frame, text="sigma (std dev) units", style="Info.TLabel").grid(row=2, column=2, sticky='w', padx=5, pady=(5, 0))

        # Apply button
        apply_frame = ttk.Frame(threshold_frame, style="Main.TFrame")
        apply_frame.pack(fill='x', pady=(10, 0))
        ttk.Button(
            apply_frame,
            text="Apply Threshold Settings",
            command=self._apply_safety_thresholds,
            style="Accent.TButton"
        ).pack(side='left')

        # Configuration Management
        config_frame = ttk.LabelFrame(main_frame, text="3. Save/Load Configuration", padding=10, style="Section.TLabelframe")
        config_frame.pack(fill='x', pady=(0, 10))

        button_frame = ttk.Frame(config_frame, style="Main.TFrame")
        button_frame.pack(fill='x')

        ttk.Button(button_frame, text="Save Safety Config", command=self._save_safety_config, style="Accent.TButton").pack(side='left', padx=5)
        ttk.Button(button_frame, text="Load Safety Config", command=self._load_safety_config, style="Accent.TButton").pack(side='left', padx=5)
        ttk.Button(button_frame, text="Clear All Baselines", command=self._clear_safety_baselines, style="Danger.TButton").pack(side='left', padx=15)

        # === Beam Loss Interlock ===
        interlock_group = ttk.LabelFrame(main_frame, text="4. Beam Loss Interlock", padding=10, style="Section.TLabelframe")
        interlock_group.pack(fill='x', pady=(0, 10))

        interlock_info = ttk.Label(
            interlock_group,
            text="Monitors radiation loss devices during scan. Automatically disables beam via L:BSTUDY if any threshold is exceeded.",
            style="Info.TLabel",
            wraplength=800
        )
        interlock_info.pack(anchor='w', pady=(0, 8))

        self.interlock_enabled_check = ttk.Checkbutton(
            interlock_group,
            text="Enable Beam Loss Interlock",
            variable=None,  # placeholder, set below
            command=self._toggle_interlock_settings
        )
        self.beam_interlock_enabled_var = tk.BooleanVar(value=BEAM_INTERLOCK_ENABLED_BY_DEFAULT)
        self.interlock_enabled_check.config(variable=self.beam_interlock_enabled_var)
        self.interlock_enabled_check.pack(anchor='w', pady=(0, 5))

        self._interlock_settings_frame = ttk.Frame(interlock_group, style="Main.TFrame")
        self._interlock_settings_frame.pack(fill='x')

        # Loss monitor table
        table_frame = ttk.Frame(self._interlock_settings_frame, style="Main.TFrame")
        table_frame.pack(fill='x', pady=(0, 5))

        ttk.Label(table_frame, text="Loss Monitor Thresholds:", font=(DEFAULT_FONT_FAMILY, DEFAULT_FONT_SIZE, 'bold')).pack(anchor='w', pady=(0, 5))

        lm_cols = ('Device', 'Max Threshold')
        self.loss_monitor_tree = ttk.Treeview(table_frame, columns=lm_cols, show='headings', height=4)
        self.loss_monitor_tree.heading('Device', text='Device')
        self.loss_monitor_tree.heading('Max Threshold', text='Max Threshold')
        self.loss_monitor_tree.column('Device', width=200, anchor='center')
        self.loss_monitor_tree.column('Max Threshold', width=150, anchor='center')
        self.loss_monitor_tree.pack(side='left', fill='x', expand=True)

        lm_scroll = ttk.Scrollbar(table_frame, command=self.loss_monitor_tree.yview)
        self.loss_monitor_tree.config(yscrollcommand=lm_scroll.set)
        lm_scroll.pack(side='right', fill='y')

        # Populate default loss monitors
        for dev, thresh in DEFAULT_LOSS_MONITORS.items():
            self.loss_monitor_tree.insert('', 'end', values=(dev, f"{thresh:.2f}"))

        # Add/Remove buttons and beam role/event
        interlock_btn_frame = ttk.Frame(self._interlock_settings_frame, style="Main.TFrame")
        interlock_btn_frame.pack(fill='x', pady=(5, 5))

        ttk.Button(interlock_btn_frame, text="Add Monitor", command=self._add_loss_monitor).pack(side='left', padx=(0, 5))
        ttk.Button(interlock_btn_frame, text="Remove Selected", command=self._remove_loss_monitor).pack(side='left', padx=5)

        ttk.Label(interlock_btn_frame, text="Beam Role:").pack(side='left', padx=(15, 5))
        self.beam_role_entry = ttk.Entry(interlock_btn_frame, width=18)
        self.beam_role_entry.pack(side='left', padx=(0, 10))

        ttk.Label(interlock_btn_frame, text="Beam Event:").pack(side='left', padx=(5, 5))
        self.beam_event_entry = ttk.Entry(interlock_btn_frame, width=10)
        self.beam_event_entry.insert(0, "@e,0A")
        self.beam_event_entry.pack(side='left')

        # Beam control buttons and status indicator
        beam_control_frame = ttk.Frame(self._interlock_settings_frame, style="Main.TFrame")
        beam_control_frame.pack(fill='x', pady=(5, 5))

        self.beam_status_label = tk.Label(
            beam_control_frame,
            text="BEAM: --",
            font=(DEFAULT_FONT_FAMILY, 10, 'bold'),
            bg=PALETTE['border'],
            fg=PALETTE['text'],
            padx=12, pady=4
        )
        self.beam_status_label.pack(side='left', padx=(0, 10))

        # Keep references so these can be disabled while a scan is running
        # (#4/#5) — ACNET button handlers must not be invoked mid-scan.
        self.beam_enable_button = ttk.Button(beam_control_frame, text="Beam ON", command=self._on_beam_enable, style="Accent.TButton")
        self.beam_enable_button.pack(side='left', padx=5)
        self.beam_disable_button = ttk.Button(beam_control_frame, text="Beam OFF", command=self._on_beam_disable)
        self.beam_disable_button.pack(side='left', padx=5)
        self.beam_refresh_button = ttk.Button(beam_control_frame, text="Refresh Status", command=self._refresh_beam_status)
        self.beam_refresh_button.pack(side='left', padx=5)

        self.disable_beam_on_completion_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self._interlock_settings_frame,
            text="Disable beam automatically after scan completes",
            variable=self.disable_beam_on_completion_var
        ).pack(anchor='w', pady=(5, 0))

        # Initial state
        self._toggle_interlock_settings()

        # Update initial status
        self._update_safety_status_display()


    def _create_response_matrix_widgets(self):
        """Widgets for the Response Matrix tab."""
        main_frame = ttk.LabelFrame(self.response_tab, text="Response Matrix Analysis", padding=12, style="Card.TLabelframe")
        main_frame.pack(padx=10, pady=10, fill="both", expand=True)

        controls_frame = ttk.Frame(main_frame, style="Main.TFrame")
        controls_frame.pack(fill='x', pady=(0, 10))

        ttk.Button(controls_frame, text="Calculate Response Matrix", command=self._calculate_response_matrix, style="Accent.TButton").pack(side='left')
        ttk.Button(controls_frame, text="Save Matrix to CSV", command=self._save_response_matrix, style="Accent.TButton").pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Open Advanced Analysis", command=self._launch_analysis_app, style="Accent.TButton").pack(side='left', padx=5)
        ttk.Checkbutton(controls_frame, text="Auto-calculate ORM after scan",
                         variable=self.auto_calc_orm, style="TCheckbutton").pack(side='left', padx=(15, 5))

        mode_frame = ttk.Frame(controls_frame, style="Main.TFrame")
        mode_frame.pack(side='right')
        ttk.Label(mode_frame, text="Display:", style="Info.TLabel").pack(side='left', padx=(0, 4))
        self.response_display_mode = tk.StringVar(value="Gain (Real)")
        mode_selector = ttk.Combobox(mode_frame, textvariable=self.response_display_mode,
                                     values=("Gain (Real)", "Uncertainty (sigma)", "Device Noise"), state="readonly", width=18)
        mode_selector.pack(side='left')
        mode_selector.bind("<<ComboboxSelected>>", lambda event: self._update_response_matrix_plot())

        self.response_fig = Figure(figsize=(10, 4.5), dpi=100, facecolor=PALETTE['card'])
        self.response_ax_h = self.response_fig.add_subplot(1, 2, 1)
        self.response_ax_v = self.response_fig.add_subplot(1, 2, 2)
        self._style_axis(self.response_ax_h)
        self._style_axis(self.response_ax_v)

        self.response_axes_positions = {
            'Horizontal': self.response_ax_h.get_position().frozen(),
            'Vertical': self.response_ax_v.get_position().frozen(),
        }

        self.response_colorbar_axes = {}
        self.response_colorbars = {}
        for plane, axis in (('Horizontal', self.response_ax_h), ('Vertical', self.response_ax_v)):
            divider = make_axes_locatable(axis)
            cax = divider.append_axes('right', size='5%', pad=0.08)
            cax.set_facecolor(PALETTE['card'])
            cax.tick_params(colors=PALETTE['muted_text'])
            for spine in cax.spines.values():
                spine.set_edgecolor(PALETTE['border'])
            cax.set_visible(False)
            self.response_colorbar_axes[plane] = cax
            self.response_colorbars[plane] = None

        self.response_heatmap_canvas = FigureCanvasTkAgg(self.response_fig, master=main_frame)
        self.response_heatmap_widget = self.response_heatmap_canvas.get_tk_widget()
        self.response_heatmap_widget.configure(bg=PALETTE['card'], highlightthickness=0)
        self.response_heatmap_widget.pack(fill="both", expand=True)

        self.response_toolbar = NavigationToolbar2Tk(self.response_heatmap_canvas, main_frame)
        self.response_toolbar.update()
        self._style_toolbar(self.response_toolbar)

        self.response_colorbar_h = None
        self.response_colorbar_v = None

        self._update_response_matrix_plot()


    def _on_treeview_edit(self, event):
        """Handle double-click to edit a cell in the scan treeview."""
        region = self.scan_tree.identify("region", event.x, event.y)
        if region != "cell":
            return

        column = self.scan_tree.identify_column(event.x)
        column_index = int(column.replace('#', '')) - 1 
        if column_index == 0: # Don't allow editing device name
            return

        item_id = self.scan_tree.focus()
        item_values = self.scan_tree.item(item_id, "values")
        
        x, y, width, height = self.scan_tree.bbox(item_id, column)

        entry_var = tk.StringVar(value=item_values[column_index])
        entry = ttk.Entry(self.scan_tree, textvariable=entry_var)
        entry.place(x=x, y=y, width=width, height=height)
        entry.focus_set()

        def save_edit(event):
            new_value = entry_var.get()
            current_values = list(self.scan_tree.item(item_id, "values"))
            current_values[column_index] = new_value
            self.scan_tree.item(item_id, values=current_values)
            entry.destroy()

        entry.bind('<Return>', save_edit)
        entry.bind('<FocusOut>', save_edit)

    def _populate_scan_tab(self):
        """Sync the scan parameter table with the current setting device list."""
        existing_configs: Dict[str, ScanDeviceConfig] = {}
        for item_id in self.scan_tree.get_children():
            values = self.scan_tree.item(item_id, "values")
            try:
                config = ScanDeviceConfig.from_tree_values(values)
                existing_configs[config.device] = config
            except ValueError:
                continue
        for item_id in list(self.scan_tree.get_children()):
            self.scan_tree.delete(item_id)
        for device in self.setting_devices:
            config = existing_configs.get(device, ScanDeviceConfig(device=device, amplitude=1.0, periods=1))
            self.scan_tree.insert('', 'end', values=config.to_tree_tuple())

    def _apply_amplitude_to_all(self):
        """Set the same amplitude for every device in the scan table."""
        children = self.scan_tree.get_children()
        if not children:
            print("[WARNING] No devices in scan table.")
            return
        try:
            amp = self._quickset_amplitude.get()
        except tk.TclError:
            print("[ERROR] Invalid amplitude value.")
            return
        if not math.isfinite(amp):
            print("[ERROR] Amplitude must be a finite number.")
            return
        for item_id in children:
            values = list(self.scan_tree.item(item_id, "values"))
            values[1] = amp
            self.scan_tree.item(item_id, values=values)
        print(f"[INFO] Amplitude {amp} applied to all {len(children)} devices.")

    def _apply_sequential_periods(self):
        """Assign monotonically increasing periods starting from the given value."""
        children = self.scan_tree.get_children()
        if not children:
            print("[WARNING] No devices in scan table.")
            return
        try:
            start = self._quickset_periods_start.get()
        except tk.TclError:
            print("[ERROR] Invalid starting period value.")
            return
        if start < 0:
            print("[ERROR] Starting period must be non-negative.")
            return
        for i, item_id in enumerate(children):
            values = list(self.scan_tree.item(item_id, "values"))
            values[2] = start + i
            self.scan_tree.item(item_id, values=values)
        print(f"[INFO] Periods {start}..{start + len(children) - 1} applied to {len(children)} devices.")

    def _collect_scan_configs(self) -> List[ScanDeviceConfig]:
        """Return the current scan table as validated configuration objects."""
        configs: List[ScanDeviceConfig] = []
        seen = set()
        for item_id in self.scan_tree.get_children():
            values = self.scan_tree.item(item_id, "values")
            config = ScanDeviceConfig.from_tree_values(values)
            if config.device in seen:
                raise ValueError(f"Duplicate scan entry found for {config.device}.")
            seen.add(config.device)
            configs.append(config)
        return configs

    def _build_scan_run_config(self) -> ScanRunConfig:
        """Compose a ScanRunConfig from UI state, raising ValueError on issues."""
        configs = self._collect_scan_configs()
        try:
            points_per_superperiod = int(self.scan_points_per_superperiod.get())
            superperiods = int(self.scan_num_superperiods.get())
        except tk.TclError as exc:
            raise ValueError(f"Invalid global scan parameter: {exc}") from exc
        run_config = ScanRunConfig(
            devices=configs,
            points_per_superperiod=points_per_superperiod,
            superperiods=superperiods,
            role=self.setting_role.get(),
        )
        run_config.validate()
        return run_config

    def _build_readback_drf_list(self, acnet_event=None):
        """Return the DRF list used when collecting scan data.

        Note: Uses actual readbacks for ALL devices (including setting devices/correctors)
        rather than .SETTING properties. This ensures the ORM calculation uses the actual
        corrector output rather than commanded setpoints, accounting for any discrepancies
        due to power supply limitations, saturation, or calibration errors.

        Setting values (.SETTING) are recorded as the analytically COMMANDED
        setpoint (ScanDeviceConfig.compute_value) and merged into the scan data
        with a .SETTING column suffix (Phase 1 — no per-step readback round-trip).
        They are excluded from the ORM, which uses the readbacks above.

        Args:
            acnet_event: Pre-captured ACNET event string. If None, reads from tkinter
                         (only safe from main thread).
        """
        if acnet_event is None:
            acnet_event = self.acnet_event.get()
        # Drop dead BPMs (all-zero at baseline) so they never enter scan data.
        all_devices = sorted(
            set(self.reading_devices + self.setting_devices) - self.dead_bpms)
        # Use readbacks for all devices (not .SETTING for correctors)
        drf_list = [f"{dev}{acnet_event}" for dev in all_devices]
        return drf_list, all_devices

    # =========================================================================
    # Device Health (per-device missed / non-finite / rejected-SET accounting)
    # =========================================================================
    #
    # self.device_health is a plain dict, keyed by device name, accumulated by
    # the scan worker thread. Each entry is:
    #   {
    #     'missed':       int,            # reading absent (None/timeout) this many times
    #     'nonfinite':    int,            # reading was NaN/inf this many times
    #     'set_rejected': int,            # a corrector SET to this device was rejected
    #     'last_seen':    datetime|None,  # last time a FINITE reading was received
    #   }
    # Individual-device problems are recorded here and NEVER abort the scan; the
    # UI pass renders this structure. Only TOTAL loss (all-None for 5 consecutive
    # steps) or a loss-monitor / safety abort stops the scan.

    def _device_health_entry(self, device: str) -> dict:
        """Return (creating if needed) the device_health record for `device`."""
        entry = self.device_health.get(device)
        if entry is None:
            entry = {'missed': 0, 'nonfinite': 0, 'set_rejected': 0,
                     'last_seen': None}
            self.device_health[device] = entry
        return entry

    def _record_device_problem(self, device: str, kind: str):
        """Increment a per-device problem counter ('missed'/'nonfinite'/
        'set_rejected'). Worker-thread only; never raises, never aborts."""
        try:
            entry = self._device_health_entry(device)
            if kind in entry:
                entry[kind] += 1
        except Exception as e:
            print(f"[WARNING] Could not record device health for {device}: {e}")

    def _record_device_seen(self, device: str, stamp=None):
        """Mark `device` as having delivered a finite reading (updates last_seen)."""
        try:
            entry = self._device_health_entry(device)
            entry['last_seen'] = stamp if stamp is not None else datetime.now()
        except Exception:
            pass

    @staticmethod
    def _meta_path_for(csv_path) -> str:
        """Derive the `_meta.json` sidecar path from a CSV path.

        Replaces ONLY the final extension (via os.path.splitext) so a path that
        contains the substring '.csv' more than once — e.g. a directory named
        'foo.csv_runs/' — is handled correctly. The old
        str.replace('.csv', '_meta.json') replaced EVERY occurrence.
        """
        base, _ext = os.path.splitext(os.fspath(csv_path))
        return base + "_meta.json"

    def _account_bpm_step_health(self, data_from_step, expected_devices,
                                 extra_nonfinite=None):
        """Per-device exclude-and-report accounting for one scan step.

        Records to self.device_health (does NOT abort):
          * devices whose slot is None/missing this step  -> 'missed'
          * devices whose reading value is NaN/inf         -> 'nonfinite'
          * devices with a finite reading                  -> updates 'last_seen'

        This is self-contained (detects non-finite values directly) so it works
        even when safety monitoring is disabled. The safety monitor's own
        non-finite flagging is recorded SEPARATELY by the caller via nonfinite_out.

        Args:
            data_from_step: BPM/scan slice of the read (list of dicts/None).
            expected_devices: device names that should have been read this step.
            extra_nonfinite: optional iterable of additional device names to mark
                non-finite (kept for symmetry; normally None).
        """
        extra_set = set(extra_nonfinite or [])
        seen_names = set()
        for item in data_from_step:
            if isinstance(item, dict):
                name = item.get('name')
                if not name:
                    continue
                seen_names.add(name)
                value = item.get('data')
                is_finite = False
                try:
                    is_finite = math.isfinite(float(value))
                except (TypeError, ValueError):
                    is_finite = False
                if is_finite and name not in extra_set:
                    self._record_device_seen(name, item.get('stamp'))
                else:
                    # Present but unusable (NaN/inf/non-scalar): reported,
                    # excluded from analysis, never aborting.
                    self._record_device_problem(name, 'nonfinite')
        # Any expected device with no slot/None this step is "missed".
        for dev in expected_devices:
            if dev not in seen_names:
                self._record_device_problem(dev, 'missed')

    def _record_rejected_correctors(self, devices):
        """Record each corrector whose SET was rejected (apply_settings_once
        returned False). Does NOT abort — the scan continues."""
        for dev in devices:
            self._record_device_problem(dev, 'set_rejected')
        print(f"[WARNING] Corrector SET reported NOT confirmed for: "
              f"{', '.join(devices)} — recorded to Device Health, continuing scan")

    # =========================================================================
    # Device Health tab (#4)
    # =========================================================================

    def _create_device_health_widgets(self):
        """Build the Device Health tab.

        Renders self.device_health (populated by the scan worker) as a table:
        device, role (BPM/corrector/loss), missed count, non-finite count, and a
        status summary (excluded/dead/set-rejected/...) plus last-seen. The table
        is (re)populated on the main thread when a scan completes.
        """
        main_frame = ttk.LabelFrame(
            self.device_health_tab, text="Per-Device Scan Health",
            padding=12, style="Card.TLabelframe")
        main_frame.pack(padx=10, pady=10, fill="both", expand=True)

        ttk.Label(
            main_frame,
            text=("Per-device accounting from the most recent scan. Individual "
                  "device problems are reported here and never abort the scan. "
                  "Absence from this table means no problems were recorded."),
            style="Info.TLabel",
            wraplength=820,
        ).pack(anchor='w', pady=(0, 8))

        self.device_health_summary_label = ttk.Label(
            main_frame, text="No scan has completed yet.", style="Info.TLabel")
        self.device_health_summary_label.pack(anchor='w', pady=(0, 8))

        table_frame = ttk.Frame(main_frame, style="Main.TFrame")
        table_frame.pack(fill="both", expand=True)

        dh_cols = ('Device', 'Role', 'Missed', 'Non-finite',
                   'Set Rejected', 'Status', 'Last Seen')
        self.device_health_tree = ttk.Treeview(
            table_frame, columns=dh_cols, show='headings')
        widths = {
            'Device': 160, 'Role': 90, 'Missed': 70, 'Non-finite': 80,
            'Set Rejected': 90, 'Status': 150, 'Last Seen': 150,
        }
        for col in dh_cols:
            self.device_health_tree.heading(col, text=col)
            anchor = 'w' if col in ('Device', 'Status', 'Last Seen') else 'center'
            self.device_health_tree.column(col, width=widths[col], anchor=anchor)
        self.device_health_tree.pack(side='left', fill='both', expand=True)

        dh_scroll = ttk.Scrollbar(table_frame, command=self.device_health_tree.yview)
        self.device_health_tree.config(yscrollcommand=dh_scroll.set)
        dh_scroll.pack(side='right', fill='y')

        btn_frame = ttk.Frame(main_frame, style="Main.TFrame")
        btn_frame.pack(fill='x', pady=(8, 0))
        ttk.Button(btn_frame, text="Refresh",
                   command=self._populate_device_health).pack(side='left')

    def _classify_device_role(self, device: str) -> str:
        """Best-effort role label for a device in the Device Health table.

        Loss-monitor and corrector lists take precedence over the reading list so
        a device that appears in more than one (rare) is labelled by its most
        specific role.
        """
        try:
            loss_devs = set(self._loss_monitor_devices_from_config(
                self.beam_interlock_config))
        except Exception:
            loss_devs = set()
        if device in loss_devs:
            return "loss"
        if device in set(self.setting_devices):
            return "corrector"
        if device in set(self.reading_devices):
            return "BPM"
        return "—"

    def _device_health_status_text(self, device: str, entry: dict) -> str:
        """Human-readable status summary for one Device Health row."""
        parts = []
        if device in getattr(self, 'dead_bpms', set()):
            parts.append("dead/excluded")
        if entry.get('set_rejected', 0):
            parts.append("set-rejected")
        if entry.get('nonfinite', 0):
            parts.append("non-finite excluded")
        if entry.get('missed', 0):
            parts.append("missed reads")
        if not parts:
            parts.append("OK")
        return ", ".join(parts)

    def _populate_device_health(self):
        """Render self.device_health into the Device Health tree (main thread).

        Best-effort: device_health is a plain dict written by the worker thread;
        call this only after the scan has completed (e.g. from _on_scan_complete).
        """
        tree = getattr(self, 'device_health_tree', None)
        if tree is None:
            return
        try:
            for item in tree.get_children():
                tree.delete(item)
        except tk.TclError:
            return

        # Snapshot to avoid surprises if the worker is still finishing up.
        health = dict(self.device_health or {})
        problem_count = 0
        for device in sorted(health.keys()):
            entry = health[device] or {}
            missed = entry.get('missed', 0)
            nonfinite = entry.get('nonfinite', 0)
            set_rejected = entry.get('set_rejected', 0)
            if missed or nonfinite or set_rejected or device in getattr(self, 'dead_bpms', set()):
                problem_count += 1
            last_seen = entry.get('last_seen')
            if isinstance(last_seen, datetime):
                last_seen_text = last_seen.strftime('%Y-%m-%d %H:%M:%S')
            elif last_seen is None:
                last_seen_text = "never"
            else:
                last_seen_text = str(last_seen)
            tree.insert('', 'end', values=(
                device,
                self._classify_device_role(device),
                missed,
                nonfinite,
                set_rejected,
                self._device_health_status_text(device, entry),
                last_seen_text,
            ))

        label = getattr(self, 'device_health_summary_label', None)
        if label is not None:
            if not health:
                label.config(text="No device problems recorded in the last scan.")
            else:
                label.config(
                    text=(f"{len(health)} device(s) tracked, "
                          f"{problem_count} with recorded problems."))

    # =========================================================================
    # Frozen-config loss-monitor helpers
    # =========================================================================

    def _loss_monitor_devices_from_config(self, config) -> list:
        """Enabled+positive-threshold loss-monitor device names from a config
        snapshot (mirrors BeamInterlockMonitor.loss_monitor_devices but reads an
        explicit config so the scan worker uses the frozen snapshot, not live)."""
        if config is None or not config.enabled:
            return []
        return [dev for dev, thresh in config.loss_monitors.items()
                if thresh > 0]

    def _restore_correctors_on_trip(self, restore_devices, restore_values, role):
        """Restore correctors to nominal on a loss-monitor trip, BEFORE tripping
        the beam (#1). Runs to completion with NO abort_check (safety-critical).
        Never raises — called from the live scan loop, mirrors the finally-block
        restore pattern (single credential-renewal retry). Records a rejected SET
        to Device Health but does not abort here (the trip is already aborting)."""
        if not (restore_devices and restore_values):
            return
        print("[WARNING] Loss-monitor trip — restoring correctors to nominal "
              "BEFORE disabling beam")
        try:
            ok = self.scanner.apply_settings_once(restore_devices, restore_values, role)
            if ok is False:
                self._record_rejected_correctors(restore_devices)
        except CredentialExpiredError:
            print("[WARNING] Credential expired during trip restore — attempting renewal...")
            if self._renew_kerberos_ticket():
                try:
                    self.scanner.apply_settings_once(restore_devices, restore_values, role)
                    print("[SUCCESS] Correctors restored after credential renewal.")
                except Exception as e2:
                    print(f"[ERROR] Failed to restore correctors after renewal: {e2}")
            else:
                print("[ERROR] Failed to restore correctors on trip — credential "
                      "renewal failed. MANUALLY VERIFY DEVICE SETTINGS!")
        except Exception as e:
            print(f"[ERROR] Failed to restore correctors on trip: {e}")
        # Read-only confirmation (never raises, no abort_check).
        self._verify_nominal_restore(restore_devices, restore_values)

    def _escalate_beam_trip_unconfirmed(self, trip_event):
        """Loss-monitor trip where beam could NOT be confirmed OFF.

        Mirrors the manual _on_beam_disable escalation: marks the beam status
        UNKNOWN and raises a loud modal so the operator disables the beam by
        hand. Marshalled onto the Tk main thread via after(0) with the closing
        guard (worker-thread safe)."""
        dev = getattr(trip_event, 'device', '?')
        val = getattr(trip_event, 'value', float('nan'))
        thr = getattr(trip_event, 'threshold', float('nan'))
        print("[ERROR] CRITICAL: loss-monitor trip but beam NOT confirmed OFF — "
              "operator escalation required")
        if self._closing:
            return
        self.after(0, self._set_beam_status, BeamStatus.UNKNOWN)

        def _show_modal():
            if self._closing:
                return
            messagebox.showerror(
                "Beam Trip NOT Confirmed",
                f"A loss monitor exceeded its threshold:\n\n"
                f"    {dev} = {val:.4f}  (threshold {thr:.4f})\n\n"
                "The automatic beam-disable command was issued, but the beam "
                "could NOT be confirmed OFF (the SET may have been rejected or "
                "the status could not be read back).\n\n"
                "DISABLE THE BEAM MANUALLY NOW and verify the beam state."
            )
        self.after(0, _show_modal)

    def _compute_scan_step_values(self, configs, nominal_map, step_index, points_per_superperiod, modulated_device=None):
        """Compute device settings for a given scan step."""
        values = []
        for config in configs:
            nominal = nominal_map.get(config.device, 0.0)
            if modulated_device is None or config.device == modulated_device:
                values.append(config.compute_value(nominal, step_index, points_per_superperiod))
            else:
                values.append(nominal)
        return values


    def _build_nominal_map(self, devices, nominal_values):

        """Normalize nominal settings into a device -> value mapping."""

        nominal_map = {}

        fallback_devices = []

        for device, value in zip(devices, nominal_values or []):

            try:

                nominal = float(value)

                if not math.isfinite(nominal):

                    raise ValueError

            except (TypeError, ValueError):

                nominal = 0.0

                fallback_devices.append(device)

            nominal_map[device] = nominal

        for device in devices:

            if device not in nominal_map:

                nominal_map[device] = 0.0

                fallback_devices.append(device)

        return nominal_map, fallback_devices




    def _style_axis(self, ax, title_color=PALETTE['text']):
        """Apply application theming to matplotlib axes."""
        ax.set_facecolor(PALETTE['surface'])
        for spine in ax.spines.values():
            spine.set_color(PALETTE['border'])
        ax.tick_params(colors=PALETTE['muted_text'])
        ax.xaxis.label.set_color(PALETTE['muted_text'])
        ax.yaxis.label.set_color(PALETTE['muted_text'])
        ax.title.set_color(title_color)
        ax.grid(color=PALETTE['border'], alpha=0.25)
        ax.figure.patch.set_facecolor(PALETTE['card'])

    @staticmethod
    def _style_toolbar(toolbar):
        """Apply dark theme styling to a NavigationToolbar2Tk."""
        toolbar.configure(background=PALETTE['surface'])
        for child in toolbar.winfo_children():
            try:
                child.configure(background=PALETTE['surface'])
            except tk.TclError:
                pass

    def _create_reading_widgets(self):
        """Widgets for the Data Reading and Plots tab."""
        control_frame = ttk.Frame(self.reading_tab, padding=12, style="Main.TFrame")
        control_frame.pack(fill='x', padx=10, pady=5)

        self.start_stop_button = ttk.Button(control_frame, text="Start Reading", command=self._toggle_reading, style="Accent.TButton")
        self.start_stop_button.pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="Save Live Data", command=self._save_acquired_data, style="Accent.TButton").pack(side='left', padx=5)

        plot_control_frame = ttk.Frame(control_frame)
        plot_control_frame.pack(side='left', padx=(10,0))
        ttk.Label(plot_control_frame, text="Device to Plot:").pack(side='left')
        self.plot_selector_combo = ttk.Combobox(plot_control_frame, textvariable=self.plotted_device, state="readonly", width=20)
        self.plot_selector_combo.pack(side='left', padx=5)
        self.plot_selector_combo.bind('<<ComboboxSelected>>', self._on_plot_device_change)

        ttk.Label(control_frame, text="ACNET Event:").pack(side='left', padx=(10, 0))
        event_combo = ttk.Combobox(control_frame, textvariable=self.acnet_event, values=['@p,1000', '@i', '@e,52'])
        event_combo.pack(side='left', padx=5)

        ttk.Label(control_frame, text="DPM:").pack(side='left', padx=(10, 0))
        self.dpm_combo = ttk.Combobox(control_frame, textvariable=self.dpm_node, width=10)
        self.dpm_combo.pack(side='left', padx=2)
        ttk.Button(control_frame, text="Discover", command=self._discover_dpms).pack(side='left', padx=2)

        self.reading_status_label = ttk.Label(control_frame, text="Status: Idle", style="Info.TLabel")
        self.reading_status_label.pack(side='right', padx=5)

        plot_frame = ttk.LabelFrame(self.reading_tab, text="Live Data Plot", padding=12, style="Card.TLabelframe")
        plot_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.fig = Figure(figsize=(5, 4), dpi=100, facecolor=PALETTE['card'])
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        plot_widget = self.plot_canvas.get_tk_widget()
        plot_widget.configure(bg=PALETTE['card'], highlightthickness=0, borderwidth=0)
        plot_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.plot_canvas.draw()
        
    def _create_stats_widgets(self):
        """Widgets for the Statistics tab."""
        main_frame = ttk.LabelFrame(self.stats_tab, text="Live Data Statistics", padding=12, style="Card.TLabelframe")
        main_frame.pack(padx=10, pady=10, fill="both", expand=True)

        cols = ('Device', 'Min', 'Max', 'Mean', 'Std Dev')
        self.stats_tree = ttk.Treeview(main_frame, columns=cols, show='headings')
        for col in cols:
            self.stats_tree.heading(col, text=col)
            self.stats_tree.column(col, width=150, anchor='center')
        self.stats_tree.pack(fill="both", expand=True, pady=(0, 5))
        
        summary_frame = ttk.Frame(main_frame, style="Main.TFrame")
        summary_frame.pack(fill='x', pady=5)
        self.overall_rms_label = ttk.Label(summary_frame, text="Overall RMS of Std Deviations: N/A", font=('Helvetica', 10, 'bold'))
        self.overall_rms_label.pack(side='left')

        button_frame = ttk.Frame(self.stats_tab, padding=(10, 0, 10, 10), style="Main.TFrame")
        button_frame.pack(fill='x')
        ttk.Button(button_frame, text="Export to CSV", command=self._export_stats, style="Accent.TButton").pack(side='left')

    def _create_fft_widgets(self):
        """Widgets for the FFT Analysis tab."""
        main_frame = ttk.Frame(self.fft_tab, padding=12, style="Main.TFrame")
        main_frame.pack(fill="both", expand=True)
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)

        # Control panel on the left
        control_panel = ttk.LabelFrame(main_frame, text="Controls", padding=12, style="Card.TLabelframe")
        control_panel.grid(row=1, column=0, sticky='ns', padx=(0, 10))

        ttk.Button(control_panel, text="Load Scan Data File", command=self._load_fft_data, style="Accent.TButton").pack(pady=5, fill='x')
        
        ttk.Label(control_panel, text="Select Devices to Plot:").pack(pady=(10,0))
        self.fft_device_listbox = tk.Listbox(control_panel, selectmode='extended')
        self.fft_device_listbox.pack(pady=5, fill='both', expand=True)

        ttk.Button(control_panel, text="Generate FFT Plot", command=self._plot_fft, style="Accent.TButton").pack(pady=5, fill='x')

        # Plotting canvas on the right
        plot_frame = ttk.LabelFrame(main_frame, text="FFT Spectrum", padding=12, style="Card.TLabelframe")
        plot_frame.grid(row=1, column=1, sticky='nsew')
        
        self.fft_fig = Figure(figsize=(6, 4), dpi=100, facecolor=PALETTE['card'])
        self.fft_canvas = FigureCanvasTkAgg(self.fft_fig, master=plot_frame)
        fft_widget = self.fft_canvas.get_tk_widget()
        fft_widget.configure(bg=PALETTE['card'], highlightthickness=0, borderwidth=0)
        fft_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fft_toolbar = NavigationToolbar2Tk(self.fft_canvas, plot_frame)
        self.fft_toolbar.update()
        self._style_toolbar(self.fft_toolbar)

    def add_devices_to_list(self, devices, list_type):
        """Adds a list of devices to the specified list, avoiding duplicates."""
        device_name = self.manual_device_entry.get().strip()
        if device_name and devices == [device_name]: # Clear manual entry if used
            self.manual_device_entry.delete(0, tk.END)

        target_list = self.setting_devices if list_type == 'setting' else self.reading_devices
        for dev in devices:
            if dev and dev not in target_list:
                target_list.append(dev)

        if list_type == 'setting':
            self.setting_file_label.config(text=f"{len(self.setting_devices)} devices in Setting List.")
            self._update_settings_display()
            self._populate_scan_tab() # Update scan tab as well
        else:
            self.reading_file_label.config(text=f"{len(self.reading_devices)} devices in Reading List.")
            self._update_plot_selector()

    def _clear_device_list(self, device_type):
        """Clears the specified device list."""
        if device_type == 'setting':
            self.setting_devices.clear()
            self.setting_file_label.config(text="0 devices in Setting List.")
            self._update_settings_display()
            self._populate_scan_tab() # Update scan tab as well
        elif device_type == 'reading':
            self.reading_devices.clear()
            self.reading_file_label.config(text="0 devices in Reading List.")
            self._update_plot_selector()
        print(f"[SUCCESS] The {device_type} device list has been cleared.")

    def _update_plot_selector(self):
        """Updates the values in the device plotting dropdown and sets a default."""
        self.plot_selector_combo['values'] = self.reading_devices
        if self.reading_devices:
            self.plotted_device.set(self.reading_devices[0])
        else:
            self.plotted_device.set('')
        self._update_plots() # Also refresh the plot to show the new default or clear it

    def _open_device_selector(self):
        """Open a dialog to select a device file, then open the selection tool."""
        filepath = filedialog.askopenfilename(
            title="Select Device File",
            filetypes=(("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if not filepath:
            return

        try:
            device_data = {}
            if filepath.endswith(('.xlsx', '.xls')):
                xls = pd.ExcelFile(filepath)
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    device_data[sheet_name] = df.iloc[:, 0].dropna().tolist()
            else: # CSV file
                df = pd.read_csv(filepath)
                sheet_name = os.path.basename(filepath)
                device_data[sheet_name] = df.iloc[:, 0].dropna().tolist()
            
            DeviceSelectionDialog(self, device_data)

        except Exception as e:
            print(f"[ERROR] Failed to read file: {e}")

    def _update_settings_display(self):
        """Clears and repopulates the settings grid with selected setting_devices."""
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.device_setting_entries.clear()
        
        ttk.Label(self.scrollable_frame, text="Device", font=('Helvetica', 10, 'bold')).grid(row=0, column=0, padx=5, pady=2, sticky='w')
        ttk.Label(self.scrollable_frame, text="New Value", font=('Helvetica', 10, 'bold')).grid(row=0, column=1, padx=5, pady=2, sticky='w')

        for i, device in enumerate(self.setting_devices, start=1):
            ttk.Label(self.scrollable_frame, text=device).grid(row=i, column=0, padx=5, pady=2, sticky='w')
            entry = ttk.Entry(self.scrollable_frame, width=20)
            entry.grid(row=i, column=1, padx=5, pady=2, sticky='ew')
            self.device_setting_entries[device] = entry

    def _fetch_nominals(self):
        """Fetches the current nominal setting values for the devices in the setting list."""
        if not self.setting_devices:
            print("[WARNING] No setting devices have been loaded.")
            return
        self._sync_dpm_node()

        try:
            print(f"Fetching nominals for: {self.setting_devices}")
            thread = threading.Thread(target=self._fetch_nominals_thread_target)
            thread.start()

        except Exception as e:
            print(f"[ERROR] Failed to start thread to fetch nominals: {e}")

    def _fetch_nominals_thread_target(self):
        """The actual logic for fetching nominals, run in a separate thread."""
        try:
            nominals = self.scanner.get_settings_once(self.setting_devices)
            if not self._closing:
                self.after(0, self._update_nominals_in_gui, nominals)
        except Exception as e:
            print(f"[ERROR] Failed to fetch nominals from ACNET: {e}")

    def _update_nominals_in_gui(self, nominals):
        """Updates the GUI with the fetched nominal values. Must be called from the main thread."""
        if self._closing:
            return
        for device, value in zip(self.setting_devices, nominals):
            if device in self.device_setting_entries:
                entry = self.device_setting_entries[device]
                entry.delete(0, tk.END)
                if value is not None:
                    entry.insert(0, f"{value:.4f}")
                else:
                    entry.insert(0, "ERROR")
        print("[SUCCESS] Nominal values have been fetched.")
        
    def _apply_settings_with_confirmation(self):
        """Apply settings with confirmation dialog for safety."""
        # Count how many settings will be applied
        count = sum(1 for entry in self.device_setting_entries.values() if entry.get().strip())

        if count == 0:
            print("[INFO] No values entered to apply.")
            return

        # Show confirmation dialog
        message = f"You are about to apply settings to {count} device(s).\n\nThis will modify hardware values.\n\nAre you sure you want to continue?"

        def on_confirm():
            self._apply_settings()

        ConfirmationDialog(self, "Confirm Apply Settings", message, on_confirm)

    def _apply_settings(self):
        """Gathers values from the UI and tells the scanner to apply them."""
        devices_to_set = []
        values_to_set = []
        for device, entry in self.device_setting_entries.items():
            value_str = entry.get()
            if value_str:
                try:
                    devices_to_set.append(device)
                    values_to_set.append(float(value_str))
                except (ValueError, TypeError):
                    print(f"[ERROR] Value for {device} is not a valid number: '{value_str}'")
                    return

        if not devices_to_set:
            print("[INFO] No values entered to apply.")
            return

        role = self.setting_role.get()
        if not role:
            print("[ERROR] ACNET Role cannot be empty.")
            return

        def _apply_thread():
            try:
                self.scanner.apply_settings_once(devices_to_set, values_to_set, role)
                self._log_settings(devices_to_set, values_to_set, role)
                print("[SUCCESS] Settings have been sent to the devices.")
            except Exception as e:
                print(f"[ERROR] Failed to apply settings: {e}")
        threading.Thread(target=_apply_thread, daemon=True).start()

    def _log_settings(self, devices, values, role):
        """Logs applied settings to a local CSV file."""
        log_file = "settings_log.csv"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df = pd.DataFrame({
            'Timestamp': [timestamp] * len(devices),
            'Role': [role] * len(devices),
            'Device': devices,
            'Value': values
        })
        try:
            df.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)
        except Exception as e:
            print(f"[ERROR] Failed to write to log file: {e}")
            
    def _preview_scan_waveform(self):
        """Generate and show a quick plot of the upcoming scan waveform."""
        try:
            run_config = self._build_scan_run_config()
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            return
        if run_config.total_steps <= 1:
            print("[WARNING] Not enough scan steps to generate a preview.")
            return
        if self.preview_scan_button is not None:
            self.preview_scan_button.config(state="disabled")
        thread = threading.Thread(
            target=self._preview_waveform_thread_target,
            args=(run_config,),
            daemon=True,
        )
        thread.start()

    def _preview_waveform_thread_target(self, run_config):
        """Background worker that assembles data for the scan preview plot."""
        try:
            try:
                nominals = self.scanner.get_settings_once(run_config.device_names)
            except Exception as exc:
                print(f"[WARNING] Unable to fetch current device settings for preview: {exc}")
                nominals = [None] * len(run_config.devices)

            total_steps = run_config.total_steps
            max_points = 2000
            if total_steps <= max_points:
                sample_indices = list(range(total_steps))
            else:
                sample_indices_array = np.linspace(0, total_steps - 1, max_points, dtype=int)
                sample_indices = np.unique(sample_indices_array).tolist()

            curves = []
            used_default_any = False
            for idx, config in enumerate(run_config.devices):
                nominal_raw = nominals[idx] if idx < len(nominals) else None
                used_default = False
                try:
                    nominal = float(nominal_raw)
                    if not math.isfinite(nominal):
                        nominal = 0.0
                        used_default = True
                except (TypeError, ValueError):
                    nominal = 0.0
                    used_default = True
                values = [
                    config.compute_value(nominal, step, run_config.points_per_superperiod)
                    for step in sample_indices
                ]
                curves.append({
                    'device': config.device,
                    'values': values,
                    'nominal': nominal,
                    'amplitude': config.amplitude,
                    'periods': config.periods,
                    'used_default': used_default,
                })
                used_default_any = used_default_any or used_default

            info = {
                'total_steps': total_steps,
                'sampled_points': len(sample_indices),
                'used_default_nominal': used_default_any,
            }
            self.after(0, self._render_scan_preview, run_config, sample_indices, curves, info)
        except Exception as exc:
            print(f"[ERROR] Failed to build scan preview: {exc}")
        finally:
            self.after(0, self._enable_preview_button)


    def _render_scan_preview(self, run_config, sample_indices, curves, info):
        """Render the scan preview plot in a dedicated window."""
        if self._closing:
            return
        window_exists = self._scan_preview_window is not None and self._scan_preview_window.winfo_exists()
        if not window_exists:
            preview_window = tk.Toplevel(self)
            preview_window.title("Scan Waveform Preview")
            preview_window.geometry("820x520")
            preview_window.configure(bg=PALETTE['background'])

            frame = ttk.Frame(preview_window, padding=12, style="Main.TFrame")
            frame.pack(fill='both', expand=True)

            fig = Figure(figsize=(6.5, 4.5), dpi=100, facecolor=PALETTE['card'])
            canvas = FigureCanvasTkAgg(fig, master=frame)
            preview_widget = canvas.get_tk_widget()
            preview_widget.configure(bg=PALETTE['card'], highlightthickness=0, borderwidth=0)
            preview_widget.pack(fill='both', expand=True)

            toolbar = NavigationToolbar2Tk(canvas, frame)
            toolbar.update()
            self._style_toolbar(toolbar)

            info_label = ttk.Label(frame, text='', wraplength=780, justify='left', style='Info.TLabel')
            info_label.pack(fill='x', pady=(10, 0))

            preview_window.figure = fig
            preview_window.canvas = canvas
            preview_window.info_label = info_label
            self._scan_preview_window = preview_window

            def on_close():
                self._scan_preview_window = None
                preview_window.destroy()

            preview_window.protocol("WM_DELETE_WINDOW", on_close)
        else:
            preview_window = self._scan_preview_window

        fig = preview_window.figure
        canvas = preview_window.canvas
        fig.clear()
        ax = fig.add_subplot(111)
        color_cycle = ['#38bdf8', '#f97316', '#22c55e', '#facc15', '#a855f7', '#f472b6']

        step_array = np.array(sample_indices, dtype=float)
        for idx, curve in enumerate(curves):
            label = f"{curve['device']} (amp={curve['amplitude']:.3f}, periods={curve['periods']})"
            color = color_cycle[idx % len(color_cycle)]
            ax.plot(step_array, curve['values'], label=label, color=color, linewidth=1.4)

        ax.set_xlabel('Step Index')
        ax.set_ylabel('Device Value')
        ax.set_title('Planned Scan Waveforms')
        self._style_axis(ax)

        legend = ax.legend()
        if legend:
            legend.get_frame().set_facecolor(PALETTE['surface'])
            legend.get_frame().set_edgecolor(PALETTE['border'])
            for text_item in legend.get_texts():
                text_item.set_color(PALETTE['text'])

        fig.tight_layout()
        canvas.draw()

        summary_parts = [
            f"Points per superperiod: {run_config.points_per_superperiod}",
            f"Superperiods: {run_config.superperiods}",
            f"Total steps: {info['total_steps']}"
        ]
        if info['sampled_points'] < info['total_steps']:
            summary_parts.append(f"Preview samples: {info['sampled_points']}")
        if info['used_default_nominal']:
            summary_parts.append("Nominal unavailable for some devices; baseline assumed 0.")
        preview_window.info_label.config(text=' | '.join(summary_parts))

        self.tabControl.select(self.scan_tab)
        preview_window.deiconify()
        preview_window.lift()

    def _enable_preview_button(self):
        """Re-enable the scan preview button after background work finishes."""
        if self._closing:
            return
        if self.preview_scan_button is not None:
            self.preview_scan_button.config(state="normal")


    def _start_scan(self):
        """Gathers parameters from the UI and starts the appropriate scan."""
        if self.is_reading or self.is_scanning:
            print("[WARNING] A scan or reading process is already in progress.")
            return
        self._sync_dpm_node()
        try:
            run_config = self._build_scan_run_config()
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            return

        self.is_scanning = True
        self.stop_scan_flag.clear()
        self.plot_data.clear()
        self.device_stats.clear()
        self._last_run_config = run_config

        # Reset safety monitor state
        if self.safety_config.enabled:
            self.safety_monitor.clear_buffers()
            self.safety_monitor.reset_abort_state()
            print("[INFO] Safety monitoring active for this scan")

        # Build beam interlock config from UI (must be on main thread)
        self.beam_interlock_config = self._build_beam_interlock_config()
        self.beam_interlock.update_config(self.beam_interlock_config)
        self.beam_interlock.reset()

        # Freeze an immutable snapshot of the interlock config for the WHOLE
        # scan. The scan worker reads ONLY this snapshot (enabled/thresholds/
        # loss list/beam_role/beam_event), never the live self.beam_interlock_config,
        # so a mid-scan UI edit cannot change what the running scan enforces.
        frozen_interlock_config = copy.deepcopy(self.beam_interlock_config)

        # Lock the beam buttons / interlock toggle now (#4/#5) so the pre-scan
        # ACNET preflight and the scan run with the controls disabled.
        self._update_scan_buttons(is_scanning=True)

        if not frozen_interlock_config.enabled:
            # No interlock: no ACNET preflight needed, launch directly.
            self._start_scan_after_preflight(run_config, frozen_interlock_config)
            return

        # Interlock enabled: run the blocking beam-status + loss-monitor preflight
        # on a short worker thread (#5, still via the single shared ACNET
        # executor) and marshal the decision back to the Tk main thread.
        print("[INFO] Beam loss interlock active for this scan")

        def _preflight():
            # Require a POSITIVELY-CONFIRMED beam-ON before starting. Unknown
            # (None) beam state is a preflight FAILURE — retry once.
            beam_on = self.beam_interlock.check_beam_on()
            if beam_on is None:
                print("[WARNING] Could not verify beam status — retrying once")
                beam_on = self.beam_interlock.check_beam_on()
            loss_readings = {}
            if beam_on is True:
                loss_readings = self.beam_interlock.read_loss_monitors()
            return beam_on, loss_readings

        def _done(result, error):
            if error is not None:
                print(f"[ERROR] Pre-scan beam check failed: {error} — cannot start scan.")
                self._set_beam_status(BeamStatus.UNKNOWN)
                self.is_scanning = False
                self._update_scan_buttons(is_scanning=False)
                return
            beam_on, loss_readings = result
            if beam_on is False:
                print("[ERROR] Beam is OFF — cannot start scan. Enable beam first.")
                self._set_beam_status(BeamStatus.OFF)
                self.is_scanning = False
                self._update_scan_buttons(is_scanning=False)
                return
            if beam_on is None:
                print("[ERROR] Beam status UNKNOWN after retry — cannot start scan "
                      "with the interlock enabled on an unverified beam state.")
                self._set_beam_status(BeamStatus.UNKNOWN)
                self.is_scanning = False
                self._update_scan_buttons(is_scanning=False)
                return

            print("[INFO] Pre-scan beam check: beam is ON")
            self._set_beam_status(BeamStatus.ON)

            # Pre-scan loss monitor check
            for dev, val in loss_readings.items():
                thresh = frozen_interlock_config.loss_monitors.get(dev)
                if thresh is not None and val > thresh:
                    print(f"[ERROR] Pre-scan loss monitor {dev} = {val:.4f} already exceeds threshold {thresh:.4f} — aborting")
                    self.is_scanning = False
                    self._update_scan_buttons(is_scanning=False)
                    return

            self._start_scan_after_preflight(run_config, frozen_interlock_config)

        self._run_beam_op_async(_preflight, _done)

    def _start_scan_after_preflight(self, run_config, frozen_interlock_config):
        """Launch the scan worker thread after the (optional) ACNET preflight.

        Runs on the Tk main thread (either directly when the interlock is
        disabled, or marshalled back from the preflight worker). Captures Tk vars
        here before spawning the worker thread.
        """
        if self._closing:
            self.is_scanning = False
            return

        self.scan_progressbar.config(maximum=run_config.total_steps, value=0)

        scan_mode = self.scan_mode.get()
        print(f"[INFO] Starting {scan_mode.lower()} scan — "
              f"{len(run_config.devices)} devices, {run_config.total_steps} steps")

        metadata = run_config.to_metadata()
        # Pre-capture tkinter var on main thread before starting scan thread
        acnet_event = self.acnet_event.get()
        use_persistent = self.use_persistent_scan.get()
        if scan_mode == "Simultaneous":
            # Opt-in persistent advance-on-event engine; default off -> serial path.
            if use_persistent:
                print("[INFO] Using the persistent advance-on-event engine "
                      "(machine-validated ~6x faster, no per-step teardown).")
            sync_target = (self._persistent_synchronous_scan_loop if use_persistent
                           else self._synchronous_scan_loop)
            thread = threading.Thread(
                target=sync_target,
                args=(run_config, metadata, acnet_event, frozen_interlock_config),
                daemon=True,
            )
        else:  # Sequential
            # Capture tkinter vars on main thread before starting scan thread
            seq_save_dir = self.scan_data_path.get()
            seq_auto_calc = self.auto_calc_orm.get()
            thread = threading.Thread(
                target=self._sequential_scan_loop,
                args=(run_config, metadata, seq_save_dir, seq_auto_calc,
                      acnet_event, frozen_interlock_config),
                daemon=True,
            )

        self._scan_thread = thread
        thread.start()
        self._update_scan_buttons(is_scanning=True)

    def _synchronous_scan_loop(self, run_config, metadata, acnet_event=None,
                               interlock_config=None):
        """Main scan loop for simultaneous modulation.

        interlock_config: immutable BeamInterlockConfig snapshot frozen at scan
        start (see _start_scan). The worker reads ONLY this snapshot for the
        interlock's enabled/thresholds/loss list/beam_role, never the live
        self.beam_interlock_config.
        """
        all_scan_data = []
        devices_to_scan = run_config.device_names
        safety_abort_occurred = False
        scan_started = False
        scan_error = False
        user_stopped = False
        restore_devices = None
        restore_values = None
        # Per-device health accounting (missed/non-finite/rejected-SET). Fresh
        # for every scan; individual-device problems are recorded here, never abort.
        self.device_health = {}
        # Frozen interlock snapshot: fall back to a deepcopy of the live config
        # if a caller invoked this without one (keeps the worker off live state).
        if interlock_config is None:
            interlock_config = copy.deepcopy(self.beam_interlock_config)
        interlock_enabled = interlock_config.enabled
        try:
            # Clear stale nominals from prior scans before storing new ones
            self.nominal_settings.clear()

            try:
                nominals = self._acnet_call_with_renewal(
                    self.scanner.get_settings_once, devices_to_scan)
            except CredentialExpiredError:
                print("[ERROR] Cannot read device nominals — Kerberos ticket expired.")
                return
            nominal_map, fallback_devices = self._build_nominal_map(devices_to_scan, nominals)
            fallback_set = set(fallback_devices)
            if fallback_devices:
                print(f"[ERROR] Could not read nominals for: {', '.join(sorted(fallback_set))}")
                print(f"[ERROR] Aborting scan — cannot safely modulate devices with unknown nominals.")
                return

            scan_started = True

            # Store nominal settings for safety restoration
            for dev in devices_to_scan:
                self.nominal_settings.store(dev, nominal_map[dev])

            drf_list, scan_drf_devices = self._build_readback_drf_list(acnet_event=acnet_event)
            # Loss monitors ride along in the same event read — avoids a second
            # blocking read_once_on_event per scan step. Use the FROZEN config.
            loss_drf_list = self._build_loss_monitor_drf_list(config=interlock_config)
            n_scan_drf = len(drf_list)
            combined_drf_list = drf_list + loss_drf_list
            nominal_vector = [nominal_map[dev] for dev in devices_to_scan]
            restore_devices = devices_to_scan
            restore_values = nominal_vector

            consecutive_timeouts = 0
            # Loss-monitor misses are tracked INDEPENDENTLY of the BPM slice: a
            # configured+enabled loss monitor that is missing/non-finite warns
            # loudly and counts toward LOSS_MONITOR_MAX_CONSECUTIVE_MISSES.
            loss_consecutive_misses = 0
            renewed = [False]
            for step in range(run_config.total_steps):
                if self.stop_scan_flag.is_set():
                    if not safety_abort_occurred:
                        user_stopped = True
                    print("Scan stopped by user.")
                    break

                step_values = self._compute_scan_step_values(
                    run_config.devices,
                    nominal_map,
                    step,
                    run_config.points_per_superperiod,
                )

                set_ok = self._acnet_call_with_renewal(
                    self.scanner.apply_settings_once, devices_to_scan, step_values, run_config.role,
                    abort_check=self.stop_scan_flag.is_set, _renewed=renewed)
                # apply_settings_once now returns a bool: if the SET was not
                # positively confirmed (and we are NOT stopping), record the
                # affected correctors to Device Health and continue — do not
                # silently treat a rejected SET as applied.
                if set_ok is False and not self.stop_scan_flag.is_set():
                    self._record_rejected_correctors(devices_to_scan)

                # Phase 1: the per-step get_settings_once readback round-trip is
                # dropped (~30% faster — it was 1 of the 2 remaining DPM opens per
                # step). The commanded setpoint is known analytically (step_values,
                # from ScanDeviceConfig.compute_value), so it is recorded directly
                # as the .SETTING column below. .SETTING is excluded from the ORM,
                # so the response matrix is unchanged.

                if self.stop_scan_flag.is_set():
                    if not safety_abort_occurred:
                        user_stopped = True
                    print("Scan stopped by user.")
                    break

                raw_data = self._acnet_call_with_renewal(
                    self.scanner.read_once_on_event, combined_drf_list,
                    abort_check=self.stop_scan_flag.is_set, _renewed=renewed) or []
                data_from_step = raw_data[:n_scan_drf]
                loss_data_from_step = raw_data[n_scan_drf:]

                # A stopped scan returns an empty/None read — break cleanly here
                # so it is not retried or miscounted as a connection timeout.
                if self.stop_scan_flag.is_set():
                    if not safety_abort_occurred:
                        user_stopped = True
                    print("Scan stopped by user.")
                    break

                # Retry once on empty read before counting as timeout
                if data_from_step and all(item is None for item in data_from_step):
                    print(f"[WARNING] Scan step {step}: no data — retrying read")
                    raw_data = self._acnet_call_with_renewal(
                        self.scanner.read_once_on_event, combined_drf_list,
                        abort_check=self.stop_scan_flag.is_set, _renewed=renewed) or []
                    data_from_step = raw_data[:n_scan_drf]
                    loss_data_from_step = raw_data[n_scan_drf:]

                # Track consecutive timeouts (all-None results)
                if data_from_step and all(item is None for item in data_from_step):
                    consecutive_timeouts += 1
                    print(f"[WARNING] Scan step {step}: no data received "
                          f"(timeout {consecutive_timeouts}/{ACNET_MAX_CONSECUTIVE_TIMEOUTS})")
                    if consecutive_timeouts >= ACNET_MAX_CONSECUTIVE_TIMEOUTS:
                        print(f"[ERROR] {ACNET_MAX_CONSECUTIVE_TIMEOUTS} consecutive timeouts — "
                              f"auto-aborting scan (beam may be off or ACNET connection lost)")
                        scan_error = True
                        self.stop_scan_flag.set()
                        break
                else:
                    consecutive_timeouts = 0

                # LOSS-MONITOR PRESENCE CHECK (independent of the BPM slice).
                # A configured+enabled loss monitor that is missing/non-finite
                # must warn loudly and count toward LOSS_MONITOR_MAX_CONSECUTIVE_MISSES;
                # never a silent "OK". This runs even when the BPM slice is empty.
                if interlock_enabled:
                    missing_loss = self._missing_loss_monitors(
                        loss_data_from_step, interlock_config)
                    if missing_loss:
                        loss_consecutive_misses += 1
                        for dev in missing_loss:
                            self._record_device_problem(dev, 'missed')
                        print(f"[WARNING] Loss monitor(s) missing/non-finite this "
                              f"step: {', '.join(missing_loss)} "
                              f"({loss_consecutive_misses}/{LOSS_MONITOR_MAX_CONSECUTIVE_MISSES})")
                        if loss_consecutive_misses >= LOSS_MONITOR_MAX_CONSECUTIVE_MISSES:
                            print(f"[ERROR] {LOSS_MONITOR_MAX_CONSECUTIVE_MISSES} consecutive "
                                  f"loss-monitor misses — cannot guarantee the loss interlock; "
                                  f"auto-aborting scan")
                            scan_error = True
                            safety_abort_occurred = True
                            self.stop_scan_flag.set()
                            break
                    else:
                        loss_consecutive_misses = 0

                if data_from_step:
                    # Per-device exclude-and-report accounting BEFORE filtering:
                    # absent/None slots -> 'missed'; finite -> 'last_seen'.
                    self._account_bpm_step_health(
                        data_from_step, scan_drf_devices, [])
                    # Filter out None values (from timeouts) and add to scan data
                    valid_data = [item for item in data_from_step if item is not None]
                    if valid_data:
                        # Normalize timestamps: all devices from the same event
                        # get slightly different stamps from ACNET; unify them so
                        # they group into one row when pivoted into a CSV.
                        common_stamp = valid_data[0]['stamp']
                        for item in valid_data:
                            item['stamp'] = common_stamp

                        # Merge the analytically COMMANDED setpoint as the .SETTING
                        # column (Phase 1). step_values is the exact commanded
                        # vector (ScanDeviceConfig.compute_value), aligned with
                        # devices_to_scan — the same vector just sent via
                        # apply_settings_once. Excluded from the ORM.
                        for dev, val in zip(devices_to_scan, step_values):
                            try:
                                valid_data.append({
                                    'stamp': common_stamp,
                                    'name': f"{dev}.SETTING",
                                    'data': float(val),
                                })
                            except (TypeError, ValueError):
                                pass

                        all_scan_data.extend(valid_data)
                        self.data_queue.put(valid_data)

                        # SAFETY CHECK: Monitor reading devices for threshold violations
                        if self.safety_config.enabled and self.safety_baselines_measured:
                            try:
                                # Collect non-finite device names so they are
                                # EXCLUDED from analysis but recorded to Device
                                # Health (they never abort on their own).
                                nonfinite_devices = []
                                violations = self.safety_monitor.check_batch(
                                    valid_data, nonfinite_out=nonfinite_devices)
                                for dev in nonfinite_devices:
                                    self._record_device_problem(dev, 'nonfinite')
                                abort_violations = [v for v in violations if v.violation_type == ViolationType.ABORT]

                                if abort_violations:
                                    print("[ERROR] Safety abort triggered - stopping scan")
                                    print("[INFO] Abort data point HAS been saved to scan data")
                                    safety_abort_occurred = True
                                    scan_error = True
                                    self._trigger_safety_abort(abort_violations)
                                    self.stop_scan_flag.set()
                                    break
                            except Exception as e:
                                # The safety check itself failed — protection can no
                                # longer be guaranteed, so stop rather than continue
                                # unmonitored. The finally block restores nominals.
                                print(f"[ERROR] Safety check failed — aborting scan: {e}")
                                scan_error = True
                                safety_abort_occurred = True
                                self.stop_scan_flag.set()
                                break

                        # BEAM LOSS INTERLOCK CHECK (frozen config)
                        if interlock_enabled:
                            trip = self._check_loss_monitor_data(
                                loss_data_from_step, config=interlock_config)
                            if trip is not None:
                                # #1: RESTORE correctors to nominal FIRST, then
                                # trip the beam. Restore runs to completion with
                                # NO abort_check (safety-critical).
                                self._restore_correctors_on_trip(
                                    restore_devices, restore_values, run_config.role)
                                beam_role = interlock_config.beam_role or run_config.role
                                beam_off = self.beam_interlock.trip_beam(trip, beam_role)
                                if beam_off:
                                    if not self._closing:
                                        self.after(0, self._set_beam_status, BeamStatus.OFF)
                                else:
                                    # Not confirmed OFF — escalate exactly like the
                                    # manual _on_beam_disable path (modal + UNKNOWN).
                                    self._escalate_beam_trip_unconfirmed(trip)
                                safety_abort_occurred = True
                                scan_error = True
                                self.stop_scan_flag.set()
                                print(f"[ERROR] Beam tripped — scan aborted: {trip}")
                                break

                if not self._closing:
                    self.after(0, self.scan_progressbar.step)
        except Exception as e:
            print(f"[ERROR] An error occurred during the scan: {e}")
            scan_error = True
        finally:
            if scan_started:
                if restore_devices and restore_values:
                    try:
                        self.scanner.apply_settings_once(restore_devices, restore_values, run_config.role)
                    except CredentialExpiredError:
                        print("[WARNING] Credential expired during nominal restore — attempting renewal...")
                        if self._renew_kerberos_ticket():
                            try:
                                self.scanner.apply_settings_once(restore_devices, restore_values, run_config.role)
                                print("[SUCCESS] Nominal settings restored after credential renewal.")
                            except Exception as e2:
                                print(f"[ERROR] Failed to restore nominal settings after renewal: {e2}")
                        else:
                            print("[ERROR] Failed to restore nominal settings — credential renewal failed. "
                                  "MANUALLY VERIFY DEVICE SETTINGS!")
                    except Exception as e:
                        print(f"[ERROR] Failed to restore nominal settings: {e}")
                    self._verify_nominal_restore(restore_devices, restore_values)
                if not self._closing:
                    self.after(0, self._on_scan_complete, all_scan_data, metadata, scan_error, user_stopped)
            else:
                # Preflight failed — reset UI without success message
                self.is_scanning = False
                if not self._closing:
                    self.after(0, self._update_scan_buttons, False)

    def _persistent_synchronous_scan_loop(self, run_config, metadata, acnet_event=None,
                                          interlock_config=None):
        """OPT-IN persistent advance-on-event variant of _synchronous_scan_loop.

        Same preflight, safety contract, and teardown as the serial loop, but the
        per-step apply+read (two fresh DPMContexts per step, ~1.2 s) is replaced by
        ONE persistent settings context + ONE persistent read subscription driven by
        AcnetScanner.run_advance_on_event_scan (machine-validated 2026-06-04: ~6x
        faster, 100/100 complete events on 71 devices, no stall). The per-step body
        runs inside on_event() and reuses the SAME helper methods as the serial loop
        (no divergence of the safety logic). The serial _synchronous_scan_loop is
        unchanged and remains the default; this path is reached only when the
        'Persistent (advance-on-event)' option is on.

        Behaviour differences vs the serial loop (validate on the machine):
          * No per-step SET-confirmation (engine fires the SET; the corrector
            READBACK in each snapshot is the confirmation). _record_rejected_
            correctors is therefore not used here.
          * No explicit retry-once on an empty read: a total read stall surfaces as
            an all-None snapshot which still feeds the consecutive-timeout abort.
          * On a loss trip the engine is stopped first, THEN correctors are restored
            and the beam tripped (correct order preserved; the single executor must
            be free for those serial ops, so they cannot run inside the callback).
        """
        all_scan_data = []
        devices_to_scan = run_config.device_names
        safety_abort_occurred = False
        scan_started = False
        scan_error = False
        user_stopped = False
        restore_devices = None
        restore_values = None
        self.device_health = {}
        if interlock_config is None:
            interlock_config = copy.deepcopy(self.beam_interlock_config)
        interlock_enabled = interlock_config.enabled
        # Cross-step state held in a dict so the on_event closure can mutate it.
        st = {'consecutive_timeouts': 0, 'loss_consecutive_misses': 0,
              'safety_abort': False, 'scan_error': False, 'pending_trip': None,
              'user_stopped': False}
        try:
            self.nominal_settings.clear()
            try:
                nominals = self._acnet_call_with_renewal(
                    self.scanner.get_settings_once, devices_to_scan)
            except CredentialExpiredError:
                print("[ERROR] Cannot read device nominals — Kerberos ticket expired.")
                return
            nominal_map, fallback_devices = self._build_nominal_map(devices_to_scan, nominals)
            if fallback_devices:
                print(f"[ERROR] Could not read nominals for: {', '.join(sorted(set(fallback_devices)))}")
                print(f"[ERROR] Aborting scan — cannot safely modulate devices with unknown nominals.")
                return
            scan_started = True
            for dev in devices_to_scan:
                self.nominal_settings.store(dev, nominal_map[dev])

            drf_list, scan_drf_devices = self._build_readback_drf_list(acnet_event=acnet_event)
            loss_drf_list = self._build_loss_monitor_drf_list(config=interlock_config)
            n_scan_drf = len(drf_list)
            combined_drf_list = drf_list + loss_drf_list
            nominal_vector = [nominal_map[dev] for dev in devices_to_scan]
            restore_devices = devices_to_scan
            restore_values = nominal_vector
            total_steps = run_config.total_steps

            def _step_values(step):
                return self._compute_scan_step_values(
                    run_config.devices, nominal_map, step, run_config.points_per_superperiod)

            def on_event(idx, snapshot, complete):
                """Per-step body — mirrors _synchronous_scan_loop. Returns the next
                setpoint vector, or None to stop the engine."""
                if self.stop_scan_flag.is_set():
                    st['user_stopped'] = not st['safety_abort']
                    return None
                data_from_step = snapshot[:n_scan_drf]
                loss_data_from_step = snapshot[n_scan_drf:]
                step_values = _step_values(idx)

                # Consecutive all-None (total read stall) -> same abort as serial.
                if data_from_step and all(item is None for item in data_from_step):
                    st['consecutive_timeouts'] += 1
                    print(f"[WARNING] Scan step {idx}: no data received "
                          f"(timeout {st['consecutive_timeouts']}/{ACNET_MAX_CONSECUTIVE_TIMEOUTS})")
                    if st['consecutive_timeouts'] >= ACNET_MAX_CONSECUTIVE_TIMEOUTS:
                        print(f"[ERROR] {ACNET_MAX_CONSECUTIVE_TIMEOUTS} consecutive timeouts — "
                              f"auto-aborting scan (beam may be off or ACNET connection lost)")
                        st['scan_error'] = True
                        self.stop_scan_flag.set()
                        return None
                else:
                    st['consecutive_timeouts'] = 0

                # Loss-monitor presence check (independent of the BPM slice).
                if interlock_enabled:
                    missing_loss = self._missing_loss_monitors(loss_data_from_step, interlock_config)
                    if missing_loss:
                        st['loss_consecutive_misses'] += 1
                        for dev in missing_loss:
                            self._record_device_problem(dev, 'missed')
                        print(f"[WARNING] Loss monitor(s) missing/non-finite this step: "
                              f"{', '.join(missing_loss)} "
                              f"({st['loss_consecutive_misses']}/{LOSS_MONITOR_MAX_CONSECUTIVE_MISSES})")
                        if st['loss_consecutive_misses'] >= LOSS_MONITOR_MAX_CONSECUTIVE_MISSES:
                            print(f"[ERROR] {LOSS_MONITOR_MAX_CONSECUTIVE_MISSES} consecutive loss-monitor "
                                  f"misses — cannot guarantee the loss interlock; auto-aborting scan")
                            st['scan_error'] = True
                            st['safety_abort'] = True
                            self.stop_scan_flag.set()
                            return None
                    else:
                        st['loss_consecutive_misses'] = 0

                if data_from_step:
                    self._account_bpm_step_health(data_from_step, scan_drf_devices, [])
                    valid_data = [item for item in data_from_step if item is not None]
                    if valid_data:
                        common_stamp = valid_data[0]['stamp']
                        for item in valid_data:
                            item['stamp'] = common_stamp
                        # Analytically COMMANDED setpoint as .SETTING (excluded from ORM).
                        for dev, val in zip(devices_to_scan, step_values):
                            try:
                                valid_data.append({'stamp': common_stamp,
                                                   'name': f"{dev}.SETTING", 'data': float(val)})
                            except (TypeError, ValueError):
                                pass
                        all_scan_data.extend(valid_data)
                        self.data_queue.put(valid_data)

                        if self.safety_config.enabled and self.safety_baselines_measured:
                            try:
                                nonfinite_devices = []
                                violations = self.safety_monitor.check_batch(
                                    valid_data, nonfinite_out=nonfinite_devices)
                                for dev in nonfinite_devices:
                                    self._record_device_problem(dev, 'nonfinite')
                                abort_violations = [v for v in violations
                                                    if v.violation_type == ViolationType.ABORT]
                                if abort_violations:
                                    print("[ERROR] Safety abort triggered - stopping scan")
                                    print("[INFO] Abort data point HAS been saved to scan data")
                                    st['safety_abort'] = True
                                    st['scan_error'] = True
                                    self._trigger_safety_abort(abort_violations)
                                    self.stop_scan_flag.set()
                                    return None
                            except Exception as e:
                                print(f"[ERROR] Safety check failed — aborting scan: {e}")
                                st['scan_error'] = True
                                st['safety_abort'] = True
                                self.stop_scan_flag.set()
                                return None

                        if interlock_enabled:
                            trip = self._check_loss_monitor_data(loss_data_from_step, config=interlock_config)
                            if trip is not None:
                                # Restore + beam-trip cannot run inside the engine
                                # coroutine (the single executor is busy). Record the
                                # trip and STOP; the post-engine handler restores
                                # correctors BEFORE tripping the beam (serial loop order).
                                st['pending_trip'] = trip
                                st['safety_abort'] = True
                                st['scan_error'] = True
                                self.stop_scan_flag.set()
                                return None

                if not self._closing:
                    self.after(0, self.scan_progressbar.step)
                if idx + 1 >= total_steps:
                    return None
                return _step_values(idx + 1)

            # Drive the persistent engine. per_event_tmo > @e,52's ~730 ms bursty gap
            # AND long enough that a beam-gated pause is not falsely seen as a stall;
            # 5 consecutive stalls still auto-abort (~25 s), like the serial loop.
            self.scanner.run_advance_on_event_scan(
                combined_drf_list, devices_to_scan, _step_values(0), run_config.role,
                on_event, abort_check=self.stop_scan_flag.is_set, per_event_tmo=5.0)

            safety_abort_occurred = st['safety_abort']
            scan_error = st['scan_error']
            user_stopped = st['user_stopped']

            # Loss trip: engine has stopped, so the executor is free for these serial
            # ops. Restore correctors FIRST, then trip the beam (serial loop order).
            if st['pending_trip'] is not None:
                trip = st['pending_trip']
                self._restore_correctors_on_trip(restore_devices, restore_values, run_config.role)
                beam_role = interlock_config.beam_role or run_config.role
                beam_off = self.beam_interlock.trip_beam(trip, beam_role)
                if beam_off:
                    if not self._closing:
                        self.after(0, self._set_beam_status, BeamStatus.OFF)
                else:
                    self._escalate_beam_trip_unconfirmed(trip)
                print(f"[ERROR] Beam tripped — scan aborted: {trip}")
        except Exception as e:
            print(f"[ERROR] An error occurred during the scan: {e}")
            scan_error = True
        finally:
            if scan_started:
                if restore_devices and restore_values:
                    try:
                        self.scanner.apply_settings_once(restore_devices, restore_values, run_config.role)
                    except CredentialExpiredError:
                        print("[WARNING] Credential expired during nominal restore — attempting renewal...")
                        if self._renew_kerberos_ticket():
                            try:
                                self.scanner.apply_settings_once(restore_devices, restore_values, run_config.role)
                                print("[SUCCESS] Nominal settings restored after credential renewal.")
                            except Exception as e2:
                                print(f"[ERROR] Failed to restore nominal settings after renewal: {e2}")
                        else:
                            print("[ERROR] Failed to restore nominal settings — credential renewal failed. "
                                  "MANUALLY VERIFY DEVICE SETTINGS!")
                    except Exception as e:
                        print(f"[ERROR] Failed to restore nominal settings: {e}")
                    self._verify_nominal_restore(restore_devices, restore_values)
                if not self._closing:
                    self.after(0, self._on_scan_complete, all_scan_data, metadata, scan_error, user_stopped)
            else:
                self.is_scanning = False
                if not self._closing:
                    self.after(0, self._update_scan_buttons, False)

    def _sequential_scan_loop(self, run_config, metadata, save_dir=None, auto_calc=None, acnet_event=None,
                              interlock_config=None):
        """Scan loop that modulates one device at a time.

        interlock_config: immutable BeamInterlockConfig snapshot frozen at scan
        start (see _start_scan). The worker reads ONLY this snapshot for the
        interlock, never the live self.beam_interlock_config.
        """
        devices_to_scan = run_config.device_names
        total_steps = run_config.total_steps
        safety_abort_occurred = False
        scan_started = False
        scan_error = False
        user_stopped = False
        restore_devices = None
        restore_values = None
        # Per-device health accounting (missed/non-finite/rejected-SET). Fresh
        # for every scan; individual-device problems are recorded here, never abort.
        self.device_health = {}
        if interlock_config is None:
            interlock_config = copy.deepcopy(self.beam_interlock_config)
        interlock_enabled = interlock_config.enabled
        try:
            # Clear stale nominals from prior scans before storing new ones
            self.nominal_settings.clear()

            try:
                nominals = self._acnet_call_with_renewal(
                    self.scanner.get_settings_once, devices_to_scan)
            except CredentialExpiredError:
                print("[ERROR] Cannot read device nominals — Kerberos ticket expired.")
                return
            nominal_map, fallback_devices = self._build_nominal_map(devices_to_scan, nominals)
            fallback_set = set(fallback_devices)
            if fallback_devices:
                print(f"[ERROR] Could not read nominals for: {', '.join(sorted(fallback_set))}")
                print(f"[ERROR] Aborting scan — cannot safely modulate devices with unknown nominals.")
                return

            scan_started = True

            # Store nominal settings for safety restoration
            for dev in devices_to_scan:
                self.nominal_settings.store(dev, nominal_map[dev])

            drf_list, scan_drf_devices = self._build_readback_drf_list(acnet_event=acnet_event)
            # Loss monitors ride along in the same event read — avoids a second
            # blocking read_once_on_event per scan step. Use the FROZEN config.
            loss_drf_list = self._build_loss_monitor_drf_list(config=interlock_config)
            n_scan_drf = len(drf_list)
            combined_drf_list = drf_list + loss_drf_list
            nominal_vector = [nominal_map[dev] for dev in devices_to_scan]
            restore_devices = devices_to_scan
            restore_values = nominal_vector

            for index, config in enumerate(run_config.devices):
                if self.stop_scan_flag.is_set():
                    if not safety_abort_occurred:
                        user_stopped = True
                    print("Scan sequence stopped by user.")
                    break

                print(f"--- Starting sequential scan {index + 1}/{len(run_config.devices)} for {config.device} ---")
                if not self._closing:
                    self.after(0, self.scan_progressbar.config, {'value': 0})
                single_scan_data = []

                consecutive_timeouts = 0
                # Loss-monitor misses tracked INDEPENDENTLY of the BPM slice.
                loss_consecutive_misses = 0
                renewed = [False]
                for step in range(total_steps):
                    if self.stop_scan_flag.is_set():
                        break

                    step_values = self._compute_scan_step_values(
                        run_config.devices,
                        nominal_map,
                        step,
                        run_config.points_per_superperiod,
                        modulated_device=config.device,
                    )

                    set_ok = self._acnet_call_with_renewal(
                        self.scanner.apply_settings_once, devices_to_scan, step_values, run_config.role,
                        abort_check=self.stop_scan_flag.is_set, _renewed=renewed)
                    # Record rejected correctors (#3) — continue, do not abort.
                    if set_ok is False and not self.stop_scan_flag.is_set():
                        self._record_rejected_correctors(devices_to_scan)

                    # Phase 1: dropped the per-step get_settings_once round-trip;
                    # the commanded setpoint (step_values, ScanDeviceConfig.
                    # compute_value) is recorded directly as .SETTING below.
                    # Excluded from the ORM, so the response matrix is unchanged.

                    if self.stop_scan_flag.is_set():
                        break

                    raw_data = self._acnet_call_with_renewal(
                        self.scanner.read_once_on_event, combined_drf_list,
                        abort_check=self.stop_scan_flag.is_set, _renewed=renewed) or []
                    data_from_step = raw_data[:n_scan_drf]
                    loss_data_from_step = raw_data[n_scan_drf:]

                    # A stopped scan returns an empty/None read — break cleanly
                    # here so it is not retried or miscounted as a timeout.
                    if self.stop_scan_flag.is_set():
                        break

                    # Retry once on empty read before counting as timeout
                    if data_from_step and all(item is None for item in data_from_step):
                        print(f"[WARNING] Scan step {step}: no data — retrying read")
                        raw_data = self._acnet_call_with_renewal(
                            self.scanner.read_once_on_event, combined_drf_list,
                            abort_check=self.stop_scan_flag.is_set, _renewed=renewed) or []
                        data_from_step = raw_data[:n_scan_drf]
                        loss_data_from_step = raw_data[n_scan_drf:]

                    # Track consecutive timeouts (all-None results)
                    if data_from_step and all(item is None for item in data_from_step):
                        consecutive_timeouts += 1
                        print(f"[WARNING] Scan step {step}: no data received "
                              f"(timeout {consecutive_timeouts}/{ACNET_MAX_CONSECUTIVE_TIMEOUTS})")
                        if consecutive_timeouts >= ACNET_MAX_CONSECUTIVE_TIMEOUTS:
                            print(f"[ERROR] {ACNET_MAX_CONSECUTIVE_TIMEOUTS} consecutive timeouts — "
                                  f"auto-aborting scan (beam may be off or ACNET connection lost)")
                            scan_error = True
                            self.stop_scan_flag.set()
                            break
                    else:
                        consecutive_timeouts = 0

                    # LOSS-MONITOR PRESENCE CHECK (independent of the BPM slice).
                    # A configured+enabled loss monitor that is missing/non-finite
                    # warns loudly and counts toward LOSS_MONITOR_MAX_CONSECUTIVE_MISSES;
                    # never a silent "OK". Runs even when the BPM slice is empty.
                    if interlock_enabled:
                        missing_loss = self._missing_loss_monitors(
                            loss_data_from_step, interlock_config)
                        if missing_loss:
                            loss_consecutive_misses += 1
                            for dev in missing_loss:
                                self._record_device_problem(dev, 'missed')
                            print(f"[WARNING] Loss monitor(s) missing/non-finite this "
                                  f"step: {', '.join(missing_loss)} "
                                  f"({loss_consecutive_misses}/{LOSS_MONITOR_MAX_CONSECUTIVE_MISSES})")
                            if loss_consecutive_misses >= LOSS_MONITOR_MAX_CONSECUTIVE_MISSES:
                                print(f"[ERROR] {LOSS_MONITOR_MAX_CONSECUTIVE_MISSES} consecutive "
                                      f"loss-monitor misses — cannot guarantee the loss interlock; "
                                      f"auto-aborting scan")
                                scan_error = True
                                safety_abort_occurred = True
                                self.stop_scan_flag.set()
                                break
                        else:
                            loss_consecutive_misses = 0

                    if data_from_step:
                        # Per-device exclude-and-report accounting BEFORE filtering:
                        # absent/None slots -> 'missed'; finite -> 'last_seen'.
                        self._account_bpm_step_health(
                            data_from_step, scan_drf_devices, [])
                        # Filter out None values (from timeouts) and add to scan data
                        valid_data = [item for item in data_from_step if item is not None]
                        if valid_data:
                            # Normalize timestamps: all devices from the same event
                            # get slightly different stamps from ACNET; unify them so
                            # they group into one row when pivoted into a CSV.
                            common_stamp = valid_data[0]['stamp']
                            for item in valid_data:
                                item['stamp'] = common_stamp

                            # Merge the analytically COMMANDED setpoint as the
                            # .SETTING column (Phase 1). step_values is the exact
                            # commanded vector aligned with devices_to_scan.
                            for dev, val in zip(devices_to_scan, step_values):
                                try:
                                    valid_data.append({
                                        'stamp': common_stamp,
                                        'name': f"{dev}.SETTING",
                                        'data': float(val),
                                    })
                                except (TypeError, ValueError):
                                    pass

                            single_scan_data.extend(valid_data)
                            self.data_queue.put(valid_data)

                            # SAFETY CHECK: Monitor reading devices for threshold violations
                            if self.safety_config.enabled and self.safety_baselines_measured:
                                try:
                                    # Collect non-finite device names so the safety
                                    # monitor EXCLUDES them from the mean (they never
                                    # poison the buffer and never abort on their own).
                                    # device_health recording of non-finite is done
                                    # once in _account_bpm_step_health from raw data,
                                    # so we do not re-record here (avoids double count).
                                    nonfinite_devices = []
                                    violations = self.safety_monitor.check_batch(
                                        valid_data, nonfinite_out=nonfinite_devices)
                                    if nonfinite_devices:
                                        print(f"[WARNING] Safety monitor flagged "
                                              f"non-finite readings (excluded from "
                                              f"analysis): {', '.join(nonfinite_devices)}")
                                    abort_violations = [v for v in violations if v.violation_type == ViolationType.ABORT]

                                    if abort_violations:
                                        print("[ERROR] Safety abort triggered - stopping scan")
                                        print("[INFO] Abort data point HAS been saved to scan data")
                                        safety_abort_occurred = True
                                        scan_error = True
                                        self._trigger_safety_abort(abort_violations)
                                        self.stop_scan_flag.set()
                                        break
                                except Exception as e:
                                    # The safety check itself failed — protection can no
                                    # longer be guaranteed, so stop rather than continue
                                    # unmonitored. The finally block restores nominals.
                                    print(f"[ERROR] Safety check failed — aborting scan: {e}")
                                    scan_error = True
                                    safety_abort_occurred = True
                                    self.stop_scan_flag.set()
                                    break

                            # BEAM LOSS INTERLOCK CHECK (frozen config)
                            if interlock_enabled:
                                trip = self._check_loss_monitor_data(
                                    loss_data_from_step, config=interlock_config)
                                if trip is not None:
                                    # #1: RESTORE correctors to nominal FIRST, then
                                    # trip beam; restore runs to completion, no abort_check.
                                    self._restore_correctors_on_trip(
                                        restore_devices, restore_values, run_config.role)
                                    beam_role = interlock_config.beam_role or run_config.role
                                    beam_off = self.beam_interlock.trip_beam(trip, beam_role)
                                    if beam_off:
                                        if not self._closing:
                                            self.after(0, self._set_beam_status, BeamStatus.OFF)
                                    else:
                                        self._escalate_beam_trip_unconfirmed(trip)
                                    safety_abort_occurred = True
                                    scan_error = True
                                    self.stop_scan_flag.set()
                                    print(f"[ERROR] Beam tripped — scan aborted: {trip}")
                                    break

                    if not self._closing:
                        self.after(0, self.scan_progressbar.step)

                # Save scan data (full or partial)
                if not self.stop_scan_flag.is_set():
                    self._save_scan_data_sequentially(single_scan_data, [config.to_metadata_dict()], config.device,
                                                      save_dir=save_dir, auto_calc=auto_calc)
                else:
                    if not safety_abort_occurred:
                        user_stopped = True
                    if single_scan_data:
                        self._save_scan_data_sequentially(single_scan_data, [config.to_metadata_dict()], config.device,
                                                          save_dir=save_dir, auto_calc=False, partial=True)

                # Restore nominals after each device scan (only if no safety abort)
                # Only restore devices with known nominals (skip 0.0 fallbacks)
                if not safety_abort_occurred and restore_devices and restore_values:
                    self._acnet_call_with_renewal(
                        self.scanner.apply_settings_once, restore_devices, restore_values, run_config.role)
                    self._verify_nominal_restore(restore_devices, restore_values)
        except Exception as e:
            print(f"[ERROR] An error occurred during the sequential scan: {e}")
            scan_error = True
        finally:
            if scan_started:
                if restore_devices and restore_values:
                    try:
                        self.scanner.apply_settings_once(restore_devices, restore_values, run_config.role)
                    except CredentialExpiredError:
                        print("[WARNING] Credential expired during nominal restore — attempting renewal...")
                        if self._renew_kerberos_ticket():
                            try:
                                self.scanner.apply_settings_once(restore_devices, restore_values, run_config.role)
                                print("[SUCCESS] Nominal settings restored after credential renewal.")
                            except Exception as e2:
                                print(f"[ERROR] Failed to restore nominal settings after renewal: {e2}")
                        else:
                            print("[ERROR] Failed to restore nominal settings — credential renewal failed. "
                                  "MANUALLY VERIFY DEVICE SETTINGS!")
                    except Exception as e:
                        print(f"[ERROR] Failed to restore nominal settings: {e}")
                    self._verify_nominal_restore(restore_devices, restore_values)
                if not self._closing:
                    self.after(0, self._on_scan_complete, [], [], scan_error, user_stopped)
            else:
                # Preflight failed — reset UI without success message
                self.is_scanning = False
                if not self._closing:
                    self.after(0, self._update_scan_buttons, False)

    def _save_scan_data_sequentially(self, scan_data, scan_params, device_name,
                                      save_dir=None, auto_calc=None, partial=False):
        """Saves data for a single device from a sequential scan.

        Args:
            save_dir: Pre-captured save directory (avoids tkinter access from thread).
            auto_calc: Pre-captured auto_calc_orm flag (avoids tkinter access from thread).
            partial: If True, saves with PARTIAL_ prefix and skips auto ORM calculation.
        """
        if not scan_data:
            return

        timestr = datetime.now().strftime('%Y%m%d_%H%M%S')
        if save_dir is None:
            # Fallback: schedule on main thread to read tkinter var safely
            print("[WARNING] save_dir not pre-captured, using default '.'")
            save_dir = '.'
        if auto_calc is None:
            auto_calc = False

        safe_device_name = device_name.replace(":", "_").replace("-", "_")
        prefix = "PARTIAL_scan_data_SEQUENTIAL" if partial else "scan_data_SEQUENTIAL"
        data_filename = os.path.join(save_dir, f"{prefix}_{safe_device_name}_{timestr}.csv")
        meta_filename = os.path.join(save_dir, f"{prefix}_{safe_device_name}_{timestr}_meta.json")

        df = pd.DataFrame.from_records(scan_data)
        if not df.empty:
            try:
                pivot_df = df.pivot_table(index='stamp', columns='name', values='data')
                pivot_df.sort_index(inplace=True)
                atomic_to_csv(pivot_df, data_filename, index_label='Timestamp')
                if partial:
                    print(f"[INFO] Partial sequential scan data for {device_name} saved ({len(scan_data)} points) to:\n{data_filename}")
                else:
                    print(f"[SUCCESS] Sequential scan data for {device_name} saved to:\n{data_filename}")
                self.last_analysis_source = data_filename

                atomic_write_text(meta_filename,
                                  json.dumps(scan_params, indent=4),
                                  encoding='utf-8')
                print(f"[SUCCESS] Metadata for {device_name} saved to:\n{meta_filename}")

                # Auto-calculate and save ORM on the main thread (skip for partial scans)
                if auto_calc and not partial and not self._closing:
                    orm_timestr = f"SEQUENTIAL_{safe_device_name}_{timestr}"
                    self.after(0, self._auto_orm_for_sequential, data_filename, save_dir, orm_timestr, device_name)
            except Exception as e:
                print(f"[ERROR] Failed to save sequential scan data for {device_name}: {e}")

    def _auto_orm_for_sequential(self, data_filename, save_dir, orm_timestr, device_name):
        """Calculate and save ORM for a sequential device (runs on main thread)."""
        if self._closing:
            return
        print(f"[INFO] Auto-calculating ORM for sequential device {device_name}...")
        self._load_fft_data(filepath=data_filename)
        self._calculate_response_matrix()
        if self.response_matrix_data is not None and np.any(np.isfinite(self.response_matrix_data)):
            self._auto_save_response_matrix(save_dir, orm_timestr)
        else:
            print(f"[WARNING] Auto ORM for {device_name} produced no valid matrix data.")


    def _on_scan_complete(self, all_scan_data, scan_params, scan_error=False, user_stopped=False):
        """Callback function to update the GUI when the scan finishes."""
        self.is_scanning = False
        if scan_error:
            print("[WARNING] The device scan finished with errors. Data may be incomplete.")
        elif user_stopped:
            print("[INFO] Scan was stopped by user. Partial data may have been collected.")
        else:
            print("[SUCCESS] The device scan has finished.")

        # Auto-disable beam after successful completion if configured
        if (not scan_error and not user_stopped
                and self.beam_interlock_config.enabled
                and self.beam_interlock_config.disable_beam_on_completion):
            role = self.beam_interlock.get_effective_role(
                self._last_run_config.role if self._last_run_config else "OPERATOR")
            try:
                self.scanner.disable_beam(BEAM_CONTROL_DRF, role)
                print("[INFO] Post-scan beam disable sent")
            except Exception as e:
                print(f"[ERROR] Post-scan beam disable failed: {e}")
            if not self._closing:
                self._refresh_beam_status()

        if self._closing:
            return
        self.scan_progressbar.config(value=0)
        self._update_scan_buttons(is_scanning=False)

        # Render per-device health accumulated by the scan worker (#4). The
        # worker has finished by the time this main-thread callback runs.
        self._populate_device_health()

        # Save and analyze data for simultaneous scans
        if self.scan_mode.get() == "Simultaneous" and all_scan_data:
            timestr = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_dir = self.scan_data_path.get()
            is_partial = scan_error or user_stopped
            prefix = "PARTIAL_scan_data" if is_partial else "scan_data"

            data_filename = os.path.join(save_dir, f"{prefix}_{timestr}.csv")
            meta_filename = os.path.join(save_dir, f"{prefix}_{timestr}_meta.json")

            df = pd.DataFrame.from_records(all_scan_data)
            if not df.empty:
                try:
                    pivot_df = df.pivot_table(index='stamp', columns='name', values='data')
                    pivot_df.sort_index(inplace=True)
                    atomic_to_csv(pivot_df, data_filename, index_label='Timestamp')
                    if is_partial:
                        print(f"[INFO] Partial scan data saved ({len(all_scan_data)} points) to:\n{data_filename}")
                    else:
                        print(f"[SUCCESS] Scan data automatically saved to:\n{data_filename}")

                    atomic_write_text(meta_filename,
                                      json.dumps(scan_params, indent=4),
                                      encoding='utf-8')
                    print(f"[SUCCESS] Scan metadata saved to:\n{meta_filename}")

                    # Auto-analyze only for complete scans
                    if not is_partial:
                        # Now automate the FFT plot
                        self.tabControl.select(self.fft_tab)
                        self._load_fft_data(filepath=data_filename)
                        self.fft_device_listbox.select_set(0, tk.END) # Select all devices
                        self._plot_fft()

                        # Auto-calculate and save ORM if enabled
                        if self.auto_calc_orm.get():
                            print("[INFO] Auto-calculating Orbit Response Matrix...")
                            self._calculate_response_matrix()
                            if self.response_matrix_data is not None and np.any(np.isfinite(self.response_matrix_data)):
                                self._auto_save_response_matrix(save_dir, timestr)
                                self.tabControl.select(self.response_tab)
                            else:
                                print("[WARNING] Auto ORM calculation produced no valid matrix data.")
                except Exception as e:
                    print(f"[ERROR] Failed to save scan data: {e}")

    def _stop_scan(self):
        """Sets a flag to gracefully stop the scan loop."""
        self.stop_scan_flag.set()
        print("[INFO] Stopping scan - nominals will be restored by scan loop")

    def _update_scan_buttons(self, is_scanning):
        """Updates the state of the scan buttons."""
        self.start_scan_button.config(state="disabled" if is_scanning else "normal")
        self.stop_scan_button.config(state="normal" if is_scanning else "disabled")
        self.reading_status_label.config(text="Status: Scanning..." if is_scanning else "Status: Idle")
        self.start_stop_button.config(state="disabled" if is_scanning else "normal")

        # Disable the beam ON/OFF/Refresh buttons and the interlock-enable toggle
        # while a scan is running (#4/#5). These ACNET handlers must not be fired
        # mid-scan; they re-enable on completion.
        beam_state = "disabled" if is_scanning else "normal"
        for attr in ("beam_enable_button", "beam_disable_button",
                     "beam_refresh_button", "interlock_enabled_check"):
            widget = getattr(self, attr, None)
            if widget is not None:
                try:
                    widget.config(state=beam_state)
                except tk.TclError:
                    pass

    def _toggle_reading(self):
        """Starts or stops the data reading thread for live monitoring (non-scan)."""
        if self.is_scanning:
            print("[WARNING] Cannot start live reading while a scan is in progress.")
            return

        if self.is_reading:
            self.is_reading = False
            self.start_stop_button.config(text="Start Reading")
            self.reading_status_label.config(text="Status: Stopping...")
            threading.Thread(target=self.scanner.stop_thread, args=('live_reading',), daemon=True).start()
            return
        else:
            if not self.reading_devices:
                print("[WARNING] No reading devices selected.")
                return
            self.is_reading = True
            self._sync_dpm_node()

            event_str = self.acnet_event.get()
            devices_with_event = [f"{dev}{event_str}" for dev in self.reading_devices]
            self.plot_data.clear() 
            self.device_stats.clear()
            
            self._update_plot_selector()

            self.scanner.start_read_thread('live_reading', devices_with_event, self.data_queue, on_complete=lambda: self.after(0, self._on_read_complete))
            self.start_stop_button.config(text="Stop Reading")
            self.reading_status_label.config(text="Status: Reading...")

    def _on_read_complete(self):
        self.is_reading = False
        if self._closing:
            return
        self.start_stop_button.config(text="Start Reading")
        self.reading_status_label.config(text="Status: Idle")

    def _poll_data_queue(self):
        """Periodically checks the queue for new data and updates the GUI."""
        if self._closing:
            return
        had_data = False
        try:
            while True:
                new_data_chunk = self.data_queue.get_nowait()
                self._process_data(new_data_chunk)
                had_data = True
        except queue.Empty:
            pass
        if had_data:
            self._update_plots()
            self._update_stats_tab()
        if not self._closing:
            self.after(200, self._poll_data_queue)

    def _process_data(self, new_data):
        """Processes a chunk of data from the queue to update plots and stats."""
        for item in new_data:
            if not isinstance(item, dict): continue # Defensive check
            dev_name = item['name']
            if dev_name not in self.plot_data:
                self.plot_data[dev_name] = {'timestamps': [], 'values': []}
            self.plot_data[dev_name]['timestamps'].append(item['stamp'])
            self.plot_data[dev_name]['values'].append(item['data'])
            
            if len(self.plot_data[dev_name]['timestamps']) > 2000:
                 self.plot_data[dev_name]['timestamps'].pop(0)
                 self.plot_data[dev_name]['values'].pop(0)

            if dev_name in self.reading_devices:
                values = self.plot_data[dev_name]['values']
                if values:
                    self.device_stats[dev_name] = {
                        'min': np.min(values),
                        'max': np.max(values),
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }
        

    def _update_plots(self):
        """Updates the matplotlib plots to show only the selected device."""
        self.fig.clear()

        device_to_plot = self.plotted_device.get()
        if not device_to_plot or device_to_plot not in self.plot_data:
            self.plot_canvas.draw()
            return

        ax = self.fig.add_subplot(111)
        data = self.plot_data[device_to_plot]

        if data['timestamps']:
            ax.plot(data['timestamps'], data['values'], color=PALETTE['accent'], linewidth=1.6)
            ax.set_title(device_to_plot)
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=45)

        self._style_axis(ax)
        self.fig.tight_layout()
        self.plot_canvas.draw()

    def _on_plot_device_change(self, event=None):
            """Callback to redraw the plot when the user selects a new device."""
            self._update_plots()

    def _update_stats_tab(self):
            """Clears and repopulates the statistics table and summary."""
            for item in self.stats_tree.get_children():
                self.stats_tree.delete(item)
        
            std_devs = []
            for device in self.reading_devices:
                if device in self.device_stats:
                    stats = self.device_stats[device]
                    std_devs.append(stats['std'])
                    self.stats_tree.insert('', 'end', values=(
                        device,
                        f"{stats['min']:.4f}",
                        f"{stats['max']:.4f}",
                        f"{stats['mean']:.4f}",
                        f"{stats['std']:.4f}"
                    ))
        
            if std_devs:
                overall_rms = np.sqrt(np.mean(np.square(std_devs)))
                self.overall_rms_label.config(text=f"Overall RMS of Std Deviations: {overall_rms:.4f}")
            else:
                self.overall_rms_label.config(text="Overall RMS of Std Deviations: N/A")

    def _export_stats(self):
            """Exports the current statistics table to a CSV file."""
            if not self.device_stats:
                print("[WARNING] No statistics to export.")
                return

            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                title="Save Statistics to CSV"
            )
            if not filepath:
                return

            try:
                with open(filepath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    headers = self.stats_tree['columns']
                    writer.writerow(headers)
                    for item_id in self.stats_tree.get_children():
                        row = self.stats_tree.item(item_id, 'values')
                        writer.writerow(row)
                print(f"[SUCCESS] Statistics successfully exported to {filepath}")
            except Exception as e:
                print(f"[ERROR] Failed to export statistics: {e}")


    def _save_acquired_data(self):
            """Saves all currently held data (from live reading or scans) to a CSV file."""
            if not self.plot_data:
                print("[WARNING] No data to save.")
                return

            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                title="Save All Acquired Data to CSV"
            )
            if not filepath:
                return
            
            thread = threading.Thread(target=self._save_data_in_background, args=(filepath, copy.deepcopy(self.plot_data)))
            thread.start()
            self.reading_status_label.config(text="Status: Saving data...")

    def _save_data_in_background(self, filepath, data_to_save):
            """Handles the potentially long process of formatting and saving data."""
            try:
                all_data_list = []
                for device, data_dict in data_to_save.items():
                    for i, timestamp in enumerate(data_dict['timestamps']):
                        all_data_list.append({'stamp': timestamp, 'name': device, 'data': data_dict['values'][i]})
            
                if all_data_list:
                    df = pd.DataFrame.from_records(all_data_list)
                    pivot_df = df.pivot_table(index='stamp', columns='name', values='data')
                    pivot_df.sort_index(inplace=True)
                    atomic_to_csv(pivot_df, filepath, index_label='Timestamp')
                    self.last_analysis_source = filepath

                    print(f"[SUCCESS] All acquired data saved to {filepath}")
            except Exception as e:
                print(f"[ERROR] Failed to save data: {e}")
            finally:
                if not self._closing:
                    self.after(0, self.reading_status_label.config, {'text': 'Status: Idle'})


    def _load_fft_data(self, filepath=None):
            """Loads a CSV file for FFT analysis. If no filepath is provided, opens a dialog."""
            if filepath is None:
                filepath = filedialog.askopenfilename(
                    title="Load Scan Data File",
                    filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
                )
            if not filepath:
                return

            self._aligned_fft_data = None

            try:
                self.fft_data = pd.read_csv(filepath, index_col='Timestamp', parse_dates=True)
                self.fft_device_listbox.delete(0, tk.END)
                for col in self.fft_data.columns:
                    self.fft_device_listbox.insert(tk.END, col)
            
                # Also try to load the metadata file
                meta_path = self._meta_path_for(filepath)
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        self.fft_scan_params = json.load(f)
                    print(f"[SUCCESS] Successfully loaded {filepath} and its metadata.")
                else:
                    self.fft_scan_params = None
                    print(f"[SUCCESS] Successfully loaded {filepath}. No metadata file found.")

                self._aligned_fft_data = self._prepare_aligned_fft_data(self.fft_data, self.fft_scan_params)
                self.last_analysis_source = filepath

            except Exception as e:
                print(f"[ERROR] Failed to load or parse file: {e}")


    def _prepare_aligned_fft_data(self, data_frame, scan_params):
            """Ensure setting and readback samples share a common timeline for FFT analysis."""
            if data_frame is None or data_frame.empty:
                return None

            df = data_frame.copy()
            df.sort_index(inplace=True)
            df = df.loc[df.index.notnull()]

            setting_devices = []
            if scan_params:
                setting_devices = [
                    entry.get('device')
                    for entry in scan_params
                    if isinstance(entry, dict) and entry.get('device')
                ]

            # Resolve setting device column names - supports both:
            # - New format: plain device names (readbacks)
            # - Legacy format: device names with .SETTING suffix
            alias_map = {}
            existing_setting_devices = []
            missing_setting_devices = []
            for device_name in setting_devices:
                resolved = None
                # First try exact match (new format with readbacks)
                if device_name in df.columns:
                    resolved = device_name
                else:
                    # Fallback for legacy data with .SETTING suffix
                    suffix_alt = f"{device_name}.SETTING"
                    if suffix_alt in df.columns:
                        resolved = suffix_alt
                    elif device_name.endswith('.SETTING'):
                        base_name = device_name[:-8]
                        if base_name in df.columns:
                            resolved = base_name
                if resolved:
                    alias_map[device_name] = resolved
                    if resolved not in existing_setting_devices:
                        existing_setting_devices.append(resolved)
                else:
                    missing_setting_devices.append(device_name)

            self._setting_device_aliases = alias_map

            if missing_setting_devices:
                preview = ', '.join(missing_setting_devices[:3])
                if len(missing_setting_devices) > 3:
                    preview += ', ...'
                print(f"[WARNING] Setting devices missing from data columns: {preview}")

            readback_cols = [col for col in df.columns
                           if col not in existing_setting_devices
                           and not col.endswith('.SETTING')]

            if existing_setting_devices:
                df[existing_setting_devices] = df[existing_setting_devices].ffill()

            if readback_cols:
                df = df.dropna(subset=readback_cols, how='all')

            df = df.dropna(how='all')
            df = df.loc[~df.index.duplicated(keep='last')]

            return df


    def _plot_fft(self):
            """Calculates and plots the FFT for selected devices."""
            if self.fft_data is None:
                print("[WARNING] No Data: Please load a scan data file first.")
                return

            selected_indices = self.fft_device_listbox.curselection()
            if not selected_indices:
                print("[WARNING] No Selection: Please select one or more devices to plot.")
                return

            self.fft_fig.clear()
            ax = self.fft_fig.add_subplot(111)
            self.fft_lines = []  # Store plotted lines for interactivity
            color_cycle = ['#38bdf8', '#f97316', '#22c55e', '#facc15', '#a855f7', '#f472b6']

            for idx, i in enumerate(selected_indices):
                device_name = self.fft_device_listbox.get(i)
                column = self.fft_data[device_name].dropna()
                signal = column - column.mean()

                N = len(signal)
                if N < 2:
                    continue

                if pd.api.types.is_datetime64_any_dtype(signal.index):
                    total_duration = (signal.index[-1] - signal.index[0]).total_seconds()
                    time_diffs = signal.index.to_series().diff().dt.total_seconds()
                    T = time_diffs.mean()
                else:
                    T = 1.0
                    total_duration = N * T

                if T == 0 or pd.isna(T) or total_duration == 0:
                    print(f"[WARNING] Cannot calculate valid timing for {device_name}. Skipping.")
                    continue

                yf = fft(signal.values)
                xf_hz = fftfreq(N, T)[:N // 2]
                xf_periods = xf_hz * total_duration
                normalized_yf = 2.0 / N * np.abs(yf[0:N // 2])

                color = color_cycle[idx % len(color_cycle)]
                line, = ax.plot(xf_periods, normalized_yf, label=device_name, color=color, linewidth=1.4)
                self.fft_lines.append(line)

            self.fft_annot = ax.annotate(
                "",
                xy=(0, 0),
                xytext=(20, 20),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc=PALETTE['surface'], ec=PALETTE['border']),
                arrowprops=dict(arrowstyle="->", color=PALETTE['accent']),
                color=PALETTE['text'],
            )
            self.fft_annot.set_visible(False)
            self.fft_canvas.mpl_connect("motion_notify_event", self._hover_fft)

            ax.set_title("FFT Spectrum")
            ax.set_xlabel("Number of Periods")
            ax.set_ylabel("Normalized Amplitude")
            self._style_axis(ax)

            legend = ax.legend()
            if legend:
                legend.get_frame().set_facecolor(PALETTE['surface'])
                legend.get_frame().set_edgecolor(PALETTE['border'])
                for text_item in legend.get_texts():
                    text_item.set_color(PALETTE['text'])

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                self.fft_fig.tight_layout()
            self.fft_canvas.draw()

    def _hover_fft(self, event):
            """Show an annotation when hovering over a data point on the FFT plot."""
            vis = self.fft_annot.get_visible()
            if event.inaxes == self.fft_annot.axes:
                found_target = False
                for line in self.fft_lines:
                    cont, ind = line.contains(event)
                    if cont and len(ind["ind"]) > 0:
                        pos = line.get_xydata()[ind["ind"][0]]
                        self.fft_annot.xy = pos
                        text = f"Periods={pos[0]:.2f}\nAmp={pos[1]:.3f}"
                        self.fft_annot.set_text(text)
                        self.fft_annot.get_bbox_patch().set_alpha(0.4)
                        self.fft_annot.set_visible(True)
                        self.fft_canvas.draw_idle()
                        found_target = True
                        break
            
                if not found_target and vis:
                    self.fft_annot.set_visible(False)
                    self.fft_canvas.draw_idle()


    def _start_rms_measurement(self):
            """Starts the process to measure baseline RMS."""
            if self.is_reading or self.is_scanning:
                print("[WARNING] Cannot measure RMS while another process is active.")
                return
            self._sync_dpm_node()
        
            all_devices = sorted(list(set(self.setting_devices + self.reading_devices)))
            if not all_devices:
                print("[WARNING] No Devices: Please load a setting or reading device list first.")
                return

            try:
                num_samples = self.rms_samples.get()
                if num_samples <= 1:
                    print("[ERROR] Number of samples must be greater than 1.")
                    return
            except tk.TclError:
                print("[ERROR] Invalid number of samples.")
                return

            self.rms_status_label.config(text="Status: Measuring...")
            self.rms_button.config(state="disabled")
            self.rms_progressbar.config(maximum=num_samples, value=0)
            # Pre-capture tkinter var before starting background thread
            acnet_event = self.acnet_event.get()
            thread = threading.Thread(target=self._measure_rms_thread_target, args=(all_devices, num_samples, acnet_event))
            thread.start()

    def _measure_rms_thread_target(self, devices, num_samples, acnet_event=None):
            """Thread target for collecting data and calculating RMS."""
            try:
                if acnet_event is None:
                    acnet_event = ""
                # Use readbacks for all devices (not .SETTING for correctors)
                drf_list = [f"{dev}{acnet_event}" for dev in devices]
            
                collected_data = {dev: [] for dev in devices}

                consecutive_timeouts = 0
                for i in range(num_samples):
                    if self._closing:
                        break
                    # Update status in the main thread
                    self.after(0, self.rms_status_label.config, {'text': f'Status: Acquiring sample {i+1}/{num_samples}'})

                    data_from_step = self.scanner.read_once_on_event(drf_list)

                    # Track consecutive timeouts (all-None results)
                    if data_from_step and all(item is None for item in data_from_step):
                        consecutive_timeouts += 1
                        print(f"[WARNING] RMS sample {i+1}: no data received "
                              f"(timeout {consecutive_timeouts}/{ACNET_MAX_CONSECUTIVE_TIMEOUTS})")
                        if consecutive_timeouts >= ACNET_MAX_CONSECUTIVE_TIMEOUTS:
                            print(f"[ERROR] {ACNET_MAX_CONSECUTIVE_TIMEOUTS} consecutive timeouts — "
                                  f"aborting RMS measurement (beam may be off or ACNET connection lost)")
                            break
                    else:
                        consecutive_timeouts = 0

                    for item in data_from_step:
                        if item and item['name'] in collected_data:
                            collected_data[item['name']].append(item['data'])
                    if not self._closing:
                        self.after(0, self.rms_progressbar.step)
            
                # Calculate RMS (standard deviation)
                rms_results = {}
                for device, values in collected_data.items():
                    if len(values) > 1:
                        rms_results[device] = np.std(values)
                    else:
                        rms_results[device] = 0.0

                # Schedule GUI update
                if not self._closing:
                    self.after(0, self._update_rms_table, rms_results)

            except Exception as e:
                print(f"[ERROR] Failed during RMS measurement: {e}")
            finally:
                if not self._closing:
                    self.after(0, self.rms_button.config, {'state': 'normal'})
                    self.after(0, self.rms_status_label.config, {'text': ''})
                    self.after(0, self.rms_progressbar.config, {'value': 0})


    def _update_rms_table(self, rms_results):
            """Populates the RMS table with calculated values."""
            if self._closing:
                return
            for item in self.rms_tree.get_children():
                self.rms_tree.delete(item)

            for device, rms_val in rms_results.items():
                self.rms_tree.insert('', 'end', values=(device, f"{rms_val:.4f}"))
            print("[SUCCESS] Baseline RMS measurement complete.")

    def _export_rms_data(self):
            """Exports the RMS baseline data to a CSV file."""
            if not self.rms_tree.get_children():
                print("[WARNING] No RMS data to export. Please run a baseline RMS measurement first.")
                return

            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Save RMS Data to CSV",
                initialfile=f"baseline_rms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )

            if not filepath:
                return

            try:
                with open(filepath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)

                    # Write header
                    writer.writerow(['Device', 'RMS Fluctuation'])

                    # Write data rows
                    for item_id in self.rms_tree.get_children():
                        row = self.rms_tree.item(item_id, 'values')
                        writer.writerow(row)

                print(f"[SUCCESS] RMS data successfully exported to {filepath}")
            except Exception as e:
                print(f"[ERROR] Failed to export RMS data: {e}")

    def _get_color_for_value(self, value, vmax):
        """Calculates a color for a heatmap based on a value and a maximum."""
        if vmax == 0:
            return "#FFFFFF"  # White

        norm_val = value / vmax

        if norm_val > 0:
            # White to Red
            g = int(255 * (1 - norm_val))
            b = int(255 * (1 - norm_val))
            return f'#FF{g:02x}{b:02x}'
        else:
            # White to Blue
            r = int(255 * (1 - abs(norm_val)))
            g = int(255 * (1 - abs(norm_val)))
            return f'#{r:02x}{g:02x}FF'

    def _get_contrast_text_color(self, hex_color):
        """Return a contrasting text color for a given background color."""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            return PALETTE['text']
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        brightness = (0.299 * r) + (0.587 * g) + (0.114 * b)
        return PALETTE['background'] if brightness > 150 else PALETTE['text']

    def _calculate_response_matrix(self):
        """Calculate response matrix and uncertainties via FFT-based transfer functions."""
        if self.fft_data is None or self.fft_scan_params is None:
            print("[WARNING] Please load a scan data file with corresponding metadata first.")
            return

        aligned_data = self._prepare_aligned_fft_data(self.fft_data, self.fft_scan_params)
        if aligned_data is None or aligned_data.empty:
            print("[WARNING] Unable to align scan data with metadata for response matrix computation.")
            return

        self._aligned_fft_data = aligned_data

        alias_map = getattr(self, '_setting_device_aliases', {}) or {}
        column_names = set(aligned_data.columns)
        setting_entries = []
        skipped_setting_devices = []
        if self.fft_scan_params:
            for entry in self.fft_scan_params:
                if not isinstance(entry, dict):
                    continue
                raw_name = entry.get('device')
                if not raw_name:
                    continue
                actual_name = alias_map.get(raw_name)
                if not actual_name:
                    # First try exact match (new format with readbacks)
                    if raw_name in column_names:
                        actual_name = raw_name
                    else:
                        # Fallback for legacy data with .SETTING suffix
                        suffix_alt = f"{raw_name}.SETTING"
                        if suffix_alt in column_names:
                            actual_name = suffix_alt
                        elif raw_name.endswith('.SETTING') and raw_name[:-8] in column_names:
                            actual_name = raw_name[:-8]
                if actual_name:
                    setting_entries.append((raw_name, actual_name, entry))
                else:
                    skipped_setting_devices.append(raw_name)

        if skipped_setting_devices:
            ordered = list(dict.fromkeys(skipped_setting_devices))
            preview = ', '.join(ordered[:3])
            if len(ordered) > 3:
                preview += ', ...'
            print(f"[WARNING] Skipping response computation for missing setting devices: {preview}")

        ordered_entries = []
        seen_actual = set()
        for raw_name, actual_name, entry in setting_entries:
            if actual_name in seen_actual:
                continue
            seen_actual.add(actual_name)
            ordered_entries.append((raw_name, actual_name, entry))
        setting_entries = ordered_entries

        if not setting_entries:
            print("[WARNING] No setting devices yielded usable FFT data.")
            return

        self.response_setting_devices = [actual for (_, actual, _) in setting_entries]
        self.response_setting_labels = [self._format_device_label(raw) for (raw, _, _) in setting_entries]

        # Filter out setting device columns, .SETTING columns, and BPMs with all-zero readings
        self.response_reading_devices = [
            col for col in aligned_data.columns
            if col not in self.response_setting_devices
            and not col.endswith('.SETTING')
            and not (aligned_data[col].dropna().empty or np.all(aligned_data[col].dropna().values == 0))
        ]
        self.response_reading_labels = [self._format_device_label(name) for name in self.response_reading_devices]

        self._horizontal_indices = [idx for idx, name in enumerate(self.response_reading_devices) if 'BPH' in name.upper()]
        self._vertical_indices = [idx for idx, name in enumerate(self.response_reading_devices) if 'BPV' in name.upper()]

        if not self.response_reading_devices:
            print("[WARNING] No reading devices available after alignment.")
            return

        def _index_to_seconds(idx):
            if pd.api.types.is_datetime64_any_dtype(idx):
                conv = idx.tz_convert('UTC') if getattr(idx, 'tz', None) is not None else idx
                return conv.asi8.astype(np.float64) / 1e9

            # Try to parse as datetime if index contains strings
            try:
                return idx.to_numpy(dtype=np.float64, copy=False)
            except (ValueError, TypeError):
                # If direct conversion fails, try parsing as datetime
                try:
                    dt_idx = pd.to_datetime(idx, format='mixed', utc=True)
                    return dt_idx.asi8.astype(np.float64) / 1e9
                except Exception as e:
                    print(f"[ERROR] Could not convert index to seconds. Index dtype: {idx.dtype}, sample: {idx[0] if len(idx) > 0 else 'empty'}")
                    print(f"[ERROR] Exception: {e}")
                    raise

        corrector_info = {}
        usable_setting_entries = []
        for raw_name, s_dev, entry in setting_entries:
            series = aligned_data[s_dev].dropna()
            if series.empty or len(series) < 2:
                continue

            timestamps = _index_to_seconds(series.index)
            if timestamps.size < 2:
                continue
            timestamps = timestamps - timestamps[0]
            deltas = np.diff(timestamps)
            positive_deltas = deltas[deltas > 0]
            if positive_deltas.size == 0:
                continue
            dt = float(np.median(positive_deltas))
            if dt <= 0:
                continue

            signal = series.values - series.values.mean()
            N = len(signal)
            if N < 2:
                continue

            corr_fft = fft(signal)
            freqs = fftfreq(N, dt)
            half = max(N // 2, 1)
            freqs_pos = freqs[:half]
            amps = (2.0 / N) * np.abs(corr_fft[:half])
            if amps.size <= 1:
                continue
            amps[0] = 0.0
            idx_max = int(np.argmax(amps[1:]) + 1)
            if idx_max >= amps.size:
                continue
            dominant_freq = freqs_pos[idx_max]
            if abs(dominant_freq) < 1e-9:
                continue

            corr_component = (2.0 / N) * corr_fft[idx_max]
            corr_amp = float(np.abs(corr_component))
            amps_noise = amps.copy()
            for offset in range(-2, 3):  # -2, -1, 0, +1, +2
                idx = idx_max + offset
                if 0 <= idx < len(amps_noise):
                    amps_noise[idx] = 0.0
            corr_noise = float(np.sqrt(np.mean(amps_noise**2)))

            corrector_info[s_dev] = {
                'frequency': float(abs(dominant_freq)),
                'noise': corr_noise,
                'dt': dt
            }
            usable_setting_entries.append((raw_name, s_dev, entry))

        if not usable_setting_entries:
            print("[WARNING] No valid setting device spectra found for response matrix computation.")
            return

        setting_entries = usable_setting_entries
        self.response_setting_devices = [actual for (_, actual, _) in setting_entries]
        self.response_setting_labels = [self._format_device_label(raw) for (raw, _, _) in setting_entries]

        # Collect corrector drive frequencies (Hz) for noise exclusion
        corrector_frequencies = {s_dev: info['frequency'] for s_dev, info in corrector_info.items()}

        # Calculate BPM noise from full data, excluding corrector frequency bins (like ORM.py)
        bpm_noise = {}
        for r_dev in self.response_reading_devices:
            series = aligned_data[r_dev].dropna()
            if series.empty or len(series) < 2:
                continue

            # Compute this BPM's own dt for correct frequency-to-bin mapping
            timestamps = _index_to_seconds(series.index)
            if timestamps.size < 2:
                continue
            timestamps = timestamps - timestamps[0]
            deltas = np.diff(timestamps)
            positive_deltas = deltas[deltas > 0]
            if positive_deltas.size == 0:
                continue
            bpm_dt = float(np.median(positive_deltas))
            if bpm_dt <= 0:
                continue

            data = series.values  # NO mean subtraction for noise calculation (like ORM.py)
            N = len(data)
            bpm_fft = fft(data)
            amps = (2.0 / N) * np.abs(bpm_fft[:N//2])
            if amps.size <= 1:
                continue
            amps[0] = 0.0

            # Exclude corrector frequency bins ±2 (frequency-based lookup per BPM grid)
            for freq_hz in corrector_frequencies.values():
                cidx = int(round(freq_hz * N * bpm_dt))
                for offset in range(-2, 3):
                    idx = cidx + offset
                    if 0 <= idx < len(amps):
                        amps[idx] = 0.0

            bpm_noise[r_dev] = float(np.sqrt(np.mean(amps**2)))

        self.response_bpm_noise = bpm_noise
        self.response_corrector_noise = {dev: info['noise'] for dev, info in corrector_info.items()}
        self.response_frequency_map = {dev: info['frequency'] for dev, info in corrector_info.items()}

        # DIAGNOSTIC: Print noise values for comparison
        print("\n[DIAGNOSTIC] Noise values calculated:")
        print("\nCorrector drive frequencies:")
        for dev, freq in corrector_frequencies.items():
            print(f"  {dev}: {freq:.6f} Hz")
        print("\nBPM noise (first 5):")
        for i, (dev, noise) in enumerate(list(bpm_noise.items())[:5]):
            print(f"  {dev}: {noise:.6e}")
        print("\nCorrector noise (all):")
        for dev, info in corrector_info.items():
            print(f"  {dev}: {info['noise']:.6e}")

        # Print detailed calculation for first BPM to debug
        if self.response_reading_devices:
            first_bpm = self.response_reading_devices[0]
            series = aligned_data[first_bpm].dropna()
            timestamps = _index_to_seconds(series.index)
            timestamps = timestamps - timestamps[0]
            deltas = np.diff(timestamps)
            positive_deltas = deltas[deltas > 0]
            dbg_dt = float(np.median(positive_deltas)) if positive_deltas.size > 0 else 0.0
            data = series.values
            N = len(data)
            bpm_fft = fft(data)
            amps = (2.0 / N) * np.abs(bpm_fft[:N//2])
            print(f"\n[DEBUG] First BPM ({first_bpm}) noise calculation:")
            print(f"  Data length: {N}, dt: {dbg_dt:.6f} s")
            print(f"  FFT bins (half): {len(amps)}")
            print(f"  DC amplitude (bin 0): {amps[0]:.6e}")
            excluded_bins = set([0])
            for freq_hz in corrector_frequencies.values():
                cidx = int(round(freq_hz * N * dbg_dt))
                for offset in range(-2, 3):
                    idx = cidx + offset
                    if 0 <= idx < len(amps):
                        excluded_bins.add(idx)
            print(f"  Excluded bins: {sorted(excluded_bins)}")
            amps_copy = amps.copy()
            for idx in excluded_bins:
                if 0 <= idx < len(amps_copy):
                    amps_copy[idx] = 0.0
            print(f"  RMS of remaining bins: {np.sqrt(np.mean(amps_copy**2)):.6e}")

        num_read = len(self.response_reading_devices)
        num_set = len(setting_entries)
        self.response_matrix_data = np.full((num_read, num_set), np.nan)
        self.response_error_matrix = np.full((num_read, num_set), np.nan)
        self.response_bpm_amplitudes = np.full((num_read, num_set), np.nan)
        self.response_corrector_amplitudes = np.full((num_read, num_set), np.nan)

        for c_idx, (raw_name, s_dev, entry) in enumerate(setting_entries):
            for r_idx, r_dev in enumerate(self.response_reading_devices):
                working = aligned_data[[r_dev, s_dev]].dropna()
                if working.empty or len(working) < 2:
                    continue

                timestamps = _index_to_seconds(working.index)
                if timestamps.size < 2:
                    continue
                timestamps = timestamps - timestamps[0]
                deltas = np.diff(timestamps)
                positive_deltas = deltas[deltas > 0]
                if positive_deltas.size == 0:
                    continue
                dt = float(np.median(positive_deltas))
                if dt <= 0:
                    continue

                response_values = working[r_dev].values - working[r_dev].values.mean()
                drive_values = working[s_dev].values - working[s_dev].values.mean()
                N = len(working)
                if N < 2:
                    continue

                response_fft = fft(response_values)
                drive_fft = fft(drive_values)
                freqs = fftfreq(N, dt)
                half = max(N // 2, 1)

                # FIX: Find the bin with maximum amplitude in the corrector FFT (like ORM.py)
                # Don't try to match pre-calculated frequency - find actual max in this subset
                drive_amps = np.abs(drive_fft[:half])
                if drive_amps.size <= 1:
                    continue
                # Skip DC component (index 0), find max in rest
                freq_idx = int(np.argmax(drive_amps[1:]) + 1)
                if not (0 <= freq_idx < half):
                    continue

                response_component = (2.0 / N) * response_fft[freq_idx]
                drive_component = (2.0 / N) * drive_fft[freq_idx]

                bpm_amp = float(np.abs(response_component))
                corr_amp = float(np.abs(drive_component))
                self.response_bpm_amplitudes[r_idx, c_idx] = bpm_amp
                self.response_corrector_amplitudes[r_idx, c_idx] = corr_amp

                if corr_amp <= 1e-12:
                    continue

                transfer = response_component / drive_component
                matrix_value = float(transfer.real)
                self.response_matrix_data[r_idx, c_idx] = matrix_value

                # Use pre-calculated noise values (like ORM.py)
                eb = bpm_noise.get(r_dev, 0.0)
                ec = corrector_info[s_dev]['noise']

                if corr_amp > 0:
                    # Error propagation per the manual (eq:errorprop): each term
                    # carries a 1/2 factor (the FFT amplitude estimate variance).
                    error_term = (eb ** 2) / (2.0 * corr_amp ** 2)
                    error_term += ((bpm_amp ** 2) / (2.0 * corr_amp ** 4)) * (ec ** 2)
                    self.response_error_matrix[r_idx, c_idx] = float(np.sqrt(error_term)) if error_term > 0 else 0.0

        # DIAGNOSTIC: Print error matrix statistics
        finite_errors = self.response_error_matrix[np.isfinite(self.response_error_matrix)]
        print("\n[DIAGNOSTIC] Error matrix statistics:")
        print(f"  Filled entries: {len(finite_errors)} / {self.response_error_matrix.size}")
        if len(finite_errors) > 0:
            print(f"  Mean error: {np.mean(finite_errors):.6e}")
            print(f"  Max error: {np.max(finite_errors):.6e}")
            positive_errors = finite_errors[finite_errors > 0]
            print(f"  Min error (positive): {np.min(positive_errors):.6e}" if len(positive_errors) > 0 else "  Min error (positive): N/A")
        print("  First few error values:")
        for i in range(min(3, self.response_error_matrix.shape[0])):
            for j in range(min(3, self.response_error_matrix.shape[1])):
                val = self.response_error_matrix[i, j]
                if np.isfinite(val) and val > 0:
                    print(f"    [{i},{j}] = {val:.6e} (BPM_amp={self.response_bpm_amplitudes[i,j]:.6e}, Corr_amp={self.response_corrector_amplitudes[i,j]:.6e})")

        self._update_response_matrix_plot()

        positive_errs = self.response_error_matrix[(np.isfinite(self.response_error_matrix)) & (self.response_error_matrix > 0)]
        if len(positive_errs) > 0:
            mean_err = np.mean(positive_errs)
            print(f"[SUCCESS] Response matrix calculation complete. Mean entry uncertainty: {mean_err:.4e}")
        else:
            print("[SUCCESS] Response matrix calculation complete.")

    def _plot_device_noise(self):
        """Plot device noise levels (one value per device) like ORM.py's plotNoiseLevels()."""
        self.response_ax_h.clear()
        self.response_ax_v.clear()
        self._style_axis(self.response_ax_h)
        self._style_axis(self.response_ax_v)

        # Clear any existing colorbars
        for attr in ('response_colorbar_h', 'response_colorbar_v'):
            cbar = getattr(self, attr, None)
            if cbar is not None:
                try:
                    cbar.remove()
                except Exception:
                    pass
                setattr(self, attr, None)

        # Get noise values from correctors and BPMs
        corrector_noise = getattr(self, 'response_corrector_noise', {})
        bpm_noise = getattr(self, 'response_bpm_noise', {})

        if not corrector_noise and not bpm_noise:
            self.response_ax_h.text(0.5, 0.5, 'No noise data available.\nCalculate response matrix first.',
                                   ha='center', va='center', color=PALETTE['muted_text'], transform=self.response_ax_h.transAxes)
            self.response_ax_h.set_xticks([])
            self.response_ax_h.set_yticks([])
            self.response_ax_v.set_visible(False)
            self.response_heatmap_canvas.draw()
            return

        # Combine all devices and their noise values (like ORM.py line 889-890)
        all_devices = list(corrector_noise.keys()) + list(bpm_noise.keys())
        noise_vals = [corrector_noise.get(dev, bpm_noise.get(dev, 0)) for dev in all_devices]

        # Split into correctors and BPMs for separate plots
        corrector_devices = list(corrector_noise.keys())
        corrector_vals = list(corrector_noise.values())
        bpm_devices = list(bpm_noise.keys())
        bpm_vals = list(bpm_noise.values())

        # Plot correctors on left axis
        if corrector_devices:
            x_positions = np.arange(len(corrector_devices))
            colors_corr = matplotlib.cm.rainbow(np.linspace(0, 1, len(corrector_devices)))
            self.response_ax_h.bar(x_positions, corrector_vals, color=colors_corr)
            self.response_ax_h.set_title('Corrector Noise Levels (DC Removed)', color=PALETTE['text'])
            self.response_ax_h.set_xlabel('Correctors', color=PALETTE['text'])
            self.response_ax_h.set_ylabel('Noise (RMS, ignoring DC)', color=PALETTE['text'])
            self.response_ax_h.set_xticks(x_positions)
            self.response_ax_h.set_xticklabels(corrector_devices, rotation=45, ha='right')
            self.response_ax_h.tick_params(colors=PALETTE['muted_text'])
        else:
            self.response_ax_h.text(0.5, 0.5, 'No corrector noise data',
                                   ha='center', va='center', color=PALETTE['muted_text'], transform=self.response_ax_h.transAxes)
            self.response_ax_h.set_xticks([])
            self.response_ax_h.set_yticks([])

        # Plot BPMs on right axis
        if bpm_devices:
            x_positions = np.arange(len(bpm_devices))
            colors_bpm = matplotlib.cm.rainbow(np.linspace(0, 1, len(bpm_devices)))
            self.response_ax_v.bar(x_positions, bpm_vals, color=colors_bpm)
            self.response_ax_v.set_title('BPM Noise Levels (DC & Corrector Peaks Removed)', color=PALETTE['text'])
            self.response_ax_v.set_xlabel('BPMs', color=PALETTE['text'])
            self.response_ax_v.set_ylabel('Noise (RMS, ignoring DC & peaks)', color=PALETTE['text'])
            self.response_ax_v.set_xticks(x_positions)
            self.response_ax_v.set_xticklabels(bpm_devices, rotation=45, ha='right')
            self.response_ax_v.tick_params(colors=PALETTE['muted_text'])
        else:
            self.response_ax_v.text(0.5, 0.5, 'No BPM noise data',
                                   ha='center', va='center', color=PALETTE['muted_text'], transform=self.response_ax_v.transAxes)
            self.response_ax_v.set_xticks([])
            self.response_ax_v.set_yticks([])

        self.response_ax_v.set_visible(True)
        self.response_fig.tight_layout()
        self.response_heatmap_canvas.draw()



    def _update_response_matrix_plot(self):
        """Render the response matrix heatmaps based on the selected display mode."""
        if not hasattr(self, 'response_fig'):
            return

        self.response_ax_h.clear()
        self.response_ax_v.clear()
        self._style_axis(self.response_ax_h)
        self._style_axis(self.response_ax_v)

        if self.response_matrix_data is None or not self.response_setting_devices or not self.response_reading_devices:
            for ax in (self.response_ax_h, self.response_ax_v):
                ax.text(0.5, 0.5, 'No response matrix calculated yet.',
                        ha='center', va='center', color=PALETTE['muted_text'], transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            colorbar_store = getattr(self, 'response_colorbars', None)
            colorbar_axes = getattr(self, 'response_colorbar_axes', None)
            for attr, plane in (('response_colorbar_h', 'Horizontal'), ('response_colorbar_v', 'Vertical')):
                cbar = getattr(self, attr, None)
                if cbar is not None:
                    try:
                        cbar.remove()
                    except Exception:
                        pass
                    setattr(self, attr, None)
                if isinstance(colorbar_store, dict):
                    colorbar_store[plane] = None
                if isinstance(colorbar_axes, dict):
                    cax = colorbar_axes.get(plane)
                    if cax is not None and getattr(cax, 'figure', None) is not None:
                        cax.cla()
                        cax.set_visible(False)
            if isinstance(colorbar_store, dict):
                self.response_colorbars = colorbar_store
            self.response_heatmap_canvas.draw()
            return

        mode = getattr(self, 'response_display_mode', None)
        mode_value = mode.get() if mode is not None else 'Gain (Real)'

        # Handle Device Noise mode separately (1D bar plot instead of 2D heatmap)
        if mode_value == 'Device Noise':
            self._plot_device_noise()
            return

        if mode_value == 'Uncertainty (sigma)' and self.response_error_matrix is not None:
            matrix = self.response_error_matrix
            title_suffix = 'Uncertainty (sigma)'
            cmap = 'magma'
            colorbar_label = 'sigma (abs)'
            is_uncertainty = True
        else:
            matrix = self.response_matrix_data
            title_suffix = 'Real Gain'
            cmap = 'coolwarm'
            colorbar_label = 'Delta Reading / Delta Setting'
            is_uncertainty = False

        matrix = np.asarray(matrix, dtype=float) if matrix is not None else np.zeros((0, 0))

        plane_map = {
            'Horizontal': self._horizontal_indices,
            'Vertical': self._vertical_indices,
        }

        axes_positions = getattr(self, 'response_axes_positions', {})
        colorbar_axes = getattr(self, 'response_colorbar_axes', {})
        colorbar_store = getattr(self, 'response_colorbars', {})
        if not isinstance(colorbar_store, dict):
            colorbar_store = {}

        for ax, attr, plane in ((self.response_ax_h, 'response_colorbar_h', 'Horizontal'),
                                 (self.response_ax_v, 'response_colorbar_v', 'Vertical')):
            base_position = axes_positions.get(plane) if isinstance(axes_positions, dict) else None
            if base_position is not None:
                ax.set_position(base_position)

            existing_cbar = getattr(self, attr, None)
            if existing_cbar is not None:
                try:
                    existing_cbar.remove()
                except Exception:
                    pass
                setattr(self, attr, None)

            if isinstance(colorbar_store, dict):
                stored_cbar = colorbar_store.get(plane)
                if stored_cbar is not None and stored_cbar is not existing_cbar:
                    try:
                        stored_cbar.remove()
                    except Exception:
                        pass
                colorbar_store[plane] = None

            cax = colorbar_axes.get(plane) if isinstance(colorbar_axes, dict) else None
            if cax is None or getattr(cax, 'figure', None) is None:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.08)
                cax.set_facecolor(PALETTE['card'])
                cax.tick_params(colors=PALETTE['muted_text'])
                for spine in cax.spines.values():
                    spine.set_edgecolor(PALETTE['border'])
                if isinstance(colorbar_axes, dict):
                    colorbar_axes[plane] = cax
            else:
                cax.cla()
                cax.set_facecolor(PALETTE['card'])
                cax.tick_params(colors=PALETTE['muted_text'])
                for spine in cax.spines.values():
                    spine.set_edgecolor(PALETTE['border'])
            if cax is not None:
                cax.set_visible(True)

            indices = plane_map[plane]
            if not indices:
                ax.text(0.5, 0.5, f'No {plane.lower()} BPMs available.',
                        ha='center', va='center', color=PALETTE['muted_text'], transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                if cax is not None:
                    cax.set_visible(False)
                continue

            sub_matrix = matrix[indices, :]
            if sub_matrix.size == 0:
                ax.text(0.5, 0.5, 'No data available.',
                        ha='center', va='center', color=PALETTE['muted_text'], transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                if cax is not None:
                    cax.set_visible(False)
                continue

            if is_uncertainty:
                vmin = 0.0
                try:
                    vmax_candidate = float(np.nanmax(sub_matrix))
                except ValueError:
                    vmax_candidate = 0.0
                vmax = vmax_candidate if np.isfinite(vmax_candidate) and vmax_candidate > 0 else 1.0
            else:
                try:
                    abs_max = float(np.nanmax(np.abs(sub_matrix)))
                except ValueError:
                    abs_max = 0.0
                abs_max = abs_max if np.isfinite(abs_max) and abs_max != 0.0 else 1.0
                vmin, vmax = -abs_max, abs_max

            im = ax.imshow(sub_matrix, aspect='auto', origin='upper', cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f'{plane} BPMs - {title_suffix}', color=PALETTE['text'])
            ax.set_xlabel('Setting Devices', color=PALETTE['muted_text'])
            ax.set_ylabel('Reading Devices', color=PALETTE['muted_text'])

            x_labels = self.response_setting_labels or [self._format_device_label(dev) for dev in self.response_setting_devices]
            valid_indices = [idx for idx in indices if idx < len(self.response_reading_labels)]
            y_labels = [self.response_reading_labels[idx] for idx in valid_indices]

            ax.set_xticks(range(len(self.response_setting_devices)))
            ax.set_xticklabels(x_labels, rotation=90, ha='right')
            ax.set_yticks(range(len(valid_indices)))
            ax.set_yticklabels(y_labels)
            ax.tick_params(axis='both', colors=PALETTE['muted_text'])

            if cax is not None:
                cbar = self.response_fig.colorbar(im, cax=cax)
            else:
                cbar = self.response_fig.colorbar(im, ax=ax, pad=0.02)
            cbar.ax.set_ylabel(colorbar_label, color=PALETTE['muted_text'])
            cbar.ax.tick_params(colors=PALETTE['muted_text'])
            cbar.outline.set_edgecolor(PALETTE['muted_text'])
            setattr(self, attr, cbar)
            if isinstance(colorbar_store, dict):
                colorbar_store[plane] = cbar

        if isinstance(colorbar_store, dict):
            self.response_colorbars = colorbar_store


        self.response_heatmap_canvas.draw()


    def _format_device_label(self, name: str) -> str:
        """Return a display-friendly device label."""
        if not name:
            return name
        if name.endswith('.SETTING'):
            return name[:-8]
        return name

    def _save_plots(self, output_dir, base_stem):
        """Save all current plot figures as PNG images.

        Saves the FFT and readings figures as-is, then cycles through all
        three response matrix display modes (Gain, Uncertainty, Noise) so
        each view is captured.

        Args:
            output_dir: Directory to save into (Path object)
            base_stem: Filename stem prefix for the saved images
        """
        output_dir = Path(output_dir)
        plot_files = []

        def _save_fig(fig, suffix):
            try:
                if fig is not None and fig.get_axes():
                    path = output_dir / f"{base_stem}_{suffix}.png"
                    fig.savefig(str(path), dpi=150, bbox_inches='tight',
                                facecolor=fig.get_facecolor(), edgecolor='none')
                    plot_files.append(path)
            except Exception as e:
                print(f"[WARNING] Failed to save {suffix} plot: {e}")

        # Save FFT and live-readings figures
        _save_fig(self.fft_fig, "fft")
        _save_fig(self.fig, "readings")

        # Save all three response matrix views
        if hasattr(self, 'response_fig') and self.response_matrix_data is not None:
            original_mode = self.response_display_mode.get()

            mode_map = [
                ("Gain (Real)", "heatmap"),
                ("Uncertainty (sigma)", "uncertainty"),
                ("Device Noise", "noise"),
            ]
            for mode, suffix in mode_map:
                # Uncertainty view requires the error matrix
                if mode == "Uncertainty (sigma)" and self.response_error_matrix is None:
                    continue
                try:
                    self.response_display_mode.set(mode)
                    self._update_response_matrix_plot()
                    _save_fig(self.response_fig, suffix)
                except Exception as e:
                    print(f"[WARNING] Failed to save {suffix} plot: {e}")

            # Restore original display mode
            self.response_display_mode.set(original_mode)
            self._update_response_matrix_plot()

        if plot_files:
            print(f"[SUCCESS] Plots saved: {', '.join(str(p) for p in plot_files)}")

    def _save_response_matrix(self):
            """Saves the calculated response matrix to separate CSV files for each plane."""
            if self.response_matrix_data is None:
                print("[WARNING] No response matrix has been calculated yet.")
                return

            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                title="Save Response Matrix to CSV",
                initialdir=self.scan_data_path.get()
            )
            if not filepath:
                return

            try:
                base_path = Path(filepath)
                output_dir = base_path.parent
                base_stem = base_path.stem or "response_matrix"

                display_settings = self.response_setting_labels or [self._format_device_label(dev) for dev in self.response_setting_devices]
                reading_labels = self.response_reading_labels or [self._format_device_label(dev) for dev in self.response_reading_devices]

                plane_indices = {
                    "Horizontal": getattr(self, '_horizontal_indices', []),
                    "Vertical": getattr(self, '_vertical_indices', []),
                }

                saved_files = []
                for plane, indices in plane_indices.items():
                    if not indices:
                        print(f"[INFO] No {plane.lower()} readings available; skipping {plane.lower()} export.")
                        continue

                    sub_matrix = np.asarray(self.response_matrix_data)[indices, :]
                    if sub_matrix.size == 0:
                        print(f"[INFO] No {plane.lower()} data present in matrix; skipping {plane.lower()} export.")
                        continue

                    plane_labels = [reading_labels[idx] for idx in indices if 0 <= idx < len(reading_labels)]
                    if not plane_labels:
                        print(f"[INFO] Unable to resolve labels for {plane.lower()} readings; skipping {plane.lower()} export.")
                        continue

                    df = pd.DataFrame(sub_matrix, index=plane_labels, columns=display_settings)
                    df.index.name = "Reading Device"

                    plane_filename = f"{base_stem}_{plane.lower()}.csv"
                    plane_path = output_dir / plane_filename
                    atomic_to_csv(df, plane_path)
                    saved_files.append(plane_path)

                    # Export the error matrix for this plane if available
                    if self.response_error_matrix is not None:
                        sub_error = np.asarray(self.response_error_matrix)[indices, :]
                        if sub_error.size > 0:
                            df_err = pd.DataFrame(sub_error, index=plane_labels, columns=display_settings)
                            df_err.index.name = "Reading Device"
                            error_filename = f"{base_stem}_{plane.lower()}_error.csv"
                            error_path = output_dir / error_filename
                            atomic_to_csv(df_err, error_path)
                            saved_files.append(error_path)

                if saved_files:
                    saved_list = ', '.join(str(path) for path in saved_files)
                    self.last_analysis_source = str(saved_files[0])
                    print(f"[SUCCESS] Response matrix saved to: {saved_list}")
                    self._save_plots(output_dir, base_stem)
                else:
                    print("[WARNING] No plane-specific response matrices were saved.")
            except Exception as e:
                print(f"[ERROR] Failed to save response matrix: {e}")

    def _auto_save_response_matrix(self, save_dir, timestr):
        """Automatically save the calculated response matrix to CSV files (no dialog)."""
        if self.response_matrix_data is None:
            return

        try:
            output_dir = Path(save_dir)
            base_stem = f"response_matrix_{timestr}"

            display_settings = self.response_setting_labels or [self._format_device_label(dev) for dev in self.response_setting_devices]
            reading_labels = self.response_reading_labels or [self._format_device_label(dev) for dev in self.response_reading_devices]

            plane_indices = {
                "Horizontal": getattr(self, '_horizontal_indices', []),
                "Vertical": getattr(self, '_vertical_indices', []),
            }

            saved_files = []
            for plane, indices in plane_indices.items():
                if not indices:
                    continue

                sub_matrix = np.asarray(self.response_matrix_data)[indices, :]
                if sub_matrix.size == 0:
                    continue

                plane_labels = [reading_labels[idx] for idx in indices if 0 <= idx < len(reading_labels)]
                if not plane_labels:
                    continue

                df = pd.DataFrame(sub_matrix, index=plane_labels, columns=display_settings)
                df.index.name = "Reading Device"

                plane_filename = f"{base_stem}_{plane.lower()}.csv"
                plane_path = output_dir / plane_filename
                atomic_to_csv(df, plane_path)
                saved_files.append(plane_path)

                if self.response_error_matrix is not None:
                    sub_error = np.asarray(self.response_error_matrix)[indices, :]
                    if sub_error.size > 0:
                        df_err = pd.DataFrame(sub_error, index=plane_labels, columns=display_settings)
                        df_err.index.name = "Reading Device"
                        error_filename = f"{base_stem}_{plane.lower()}_error.csv"
                        error_path = output_dir / error_filename
                        atomic_to_csv(df_err, error_path)
                        saved_files.append(error_path)

            if saved_files:
                saved_list = ', '.join(str(p) for p in saved_files)
                print(f"[SUCCESS] ORM auto-saved to: {saved_list}")
                self._save_plots(output_dir, base_stem)
            else:
                print("[WARNING] Auto-save: No plane-specific response matrices were saved.")
        except Exception as e:
            print(f"[ERROR] Failed to auto-save response matrix: {e}")

    def _launch_analysis_app(self):
        """Launch the advanced analysis tool using in-memory data when available."""
        try:
            from analysis_viewer import launch_from_arrays, launch_from_csv
        except ImportError as exc:
            messagebox.showerror("Advanced Analysis", f"PyQt5 analysis module not available: {exc}")
            return

        process = None
        matrix = self.response_matrix_data
        if matrix is not None and self.response_setting_devices and self.response_reading_devices:
            matrix_array = np.asarray(matrix, dtype=float)
            row_labels = self.response_reading_labels or [self._format_device_label(dev) for dev in self.response_reading_devices]
            col_labels = self.response_setting_labels or [self._format_device_label(dev) for dev in self.response_setting_devices]
            if len(row_labels) != matrix_array.shape[0]:
                row_labels = [f"Reading {idx}" for idx in range(matrix_array.shape[0])]
            if len(col_labels) != matrix_array.shape[1]:
                col_labels = [f"Setting {idx}" for idx in range(matrix_array.shape[1])]
            error_matrix = None
            if self.response_error_matrix is not None:
                error_matrix = np.asarray(self.response_error_matrix, dtype=float)
            plane_map = {}
            if getattr(self, '_horizontal_indices', []):
                plane_map['Horizontal'] = [int(idx) for idx in self._horizontal_indices]
            if getattr(self, '_vertical_indices', []):
                plane_map['Vertical'] = [int(idx) for idx in self._vertical_indices]
            if not plane_map:
                plane_map = None
            process = launch_from_arrays(matrix_array, row_labels, col_labels, error_matrix, plane_map)
        elif self.last_analysis_source and os.path.exists(self.last_analysis_source):
            process = launch_from_csv(Path(self.last_analysis_source))
        else:
            messagebox.showinfo("Advanced Analysis", "No response matrix available. Calculate the matrix or save scan data first.")
            return

        if process is not None:
            self._analysis_processes.append(process)
            self.after(1500, lambda p=process: self._check_analysis_process(p))
            print(f"[INFO] Launching advanced analysis (PID {process.pid})")


    def _check_analysis_process(self, process):
        if self._closing:
            return
        processes = getattr(self, '_analysis_processes', [])
        if process not in processes:
            return
        if process.is_alive():
            if not self._closing:
                self.after(1500, lambda p=process: self._check_analysis_process(p))
            return
        exitcode = process.exitcode
        if exitcode not in (None, 0) and not self._closing:
            messagebox.showwarning("Advanced Analysis", f"Analysis tool exited with code {exitcode}")
        if process in processes:
            processes.remove(process)

    def _save_scan_setup(self):
            """Saves the current scan configuration to a JSON file."""
            filepath = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")],
                title="Save Scan Setup"
            )
            if not filepath:
                return

            try:
                configs = self._collect_scan_configs()
            except ValueError as exc:
                print(f"[ERROR] {exc}")
                return

            scan_setup = [cfg.to_setup_dict() for cfg in configs]

            try:
                atomic_write_text(filepath, json.dumps(scan_setup, indent=4),
                                  encoding='utf-8')
                print(f"[SUCCESS] Scan setup saved to {filepath}")
            except Exception as e:
                print(f"[ERROR] Failed to save scan setup: {e}")

    def _load_scan_setup(self):
            """Loads a scan configuration from a JSON file."""
            filepath = filedialog.askopenfilename(
                title="Load Scan Setup",
                filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
            )
            if not filepath:
                return

            try:
                with open(filepath, 'r') as f:
                    scan_setup = json.load(f)

                for item_id in list(self.scan_tree.get_children()):
                    self.scan_tree.delete(item_id)

                loaded_devices = []
                for params in scan_setup:
                    try:
                        config = ScanDeviceConfig.from_dict(params)
                    except ValueError as exc:
                        print(f"[WARNING] Skipping invalid scan entry: {exc}")
                        continue
                    self.scan_tree.insert('', 'end', values=config.to_tree_tuple())
                    loaded_devices.append(config.device)

                # Replace self.setting_devices to match loaded tree (remove stale entries)
                self.setting_devices = loaded_devices
                self.setting_file_label.config(text=f"{len(self.setting_devices)} devices in Setting List.")
                self._update_settings_display()

                print(f"[SUCCESS] Scan setup loaded from {filepath}")
            except Exception as e:
                print(f"[ERROR] Failed to load scan setup file: {e}")

    def _select_scan_data_directory(self):
            """Opens a dialog to select a directory for saving scan data."""
            dir_path = filedialog.askdirectory(
                title="Select Save Location for Scan Data",
                initialdir=self.scan_data_path.get()
            )
            if dir_path:
                self.scan_data_path.set(dir_path)
                print(f"[INFO] Scan data will be saved to: {dir_path}")

    # ############################################################################
    #
    # Safety System Methods
    #
    # ############################################################################

    def _toggle_safety(self):
        """Toggle safety monitoring on/off."""
        enabled = self.safety_enabled.get()
        self.safety_config.enabled = enabled
        self.safety_monitor.update_config(self.safety_config)
        self._update_safety_status_display()

        if enabled:
            print("[SUCCESS] Safety monitoring ENABLED")
        else:
            print("[WARNING] Safety monitoring DISABLED - scans will run without protection!")

    def _measure_safety_baseline(self):
        """Measure baseline RMS values for all reading devices."""
        if not self.reading_devices:
            messagebox.showwarning("No Devices", "Please add reading devices first.")
            return
        self._sync_dpm_node()

        num_samples = self.safety_baseline_samples.get()
        if num_samples < 10:
            messagebox.showwarning("Insufficient Samples", "Please use at least 10 samples for baseline measurement.")
            return

        # Disable button during measurement
        self.safety_measure_button.config(state='disabled')
        self.safety_baseline_status.config(text=f"Measuring... (0/{num_samples})")

        # Pre-capture tkinter vars before starting background thread
        acnet_event = self.acnet_event.get()
        reading_devices = list(self.reading_devices)

        # Start measurement in background thread
        threading.Thread(
            target=self._baseline_measurement_thread,
            args=(num_samples, acnet_event, reading_devices),
            daemon=True
        ).start()

    def _baseline_measurement_thread(self, num_samples, acnet_event=None, reading_devices=None):
        """Background thread for baseline measurement."""
        try:
            # IMPORTANT: Only measure reading devices (BPMs) for safety baseline, not correctors
            if acnet_event is None:
                print("[WARNING] acnet_event not pre-captured for baseline thread")
                return
            if reading_devices is None:
                reading_devices = []
            if not reading_devices:
                if not self._closing:
                    self.after(0, messagebox.showerror, "Error", "No reading devices available.")
                    self.after(0, self.safety_measure_button.config, {'state': 'normal'})
                return

            drf_list = [f"{dev}{acnet_event}" for dev in reading_devices]
            device_names = reading_devices

            # Collect data
            device_data = {dev: [] for dev in device_names}

            print(f"[INFO] Measuring safety baseline for {len(reading_devices)} reading devices (BPMs only)")

            for i in range(num_samples):
                if self._closing:
                    break
                data = self.scanner.read_once_on_event(drf_list)
                if data:
                    for item in data:
                        if item is None:
                            continue
                        dev_name = item.get('name')
                        value = item.get('data')
                        if dev_name and dev_name in device_data and value is not None:
                            device_data[dev_name].append(value)

                # Update status
                if not self._closing:
                    self.after(0, self.safety_baseline_status.config, {'text': f"Measuring... ({i+1}/{num_samples})"})
                time.sleep(0.1)  # Small delay between samples

            # Calculate baselines and detect dead BPMs. A device is only
            # classified dead if it returned a solid run of samples that were
            # ALL zero — a thin/partial run is "couldn't measure", not "dead".
            min_baseline_samples = SafetyMonitor.MIN_SAMPLES_FOR_CHECK
            baselines_computed = 0
            baselined = set()
            dead_bpms = set()
            sparse_devices = []
            for device, values in device_data.items():
                if len(values) < min_baseline_samples:
                    if values:
                        sparse_devices.append(device)
                    continue
                if all(v == 0 for v in values):
                    dead_bpms.add(device)
                    continue
                baseline = SafetyMonitor.calculate_baseline_from_data(device, values)
                self.safety_config.set_baseline(device, baseline)
                baselined.add(device)
                baselines_computed += 1

            # If NOTHING showed real signal, all-zero readings almost certainly
            # mean "no beam" or a systemic fault — not dead hardware. Exclude
            # nothing in that case rather than nuking every BPM.
            if dead_bpms and baselines_computed == 0:
                print(f"[WARNING] All {len(dead_bpms)} measured BPM(s) read all-zero — "
                      f"likely no beam or a systemic fault, NOT dead hardware. "
                      f"No BPMs excluded; re-measure with beam on.")
                dead_bpms = set()

            # Any measured device that did NOT get a fresh baseline this run
            # (dead, sparse, or no data) must not be left monitored against a
            # stale baseline from a previous measurement — drop it.
            for device in device_data:
                if device not in baselined:
                    self.safety_config.device_baselines.pop(device, None)

            self.dead_bpms = dead_bpms

            if sparse_devices:
                print(f"[WARNING] {len(sparse_devices)} device(s) returned fewer than "
                      f"{min_baseline_samples} samples — skipped, no baseline: "
                      f"{', '.join(sorted(sparse_devices))}")

            # Update monitor config
            self.safety_monitor.update_config(self.safety_config)
            self.safety_baselines_measured = baselines_computed > 0

            if dead_bpms:
                print(f"[WARNING] {len(dead_bpms)} dead BPM(s) read all-zero — excluded "
                      f"from scans, CSV, and response matrix: {', '.join(sorted(dead_bpms))}")

            # Update GUI
            if not self._closing:
                self.after(0, self._update_baseline_display)
                self.after(0, self._update_safety_status_display)
                if baselines_computed > 0:
                    status = f"Status: {baselines_computed}/{len(device_data)} baselines measured"
                    if dead_bpms:
                        status += f" ({len(dead_bpms)} dead BPM(s) excluded)"
                    self.after(0, self.safety_baseline_status.config, {'text': status})
                else:
                    self.after(0, self.safety_baseline_status.config,
                               {'text': "Status: No valid baselines (no data received)"})
            if baselines_computed > 0:
                print(f"[SUCCESS] Safety baselines measured for {baselines_computed}/{len(device_data)} devices")
            else:
                print(f"[WARNING] No valid baselines produced — safety monitoring will not be active")

        except Exception as e:
            error_msg = str(e)
            if not self._closing:
                self.after(0, messagebox.showerror, "Error", f"Baseline measurement failed: {error_msg}")
            print(f"[ERROR] Baseline measurement failed: {error_msg}")
        finally:
            if not self._closing:
                self.after(0, self.safety_measure_button.config, {'state': 'normal'})

    def _update_baseline_display(self):
        """Update the baseline data tree view."""
        if self._closing:
            return
        # Clear existing items
        for item in self.safety_baseline_tree.get_children():
            self.safety_baseline_tree.delete(item)

        # Add baseline data
        for device, baseline in self.safety_config.device_baselines.items():
            self.safety_baseline_tree.insert('', 'end', values=(
                device,
                f"{baseline.mean:.6f}",
                f"{baseline.rms:.6f}",
                f"{baseline.std:.6f}",
                f"{baseline.min_val:.6f}",
                f"{baseline.max_val:.6f}",
                baseline.sample_count
            ))

    def _update_safety_config(self):
        """Update safety configuration based on UI changes."""
        # Update threshold type
        threshold_type = self.safety_threshold_type.get()
        if threshold_type == "per_device":
            self.safety_config.threshold_type = SafetyThresholdType.PER_DEVICE_MEAN
        else:
            self.safety_config.threshold_type = SafetyThresholdType.OVERALL_MEAN

        self.safety_monitor.update_config(self.safety_config)

    def _apply_safety_thresholds(self):
        """Apply threshold settings from UI to configuration."""
        try:
            # Read values into locals first for validation before applying
            pd_warn = self.safety_per_device_warning.get()
            pd_abort = self.safety_per_device_abort.get()
            ov_warn = self.safety_overall_warning.get()
            ov_abort = self.safety_overall_abort.get()
            buffer_size = self.safety_buffer_size.get()

            # Validate before applying
            if buffer_size < MIN_BUFFER_SIZE:
                messagebox.showwarning(
                    "Invalid Buffer Size",
                    f"Buffer size must be at least {MIN_BUFFER_SIZE} — safety checks "
                    f"need that many samples before they can fire.")
                return

            if pd_warn >= pd_abort:
                messagebox.showwarning("Invalid Thresholds", "Per-device warning threshold must be less than abort threshold.")
                return

            if ov_warn >= ov_abort:
                messagebox.showwarning("Invalid Thresholds", "Overall warning threshold must be less than abort threshold.")
                return

            # Validation passed — apply to config
            self.safety_config.per_device_warning_threshold = pd_warn
            self.safety_config.per_device_abort_threshold = pd_abort
            self.safety_config.overall_warning_threshold = ov_warn
            self.safety_config.overall_abort_threshold = ov_abort
            self.safety_config.buffer_size = buffer_size

            self.safety_monitor.update_config(self.safety_config)
            print("[SUCCESS] Safety thresholds applied successfully")
            messagebox.showinfo("Success", "Safety thresholds updated successfully.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply thresholds: {e}")

    def _save_safety_config(self):
        """Save safety configuration to JSON file."""
        filepath = filedialog.asksaveasfilename(
            title="Save Safety Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            config_dict = self.safety_config.to_dict()
            atomic_write_text(filepath, json.dumps(config_dict, indent=2),
                              encoding='utf-8')
            print(f"[SUCCESS] Safety configuration saved to: {filepath}")
            messagebox.showinfo("Success", "Safety configuration saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {e}")
            print(f"[ERROR] Failed to save safety configuration: {e}")

    def _load_safety_config(self):
        """Load safety configuration from JSON file."""
        filepath = filedialog.askopenfilename(
            title="Load Safety Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)

            self.safety_config = SafetyConfiguration.from_dict(config_dict)
            self.safety_monitor.update_config(self.safety_config)
            # A loaded config defines its own known-good baselines; drop any
            # dead-BPM exclusions from a prior in-session baseline measurement.
            self.dead_bpms = set()

            # Update UI
            self.safety_enabled.set(self.safety_config.enabled)
            self.safety_per_device_warning.set(self.safety_config.per_device_warning_threshold)
            self.safety_per_device_abort.set(self.safety_config.per_device_abort_threshold)
            self.safety_overall_warning.set(self.safety_config.overall_warning_threshold)
            self.safety_overall_abort.set(self.safety_config.overall_abort_threshold)
            self.safety_buffer_size.set(self.safety_config.buffer_size)

            if self.safety_config.threshold_type == SafetyThresholdType.PER_DEVICE_MEAN:
                self.safety_threshold_type.set("per_device")
            else:
                self.safety_threshold_type.set("overall")

            self._update_baseline_display()

            num_loaded = len(self.safety_config.device_baselines)
            self.safety_baselines_measured = num_loaded > 0

            self._update_safety_status_display()

            if num_loaded > 0:
                print(f"[SUCCESS] Safety configuration loaded from: {filepath} ({num_loaded} baselines)")
            else:
                print(f"[WARNING] Safety configuration loaded but contains no baselines — safety monitoring will not be active")
            messagebox.showinfo("Success", "Safety configuration loaded successfully.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration: {e}")
            print(f"[ERROR] Failed to load safety configuration: {e}")

    def _clear_safety_baselines(self):
        """Clear all safety baselines."""
        result = messagebox.askyesno(
            "Confirm Clear",
            "Are you sure you want to clear all safety baselines? This will disable safety monitoring until new baselines are measured.",
            icon='warning'
        )

        if result:
            self.safety_config.clear_baselines()
            self.dead_bpms = set()
            self.safety_monitor.update_config(self.safety_config)
            self.safety_baselines_measured = False

            # Clear tree view
            for item in self.safety_baseline_tree.get_children():
                self.safety_baseline_tree.delete(item)

            self.safety_baseline_status.config(text="Status: Not measured")
            self._update_safety_status_display()
            print("[WARNING] All safety baselines cleared")

    def _update_safety_status_display(self):
        """Update the safety status banner."""
        if self._closing:
            return
        if not self.safety_config.enabled:
            self.safety_status_label.config(
                text="⚠ SAFETY: DISABLED",
                bg=SAFETY_DISABLED_COLOR
            )
        elif not self.safety_baselines_measured:
            self.safety_status_label.config(
                text="⚠ SAFETY: NOT CONFIGURED (No baselines)",
                bg=SAFETY_WARNING_COLOR
            )
        else:
            num_baselines = len(self.safety_config.device_baselines)
            threshold_type = "Per-Device" if self.safety_config.threshold_type == SafetyThresholdType.PER_DEVICE_MEAN else "Overall"
            self.safety_status_label.config(
                text=f"✓ SAFETY: ACTIVE ({num_baselines} devices, {threshold_type} mode)",
                bg=SAFETY_ACTIVE_COLOR
            )

    def _handle_safety_violation(self, violation):
        """Handle ABORT-level safety violation."""
        print(f"[ERROR] {violation}")

        # Set stop flag (will be caught in scan loop)
        self.stop_scan_flag.set()

        # Log to file if enabled
        if self.safety_config.log_to_file:
            self._log_safety_event(violation)

    def _handle_safety_warning(self, violation):
        """Handle WARNING-level safety violation."""
        if self.safety_config.log_warnings:
            print(f"[WARNING] {violation}")

        # Log to file if enabled
        if self.safety_config.log_to_file:
            self._log_safety_event(violation)

    def _log_safety_event(self, violation):
        """Log safety event to file."""
        try:
            log_dir = Path(SAFETY_LOG_DIR)
            log_dir.mkdir(exist_ok=True)

            log_file = log_dir / SAFETY_LOG_FILE

            # Read existing log or create new
            if log_file.exists():
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
            else:
                log_data = {'events': []}

            # Add new event
            event = {
                'timestamp': violation.timestamp.isoformat(),
                'type': violation.violation_type.value,
                'device': violation.device,
                'value': violation.value,
                'baseline_mean': violation.baseline_mean,
                'current_mean': violation.current_mean,
                'mean_shift': violation.mean_shift,
                'sigma_deviation': violation.sigma_deviation,
                'threshold': violation.threshold,
                'scan_active': self.is_scanning
            }

            log_data['events'].append(event)

            # Write back atomically (temp file + os.replace) so a crash mid-write
            # never leaves a truncated safety log.
            atomic_write_text(log_file, json.dumps(log_data, indent=2),
                              encoding='utf-8')

        except Exception as e:
            print(f"[ERROR] Failed to log safety event: {e}")

    def _store_nominal_settings(self, devices):
        """Store nominal settings before scan starts."""
        try:
            nominals = self.scanner.get_settings_once(devices)
            self.nominal_settings.clear()

            for device, value in zip(devices, nominals):
                if value is not None:
                    self.nominal_settings.store(device, value)

            print(f"[INFO] Stored nominal settings for {len(devices)} devices")

        except Exception as e:
            print(f"[WARNING] Failed to store nominal settings: {e}")

    def _verify_nominal_restore(self, devices, values):
        """Read devices back after a nominal restore and loudly flag any that
        did not return to nominal. Never raises — safe to call from finally
        blocks. This is a read-back only; it changes no device settings."""
        try:
            try:
                readback = self.scanner.get_settings_once(list(devices))
            except Exception as e:
                readback = None
                print(f"[WARNING] Could not read devices back to verify nominal restore: {e}")

            unverified = []
            for i, (dev, target) in enumerate(zip(devices, values)):
                actual = readback[i] if (readback is not None and i < len(readback)) else None
                if actual is None:
                    unverified.append(dev)
                    continue
                try:
                    ok = math.isclose(float(actual), float(target),
                                      rel_tol=ACNET_RESTORE_VERIFY_TOLERANCE, abs_tol=1e-6)
                except (TypeError, ValueError):
                    ok = False
                if not ok:
                    unverified.append(dev)

            if not unverified:
                print(f"[SUCCESS] Nominal restore verified for {len(devices)} device(s).")
                return

            banner = "=" * 64
            print(banner)
            print("[ERROR] CRITICAL: NOMINAL RESTORE FAILED OR UNVERIFIED")
            print(f"[ERROR] MANUALLY VERIFY DEVICE SETTINGS: {', '.join(map(str, unverified))}")
            print(banner)
            if not self._closing:
                try:
                    msg = ("These devices could not be confirmed back at nominal:\n\n"
                           f"{', '.join(map(str, unverified))}\n\n"
                           "MANUALLY VERIFY their settings.")
                    self.after(0, messagebox.showerror, "Nominal Restore Unverified", msg)
                except Exception:
                    pass
        except Exception as e:
            # Runs from finally blocks — must never let an exception escape and
            # mask the original error.
            print(f"[ERROR] Nominal-restore verification itself failed: {e}")

    def _restore_nominal_settings(self, role="OPERATOR"):
        """Restore nominal settings after scan abort/stop.

        Raises CredentialExpiredError so callers can attempt renewal.
        All other exceptions are caught and logged.
        """
        if not self.nominal_settings.has_values():
            print("[WARNING] No nominal settings to restore")
            return

        devices = list(self.nominal_settings.get_all().keys())
        values = [self.nominal_settings.get(dev) for dev in devices]

        try:
            self.scanner.apply_settings_once(devices, values, role)
        except CredentialExpiredError:
            raise
        except Exception as e:
            print(f"[ERROR] Failed to restore nominal settings: {e}")

        # Verify regardless of whether the apply reported success — a failed
        # apply means the devices are still off, which is exactly what the
        # verification banner is for.
        self._verify_nominal_restore(devices, values)

    def _save_abort_log(self, violations):
        """
        Save comprehensive abort log with all diagnostic information.

        Args:
            violations: List of SafetyViolation objects that triggered abort
        """
        print("[INFO] Saving detailed abort log...")
        try:
            from pathlib import Path
            import json
            from datetime import datetime
            import os

            # Create abort logs directory (use absolute path)
            abort_log_dir = Path(SAFETY_LOG_DIR).resolve() / "abort_logs"
            abort_log_dir.mkdir(parents=True, exist_ok=True)

            print(f"[INFO] Abort log directory: {abort_log_dir}")

            # Create timestamped filename
            timestamp = datetime.now()
            filename = f"ABORT_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = abort_log_dir / filename

            print(f"[INFO] Creating log file: {filename}")

            # Gather comprehensive abort information
            abort_data = {
                "abort_info": {
                    "timestamp": timestamp.isoformat(),
                    "trigger_count": len(violations),
                    "scan_was_active": self.is_scanning
                },

                "violations": [
                    {
                        "device": v.device,
                        "baseline_mean": v.baseline_mean,
                        "current_mean": v.current_mean,
                        "reading_at_abort": v.current_mean,  # Explicit: the reading that triggered abort
                        "baseline_std": v.baseline_std,
                        "mean_shift": v.mean_shift,
                        "sigma_deviation": v.sigma_deviation,
                        "threshold_sigma": v.threshold,
                        "threshold_in_position_units": v.threshold * v.baseline_std,
                        "exceeded_by_sigma": v.sigma_deviation - v.threshold,
                        "status": "EXCEEDED" if v.sigma_deviation >= v.threshold else "OK",
                        "violation_type": v.violation_type.value,
                        "timestamp": v.timestamp.isoformat()
                    }
                    for v in violations
                ],

                "safety_configuration": {
                    "enabled": self.safety_config.enabled,
                    "threshold_type": self.safety_config.threshold_type.value,
                    "per_device_warning_threshold": self.safety_config.per_device_warning_threshold,
                    "per_device_abort_threshold": self.safety_config.per_device_abort_threshold,
                    "overall_warning_threshold": self.safety_config.overall_warning_threshold,
                    "overall_abort_threshold": self.safety_config.overall_abort_threshold,
                    "buffer_size": self.safety_config.buffer_size,
                    "auto_abort_on_violation": self.safety_config.auto_abort_on_violation,
                    "restore_nominals_on_abort": self.safety_config.restore_nominals_on_abort
                },

                "baseline_measurements": {
                    device: {
                        "mean": baseline.mean,
                        "rms": baseline.rms,
                        "std": baseline.std,
                        "min": baseline.min_val,
                        "max": baseline.max_val,
                        "sample_count": baseline.sample_count,
                        "measured_at": baseline.timestamp.isoformat()
                    }
                    for device, baseline in self.safety_config.device_baselines.items()
                },

                "scan_configuration": {},
                "corrector_settings": {},
                "current_readings": {}
            }

            # Add scan configuration if available
            if self._last_run_config:
                # Get device configurations
                device_configs = []
                for dev_config in self._last_run_config.devices:
                    device_configs.append({
                        "device": dev_config.device,
                        "amplitude": dev_config.amplitude,
                        "periods": dev_config.periods
                    })

                abort_data["scan_configuration"] = {
                    "device_names": self._last_run_config.device_names,
                    "device_configs": device_configs,
                    "points_per_superperiod": self._last_run_config.points_per_superperiod,
                    "num_superperiods": self._last_run_config.superperiods,
                    "total_steps": self._last_run_config.total_steps,
                    "acnet_role": self._last_run_config.role
                }

            # Add current corrector settings (nominal values)
            if self.nominal_settings.has_values():
                abort_data["corrector_settings"] = {
                    "nominal_values": self.nominal_settings.get_all(),
                    "stored_at": self.nominal_settings.timestamp.isoformat() if self.nominal_settings.timestamp else None
                }

            # Try to get current readings from all BPMs
            try:
                if self.reading_devices:
                    current_readings = {}
                    for device in self.reading_devices:
                        # Get from data buffers if available
                        buffer_data = self.safety_monitor._data_buffers.get(device)
                        if buffer_data and len(buffer_data) > 0:
                            current_readings[device] = {
                                "latest_value": list(buffer_data)[-1],
                                "buffer_mean": float(sum(buffer_data) / len(buffer_data)),
                                "buffer_size": len(buffer_data)
                            }
                    abort_data["current_readings"] = current_readings
            except Exception as e:
                abort_data["current_readings"] = {"error": f"Failed to capture: {str(e)}"}

            # Add violation history summary
            abort_data["violation_history_summary"] = {
                "total_violations": len(self.safety_monitor.violation_history),
                "total_warnings": len(self.safety_monitor.warning_history),
                "recent_warnings": [
                    {
                        "device": v.device,
                        "sigma_deviation": v.sigma_deviation,
                        "timestamp": v.timestamp.isoformat()
                    }
                    for v in self.safety_monitor.warning_history[-10:]  # Last 10 warnings
                ]
            }

            # Save to file atomically
            atomic_write_text(filepath, json.dumps(abort_data, indent=2),
                              encoding='utf-8')

            # Also save human-readable text version
            text_filename = f"ABORT_{timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
            text_filepath = abort_log_dir / text_filename

            # Build the human-readable report in a buffer, then write atomically.
            f = io.StringIO()
            if True:
                f.write("=" * 80 + "\n")
                f.write("SAFETY ABORT LOG\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Scan Active: {self.is_scanning}\n\n")

                f.write("-" * 80 + "\n")
                f.write("VIOLATIONS THAT TRIGGERED ABORT:\n")
                f.write("-" * 80 + "\n")
                for i, v in enumerate(violations, 1):
                    # Calculate threshold in position units
                    threshold_in_units = v.threshold * v.baseline_std

                    f.write(f"\n{i}. {v.device}\n")
                    f.write(f"   " + "─" * 70 + "\n")
                    f.write(f"   BASELINE (Normal Operation):\n")
                    f.write(f"     Mean Position:  {v.baseline_mean:.6f} mm\n")
                    f.write(f"     Std Deviation:  {v.baseline_std:.6f} mm\n")
                    f.write(f"     Noise Level:    ±{v.baseline_std:.6f} mm (1σ)\n")
                    f.write(f"   \n")
                    f.write(f"   AT ABORT:\n")
                    f.write(f"     Current Position: {v.current_mean:.6f} mm  ← READING AT ABORT\n")
                    f.write(f"     Position Shift:   {v.mean_shift:.6f} mm\n")
                    f.write(f"     Deviation:        {v.sigma_deviation:.2f}σ (sigma from baseline)\n")
                    f.write(f"   \n")
                    f.write(f"   THRESHOLD vs ACTUAL:\n")
                    f.write(f"     Abort Threshold:  {v.threshold:.2f}σ = {threshold_in_units:.6f} mm shift allowed\n")
                    f.write(f"     Actual Shift:     {v.sigma_deviation:.2f}σ = {v.mean_shift:.6f} mm shift\n")
                    f.write(f"     Status:           {'EXCEEDED' if v.sigma_deviation >= v.threshold else 'OK'} ({'exceeded by ' + str(round(v.sigma_deviation - v.threshold, 2)) + 'σ'})\n")
                    f.write(f"   \n")
                    f.write(f"   Time: {v.timestamp.strftime('%H:%M:%S.%f')[:-3]}\n")

                f.write("\n" + "-" * 80 + "\n")
                f.write("SAFETY CONFIGURATION:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Threshold Type: {self.safety_config.threshold_type.value}\n")
                f.write(f"Per-Device Warning: {self.safety_config.per_device_warning_threshold}σ\n")
                f.write(f"Per-Device Abort:   {self.safety_config.per_device_abort_threshold}σ\n")
                f.write(f"Buffer Size: {self.safety_config.buffer_size} samples\n")

                if abort_data["scan_configuration"]:
                    f.write("\n" + "-" * 80 + "\n")
                    f.write("SCAN CONFIGURATION:\n")
                    f.write("-" * 80 + "\n")
                    sc = abort_data["scan_configuration"]
                    f.write(f"Devices Scanned: {', '.join(sc.get('device_names', []))}\n")
                    f.write(f"Points/Superperiod: {sc.get('points_per_superperiod', 'N/A')}\n")
                    f.write(f"Num Superperiods: {sc.get('num_superperiods', 'N/A')}\n")
                    f.write(f"Total Steps: {sc.get('total_steps', 'N/A')}\n")
                    f.write(f"ACNET Role: {sc.get('acnet_role', 'N/A')}\n")

                    # Write individual device configurations
                    if 'device_configs' in sc:
                        f.write("\nDevice Scan Details:\n")
                        for dev_cfg in sc['device_configs']:
                            f.write(f"  {dev_cfg['device']}: amplitude={dev_cfg['amplitude']}, periods={dev_cfg['periods']}\n")

                f.write("\n" + "-" * 80 + "\n")
                f.write("ALL BASELINE MEASUREMENTS:\n")
                f.write("-" * 80 + "\n")
                for device, baseline in abort_data["baseline_measurements"].items():
                    f.write(f"\n{device}:\n")
                    f.write(f"  Mean: {baseline['mean']:.6f}, Std: {baseline['std']:.6f}\n")
                    f.write(f"  RMS:  {baseline['rms']:.6f}\n")
                    f.write(f"  Range: [{baseline['min']:.6f}, {baseline['max']:.6f}]\n")
                    f.write(f"  Samples: {baseline['sample_count']}\n")

                f.write("\n" + "=" * 80 + "\n")
                f.write(f"JSON data saved to: {filename}\n")
                f.write("=" * 80 + "\n")

            atomic_write_text(text_filepath, f.getvalue(), encoding='utf-8')

            print(f"[SUCCESS] Detailed abort log saved:")
            print(f"          JSON: {filepath.absolute()}")
            print(f"          Text: {text_filepath.absolute()}")
            print(f"[INFO] You can find these files in: {abort_log_dir}")

        except Exception as e:
            print(f"[ERROR] Failed to save abort log: {e}")
            print(f"[ERROR] This error occurred while trying to save to: {abort_log_dir if 'abort_log_dir' in locals() else 'unknown path'}")
            import traceback
            traceback.print_exc()

    def _trigger_safety_abort(self, violations):
        """Emergency abort procedure for safety violations."""
        print("")
        print("=" * 80)
        print("[ERROR] ===== SAFETY ABORT TRIGGERED =====")
        print("=" * 80)

        for v in violations:
            print(f"[ERROR] {v}")

        print("[ERROR] Scan has been automatically stopped for safety.")

        # Save detailed abort log
        self._save_abort_log(violations)

        # Restore nominals if configured
        if self.safety_config.restore_nominals_on_abort:
            role = self._last_run_config.role if self._last_run_config else "OPERATOR"
            self._restore_nominal_settings(role)

        # Update safety monitor state
        self.safety_monitor.clear_buffers()

        # Show abort dialog
        if not self._closing:
            self.after(0, self._show_safety_abort_dialog, violations)

        print("=" * 80)
        print("")

    def _show_safety_abort_dialog(self, violations):
        """Show safety abort dialog to user."""
        if self._closing:
            return
        violation_text = "\n".join([str(v) for v in violations[:5]])  # Show first 5

        messagebox.showerror(
            "SAFETY ABORT",
            f"Scan aborted due to safety threshold violation!\n\n{violation_text}\n\n"
            f"Devices with known nominals have been restored.\n"
            f"Devices with unknown nominals were NOT restored — verify manually.\n"
            f"Check the logs for details.",
            icon='error'
        )

    # =========================================================================
    # Beam Interlock Methods
    # =========================================================================

    def _build_beam_interlock_config(self) -> BeamInterlockConfig:
        """Build a BeamInterlockConfig from the current UI state."""
        loss_monitors = {}
        for item in self.loss_monitor_tree.get_children():
            vals = self.loss_monitor_tree.item(item, 'values')
            if vals and len(vals) >= 2:
                try:
                    loss_monitors[str(vals[0])] = float(vals[1])
                except ValueError:
                    pass
        return BeamInterlockConfig(
            enabled=self.beam_interlock_enabled_var.get(),
            loss_monitors=loss_monitors,
            beam_event=self.beam_event_entry.get().strip() or "@e,0A",
            beam_role=self.beam_role_entry.get().strip(),
            disable_beam_on_completion=self.disable_beam_on_completion_var.get(),
        )

    def _toggle_interlock_settings(self):
        """Enable/disable interlock settings widgets based on checkbox."""
        enabled = self.beam_interlock_enabled_var.get()
        state = 'normal' if enabled else 'disabled'
        for child in self._interlock_settings_frame.winfo_children():
            try:
                child.config(state=state)
            except tk.TclError:
                # Frame widgets don't have a state option; recurse into their children
                for subchild in child.winfo_children():
                    try:
                        subchild.config(state=state)
                    except tk.TclError:
                        pass

    def _add_loss_monitor(self):
        """Add a new loss monitor to the table via dialog."""
        dialog = tk.Toplevel(self)
        dialog.title("Add Loss Monitor")
        dialog.geometry("300x150")
        dialog.transient(self)
        dialog.grab_set()

        ttk.Label(dialog, text="Device Name:").pack(pady=(10, 2))
        dev_entry = ttk.Entry(dialog, width=25)
        dev_entry.pack()

        ttk.Label(dialog, text="Max Threshold:").pack(pady=(5, 2))
        thresh_entry = ttk.Entry(dialog, width=25)
        thresh_entry.pack()

        def _do_add():
            dev = dev_entry.get().strip()
            try:
                thresh = float(thresh_entry.get().strip())
            except ValueError:
                messagebox.showwarning("Invalid", "Threshold must be a number.", parent=dialog)
                return
            if dev:
                self.loss_monitor_tree.insert('', 'end', values=(dev, f"{thresh:.2f}"))
                dialog.destroy()

        ttk.Button(dialog, text="Add", command=_do_add, style="Accent.TButton").pack(pady=10)

    def _remove_loss_monitor(self):
        """Remove selected loss monitor from the table."""
        selected = self.loss_monitor_tree.selection()
        for item in selected:
            self.loss_monitor_tree.delete(item)

    def _get_beam_role(self) -> str:
        """Get the ACNET role for beam control."""
        config = self._build_beam_interlock_config()
        self.beam_interlock.update_config(config)
        return self.beam_interlock.get_effective_role(self.setting_role.get())

    def _run_beam_op_async(self, work, on_done):
        """Run a blocking ACNET beam op off the Tk main thread (#5).

        `work` is a 0-arg callable executed on a short daemon worker thread. It
        invokes scanner/interlock methods which still funnel through the SINGLE
        shared ACNET executor — this helper does NOT create an executor or call
        acsys.run_client itself. The result (or the exception) is marshalled back
        to the Tk main thread via self.after(0, ...) and passed to
        on_done(result, error) there, guarded by self._closing.
        """
        def _worker():
            result = None
            error = None
            try:
                result = work()
            except Exception as e:  # surface to the UI rather than crash the thread
                error = e
            if self._closing:
                return
            try:
                self.after(0, _deliver, result, error)
            except RuntimeError:
                pass  # interpreter/Tk shutting down

        def _deliver(result, error):
            if self._closing:
                return
            on_done(result, error)

        threading.Thread(target=_worker, daemon=True).start()

    def _refresh_beam_status(self):
        """Re-read beam status from ACNET and update indicator (off-thread)."""
        # Build/snapshot config on the Tk main thread, then read status on a
        # worker so the GUI never blocks on ACNET.
        self.beam_interlock_config = self._build_beam_interlock_config()
        self.beam_interlock.update_config(self.beam_interlock_config)

        def _done(beam_on, error):
            if error is not None:
                print(f"[WARNING] Failed to refresh beam status: {error}")
                self._set_beam_status(BeamStatus.UNKNOWN)
                return
            self._set_beam_status(
                BeamStatus.ON if beam_on is True
                else BeamStatus.OFF if beam_on is False
                else BeamStatus.UNKNOWN)

        self._run_beam_op_async(self.beam_interlock.check_beam_on, _done)

    def _set_beam_status(self, status: BeamStatus):
        """Update beam status indicator label."""
        if status == BeamStatus.ON:
            self.beam_status_label.config(text="BEAM: ON", bg=PALETTE['success'], fg='white')
        elif status == BeamStatus.OFF:
            self.beam_status_label.config(text="BEAM: OFF", bg=PALETTE['error'], fg='white')
        else:
            self.beam_status_label.config(text="BEAM: --", bg=PALETTE['border'], fg=PALETTE['text'])

    def _on_beam_enable(self):
        """Enable beam via L:BSTUDY (ACNET call off the Tk thread, #5)."""
        role = self._get_beam_role()

        def _done(ok, error):
            if error is not None:
                print(f"[ERROR] Beam enable failed: {error}")
                self._set_beam_status(BeamStatus.UNKNOWN)
                return
            self._refresh_beam_status()
            if ok:
                print("[SUCCESS] Beam enable command sent")
            else:
                print("[ERROR] Beam enable failed — check log")

        self._run_beam_op_async(lambda: self.beam_interlock.enable_beam(role), _done)

    def _on_beam_disable(self):
        """Disable beam via L:BSTUDY, then verify it actually went off.

        A sent command is not a confirmed state — if the control value is wrong
        or the setting is rejected, the disable can silently no-op. The blocking
        disable + L:BSTUDY status readback run on a short worker thread (#5, still
        through the single shared ACNET executor); the verify result and the loud
        alarm modals are marshalled back to the Tk main thread. The confirm
        prompt, scan-abort, and all modal/status-label behavior are preserved.
        """
        result = messagebox.askyesno(
            "Disable Beam",
            "Are you sure you want to disable the beam?",
            icon='warning'
        )
        if not result:
            return

        role = self._get_beam_role()

        # Abort any running scan regardless of how the disable resolves.
        if self.is_scanning:
            self.stop_scan_flag.set()
            print("[WARNING] Beam disable requested — scan abort requested")

        def _work():
            # Returns (sent_ok, beam_on). Raises only if the disable command
            # itself could not be sent (surfaced as `error` in _done).
            self.scanner.disable_beam(BEAM_CONTROL_DRF, role)
            # Verify: read L:BSTUDY status back rather than trusting the command.
            beam_on = self.beam_interlock.check_beam_on()
            return beam_on

        def _done(beam_on, error):
            if error is not None:
                print(f"[ERROR] Beam disable command failed to send: {error}")
                self._set_beam_status(BeamStatus.UNKNOWN)
                messagebox.showerror(
                    "Beam Disable Failed",
                    f"The beam disable command could not be sent:\n\n{error}\n\n"
                    "MANUALLY VERIFY THE BEAM STATE."
                )
                return

            if beam_on is False:
                self._set_beam_status(BeamStatus.OFF)
                print("[SUCCESS] Beam disable confirmed — beam is OFF")
            elif beam_on is True:
                self._set_beam_status(BeamStatus.ON)
                print("[ERROR] CRITICAL: beam disable command sent but beam is STILL ON")
                messagebox.showerror(
                    "Beam Still On",
                    "The beam disable command was sent, but L:BSTUDY reports the "
                    "beam is STILL ON.\n\nThe command may have been rejected or the "
                    "control value is wrong.\n\nDISABLE THE BEAM MANUALLY NOW."
                )
            else:
                self._set_beam_status(BeamStatus.UNKNOWN)
                print("[ERROR] Beam disable command sent but status could NOT be verified")
                messagebox.showwarning(
                    "Beam State Unknown",
                    "The beam disable command was sent, but the beam status could "
                    "not be read back to confirm it.\n\nMANUALLY VERIFY THE BEAM STATE."
                )

        self._run_beam_op_async(_work, _done)

    def _build_loss_monitor_drf_list(self, config=None) -> list:
        """DRF list for beam-loss monitors, read in the same event batch as scan data.

        Returns an empty list when the interlock is disabled so callers can
        unconditionally append it to the readback DRF list.

        Args:
            config: optional frozen BeamInterlockConfig snapshot. The scan worker
                    passes its immutable snapshot so it reads only frozen state;
                    None falls back to the live self.beam_interlock_config.
        """
        cfg = config if config is not None else self.beam_interlock_config
        if not cfg.enabled:
            return []
        event = cfg.beam_event
        return [f"{dev}{event}" for dev in self._loss_monitor_devices_from_config(cfg)]

    def _check_loss_monitor_data(self, loss_data, config=None) -> Optional[BeamTripEvent]:
        """Check loss-monitor readings (already read alongside scan data).

        Args:
            loss_data: the loss-monitor slice of a read_once_on_event result —
                       a list of reading dicts (or None entries on timeout).
            config: optional frozen BeamInterlockConfig snapshot (see
                    _build_loss_monitor_drf_list). None uses the live config.

        Returns:
            BeamTripEvent if a threshold was exceeded, None otherwise.
        """
        cfg = config if config is not None else self.beam_interlock_config
        if not cfg.enabled:
            return None

        readings = {}
        for item in loss_data:
            if isinstance(item, dict) and 'name' in item and 'data' in item:
                try:
                    readings[item['name']] = float(item['data'])
                except (TypeError, ValueError):
                    # Non-scalar reading — can't evaluate this monitor; skip it.
                    pass

        # Evaluate against the frozen thresholds directly so the worker never
        # reads live self.beam_interlock_config via the monitor object.
        for device, threshold in cfg.loss_monitors.items():
            value = readings.get(device)
            if value is not None and value > threshold:
                event = BeamTripEvent(
                    device=device,
                    value=float(value),
                    threshold=threshold,
                    timestamp=datetime.now(),
                )
                print(f"[ERROR] LOSS MONITOR EXCEEDED: {device} = {value:.4f} "
                      f"> threshold {threshold:.4f}")
                return event
        return None

    def _missing_loss_monitors(self, loss_data, config) -> list:
        """Configured+enabled loss monitors with NO usable reading this step.

        A loss monitor is "missing" if its slot is absent/None, or its reading
        is non-finite (NaN/inf). Returned so the scan loop can track loss-monitor
        misses INDEPENDENTLY of the BPM slice and never silently treat an absent
        safety monitor as OK. Returns [] when the interlock is disabled.
        """
        if config is None or not config.enabled:
            return []
        expected = self._loss_monitor_devices_from_config(config)
        if not expected:
            return []
        readings = {}
        for item in loss_data:
            if isinstance(item, dict) and 'name' in item and 'data' in item:
                name = item['name']
                try:
                    val = float(item['data'])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(val):
                    readings[name] = val
        return [dev for dev in expected if dev not in readings]

    def _on_closing(self):
            """Handle window closing event with proper cleanup."""
            self._closing = True
            print("[INFO] Shutting down application...")

            # Stop reading thread if active
            if self.is_reading:
                self.scanner.stop_thread('live_reading')

            # Stop scan if active
            self.stop_scan_flag.set()

            # Wait for scan thread to finish (gives finally block time to restore nominals)
            if self._scan_thread is not None and self._scan_thread.is_alive():
                self._scan_thread.join(timeout=3.0)

            # Safety net: if scan thread didn't finish, force restore nominals now
            if self._scan_thread is not None and self._scan_thread.is_alive():
                print("[WARNING] Scan thread still running — forcing nominal restoration")
                try:
                    self._restore_nominal_settings(
                        self._last_run_config.role if self._last_run_config else "OPERATOR"
                    )
                except CredentialExpiredError:
                    print("[WARNING] Credential expired during shutdown restore — attempting renewal...")
                    if self._renew_kerberos_ticket():
                        try:
                            self._restore_nominal_settings(
                                self._last_run_config.role if self._last_run_config else "OPERATOR"
                            )
                        except Exception as e2:
                            print(f"[ERROR] CRITICAL: Nominal restore failed after renewal: {e2}")
                            print("[ERROR] MANUALLY VERIFY DEVICE SETTINGS!")
                    else:
                        print("[ERROR] CRITICAL: Credential renewal failed during shutdown. "
                              "MANUALLY VERIFY DEVICE SETTINGS!")
                except Exception as e:
                    print(f"[ERROR] Safety-net nominal restore failed: {e}")

            # Stop all ACNET threads
            self.scanner.stop_all_threads()

            # Terminate analysis processes
            for proc in getattr(self, '_analysis_processes', []):
                try:
                    if proc.is_alive():
                        proc.terminate()
                    proc.join(timeout=1.0)
                except Exception:
                    pass

            # Wait for background file operations
            from utils.file_operations import BackgroundFileOps
            BackgroundFileOps.wait_for_completion(timeout=3.0)

            # Give threads time to stop
            time.sleep(0.2)

            print("[INFO] Cleanup complete. Exiting...")
            self.destroy()

if __name__ == "__main__":
    print("[INFO] Starting FORMA...")
    app = DeviceControlApp()
    app.protocol("WM_DELETE_WINDOW", app._on_closing)
    app.mainloop()
