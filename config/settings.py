"""
Application Configuration Settings
"""

# UI Color Palette
PALETTE = {
    'background': '#0f172a',
    'surface': '#1e293b',
    'card': '#16213c',
    'accent': '#38bdf8',
    'accent_hover': '#0ea5e9',
    'text': '#f8fafc',
    'muted_text': '#cbd5f5',
    'border': '#334155',
    'success': '#22c55e',
    'warning': '#fbbf24',
    'error': '#f87171',
}

# Font Configuration
DEFAULT_FONT_FAMILY = 'Segoe UI'
DEFAULT_FONT_SIZE = 10
HEADING_FONT_SIZE = 11

# ACNET Configuration
ACNET_ROLES = ["OPERATOR", "testing", "linac_trims", "linac_quads"]
DEFAULT_ACNET_ROLE = "OPERATOR"
DEFAULT_ACNET_EVENT = "@p,1000"

# Scan Configuration
DEFAULT_POINTS_PER_SUPERPERIOD = 100
DEFAULT_NUM_SUPERPERIODS = 1
DEFAULT_RMS_SAMPLES = 50
DEFAULT_SCAN_MODE = "Simultaneous"

# Thread Configuration
THREAD_TIMEOUT = 60
THREAD_JOIN_TIMEOUT = 5.0
DATA_QUEUE_POLL_INTERVAL = 100  # milliseconds

# ACNET operation timeouts (seconds)
ACNET_OPERATION_TIMEOUT = 30.0       # Hard outer timeout per acsys.run_client() call
ACNET_MAX_CONSECUTIVE_TIMEOUTS = 5   # Auto-abort scan after this many consecutive empty reads

# Relative tolerance for confirming a device returned to nominal after a restore
# (commanded .SETTING vs read-back .SETTING@i). Catches "restore didn't run" /
# "device far off"; loose enough for device-side quantization. Tunable.
ACNET_RESTORE_VERIFY_TOLERANCE = 1e-3

# Plot Configuration
MAX_PLOT_DATA_POINTS = 10000  # Maximum points to keep in memory per device
PLOT_UPDATE_INTERVAL = 100  # milliseconds

# Kerberos Configuration
KERBEROS_TIMEOUT = 60  # seconds

# Safety Configuration
SAFETY_ENABLED_BY_DEFAULT = True
SAFETY_LOG_DIR = "safety_logs"
SAFETY_LOG_FILE = "safety_events.json"

# Default safety thresholds (in units of sigma/standard deviation)
DEFAULT_PER_DEVICE_WARNING_THRESHOLD = 3.0  # 3-sigma mean shift triggers warning
DEFAULT_PER_DEVICE_ABORT_THRESHOLD = 5.0    # 5-sigma mean shift triggers abort
DEFAULT_OVERALL_WARNING_THRESHOLD = 2.5     # 2.5-sigma overall mean shift warning
DEFAULT_OVERALL_ABORT_THRESHOLD = 4.0       # 4-sigma overall mean shift abort

# Baseline measurement configuration
DEFAULT_BASELINE_SAMPLES = 50  # Number of samples for baseline measurement

# Safety monitoring buffer configuration
DEFAULT_SAFETY_BUFFER_SIZE = 100  # Number of samples to average for current mean (1 = no averaging)

# Safety UI colors
SAFETY_ACTIVE_COLOR = '#22c55e'    # Green when monitoring
SAFETY_WARNING_COLOR = '#fbbf24'   # Yellow for warnings
SAFETY_ABORT_COLOR = '#ef4444'     # Red for aborts
SAFETY_DISABLED_COLOR = '#94a3b8'  # Gray when disabled

# =============================================================================
# Beam Interlock Settings
# =============================================================================

BEAM_CONTROL_DEVICE = "L:BSTUDY"
BEAM_STATUS_DRF = "L:BSTUDY.STATUS"
BEAM_CONTROL_DRF = "L:BSTUDY.CONTROL@N"
BEAM_OFF_VALUE = "off"   # Value sent to L:BSTUDY.CONTROL to disable beam
BEAM_ON_VALUE = "on"     # Value sent to L:BSTUDY.CONTROL to enable beam
BEAM_INTERLOCK_ENABLED_BY_DEFAULT = False

# Default loss monitor thresholds (device -> max allowed value)
DEFAULT_LOSS_MONITORS = {
    "L:DELM2": 25.0,
    "L:DELM5": 10.0,
    "L:400SCA": 0.2,
    "L:D7LMSM": 40.0,
}
