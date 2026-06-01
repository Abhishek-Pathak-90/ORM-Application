"""
acsys_connection_monitor.py  --  Read-only ACSys connectivity diagnostic for FORMA

WHAT IT DOES
------------
Opens ONE persistent ACSys connection and streams a single device -- default
G:AMANDA at 5 Hz -- using FORMA's OWN backend (backend/acnet_scanner.py). This
is the exact thing a FORMA scan does: one DPMContext, data streaming through
it. The monitor then watches that stream and records, in detail:

  * every reading received (count, rate, value)
  * every STALL  -- the connection is alive but no data arrives for a while
  * every DROP   -- the connection / read thread dies entirely
  * every RECOVERY, with the outage duration

At the end it prints a session summary: how many readings arrived vs. how many
were expected, the failure rate, and every outage window with its duration.

It answers one question: over a long run, how often does FORMA lose ACSys?

Why streaming (and not a fetch-in-a-loop)? Calling a one-shot fetch repeatedly
opens a new socket every time; Windows then runs out of ephemeral ports
(WinError 10048) -- a local artifact that has nothing to do with ACSys. One
persistent connection has no such churn and matches what a real scan does.

SAFETY  --  STRICTLY READ-ONLY
------------------------------
It only calls AcnetScanner.start_read_thread(), which performs a streaming
READ. It never calls apply_settings_once(), enable_beam(), disable_beam(), or
any other write/command path -- those are never invoked.

USAGE
-----
  python acsys_connection_monitor.py
  python acsys_connection_monitor.py --device G:AMANDA --duration 3600
  python acsys_connection_monitor.py --device "G:AMANDA@p,200" --logfile diag.log

Run it from the FORMA_v2 folder, in the SAME Python environment RUN_FORMA.bat
uses. Leave it running to capture intermittent drops. Press Ctrl+C (or use
--duration) to stop -- the session summary is printed and written to the log.
Share the log file with Controls.
"""

import argparse
import getpass
import os
import platform
import queue
import re
import socket
import sys
import threading
import time
import traceback
from datetime import datetime

# Import FORMA's backend. This also installs FORMA's asyncio-noise suppression
# (acnet_scanner sets sys.unraisablehook at import time), so the log stays
# clean without us having to re-implement any of that.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import acsys
    from backend.acnet_scanner import AcnetScanner
except Exception as exc:
    sys.stderr.write(
        "ERROR: could not import FORMA's backend (backend.acnet_scanner):\n"
        "  %s\n"
        "Run this script from the FORMA_v2 folder, with the same Python\n"
        "environment that RUN_FORMA.bat uses.\n" % exc
    )
    sys.exit(2)

THREAD_NAME = "acsys_monitor"
POLL_INTERVAL = 0.5       # seconds between draining the reading queue
HEARTBEAT_SECONDS = 20    # how often to print a status line while running
RESTART_DELAY = 2.0       # seconds to wait between reconnect attempts


class _Tee:
    """Mirror a stream to the console AND the log file, so acsys's own output
    (e.g. '*** unable to connect to ACSys') and FORMA's [WARNING]/[ERROR] lines
    are all captured for Controls. Locked so concurrent writers don't
    interleave mid-line."""

    def __init__(self, stream, logfile):
        self._stream = stream
        self._logfile = logfile
        self._lock = threading.Lock()

    def write(self, text):
        with self._lock:
            self._stream.write(text)
            self._stream.flush()
            self._logfile.write(text)
            self._logfile.flush()

    def flush(self):
        with self._lock:
            self._stream.flush()
            self._logfile.flush()


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _fmt_dur(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return "%dh %dm %ds" % (h, m, s)
    if m:
        return "%dm %ds" % (m, s)
    return "%ds" % s


def _print_error_detail(error, indent="    "):
    """Dump everything known about a failure -- this is what Controls needs to
    decode an opaque ACNET error like [1 -34]."""
    print("%serror type   : %s" % (indent, type(error).__name__))
    print("%serror module : %s" % (indent, getattr(type(error), "__module__", "?")))
    print("%serror str    : %s" % (indent, str(error) if str(error) else "<empty>"))
    print("%serror repr   : %r" % (indent, error))
    for attr in ("args", "status", "errno", "code", "facility", "error_code",
                 "acnet_status", "message"):
        if hasattr(error, attr):
            try:
                print("%serror.%-7s: %r" % (indent, attr, getattr(error, attr)))
            except Exception:
                pass
    if getattr(error, "__traceback__", None) is not None:
        print("%straceback:" % indent)
        tb = traceback.format_exception(type(error), error, error.__traceback__)
        for line in "".join(tb).rstrip().splitlines():
            print("%s  %s" % (indent, line))


def main():
    parser = argparse.ArgumentParser(
        description="Read-only ACSys connectivity monitor. Streams one device "
                    "through FORMA's own backend (one persistent connection, "
                    "exactly like a scan) and logs every stall and drop.")
    parser.add_argument("--device", default="G:AMANDA@p,200",
                        help="device DRF to stream (default: G:AMANDA@p,200 -- "
                             "G:AMANDA sampled periodically at 200 ms = 5 Hz). "
                             "If you pass a name with no '@event', '@p,200' is "
                             "appended.")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="stop after this many seconds (default: 0 = run "
                             "until Ctrl+C)")
    parser.add_argument("--stall-seconds", type=float, default=3.0,
                        help="if no reading arrives for this long while the "
                             "connection is still alive, count it as a stall "
                             "(default: 3.0)")
    parser.add_argument("--logfile", default=None,
                        help="log file path (default: "
                             "acsys_monitor_<timestamp>.log in current dir)")
    args = parser.parse_args()

    drf = args.device if "@" in args.device else args.device + "@p,200"

    # Expected sample period, parsed from an '@p,<ms>' event, used for the
    # "readings received vs. expected" statistic. None if not a periodic DRF.
    m = re.search(r"@p,(\d+)", drf)
    expected_period = (int(m.group(1)) / 1000.0) if m else None

    logpath = os.path.abspath(
        args.logfile or
        "acsys_monitor_%s.log" % datetime.now().strftime("%Y%m%d_%H%M%S"))
    logfile = open(logpath, "a", encoding="utf-8", buffering=1)
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_stdout, logfile)
    sys.stderr = _Tee(orig_stderr, logfile)

    scanner = AcnetScanner()

    try:
        host = socket.gethostname()
    except Exception:
        host = "<unknown>"

    print("=" * 72)
    print(" ACSys CONNECTION MONITOR   (read-only diagnostic for FORMA)")
    print("=" * 72)
    print(" Started    : %s   (local time)" % _ts())
    print(" Host       : %s" % host)
    print(" User       : %s" % getpass.getuser())
    print(" OS         : %s" % platform.platform())
    print(" Python     : %s" % sys.version.replace("\n", " "))
    print(" acsys ver  : %s" % getattr(acsys, "__version__", "<no __version__>"))
    print(" acsys path : %s" % getattr(acsys, "__file__", "<unknown>"))
    print(" Streaming  : %s" % drf)
    if expected_period:
        print(" Expected   : 1 reading every %.3f s  (%.2f Hz)"
              % (expected_period, 1.0 / expected_period))
    print(" Stall after: %.1f s of silence (connection alive but no data)"
          % args.stall_seconds)
    print(" Duration   : %s"
          % (_fmt_dur(args.duration) if args.duration
             else "unlimited (Ctrl+C to stop)"))
    print(" Log file   : %s" % logpath)
    print(" Method     : backend.acnet_scanner.AcnetScanner.start_read_thread()")
    print("              -- ONE persistent connection, the same way a scan runs")
    print("=" * 72)
    print(" SAFETY: strictly READ-ONLY. Only a streaming device read is opened.")
    print("         No settings, no beam control, no writes of any kind.")
    print("=" * 72)
    print("")

    # ---- streaming plumbing -------------------------------------------------
    data_queue = queue.Queue()
    stream_ended = threading.Event()      # set by on_complete when the read thread dies
    err_holder = {"exc": None}            # last error reported by the read thread

    def _on_error(_thread_name, exc):
        err_holder["exc"] = exc

    def _on_complete():
        stream_ended.set()

    scanner.set_error_callback(_on_error)

    def _start_stream():
        err_holder["exc"] = None
        stream_ended.clear()
        scanner.start_read_thread(THREAD_NAME, [drf], data_queue,
                                  on_complete=_on_complete)

    # ---- running state ------------------------------------------------------
    readings = 0
    up = None                       # None = no data yet, True = up, False = down
    last_value = None
    last_reading_dt = None
    last_reading_mono = time.monotonic()
    outage_start = None             # datetime an outage began
    outage_cause = None             # "stall" / "drop" description
    restarts_this_outage = 0
    total_restarts = 0
    outages = []                    # list of (start_dt, end_dt, cause, restarts)
    session_start = datetime.now()
    session_mono = time.monotonic()

    def print_summary():
        end = datetime.now()
        elapsed = time.monotonic() - session_mono
        print("")
        print("=" * 72)
        print(" SESSION SUMMARY")
        print("=" * 72)
        print(" Started        : %s" % session_start.strftime("%Y-%m-%d %H:%M:%S"))
        print(" Ended          : %s" % end.strftime("%Y-%m-%d %H:%M:%S"))
        print(" Duration       : %s" % _fmt_dur(elapsed))
        print(" Device         : %s" % drf)
        print(" Readings rec'd : %d" % readings)
        if expected_period and elapsed > 0:
            expected = elapsed / expected_period
            pct = (100.0 * readings / expected) if expected > 0 else 0.0
            print(" Expected       : ~%d  (at %.2f Hz for %s)"
                  % (int(expected), 1.0 / expected_period, _fmt_dur(elapsed)))
            print(" Received       : %.2f%% of expected readings" % pct)
            print(" Missed         : ~%d readings (%.2f%%)"
                  % (max(0, int(expected) - readings),
                     max(0.0, 100.0 - pct)))
        if readings and elapsed > 0:
            print(" Average rate   : %.2f Hz" % (readings / elapsed))
        if last_value is not None:
            print(" Last value     : %s = %r" % (drf.split("@")[0], last_value))
        all_outages = list(outages)
        if outage_start is not None:
            all_outages.append((outage_start, end, outage_cause,
                                restarts_this_outage))
        if all_outages:
            total_outage = sum((e - s).total_seconds() for s, e, _, _ in all_outages)
            print(" Outages        : %d   (connection lost %d time(s) this run)"
                  % (len(all_outages), len(all_outages)))
            print(" Total downtime : %s" % _fmt_dur(total_outage))
            print(" Reconnect tries: %d" % total_restarts)
            for i, (s, e, cause, rs) in enumerate(all_outages, 1):
                print("   %2d. %s  ->  %s   (%s)"
                      % (i, s.strftime("%Y-%m-%d %H:%M:%S"),
                         e.strftime("%H:%M:%S"),
                         _fmt_dur((e - s).total_seconds())))
                print("       cause: %s" % cause)
                if rs:
                    print("       reconnect attempts during this outage: %d" % rs)
        else:
            print(" Outages        : none -- the stream was unbroken all session")
        print("=" * 72)
        print(" Share this log file with Controls: %s" % logpath)
        print("=" * 72)

    # ---- start streaming ----------------------------------------------------
    print("[%s] opening ACSys stream for %s ..." % (_ts(), drf))
    _start_stream()
    last_heartbeat = time.monotonic()

    try:
        while True:
            if args.duration and (time.monotonic() - session_mono) >= args.duration:
                print("")
                print("[%s] reached duration limit (%s) -- stopping."
                      % (_ts(), _fmt_dur(args.duration)))
                break

            # Drain every reading currently in the queue.
            got = 0
            while True:
                try:
                    item = data_queue.get_nowait()
                except queue.Empty:
                    break
                got += 1
                readings += 1
                try:
                    last_value = item[0]["data"]
                except (IndexError, KeyError, TypeError):
                    last_value = item
                last_reading_dt = datetime.now()

            now_mono = time.monotonic()
            now_dt = datetime.now()

            if got:
                last_reading_mono = now_mono
                if up is False:
                    # recovery: an outage just ended
                    outages.append((outage_start, now_dt, outage_cause,
                                    restarts_this_outage))
                    print("")
                    print("[%s] *** CONNECTION RESTORED ***" % _ts())
                    print("      was down since : %s"
                          % outage_start.strftime("%Y-%m-%d %H:%M:%S"))
                    print("      outage lasted  : %s"
                          % _fmt_dur((now_dt - outage_start).total_seconds()))
                    print("      cause was      : %s" % outage_cause)
                    if restarts_this_outage:
                        print("      reconnect tries: %d" % restarts_this_outage)
                    print("      data flowing   : %s = %r"
                          % (drf.split("@")[0], last_value))
                    print("")
                    outage_start = None
                    outage_cause = None
                    restarts_this_outage = 0
                elif up is None:
                    print("[%s] stream is UP -- first reading: %s = %r"
                          % (_ts(), drf.split("@")[0], last_value))
                up = True

            else:
                silence = now_mono - last_reading_mono

                if stream_ended.is_set():
                    # The read thread died -- connection dropped, or a
                    # reconnect attempt failed to connect at all.
                    if up is not False:
                        outage_start = now_dt
                        outage_cause = "connection dropped (read thread ended)"
                        up = False
                        print("")
                        print("[%s] *** CONNECTION LOST -- read thread ended ***"
                              % _ts())
                        if last_reading_dt:
                            print("      last reading at: %s"
                                  % last_reading_dt.strftime("%Y-%m-%d %H:%M:%S"))
                        if err_holder["exc"] is not None:
                            print("      the read thread failed with:")
                            _print_error_detail(err_holder["exc"], indent="      ")
                        else:
                            print("      the read thread ended without raising an "
                                  "error")
                            print("      (typically a >60 s silence -- the "
                                  "connection stopped delivering data)")
                        print("")
                    # Try to reconnect (paced by RESTART_DELAY so a sustained
                    # outage doesn't tight-loop).
                    restarts_this_outage += 1
                    total_restarts += 1
                    print("[%s] reconnect attempt #%d (outage attempt #%d) ..."
                          % (_ts(), total_restarts, restarts_this_outage))
                    scanner.stop_thread(THREAD_NAME)
                    time.sleep(RESTART_DELAY)
                    _start_stream()
                    last_reading_mono = time.monotonic()  # grace for the new stream

                elif silence >= args.stall_seconds and up is not False:
                    # Connection still alive (thread running) but no data.
                    outage_start = now_dt
                    outage_cause = ("stream stalled -- connection alive but no "
                                    "data for %.1f s" % silence)
                    up = False
                    print("")
                    print("[%s] *** CONNECTION STALLED ***" % _ts())
                    print("      no reading for : %.1f s "
                          "(connection still open, but data stopped)" % silence)
                    if last_reading_dt:
                        print("      last reading at: %s"
                              % last_reading_dt.strftime("%Y-%m-%d %H:%M:%S"))
                    print("")

            # Periodic heartbeat so a healthy run still has a visible pulse,
            # and a long outage still shows it is being watched.
            if now_mono - last_heartbeat >= HEARTBEAT_SECONDS:
                elapsed = now_mono - session_mono
                if up is True:
                    if expected_period and elapsed > 0:
                        expected = elapsed / expected_period
                        pct = (100.0 * readings / expected) if expected > 0 else 0.0
                        print("[%s] heartbeat: UP -- %d readings, %.1f%% of "
                              "expected, last %s = %r"
                              % (_ts(), readings, pct, drf.split("@")[0],
                                 last_value))
                    else:
                        print("[%s] heartbeat: UP -- %d readings, last %s = %r"
                              % (_ts(), readings, drf.split("@")[0], last_value))
                else:
                    down_for = ((now_dt - outage_start).total_seconds()
                                if outage_start else 0.0)
                    print("[%s] heartbeat: DOWN for %s -- %d reconnect attempt(s)"
                          % (_ts(), _fmt_dur(down_for), restarts_this_outage))
                last_heartbeat = now_mono

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("")
        print("[%s] interrupted by user (Ctrl+C) -- stopping." % _ts())
    except Exception as exc:
        # A bug in the monitor itself should still produce a usable log.
        print("")
        print("[%s] MONITOR SCRIPT INTERNAL ERROR (not an ACSys failure):" % _ts())
        _print_error_detail(exc)
    finally:
        try:
            scanner.stop_thread(THREAD_NAME)
        except Exception:
            pass
        print_summary()
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        try:
            logfile.close()
        except Exception:
            pass
        print("Monitor stopped. Diagnostic log saved to:")
        print("  %s" % logpath)


if __name__ == "__main__":
    main()
