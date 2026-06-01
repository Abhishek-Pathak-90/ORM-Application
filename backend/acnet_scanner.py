"""
ACNET Communication Backend

Handles all backend ACNET communication using acsys.dpm.
This module runs requests in separate threads to keep the GUI responsive.
"""

import threading
import asyncio
import concurrent.futures
import sys
import time
import acsys.dpm
from typing import List, Callable, Optional, Dict

from config.settings import ACNET_OPERATION_TIMEOUT, BEAM_OFF_VALUE, BEAM_ON_VALUE


class CredentialExpiredError(Exception):
    """Raised when a Kerberos/GSS credential has expired during an ACNET operation."""
    pass


def _quiet_exception_handler(_loop, context):
    """Suppress asyncio noise from DPM task teardown after loop closure."""
    msg = context.get('message', '')
    if 'Task was destroyed but it is pending' in msg:
        return
    exc = context.get('exception')
    if isinstance(exc, RuntimeError) and 'Event loop is closed' in str(exc):
        return
    _loop.default_exception_handler(context)


# "Event loop is closed" unraisable errors are the fingerprint of an orphaned
# event loop — exactly the symptom that hid the v0->v2 connection-leak bug.
# After the persistent-executor fix they should not occur; if one does, LOG it
# (do not silently swallow) so this class of problem can never hide again.
_original_unraisablehook = sys.unraisablehook


def _log_unraisablehook(unraisable):
    exc = unraisable.exc_value
    if isinstance(exc, RuntimeError) and 'Event loop is closed' in str(exc):
        # Write to the real console stderr, not sys.stderr — the app redirects
        # sys.stderr to a Tk widget, and this hook can fire during teardown
        # when that widget is already destroyed.
        try:
            print("[WARNING] 'Event loop is closed' unraisable exception — should "
                  "not occur after the persistent-executor fix; investigate if it "
                  "recurs (this was the v0->v2 connection-leak fingerprint).",
                  file=sys.__stderr__)
        except Exception:
            pass
        return
    _original_unraisablehook(unraisable)


sys.unraisablehook = _log_unraisablehook


class AcnetScanner:
    """
    Handles all backend ACNET communication using acsys.dpm.
    This class runs requests in separate threads to keep the GUI responsive.
    """

    def __init__(self):
        self.thread_dict = {}
        self._dict_lock = threading.Lock()
        self._error_callback = None
        self.dpm_node = None  # None = auto-discover via multicast
        # All acsys.run_client calls share ONE persistent worker thread, hence
        # ONE event loop. acsys defers connection teardown onto the loop, so
        # reusing the loop lets each call flush the previous call's teardown
        # (the way Scheduler_v5 / FORMAv0 do). A fresh thread/loop per call
        # strands that teardown and leaks sockets until GC.
        self._executor_lock = threading.Lock()
        self._acnet_executor = self._new_acnet_executor()

    def set_error_callback(self, callback: Callable):
        """Set callback for error notifications."""
        self._error_callback = callback

    @staticmethod
    def _install_loop_cleanup_handler():
        """Install handler to suppress 'Event loop is closed' noise during teardown."""
        asyncio.get_running_loop().set_exception_handler(_quiet_exception_handler)

    @staticmethod
    def _new_acnet_executor():
        """Single-thread executor for acsys.run_client. max_workers=1 is
        load-bearing: it pins every ACNET call to one thread / one event loop."""
        return concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='acnet-call')

    def _submit_acnet_call(self, async_fn, kwargs):
        """Submit acsys.run_client to the shared executor, recreating it if it
        was shut down — by close() on app exit, or retired after a prior hard
        timeout. Keeps the scanner usable for safety-critical restores even
        during shutdown. Returns (executor, future)."""
        with self._executor_lock:
            executor = self._acnet_executor
        try:
            return executor, executor.submit(acsys.run_client, async_fn, **kwargs)
        except RuntimeError:
            with self._executor_lock:
                if self._acnet_executor is executor:
                    self._acnet_executor = self._new_acnet_executor()
                executor = self._acnet_executor
            return executor, executor.submit(acsys.run_client, async_fn, **kwargs)

    def _call_with_timeout(self, async_fn, timeout, default_return=None,
                           abort_check=None, **kwargs):
        """Run acsys.run_client with a hard outer timeout to prevent indefinite blocking.

        Submits to the shared persistent executor so every call reuses one
        worker thread and one event loop (see __init__). Waits for the result
        in short slices: if `abort_check` is supplied and returns True (e.g. the
        user pressed Stop), the call returns `default_return` promptly instead
        of blocking for the full timeout. Wraps credential-expired GSS errors as
        CredentialExpiredError. All other exceptions propagate to the caller so
        existing try/except blocks continue to work — safety-critical nominal
        restores pass no abort_check, so they always run to completion.
        """
        executor, future = self._submit_acnet_call(async_fn, kwargs)
        deadline = time.monotonic() + timeout
        aborted = False
        while True:
            if abort_check is not None and abort_check():
                aborted = True
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                return future.result(timeout=min(remaining, 0.5))
            except concurrent.futures.TimeoutError:
                continue  # slice expired — re-check abort_check and deadline
            except Exception as e:
                err_lower = str(e).lower()
                is_credential_error = (
                    ('credential' in err_lower and 'expired' in err_lower) or
                    ('ticket' in err_lower and 'expired' in err_lower) or
                    ('gss' in err_lower and 'expired' in err_lower)
                )
                if is_credential_error:
                    raise CredentialExpiredError(str(e)) from e
                raise
        # Loop exited without a result: aborted, or hard timeout. The worker is
        # still running run_client, so retire this executor (its stuck thread
        # goes with it) and install a fresh one — only on this rare path, not
        # the normal per-call flow.
        if aborted:
            print("[INFO] ACNET operation interrupted by stop request")
        else:
            print(f"[WARNING] ACNET operation timed out after {timeout}s — connection may be stale")
        with self._executor_lock:
            if self._acnet_executor is executor:
                self._acnet_executor = self._new_acnet_executor()
                retired = executor
            else:
                retired = None
        if retired is not None:
            retired.shutdown(wait=False)
        return default_return

    def _acnet_daq_thread_target(self, thread_name):
        """The function that is run in a separate thread to handle DAQ."""
        with self._dict_lock:
            context = self.thread_dict.get(thread_name)
        if context is None:
            print(f"[WARNING] Thread context for '{thread_name}' was not found.")
            return
        try:
            if context['task'] == 'read_many':
                acsys.run_client(self._read_many_async, thread_context=context)
        except Exception as e:
            error_msg = f"[ERROR] Thread '{thread_name}' encountered error: {repr(e)}"
            print(error_msg, file=sys.stderr)
            if self._error_callback:
                self._error_callback(thread_name, e)
        finally:
            context['stop'].set()
            print(f"Thread '{thread_name}' has finished.")
            try:
                if context.get('on_complete'):
                    context['on_complete']()
            except Exception as cb_err:
                print(f"[ERROR] Thread '{thread_name}' on_complete callback failed: {repr(cb_err)}", file=sys.stderr)
            with self._dict_lock:
                self.thread_dict.pop(thread_name, None)

    def start_read_thread(self, thread_name, param_list, data_queue, on_complete=None):
        """Starts a non-blocking thread to continuously read devices."""
        print(f"Starting read thread '{thread_name}'...")
        if self.is_thread_running(thread_name):
            print(f"Thread '{thread_name}' is already running.")
            return

        daq_thread = threading.Thread(
            target=self._acnet_daq_thread_target,
            args=(thread_name,),
            daemon=True  # Make daemon so it closes with main app
        )
        with self._dict_lock:
            self.thread_dict[thread_name] = {
                'thread': daq_thread,
                'lock': threading.Lock(),
                'timeout': 60,
                'data_queue': data_queue,
                'param_list': param_list,
                'stop': threading.Event(),
                'task': 'read_many',
                'on_complete': on_complete
            }
        daq_thread.start()

    def is_thread_running(self, thread_name):
        """Check if a thread is currently running."""
        with self._dict_lock:
            return thread_name in self.thread_dict and self.thread_dict[thread_name]['thread'].is_alive()

    def stop_thread(self, thread_name):
        """Stops a running thread."""
        if self.is_thread_running(thread_name):
            print(f"Stopping thread '{thread_name}'...")
            with self._dict_lock:
                context = self.thread_dict.get(thread_name)
            if context:
                context['stop'].set()
                context['thread'].join(timeout=2.0)
            with self._dict_lock:
                self.thread_dict.pop(thread_name, None)
        else:
            print(f"Thread '{thread_name}' not found or not running.")

    def stop_all_threads(self):
        """Stop all running threads. Call this on application exit."""
        print("[INFO] Stopping all ACNET threads...")
        with self._dict_lock:
            thread_names = list(self.thread_dict.keys())
        for thread_name in thread_names:
            self.stop_thread(thread_name)
        self.close()
        print("[INFO] All ACNET threads stopped.")

    def close(self):
        """Release the persistent ACNET-call executor. Called on app exit.
        A later call (e.g. a safety-net nominal restore) will transparently
        recreate it via _submit_acnet_call."""
        with self._executor_lock:
            self._acnet_executor.shutdown(wait=False)

    async def _read_many_async(self, con, thread_context):
        """Async function to continuously read device values and put them on a queue."""
        self._install_loop_cleanup_handler()
        async with acsys.dpm.DPMContext(con, dpm_node=self.dpm_node) as dpm:
            await dpm.add_entries(list(enumerate(thread_context['param_list'])))
            if thread_context['stop'].is_set():
                return
            await dpm.start()
            if thread_context['stop'].is_set():
                return
            try:
                async for reply in dpm.replies(tmo=60.0):
                    if thread_context['stop'].is_set():
                        break
                    if reply.isReading:
                        if 0 <= reply.tag < len(thread_context['param_list']):
                            thread_context['data_queue'].put([{
                                'stamp': reply.stamp,
                                'name': thread_context['param_list'][reply.tag].split('@')[0],
                                'data': reply.data
                            }])
                        else:
                            print(f"[WARNING] Received reply with out-of-range tag: {reply.tag}")
            except asyncio.TimeoutError:
                if not thread_context['stop'].is_set():
                    print(f"[WARNING] Read timeout in continuous reading - connection may have dropped")
            except Exception as e:
                print(f"[ERROR] Error in read_many loop: {repr(e)}", file=sys.stderr)
                raise

    async def _get_settings_async(self, con, drf_list):
        """Async function to read setting values once."""
        self._install_loop_cleanup_handler()
        settings = [None] * len(drf_list)
        async with acsys.dpm.DPMContext(con, dpm_node=self.dpm_node) as dpm:
            await dpm.add_entries(list(enumerate(drf_list)))
            await dpm.start()
            try:
                async for reply in dpm.replies(tmo=10.0):
                    if reply.isReading:
                        if 0 <= reply.tag < len(settings):
                            settings[reply.tag] = reply.data
                    if None not in settings:
                        break
            except asyncio.TimeoutError:
                print("[WARNING] Timed out waiting for some device settings.")
        return settings

    async def _set_once_async(self, con, drf_list, value_list, settings_role):
        """Async function to set device values once."""
        self._install_loop_cleanup_handler()
        async with acsys.dpm.DPMContext(con, dpm_node=self.dpm_node) as dpm:
            await dpm.enable_settings(role=settings_role)
            await dpm.add_entries(list(enumerate(drf_list)))
            setpairs = list(enumerate(value_list))
            await dpm.apply_settings(setpairs)
            await dpm.start()
            statuses_received = 0
            try:
                async for reply in dpm.replies(tmo=5.0):
                    if reply.isStatus:
                        statuses_received += 1
                        if statuses_received >= len(drf_list):
                            break
            except asyncio.TimeoutError:
                print(f"[WARNING] Timed out waiting for settings confirmation "
                      f"({statuses_received}/{len(drf_list)} confirmed).")

    async def _read_once_on_event_async(self, con, drf_list):
        """Reads a list of devices upon the next event."""
        self._install_loop_cleanup_handler()
        data = [None] * len(drf_list)
        names = [drf.split('@')[0] for drf in drf_list]
        async with acsys.dpm.DPMContext(con, dpm_node=self.dpm_node) as dpm:
            await dpm.add_entries(list(enumerate(drf_list)))
            await dpm.start()
            try:
                async for reply in dpm.replies(tmo=10.0):
                    if reply.isReading:
                        if 0 <= reply.tag < len(data):
                            data[reply.tag] = {
                                'stamp': reply.stamp,
                                'name': names[reply.tag],
                                'data': reply.data
                            }
                    if all(item is not None for item in data):
                        break
            except asyncio.TimeoutError:
                print("[WARNING] Timed out waiting for ACNET event.")
        return data

    def get_settings_once(self, paramlist: List[str], abort_check=None) -> List:
        """Public method to get device settings. Runs the async version.

        abort_check: optional () -> bool; if it returns True the call returns
        promptly instead of blocking for the full timeout (used by scan loops
        so Stop is responsive). Restore/preflight callers omit it.
        """
        if not paramlist:
            print('[WARNING] Device list empty. Aborting get settings operation.')
            return []
        drf_list = [f'{dev}.SETTING@i' for dev in paramlist]
        return self._call_with_timeout(
            self._get_settings_async,
            timeout=ACNET_OPERATION_TIMEOUT,
            default_return=[None] * len(paramlist),
            abort_check=abort_check,
            drf_list=drf_list,
        )

    def apply_settings_once(self, paramlist: List[str], values: List, role: str, abort_check=None):
        """Public method to set device values. Runs the async version.

        abort_check: optional () -> bool for responsive Stop (see
        get_settings_once). Nominal-restore callers omit it so the restore
        always runs to completion.
        """
        if not paramlist:
            print('[WARNING] Device list empty. Aborting setting operation.')
            return
        drf_list = [f'{dev}.SETTING' for dev in paramlist]
        self._call_with_timeout(
            self._set_once_async,
            timeout=ACNET_OPERATION_TIMEOUT,
            default_return=None,
            abort_check=abort_check,
            drf_list=drf_list,
            value_list=values,
            settings_role=role,
        )

    def read_once_on_event(self, drf_list: List[str], abort_check=None) -> List:
        """Public method to read devices on the next event.

        abort_check: optional () -> bool for responsive Stop (see
        get_settings_once).
        """
        if not drf_list:
            return []
        return self._call_with_timeout(
            self._read_once_on_event_async,
            timeout=ACNET_OPERATION_TIMEOUT,
            default_return=[None] * len(drf_list),
            abort_check=abort_check,
            drf_list=drf_list,
        )

    # =========================================================================
    # DPM Discovery
    # =========================================================================

    async def _find_all_dpms_async(self, con):
        """Discover all active DPM nodes."""
        self._install_loop_cleanup_handler()
        return await acsys.dpm.available_dpms(con)

    def discover_dpms(self) -> List[str]:
        """Return a list of available DPM node names."""
        result = self._call_with_timeout(
            self._find_all_dpms_async,
            timeout=ACNET_OPERATION_TIMEOUT,
            default_return=[],
        )
        return result if result else []

    # =========================================================================
    # Beam Control (L:BSTUDY)
    # =========================================================================

    async def _check_beam_status_async(self, con, status_drf):
        """Read beam status."""
        self._install_loop_cleanup_handler()
        async with acsys.dpm.DPMContext(con, dpm_node=self.dpm_node) as dpm:
            await dpm.add_entries([(0, status_drf)])
            await dpm.start()
            try:
                async for reply in dpm.replies(tmo=10.0):
                    if reply.isReading:
                        return reply.data
            except asyncio.TimeoutError:
                print("[WARNING] Timeout reading beam status")
                return None
        return None

    def check_beam_status(self, status_drf: str, event: str = "@e,0A") -> Optional[bool]:
        """Check if studies beam is on.

        Returns:
            True if beam is on, False if off, None if unknown/error
        """
        full_drf = f"{status_drf}{event}"
        try:
            result = self._call_with_timeout(
                self._check_beam_status_async,
                timeout=ACNET_OPERATION_TIMEOUT,
                default_return=None,
                status_drf=full_drf)
            if result is None:
                return None
            if isinstance(result, dict):
                return result.get("on", False)
            return bool(result)
        except Exception as e:
            print(f"[WARNING] Failed to check beam status: {e}")
            return None

    async def _disable_beam_async(self, con, control_drf, settings_role):
        """Disable beam via L:BSTUDY.CONTROL."""
        self._install_loop_cleanup_handler()
        async with acsys.dpm.DPMContext(con, dpm_node=self.dpm_node) as dpm:
            await dpm.enable_settings(role=settings_role)
            await dpm.add_entries([(0, control_drf)])
            await dpm.apply_settings([(0, BEAM_OFF_VALUE)])
            await dpm.start()
            try:
                async for reply in dpm.replies(tmo=10.0):
                    if reply.isStatus:
                        break
            except asyncio.TimeoutError:
                print("[WARNING] Timed out waiting for beam disable confirmation")

    def disable_beam(self, control_drf: str, role: str):
        """Disable studies beam."""
        print(f"[WARNING] DISABLING BEAM via {control_drf} (role={role})")
        self._call_with_timeout(
            self._disable_beam_async,
            timeout=ACNET_OPERATION_TIMEOUT,
            default_return=None,
            control_drf=control_drf,
            settings_role=role)
        print("[INFO] Beam disable command sent")

    async def _enable_beam_async(self, con, control_drf, settings_role):
        """Enable beam via L:BSTUDY.CONTROL."""
        self._install_loop_cleanup_handler()
        async with acsys.dpm.DPMContext(con, dpm_node=self.dpm_node) as dpm:
            await dpm.enable_settings(role=settings_role)
            await dpm.add_entries([(0, control_drf)])
            await dpm.apply_settings([(0, BEAM_ON_VALUE)])
            await dpm.start()
            try:
                async for reply in dpm.replies(tmo=10.0):
                    if reply.isStatus:
                        break
            except asyncio.TimeoutError:
                print("[WARNING] Timed out waiting for beam enable confirmation")

    def enable_beam(self, control_drf: str, role: str):
        """Enable studies beam."""
        print(f"[INFO] ENABLING BEAM via {control_drf} (role={role})")
        self._call_with_timeout(
            self._enable_beam_async,
            timeout=ACNET_OPERATION_TIMEOUT,
            default_return=None,
            control_drf=control_drf,
            settings_role=role)
        print("[INFO] Beam enable command sent")
