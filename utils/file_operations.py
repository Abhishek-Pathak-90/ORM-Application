"""
Background File Operations

Provides threaded file I/O operations to prevent GUI freezing.
Uses non-daemon threads to prevent file corruption on app exit.
"""

import os
import tempfile
import threading
import pandas as pd
import json
from pathlib import Path
from typing import Callable, Optional, Any, Dict
from datetime import datetime


def atomic_write_text(path, text, encoding='utf-8'):
    """
    Atomically write ``text`` to ``path``.

    Writes to a temporary file in the SAME directory as the target, flushes and
    fsyncs to disk, then os.replace() onto the final path so a reader never sees
    a partially written file. The temp file is removed on error.

    Args:
        path: Destination file path (str or Path).
        text: String contents to write.
        encoding: Text encoding (default 'utf-8').

    Returns:
        The final path as a string.
    """
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=directory, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding=encoding) as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
        except BaseException:
            # fdopen took ownership of fd; if it failed before that, close fd.
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        os.replace(tmp_path, path)
        tmp_path = None
        return path
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def atomic_write_bytes(path, data):
    """
    Atomically write ``data`` (bytes) to ``path``.

    Writes to a temporary file in the SAME directory as the target, flushes and
    fsyncs to disk, then os.replace() onto the final path. The temp file is
    removed on error.

    Args:
        path: Destination file path (str or Path).
        data: Bytes-like object to write.

    Returns:
        The final path as a string.
    """
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=directory, suffix='.tmp')
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        os.replace(tmp_path, path)
        tmp_path = None
        return path
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def atomic_to_csv(df, path, **to_csv_kwargs):
    """
    Atomically write a DataFrame to ``path`` as CSV.

    Writes ``df`` to a temporary file in the SAME directory as the target, then
    os.replace() onto the final path so the destination is never partially
    written. Drop-in replacement for ``df.to_csv(path, **kwargs)``.

    Args:
        df: pandas DataFrame to write.
        path: Destination CSV path (str or Path).
        **to_csv_kwargs: Passed through to ``DataFrame.to_csv``.

    Returns:
        The final path as a string.
    """
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=directory, suffix='.tmp')
        os.close(fd)
        df.to_csv(tmp_path, **to_csv_kwargs)
        os.replace(tmp_path, path)
        tmp_path = None
        return path
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


class BackgroundFileOps:
    """Handle file operations in background threads."""

    _active_threads = []
    _threads_lock = threading.Lock()

    @classmethod
    def _track_thread(cls, thread):
        """Add a thread to the tracking list."""
        with cls._threads_lock:
            cls._active_threads.append(thread)

    @classmethod
    def _untrack_thread(cls, thread):
        """Remove a thread from the tracking list."""
        with cls._threads_lock:
            cls._active_threads = [t for t in cls._active_threads if t is not thread]

    @classmethod
    def wait_for_completion(cls, timeout=5.0):
        """Wait for all file operations to complete. Call on app exit."""
        with cls._threads_lock:
            threads = list(cls._active_threads)
        for t in threads:
            t.join(timeout=timeout)

    @staticmethod
    def save_dataframe_async(filepath: str, data: pd.DataFrame, callback: Optional[Callable] = None):
        """
        Save a DataFrame to CSV in a background thread.

        Args:
            filepath: Path to save the CSV file
            data: DataFrame to save
            callback: Optional callback function(success: bool, path_or_error: str)
        """
        def _save():
            try:
                data.to_csv(filepath, index=False)
                print(f"[SUCCESS] Data saved to {filepath}")
                if callback:
                    callback(True, filepath)
            except Exception as e:
                print(f"[ERROR] Failed to save data: {e}")
                if callback:
                    callback(False, str(e))
            finally:
                BackgroundFileOps._untrack_thread(thread)

        thread = threading.Thread(target=_save, daemon=False)
        BackgroundFileOps._track_thread(thread)
        thread.start()

    @staticmethod
    def save_json_async(filepath: str, data: Dict, callback: Optional[Callable] = None):
        """
        Save JSON data in a background thread.

        Args:
            filepath: Path to save the JSON file
            data: Dictionary to save as JSON
            callback: Optional callback function(success: bool, path_or_error: str)
        """
        def _save():
            try:
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"[SUCCESS] JSON saved to {filepath}")
                if callback:
                    callback(True, filepath)
            except Exception as e:
                print(f"[ERROR] Failed to save JSON: {e}")
                if callback:
                    callback(False, str(e))
            finally:
                BackgroundFileOps._untrack_thread(thread)

        thread = threading.Thread(target=_save, daemon=False)
        BackgroundFileOps._track_thread(thread)
        thread.start()

    @staticmethod
    def load_json_async(filepath: str, callback: Callable):
        """
        Load JSON data in a background thread.

        Args:
            filepath: Path to load the JSON file from
            callback: Callback function(success: bool, data: dict or error_msg: str)
        """
        def _load():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                callback(True, data)
            except Exception as e:
                print(f"[ERROR] Failed to load JSON: {e}")
                callback(False, str(e))
            finally:
                BackgroundFileOps._untrack_thread(thread)

        thread = threading.Thread(target=_load, daemon=False)
        BackgroundFileOps._track_thread(thread)
        thread.start()

    @staticmethod
    def save_scan_data(base_path: str, scan_data: pd.DataFrame, metadata: Dict,
                       callback: Optional[Callable] = None):
        """
        Save scan data with metadata in a background thread.

        Args:
            base_path: Base path for saving (without extension)
            scan_data: DataFrame containing scan data
            metadata: Dictionary containing scan metadata
            callback: Optional callback function(success: bool, csv_path_or_error: str, json_path: str or None)
        """
        def _save():
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path = f"{base_path}_{timestamp}.csv"
                json_path = f"{base_path}_{timestamp}_meta.json"

                scan_data.to_csv(csv_path, index=False)
                with open(json_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

                print(f"[SUCCESS] Scan data saved to {csv_path}")
                print(f"[SUCCESS] Metadata saved to {json_path}")

                if callback:
                    callback(True, csv_path, json_path)
            except Exception as e:
                print(f"[ERROR] Failed to save scan data: {e}")
                if callback:
                    callback(False, str(e), None)
            finally:
                BackgroundFileOps._untrack_thread(thread)

        thread = threading.Thread(target=_save, daemon=False)
        BackgroundFileOps._track_thread(thread)
        thread.start()
