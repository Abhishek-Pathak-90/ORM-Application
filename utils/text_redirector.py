"""
Text Redirector Utility

Redirects stdout/stderr to a tkinter Text widget with proper tagging.
Thread-safe: uses widget.after() to schedule GUI updates on the main thread.
"""

import sys
import tkinter as tk
from typing import Optional


class TextRedirector:
    """A class to redirect stdout/stderr to a tkinter Text widget (thread-safe)."""

    def __init__(self, widget, tag="INFO"):
        self.widget = widget
        self.tag = tag
        self._original_stdout = None
        self._original_stderr = None

    def write(self, str_in: str):
        """Write text to the widget with appropriate tag (thread-safe)."""
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
        """Perform the actual widget insert on the main thread."""
        try:
            self.widget.configure(state="normal")
            self.widget.insert("end", str_in, (tag,))
            self.widget.see("end")
            self.widget.configure(state="disabled")
        except tk.TclError:
            pass  # Widget destroyed

    def flush(self):
        """Flush method (required for file-like objects)."""
        pass

    def restore(self):
        """Restore original stdout/stderr."""
        if self._original_stdout:
            sys.stdout = self._original_stdout
        if self._original_stderr:
            sys.stderr = self._original_stderr
