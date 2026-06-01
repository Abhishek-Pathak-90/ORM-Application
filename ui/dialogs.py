"""
UI Dialogs

Kerberos login and device selection dialogs.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, Optional, Dict, List

from auth.kerberos_manager import KerberosManager
from config.settings import PALETTE


class KerberosLoginDialog(tk.Toplevel):
    """A dialog to handle Kerberos login choices."""

    def __init__(self, parent, status_callback: Optional[Callable] = None):
        super().__init__(parent)
        self.title("Kerberos Authentication")
        self.parent = parent
        self.manager = KerberosManager()
        self.status_callback = status_callback
        self.transient(parent)
        self.grab_set()
        self.configure(bg=PALETTE['background'])

        has_ticket = self.manager.check_existing_ticket()

        main_frame = ttk.Frame(self, padding=20, style="Main.TFrame")
        main_frame.pack(expand=True, fill="both")

        if has_ticket:
            ttk.Label(
                main_frame,
                text="A valid Kerberos ticket was found.",
                font=('Helvetica', 10, 'bold')
            ).pack(pady=10)
            ttk.Button(
                main_frame,
                text="Use Existing Ticket",
                command=lambda: self._close(True, "[SUCCESS] Using existing Kerberos ticket."),
                style="Accent.TButton"
            ).pack(fill='x', pady=5)
            ttk.Button(
                main_frame,
                text="Create New Ticket",
                command=self.create_new_ticket,
                style="Accent.TButton"
            ).pack(fill='x', pady=5)
        else:
            ttk.Label(
                main_frame,
                text="No valid Kerberos ticket found.",
                font=('Helvetica', 10, 'bold')
            ).pack(pady=10)
            ttk.Button(
                main_frame,
                text="Create New Ticket",
                command=self.create_new_ticket,
                style="Accent.TButton"
            ).pack(fill='x', pady=5)

        ttk.Button(
            main_frame,
            text="Cancel",
            command=lambda: self._close(False, "[INFO] Kerberos login cancelled."),
            style="TButton"
        ).pack(fill='x', pady=(10, 0))

    def _close(self, success: bool, message: Optional[str] = None):
        """Close dialog and call callback."""
        if self.status_callback:
            try:
                self.status_callback(success, message)
            except Exception as exc:
                print(f"[WARNING] Failed to propagate Kerberos status: {exc}")
        self.destroy()

    def create_new_ticket(self):
        """Handles the process of creating a new ticket."""
        if self.manager.get_kerberos_ticket(force_new=True):
            print("[SUCCESS] Kerberos ticket obtained successfully.")
            self._close(True, "[SUCCESS] Kerberos ticket obtained successfully.")
        else:
            messagebox.showerror("Authentication Failed",
                "Failed to obtain Kerberos ticket. Check your credentials.", parent=self)


class DeviceSelectionDialog(tk.Toplevel):
    """A dialog for advanced device selection from a dictionary of device data."""

    def __init__(self, parent, device_data: Dict[str, List[str]]):
        super().__init__(parent)
        self.title("Select Devices")
        self.parent = parent
        self.device_data = device_data
        self.transient(parent)
        self.grab_set()
        self.geometry("600x400")
        self.configure(bg=PALETTE['background'])

        # Main frame
        main_frame = ttk.Frame(self, padding=12, style="Main.TFrame")
        main_frame.pack(fill="both", expand=True)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)

        # Left side: Categories (Sheets)
        ttk.Label(
            main_frame,
            text="Device Categories",
            font=('Helvetica', 10, 'bold')
        ).grid(row=0, column=0, pady=5, sticky='w')

        category_frame = ttk.Frame(main_frame, style="Main.TFrame")
        category_frame.grid(row=1, column=0, sticky='nsew', padx=(0, 5))

        self.category_listbox = tk.Listbox(
            category_frame,
            selectmode='extended',
            bg=PALETTE['surface'],
            fg=PALETTE['text'],
            highlightthickness=0,
            borderwidth=0,
            selectbackground=PALETTE['accent'],
            selectforeground=PALETTE['background']
        )
        self.category_listbox.pack(side='left', fill='both', expand=True)

        cat_scroll = ttk.Scrollbar(category_frame, orient='vertical', command=self.category_listbox.yview)
        cat_scroll.pack(side='right', fill='y')
        self.category_listbox.config(yscrollcommand=cat_scroll.set)

        for category_name in self.device_data.keys():
            self.category_listbox.insert(tk.END, category_name)
        self.category_listbox.bind('<<ListboxSelect>>', self.update_device_list)

        # Right side: Devices
        ttk.Label(
            main_frame,
            text="Devices",
            font=('Helvetica', 10, 'bold')
        ).grid(row=0, column=1, pady=5, sticky='w')

        device_frame = ttk.Frame(main_frame, style="Main.TFrame")
        device_frame.grid(row=1, column=1, sticky='nsew', padx=(5, 0))

        self.device_listbox = tk.Listbox(
            device_frame,
            selectmode='extended',
            bg=PALETTE['surface'],
            fg=PALETTE['text'],
            highlightthickness=0,
            borderwidth=0,
            selectbackground=PALETTE['accent'],
            selectforeground=PALETTE['background']
        )
        self.device_listbox.pack(side='left', fill='both', expand=True)

        dev_scroll = ttk.Scrollbar(device_frame, orient='vertical', command=self.device_listbox.yview)
        dev_scroll.pack(side='right', fill='y')
        self.device_listbox.config(yscrollcommand=dev_scroll.set)

        # Bottom buttons
        button_frame = ttk.Frame(main_frame, style="Main.TFrame")
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)

        ttk.Button(
            button_frame,
            text="Add to Setting List",
            command=lambda: self.add_to_list('setting'),
            style="Accent.TButton"
        ).pack(side='left', padx=5)

        ttk.Button(
            button_frame,
            text="Add to Reading List",
            command=lambda: self.add_to_list('reading'),
            style="Accent.TButton"
        ).pack(side='left', padx=5)

        ttk.Button(
            button_frame,
            text="Done",
            command=self.destroy,
            style="TButton"
        ).pack(side='right', padx=5)

    def update_device_list(self, event=None):
        """Update device list based on selected categories."""
        selected_indices = self.category_listbox.curselection()
        if not selected_indices:
            return

        self.device_listbox.delete(0, tk.END)
        all_devices = []
        for i in selected_indices:
            category_name = self.category_listbox.get(i)
            devices = self.device_data.get(category_name, [])
            all_devices.extend(devices)

        for device in sorted(list(set(all_devices))):
            self.device_listbox.insert(tk.END, device)

    def add_to_list(self, list_type: str):
        """Add selected devices to a list."""
        selected_indices = self.device_listbox.curselection()
        devices_to_add = [self.device_listbox.get(i) for i in selected_indices]

        if not devices_to_add:
            print("[WARNING] No devices selected from the list.")
            return

        self.parent.add_devices_to_list(devices_to_add, list_type)
        print(f"[SUCCESS] {len(devices_to_add)} devices added to the {list_type} list.")


class ConfirmationDialog(tk.Toplevel):
    """A confirmation dialog for critical operations."""

    def __init__(self, parent, title: str, message: str, on_confirm: Callable):
        super().__init__(parent)
        self.title(title)
        self.on_confirm = on_confirm
        self.result = False
        self.transient(parent)
        self.grab_set()
        self.configure(bg=PALETTE['background'])

        main_frame = ttk.Frame(self, padding=20, style="Main.TFrame")
        main_frame.pack(expand=True, fill="both")

        # Message
        ttk.Label(
            main_frame,
            text=message,
            font=('Helvetica', 10),
            wraplength=350
        ).pack(pady=15)

        # Buttons
        button_frame = ttk.Frame(main_frame, style="Main.TFrame")
        button_frame.pack(fill='x', pady=10)

        ttk.Button(
            button_frame,
            text="Confirm",
            command=self._confirm,
            style="Accent.TButton"
        ).pack(side='left', padx=5, expand=True, fill='x')

        ttk.Button(
            button_frame,
            text="Cancel",
            command=self._cancel,
            style="TButton"
        ).pack(side='right', padx=5, expand=True, fill='x')

    def _confirm(self):
        """Handle confirmation."""
        self.result = True
        if self.on_confirm:
            self.on_confirm()
        self.destroy()

    def _cancel(self):
        """Handle cancellation."""
        self.result = False
        self.destroy()
