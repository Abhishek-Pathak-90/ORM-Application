"""
Bounded Data Storage

Provides a collections.deque-based storage with maximum size limits
to prevent unbounded memory growth during continuous data acquisition.
"""

from collections import deque
from typing import Dict, Any, List
import threading


class BoundedDataStore:
    """Thread-safe bounded data storage for device readings."""

    def __init__(self, max_points: int = 10000):
        """
        Initialize bounded data store.

        Args:
            max_points: Maximum number of data points to keep per device
        """
        self.max_points = max_points
        self._data: Dict[str, deque] = {}
        self._lock = threading.Lock()

    def add_data(self, device: str, value: Any):
        """Add a data point for a device."""
        with self._lock:
            if device not in self._data:
                self._data[device] = deque(maxlen=self.max_points)
            self._data[device].append(value)

    def get_data(self, device: str) -> List[Any]:
        """Get all data for a device."""
        with self._lock:
            if device not in self._data:
                return []
            return list(self._data[device])

    def get_latest(self, device: str) -> Any:
        """Get the latest value for a device."""
        with self._lock:
            if device not in self._data or len(self._data[device]) == 0:
                return None
            return self._data[device][-1]

    def clear_device(self, device: str):
        """Clear data for a specific device."""
        with self._lock:
            if device in self._data:
                self._data[device].clear()

    def clear_all(self):
        """Clear all data."""
        with self._lock:
            self._data.clear()

    def get_all_devices(self) -> List[str]:
        """Get list of all devices with data."""
        with self._lock:
            return list(self._data.keys())

    def get_data_length(self, device: str) -> int:
        """Get number of data points for a device."""
        with self._lock:
            if device not in self._data:
                return 0
            return len(self._data[device])
