"""
Scan Configuration Data Structures

Typed representations for scan configurations with validation.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any


@dataclass
class ScanDeviceConfig:
    """Typed representation of a single device scan configuration."""
    device: str
    amplitude: float
    periods: int

    @classmethod
    def from_tree_values(cls, values: Tuple):
        """Create config from treeview values."""
        if len(values) < 3:
            raise ValueError('Scan table row must contain device, amplitude, and periods.')
        device = str(values[0]).strip()
        if not device:
            raise ValueError('Scan device name cannot be empty.')
        try:
            amplitude = float(values[1])
        except (TypeError, ValueError):
            raise ValueError(f"Amplitude for {device} must be numeric.")
        if not math.isfinite(amplitude):
            raise ValueError(f"Amplitude for {device} must be a finite number.")
        try:
            periods = int(values[2])
        except (TypeError, ValueError):
            raise ValueError(f"Number of periods for {device} must be an integer.")
        if periods < 0:
            raise ValueError(f"Number of periods for {device} must be non-negative.")
        return cls(device=device, amplitude=amplitude, periods=periods)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create config from dictionary."""
        if data is None:
            raise ValueError('Scan configuration entry cannot be empty.')
        amplitude_key = 'amp' if 'amp' in data else 'amplitude'
        if amplitude_key not in data:
            raise ValueError('Scan configuration missing amplitude information.')
        return cls.from_tree_values((
            data.get('device', ''),
            data.get(amplitude_key, 0.0),
            data.get('periods', 0)
        ))

    def to_tree_tuple(self) -> Tuple:
        """Convert to tuple for treeview display."""
        return (self.device, self.amplitude, self.periods)

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Convert to metadata dictionary."""
        return {'device': self.device, 'amp': self.amplitude, 'periods': self.periods}

    def to_setup_dict(self) -> Dict[str, Any]:
        """Convert to setup dictionary."""
        return {'device': self.device, 'amplitude': self.amplitude, 'periods': self.periods}

    def compute_value(self, nominal_value: float, step_index: int, points_per_superperiod: int) -> float:
        """Compute the scan value for a given step."""
        if self.periods == 0 or points_per_superperiod == 0:
            return nominal_value
        steps_per_period = points_per_superperiod / self.periods
        angle = 2 * np.pi * (step_index % points_per_superperiod) / steps_per_period
        return nominal_value + self.amplitude * np.sin(angle)


@dataclass
class ScanRunConfig:
    """Container for a full scan setup including validation helpers."""
    devices: List[ScanDeviceConfig]
    points_per_superperiod: int
    superperiods: int
    role: str

    def validate(self):
        """Validate the scan configuration."""
        if not self.devices:
            raise ValueError('No devices in the scan list.')
        if self.points_per_superperiod <= 0:
            raise ValueError('Points per superperiod must be greater than zero.')
        if self.superperiods <= 0:
            raise ValueError('Number of superperiods must be greater than zero.')
        role = (self.role or '').strip()
        if not role:
            raise ValueError('ACNET Role cannot be empty.')
        device_names = {cfg.device for cfg in self.devices}
        if len(device_names) != len(self.devices):
            raise ValueError('Duplicate device entries detected in scan configuration.')
        self.role = role

    @property
    def total_steps(self) -> int:
        """Get total number of steps in the scan."""
        return self.points_per_superperiod * self.superperiods

    @property
    def device_names(self) -> List[str]:
        """Get list of device names."""
        return [cfg.device for cfg in self.devices]

    def to_metadata(self) -> List[Dict[str, Any]]:
        """Convert to metadata format."""
        return [cfg.to_metadata_dict() for cfg in self.devices]
