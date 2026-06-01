"""
Safety Configuration Data Structures

Typed representations for safety monitoring configurations.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
from datetime import datetime
from enum import Enum


# A device's reading buffer must hold at least this many samples for a safety
# check to ever fire — keep this >= SafetyMonitor.MIN_SAMPLES_FOR_CHECK.
MIN_BUFFER_SIZE = 5


class ViolationType(Enum):
    """Type of safety violation."""
    WARNING = "WARNING"
    ABORT = "ABORT"


class SafetyThresholdType(Enum):
    """Type of threshold monitoring."""
    PER_DEVICE_MEAN = "per_device_mean"  # Monitor each device's mean shift individually
    OVERALL_MEAN = "overall_mean"        # Monitor overall mean shift across all devices


@dataclass
class SafetyViolation:
    """Represents a safety limit violation."""
    device: str
    value: float
    baseline_mean: float
    current_mean: float
    baseline_std: float
    mean_shift: float  # Absolute difference from baseline
    sigma_deviation: float  # How many sigmas away from baseline
    threshold: float  # Threshold in units of sigma
    violation_type: ViolationType
    timestamp: datetime

    def __str__(self):
        return (f"{self.violation_type.value}: {self.device} Mean={self.current_mean:.6f} "
                f"(shifted {self.mean_shift:.6f} from baseline {self.baseline_mean:.6f}, "
                f"{self.sigma_deviation:.2f}σ deviation, threshold: {self.threshold:.1f}σ)")


@dataclass
class DeviceBaseline:
    """Baseline measurements for a device."""
    device: str
    mean: float
    rms: float
    std: float
    min_val: float
    max_val: float
    sample_count: int
    timestamp: datetime

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'device': self.device,
            'mean': self.mean,
            'rms': self.rms,
            'std': self.std,
            'min': self.min_val,
            'max': self.max_val,
            'sample_count': self.sample_count,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class SafetyConfiguration:
    """Complete safety monitoring configuration."""
    enabled: bool = True

    # Threshold type selection
    threshold_type: SafetyThresholdType = SafetyThresholdType.PER_DEVICE_MEAN

    # Per-device mean shift thresholds (in units of sigma/standard deviation)
    per_device_warning_threshold: float = 3.0  # 3-sigma warning
    per_device_abort_threshold: float = 5.0    # 5-sigma abort

    # Overall mean shift thresholds (in units of sigma)
    overall_warning_threshold: float = 2.5     # 2.5-sigma overall warning
    overall_abort_threshold: float = 4.0       # 4-sigma overall abort

    # Monitoring buffer configuration
    buffer_size: int = 100  # Number of samples to average for current mean (1 = no averaging)

    # Baselines for reading devices
    device_baselines: Dict[str, DeviceBaseline] = field(default_factory=dict)

    # Emergency stop behavior
    auto_abort_on_violation: bool = True
    restore_nominals_on_abort: bool = True

    # Logging
    log_warnings: bool = True
    log_to_file: bool = True

    def has_baseline(self, device: str) -> bool:
        """Check if baseline exists for device."""
        return device in self.device_baselines

    def get_baseline(self, device: str) -> Optional[DeviceBaseline]:
        """Get baseline for device."""
        return self.device_baselines.get(device)

    def set_baseline(self, device: str, baseline: DeviceBaseline):
        """Set baseline for device."""
        self.device_baselines[device] = baseline

    def clear_baselines(self):
        """Clear all baselines."""
        self.device_baselines.clear()

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'enabled': self.enabled,
            'threshold_type': self.threshold_type.value,
            'per_device_warning_threshold': self.per_device_warning_threshold,
            'per_device_abort_threshold': self.per_device_abort_threshold,
            'overall_warning_threshold': self.overall_warning_threshold,
            'overall_abort_threshold': self.overall_abort_threshold,
            'buffer_size': self.buffer_size,
            'auto_abort_on_violation': self.auto_abort_on_violation,
            'restore_nominals_on_abort': self.restore_nominals_on_abort,
            'log_warnings': self.log_warnings,
            'log_to_file': self.log_to_file,
            'baselines': {dev: baseline.to_dict() for dev, baseline in self.device_baselines.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SafetyConfiguration':
        """Create configuration from dictionary."""
        config = cls()
        config.enabled = data.get('enabled', True)
        config.threshold_type = SafetyThresholdType(data.get('threshold_type', 'per_device_mean'))
        config.per_device_warning_threshold = data.get('per_device_warning_threshold', 3.0)
        config.per_device_abort_threshold = data.get('per_device_abort_threshold', 5.0)
        config.overall_warning_threshold = data.get('overall_warning_threshold', 2.5)
        config.overall_abort_threshold = data.get('overall_abort_threshold', 4.0)
        config.buffer_size = data.get('buffer_size', 100)
        config.auto_abort_on_violation = data.get('auto_abort_on_violation', True)
        config.restore_nominals_on_abort = data.get('restore_nominals_on_abort', True)
        config.log_warnings = data.get('log_warnings', True)
        config.log_to_file = data.get('log_to_file', True)

        # Load baselines
        baselines = data.get('baselines', {})
        for dev, bl_data in baselines.items():
            try:
                config.device_baselines[dev] = DeviceBaseline(
                    device=dev,
                    mean=bl_data['mean'],
                    rms=bl_data['rms'],
                    std=bl_data['std'],
                    min_val=bl_data['min'],
                    max_val=bl_data['max'],
                    sample_count=bl_data['sample_count'],
                    timestamp=datetime.fromisoformat(bl_data['timestamp'])
                )
            except (KeyError, ValueError) as e:
                print(f"[WARNING] Failed to load baseline for {dev}: {e}")

        # Reject an invalid config outright. A config with broken thresholds
        # (e.g. warning >= abort) that loads with only a warning silently
        # disables real protection — fail loudly instead.
        config.validate()
        return config

    def validate(self):
        """Validate threshold configuration.

        Raises:
            ValueError: If thresholds are non-positive, buffer_size too small,
                        or warning threshold >= abort threshold
        """
        for name, val in [
            ('per_device_warning_threshold', self.per_device_warning_threshold),
            ('per_device_abort_threshold', self.per_device_abort_threshold),
            ('overall_warning_threshold', self.overall_warning_threshold),
            ('overall_abort_threshold', self.overall_abort_threshold),
        ]:
            if val <= 0:
                raise ValueError(f"{name} must be > 0, got {val}")

        if self.buffer_size < MIN_BUFFER_SIZE:
            raise ValueError(
                f"buffer_size must be >= {MIN_BUFFER_SIZE} — safety checks need "
                f"that many samples to fire — got {self.buffer_size}")

        if self.per_device_warning_threshold >= self.per_device_abort_threshold:
            raise ValueError("Per-device warning threshold must be less than abort threshold")
        if self.overall_warning_threshold >= self.overall_abort_threshold:
            raise ValueError("Overall warning threshold must be less than abort threshold")


@dataclass
class NominalSettings:
    """Storage for nominal device settings to restore after abort."""
    device_values: Dict[str, float] = field(default_factory=dict)
    timestamp: Optional[datetime] = None

    def store(self, device: str, value: float):
        """Store nominal value for a device."""
        self.device_values[device] = value
        self.timestamp = datetime.now()

    def get(self, device: str) -> Optional[float]:
        """Get nominal value for a device."""
        return self.device_values.get(device)

    def get_all(self) -> Dict[str, float]:
        """Get all nominal values."""
        return self.device_values.copy()

    def clear(self):
        """Clear all stored nominals."""
        self.device_values.clear()
        self.timestamp = None

    def has_values(self) -> bool:
        """Check if any nominals are stored."""
        return len(self.device_values) > 0
