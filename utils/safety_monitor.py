"""
Safety Monitoring System

Real-time monitoring of device readings to prevent beam trips and radiation events.
"""

import math
import numpy as np
import threading
from datetime import datetime
from typing import Dict, List, Optional, Callable
from collections import deque
import json
from pathlib import Path

from models.safety_config import (
    SafetyConfiguration, SafetyViolation, ViolationType,
    SafetyThresholdType, DeviceBaseline
)


class SafetyMonitor:
    """
    Monitors reading device values in real-time and triggers alerts/aborts
    when thresholds are exceeded.
    """

    MIN_SAMPLES_FOR_CHECK = 5   # Minimum buffer samples before checks fire
    MIN_STD_FLOOR = 1e-10       # Floor to prevent inf sigma from zero-std baselines

    def __init__(self, config: SafetyConfiguration,
                 violation_callback: Optional[Callable] = None,
                 warning_callback: Optional[Callable] = None):
        """
        Initialize safety monitor.

        Args:
            config: SafetyConfiguration object
            violation_callback: Function to call on ABORT-level violations
            warning_callback: Function to call on WARNING-level violations
        """
        self.config = config
        self.violation_callback = violation_callback
        self.warning_callback = warning_callback

        # Real-time data buffers for mean calculation (circular buffers)
        self._data_buffers: Dict[str, deque] = {}

        # Violation tracking
        self.violation_history: List[SafetyViolation] = []
        self.warning_history: List[SafetyViolation] = []
        self._lock = threading.Lock()

        # Safety state
        self._abort_triggered = False
        self._last_check_time = None

        # Non-finite (NaN/inf) readings seen on the most recent check_batch call.
        # These are reported (for Device Health) but never trigger an abort.
        self._last_nonfinite_devices: List[str] = []

    def update_config(self, config: SafetyConfiguration):
        """Update the safety configuration."""
        with self._lock:
            old_buffer_size = self.config.buffer_size if hasattr(self.config, 'buffer_size') else 100
            self.config = config

            # If buffer size changed, recreate buffers with new size
            if config.buffer_size != old_buffer_size:
                # Preserve existing data but resize buffers
                for device in list(self._data_buffers.keys()):
                    old_data = list(self._data_buffers[device])
                    self._data_buffers[device] = deque(old_data[-config.buffer_size:], maxlen=config.buffer_size)

    def add_reading(self, device: str, value: float, timestamp: Optional[datetime] = None):
        """
        Add a single reading to the monitoring system.

        Args:
            device: Device name
            value: Reading value
            timestamp: Optional timestamp (defaults to now)
        """
        with self._lock:
            self._add_reading_unlocked(device, value)

    @staticmethod
    def _is_finite(value) -> bool:
        """
        Return True if value is a real, finite number (not NaN/inf/None).
        Non-finite readings must never be buffered or fed into a mean.
        """
        try:
            return math.isfinite(value)
        except (TypeError, ValueError):
            return False

    def _add_reading_unlocked(self, device: str, value: float) -> bool:
        """
        Add a reading without acquiring the lock (caller must hold self._lock).

        Non-finite values (NaN/inf) are NOT buffered so they cannot poison the
        running mean.

        Returns:
            True if the value was buffered, False if it was dropped as non-finite.
        """
        if not self._is_finite(value):
            return False
        if device not in self._data_buffers:
            self._data_buffers[device] = deque(maxlen=self.config.buffer_size)
        self._data_buffers[device].append(value)
        return True

    def _check_reading_unlocked(self, device: str, value: float,
                                timestamp: datetime) -> Optional[SafetyViolation]:
        """
        Check a single reading against safety limits (caller must hold self._lock).
        Returns a violation object if threshold exceeded, but does NOT record it.

        Args:
            device: Device name
            value: Current reading value
            timestamp: Timestamp for the reading

        Returns:
            SafetyViolation if threshold exceeded, None otherwise
        """
        # Add to buffer. A non-finite reading is dropped (not buffered) and
        # must never produce a violation/abort on its own.
        if not self._add_reading_unlocked(device, value):
            return None

        # Check if we have a baseline for this device
        baseline = self.config.get_baseline(device)
        if baseline is None:
            return None  # Can't check without baseline

        # Need minimum samples before checks fire
        buffer = self._data_buffers.get(device)
        if buffer is None or len(buffer) < self.MIN_SAMPLES_FOR_CHECK:
            return None

        current_mean = float(np.mean(np.array(list(buffer))))

        # Calculate mean shift (absolute difference from baseline)
        mean_shift = abs(current_mean - baseline.mean)

        # Use std floor to prevent inf sigma from zero-std baselines
        effective_std = max(baseline.std, self.MIN_STD_FLOOR)
        sigma_deviation = mean_shift / effective_std

        # Check thresholds based on configuration
        if self.config.threshold_type == SafetyThresholdType.PER_DEVICE_MEAN:
            # Check per-device thresholds
            if sigma_deviation >= self.config.per_device_abort_threshold:
                return SafetyViolation(
                    device=device,
                    value=value,
                    baseline_mean=baseline.mean,
                    current_mean=current_mean,
                    baseline_std=baseline.std,
                    mean_shift=mean_shift,
                    sigma_deviation=sigma_deviation,
                    threshold=self.config.per_device_abort_threshold,
                    violation_type=ViolationType.ABORT,
                    timestamp=timestamp
                )

            elif sigma_deviation >= self.config.per_device_warning_threshold:
                return SafetyViolation(
                    device=device,
                    value=value,
                    baseline_mean=baseline.mean,
                    current_mean=current_mean,
                    baseline_std=baseline.std,
                    mean_shift=mean_shift,
                    sigma_deviation=sigma_deviation,
                    threshold=self.config.per_device_warning_threshold,
                    violation_type=ViolationType.WARNING,
                    timestamp=timestamp
                )

        return None

    def check_reading(self, device: str, value: float,
                     timestamp: Optional[datetime] = None) -> Optional[SafetyViolation]:
        """
        Check a single reading against safety limits.
        All computation inside lock; violation/warning recorded outside lock.

        Args:
            device: Device name
            value: Current reading value
            timestamp: Optional timestamp

        Returns:
            SafetyViolation if threshold exceeded, None otherwise
        """
        if not self.config.enabled:
            return None

        if timestamp is None:
            timestamp = datetime.now()

        with self._lock:
            violation = self._check_reading_unlocked(device, value, timestamp)

        # Record outside lock to avoid deadlock (_record_* acquire self._lock)
        if violation is not None:
            if violation.violation_type == ViolationType.ABORT:
                self._record_violation(violation)
            else:
                self._record_warning(violation)

        return violation

    def check_batch(self, data_batch: List[Dict],
                    nonfinite_out: Optional[List[str]] = None) -> List[SafetyViolation]:
        """
        Check a batch of readings (from ACNET scan step).
        Per-device mode: single lock for entire batch, record outside lock.
        Overall mode: delegates to _check_overall_mean (handles own lock).

        Non-finite readings (NaN/inf) are NOT buffered and never poison the
        mean. The offending device names are collected and reported (for a
        Device Health report) but do NOT trigger an abort and never raise.
        They are exposed two ways:
          * appended to the optional ``nonfinite_out`` list, if provided;
          * available afterward via :meth:`get_last_nonfinite_devices`.

        Args:
            data_batch: List of dicts with keys: 'name', 'data', 'stamp'
            nonfinite_out: Optional list the caller passes in; offending device
                names for this batch are appended to it.

        Returns:
            List of SafetyViolation objects (empty if all safe). Non-finite
            readings are NOT represented here.
        """
        if not self.config.enabled:
            return []

        violations = []
        nonfinite_devices: List[str] = []

        # Per-device monitoring — single lock for entire batch
        if self.config.threshold_type == SafetyThresholdType.PER_DEVICE_MEAN:
            with self._lock:
                for item in data_batch:
                    device = item.get('name')
                    value = item.get('data')
                    timestamp = item.get('stamp')
                    if timestamp is None:
                        timestamp = datetime.now()

                    if device and value is not None:
                        # Flag (but do not buffer) non-finite readings.
                        if not self._is_finite(value):
                            nonfinite_devices.append(device)
                            continue

                        violation = self._check_reading_unlocked(device, value, timestamp)
                        if violation:
                            violations.append(violation)

                self._last_check_time = datetime.now()
                self._last_nonfinite_devices = list(nonfinite_devices)

            # Record all violations/warnings outside lock
            for violation in violations:
                if violation.violation_type == ViolationType.ABORT:
                    self._record_violation(violation)
                else:
                    self._record_warning(violation)

        # Overall mean monitoring (across all devices)
        elif self.config.threshold_type == SafetyThresholdType.OVERALL_MEAN:
            overall_violation = self._check_overall_mean(data_batch, nonfinite_devices)
            if overall_violation:
                violations.append(overall_violation)

            with self._lock:
                self._last_check_time = datetime.now()
                self._last_nonfinite_devices = list(nonfinite_devices)

        # Surface offending device names to the caller (Device Health report).
        if nonfinite_out is not None:
            nonfinite_out.extend(nonfinite_devices)

        return violations

    def get_last_nonfinite_devices(self) -> List[str]:
        """
        Return the device names whose reading was non-finite (NaN/inf) on the
        most recent check_batch call. Reported for Device Health; never an abort.
        """
        with self._lock:
            return list(self._last_nonfinite_devices)

    def _check_overall_mean(self, data_batch: List[Dict],
                            nonfinite_devices: Optional[List[str]] = None
                            ) -> Optional[SafetyViolation]:
        """
        Check overall mean shift across all devices.
        All computation inside lock; violation/warning recorded outside lock.

        Args:
            data_batch: List of readings
            nonfinite_devices: Optional list to collect device names whose
                reading was non-finite (NaN/inf). Such readings are not buffered.

        Returns:
            SafetyViolation if overall threshold exceeded
        """
        violation = None

        with self._lock:
            # Add all readings to buffers. Non-finite readings are dropped by
            # _add_reading_unlocked; flag them so the caller can report them.
            for item in data_batch:
                device = item.get('name')
                value = item.get('data')
                if device and value is not None:
                    if not self._add_reading_unlocked(device, value):
                        if nonfinite_devices is not None:
                            nonfinite_devices.append(device)

            # Devices ready to contribute: baselined AND enough samples buffered.
            # The minimum-samples gate must be PER-DEVICE (same as the per-device
            # path) — checking it against the total sample count across all devices
            # lets the check fire after a single reading per device.
            ready_devices = [
                device for device, buffer in self._data_buffers.items()
                if device in self.config.device_baselines
                and len(buffer) >= self.MIN_SAMPLES_FOR_CHECK
            ]

            if not ready_devices:
                return None

            # Current and baseline overall stats — computed over the SAME device
            # set so the comparison is apples-to-apples.
            all_values = []
            for device in ready_devices:
                all_values.extend(list(self._data_buffers[device]))
            current_overall_mean = float(np.mean(np.array(all_values)))

            baseline_means = [self.config.device_baselines[d].mean for d in ready_devices]
            baseline_stds = [self.config.device_baselines[d].std for d in ready_devices]
            baseline_overall_mean = float(np.mean(np.array(baseline_means)))
            baseline_overall_std = float(np.mean(np.array(baseline_stds)))

            # Calculate mean shift and sigma deviation
            mean_shift = abs(current_overall_mean - baseline_overall_mean)

            # Use std floor to prevent inf sigma from zero-std baselines
            effective_std = max(baseline_overall_std, self.MIN_STD_FLOOR)
            sigma_deviation = mean_shift / effective_std

            # Check thresholds
            timestamp = datetime.now()
            if sigma_deviation >= self.config.overall_abort_threshold:
                violation = SafetyViolation(
                    device="OVERALL",
                    value=current_overall_mean,
                    baseline_mean=baseline_overall_mean,
                    current_mean=current_overall_mean,
                    baseline_std=baseline_overall_std,
                    mean_shift=mean_shift,
                    sigma_deviation=sigma_deviation,
                    threshold=self.config.overall_abort_threshold,
                    violation_type=ViolationType.ABORT,
                    timestamp=timestamp
                )

            elif sigma_deviation >= self.config.overall_warning_threshold:
                violation = SafetyViolation(
                    device="OVERALL",
                    value=current_overall_mean,
                    baseline_mean=baseline_overall_mean,
                    current_mean=current_overall_mean,
                    baseline_std=baseline_overall_std,
                    mean_shift=mean_shift,
                    sigma_deviation=sigma_deviation,
                    threshold=self.config.overall_warning_threshold,
                    violation_type=ViolationType.WARNING,
                    timestamp=timestamp
                )

        # Record outside lock to avoid deadlock
        if violation is not None:
            if violation.violation_type == ViolationType.ABORT:
                self._record_violation(violation)
            else:
                self._record_warning(violation)

        return violation

    def _calculate_current_mean(self, device: str) -> Optional[float]:
        """Calculate mean from current data buffer."""
        with self._lock:
            if device not in self._data_buffers:
                return None

            buffer = self._data_buffers[device]
            if len(buffer) == 0:
                return None

            values = np.array(list(buffer))
            mean = np.mean(values)
            return float(mean)

    def _record_violation(self, violation: SafetyViolation):
        """Record an ABORT-level violation."""
        with self._lock:
            self.violation_history.append(violation)
            self._abort_triggered = True

        # Call callback if provided
        if self.violation_callback:
            self.violation_callback(violation)

    def _record_warning(self, violation: SafetyViolation):
        """Record a WARNING-level violation."""
        with self._lock:
            self.warning_history.append(violation)

        # Call callback if provided
        if self.warning_callback:
            self.warning_callback(violation)

    def is_abort_triggered(self) -> bool:
        """Check if abort has been triggered."""
        with self._lock:
            return self._abort_triggered

    def reset_abort_state(self):
        """Reset the abort trigger state."""
        with self._lock:
            self._abort_triggered = False

    def clear_buffers(self):
        """Clear all data buffers."""
        with self._lock:
            self._data_buffers.clear()

    def clear_history(self):
        """Clear violation and warning history."""
        with self._lock:
            self.violation_history.clear()
            self.warning_history.clear()

    def get_violation_summary(self) -> Dict:
        """Get summary of violations and warnings."""
        with self._lock:
            return {
                'total_violations': len(self.violation_history),
                'total_warnings': len(self.warning_history),
                'abort_triggered': self._abort_triggered,
                'last_check': self._last_check_time,
                'recent_violations': [str(v) for v in self.violation_history[-5:]],
                'recent_warnings': [str(v) for v in self.warning_history[-5:]]
            }

    def save_event_log(self, filepath: str):
        """
        Save violation and warning history to JSON file.

        Args:
            filepath: Path to save log file
        """
        with self._lock:
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'violations': [
                    {
                        'device': v.device,
                        'value': v.value,
                        'baseline_mean': v.baseline_mean,
                        'current_mean': v.current_mean,
                        'baseline_std': v.baseline_std,
                        'mean_shift': v.mean_shift,
                        'sigma_deviation': v.sigma_deviation,
                        'threshold': v.threshold,
                        'type': v.violation_type.value,
                        'timestamp': v.timestamp.isoformat()
                    }
                    for v in self.violation_history
                ],
                'warnings': [
                    {
                        'device': v.device,
                        'value': v.value,
                        'baseline_mean': v.baseline_mean,
                        'current_mean': v.current_mean,
                        'baseline_std': v.baseline_std,
                        'mean_shift': v.mean_shift,
                        'sigma_deviation': v.sigma_deviation,
                        'threshold': v.threshold,
                        'type': v.violation_type.value,
                        'timestamp': v.timestamp.isoformat()
                    }
                    for v in self.warning_history
                ]
            }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)

    @staticmethod
    def calculate_baseline_from_data(device: str, values: List[float]) -> DeviceBaseline:
        """
        Calculate baseline statistics from a set of readings.

        Args:
            device: Device name
            values: List of reading values

        Returns:
            DeviceBaseline object

        Raises:
            ValueError: If values is empty (or contains no finite samples)
        """
        if not values:
            raise ValueError(f"Cannot calculate baseline for {device}: no data")
        # Exclude non-finite (NaN/inf) samples so they cannot poison the baseline.
        finite_values = [v for v in values if SafetyMonitor._is_finite(v)]
        if not finite_values:
            raise ValueError(
                f"Cannot calculate baseline for {device}: no finite data")
        arr = np.array(finite_values)
        return DeviceBaseline(
            device=device,
            mean=float(np.mean(arr)),
            rms=float(np.sqrt(np.mean(arr ** 2))),
            std=float(np.std(arr)),
            min_val=float(np.min(arr)),
            max_val=float(np.max(arr)),
            sample_count=len(finite_values),
            timestamp=datetime.now()
        )
