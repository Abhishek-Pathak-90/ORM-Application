"""
Beam Interlock Monitor

Monitors radiation loss devices and disables beam when thresholds are exceeded.
Uses L:BSTUDY for beam control per Fermilab Linac operations.
"""

from datetime import datetime
from typing import Dict, List, Optional

from models.beam_interlock import BeamInterlockConfig, BeamTripEvent
from config.settings import BEAM_STATUS_DRF, BEAM_CONTROL_DRF


class BeamInterlockMonitor:
    """
    Monitors loss devices and can disable beam on threshold violations.

    Default loss monitors:
    - L:DELM2 (threshold 25)
    - L:DELM5 (threshold 10)
    - L:400SCA (threshold 0.2)
    - L:D7LMSM (threshold 40)

    On violation: disables beam via L:BSTUDY.CONTROL, logs the event.
    """

    def __init__(self, config: BeamInterlockConfig, scanner):
        self.config = config
        self.scanner = scanner
        self._tripped = False
        self._trip_event: Optional[BeamTripEvent] = None
        self._trip_history: List[BeamTripEvent] = []

    def update_config(self, config: BeamInterlockConfig):
        self.config = config

    @property
    def is_tripped(self) -> bool:
        return self._tripped

    @property
    def last_trip(self) -> Optional[BeamTripEvent]:
        return self._trip_event

    def reset(self):
        self._tripped = False
        self._trip_event = None

    @property
    def loss_monitor_devices(self) -> List[str]:
        if not self.config.enabled:
            return []
        return [dev for dev, thresh in self.config.loss_monitors.items()
                if thresh > 0]

    def check_losses(self, readings: Dict[str, float]) -> Optional[BeamTripEvent]:
        """Check loss monitor readings against thresholds.

        Args:
            readings: Dict of device name -> value

        Returns:
            BeamTripEvent if any threshold exceeded, None if all OK
        """
        if not self.config.enabled:
            return None

        for device, threshold in self.config.loss_monitors.items():
            value = readings.get(device)
            if value is not None and value > threshold:
                event = BeamTripEvent(
                    device=device,
                    value=float(value),
                    threshold=threshold,
                    timestamp=datetime.now()
                )
                print(f"[ERROR] LOSS MONITOR EXCEEDED: {device} = {value:.4f} "
                      f"> threshold {threshold:.4f}")
                return event

        return None

    def trip_beam(self, trip_event: BeamTripEvent, role: str) -> bool:
        """Disable beam due to loss monitor violation.

        Returns True ONLY if the beam is confirmed OFF: disable_beam reports a
        positively-confirmed disable AND a check_beam_on() readback shows the
        beam is OFF. On any unconfirmed/failed outcome returns False so the scan
        loop can escalate (e.g. to a modal).
        """
        self._tripped = True
        self._trip_event = trip_event
        self._trip_history.append(trip_event)

        print(f"[ERROR] BEAM TRIP INITIATED: {trip_event}")

        try:
            disabled = self.scanner.disable_beam(BEAM_CONTROL_DRF, role)
        except Exception as e:
            print(f"[ERROR] CRITICAL: Failed to disable beam: {e}")
            return False

        # Independent readback confirmation: beam must be positively OFF.
        try:
            beam_on = self.check_beam_on()
        except Exception as e:
            print(f"[ERROR] CRITICAL: Failed to read back beam status after "
                  f"disable: {e}")
            beam_on = None

        if disabled and beam_on is False:
            print("[INFO] Beam trip confirmed: beam is OFF")
            return True

        print(f"[ERROR] CRITICAL: Beam trip NOT confirmed OFF "
              f"(disable_confirmed={disabled}, beam_on={beam_on}) — escalation "
              f"required")
        return False

    def check_beam_on(self) -> Optional[bool]:
        """Check if beam is currently on.

        Returns:
            True if on, False if off, None if unknown
        """
        return self.scanner.check_beam_status(
            BEAM_STATUS_DRF, self.config.beam_event)

    def enable_beam(self, role: str) -> bool:
        try:
            ok = bool(self.scanner.enable_beam(BEAM_CONTROL_DRF, role))
            if ok:
                print("[INFO] Beam enable command confirmed")
            else:
                print("[WARNING] Beam enable command NOT confirmed")
            return ok
        except Exception as e:
            print(f"[ERROR] Failed to enable beam: {e}")
            return False

    def disable_beam(self, role: str) -> bool:
        try:
            ok = bool(self.scanner.disable_beam(BEAM_CONTROL_DRF, role))
            if ok:
                print("[INFO] Beam disable confirmed OFF")
            else:
                print("[WARNING] Beam disable NOT confirmed")
            return ok
        except Exception as e:
            print(f"[ERROR] Failed to disable beam: {e}")
            return False

    def read_loss_monitors(self) -> Dict[str, float]:
        """Read current loss monitor values."""
        devices = self.loss_monitor_devices
        if not devices:
            return {}
        try:
            drf_list = [f"{dev}{self.config.beam_event}" for dev in devices]
            data = self.scanner.read_once_on_event(drf_list)
            result = {}
            for i, dev in enumerate(devices):
                if data and i < len(data) and data[i] is not None:
                    result[dev] = float(data[i]['data']) if isinstance(data[i], dict) else float(data[i])
            return result
        except Exception as e:
            print(f"[WARNING] Failed to read loss monitors: {e}")
            return {}

    def get_effective_role(self, fallback_role: str) -> str:
        return self.config.beam_role or fallback_role
