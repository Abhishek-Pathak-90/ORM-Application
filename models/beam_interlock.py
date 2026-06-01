"""
Beam Interlock Data Structures

Data models for beam loss interlock monitoring and beam control.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime
from enum import Enum


class BeamStatus(Enum):
    """Beam status."""
    ON = "ON"
    OFF = "OFF"
    UNKNOWN = "UNKNOWN"


@dataclass
class BeamTripEvent:
    """Record of a loss-monitor-triggered beam trip."""
    device: str
    value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self):
        return (f"BEAM TRIP: {self.device} = {self.value:.4f} "
                f"exceeded threshold {self.threshold:.4f}")

    def to_dict(self) -> Dict:
        return {
            'device': self.device,
            'value': self.value,
            'threshold': self.threshold,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class BeamInterlockConfig:
    """Configuration for the beam loss interlock system."""
    enabled: bool = False

    # Loss monitor thresholds: device name -> max allowed value
    loss_monitors: Dict[str, float] = field(default_factory=dict)

    # ACNET event for reading beam status and loss monitors
    beam_event: str = "@e,0A"

    # Role that has L:BSTUDY access (empty = use corrector_role)
    beam_role: str = ""

    # Automatically disable beam when scan completes successfully
    disable_beam_on_completion: bool = False

    def to_dict(self) -> Dict:
        return {
            'enabled': self.enabled,
            'loss_monitors': dict(self.loss_monitors),
            'beam_event': self.beam_event,
            'beam_role': self.beam_role,
            'disable_beam_on_completion': self.disable_beam_on_completion,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'BeamInterlockConfig':
        return cls(
            enabled=data.get('enabled', False),
            loss_monitors=data.get('loss_monitors', {}),
            beam_event=data.get('beam_event', '@e,0A'),
            beam_role=data.get('beam_role', ''),
            disable_beam_on_completion=data.get('disable_beam_on_completion', False),
        )
