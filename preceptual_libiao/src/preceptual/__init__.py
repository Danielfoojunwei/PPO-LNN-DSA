"""
Preceptual.ai (Libiao Edition)

Airtime OS + Handover Orchestrator for Libiao robot fleets.

This package provides:
- RCS (Robot Control System) client integration
- LBAP (900MHz gateway) device management
- Traffic slicing and congestion control
- PPO-LNN based handover orchestration
- Federated learning for multi-site deployment
- Fleet simulation for training and testing
"""

__version__ = "0.1.0"
__author__ = "Preceptual Team"

from .libiao_rcs import (
    RCSClient,
    RCSClientConfig,
    RobotInfo,
    FleetSnapshot,
    SwitchCommand,
)

from .lbap import (
    LBAPInventory,
    LBAPDevice,
    RadioProfile,
    HealthMonitor,
    LoadEstimator,
)

from .slicing import (
    SliceManager,
    SliceID,
    CongestionModeManager,
    CongestionMode,
)

from .decision import (
    CommSwitchFSM,
    SwitchOperation,
)

__all__ = [
    # Version
    "__version__",
    # RCS
    "RCSClient",
    "RCSClientConfig",
    "RobotInfo",
    "FleetSnapshot",
    "SwitchCommand",
    # LBAP
    "LBAPInventory",
    "LBAPDevice",
    "RadioProfile",
    "HealthMonitor",
    "LoadEstimator",
    # Slicing
    "SliceManager",
    "SliceID",
    "CongestionModeManager",
    "CongestionMode",
    # Decision
    "CommSwitchFSM",
    "SwitchOperation",
]
