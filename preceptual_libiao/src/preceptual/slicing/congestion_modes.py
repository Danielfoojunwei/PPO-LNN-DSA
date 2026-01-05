"""
Congestion Modes for Airtime OS

Implements deterministic degradation policies for different
congestion/failure scenarios.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
import time
import logging

from .slices import SliceManager, SliceID, SliceState

logger = logging.getLogger(__name__)


class CongestionMode(Enum):
    """
    System-wide congestion modes.

    Each mode defines specific behaviors for:
    - Slice budget adjustments
    - Switching policy
    - Admission control
    """
    NORMAL = 0
    PROTECT_CONTROL = 1  # Freeze switching, throttle best-effort
    ROAM_RECOVERY = 2    # Allow only top-K risk moves
    INCIDENT_CONTAINMENT = 3  # Cap misbehaving robots


@dataclass
class ModeConfig:
    """Configuration for a congestion mode."""
    mode: CongestionMode
    name: str
    description: str

    # Slice budget adjustments (multipliers)
    slice_budget_multipliers: Dict[SliceID, float] = field(default_factory=dict)

    # Switching policy
    switching_enabled: bool = True
    max_switches_per_tick: int = 10
    switch_cooldown_seconds: float = 5.0

    # Admission control
    admission_enabled: bool = True
    admission_threshold: float = 0.9  # Queue depth threshold

    # Auto-recovery
    recovery_threshold_seconds: float = 30.0  # Time before considering recovery


# Default mode configurations
MODE_CONFIGS = {
    CongestionMode.NORMAL: ModeConfig(
        mode=CongestionMode.NORMAL,
        name="Normal",
        description="Normal operation with all features enabled",
        slice_budget_multipliers={
            SliceID.A: 1.0, SliceID.B: 1.0, SliceID.C: 1.0,
            SliceID.D: 1.0, SliceID.E: 1.0, SliceID.F: 1.0,
        },
        switching_enabled=True,
        max_switches_per_tick=10,
        switch_cooldown_seconds=5.0,
        admission_enabled=True,
        admission_threshold=0.9,
        recovery_threshold_seconds=30.0,
    ),
    CongestionMode.PROTECT_CONTROL: ModeConfig(
        mode=CongestionMode.PROTECT_CONTROL,
        name="Protect Control",
        description="Freeze switching, throttle best-effort to protect control traffic",
        slice_budget_multipliers={
            SliceID.A: 1.5, SliceID.B: 1.3, SliceID.C: 1.0,
            SliceID.D: 0.8, SliceID.E: 0.5, SliceID.F: 0.3,
        },
        switching_enabled=False,
        max_switches_per_tick=0,
        switch_cooldown_seconds=30.0,
        admission_enabled=True,
        admission_threshold=0.5,  # More aggressive throttling
        recovery_threshold_seconds=60.0,
    ),
    CongestionMode.ROAM_RECOVERY: ModeConfig(
        mode=CongestionMode.ROAM_RECOVERY,
        name="Roam Recovery",
        description="Allow only top-K high-risk robot moves",
        slice_budget_multipliers={
            SliceID.A: 1.2, SliceID.B: 1.1, SliceID.C: 1.0,
            SliceID.D: 0.9, SliceID.E: 0.7, SliceID.F: 0.5,
        },
        switching_enabled=True,
        max_switches_per_tick=3,  # Only top-K
        switch_cooldown_seconds=10.0,
        admission_enabled=True,
        admission_threshold=0.7,
        recovery_threshold_seconds=45.0,
    ),
    CongestionMode.INCIDENT_CONTAINMENT: ModeConfig(
        mode=CongestionMode.INCIDENT_CONTAINMENT,
        name="Incident Containment",
        description="Cap misbehaving robots, isolate problem areas",
        slice_budget_multipliers={
            SliceID.A: 1.5, SliceID.B: 1.2, SliceID.C: 0.8,
            SliceID.D: 0.5, SliceID.E: 0.3, SliceID.F: 0.1,
        },
        switching_enabled=True,
        max_switches_per_tick=5,
        switch_cooldown_seconds=15.0,
        admission_enabled=True,
        admission_threshold=0.3,  # Very aggressive throttling
        recovery_threshold_seconds=120.0,
    ),
}


@dataclass
class ModeTransition:
    """Record of a mode transition."""
    from_mode: CongestionMode
    to_mode: CongestionMode
    timestamp: float
    reason: str
    triggered_by: str  # "policy", "safety", "manual"


@dataclass
class CongestionTrigger:
    """Condition that triggers mode transition."""
    name: str
    target_mode: CongestionMode
    check_fn: Callable[[SliceManager, Dict[str, Any]], bool]
    priority: int  # Higher = checked first
    cooldown_seconds: float = 30.0
    last_triggered: float = 0.0


class CongestionModeManager:
    """
    Manages congestion mode transitions and enforcement.

    Provides:
    - Mode state machine
    - Trigger evaluation
    - Policy enforcement
    - Recovery tracking
    """

    def __init__(
        self,
        slice_manager: SliceManager,
        configs: Optional[Dict[CongestionMode, ModeConfig]] = None,
    ):
        self.slice_manager = slice_manager
        self.configs = configs or MODE_CONFIGS.copy()

        # Current state
        self.current_mode: CongestionMode = CongestionMode.NORMAL
        self.mode_entry_time: float = time.time()
        self.mode_stable_time: float = 0.0

        # History
        self.transitions: List[ModeTransition] = []
        self.max_history = 100

        # Triggers
        self.triggers: List[CongestionTrigger] = []
        self._setup_default_triggers()

        # Capped robots (for incident containment)
        self.capped_robots: Dict[str, float] = {}  # robot_id -> cap_time

    @property
    def config(self) -> ModeConfig:
        """Get current mode config."""
        return self.configs[self.current_mode]

    @property
    def time_in_mode(self) -> float:
        """Time spent in current mode."""
        return time.time() - self.mode_entry_time

    def _setup_default_triggers(self) -> None:
        """Set up default congestion triggers."""
        # Control plane degradation -> PROTECT_CONTROL
        self.triggers.append(CongestionTrigger(
            name="control_plane_degradation",
            target_mode=CongestionMode.PROTECT_CONTROL,
            check_fn=self._check_control_degradation,
            priority=100,
            cooldown_seconds=60.0,
        ))

        # High switch failure rate -> ROAM_RECOVERY
        self.triggers.append(CongestionTrigger(
            name="high_switch_failures",
            target_mode=CongestionMode.ROAM_RECOVERY,
            check_fn=self._check_switch_failures,
            priority=80,
            cooldown_seconds=45.0,
        ))

        # Slice A/B drops -> PROTECT_CONTROL
        self.triggers.append(CongestionTrigger(
            name="critical_slice_drops",
            target_mode=CongestionMode.PROTECT_CONTROL,
            check_fn=self._check_critical_drops,
            priority=90,
            cooldown_seconds=30.0,
        ))

        # Hotspot collapse -> INCIDENT_CONTAINMENT
        self.triggers.append(CongestionTrigger(
            name="hotspot_collapse",
            target_mode=CongestionMode.INCIDENT_CONTAINMENT,
            check_fn=self._check_hotspot_collapse,
            priority=70,
            cooldown_seconds=120.0,
        ))

        # Recovery to normal
        self.triggers.append(CongestionTrigger(
            name="recovery_to_normal",
            target_mode=CongestionMode.NORMAL,
            check_fn=self._check_recovery,
            priority=10,
            cooldown_seconds=60.0,
        ))

    def _check_control_degradation(
        self,
        slice_manager: SliceManager,
        context: Dict[str, Any]
    ) -> bool:
        """Check for control plane degradation."""
        # Check device timeout rate
        avg_timeout = context.get("avg_device_timeout_rate", 0.0)
        if avg_timeout > 0.2:
            return True

        # Check slice A latency
        slice_a = slice_manager.get_slice(SliceID.A)
        if slice_a.avg_latency_ms > slice_a.config.max_latency_ms * 0.8:
            return True

        return False

    def _check_switch_failures(
        self,
        slice_manager: SliceManager,
        context: Dict[str, Any]
    ) -> bool:
        """Check for high switch failure rate."""
        failure_rate = context.get("switch_failure_rate", 0.0)
        return failure_rate > 0.3

    def _check_critical_drops(
        self,
        slice_manager: SliceManager,
        context: Dict[str, Any]
    ) -> bool:
        """Check for drops in critical slices."""
        slice_a = slice_manager.get_slice(SliceID.A)
        slice_b = slice_manager.get_slice(SliceID.B)

        return (
            slice_a.drop_rate > slice_a.config.max_drop_rate or
            slice_b.drop_rate > slice_b.config.max_drop_rate * 2
        )

    def _check_hotspot_collapse(
        self,
        slice_manager: SliceManager,
        context: Dict[str, Any]
    ) -> bool:
        """Check for hotspot collapse conditions."""
        hotspot_count = context.get("hotspot_device_count", 0)
        return hotspot_count >= 2

    def _check_recovery(
        self,
        slice_manager: SliceManager,
        context: Dict[str, Any]
    ) -> bool:
        """Check if conditions allow recovery to normal."""
        if self.current_mode == CongestionMode.NORMAL:
            return False

        # Check stability time requirement
        if self.time_in_mode < self.config.recovery_threshold_seconds:
            return False

        # Check all slices healthy
        unhealthy = slice_manager.get_unhealthy_slices()
        if unhealthy:
            return False

        # Check device health
        avg_timeout = context.get("avg_device_timeout_rate", 0.0)
        if avg_timeout > 0.05:
            return False

        return True

    def evaluate_triggers(self, context: Dict[str, Any]) -> Optional[CongestionMode]:
        """
        Evaluate all triggers and return recommended mode.

        Args:
            context: Context dict with metrics

        Returns:
            Recommended mode, or None if no change needed
        """
        now = time.time()

        # Sort triggers by priority (highest first)
        sorted_triggers = sorted(self.triggers, key=lambda t: t.priority, reverse=True)

        for trigger in sorted_triggers:
            # Skip if on cooldown
            if now - trigger.last_triggered < trigger.cooldown_seconds:
                continue

            # Skip if already in target mode
            if trigger.target_mode == self.current_mode:
                continue

            # Check trigger condition
            try:
                if trigger.check_fn(self.slice_manager, context):
                    trigger.last_triggered = now
                    return trigger.target_mode
            except Exception as e:
                logger.error(f"Trigger check failed for {trigger.name}: {e}")

        return None

    def transition_to(
        self,
        mode: CongestionMode,
        reason: str,
        triggered_by: str = "policy"
    ) -> bool:
        """
        Transition to a new congestion mode.

        Args:
            mode: Target mode
            reason: Reason for transition
            triggered_by: What triggered the transition

        Returns:
            True if transition occurred
        """
        if mode == self.current_mode:
            return False

        # Record transition
        transition = ModeTransition(
            from_mode=self.current_mode,
            to_mode=mode,
            timestamp=time.time(),
            reason=reason,
            triggered_by=triggered_by,
        )
        self.transitions.append(transition)

        # Trim history
        if len(self.transitions) > self.max_history:
            self.transitions = self.transitions[-self.max_history:]

        # Update state
        old_mode = self.current_mode
        self.current_mode = mode
        self.mode_entry_time = time.time()

        logger.info(
            f"Congestion mode transition: {old_mode.name} -> {mode.name} "
            f"(reason: {reason}, triggered_by: {triggered_by})"
        )

        # Apply mode-specific budget adjustments
        self._apply_mode_budgets()

        return True

    def _apply_mode_budgets(self) -> None:
        """Apply budget adjustments for current mode."""
        config = self.config
        multipliers = config.slice_budget_multipliers

        for slice_id, multiplier in multipliers.items():
            state = self.slice_manager.get_slice(slice_id)
            new_budget = state.config.default_budget * multiplier
            state.set_budget(new_budget)

    def is_switching_allowed(self, robot_id: Optional[str] = None) -> bool:
        """Check if switching is allowed under current mode."""
        if not self.config.switching_enabled:
            return False

        # Check if robot is capped
        if robot_id and robot_id in self.capped_robots:
            cap_time = self.capped_robots[robot_id]
            if time.time() - cap_time < 300:  # 5 minute cap
                return False

        return True

    def get_max_switches(self) -> int:
        """Get maximum allowed switches per tick."""
        return self.config.max_switches_per_tick

    def get_switch_cooldown(self) -> float:
        """Get switch cooldown in seconds."""
        return self.config.switch_cooldown_seconds

    def cap_robot(self, robot_id: str, reason: str = "") -> None:
        """Cap a misbehaving robot."""
        self.capped_robots[robot_id] = time.time()
        logger.warning(f"Capped robot {robot_id}: {reason}")

    def uncap_robot(self, robot_id: str) -> None:
        """Remove cap from a robot."""
        self.capped_robots.pop(robot_id, None)

    def get_capped_robots(self) -> List[str]:
        """Get list of currently capped robots."""
        now = time.time()
        return [
            rid for rid, cap_time in self.capped_robots.items()
            if now - cap_time < 300
        ]

    def force_mode(self, mode: CongestionMode, reason: str = "") -> None:
        """Force transition to a mode (safety override)."""
        self.transition_to(mode, reason or f"Forced to {mode.name}", "safety")

    def tick(self, context: Dict[str, Any]) -> Optional[CongestionMode]:
        """
        Process a tick - evaluate triggers and return new mode if changed.

        Args:
            context: Current system context

        Returns:
            New mode if transition occurred, else None
        """
        # Check for trigger-based transitions
        recommended = self.evaluate_triggers(context)
        if recommended:
            self.transition_to(
                recommended,
                f"Trigger evaluation recommended {recommended.name}",
                "policy"
            )
            return recommended

        # Update stable time
        self.mode_stable_time = self.time_in_mode

        return None

    def get_summary(self) -> Dict[str, Any]:
        """Get mode manager summary."""
        return {
            "current_mode": self.current_mode.name,
            "time_in_mode_seconds": self.time_in_mode,
            "switching_enabled": self.config.switching_enabled,
            "max_switches_per_tick": self.config.max_switches_per_tick,
            "capped_robot_count": len(self.get_capped_robots()),
            "recent_transitions": [
                {
                    "from": t.from_mode.name,
                    "to": t.to_mode.name,
                    "reason": t.reason,
                    "timestamp": t.timestamp,
                }
                for t in self.transitions[-5:]
            ],
        }
