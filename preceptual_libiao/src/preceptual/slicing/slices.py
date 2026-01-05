"""
Traffic Slices for Airtime OS

Defines traffic slices (A-F) with priorities, budgets, and QoS parameters.
Implements per-slice airtime allocation and admission control.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import time


class SliceID(Enum):
    """Traffic slice identifiers, ordered by priority."""
    A = "A"  # Critical control (emergency, safety)
    B = "B"  # Real-time control (motion commands)
    C = "C"  # Telemetry/status (periodic reporting)
    D = "D"  # Task coordination (job assignments)
    E = "E"  # Bulk data (logs, diagnostics)
    F = "F"  # Best effort (updates, optional)


@dataclass
class SliceConfig:
    """Configuration for a traffic slice."""
    slice_id: SliceID
    priority: int  # Higher = more important (A=6, F=1)

    # Budget configuration
    min_budget: float  # Minimum guaranteed airtime fraction
    max_budget: float  # Maximum allowed airtime fraction
    default_budget: float  # Initial budget

    # QoS targets
    target_latency_ms: float  # Target max latency
    max_latency_ms: float  # Hard latency limit
    max_queue_depth: int  # Maximum queue size
    max_drop_rate: float  # Maximum acceptable drop rate

    # Behavior
    preemptible: bool  # Can be preempted by higher priority
    elastic: bool  # Budget can be dynamically adjusted

    def validate(self) -> bool:
        """Validate configuration."""
        return (
            0 <= self.min_budget <= self.max_budget <= 1.0 and
            self.min_budget <= self.default_budget <= self.max_budget and
            self.target_latency_ms <= self.max_latency_ms and
            self.priority >= 1
        )


# Default slice configurations for Libiao fleet
DEFAULT_SLICE_CONFIGS = {
    SliceID.A: SliceConfig(
        slice_id=SliceID.A,
        priority=6,
        min_budget=0.10,
        max_budget=0.25,
        default_budget=0.15,
        target_latency_ms=5.0,
        max_latency_ms=20.0,
        max_queue_depth=10,
        max_drop_rate=0.001,
        preemptible=False,
        elastic=False,
    ),
    SliceID.B: SliceConfig(
        slice_id=SliceID.B,
        priority=5,
        min_budget=0.20,
        max_budget=0.40,
        default_budget=0.25,
        target_latency_ms=10.0,
        max_latency_ms=50.0,
        max_queue_depth=20,
        max_drop_rate=0.01,
        preemptible=False,
        elastic=True,
    ),
    SliceID.C: SliceConfig(
        slice_id=SliceID.C,
        priority=4,
        min_budget=0.15,
        max_budget=0.30,
        default_budget=0.20,
        target_latency_ms=50.0,
        max_latency_ms=200.0,
        max_queue_depth=50,
        max_drop_rate=0.02,
        preemptible=True,
        elastic=True,
    ),
    SliceID.D: SliceConfig(
        slice_id=SliceID.D,
        priority=3,
        min_budget=0.10,
        max_budget=0.25,
        default_budget=0.15,
        target_latency_ms=100.0,
        max_latency_ms=500.0,
        max_queue_depth=100,
        max_drop_rate=0.05,
        preemptible=True,
        elastic=True,
    ),
    SliceID.E: SliceConfig(
        slice_id=SliceID.E,
        priority=2,
        min_budget=0.05,
        max_budget=0.20,
        default_budget=0.10,
        target_latency_ms=500.0,
        max_latency_ms=2000.0,
        max_queue_depth=200,
        max_drop_rate=0.10,
        preemptible=True,
        elastic=True,
    ),
    SliceID.F: SliceConfig(
        slice_id=SliceID.F,
        priority=1,
        min_budget=0.05,
        max_budget=0.15,
        default_budget=0.10,
        target_latency_ms=1000.0,
        max_latency_ms=5000.0,
        max_queue_depth=500,
        max_drop_rate=0.20,
        preemptible=True,
        elastic=True,
    ),
}


@dataclass
class SliceState:
    """Runtime state of a traffic slice."""
    config: SliceConfig
    current_budget: float = 0.0
    used_airtime: float = 0.0  # Airtime used this interval

    # Queue state
    queue_depth: int = 0
    enqueue_count: int = 0
    dequeue_count: int = 0
    drop_count: int = 0

    # Latency tracking
    total_latency_ms: float = 0.0
    packet_count: int = 0
    max_observed_latency_ms: float = 0.0

    # Timestamps
    last_update: float = field(default_factory=time.time)
    interval_start: float = field(default_factory=time.time)

    def __post_init__(self):
        self.current_budget = self.config.default_budget

    @property
    def slice_id(self) -> SliceID:
        return self.config.slice_id

    @property
    def priority(self) -> int:
        return self.config.priority

    @property
    def utilization(self) -> float:
        """Budget utilization ratio."""
        if self.current_budget <= 0:
            return 0.0
        return self.used_airtime / self.current_budget

    @property
    def drop_rate(self) -> float:
        """Current drop rate."""
        total = self.enqueue_count
        if total <= 0:
            return 0.0
        return self.drop_count / total

    @property
    def avg_latency_ms(self) -> float:
        """Average latency."""
        if self.packet_count <= 0:
            return 0.0
        return self.total_latency_ms / self.packet_count

    @property
    def is_healthy(self) -> bool:
        """Check if slice is meeting QoS targets."""
        return (
            self.drop_rate <= self.config.max_drop_rate and
            self.avg_latency_ms <= self.config.max_latency_ms and
            self.queue_depth <= self.config.max_queue_depth
        )

    @property
    def remaining_budget(self) -> float:
        """Remaining budget in current interval."""
        return max(0, self.current_budget - self.used_airtime)

    def can_admit(self, airtime_required: float) -> bool:
        """Check if slice can admit more traffic."""
        return (
            self.queue_depth < self.config.max_queue_depth and
            self.remaining_budget >= airtime_required
        )

    def record_packet(self, latency_ms: float, airtime_used: float) -> None:
        """Record a transmitted packet."""
        self.packet_count += 1
        self.dequeue_count += 1
        self.total_latency_ms += latency_ms
        self.used_airtime += airtime_used

        if latency_ms > self.max_observed_latency_ms:
            self.max_observed_latency_ms = latency_ms

        if self.queue_depth > 0:
            self.queue_depth -= 1

        self.last_update = time.time()

    def record_enqueue(self) -> bool:
        """Record an enqueued packet. Returns False if dropped."""
        self.enqueue_count += 1

        if self.queue_depth >= self.config.max_queue_depth:
            self.drop_count += 1
            return False

        self.queue_depth += 1
        return True

    def record_drop(self) -> None:
        """Record a dropped packet."""
        self.drop_count += 1

    def reset_interval(self) -> None:
        """Reset per-interval counters."""
        self.used_airtime = 0.0
        self.enqueue_count = 0
        self.dequeue_count = 0
        self.drop_count = 0
        self.total_latency_ms = 0.0
        self.packet_count = 0
        self.max_observed_latency_ms = 0.0
        self.interval_start = time.time()

    def set_budget(self, budget: float) -> None:
        """Set current budget (clamped to config limits)."""
        self.current_budget = max(
            self.config.min_budget,
            min(self.config.max_budget, budget)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slice_id": self.slice_id.value,
            "priority": self.priority,
            "current_budget": self.current_budget,
            "used_airtime": self.used_airtime,
            "utilization": self.utilization,
            "queue_depth": self.queue_depth,
            "drop_rate": self.drop_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "is_healthy": self.is_healthy,
        }


class SliceManager:
    """
    Manages all traffic slices for Airtime OS.

    Provides:
    - Slice initialization and configuration
    - Budget allocation and rebalancing
    - Admission control decisions
    - QoS monitoring
    """

    def __init__(
        self,
        configs: Optional[Dict[SliceID, SliceConfig]] = None,
        interval_seconds: float = 1.0,
    ):
        self.configs = configs or DEFAULT_SLICE_CONFIGS.copy()
        self.interval_seconds = interval_seconds

        # Initialize slice states
        self.slices: Dict[SliceID, SliceState] = {}
        for slice_id, config in self.configs.items():
            self.slices[slice_id] = SliceState(config=config)

        # Validate total budget
        total_budget = sum(s.current_budget for s in self.slices.values())
        if total_budget > 1.0:
            self._normalize_budgets()

    def _normalize_budgets(self) -> None:
        """Normalize budgets to sum to 1.0."""
        total = sum(s.current_budget for s in self.slices.values())
        if total > 0:
            for state in self.slices.values():
                state.current_budget /= total

    def get_slice(self, slice_id: SliceID) -> SliceState:
        """Get slice state."""
        return self.slices[slice_id]

    def set_budgets(self, budgets: Dict[SliceID, float]) -> None:
        """
        Set budgets for multiple slices.

        Args:
            budgets: Map of slice_id to new budget
        """
        for slice_id, budget in budgets.items():
            if slice_id in self.slices:
                self.slices[slice_id].set_budget(budget)

        # Normalize if needed
        total = sum(s.current_budget for s in self.slices.values())
        if total > 1.0:
            self._normalize_budgets()

    def admit(self, slice_id: SliceID, airtime_required: float = 0.01) -> bool:
        """
        Attempt to admit traffic to a slice.

        Args:
            slice_id: Target slice
            airtime_required: Airtime requirement

        Returns:
            True if admitted, False if rejected
        """
        state = self.slices[slice_id]

        if state.can_admit(airtime_required):
            return state.record_enqueue()
        else:
            state.record_drop()
            return False

    def record_transmission(
        self,
        slice_id: SliceID,
        latency_ms: float,
        airtime_used: float
    ) -> None:
        """Record a successful transmission."""
        self.slices[slice_id].record_packet(latency_ms, airtime_used)

    def tick(self) -> None:
        """Process interval tick - reset counters and rebalance."""
        for state in self.slices.values():
            state.reset_interval()

    def get_priority_order(self) -> List[SliceID]:
        """Get slices ordered by priority (highest first)."""
        return sorted(
            self.slices.keys(),
            key=lambda s: self.slices[s].priority,
            reverse=True
        )

    def get_unhealthy_slices(self) -> List[SliceID]:
        """Get slices not meeting QoS targets."""
        return [
            slice_id for slice_id, state in self.slices.items()
            if not state.is_healthy
        ]

    def get_critical_slices(self) -> List[SliceID]:
        """Get critical priority slices (A, B)."""
        return [SliceID.A, SliceID.B]

    def get_preemptible_slices(self) -> List[SliceID]:
        """Get preemptible slices ordered by priority (lowest first)."""
        return sorted(
            [s for s, state in self.slices.items() if state.config.preemptible],
            key=lambda s: self.slices[s].priority
        )

    def get_total_utilization(self) -> float:
        """Get total airtime utilization."""
        return sum(s.used_airtime for s in self.slices.values())

    def get_summary(self) -> Dict[str, Any]:
        """Get slice manager summary."""
        return {
            "total_utilization": self.get_total_utilization(),
            "unhealthy_slices": [s.value for s in self.get_unhealthy_slices()],
            "slices": {
                slice_id.value: state.to_dict()
                for slice_id, state in self.slices.items()
            }
        }

    def to_observation_vector(self) -> List[float]:
        """Convert slice states to observation vector."""
        obs = []
        for slice_id in [SliceID.A, SliceID.B, SliceID.C, SliceID.D, SliceID.E, SliceID.F]:
            state = self.slices[slice_id]
            obs.extend([
                state.current_budget,
                state.utilization,
                min(state.queue_depth / state.config.max_queue_depth, 1.0),
                state.drop_rate,
                min(state.avg_latency_ms / state.config.max_latency_ms, 1.0),
            ])
        return obs
