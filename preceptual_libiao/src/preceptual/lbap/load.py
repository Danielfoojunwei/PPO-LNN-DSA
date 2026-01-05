"""
LBAP Device Load Estimation

Estimates and tracks load metrics for LBAP-102LU-900 devices.
Combines robot assignment data with device health metrics.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from collections import deque
from enum import Enum

from .inventory import LBAPDevice, LBAPInventory, DeviceState
from .health import HealthMonitor, HealthLevel

logger = logging.getLogger(__name__)


class LoadLevel(Enum):
    """Device load level classification."""
    IDLE = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4
    OVERLOADED = 5


@dataclass
class LoadSample:
    """Single load sample for a device."""
    timestamp: float
    robot_count: int
    traffic_rate_proxy: float  # Estimated traffic rate (arbitrary units)
    queue_depth: int
    active_channels: int
    avg_rtt_ms: float


@dataclass
class LoadHistory:
    """Rolling load history for trend analysis."""
    device_id: str
    max_samples: int = 500
    samples: deque = field(default_factory=lambda: deque(maxlen=500))

    def add_sample(self, sample: LoadSample) -> None:
        self.samples.append(sample)

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    def get_peak_load(self, window_seconds: float = 300.0) -> Optional[LoadSample]:
        """Get peak load sample in time window."""
        if not self.samples:
            return None

        cutoff = time.time() - window_seconds
        window_samples = [s for s in self.samples if s.timestamp >= cutoff]

        if not window_samples:
            return None

        return max(window_samples, key=lambda s: s.robot_count)

    def get_avg_load(self, window_seconds: float = 60.0) -> Dict[str, float]:
        """Get average load metrics over time window."""
        if not self.samples:
            return {"robot_count": 0, "traffic_rate": 0, "queue_depth": 0}

        cutoff = time.time() - window_seconds
        window_samples = [s for s in self.samples if s.timestamp >= cutoff]

        if not window_samples:
            return {"robot_count": 0, "traffic_rate": 0, "queue_depth": 0}

        return {
            "robot_count": sum(s.robot_count for s in window_samples) / len(window_samples),
            "traffic_rate": sum(s.traffic_rate_proxy for s in window_samples) / len(window_samples),
            "queue_depth": sum(s.queue_depth for s in window_samples) / len(window_samples),
        }


@dataclass
class LoadThresholds:
    """Thresholds for load classification."""
    # Robot count thresholds
    robots_low: int = 10
    robots_moderate: int = 25
    robots_high: int = 40
    robots_critical: int = 50

    # Queue depth thresholds
    queue_low: int = 10
    queue_moderate: int = 30
    queue_high: int = 50
    queue_critical: int = 80

    # Traffic rate proxy thresholds
    traffic_low: float = 0.2
    traffic_moderate: float = 0.5
    traffic_high: float = 0.8
    traffic_critical: float = 0.95


@dataclass
class DeviceLoadStatus:
    """Current load status for a device."""
    device_id: str
    load_level: LoadLevel
    load_score: float  # 0-100 (higher = more loaded)

    # Component metrics
    robot_count: int
    robot_level: LoadLevel
    queue_depth: int
    queue_level: LoadLevel
    traffic_rate_proxy: float
    traffic_level: LoadLevel

    # Capacity
    robot_capacity: int
    remaining_capacity: int
    utilization: float

    # Trends
    load_trend: Optional[float] = None  # Positive = increasing

    # Hotspot indicator
    is_hotspot: bool = False
    hotspot_score: float = 0.0

    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "load_level": self.load_level.name,
            "load_score": self.load_score,
            "robot_count": self.robot_count,
            "queue_depth": self.queue_depth,
            "traffic_rate_proxy": self.traffic_rate_proxy,
            "utilization": self.utilization,
            "remaining_capacity": self.remaining_capacity,
            "is_hotspot": self.is_hotspot,
            "load_trend": self.load_trend,
        }


@dataclass
class ChannelLoadStatus:
    """Load status for a specific channel across devices."""
    channel: int
    robot_count: int
    device_count: int
    avg_load_score: float
    is_congested: bool
    congestion_score: float


@dataclass
class FleetLoadSummary:
    """Load summary across entire fleet."""
    timestamp: float
    total_robots: int
    total_capacity: int
    overall_utilization: float

    device_loads: Dict[str, DeviceLoadStatus]
    channel_loads: Dict[int, ChannelLoadStatus]

    # Load distribution stats
    load_std_dev: float
    load_balance_score: float  # Higher = more balanced

    # Hotspots
    hotspot_devices: List[str]
    hotspot_channels: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_robots": self.total_robots,
            "total_capacity": self.total_capacity,
            "overall_utilization": self.overall_utilization,
            "load_balance_score": self.load_balance_score,
            "hotspot_devices": self.hotspot_devices,
            "hotspot_channels": self.hotspot_channels,
        }


class LoadEstimator:
    """
    LBAP device load estimation service.

    Provides:
    - Per-device load scoring
    - Channel congestion analysis
    - Hotspot detection
    - Load balancing recommendations
    """

    def __init__(
        self,
        inventory: LBAPInventory,
        health_monitor: Optional[HealthMonitor] = None,
        thresholds: Optional[LoadThresholds] = None,
        robot_capacity_per_device: int = 50,
    ):
        self.inventory = inventory
        self.health_monitor = health_monitor
        self.thresholds = thresholds or LoadThresholds()
        self.robot_capacity = robot_capacity_per_device

        self._history: Dict[str, LoadHistory] = {}
        self._device_status: Dict[str, DeviceLoadStatus] = {}
        self._channel_status: Dict[int, ChannelLoadStatus] = {}
        self._latest_summary: Optional[FleetLoadSummary] = None

        # Robot to device mapping (from RCS data)
        self._robot_device_map: Dict[str, str] = {}

    @property
    def device_loads(self) -> Dict[str, DeviceLoadStatus]:
        """Get all device load statuses."""
        return dict(self._device_status)

    @property
    def channel_loads(self) -> Dict[int, ChannelLoadStatus]:
        """Get all channel load statuses."""
        return dict(self._channel_status)

    @property
    def latest_summary(self) -> Optional[FleetLoadSummary]:
        """Get latest fleet load summary."""
        return self._latest_summary

    def _classify_robot_load(self, count: int) -> LoadLevel:
        """Classify robot count into load level."""
        t = self.thresholds
        if count <= 0:
            return LoadLevel.IDLE
        elif count <= t.robots_low:
            return LoadLevel.LOW
        elif count <= t.robots_moderate:
            return LoadLevel.MODERATE
        elif count <= t.robots_high:
            return LoadLevel.HIGH
        elif count <= t.robots_critical:
            return LoadLevel.CRITICAL
        else:
            return LoadLevel.OVERLOADED

    def _classify_queue_load(self, depth: int) -> LoadLevel:
        """Classify queue depth into load level."""
        t = self.thresholds
        if depth <= 0:
            return LoadLevel.IDLE
        elif depth <= t.queue_low:
            return LoadLevel.LOW
        elif depth <= t.queue_moderate:
            return LoadLevel.MODERATE
        elif depth <= t.queue_high:
            return LoadLevel.HIGH
        elif depth <= t.queue_critical:
            return LoadLevel.CRITICAL
        else:
            return LoadLevel.OVERLOADED

    def _classify_traffic_load(self, rate: float) -> LoadLevel:
        """Classify traffic rate into load level."""
        t = self.thresholds
        if rate <= 0:
            return LoadLevel.IDLE
        elif rate <= t.traffic_low:
            return LoadLevel.LOW
        elif rate <= t.traffic_moderate:
            return LoadLevel.MODERATE
        elif rate <= t.traffic_high:
            return LoadLevel.HIGH
        elif rate <= t.traffic_critical:
            return LoadLevel.CRITICAL
        else:
            return LoadLevel.OVERLOADED

    def _estimate_traffic_rate(self, device: LBAPDevice) -> float:
        """
        Estimate traffic rate proxy for a device.

        Combines robot count, queue depth, and RTT to estimate load.
        """
        robot_factor = device.total_robot_count / max(1, self.robot_capacity)
        queue_factor = device.queue_depth / 100.0  # Assume max queue 100
        rtt_factor = min(1.0, device.avg_rtt_ms / 200.0)  # RTT > 200ms = maxed

        # Weighted combination
        return (robot_factor * 0.5 + queue_factor * 0.3 + rtt_factor * 0.2)

    def _compute_load_score(self, levels: List[LoadLevel]) -> float:
        """Compute load score (0-100) from component levels."""
        if not levels:
            return 0.0

        total = sum(level.value for level in levels)
        max_total = len(levels) * LoadLevel.OVERLOADED.value
        return (total / max_total) * 100

    def evaluate_device(self, device: LBAPDevice) -> DeviceLoadStatus:
        """
        Evaluate load of a single device.

        Args:
            device: LBAPDevice to evaluate

        Returns:
            DeviceLoadStatus
        """
        # Estimate traffic rate
        traffic_rate = self._estimate_traffic_rate(device)

        # Classify components
        robot_level = self._classify_robot_load(device.total_robot_count)
        queue_level = self._classify_queue_load(device.queue_depth)
        traffic_level = self._classify_traffic_load(traffic_rate)

        # Overall level (max of components)
        levels = [robot_level, queue_level, traffic_level]
        overall_level = max(levels, key=lambda x: x.value)

        # Compute score
        load_score = self._compute_load_score(levels)

        # Capacity metrics
        remaining = max(0, self.robot_capacity - device.total_robot_count)
        utilization = device.total_robot_count / max(1, self.robot_capacity)

        # Get load trend from history
        history = self._history.get(device.device_id)
        load_trend = None
        if history and history.sample_count >= 10:
            recent_avg = history.get_avg_load(30.0)
            older_avg = history.get_avg_load(120.0)
            if recent_avg and older_avg:
                load_trend = recent_avg["robot_count"] - older_avg["robot_count"]

        # Hotspot detection
        is_hotspot = (
            overall_level.value >= LoadLevel.HIGH.value or
            utilization > 0.8
        )
        hotspot_score = load_score if is_hotspot else 0.0

        status = DeviceLoadStatus(
            device_id=device.device_id,
            load_level=overall_level,
            load_score=load_score,
            robot_count=device.total_robot_count,
            robot_level=robot_level,
            queue_depth=device.queue_depth,
            queue_level=queue_level,
            traffic_rate_proxy=traffic_rate,
            traffic_level=traffic_level,
            robot_capacity=self.robot_capacity,
            remaining_capacity=remaining,
            utilization=utilization,
            load_trend=load_trend,
            is_hotspot=is_hotspot,
            hotspot_score=hotspot_score,
        )

        # Update stored status
        self._device_status[device.device_id] = status

        # Record load sample
        self._record_sample(device, traffic_rate)

        return status

    def _record_sample(self, device: LBAPDevice, traffic_rate: float) -> None:
        """Record load sample for history."""
        if device.device_id not in self._history:
            self._history[device.device_id] = LoadHistory(device_id=device.device_id)

        sample = LoadSample(
            timestamp=time.time(),
            robot_count=device.total_robot_count,
            traffic_rate_proxy=traffic_rate,
            queue_depth=device.queue_depth,
            active_channels=len(device.module1.channel if hasattr(device.module1, 'channel') else [0]),
            avg_rtt_ms=device.avg_rtt_ms,
        )
        self._history[device.device_id].add_sample(sample)

    def update_robot_assignment(self, robot_id: str, device_id: str) -> None:
        """Update robot to device mapping."""
        self._robot_device_map[robot_id] = device_id

    def get_robot_device(self, robot_id: str) -> Optional[str]:
        """Get device ID for a robot."""
        return self._robot_device_map.get(robot_id)

    def evaluate_all(self) -> FleetLoadSummary:
        """Evaluate load across entire fleet."""
        device_statuses = {}
        total_robots = 0
        total_capacity = 0

        for device in self.inventory.devices:
            status = self.evaluate_device(device)
            device_statuses[device.device_id] = status
            total_robots += status.robot_count
            total_capacity += status.robot_capacity

        # Compute fleet-level metrics
        overall_utilization = total_robots / max(1, total_capacity)

        # Load balance score (inverse of std dev)
        if device_statuses:
            loads = [s.load_score for s in device_statuses.values()]
            mean_load = sum(loads) / len(loads)
            variance = sum((l - mean_load) ** 2 for l in loads) / len(loads)
            std_dev = variance ** 0.5
            # Normalize: std_dev of 50 = balance score of 0, std_dev of 0 = 100
            load_balance_score = max(0, 100 - std_dev * 2)
        else:
            std_dev = 0.0
            load_balance_score = 100.0

        # Identify hotspots
        hotspot_devices = [
            s.device_id for s in device_statuses.values()
            if s.is_hotspot
        ]

        # TODO: Channel load analysis (requires RCS integration)
        channel_loads: Dict[int, ChannelLoadStatus] = {}
        hotspot_channels: List[int] = []

        summary = FleetLoadSummary(
            timestamp=time.time(),
            total_robots=total_robots,
            total_capacity=total_capacity,
            overall_utilization=overall_utilization,
            device_loads=device_statuses,
            channel_loads=channel_loads,
            load_std_dev=std_dev,
            load_balance_score=load_balance_score,
            hotspot_devices=hotspot_devices,
            hotspot_channels=hotspot_channels,
        )

        self._latest_summary = summary
        return summary

    def recommend_target_device(
        self,
        robot_count: int = 1,
        exclude_devices: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Recommend best device for new robot assignment.

        Args:
            robot_count: Number of robots to assign
            exclude_devices: Devices to exclude from consideration

        Returns:
            Recommended device_id, or None if no suitable target
        """
        exclude = set(exclude_devices or [])
        candidates = []

        for device in self.inventory.online_devices:
            if device.device_id in exclude:
                continue

            status = self._device_status.get(device.device_id)
            if not status:
                continue

            # Skip overloaded devices
            if status.remaining_capacity < robot_count:
                continue

            # Score: prioritize low load and available capacity
            load_penalty = status.load_score / 100.0
            capacity_bonus = status.remaining_capacity / self.robot_capacity
            health_bonus = 0.0

            if self.health_monitor:
                health_status = self.health_monitor._status.get(device.device_id)
                if health_status:
                    health_bonus = health_status.health_score / 200.0  # Max 0.5 bonus

            score = capacity_bonus - load_penalty + health_bonus
            candidates.append((device.device_id, score))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def get_rebalance_suggestions(
        self,
        max_suggestions: int = 5
    ) -> List[Tuple[str, str, int]]:
        """
        Get suggestions for rebalancing load.

        Returns:
            List of (from_device, to_device, robot_count) tuples
        """
        if not self._latest_summary:
            return []

        suggestions = []
        statuses = list(self._device_status.values())

        # Find overloaded and underloaded devices
        overloaded = [
            s for s in statuses
            if s.load_level.value >= LoadLevel.HIGH.value
        ]
        underloaded = [
            s for s in statuses
            if s.load_level.value <= LoadLevel.MODERATE.value
            and s.remaining_capacity > 5
        ]

        overloaded.sort(key=lambda s: s.load_score, reverse=True)
        underloaded.sort(key=lambda s: s.remaining_capacity, reverse=True)

        for src in overloaded[:max_suggestions]:
            if not underloaded:
                break

            dst = underloaded[0]
            # Suggest moving some robots
            move_count = min(
                src.robot_count - self.thresholds.robots_moderate,
                dst.remaining_capacity // 2,
                10  # Max move at once
            )

            if move_count > 0:
                suggestions.append((
                    src.device_id,
                    dst.device_id,
                    move_count
                ))

        return suggestions

    def to_observation_vector(self) -> List[float]:
        """
        Convert fleet load state to observation vector for RL.

        Returns normalized values suitable for neural network input.
        """
        if not self._latest_summary:
            return [0.0] * 10

        s = self._latest_summary
        return [
            s.overall_utilization,
            s.load_balance_score / 100.0,
            s.load_std_dev / 50.0,  # Normalize std dev
            len(s.hotspot_devices) / max(1, len(s.device_loads)),
            s.total_robots / max(1, s.total_capacity),
            # Per-level counts
            sum(1 for d in s.device_loads.values() if d.load_level == LoadLevel.IDLE) / max(1, len(s.device_loads)),
            sum(1 for d in s.device_loads.values() if d.load_level == LoadLevel.MODERATE) / max(1, len(s.device_loads)),
            sum(1 for d in s.device_loads.values() if d.load_level == LoadLevel.HIGH) / max(1, len(s.device_loads)),
            sum(1 for d in s.device_loads.values() if d.load_level == LoadLevel.CRITICAL) / max(1, len(s.device_loads)),
            sum(1 for d in s.device_loads.values() if d.load_level == LoadLevel.OVERLOADED) / max(1, len(s.device_loads)),
        ]
