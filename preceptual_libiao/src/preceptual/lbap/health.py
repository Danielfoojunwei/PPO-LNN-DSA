"""
LBAP Device Health Monitoring

Tracks health metrics for LBAP-102LU-900 devices.
Provides health scores, alerts, and trend analysis.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from collections import deque
from enum import Enum

from .inventory import LBAPDevice, LBAPInventory, DeviceState, ModuleState
from .udp_client import LBAPUDPClient

logger = logging.getLogger(__name__)


class HealthLevel(Enum):
    """Device health level classification."""
    CRITICAL = 0
    POOR = 1
    DEGRADED = 2
    FAIR = 3
    GOOD = 4
    EXCELLENT = 5


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthAlert:
    """Health alert for a device."""
    device_id: str
    alert_type: str
    severity: AlertSeverity
    message: str
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolved_at: Optional[float] = None

    def resolve(self) -> None:
        """Mark alert as resolved."""
        self.resolved = True
        self.resolved_at = time.time()


@dataclass
class HealthSample:
    """Single health sample for a device."""
    timestamp: float
    rtt_ms: float
    loss_rate: float
    timeout_rate: float
    module1_active: bool
    module2_active: bool
    robot_count: int
    queue_depth: int
    error_count: int


@dataclass
class HealthHistory:
    """Rolling health history for trend analysis."""
    device_id: str
    max_samples: int = 1000
    samples: deque = field(default_factory=lambda: deque(maxlen=1000))

    def add_sample(self, sample: HealthSample) -> None:
        """Add a health sample."""
        self.samples.append(sample)

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    def get_trend(self, metric: str, window_size: int = 10) -> Optional[float]:
        """
        Get trend direction for a metric.

        Returns:
            Positive = improving, Negative = degrading, None = insufficient data
        """
        if len(self.samples) < window_size * 2:
            return None

        samples = list(self.samples)
        recent = samples[-window_size:]
        previous = samples[-window_size*2:-window_size]

        recent_avg = sum(getattr(s, metric, 0) for s in recent) / len(recent)
        previous_avg = sum(getattr(s, metric, 0) for s in previous) / len(previous)

        # For RTT and loss, lower is better (invert)
        if metric in ('rtt_ms', 'loss_rate', 'timeout_rate', 'error_count'):
            return previous_avg - recent_avg
        else:
            return recent_avg - previous_avg

    def get_stats(self, metric: str, window_size: int = 100) -> Dict[str, float]:
        """Get statistics for a metric over recent window."""
        if not self.samples:
            return {"min": 0, "max": 0, "avg": 0, "latest": 0}

        samples = list(self.samples)[-window_size:]
        values = [getattr(s, metric, 0) for s in samples]

        return {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1] if values else 0,
        }


@dataclass
class HealthThresholds:
    """Thresholds for health classification."""
    # RTT thresholds (ms)
    rtt_excellent: float = 10.0
    rtt_good: float = 25.0
    rtt_fair: float = 50.0
    rtt_degraded: float = 100.0
    rtt_poor: float = 200.0

    # Loss rate thresholds
    loss_excellent: float = 0.0
    loss_good: float = 0.01
    loss_fair: float = 0.03
    loss_degraded: float = 0.05
    loss_poor: float = 0.10

    # Timeout rate thresholds
    timeout_excellent: float = 0.0
    timeout_good: float = 0.01
    timeout_fair: float = 0.05
    timeout_degraded: float = 0.10
    timeout_poor: float = 0.20

    # Queue depth thresholds (as fraction of max)
    queue_excellent: float = 0.1
    queue_good: float = 0.3
    queue_fair: float = 0.5
    queue_degraded: float = 0.7
    queue_poor: float = 0.9


@dataclass
class DeviceHealthStatus:
    """Current health status for a device."""
    device_id: str
    overall_level: HealthLevel
    health_score: float  # 0-100
    rtt_level: HealthLevel
    loss_level: HealthLevel
    timeout_level: HealthLevel
    module_level: HealthLevel
    queue_level: HealthLevel

    # Trend indicators
    rtt_trend: Optional[float] = None
    loss_trend: Optional[float] = None

    # Active alerts
    active_alerts: List[HealthAlert] = field(default_factory=list)

    # Timestamps
    last_update: float = field(default_factory=time.time)
    last_success: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "overall_level": self.overall_level.name,
            "health_score": self.health_score,
            "rtt_level": self.rtt_level.name,
            "loss_level": self.loss_level.name,
            "timeout_level": self.timeout_level.name,
            "module_level": self.module_level.name,
            "queue_level": self.queue_level.name,
            "rtt_trend": self.rtt_trend,
            "loss_trend": self.loss_trend,
            "active_alert_count": len(self.active_alerts),
            "last_update": self.last_update,
        }


class HealthMonitor:
    """
    LBAP device health monitoring service.

    Provides:
    - Real-time health scoring
    - Trend analysis
    - Alert generation
    - Failover recommendations
    """

    def __init__(
        self,
        inventory: LBAPInventory,
        thresholds: Optional[HealthThresholds] = None,
        max_queue_depth: int = 100,
    ):
        self.inventory = inventory
        self.thresholds = thresholds or HealthThresholds()
        self.max_queue_depth = max_queue_depth

        self._history: Dict[str, HealthHistory] = {}
        self._status: Dict[str, DeviceHealthStatus] = {}
        self._alerts: List[HealthAlert] = []
        self._alert_callbacks: List[callable] = []

        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None

    @property
    def device_statuses(self) -> Dict[str, DeviceHealthStatus]:
        """Get all device health statuses."""
        return dict(self._status)

    @property
    def active_alerts(self) -> List[HealthAlert]:
        """Get all active (unresolved) alerts."""
        return [a for a in self._alerts if not a.resolved]

    def on_alert(self, callback: callable) -> None:
        """Register callback for new alerts."""
        self._alert_callbacks.append(callback)

    def _classify_level(
        self,
        value: float,
        excellent: float,
        good: float,
        fair: float,
        degraded: float,
        poor: float,
        higher_is_worse: bool = True
    ) -> HealthLevel:
        """Classify a metric value into health level."""
        if higher_is_worse:
            if value <= excellent:
                return HealthLevel.EXCELLENT
            elif value <= good:
                return HealthLevel.GOOD
            elif value <= fair:
                return HealthLevel.FAIR
            elif value <= degraded:
                return HealthLevel.DEGRADED
            elif value <= poor:
                return HealthLevel.POOR
            else:
                return HealthLevel.CRITICAL
        else:
            # Invert for metrics where higher is better
            if value >= excellent:
                return HealthLevel.EXCELLENT
            elif value >= good:
                return HealthLevel.GOOD
            elif value >= fair:
                return HealthLevel.FAIR
            elif value >= degraded:
                return HealthLevel.DEGRADED
            elif value >= poor:
                return HealthLevel.POOR
            else:
                return HealthLevel.CRITICAL

    def _compute_health_score(self, levels: List[HealthLevel]) -> float:
        """Compute overall health score (0-100) from component levels."""
        if not levels:
            return 0.0

        # Weight each level (0-5 scale)
        total = sum(level.value for level in levels)
        max_total = len(levels) * HealthLevel.EXCELLENT.value
        return (total / max_total) * 100

    def evaluate_device(self, device: LBAPDevice) -> DeviceHealthStatus:
        """
        Evaluate health of a single device.

        Args:
            device: LBAPDevice to evaluate

        Returns:
            DeviceHealthStatus
        """
        t = self.thresholds

        # RTT level
        rtt_level = self._classify_level(
            device.avg_rtt_ms,
            t.rtt_excellent, t.rtt_good, t.rtt_fair,
            t.rtt_degraded, t.rtt_poor
        )

        # Loss level
        loss_level = self._classify_level(
            device.loss_rate,
            t.loss_excellent, t.loss_good, t.loss_fair,
            t.loss_degraded, t.loss_poor
        )

        # Timeout level
        timeout_level = self._classify_level(
            device.timeout_rate,
            t.timeout_excellent, t.timeout_good, t.timeout_fair,
            t.timeout_degraded, t.timeout_poor
        )

        # Module level
        active_modules = device.active_module_count
        if active_modules == 2:
            module_level = HealthLevel.EXCELLENT
        elif active_modules == 1:
            module_level = HealthLevel.DEGRADED
        else:
            module_level = HealthLevel.CRITICAL

        # Queue level
        queue_fraction = device.queue_depth / max(1, self.max_queue_depth)
        queue_level = self._classify_level(
            queue_fraction,
            t.queue_excellent, t.queue_good, t.queue_fair,
            t.queue_degraded, t.queue_poor
        )

        # Compute overall level (minimum of components)
        component_levels = [rtt_level, loss_level, timeout_level, module_level, queue_level]
        overall_level = min(component_levels, key=lambda x: x.value)

        # Compute score
        health_score = self._compute_health_score(component_levels)

        # Get trends from history
        history = self._history.get(device.device_id)
        rtt_trend = history.get_trend('rtt_ms') if history else None
        loss_trend = history.get_trend('loss_rate') if history else None

        # Get active alerts for this device
        device_alerts = [
            a for a in self._alerts
            if a.device_id == device.device_id and not a.resolved
        ]

        status = DeviceHealthStatus(
            device_id=device.device_id,
            overall_level=overall_level,
            health_score=health_score,
            rtt_level=rtt_level,
            loss_level=loss_level,
            timeout_level=timeout_level,
            module_level=module_level,
            queue_level=queue_level,
            rtt_trend=rtt_trend,
            loss_trend=loss_trend,
            active_alerts=device_alerts,
            last_success=device.last_seen if device.is_online else None,
        )

        # Update stored status
        self._status[device.device_id] = status

        # Record health sample
        self._record_sample(device)

        # Check for alerts
        self._check_alerts(device, status)

        return status

    def _record_sample(self, device: LBAPDevice) -> None:
        """Record health sample for history."""
        if device.device_id not in self._history:
            self._history[device.device_id] = HealthHistory(device_id=device.device_id)

        sample = HealthSample(
            timestamp=time.time(),
            rtt_ms=device.avg_rtt_ms,
            loss_rate=device.loss_rate,
            timeout_rate=device.timeout_rate,
            module1_active=device.module1.state == ModuleState.ACTIVE,
            module2_active=device.module2.state == ModuleState.ACTIVE,
            robot_count=device.total_robot_count,
            queue_depth=device.queue_depth,
            error_count=device.error_count,
        )
        self._history[device.device_id].add_sample(sample)

    def _check_alerts(self, device: LBAPDevice, status: DeviceHealthStatus) -> None:
        """Check for alert conditions and raise alerts."""
        alerts_to_raise = []

        # Module failure
        if device.module1.state == ModuleState.FAILED:
            alerts_to_raise.append((
                "module1_failed",
                AlertSeverity.ERROR,
                f"Module 1 failed on device {device.device_id}"
            ))

        if device.module2.state == ModuleState.FAILED:
            alerts_to_raise.append((
                "module2_failed",
                AlertSeverity.ERROR,
                f"Module 2 failed on device {device.device_id}"
            ))

        # High loss rate
        if device.loss_rate > self.thresholds.loss_poor:
            alerts_to_raise.append((
                "high_loss_rate",
                AlertSeverity.WARNING,
                f"High packet loss ({device.loss_rate:.1%}) on {device.device_id}"
            ))

        # High timeout rate
        if device.timeout_rate > self.thresholds.timeout_poor:
            alerts_to_raise.append((
                "high_timeout_rate",
                AlertSeverity.WARNING,
                f"High timeout rate ({device.timeout_rate:.1%}) on {device.device_id}"
            ))

        # Device offline
        if device.state == DeviceState.OFFLINE:
            alerts_to_raise.append((
                "device_offline",
                AlertSeverity.CRITICAL,
                f"Device {device.device_id} is offline"
            ))

        # Critical health
        if status.overall_level == HealthLevel.CRITICAL:
            alerts_to_raise.append((
                "critical_health",
                AlertSeverity.CRITICAL,
                f"Device {device.device_id} health is critical (score: {status.health_score:.0f})"
            ))

        # Raise new alerts (avoid duplicates)
        for alert_type, severity, message in alerts_to_raise:
            existing = [
                a for a in self._alerts
                if a.device_id == device.device_id
                and a.alert_type == alert_type
                and not a.resolved
            ]
            if not existing:
                alert = HealthAlert(
                    device_id=device.device_id,
                    alert_type=alert_type,
                    severity=severity,
                    message=message,
                )
                self._alerts.append(alert)

                # Notify callbacks
                for callback in self._alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"Alert callback error: {e}")

    def evaluate_all(self) -> Dict[str, DeviceHealthStatus]:
        """Evaluate health of all devices in inventory."""
        results = {}
        for device in self.inventory.devices:
            status = self.evaluate_device(device)
            results[device.device_id] = status
        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get health summary across all devices."""
        if not self._status:
            return {
                "device_count": 0,
                "avg_health_score": 0,
                "level_counts": {},
                "active_alert_count": 0,
            }

        statuses = list(self._status.values())
        level_counts = {}
        for status in statuses:
            level = status.overall_level.name
            level_counts[level] = level_counts.get(level, 0) + 1

        return {
            "device_count": len(statuses),
            "avg_health_score": sum(s.health_score for s in statuses) / len(statuses),
            "level_counts": level_counts,
            "active_alert_count": len(self.active_alerts),
            "critical_devices": [
                s.device_id for s in statuses
                if s.overall_level == HealthLevel.CRITICAL
            ],
        }

    def recommend_failover(self, device_id: str) -> Optional[str]:
        """
        Recommend failover target for a degraded device.

        Args:
            device_id: Device needing failover

        Returns:
            Recommended target device_id, or None if no suitable target
        """
        source = self.inventory.get_device(device_id)
        if not source:
            return None

        # Find healthy devices in same zone
        candidates = []
        for device in self.inventory.online_devices:
            if device.device_id == device_id:
                continue

            status = self._status.get(device.device_id)
            if not status:
                continue

            if status.overall_level.value >= HealthLevel.FAIR.value:
                # Score based on health and capacity
                capacity_score = 1.0 - (device.total_robot_count / 50.0)
                health_score = status.health_score / 100.0
                same_zone = device.zone_id == source.zone_id
                zone_bonus = 0.2 if same_zone else 0.0

                score = health_score * 0.6 + capacity_score * 0.3 + zone_bonus * 0.1
                candidates.append((device.device_id, score))

        if not candidates:
            return None

        # Return best candidate
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    async def start(self) -> None:
        """Start background health monitoring."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitor started")

    async def stop(self) -> None:
        """Stop health monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitor stopped")

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                self.evaluate_all()
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")

            await asyncio.sleep(5.0)  # Evaluate every 5 seconds
