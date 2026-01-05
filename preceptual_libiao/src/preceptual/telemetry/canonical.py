"""
Canonical Telemetry Data Structures

Unified data structures for fleet telemetry that combine
RCS robot data with LBAP device metrics.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import numpy as np


class LinkType(Enum):
    """Network link type."""
    LBAP_900MHZ = "lbap_900"  # LBAP-102LU-900 (900MHz)
    WIFI_24GHZ = "wifi_24"
    WIFI_5GHZ = "wifi_5"
    UNKNOWN = "unknown"


@dataclass
class Position:
    """2D position with zone information."""
    x: float
    y: float
    zone_id: Optional[str] = None

    def distance_to(self, other: "Position") -> float:
        """Euclidean distance."""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x, y]."""
        return np.array([self.x, self.y])


@dataclass
class RobotState:
    """
    Canonical robot state combining RCS and LBAP data.

    Per-robot observation for RL policy.
    """
    # Identity
    robot_id: str
    timestamp: float = field(default_factory=time.time)

    # Position
    position: Position = field(default_factory=lambda: Position(0, 0))

    # Current link assignment
    ap_ip: str = ""
    channel: int = 0
    link_type: LinkType = LinkType.LBAP_900MHZ

    # Signal quality
    rssi: int = 0
    rssi_slope: float = 0.0  # dBm/s trend

    # Switching history
    time_since_last_switch: float = float('inf')
    recent_switch_count: int = 0  # Switches in last 60s
    switch_history: List[float] = field(default_factory=list)  # Timestamps

    # Local density
    robots_on_same_ap: int = 0
    robots_on_same_channel: int = 0
    robots_in_same_zone: int = 0
    local_density_radius: float = 10.0  # meters

    # Traffic/QoS proxy
    latency_proxy_ms: float = 0.0
    packet_loss_proxy: float = 0.0

    # Risk indicators
    switch_risk_score: float = 0.0
    contention_risk_score: float = 0.0

    def add_switch_event(self, timestamp: Optional[float] = None) -> None:
        """Record a switch event."""
        ts = timestamp or time.time()
        self.switch_history.append(ts)
        self.time_since_last_switch = 0.0

        # Trim old events (keep last 60s)
        cutoff = ts - 60.0
        self.switch_history = [t for t in self.switch_history if t >= cutoff]
        self.recent_switch_count = len(self.switch_history)

    def update_time_since_switch(self) -> None:
        """Update time since last switch."""
        if self.switch_history:
            self.time_since_last_switch = time.time() - max(self.switch_history)

    def to_observation_vector(self) -> np.ndarray:
        """
        Convert to observation vector for RL.

        Returns:
            Normalized feature vector
        """
        return np.array([
            # Position (normalized to typical warehouse scale)
            self.position.x / 100.0,
            self.position.y / 100.0,
            # Signal
            (self.rssi + 100) / 100.0,  # Normalize from [-100, 0] to [0, 1]
            np.clip(self.rssi_slope / 10.0, -1.0, 1.0),
            # Channel (normalized)
            self.channel / 100.0,
            # Switching
            min(self.time_since_last_switch / 300.0, 1.0),  # Normalize to 5 min
            min(self.recent_switch_count / 10.0, 1.0),
            # Density
            min(self.robots_on_same_ap / 50.0, 1.0),
            min(self.robots_on_same_channel / 30.0, 1.0),
            min(self.robots_in_same_zone / 20.0, 1.0),
            # QoS proxies
            min(self.latency_proxy_ms / 200.0, 1.0),
            min(self.packet_loss_proxy, 1.0),
            # Risk scores
            np.clip(self.switch_risk_score, 0.0, 1.0),
            np.clip(self.contention_risk_score, 0.0, 1.0),
        ], dtype=np.float32)

    @staticmethod
    def observation_dim() -> int:
        """Dimension of observation vector."""
        return 14


@dataclass
class DeviceState:
    """
    Canonical LBAP device state.

    Per-device observation component.
    """
    device_id: str
    ip_address: str
    timestamp: float = field(default_factory=time.time)

    # Health
    is_online: bool = True
    module1_active: bool = True
    module2_active: bool = True

    # Load
    robot_count: int = 0
    queue_depth: int = 0
    traffic_rate_proxy: float = 0.0

    # Performance
    udp_rtt_ms: float = 0.0
    loss_rate: float = 0.0
    timeout_rate: float = 0.0

    # Per-channel breakdown
    robots_per_channel: Dict[int, int] = field(default_factory=dict)

    def to_observation_vector(self) -> np.ndarray:
        """Convert to observation vector."""
        return np.array([
            float(self.is_online),
            float(self.module1_active),
            float(self.module2_active),
            min(self.robot_count / 50.0, 1.0),
            min(self.queue_depth / 100.0, 1.0),
            np.clip(self.traffic_rate_proxy, 0.0, 1.0),
            min(self.udp_rtt_ms / 100.0, 1.0),
            np.clip(self.loss_rate, 0.0, 1.0),
            np.clip(self.timeout_rate, 0.0, 1.0),
        ], dtype=np.float32)

    @staticmethod
    def observation_dim() -> int:
        return 9


@dataclass
class ChannelState:
    """
    Channel-level aggregated state.
    """
    channel: int
    robot_count: int = 0
    device_count: int = 0
    avg_rssi: float = 0.0
    contention_proxy: float = 0.0
    timeout_rate_by_channel: float = 0.0

    def to_observation_vector(self) -> np.ndarray:
        return np.array([
            self.channel / 100.0,
            min(self.robot_count / 30.0, 1.0),
            min(self.device_count / 10.0, 1.0),
            (self.avg_rssi + 100) / 100.0,
            np.clip(self.contention_proxy, 0.0, 1.0),
            np.clip(self.timeout_rate_by_channel, 0.0, 1.0),
        ], dtype=np.float32)


@dataclass
class SliceState:
    """
    Traffic slice state for Airtime OS.

    Tracks per-slice QoS metrics.
    """
    slice_id: str  # A, B, C, D, E, F
    priority: int  # Higher = more important

    # Budget
    allocated_budget: float = 0.0  # Fraction of airtime
    used_budget: float = 0.0

    # QoS metrics
    queue_depth: int = 0
    drop_count: int = 0
    latency_proxy_ms: float = 0.0

    def to_observation_vector(self) -> np.ndarray:
        return np.array([
            self.priority / 10.0,
            self.allocated_budget,
            self.used_budget,
            min(self.queue_depth / 100.0, 1.0),
            min(self.drop_count / 100.0, 1.0),
            min(self.latency_proxy_ms / 100.0, 1.0),
        ], dtype=np.float32)


class CongestionMode(Enum):
    """System-wide congestion mode."""
    NORMAL = 0
    PROTECT_CONTROL = 1
    ROAM_RECOVERY = 2
    INCIDENT_CONTAINMENT = 3


@dataclass
class FleetState:
    """
    Complete fleet state snapshot.

    Global observation for RL policy.
    """
    timestamp: float = field(default_factory=time.time)
    dt_since_last_tick: float = 0.0

    # Fleet composition
    n_robots: int = 0
    n_devices: int = 0
    n_channels_in_use: int = 0

    # Robot states (top-K by risk for policy efficiency)
    robot_states: List[RobotState] = field(default_factory=list)

    # Device states
    device_states: List[DeviceState] = field(default_factory=list)

    # Channel states
    channel_states: List[ChannelState] = field(default_factory=list)

    # Slice states
    slice_states: List[SliceState] = field(default_factory=list)

    # AP/Channel distribution
    robots_per_ap: Dict[str, int] = field(default_factory=dict)
    robots_per_channel: Dict[int, int] = field(default_factory=dict)

    # Current congestion mode
    congestion_mode: CongestionMode = CongestionMode.NORMAL

    # Site radio profile summary
    radio_profile_vector: List[float] = field(default_factory=list)

    def get_global_observation(self) -> np.ndarray:
        """
        Get global observation vector.

        Used by PPO-LNN global head.
        """
        # Fleet metrics
        global_features = np.array([
            self.n_robots / 300.0,  # Normalize to max expected
            self.n_devices / 20.0,
            self.n_channels_in_use / 50.0,
            self.congestion_mode.value / 3.0,
            self.dt_since_last_tick / 1.0,  # Normalize to 1 second
        ], dtype=np.float32)

        # Aggregated device health
        if self.device_states:
            avg_rtt = np.mean([d.udp_rtt_ms for d in self.device_states])
            avg_loss = np.mean([d.loss_rate for d in self.device_states])
            avg_timeout = np.mean([d.timeout_rate for d in self.device_states])
            online_ratio = np.mean([float(d.is_online) for d in self.device_states])
        else:
            avg_rtt = avg_loss = avg_timeout = 0.0
            online_ratio = 1.0

        device_features = np.array([
            min(avg_rtt / 100.0, 1.0),
            np.clip(avg_loss, 0.0, 1.0),
            np.clip(avg_timeout, 0.0, 1.0),
            online_ratio,
        ], dtype=np.float32)

        # Slice QoS summary
        if self.slice_states:
            slice_drops = sum(s.drop_count for s in self.slice_states)
            slice_latency = max(s.latency_proxy_ms for s in self.slice_states)
        else:
            slice_drops = 0
            slice_latency = 0.0

        slice_features = np.array([
            min(slice_drops / 100.0, 1.0),
            min(slice_latency / 100.0, 1.0),
        ], dtype=np.float32)

        # Load distribution (Jain index proxy)
        if self.robots_per_ap:
            loads = list(self.robots_per_ap.values())
            mean_load = np.mean(loads) if loads else 0
            if mean_load > 0:
                jain_index = (sum(loads) ** 2) / (len(loads) * sum(l**2 for l in loads))
            else:
                jain_index = 1.0
        else:
            jain_index = 1.0

        balance_features = np.array([jain_index], dtype=np.float32)

        return np.concatenate([
            global_features,
            device_features,
            slice_features,
            balance_features,
        ])

    @staticmethod
    def global_observation_dim() -> int:
        """Dimension of global observation vector."""
        return 12  # 5 + 4 + 2 + 1

    def get_robot_observations(self, max_robots: int = 50) -> np.ndarray:
        """
        Get stacked robot observation matrix.

        Args:
            max_robots: Maximum number of robots (pad/truncate)

        Returns:
            Array of shape (max_robots, robot_obs_dim)
        """
        obs_dim = RobotState.observation_dim()
        result = np.zeros((max_robots, obs_dim), dtype=np.float32)

        for i, robot in enumerate(self.robot_states[:max_robots]):
            result[i] = robot.to_observation_vector()

        return result


@dataclass
class Transition:
    """
    State transition for RL training.
    """
    state: FleetState
    action: Dict[str, Any]
    reward: float
    next_state: FleetState
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)

    # Timing
    dt: float = 0.0
    timestamp: float = field(default_factory=time.time)
