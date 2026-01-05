"""
Feature Extractors for Fleet Telemetry

Transforms raw RCS and LBAP data into canonical observations
suitable for PPO-LNN policy.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from collections import deque
import numpy as np

from .canonical import (
    RobotState, DeviceState, ChannelState, SliceState,
    FleetState, CongestionMode, Position, LinkType
)


@dataclass
class RSSITracker:
    """Tracks RSSI history for slope calculation."""
    robot_id: str
    max_samples: int = 20
    samples: deque = field(default_factory=lambda: deque(maxlen=20))

    def add_sample(self, rssi: int, timestamp: float) -> None:
        self.samples.append((timestamp, rssi))

    def get_slope(self) -> float:
        """Calculate RSSI slope (dBm/second)."""
        if len(self.samples) < 2:
            return 0.0

        samples = list(self.samples)
        times = np.array([s[0] for s in samples])
        rssis = np.array([s[1] for s in samples])

        # Normalize time
        times = times - times[0]

        if times[-1] <= 0:
            return 0.0

        # Simple linear regression
        n = len(samples)
        sum_t = np.sum(times)
        sum_r = np.sum(rssis)
        sum_tr = np.sum(times * rssis)
        sum_t2 = np.sum(times ** 2)

        denom = n * sum_t2 - sum_t ** 2
        if abs(denom) < 1e-6:
            return 0.0

        slope = (n * sum_tr - sum_t * sum_r) / denom
        return float(slope)


@dataclass
class SwitchTracker:
    """Tracks switch history for each robot."""
    robot_id: str
    window_seconds: float = 60.0
    history: List[float] = field(default_factory=list)
    last_ap: str = ""
    last_channel: int = 0

    def check_and_record(self, ap: str, channel: int, timestamp: float) -> bool:
        """Check if switch occurred and record it."""
        switched = (
            self.last_ap and self.last_channel and
            (ap != self.last_ap or channel != self.last_channel)
        )

        if switched:
            self.history.append(timestamp)

        self.last_ap = ap
        self.last_channel = channel

        # Trim old events
        cutoff = timestamp - self.window_seconds
        self.history = [t for t in self.history if t >= cutoff]

        return switched

    @property
    def recent_count(self) -> int:
        return len(self.history)

    @property
    def time_since_last(self) -> float:
        if not self.history:
            return float('inf')
        return time.time() - max(self.history)


@dataclass
class DensityCalculator:
    """Calculates local robot density."""
    positions: Dict[str, Position] = field(default_factory=dict)
    assignments: Dict[str, Tuple[str, int]] = field(default_factory=dict)  # robot_id -> (ap, channel)
    zones: Dict[str, str] = field(default_factory=dict)  # robot_id -> zone_id

    def update(self, robot_id: str, position: Position, ap: str, channel: int) -> None:
        """Update robot tracking data."""
        self.positions[robot_id] = position
        self.assignments[robot_id] = (ap, channel)
        if position.zone_id:
            self.zones[robot_id] = position.zone_id

    def get_density_metrics(
        self,
        robot_id: str,
        radius: float = 10.0
    ) -> Tuple[int, int, int]:
        """
        Get density metrics for a robot.

        Returns:
            (robots_on_same_ap, robots_on_same_channel, robots_in_zone)
        """
        if robot_id not in self.assignments:
            return (0, 0, 0)

        my_ap, my_channel = self.assignments[robot_id]
        my_zone = self.zones.get(robot_id)

        same_ap = sum(
            1 for rid, (ap, _) in self.assignments.items()
            if rid != robot_id and ap == my_ap
        )

        same_channel = sum(
            1 for rid, (_, ch) in self.assignments.items()
            if rid != robot_id and ch == my_channel
        )

        if my_zone:
            same_zone = sum(
                1 for rid, zone in self.zones.items()
                if rid != robot_id and zone == my_zone
            )
        else:
            # Fall back to distance-based density
            my_pos = self.positions.get(robot_id)
            if my_pos:
                same_zone = sum(
                    1 for rid, pos in self.positions.items()
                    if rid != robot_id and my_pos.distance_to(pos) <= radius
                )
            else:
                same_zone = 0

        return (same_ap, same_channel, same_zone)


class FeatureExtractor:
    """
    Main feature extractor for fleet telemetry.

    Converts raw RCS robot data and LBAP device metrics
    into canonical FleetState observations.
    """

    def __init__(
        self,
        rssi_window_samples: int = 20,
        switch_window_seconds: float = 60.0,
        density_radius: float = 10.0,
        max_robots_in_obs: int = 50,
    ):
        self.rssi_window_samples = rssi_window_samples
        self.switch_window_seconds = switch_window_seconds
        self.density_radius = density_radius
        self.max_robots_in_obs = max_robots_in_obs

        # Trackers
        self._rssi_trackers: Dict[str, RSSITracker] = {}
        self._switch_trackers: Dict[str, SwitchTracker] = {}
        self._density_calc = DensityCalculator()

        # Last state for dt calculation
        self._last_timestamp: float = 0.0

        # Risk scoring weights
        self._switch_risk_weights = {
            "recent_switches": 0.3,
            "low_rssi": 0.3,
            "rssi_declining": 0.2,
            "high_density": 0.2,
        }

    def _get_rssi_tracker(self, robot_id: str) -> RSSITracker:
        if robot_id not in self._rssi_trackers:
            self._rssi_trackers[robot_id] = RSSITracker(
                robot_id=robot_id,
                max_samples=self.rssi_window_samples
            )
        return self._rssi_trackers[robot_id]

    def _get_switch_tracker(self, robot_id: str) -> SwitchTracker:
        if robot_id not in self._switch_trackers:
            self._switch_trackers[robot_id] = SwitchTracker(
                robot_id=robot_id,
                window_seconds=self.switch_window_seconds
            )
        return self._switch_trackers[robot_id]

    def _compute_switch_risk(self, robot: RobotState) -> float:
        """Compute switch risk score for a robot."""
        w = self._switch_risk_weights

        # Recent switch penalty
        switch_score = min(robot.recent_switch_count / 5.0, 1.0)

        # Low RSSI risk
        rssi_score = max(0, (75 - robot.rssi) / 75.0)  # Below -75 dBm is risky

        # Declining RSSI risk
        decline_score = max(0, -robot.rssi_slope / 5.0)  # Dropping 5 dBm/s is max risk

        # High density risk
        density_score = min(robot.robots_on_same_channel / 20.0, 1.0)

        risk = (
            w["recent_switches"] * switch_score +
            w["low_rssi"] * rssi_score +
            w["rssi_declining"] * decline_score +
            w["high_density"] * density_score
        )

        return np.clip(risk, 0.0, 1.0)

    def _compute_contention_risk(self, robot: RobotState) -> float:
        """Compute contention risk for a robot."""
        # Based on local density and channel congestion
        ap_factor = min(robot.robots_on_same_ap / 50.0, 1.0)
        channel_factor = min(robot.robots_on_same_channel / 30.0, 1.0)
        zone_factor = min(robot.robots_in_same_zone / 20.0, 1.0)

        return np.clip(
            ap_factor * 0.4 + channel_factor * 0.4 + zone_factor * 0.2,
            0.0, 1.0
        )

    def extract_robot_state(
        self,
        robot_id: str,
        ap_ip: str,
        channel: int,
        rssi: int,
        coordinate_str: str,
        timestamp: Optional[float] = None
    ) -> RobotState:
        """
        Extract canonical robot state from raw data.

        Args:
            robot_id: Robot identifier
            ap_ip: Current AP IP address
            channel: Current channel
            rssi: Current RSSI
            coordinate_str: Position string "x,y" or "x,y,zone"
            timestamp: Observation timestamp

        Returns:
            RobotState with computed features
        """
        ts = timestamp or time.time()

        # Parse position
        parts = coordinate_str.split(",")
        x = float(parts[0]) if len(parts) > 0 else 0.0
        y = float(parts[1]) if len(parts) > 1 else 0.0
        zone_id = parts[2].strip() if len(parts) > 2 else None
        position = Position(x, y, zone_id)

        # Update RSSI tracker
        rssi_tracker = self._get_rssi_tracker(robot_id)
        rssi_tracker.add_sample(rssi, ts)
        rssi_slope = rssi_tracker.get_slope()

        # Update switch tracker
        switch_tracker = self._get_switch_tracker(robot_id)
        switch_tracker.check_and_record(ap_ip, channel, ts)

        # Update density calculator
        self._density_calc.update(robot_id, position, ap_ip, channel)
        same_ap, same_channel, same_zone = self._density_calc.get_density_metrics(
            robot_id, self.density_radius
        )

        state = RobotState(
            robot_id=robot_id,
            timestamp=ts,
            position=position,
            ap_ip=ap_ip,
            channel=channel,
            link_type=LinkType.LBAP_900MHZ,
            rssi=rssi,
            rssi_slope=rssi_slope,
            time_since_last_switch=switch_tracker.time_since_last,
            recent_switch_count=switch_tracker.recent_count,
            switch_history=switch_tracker.history.copy(),
            robots_on_same_ap=same_ap,
            robots_on_same_channel=same_channel,
            robots_in_same_zone=same_zone,
            local_density_radius=self.density_radius,
        )

        # Compute risk scores
        state.switch_risk_score = self._compute_switch_risk(state)
        state.contention_risk_score = self._compute_contention_risk(state)

        return state

    def extract_device_state(
        self,
        device_id: str,
        ip_address: str,
        is_online: bool,
        module1_active: bool,
        module2_active: bool,
        robot_count: int,
        queue_depth: int,
        udp_rtt_ms: float,
        loss_rate: float,
        timeout_rate: float,
        robots_per_channel: Optional[Dict[int, int]] = None,
        timestamp: Optional[float] = None
    ) -> DeviceState:
        """Extract canonical device state."""
        ts = timestamp or time.time()

        # Estimate traffic rate from robot count and queue
        traffic_rate = (robot_count / 50.0) * 0.5 + (queue_depth / 100.0) * 0.3

        return DeviceState(
            device_id=device_id,
            ip_address=ip_address,
            timestamp=ts,
            is_online=is_online,
            module1_active=module1_active,
            module2_active=module2_active,
            robot_count=robot_count,
            queue_depth=queue_depth,
            traffic_rate_proxy=np.clip(traffic_rate, 0.0, 1.0),
            udp_rtt_ms=udp_rtt_ms,
            loss_rate=loss_rate,
            timeout_rate=timeout_rate,
            robots_per_channel=robots_per_channel or {},
        )

    def extract_fleet_state(
        self,
        robot_states: List[RobotState],
        device_states: List[DeviceState],
        slice_states: Optional[List[SliceState]] = None,
        congestion_mode: CongestionMode = CongestionMode.NORMAL,
        radio_profile_vector: Optional[List[float]] = None,
        timestamp: Optional[float] = None
    ) -> FleetState:
        """
        Build complete fleet state from components.

        Args:
            robot_states: List of robot states
            device_states: List of device states
            slice_states: Optional slice states
            congestion_mode: Current system mode
            radio_profile_vector: Site radio config vector
            timestamp: Observation timestamp

        Returns:
            Complete FleetState
        """
        ts = timestamp or time.time()

        # Compute dt
        dt = ts - self._last_timestamp if self._last_timestamp > 0 else 0.0
        self._last_timestamp = ts

        # Aggregate channel states
        channel_stats: Dict[int, Dict[str, Any]] = {}
        for robot in robot_states:
            ch = robot.channel
            if ch not in channel_stats:
                channel_stats[ch] = {
                    "robot_count": 0,
                    "device_set": set(),
                    "rssi_sum": 0,
                }
            channel_stats[ch]["robot_count"] += 1
            channel_stats[ch]["device_set"].add(robot.ap_ip)
            channel_stats[ch]["rssi_sum"] += robot.rssi

        channel_states = []
        for ch, stats in channel_stats.items():
            avg_rssi = stats["rssi_sum"] / max(1, stats["robot_count"])
            contention = min(stats["robot_count"] / 30.0, 1.0)
            channel_states.append(ChannelState(
                channel=ch,
                robot_count=stats["robot_count"],
                device_count=len(stats["device_set"]),
                avg_rssi=avg_rssi,
                contention_proxy=contention,
            ))

        # Build AP/channel distribution
        robots_per_ap: Dict[str, int] = {}
        robots_per_channel: Dict[int, int] = {}
        for robot in robot_states:
            robots_per_ap[robot.ap_ip] = robots_per_ap.get(robot.ap_ip, 0) + 1
            robots_per_channel[robot.channel] = robots_per_channel.get(robot.channel, 0) + 1

        # Sort robots by risk for policy efficiency
        sorted_robots = sorted(
            robot_states,
            key=lambda r: r.switch_risk_score + r.contention_risk_score,
            reverse=True
        )

        return FleetState(
            timestamp=ts,
            dt_since_last_tick=dt,
            n_robots=len(robot_states),
            n_devices=len(device_states),
            n_channels_in_use=len(channel_stats),
            robot_states=sorted_robots[:self.max_robots_in_obs],
            device_states=device_states,
            channel_states=channel_states,
            slice_states=slice_states or [],
            robots_per_ap=robots_per_ap,
            robots_per_channel=robots_per_channel,
            congestion_mode=congestion_mode,
            radio_profile_vector=radio_profile_vector or [],
        )

    def get_top_risk_robots(
        self,
        robot_states: List[RobotState],
        k: int = 10
    ) -> List[RobotState]:
        """Get top-K robots by combined risk score."""
        sorted_robots = sorted(
            robot_states,
            key=lambda r: r.switch_risk_score + r.contention_risk_score,
            reverse=True
        )
        return sorted_robots[:k]

    def reset(self) -> None:
        """Reset all trackers."""
        self._rssi_trackers.clear()
        self._switch_trackers.clear()
        self._density_calc = DensityCalculator()
        self._last_timestamp = 0.0
