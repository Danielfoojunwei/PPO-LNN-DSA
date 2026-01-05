"""
Fleet World Simulator

Integrates RCS mock and LBAP world for complete fleet simulation.
Provides gymnasium-compatible environment for RL training.
"""

import random
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from .rcs_mock import RCSMock, RCSMockConfig, SimulatedRobot
from .lbap_world import LBAPWorld, LBAPWorldConfig
from ..telemetry.canonical import FleetState, RobotState, DeviceState, CongestionMode
from ..telemetry.feature_extractors import FeatureExtractor
from ..slicing.slices import SliceManager, SliceID
from ..slicing.congestion_modes import CongestionModeManager


@dataclass
class FleetWorldConfig:
    """Configuration for fleet world."""
    # Fleet size
    num_robots: int = 50
    num_aps: int = 5

    # Area
    area_width: float = 100.0
    area_height: float = 100.0

    # Timing
    tick_interval_seconds: float = 0.5  # Control tick interval
    max_episode_seconds: float = 300.0  # 5 minute episodes

    # Reward weights
    reward_stability: float = 1.0
    reward_balance: float = 0.5
    reward_switch_penalty: float = -0.1
    reward_ping_pong_penalty: float = -0.5
    reward_scan_penalty: float = -0.05
    reward_safety_violation: float = -2.0
    reward_fairness: float = 0.3

    # Thresholds
    hotspot_threshold: int = 40
    congested_channel_threshold: int = 20

    # Scenarios
    enable_failures: bool = True
    enable_movement: bool = True
    enable_interference: bool = True


@dataclass
class EpisodeMetrics:
    """Metrics collected during an episode."""
    total_reward: float = 0.0
    steps: int = 0
    switches_attempted: int = 0
    switches_successful: int = 0
    ping_pong_prevented: int = 0
    scans_performed: int = 0
    safety_violations: int = 0
    hotspots_detected: int = 0
    mode_changes: int = 0

    # QoS metrics
    avg_slice_a_latency: float = 0.0
    avg_slice_b_latency: float = 0.0
    slice_a_drops: int = 0
    slice_b_drops: int = 0

    # Load balance
    load_balance_score: float = 0.0  # Jain index

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_reward": self.total_reward,
            "steps": self.steps,
            "switch_success_rate": (
                self.switches_successful / max(1, self.switches_attempted)
            ),
            "ping_pong_prevented": self.ping_pong_prevented,
            "scans_performed": self.scans_performed,
            "safety_violations": self.safety_violations,
            "hotspots_detected": self.hotspots_detected,
            "load_balance_score": self.load_balance_score,
        }


class FleetWorld:
    """
    Complete fleet simulation environment.

    Combines RCS and LBAP simulation with:
    - Gymnasium-compatible step/reset interface
    - Reward calculation for RL training
    - Multiple scenario support
    """

    def __init__(self, config: Optional[FleetWorldConfig] = None):
        self.config = config or FleetWorldConfig()

        # Initialize components
        rcs_config = RCSMockConfig(
            num_robots=self.config.num_robots,
            num_aps=self.config.num_aps,
            area_width=self.config.area_width,
            area_height=self.config.area_height,
        )
        self.rcs = RCSMock(rcs_config)

        lbap_config = LBAPWorldConfig(
            num_devices=self.config.num_aps,
            area_width=self.config.area_width,
            area_height=self.config.area_height,
        )
        self.lbap = LBAPWorld(lbap_config)

        # Feature extraction
        self.feature_extractor = FeatureExtractor()

        # Slicing
        self.slice_manager = SliceManager()
        self.congestion_manager = CongestionModeManager(self.slice_manager)

        # State
        self.sim_time: float = 0.0
        self.episode_start: float = 0.0
        self.last_state: Optional[FleetState] = None

        # Metrics
        self.metrics = EpisodeMetrics()

        # Sync LBAP with RCS
        self._sync_lbap_with_rcs()

    def _sync_lbap_with_rcs(self) -> None:
        """Sync LBAP device assignments with RCS robot data."""
        for robot in self.rcs.robots.values():
            # Find or create mapping
            device = self.lbap.get_device_by_ip(robot.ap_ip)
            if device:
                self.lbap.assign_robot(
                    robot.robot_id,
                    device.device_id,
                    robot.channel
                )

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment for new episode.

        Returns:
            (observation, info)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Reset components
        self.rcs = RCSMock(RCSMockConfig(
            num_robots=self.config.num_robots,
            num_aps=self.config.num_aps,
            area_width=self.config.area_width,
            area_height=self.config.area_height,
        ))
        self.lbap = LBAPWorld(LBAPWorldConfig(
            num_devices=self.config.num_aps,
            area_width=self.config.area_width,
            area_height=self.config.area_height,
        ))
        self._sync_lbap_with_rcs()

        self.feature_extractor.reset()
        self.slice_manager = SliceManager()
        self.congestion_manager = CongestionModeManager(self.slice_manager)

        # Reset state
        self.sim_time = 0.0
        self.episode_start = time.time()
        self.metrics = EpisodeMetrics()

        # Get initial observation
        state = self._get_state()
        self.last_state = state

        obs = self._state_to_observation(state)
        info = {"state": state}

        return obs, info

    def step(self, action: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.

        Args:
            action: Action dict with:
                - slice_budgets: [bA, bB, bC, bD, bE, bF]
                - congestion_mode: 0-3
                - switch_actions: [(robot_id, ap_ip, channel), ...]
                - scan_robots: [robot_id, ...]

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        dt = self.config.tick_interval_seconds
        self.sim_time += dt
        self.metrics.steps += 1

        # Apply actions
        self._apply_slice_budgets(action.get("slice_budgets", []))
        self._apply_congestion_mode(action.get("congestion_mode", 0))
        self._apply_switches(action.get("switch_actions", []))
        self._apply_scans(action.get("scan_robots", []))

        # Advance simulation
        self.rcs.tick(dt)
        self.lbap.tick(dt)
        self._sync_lbap_with_rcs()

        # Simulate slice traffic
        self._simulate_slice_traffic()

        # Get new state
        state = self._get_state()

        # Calculate reward
        reward = self._calculate_reward(self.last_state, state, action)
        self.metrics.total_reward += reward

        self.last_state = state

        # Check termination
        terminated = False  # No natural termination
        truncated = self.sim_time >= self.config.max_episode_seconds

        # Build info
        info = {
            "state": state,
            "metrics": self.metrics.to_dict(),
            "rcs_metrics": self.rcs.get_metrics(),
            "lbap_metrics": self.lbap.get_network_metrics(),
        }

        obs = self._state_to_observation(state)

        return obs, reward, terminated, truncated, info

    def _get_state(self) -> FleetState:
        """Build current fleet state."""
        robot_states = []
        for robot_id, robot in self.rcs.robots.items():
            rs = self.feature_extractor.extract_robot_state(
                robot_id=robot_id,
                ap_ip=robot.ap_ip,
                channel=robot.channel,
                rssi=robot.rssi,
                coordinate_str=f"{robot.x},{robot.y},{robot.zone_id}",
                timestamp=self.sim_time,
            )
            robot_states.append(rs)

        device_states = []
        for device in self.lbap.devices.values():
            ds = DeviceState(
                device_id=device.device_id,
                ip_address=device.ip_address,
                is_online=device.is_online,
                module1_active=device.module1.is_active,
                module2_active=device.module2.is_active,
                robot_count=device.robot_count,
                queue_depth=device.module1.queue_depth + device.module2.queue_depth,
                udp_rtt_ms=device.get_rtt_ms(),
                loss_rate=device.get_loss_rate(),
                timeout_rate=device.get_loss_rate() * 0.5,
            )
            device_states.append(ds)

        slice_states = []
        for slice_id in SliceID:
            ss = self.slice_manager.get_slice(slice_id)
            from ..telemetry.canonical import SliceState as CanonicalSliceState
            slice_states.append(CanonicalSliceState(
                slice_id=slice_id.value,
                priority=ss.priority,
                allocated_budget=ss.current_budget,
                used_budget=ss.used_airtime,
                queue_depth=ss.queue_depth,
                drop_count=ss.drop_count,
                latency_proxy_ms=ss.avg_latency_ms,
            ))

        return self.feature_extractor.extract_fleet_state(
            robot_states=robot_states,
            device_states=device_states,
            slice_states=slice_states,
            congestion_mode=CongestionMode(self.congestion_manager.current_mode.value),
            timestamp=self.sim_time,
        )

    def _state_to_observation(self, state: FleetState) -> np.ndarray:
        """Convert fleet state to observation array."""
        global_obs = state.get_global_observation()
        robot_obs = state.get_robot_observations(max_robots=50)
        slice_obs = np.array(self.slice_manager.to_observation_vector(), dtype=np.float32)

        # Flatten and concatenate
        return np.concatenate([
            global_obs,
            slice_obs,
            robot_obs.flatten()[:500],  # Limit robot obs
        ])

    def _apply_slice_budgets(self, budgets: List[float]) -> None:
        """Apply slice budget action."""
        if len(budgets) == 6:
            budget_dict = {
                SliceID.A: budgets[0],
                SliceID.B: budgets[1],
                SliceID.C: budgets[2],
                SliceID.D: budgets[3],
                SliceID.E: budgets[4],
                SliceID.F: budgets[5],
            }
            self.slice_manager.set_budgets(budget_dict)

    def _apply_congestion_mode(self, mode: int) -> None:
        """Apply congestion mode action."""
        from ..slicing.congestion_modes import CongestionMode as CM
        mode_map = {0: CM.NORMAL, 1: CM.PROTECT_CONTROL, 2: CM.ROAM_RECOVERY, 3: CM.INCIDENT_CONTAINMENT}
        if mode in mode_map:
            new_mode = mode_map[mode]
            if new_mode != self.congestion_manager.current_mode:
                self.congestion_manager.transition_to(new_mode, "Policy action", "policy")
                self.metrics.mode_changes += 1

    def _apply_switches(self, switches: List[Tuple[str, str, int]]) -> None:
        """Apply switch actions."""
        import asyncio
        loop = asyncio.new_event_loop()

        for robot_id, ap_ip, channel in switches:
            self.metrics.switches_attempted += 1
            try:
                result = loop.run_until_complete(
                    self.rcs.set_robot(robot_id, ap_ip, channel)
                )
                if result.get("result") == 0:
                    self.metrics.switches_successful += 1
            except Exception:
                pass

        loop.close()

    def _apply_scans(self, robot_ids: List[str]) -> None:
        """Apply scan actions."""
        if robot_ids:
            self.metrics.scans_performed += len(robot_ids)
            # Get channels for robots
            channels = set()
            for rid in robot_ids:
                robot = self.rcs.robots.get(rid)
                if robot:
                    channels.add(robot.channel)
            self.lbap.start_scan(list(channels))

    def _simulate_slice_traffic(self) -> None:
        """Simulate traffic for each slice."""
        # Simplified traffic simulation
        for slice_id in SliceID:
            state = self.slice_manager.get_slice(slice_id)

            # Generate synthetic load
            base_rate = {
                SliceID.A: 5,
                SliceID.B: 20,
                SliceID.C: 30,
                SliceID.D: 15,
                SliceID.E: 10,
                SliceID.F: 20,
            }.get(slice_id, 10)

            packets = int(base_rate * self.config.tick_interval_seconds)
            for _ in range(packets):
                if self.slice_manager.admit(slice_id):
                    # Simulate transmission
                    latency = random.uniform(1, state.config.target_latency_ms * 2)
                    self.slice_manager.record_transmission(slice_id, latency, 0.01)

        # Track QoS
        slice_a = self.slice_manager.get_slice(SliceID.A)
        slice_b = self.slice_manager.get_slice(SliceID.B)
        self.metrics.avg_slice_a_latency = slice_a.avg_latency_ms
        self.metrics.avg_slice_b_latency = slice_b.avg_latency_ms
        self.metrics.slice_a_drops = slice_a.drop_count
        self.metrics.slice_b_drops = slice_b.drop_count

    def _calculate_reward(
        self,
        prev_state: Optional[FleetState],
        curr_state: FleetState,
        action: Dict[str, Any]
    ) -> float:
        """
        Calculate reward for the transition.

        Reward components:
        - Stability: Low drops and latency in critical slices
        - Balance: Even load distribution
        - Switch efficiency: Successful switches with low ping-pong
        - Scan efficiency: Minimal scan overhead
        - Safety: No violations
        - Fairness: Jain index for best-effort slices
        """
        cfg = self.config
        reward = 0.0

        # Stability reward (critical slice health)
        slice_a = self.slice_manager.get_slice(SliceID.A)
        slice_b = self.slice_manager.get_slice(SliceID.B)

        if slice_a.is_healthy and slice_b.is_healthy:
            reward += cfg.reward_stability
        else:
            # Penalty proportional to degradation
            if slice_a.drop_rate > 0:
                reward -= slice_a.drop_rate * 5
            if slice_b.drop_rate > 0:
                reward -= slice_b.drop_rate * 3

        # Load balance reward
        ap_loads = list(curr_state.robots_per_ap.values())
        if ap_loads:
            mean_load = np.mean(ap_loads)
            if mean_load > 0:
                jain = (sum(ap_loads)**2) / (len(ap_loads) * sum(l**2 for l in ap_loads))
                reward += cfg.reward_balance * jain
                self.metrics.load_balance_score = jain

        # Switch penalties
        switches = action.get("switch_actions", [])
        reward += len(switches) * cfg.reward_switch_penalty

        # Scan penalty
        scans = action.get("scan_robots", [])
        reward += len(scans) * cfg.reward_scan_penalty

        # Safety violation penalty
        hotspots = self.lbap.get_hotspot_devices(0.9)
        if hotspots:
            reward += cfg.reward_safety_violation
            self.metrics.safety_violations += 1
            self.metrics.hotspots_detected += len(hotspots)

        # Fairness bonus
        slice_e = self.slice_manager.get_slice(SliceID.E)
        slice_f = self.slice_manager.get_slice(SliceID.F)
        if slice_e.utilization > 0.1 and slice_f.utilization > 0.1:
            reward += cfg.reward_fairness

        return reward

    @staticmethod
    def observation_dim() -> int:
        """Get observation dimension."""
        return FleetState.global_observation_dim() + 30 + 500  # global + slices + robots

    @staticmethod
    def action_space_info() -> Dict[str, Any]:
        """Get action space information."""
        return {
            "slice_budgets": {"shape": (6,), "low": 0.0, "high": 1.0},
            "congestion_mode": {"n": 4},
            "max_switches": 10,
            "max_scans": 20,
        }
