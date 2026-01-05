"""
LBAP Network World Simulator

Simulates LBAP-102LU-900 gateway economics:
- UDP RTT/loss dynamics
- Dual-module failover
- Channel contention model
- Scanning airtime costs
"""

import random
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import math

from ..lbap.radio_profile import RadioProfile


@dataclass
class SimLBAPModule:
    """Simulated LBAP module."""
    module_id: int
    is_active: bool = True
    error_count: int = 0
    last_error_time: float = 0.0

    # Performance
    base_rtt_ms: float = 5.0
    queue_depth: int = 0
    max_queue: int = 100

    def get_rtt_ms(self, load_factor: float) -> float:
        """Get RTT based on load."""
        if not self.is_active:
            return float('inf')
        # RTT increases with queue depth
        queue_factor = 1 + (self.queue_depth / self.max_queue) * 2
        return self.base_rtt_ms * queue_factor * (1 + load_factor)


@dataclass
class SimLBAPDevice:
    """Simulated LBAP-102LU-900 device."""
    device_id: str
    ip_address: str
    x: float
    y: float

    # Dual modules
    module1: SimLBAPModule = field(default_factory=lambda: SimLBAPModule(1))
    module2: SimLBAPModule = field(default_factory=lambda: SimLBAPModule(2))

    # Capacity
    max_robots: int = 50
    robot_ids: set = field(default_factory=set)

    # Channel assignment
    primary_channel: int = 1
    channels_in_use: set = field(default_factory=set)

    # Performance simulation
    failure_rate: float = 0.001  # Per-tick module failure probability
    recovery_time: float = 30.0  # Seconds to recover failed module

    # Traffic tracking
    packets_sent: int = 0
    packets_dropped: int = 0
    last_update: float = field(default_factory=time.time)

    @property
    def robot_count(self) -> int:
        return len(self.robot_ids)

    @property
    def load_factor(self) -> float:
        return self.robot_count / self.max_robots

    @property
    def is_online(self) -> bool:
        return self.module1.is_active or self.module2.is_active

    @property
    def is_degraded(self) -> bool:
        return self.is_online and not (self.module1.is_active and self.module2.is_active)

    def get_active_module(self) -> Optional[SimLBAPModule]:
        """Get the active module (prefer module1)."""
        if self.module1.is_active:
            return self.module1
        elif self.module2.is_active:
            return self.module2
        return None

    def get_rtt_ms(self) -> float:
        """Get current RTT."""
        module = self.get_active_module()
        if module:
            return module.get_rtt_ms(self.load_factor)
        return float('inf')

    def get_loss_rate(self) -> float:
        """Get packet loss rate based on load and health."""
        if not self.is_online:
            return 1.0

        base_loss = 0.0
        if self.is_degraded:
            base_loss = 0.02  # Higher loss when degraded

        # Load-based loss
        if self.load_factor > 0.8:
            overload = (self.load_factor - 0.8) / 0.2
            base_loss += overload * 0.1

        return min(0.5, base_loss)

    def tick(self, dt: float) -> None:
        """Update device state."""
        # Simulate module failures
        if self.module1.is_active and random.random() < self.failure_rate * dt:
            self.module1.is_active = False
            self.module1.last_error_time = time.time()
            self.module1.error_count += 1

        if self.module2.is_active and random.random() < self.failure_rate * dt:
            self.module2.is_active = False
            self.module2.last_error_time = time.time()
            self.module2.error_count += 1

        # Simulate recovery
        now = time.time()
        if not self.module1.is_active:
            if now - self.module1.last_error_time > self.recovery_time:
                self.module1.is_active = True

        if not self.module2.is_active:
            if now - self.module2.last_error_time > self.recovery_time:
                self.module2.is_active = True

        # Update queue depths based on load
        target_queue = int(self.load_factor * 50)
        for module in [self.module1, self.module2]:
            if module.is_active:
                module.queue_depth = int(
                    module.queue_depth * 0.9 + target_queue * 0.1 +
                    random.gauss(0, 5)
                )
                module.queue_depth = max(0, min(module.max_queue, module.queue_depth))

        self.last_update = now

    def to_status_dict(self) -> Dict[str, Any]:
        """Convert to status dict."""
        return {
            "device_id": self.device_id,
            "ip_address": self.ip_address,
            "is_online": self.is_online,
            "is_degraded": self.is_degraded,
            "module1_active": self.module1.is_active,
            "module2_active": self.module2.is_active,
            "robot_count": self.robot_count,
            "load_factor": self.load_factor,
            "rtt_ms": self.get_rtt_ms(),
            "loss_rate": self.get_loss_rate(),
            "queue_depth": self.module1.queue_depth + self.module2.queue_depth,
        }


@dataclass
class ChannelModel:
    """
    Channel contention model.

    Simulates:
    - Collision probability based on robot count
    - Interference from nearby channels
    - Scanning overhead
    """
    channel: int
    frequency_mhz: float
    bandwidth_khz: float = 500.0

    # Robot tracking
    robot_ids: set = field(default_factory=set)

    # Contention
    base_collision_rate: float = 0.01
    collision_per_robot: float = 0.005

    # Scanning overhead
    scan_active: bool = False
    scan_overhead: float = 0.0  # Fraction of airtime used by scanning

    @property
    def robot_count(self) -> int:
        return len(self.robot_ids)

    def get_collision_rate(self) -> float:
        """Get collision rate based on robot count."""
        return min(
            0.5,
            self.base_collision_rate + self.collision_per_robot * self.robot_count
        )

    def get_effective_capacity(self) -> float:
        """Get effective capacity (1 - collision - scan overhead)."""
        return max(0, 1.0 - self.get_collision_rate() - self.scan_overhead)


@dataclass
class LBAPWorldConfig:
    """Configuration for LBAP world simulation."""
    # Devices
    num_devices: int = 5
    device_max_robots: int = 50
    device_failure_rate: float = 0.001

    # Channels
    num_channels: int = 44
    fundamental_freq_mhz: float = 904.25
    freq_interval_mhz: float = 0.5

    # Area
    area_width: float = 100.0
    area_height: float = 100.0

    # Scanning
    scan_duration_ms: float = 100.0
    scan_overhead_fraction: float = 0.05

    # Interference
    adjacent_channel_interference: float = 0.02


class LBAPWorld:
    """
    LBAP network world simulator.

    Provides realistic simulation of:
    - LBAP device health and failover
    - Channel contention and capacity
    - Scanning overhead
    - Network economics at scale
    """

    def __init__(
        self,
        config: Optional[LBAPWorldConfig] = None,
        radio_profile: Optional[RadioProfile] = None
    ):
        self.config = config or LBAPWorldConfig()
        self.radio_profile = radio_profile or RadioProfile.default_900mhz()

        # Devices
        self.devices: Dict[str, SimLBAPDevice] = {}

        # Channels
        self.channels: Dict[int, ChannelModel] = {}

        # Time tracking
        self.sim_time: float = 0.0

        # Metrics
        self.total_scans: int = 0
        self.scan_storms_detected: int = 0
        self.device_failures: int = 0

        self._initialize()

    def _initialize(self) -> None:
        """Initialize devices and channels."""
        # Create devices in grid
        num_devices = self.config.num_devices
        grid_size = int(num_devices**0.5) + 1
        spacing_x = self.config.area_width / grid_size
        spacing_y = self.config.area_height / grid_size

        for i in range(num_devices):
            row = i // grid_size
            col = i % grid_size
            device_id = f"lbap_{i:03d}"
            ip = f"192.168.0.{200 + i}"

            self.devices[device_id] = SimLBAPDevice(
                device_id=device_id,
                ip_address=ip,
                x=(col + 0.5) * spacing_x,
                y=(row + 0.5) * spacing_y,
                max_robots=self.config.device_max_robots,
                failure_rate=self.config.device_failure_rate,
            )

        # Create channels
        for i in range(self.config.num_channels):
            freq = self.config.fundamental_freq_mhz + i * self.config.freq_interval_mhz
            self.channels[i] = ChannelModel(
                channel=i,
                frequency_mhz=freq,
            )

    def get_device(self, device_id: str) -> Optional[SimLBAPDevice]:
        """Get device by ID."""
        return self.devices.get(device_id)

    def get_device_by_ip(self, ip: str) -> Optional[SimLBAPDevice]:
        """Get device by IP."""
        for device in self.devices.values():
            if device.ip_address == ip:
                return device
        return None

    def get_nearest_device(self, x: float, y: float) -> Optional[SimLBAPDevice]:
        """Get nearest device to a position."""
        nearest = None
        min_dist = float('inf')

        for device in self.devices.values():
            dist = ((device.x - x)**2 + (device.y - y)**2)**0.5
            if dist < min_dist:
                min_dist = dist
                nearest = device

        return nearest

    def assign_robot(
        self,
        robot_id: str,
        device_id: str,
        channel: int
    ) -> bool:
        """
        Assign a robot to a device and channel.

        Returns:
            True if assignment successful
        """
        device = self.devices.get(device_id)
        if not device:
            return False

        if device.robot_count >= device.max_robots:
            return False

        # Remove from any existing device
        for d in self.devices.values():
            d.robot_ids.discard(robot_id)

        # Remove from any existing channel
        for ch in self.channels.values():
            ch.robot_ids.discard(robot_id)

        # Add to new device and channel
        device.robot_ids.add(robot_id)
        device.channels_in_use.add(channel)

        if channel in self.channels:
            self.channels[channel].robot_ids.add(robot_id)

        return True

    def remove_robot(self, robot_id: str) -> None:
        """Remove a robot from all devices and channels."""
        for device in self.devices.values():
            device.robot_ids.discard(robot_id)
        for channel in self.channels.values():
            channel.robot_ids.discard(robot_id)

    def start_scan(self, channels: List[int]) -> float:
        """
        Start a scan on specified channels.

        Args:
            channels: Channels to scan

        Returns:
            Scan duration in seconds
        """
        self.total_scans += 1

        # Check for scan storm
        active_scans = sum(1 for ch in self.channels.values() if ch.scan_active)
        if active_scans > len(self.channels) * 0.3:
            self.scan_storms_detected += 1

        # Apply scan overhead to channels
        for ch_id in channels:
            if ch_id in self.channels:
                ch = self.channels[ch_id]
                ch.scan_active = True
                ch.scan_overhead = self.config.scan_overhead_fraction

        return self.config.scan_duration_ms / 1000.0

    def end_scan(self, channels: List[int]) -> None:
        """End scan on specified channels."""
        for ch_id in channels:
            if ch_id in self.channels:
                ch = self.channels[ch_id]
                ch.scan_active = False
                ch.scan_overhead = 0.0

    def tick(self, dt: float) -> None:
        """
        Advance simulation.

        Args:
            dt: Time delta in seconds
        """
        self.sim_time += dt

        # Update devices
        failures_before = sum(
            d.module1.error_count + d.module2.error_count
            for d in self.devices.values()
        )

        for device in self.devices.values():
            device.tick(dt)

        failures_after = sum(
            d.module1.error_count + d.module2.error_count
            for d in self.devices.values()
        )
        self.device_failures += failures_after - failures_before

    def get_network_metrics(self) -> Dict[str, Any]:
        """Get network-wide metrics."""
        online_devices = [d for d in self.devices.values() if d.is_online]
        degraded_devices = [d for d in self.devices.values() if d.is_degraded]

        if online_devices:
            avg_rtt = sum(d.get_rtt_ms() for d in online_devices) / len(online_devices)
            avg_loss = sum(d.get_loss_rate() for d in online_devices) / len(online_devices)
        else:
            avg_rtt = float('inf')
            avg_loss = 1.0

        total_robots = sum(d.robot_count for d in self.devices.values())

        channel_loads = {
            ch_id: ch.robot_count
            for ch_id, ch in self.channels.items()
            if ch.robot_count > 0
        }

        return {
            "sim_time": self.sim_time,
            "online_devices": len(online_devices),
            "degraded_devices": len(degraded_devices),
            "offline_devices": len(self.devices) - len(online_devices),
            "total_robots": total_robots,
            "avg_rtt_ms": avg_rtt,
            "avg_loss_rate": avg_loss,
            "channel_loads": channel_loads,
            "total_scans": self.total_scans,
            "scan_storms": self.scan_storms_detected,
            "device_failures": self.device_failures,
        }

    def get_device_statuses(self) -> List[Dict[str, Any]]:
        """Get status of all devices."""
        return [d.to_status_dict() for d in self.devices.values()]

    def get_hotspot_devices(self, threshold: float = 0.8) -> List[str]:
        """Get devices above load threshold."""
        return [
            d.device_id for d in self.devices.values()
            if d.load_factor >= threshold
        ]

    def get_congested_channels(self, threshold: int = 15) -> List[int]:
        """Get channels with high robot count."""
        return [
            ch_id for ch_id, ch in self.channels.items()
            if ch.robot_count >= threshold
        ]
