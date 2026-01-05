"""
LBAP Device Inventory

Manages inventory of LBAP-102LU-900 devices in the fleet.
Tracks device status, configuration, and relationships.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set
from enum import Enum
import json
from pathlib import Path

from .radio_profile import RadioProfile, SiteRadioConfig
from .udp_client import LBAPUDPClient, UDPClientConfig

logger = logging.getLogger(__name__)


class DeviceState(Enum):
    """LBAP device operational state."""
    UNKNOWN = "unknown"
    ONLINE = "online"
    DEGRADED = "degraded"  # One module down
    OFFLINE = "offline"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ModuleState(Enum):
    """Individual module state."""
    UNKNOWN = "unknown"
    ACTIVE = "active"
    STANDBY = "standby"
    FAILED = "failed"


@dataclass
class LBAPModule:
    """
    Single wireless module in LBAP device.

    LBAP-102LU-900 has dual modules for failover.
    """
    module_id: int  # 1 or 2
    state: ModuleState = ModuleState.UNKNOWN
    channel: int = 0
    tx_power_dbm: int = 14
    robot_count: int = 0
    error_count: int = 0
    last_active: float = 0.0


@dataclass
class LBAPDevice:
    """
    LBAP-102LU-900 device representation.

    Tracks:
    - Device identity and configuration
    - Dual module status for failover handling
    - Connected robots and load metrics
    - Health history
    """
    # Identity
    device_id: str
    ip_address: str
    port: int = 5000
    mac_address: str = ""

    # Metadata
    device_type: str = "LBAP-102LU-900"
    firmware_version: str = ""
    location: str = ""
    zone_id: str = ""

    # State
    state: DeviceState = DeviceState.UNKNOWN
    last_seen: float = field(default_factory=time.time)
    uptime_seconds: int = 0

    # Dual module status
    module1: LBAPModule = field(default_factory=lambda: LBAPModule(module_id=1))
    module2: LBAPModule = field(default_factory=lambda: LBAPModule(module_id=2))

    # Load tracking
    robot_ids: Set[str] = field(default_factory=set)
    total_robot_count: int = 0
    traffic_rate_bps: float = 0.0
    queue_depth: int = 0

    # Health metrics
    avg_rtt_ms: float = 0.0
    loss_rate: float = 0.0
    timeout_rate: float = 0.0
    error_count: int = 0

    # Configuration
    radio_profile: Optional[RadioProfile] = None

    @property
    def is_online(self) -> bool:
        """Check if device is online."""
        return self.state in (DeviceState.ONLINE, DeviceState.DEGRADED)

    @property
    def is_healthy(self) -> bool:
        """Check if device is healthy (both modules active)."""
        return (
            self.state == DeviceState.ONLINE and
            self.module1.state == ModuleState.ACTIVE and
            self.module2.state == ModuleState.ACTIVE
        )

    @property
    def active_module_count(self) -> int:
        """Count of active modules."""
        count = 0
        if self.module1.state == ModuleState.ACTIVE:
            count += 1
        if self.module2.state == ModuleState.ACTIVE:
            count += 1
        return count

    @property
    def staleness_seconds(self) -> float:
        """Time since last update."""
        return time.time() - self.last_seen

    def update_from_status(self, status: Dict[str, Any]) -> None:
        """Update device from status response."""
        self.last_seen = time.time()
        self.uptime_seconds = status.get("uptime_seconds", self.uptime_seconds)

        # Module status
        if status.get("module1_active", False):
            self.module1.state = ModuleState.ACTIVE
            self.module1.last_active = time.time()
        else:
            self.module1.state = ModuleState.FAILED

        if status.get("module2_active", False):
            self.module2.state = ModuleState.ACTIVE
            self.module2.last_active = time.time()
        else:
            self.module2.state = ModuleState.FAILED

        # Load metrics
        self.total_robot_count = status.get("robot_count", 0)
        self.queue_depth = status.get("queue_depth", 0)
        self.error_count = status.get("error_count", 0)

        # RTT if provided
        if "rtt_ms" in status:
            self.avg_rtt_ms = status["rtt_ms"]

        # Update overall state
        self._update_device_state()

    def _update_device_state(self) -> None:
        """Update device state based on module status."""
        active_modules = self.active_module_count

        if active_modules == 2:
            self.state = DeviceState.ONLINE
        elif active_modules == 1:
            self.state = DeviceState.DEGRADED
        else:
            self.state = DeviceState.ERROR

    def add_robot(self, robot_id: str) -> None:
        """Track robot association."""
        self.robot_ids.add(robot_id)

    def remove_robot(self, robot_id: str) -> None:
        """Remove robot association."""
        self.robot_ids.discard(robot_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "device_id": self.device_id,
            "ip_address": self.ip_address,
            "port": self.port,
            "mac_address": self.mac_address,
            "device_type": self.device_type,
            "firmware_version": self.firmware_version,
            "location": self.location,
            "zone_id": self.zone_id,
            "state": self.state.value,
            "last_seen": self.last_seen,
            "uptime_seconds": self.uptime_seconds,
            "module1_state": self.module1.state.value,
            "module2_state": self.module2.state.value,
            "total_robot_count": self.total_robot_count,
            "queue_depth": self.queue_depth,
            "avg_rtt_ms": self.avg_rtt_ms,
            "loss_rate": self.loss_rate,
            "timeout_rate": self.timeout_rate,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LBAPDevice":
        """Deserialize from dictionary."""
        device = cls(
            device_id=d["device_id"],
            ip_address=d["ip_address"],
            port=d.get("port", 5000),
            mac_address=d.get("mac_address", ""),
            device_type=d.get("device_type", "LBAP-102LU-900"),
            firmware_version=d.get("firmware_version", ""),
            location=d.get("location", ""),
            zone_id=d.get("zone_id", ""),
        )
        device.state = DeviceState(d.get("state", "unknown"))
        device.last_seen = d.get("last_seen", time.time())
        device.uptime_seconds = d.get("uptime_seconds", 0)
        device.module1.state = ModuleState(d.get("module1_state", "unknown"))
        device.module2.state = ModuleState(d.get("module2_state", "unknown"))
        return device


@dataclass
class InventoryConfig:
    """Configuration for device inventory."""
    # Persistence
    inventory_file: Optional[str] = None
    auto_save: bool = True
    save_interval_seconds: float = 60.0

    # Discovery
    auto_discover: bool = True
    discovery_interval_seconds: float = 300.0
    discovery_broadcast_ip: str = "255.255.255.255"

    # Health checking
    health_check_interval_seconds: float = 10.0
    offline_threshold_seconds: float = 30.0
    degraded_threshold_seconds: float = 60.0


class LBAPInventory:
    """
    LBAP device inventory manager.

    Provides:
    - Device registration and lookup
    - Automatic discovery
    - Health monitoring
    - Persistence
    """

    def __init__(self, config: Optional[InventoryConfig] = None):
        self.config = config or InventoryConfig()
        self._devices: Dict[str, LBAPDevice] = {}
        self._devices_by_ip: Dict[str, LBAPDevice] = {}
        self._clients: Dict[str, LBAPUDPClient] = {}
        self._lock = asyncio.Lock()

        # Background tasks
        self._health_task: Optional[asyncio.Task] = None
        self._discovery_task: Optional[asyncio.Task] = None
        self._save_task: Optional[asyncio.Task] = None
        self._running = False

    @property
    def devices(self) -> List[LBAPDevice]:
        """Get all devices."""
        return list(self._devices.values())

    @property
    def online_devices(self) -> List[LBAPDevice]:
        """Get online devices."""
        return [d for d in self._devices.values() if d.is_online]

    @property
    def device_count(self) -> int:
        """Total device count."""
        return len(self._devices)

    @property
    def online_count(self) -> int:
        """Online device count."""
        return len(self.online_devices)

    async def start(self) -> None:
        """Start inventory management tasks."""
        if self._running:
            return

        self._running = True

        # Load persisted inventory
        if self.config.inventory_file:
            self._load_inventory()

        # Start background tasks
        self._health_task = asyncio.create_task(self._health_check_loop())

        if self.config.auto_discover:
            self._discovery_task = asyncio.create_task(self._discovery_loop())

        if self.config.auto_save and self.config.inventory_file:
            self._save_task = asyncio.create_task(self._save_loop())

        logger.info(f"LBAP inventory started with {self.device_count} devices")

    async def stop(self) -> None:
        """Stop inventory management."""
        self._running = False

        # Cancel tasks
        for task in [self._health_task, self._discovery_task, self._save_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close clients
        for client in self._clients.values():
            await client.close()
        self._clients.clear()

        # Save inventory
        if self.config.inventory_file:
            self._save_inventory()

        logger.info("LBAP inventory stopped")

    async def add_device(
        self,
        device_id: str,
        ip_address: str,
        port: int = 5000,
        **kwargs
    ) -> LBAPDevice:
        """
        Add or update device in inventory.

        Args:
            device_id: Unique device identifier
            ip_address: Device IP address
            port: UDP port
            **kwargs: Additional device attributes

        Returns:
            LBAPDevice instance
        """
        async with self._lock:
            if device_id in self._devices:
                device = self._devices[device_id]
                device.ip_address = ip_address
                device.port = port
                for key, value in kwargs.items():
                    if hasattr(device, key):
                        setattr(device, key, value)
            else:
                device = LBAPDevice(
                    device_id=device_id,
                    ip_address=ip_address,
                    port=port,
                    **kwargs
                )
                self._devices[device_id] = device
                self._devices_by_ip[ip_address] = device
                logger.info(f"Added LBAP device: {device_id} at {ip_address}")

            return device

    async def remove_device(self, device_id: str) -> bool:
        """Remove device from inventory."""
        async with self._lock:
            if device_id in self._devices:
                device = self._devices.pop(device_id)
                self._devices_by_ip.pop(device.ip_address, None)

                if device_id in self._clients:
                    await self._clients[device_id].close()
                    del self._clients[device_id]

                logger.info(f"Removed LBAP device: {device_id}")
                return True
            return False

    def get_device(self, device_id: str) -> Optional[LBAPDevice]:
        """Get device by ID."""
        return self._devices.get(device_id)

    def get_device_by_ip(self, ip_address: str) -> Optional[LBAPDevice]:
        """Get device by IP address."""
        return self._devices_by_ip.get(ip_address)

    async def get_client(self, device_id: str) -> Optional[LBAPUDPClient]:
        """Get or create UDP client for device."""
        if device_id not in self._devices:
            return None

        if device_id not in self._clients:
            device = self._devices[device_id]
            config = UDPClientConfig(
                device_ip=device.ip_address,
                device_port=device.port
            )
            client = LBAPUDPClient(config)
            await client.connect()
            self._clients[device_id] = client

        return self._clients[device_id]

    async def refresh_device(self, device_id: str) -> bool:
        """
        Refresh device status via UDP query.

        Returns:
            True if device responded
        """
        device = self.get_device(device_id)
        if not device:
            return False

        client = await self.get_client(device_id)
        if not client:
            return False

        status = await client.get_status()
        if status:
            device.update_from_status(status)
            # Update metrics from client
            metrics = client.get_metrics()
            device.loss_rate = metrics.get("loss_rate", 0.0)
            device.timeout_rate = metrics.get("timeout_rate", 0.0)
            return True
        else:
            # Mark as potentially offline
            if device.staleness_seconds > self.config.offline_threshold_seconds:
                device.state = DeviceState.OFFLINE
            return False

    async def refresh_all(self) -> Dict[str, bool]:
        """Refresh all devices."""
        results = {}
        for device_id in list(self._devices.keys()):
            results[device_id] = await self.refresh_device(device_id)
        return results

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self._running:
            try:
                await self.refresh_all()
            except Exception as e:
                logger.error(f"Health check error: {e}")

            await asyncio.sleep(self.config.health_check_interval_seconds)

    async def _discovery_loop(self) -> None:
        """Background discovery loop."""
        # Initial delay to let existing devices register
        await asyncio.sleep(5.0)

        while self._running:
            try:
                await self.discover()
            except Exception as e:
                logger.error(f"Discovery error: {e}")

            await asyncio.sleep(self.config.discovery_interval_seconds)

    async def _save_loop(self) -> None:
        """Background save loop."""
        while self._running:
            await asyncio.sleep(self.config.save_interval_seconds)
            try:
                self._save_inventory()
            except Exception as e:
                logger.error(f"Save error: {e}")

    async def discover(
        self,
        broadcast_ip: Optional[str] = None,
        timeout: float = 3.0
    ) -> List[LBAPDevice]:
        """
        Discover LBAP devices on the network.

        Args:
            broadcast_ip: Broadcast address (uses config default)
            timeout: Discovery timeout

        Returns:
            List of newly discovered devices
        """
        broadcast_ip = broadcast_ip or self.config.discovery_broadcast_ip

        # Create temporary discovery client
        config = UDPClientConfig(
            device_ip=broadcast_ip,
            device_port=5000
        )
        client = LBAPUDPClient(config)

        try:
            await client.connect()
            discovered = await client.discover_devices(
                broadcast_ip=broadcast_ip,
                timeout=timeout
            )

            new_devices = []
            for info in discovered:
                device_id = info.get("device_id", info["ip_address"])
                if device_id not in self._devices:
                    device = await self.add_device(
                        device_id=device_id,
                        ip_address=info["ip_address"],
                        port=info.get("port", 5000),
                        device_type=info.get("device_type", "LBAP-102LU-900"),
                        firmware_version=info.get("firmware_version", ""),
                        mac_address=info.get("mac_address", ""),
                    )
                    device.state = DeviceState.ONLINE
                    new_devices.append(device)

            if new_devices:
                logger.info(f"Discovered {len(new_devices)} new LBAP devices")

            return new_devices

        finally:
            await client.close()

    def _load_inventory(self) -> None:
        """Load inventory from file."""
        path = Path(self.config.inventory_file)
        if not path.exists():
            return

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            for device_data in data.get("devices", []):
                device = LBAPDevice.from_dict(device_data)
                self._devices[device.device_id] = device
                self._devices_by_ip[device.ip_address] = device

            logger.info(f"Loaded {len(self._devices)} devices from {path}")

        except Exception as e:
            logger.error(f"Failed to load inventory: {e}")

    def _save_inventory(self) -> None:
        """Save inventory to file."""
        if not self.config.inventory_file:
            return

        path = Path(self.config.inventory_file)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 1,
            "saved_at": time.time(),
            "devices": [d.to_dict() for d in self._devices.values()]
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved {len(self._devices)} devices to {path}")

    def get_summary(self) -> Dict[str, Any]:
        """Get inventory summary."""
        states = {}
        for device in self._devices.values():
            state = device.state.value
            states[state] = states.get(state, 0) + 1

        return {
            "total_devices": self.device_count,
            "online_devices": self.online_count,
            "states": states,
            "total_robots": sum(d.total_robot_count for d in self._devices.values()),
        }
