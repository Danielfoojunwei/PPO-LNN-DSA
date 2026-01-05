"""
RCS Mock Server for Simulation

Simulates Libiao RCS API for testing and training.
Provides /com-api/query-robots and /com-api/set-robot endpoints.
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class SimulatedRobot:
    """Simulated robot state."""
    robot_id: str
    ap_ip: str
    channel: int
    rssi: int
    x: float
    y: float
    zone_id: str = ""

    # Dynamics
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    target_x: Optional[float] = None
    target_y: Optional[float] = None

    # Switch state
    switch_pending: bool = False
    switch_target_ap: str = ""
    switch_target_channel: int = 0
    switch_start_time: float = 0.0

    # Error simulation
    error_rate: float = 0.0
    last_error: float = 0.0

    def to_api_response(self) -> Dict[str, Any]:
        """Convert to API response format."""
        coord = f"{self.x:.1f},{self.y:.1f}"
        if self.zone_id:
            coord += f",{self.zone_id}"
        return {
            "robotId": self.robot_id,
            "ap": self.ap_ip,
            "channel": self.channel,
            "rssi": self.rssi,
            "coordinate": coord,
        }

    def update_position(self, dt: float) -> None:
        """Update position based on velocity."""
        if self.target_x is not None:
            dx = self.target_x - self.x
            dy = self.target_y - self.y
            dist = (dx**2 + dy**2)**0.5

            if dist < 0.5:
                self.target_x = None
                self.target_y = None
                self.velocity_x = 0
                self.velocity_y = 0
            else:
                speed = 2.0  # m/s
                self.velocity_x = (dx / dist) * speed
                self.velocity_y = (dy / dist) * speed

        self.x += self.velocity_x * dt
        self.y += self.velocity_y * dt


@dataclass
class SimulatedAP:
    """Simulated AP/LBAP device."""
    ap_ip: str
    x: float
    y: float
    max_robots: int = 50
    channels: List[int] = field(default_factory=lambda: list(range(1, 45)))

    # Load
    robot_ids: set = field(default_factory=set)

    # Performance simulation
    base_latency_ms: float = 5.0
    overload_threshold: int = 40

    def get_latency_ms(self) -> float:
        """Get current latency based on load."""
        load = len(self.robot_ids)
        if load <= self.overload_threshold:
            return self.base_latency_ms
        overload_factor = (load - self.overload_threshold) / 10.0
        return self.base_latency_ms * (1 + overload_factor**2)

    def get_loss_rate(self) -> float:
        """Get packet loss rate based on load."""
        load = len(self.robot_ids)
        if load <= self.overload_threshold:
            return 0.0
        overload_factor = (load - self.overload_threshold) / self.max_robots
        return min(0.5, overload_factor**2)


@dataclass
class RCSMockConfig:
    """Configuration for RCS mock."""
    # Fleet size
    num_robots: int = 50
    num_aps: int = 5

    # Area
    area_width: float = 100.0
    area_height: float = 100.0

    # Robot dynamics
    robot_speed: float = 2.0  # m/s
    movement_probability: float = 0.3

    # RSSI simulation
    rssi_base: int = -60
    rssi_distance_factor: float = -2.0  # dBm per meter

    # Switch simulation
    switch_success_rate: float = 0.95
    switch_latency_seconds: float = 1.0

    # Error simulation
    api_error_rate: float = 0.01
    api_latency_min_ms: float = 5.0
    api_latency_max_ms: float = 50.0


class RCSMock:
    """
    Mock RCS server for simulation.

    Simulates:
    - Robot fleet with positions and assignments
    - AP load and performance
    - Switch execution with realistic delays
    - RSSI based on distance
    """

    def __init__(self, config: Optional[RCSMockConfig] = None):
        self.config = config or RCSMockConfig()

        # State
        self.robots: Dict[str, SimulatedRobot] = {}
        self.aps: Dict[str, SimulatedAP] = {}

        # Time tracking
        self.sim_time: float = 0.0
        self.last_tick: float = time.time()

        # Metrics
        self.query_count: int = 0
        self.set_count: int = 0
        self.set_success_count: int = 0
        self.set_fail_count: int = 0

        self._initialize_fleet()

    def _initialize_fleet(self) -> None:
        """Initialize simulated fleet."""
        # Create APs in grid pattern
        num_aps = self.config.num_aps
        grid_size = int(num_aps**0.5) + 1
        ap_spacing_x = self.config.area_width / grid_size
        ap_spacing_y = self.config.area_height / grid_size

        for i in range(num_aps):
            row = i // grid_size
            col = i % grid_size
            ap_ip = f"192.168.1.{10 + i}"

            self.aps[ap_ip] = SimulatedAP(
                ap_ip=ap_ip,
                x=(col + 0.5) * ap_spacing_x,
                y=(row + 0.5) * ap_spacing_y,
            )

        # Create robots distributed across APs
        ap_list = list(self.aps.keys())
        for i in range(self.config.num_robots):
            robot_id = str(i + 1)
            ap_ip = ap_list[i % len(ap_list)]
            ap = self.aps[ap_ip]

            # Random position near AP
            x = ap.x + random.uniform(-20, 20)
            y = ap.y + random.uniform(-20, 20)
            x = max(0, min(self.config.area_width, x))
            y = max(0, min(self.config.area_height, y))

            channel = random.choice(ap.channels)
            rssi = self._calculate_rssi(x, y, ap.x, ap.y)

            robot = SimulatedRobot(
                robot_id=robot_id,
                ap_ip=ap_ip,
                channel=channel,
                rssi=rssi,
                x=x,
                y=y,
                zone_id=f"zone_{int(x/25)}_{int(y/25)}",
            )
            self.robots[robot_id] = robot
            ap.robot_ids.add(robot_id)

    def _calculate_rssi(self, rx: float, ry: float, tx: float, ty: float) -> int:
        """Calculate RSSI based on distance."""
        distance = ((rx - tx)**2 + (ry - ty)**2)**0.5
        rssi = self.config.rssi_base + self.config.rssi_distance_factor * distance
        rssi += random.gauss(0, 3)  # Noise
        return int(max(-100, min(-30, rssi)))

    def tick(self, dt: Optional[float] = None) -> None:
        """
        Advance simulation time.

        Args:
            dt: Time delta in seconds (auto-calculated if None)
        """
        if dt is None:
            now = time.time()
            dt = now - self.last_tick
            self.last_tick = now

        self.sim_time += dt

        for robot in self.robots.values():
            # Update position
            robot.update_position(dt)

            # Maybe start new movement
            if robot.target_x is None and random.random() < self.config.movement_probability * dt:
                robot.target_x = random.uniform(0, self.config.area_width)
                robot.target_y = random.uniform(0, self.config.area_height)

            # Update zone
            robot.zone_id = f"zone_{int(robot.x/25)}_{int(robot.y/25)}"

            # Update RSSI
            ap = self.aps.get(robot.ap_ip)
            if ap:
                robot.rssi = self._calculate_rssi(robot.x, robot.y, ap.x, ap.y)

            # Process pending switches
            if robot.switch_pending:
                elapsed = self.sim_time - robot.switch_start_time
                if elapsed >= self.config.switch_latency_seconds:
                    robot.switch_pending = False
                    robot.ap_ip = robot.switch_target_ap
                    robot.channel = robot.switch_target_channel

                    # Update AP associations
                    for ap in self.aps.values():
                        ap.robot_ids.discard(robot.robot_id)
                    if robot.ap_ip in self.aps:
                        self.aps[robot.ap_ip].robot_ids.add(robot.robot_id)

    async def query_robots(self) -> Dict[str, Any]:
        """
        Simulate POST /com-api/query-robots.

        Returns:
            API response dict
        """
        self.query_count += 1

        # Simulate API latency
        latency = random.uniform(
            self.config.api_latency_min_ms,
            self.config.api_latency_max_ms
        )
        await asyncio.sleep(latency / 1000)

        # Simulate errors
        if random.random() < self.config.api_error_rate:
            raise Exception("Simulated API error")

        robot_list = [
            robot.to_api_response()
            for robot in self.robots.values()
        ]

        return {"robotList": robot_list}

    async def set_robot(
        self,
        robot_id: str,
        ap_ip: str,
        channel: int
    ) -> Dict[str, Any]:
        """
        Simulate POST /com-api/set-robot.

        Args:
            robot_id: Target robot ID
            ap_ip: Target AP IP
            channel: Target channel

        Returns:
            API response dict
        """
        self.set_count += 1

        # Simulate API latency
        latency = random.uniform(
            self.config.api_latency_min_ms,
            self.config.api_latency_max_ms
        )
        await asyncio.sleep(latency / 1000)

        # Validate robot
        if robot_id not in self.robots:
            self.set_fail_count += 1
            return {"result": 1, "message": "Robot not found"}

        # Validate AP
        if ap_ip not in self.aps:
            self.set_fail_count += 1
            return {"result": 2, "message": "AP not found"}

        # Simulate switch failure
        if random.random() > self.config.switch_success_rate:
            self.set_fail_count += 1
            return {"result": 3, "message": "Switch failed"}

        # Initiate switch
        robot = self.robots[robot_id]
        robot.switch_pending = True
        robot.switch_target_ap = ap_ip
        robot.switch_target_channel = channel
        robot.switch_start_time = self.sim_time

        self.set_success_count += 1
        return {"result": 0, "message": "OK"}

    def get_robot_info(self, robot_id: str) -> Optional[Dict[str, Any]]:
        """Get current robot info (synchronous)."""
        if robot_id in self.robots:
            return self.robots[robot_id].to_api_response()
        return None

    def get_ap_load(self, ap_ip: str) -> int:
        """Get robot count for an AP."""
        if ap_ip in self.aps:
            return len(self.aps[ap_ip].robot_ids)
        return 0

    def get_channel_load(self, channel: int) -> int:
        """Get robot count for a channel."""
        return sum(
            1 for robot in self.robots.values()
            if robot.channel == channel
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get simulation metrics."""
        ap_loads = {
            ap_ip: len(ap.robot_ids)
            for ap_ip, ap in self.aps.items()
        }
        return {
            "sim_time": self.sim_time,
            "num_robots": len(self.robots),
            "num_aps": len(self.aps),
            "query_count": self.query_count,
            "set_count": self.set_count,
            "set_success_rate": (
                self.set_success_count / max(1, self.set_count)
            ),
            "ap_loads": ap_loads,
        }

    def get_hotspot_aps(self, threshold: int = 40) -> List[str]:
        """Get APs with high load."""
        return [
            ap_ip for ap_ip, ap in self.aps.items()
            if len(ap.robot_ids) > threshold
        ]


class RCSMockServer:
    """
    HTTP server wrapper for RCS mock.

    Provides actual HTTP endpoints for integration testing.
    """

    def __init__(self, mock: RCSMock, host: str = "127.0.0.1", port: int = 8080):
        self.mock = mock
        self.host = host
        self.port = port
        self._server = None

    async def start(self) -> None:
        """Start HTTP server."""
        from aiohttp import web

        app = web.Application()
        app.router.add_post("/com-api/query-robots", self._handle_query)
        app.router.add_post("/com-api/set-robot", self._handle_set)

        runner = web.AppRunner(app)
        await runner.setup()
        self._server = web.TCPSite(runner, self.host, self.port)
        await self._server.start()
        logger.info(f"RCS mock server started at http://{self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop HTTP server."""
        if self._server:
            await self._server.stop()

    async def _handle_query(self, request) -> 'web.Response':
        from aiohttp import web

        try:
            self.mock.tick()
            result = await self.mock.query_robots()
            return web.json_response(result)
        except Exception as e:
            return web.json_response(
                {"error": str(e)},
                status=500
            )

    async def _handle_set(self, request) -> 'web.Response':
        from aiohttp import web

        try:
            self.mock.tick()
            data = await request.json()
            result = await self.mock.set_robot(
                robot_id=data.get("robotId", ""),
                ap_ip=data.get("ap", ""),
                channel=int(data.get("channel", 0))
            )
            return web.json_response(result)
        except Exception as e:
            return web.json_response(
                {"result": -1, "message": str(e)},
                status=500
            )
