"""
Libiao RCS API Data Schemas

Defines Pydantic models for RCS API request/response payloads.
Based on Libiao RCS API specification for robot fleet management.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum
import time
import re


class RobotState(Enum):
    """Robot operational states."""
    UNKNOWN = "unknown"
    IDLE = "idle"
    WORKING = "working"
    CHARGING = "charging"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass(frozen=True)
class Coordinate:
    """2D coordinate with optional zone ID."""
    x: float
    y: float
    zone_id: Optional[str] = None

    @classmethod
    def from_string(cls, coord_str: str) -> "Coordinate":
        """
        Parse coordinate string format 'x,y' or 'x,y,zone'.

        Args:
            coord_str: String like '1,2' or '1.5,2.3,zone_a'

        Returns:
            Coordinate instance
        """
        if not coord_str:
            return cls(0.0, 0.0)

        parts = coord_str.split(",")
        try:
            x = float(parts[0].strip()) if len(parts) > 0 else 0.0
            y = float(parts[1].strip()) if len(parts) > 1 else 0.0
            zone_id = parts[2].strip() if len(parts) > 2 else None
            return cls(x, y, zone_id)
        except (ValueError, IndexError):
            return cls(0.0, 0.0)

    def distance_to(self, other: "Coordinate") -> float:
        """Euclidean distance to another coordinate."""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


@dataclass
class RobotInfo:
    """
    Robot information from RCS query-robots API.

    Maps to response format:
    {"robotId":"62","ap":"192.168.11.62","channel":86,"rssi":94,"coordinate":"1,2"}
    """
    robot_id: str
    ap_ip: str
    channel: int
    rssi: int
    coordinate: Coordinate

    # Computed/derived fields (not from API)
    timestamp: float = field(default_factory=time.time)
    state: RobotState = RobotState.UNKNOWN

    @classmethod
    def from_api_response(cls, data: dict) -> "RobotInfo":
        """
        Parse robot info from RCS API response.

        Args:
            data: Dict like {"robotId":"62","ap":"192.168.11.62","channel":86,"rssi":94,"coordinate":"1,2"}

        Returns:
            RobotInfo instance
        """
        return cls(
            robot_id=str(data.get("robotId", "")),
            ap_ip=data.get("ap", ""),
            channel=int(data.get("channel", 0)),
            rssi=int(data.get("rssi", 0)),
            coordinate=Coordinate.from_string(data.get("coordinate", "")),
            timestamp=time.time()
        )

    def to_api_format(self) -> dict:
        """Convert back to API format for serialization."""
        coord_str = f"{self.coordinate.x},{self.coordinate.y}"
        if self.coordinate.zone_id:
            coord_str += f",{self.coordinate.zone_id}"
        return {
            "robotId": self.robot_id,
            "ap": self.ap_ip,
            "channel": self.channel,
            "rssi": self.rssi,
            "coordinate": coord_str
        }


@dataclass
class RobotListResponse:
    """Response from POST /com-api/query-robots."""
    robot_list: List[RobotInfo]
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_api_response(cls, data: dict) -> "RobotListResponse":
        """
        Parse response from query-robots API.

        Args:
            data: Dict with "robotList" key containing list of robot dicts

        Returns:
            RobotListResponse instance
        """
        robot_list = []
        for robot_data in data.get("robotList", []):
            robot_list.append(RobotInfo.from_api_response(robot_data))
        return cls(robot_list=robot_list, timestamp=time.time())

    def get_robot(self, robot_id: str) -> Optional[RobotInfo]:
        """Get robot by ID."""
        for robot in self.robot_list:
            if robot.robot_id == robot_id:
                return robot
        return None


@dataclass
class SetRobotRequest:
    """
    Request body for POST /com-api/set-robot.

    Body format: {"robotId":"1","ap":"192.168.11.11","channel":41}
    """
    robot_id: str
    ap_ip: str
    channel: int

    def to_api_format(self) -> dict:
        """Convert to API request format."""
        return {
            "robotId": self.robot_id,
            "ap": self.ap_ip,
            "channel": self.channel
        }


@dataclass
class SetRobotResponse:
    """
    Response from POST /com-api/set-robot.

    Response format: {"result":0,"message":"OK"}  // 0 = success
    """
    result: int
    message: str

    @property
    def success(self) -> bool:
        """Check if operation was successful (result == 0)."""
        return self.result == 0

    @classmethod
    def from_api_response(cls, data: dict) -> "SetRobotResponse":
        """Parse response from set-robot API."""
        return cls(
            result=int(data.get("result", -1)),
            message=data.get("message", "")
        )


@dataclass
class APInfo:
    """
    Access Point / LBAP device information derived from robot data.

    Aggregates stats across all robots connected to this AP.
    """
    ap_ip: str
    robot_count: int = 0
    channels_in_use: set = field(default_factory=set)
    avg_rssi: float = 0.0
    min_rssi: int = 100
    max_rssi: int = 0
    robot_ids: List[str] = field(default_factory=list)

    def update_from_robot(self, robot: RobotInfo) -> None:
        """Update AP stats from a connected robot."""
        self.robot_ids.append(robot.robot_id)
        self.robot_count = len(self.robot_ids)
        self.channels_in_use.add(robot.channel)

        # Update RSSI stats
        if robot.rssi < self.min_rssi:
            self.min_rssi = robot.rssi
        if robot.rssi > self.max_rssi:
            self.max_rssi = robot.rssi

        # Recalculate average
        total_rssi = self.avg_rssi * (self.robot_count - 1) + robot.rssi
        self.avg_rssi = total_rssi / self.robot_count


@dataclass
class ChannelInfo:
    """
    Channel usage information derived from robot data.

    Tracks robots per channel for contention analysis.
    """
    channel: int
    robot_count: int = 0
    ap_ips: set = field(default_factory=set)
    robot_ids: List[str] = field(default_factory=list)
    avg_rssi: float = 0.0

    def update_from_robot(self, robot: RobotInfo) -> None:
        """Update channel stats from a robot using this channel."""
        self.robot_ids.append(robot.robot_id)
        self.robot_count = len(self.robot_ids)
        self.ap_ips.add(robot.ap_ip)

        # Update average RSSI
        total_rssi = self.avg_rssi * (self.robot_count - 1) + robot.rssi
        self.avg_rssi = total_rssi / self.robot_count


@dataclass
class FleetSnapshot:
    """
    Complete fleet state snapshot at a point in time.

    Aggregates per-robot, per-AP, and per-channel statistics.
    """
    timestamp: float
    robots: List[RobotInfo]
    aps: dict  # ap_ip -> APInfo
    channels: dict  # channel -> ChannelInfo

    @classmethod
    def from_robot_list(cls, response: RobotListResponse) -> "FleetSnapshot":
        """Build fleet snapshot from robot list response."""
        aps = {}
        channels = {}

        for robot in response.robot_list:
            # Aggregate AP stats
            if robot.ap_ip not in aps:
                aps[robot.ap_ip] = APInfo(ap_ip=robot.ap_ip)
            aps[robot.ap_ip].update_from_robot(robot)

            # Aggregate channel stats
            if robot.channel not in channels:
                channels[robot.channel] = ChannelInfo(channel=robot.channel)
            channels[robot.channel].update_from_robot(robot)

        return cls(
            timestamp=response.timestamp,
            robots=response.robot_list,
            aps=aps,
            channels=channels
        )

    @property
    def robot_count(self) -> int:
        """Total number of robots."""
        return len(self.robots)

    @property
    def ap_count(self) -> int:
        """Number of unique APs."""
        return len(self.aps)

    @property
    def channel_count(self) -> int:
        """Number of channels in use."""
        return len(self.channels)

    def get_hotspot_aps(self, threshold: int = 20) -> List[str]:
        """Get APs with more than threshold robots (potential hotspots)."""
        return [ap_ip for ap_ip, info in self.aps.items()
                if info.robot_count > threshold]

    def get_hotspot_channels(self, threshold: int = 30) -> List[int]:
        """Get channels with more than threshold robots."""
        return [ch for ch, info in self.channels.items()
                if info.robot_count > threshold]


@dataclass
class SwitchCommand:
    """
    Command to switch a robot to a new AP/channel.

    Tracks the full lifecycle of a switch attempt.
    """
    robot_id: str
    target_ap_ip: str
    target_channel: int

    # Lifecycle tracking
    created_at: float = field(default_factory=time.time)
    sent_at: Optional[float] = None
    verified_at: Optional[float] = None
    failed_at: Optional[float] = None

    # Original state (for rollback)
    original_ap_ip: Optional[str] = None
    original_channel: Optional[int] = None

    # Result
    success: Optional[bool] = None
    error_message: Optional[str] = None
    retry_count: int = 0

    def to_request(self) -> SetRobotRequest:
        """Convert to API request."""
        return SetRobotRequest(
            robot_id=self.robot_id,
            ap_ip=self.target_ap_ip,
            channel=self.target_channel
        )

    def mark_sent(self) -> None:
        """Mark command as sent to RCS."""
        self.sent_at = time.time()

    def mark_verified(self) -> None:
        """Mark switch as verified (robot moved successfully)."""
        self.verified_at = time.time()
        self.success = True

    def mark_failed(self, error: str) -> None:
        """Mark switch as failed."""
        self.failed_at = time.time()
        self.success = False
        self.error_message = error

    @property
    def is_pending(self) -> bool:
        """Check if command is still pending verification."""
        return self.sent_at is not None and self.verified_at is None and self.failed_at is None

    @property
    def latency_ms(self) -> Optional[float]:
        """Time from send to verification in milliseconds."""
        if self.sent_at and self.verified_at:
            return (self.verified_at - self.sent_at) * 1000
        return None
