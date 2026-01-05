"""
Libiao RCS Poller

Continuous polling of RCS API for robot state updates.
Provides event-driven interface for state changes.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Awaitable, Dict, Any
from enum import Enum

from .client import RCSClient, RCSClientConfig, RCSClientError
from .schemas import RobotListResponse, FleetSnapshot, RobotInfo

logger = logging.getLogger(__name__)


class PollerState(Enum):
    """Poller state machine states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPING = "stopping"


@dataclass
class PollerConfig:
    """Configuration for RCS poller."""
    poll_interval_ms: float = 500.0  # Base polling interval
    min_poll_interval_ms: float = 100.0  # Minimum interval under load
    max_poll_interval_ms: float = 5000.0  # Maximum interval on errors

    # Adaptive polling
    adaptive_polling: bool = True
    scale_up_threshold: int = 100  # Reduce interval above this robot count
    scale_down_threshold: int = 50  # Increase interval below this

    # Error handling
    max_consecutive_errors: int = 10
    error_backoff_multiplier: float = 2.0
    error_recovery_delay_seconds: float = 5.0

    # Change detection
    detect_rssi_change_threshold: int = 5  # dBm change to trigger event
    detect_ap_change: bool = True
    detect_channel_change: bool = True
    detect_coordinate_change: bool = True


# Type aliases for callbacks
RobotCallback = Callable[[RobotInfo, Optional[RobotInfo]], Awaitable[None]]
SnapshotCallback = Callable[[FleetSnapshot, Optional[FleetSnapshot]], Awaitable[None]]
ErrorCallback = Callable[[Exception], Awaitable[None]]


@dataclass
class RobotChangeEvent:
    """Event when robot state changes."""
    robot_id: str
    current: RobotInfo
    previous: Optional[RobotInfo]
    timestamp: float = field(default_factory=time.time)

    # Change flags
    ap_changed: bool = False
    channel_changed: bool = False
    rssi_changed: bool = False
    coordinate_changed: bool = False

    @classmethod
    def detect_changes(
        cls,
        current: RobotInfo,
        previous: Optional[RobotInfo],
        rssi_threshold: int = 5
    ) -> "RobotChangeEvent":
        """Create event with change detection."""
        event = cls(
            robot_id=current.robot_id,
            current=current,
            previous=previous
        )

        if previous is None:
            # New robot
            event.ap_changed = True
            event.channel_changed = True
            event.rssi_changed = True
            event.coordinate_changed = True
        else:
            event.ap_changed = current.ap_ip != previous.ap_ip
            event.channel_changed = current.channel != previous.channel
            event.rssi_changed = abs(current.rssi - previous.rssi) >= rssi_threshold
            event.coordinate_changed = (
                current.coordinate.x != previous.coordinate.x or
                current.coordinate.y != previous.coordinate.y
            )

        return event

    @property
    def has_changes(self) -> bool:
        """Check if any changes occurred."""
        return (
            self.ap_changed or
            self.channel_changed or
            self.rssi_changed or
            self.coordinate_changed
        )


class RCSPoller:
    """
    Continuous poller for RCS robot state.

    Provides:
    - Adaptive polling based on fleet size
    - Change detection with callbacks
    - Error recovery with backoff
    - Fleet statistics aggregation
    """

    def __init__(
        self,
        client: RCSClient,
        config: Optional[PollerConfig] = None
    ):
        self.client = client
        self.config = config or PollerConfig()

        # State
        self._state = PollerState.STOPPED
        self._task: Optional[asyncio.Task] = None
        self._current_interval_ms = self.config.poll_interval_ms
        self._consecutive_errors = 0

        # Data
        self._last_response: Optional[RobotListResponse] = None
        self._last_snapshot: Optional[FleetSnapshot] = None
        self._robot_history: Dict[str, RobotInfo] = {}

        # Callbacks
        self._on_robot_change: List[RobotCallback] = []
        self._on_snapshot: List[SnapshotCallback] = []
        self._on_error: List[ErrorCallback] = []

        # Metrics
        self._poll_count = 0
        self._change_count = 0
        self._error_count = 0
        self._total_poll_time_ms = 0.0

    @property
    def state(self) -> PollerState:
        """Current poller state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if poller is running."""
        return self._state == PollerState.RUNNING

    @property
    def last_snapshot(self) -> Optional[FleetSnapshot]:
        """Get last fleet snapshot."""
        return self._last_snapshot

    def on_robot_change(self, callback: RobotCallback) -> None:
        """Register callback for robot state changes."""
        self._on_robot_change.append(callback)

    def on_snapshot(self, callback: SnapshotCallback) -> None:
        """Register callback for each new snapshot."""
        self._on_snapshot.append(callback)

    def on_error(self, callback: ErrorCallback) -> None:
        """Register callback for errors."""
        self._on_error.append(callback)

    async def start(self) -> None:
        """Start the polling loop."""
        if self._state != PollerState.STOPPED:
            logger.warning(f"Poller already in state {self._state}")
            return

        self._state = PollerState.STARTING
        self._task = asyncio.create_task(self._poll_loop())
        self._state = PollerState.RUNNING
        logger.info("RCS poller started")

    async def stop(self) -> None:
        """Stop the polling loop."""
        if self._state not in (PollerState.RUNNING, PollerState.PAUSED, PollerState.ERROR):
            return

        self._state = PollerState.STOPPING
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._state = PollerState.STOPPED
        logger.info("RCS poller stopped")

    async def pause(self) -> None:
        """Pause polling temporarily."""
        if self._state == PollerState.RUNNING:
            self._state = PollerState.PAUSED
            logger.info("RCS poller paused")

    async def resume(self) -> None:
        """Resume paused polling."""
        if self._state == PollerState.PAUSED:
            self._state = PollerState.RUNNING
            logger.info("RCS poller resumed")

    async def _poll_loop(self) -> None:
        """Main polling loop."""
        while self._state in (PollerState.RUNNING, PollerState.PAUSED):
            if self._state == PollerState.PAUSED:
                await asyncio.sleep(0.1)
                continue

            start_time = time.time()

            try:
                await self._poll_once()
                self._consecutive_errors = 0
                self._adjust_interval_for_success()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._error_count += 1
                self._consecutive_errors += 1
                logger.error(f"Poll error ({self._consecutive_errors}): {e}")

                # Notify error callbacks
                for callback in self._on_error:
                    try:
                        await callback(e)
                    except Exception as cb_error:
                        logger.error(f"Error callback failed: {cb_error}")

                # Check error threshold
                if self._consecutive_errors >= self.config.max_consecutive_errors:
                    self._state = PollerState.ERROR
                    logger.error("Max consecutive errors reached, entering error state")
                    await asyncio.sleep(self.config.error_recovery_delay_seconds)
                    self._state = PollerState.RUNNING
                    self._consecutive_errors = 0

                self._adjust_interval_for_error()

            # Track poll timing
            poll_time_ms = (time.time() - start_time) * 1000
            self._total_poll_time_ms += poll_time_ms

            # Sleep for interval minus poll time
            sleep_ms = max(0, self._current_interval_ms - poll_time_ms)
            await asyncio.sleep(sleep_ms / 1000)

    async def _poll_once(self) -> None:
        """Execute single poll cycle."""
        self._poll_count += 1

        # Query robots
        response = await self.client.query_robots()

        # Build snapshot
        snapshot = FleetSnapshot.from_robot_list(response)

        # Detect changes
        changes = self._detect_changes(response)
        if changes:
            self._change_count += len(changes)

            # Notify robot change callbacks
            for event in changes:
                for callback in self._on_robot_change:
                    try:
                        await callback(event.current, event.previous)
                    except Exception as e:
                        logger.error(f"Robot change callback failed: {e}")

        # Notify snapshot callbacks
        for callback in self._on_snapshot:
            try:
                await callback(snapshot, self._last_snapshot)
            except Exception as e:
                logger.error(f"Snapshot callback failed: {e}")

        # Update history
        self._last_response = response
        self._last_snapshot = snapshot
        for robot in response.robot_list:
            self._robot_history[robot.robot_id] = robot

    def _detect_changes(self, response: RobotListResponse) -> List[RobotChangeEvent]:
        """Detect changes from previous state."""
        changes = []

        for robot in response.robot_list:
            previous = self._robot_history.get(robot.robot_id)
            event = RobotChangeEvent.detect_changes(
                robot,
                previous,
                self.config.detect_rssi_change_threshold
            )

            # Filter by config
            if previous is None:
                changes.append(event)
            elif (
                (self.config.detect_ap_change and event.ap_changed) or
                (self.config.detect_channel_change and event.channel_changed) or
                event.rssi_changed or
                (self.config.detect_coordinate_change and event.coordinate_changed)
            ):
                changes.append(event)

        return changes

    def _adjust_interval_for_success(self) -> None:
        """Adjust polling interval after successful poll."""
        if not self.config.adaptive_polling or not self._last_snapshot:
            return

        robot_count = self._last_snapshot.robot_count

        if robot_count > self.config.scale_up_threshold:
            # More robots -> faster polling
            self._current_interval_ms = max(
                self.config.min_poll_interval_ms,
                self._current_interval_ms * 0.9
            )
        elif robot_count < self.config.scale_down_threshold:
            # Fewer robots -> slower polling
            self._current_interval_ms = min(
                self.config.max_poll_interval_ms,
                self._current_interval_ms * 1.1
            )

    def _adjust_interval_for_error(self) -> None:
        """Adjust polling interval after error."""
        self._current_interval_ms = min(
            self.config.max_poll_interval_ms,
            self._current_interval_ms * self.config.error_backoff_multiplier
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get poller metrics."""
        avg_poll_time = (
            self._total_poll_time_ms / self._poll_count
            if self._poll_count > 0 else 0
        )
        return {
            "state": self._state.value,
            "poll_count": self._poll_count,
            "change_count": self._change_count,
            "error_count": self._error_count,
            "consecutive_errors": self._consecutive_errors,
            "current_interval_ms": self._current_interval_ms,
            "avg_poll_time_ms": avg_poll_time,
            "robot_count": len(self._robot_history)
        }

    def reset_metrics(self) -> None:
        """Reset poller metrics."""
        self._poll_count = 0
        self._change_count = 0
        self._error_count = 0
        self._total_poll_time_ms = 0.0
