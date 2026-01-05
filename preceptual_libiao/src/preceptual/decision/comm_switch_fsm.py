"""
Communication Switch Finite State Machine

Controls robot handover execution with verification, cooldown,
and ping-pong prevention.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Awaitable
from enum import Enum

logger = logging.getLogger(__name__)


class SwitchState(Enum):
    """FSM states for a switch operation."""
    IDLE = "idle"
    ARMED = "armed"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    FAILED = "failed"
    COOLDOWN = "cooldown"


class SwitchResult(Enum):
    """Result of a switch operation."""
    SUCCESS = "success"
    TIMEOUT = "timeout"
    REJECTED = "rejected"  # By RCS
    PING_PONG = "ping_pong"  # Prevented
    COOLDOWN = "cooldown"  # Still in cooldown
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class SwitchTarget:
    """Target specification for a switch."""
    robot_id: str
    target_ap_ip: str
    target_channel: int
    priority: int = 0  # Higher = more urgent

    # Original state (for rollback tracking)
    original_ap_ip: str = ""
    original_channel: int = 0


@dataclass
class SwitchOperation:
    """Complete switch operation with lifecycle tracking."""
    # Target
    target: SwitchTarget

    # State
    state: SwitchState = SwitchState.IDLE

    # Timestamps
    created_at: float = field(default_factory=time.time)
    armed_at: Optional[float] = None
    executed_at: Optional[float] = None
    verified_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Result
    result: Optional[SwitchResult] = None
    error_message: str = ""
    retry_count: int = 0
    max_retries: int = 3

    # Verification
    verification_attempts: int = 0
    max_verification_attempts: int = 10

    @property
    def robot_id(self) -> str:
        return self.target.robot_id

    @property
    def is_complete(self) -> bool:
        return self.state in (SwitchState.VERIFIED, SwitchState.FAILED)

    @property
    def latency_ms(self) -> Optional[float]:
        """Time from execution to verification in milliseconds."""
        if self.executed_at and self.verified_at:
            return (self.verified_at - self.executed_at) * 1000
        return None

    @property
    def age_seconds(self) -> float:
        """Time since creation."""
        return time.time() - self.created_at

    def arm(self) -> None:
        """Transition to ARMED state."""
        self.state = SwitchState.ARMED
        self.armed_at = time.time()

    def execute(self) -> None:
        """Transition to EXECUTING state."""
        self.state = SwitchState.EXECUTING
        self.executed_at = time.time()

    def start_verification(self) -> None:
        """Transition to VERIFYING state."""
        self.state = SwitchState.VERIFYING

    def verify_success(self) -> None:
        """Mark verification as successful."""
        self.state = SwitchState.VERIFIED
        self.verified_at = time.time()
        self.completed_at = time.time()
        self.result = SwitchResult.SUCCESS

    def fail(self, result: SwitchResult, message: str = "") -> None:
        """Mark operation as failed."""
        self.state = SwitchState.FAILED
        self.completed_at = time.time()
        self.result = result
        self.error_message = message

    def enter_cooldown(self) -> None:
        """Transition to COOLDOWN state."""
        self.state = SwitchState.COOLDOWN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "target_ap": self.target.target_ap_ip,
            "target_channel": self.target.target_channel,
            "state": self.state.value,
            "result": self.result.value if self.result else None,
            "latency_ms": self.latency_ms,
            "retry_count": self.retry_count,
            "age_seconds": self.age_seconds,
        }


@dataclass
class FSMConfig:
    """Configuration for CommSwitchFSM."""
    # Timing
    min_dwell_time_seconds: float = 10.0  # Minimum time between switches
    verification_timeout_seconds: float = 5.0
    verification_poll_interval_seconds: float = 0.5
    execution_timeout_seconds: float = 2.0
    cooldown_duration_seconds: float = 30.0

    # Limits
    max_retries: int = 3
    max_concurrent_operations: int = 5
    max_switches_per_robot_per_minute: int = 3

    # Ping-pong prevention
    ping_pong_window_seconds: float = 60.0
    ping_pong_threshold: int = 2  # Max switches back to same AP in window


class RobotSwitchHistory:
    """Tracks switch history for a robot."""

    def __init__(self, robot_id: str, window_seconds: float = 60.0):
        self.robot_id = robot_id
        self.window_seconds = window_seconds
        self.history: List[Dict[str, Any]] = []  # List of {timestamp, from_ap, to_ap}
        self.last_switch_time: float = 0.0

    def record_switch(self, from_ap: str, to_ap: str, timestamp: float) -> None:
        """Record a completed switch."""
        self.history.append({
            "timestamp": timestamp,
            "from_ap": from_ap,
            "to_ap": to_ap,
        })
        self.last_switch_time = timestamp

        # Trim old events
        cutoff = timestamp - self.window_seconds
        self.history = [h for h in self.history if h["timestamp"] >= cutoff]

    def get_switch_count(self, window_seconds: Optional[float] = None) -> int:
        """Get switch count in window."""
        cutoff = time.time() - (window_seconds or self.window_seconds)
        return sum(1 for h in self.history if h["timestamp"] >= cutoff)

    def is_ping_pong(self, target_ap: str, threshold: int = 2) -> bool:
        """Check if switching to target_ap would be ping-pong."""
        # Count recent switches TO this AP
        count = sum(1 for h in self.history if h["to_ap"] == target_ap)
        return count >= threshold

    def time_since_last_switch(self) -> float:
        """Time since last switch."""
        if self.last_switch_time == 0:
            return float('inf')
        return time.time() - self.last_switch_time


class CommSwitchFSM:
    """
    Finite State Machine for robot communication switching.

    Manages the lifecycle of switch operations with:
    - Verification after execution
    - Cooldown enforcement
    - Ping-pong prevention
    - Retry logic with exponential backoff
    - Concurrent operation limiting
    """

    def __init__(
        self,
        config: Optional[FSMConfig] = None,
        execute_fn: Optional[Callable[[SwitchTarget], Awaitable[bool]]] = None,
        verify_fn: Optional[Callable[[str], Awaitable[Optional[Dict[str, Any]]]]] = None,
    ):
        self.config = config or FSMConfig()
        self.execute_fn = execute_fn
        self.verify_fn = verify_fn

        # Active operations
        self._operations: Dict[str, SwitchOperation] = {}  # robot_id -> operation
        self._pending_queue: List[SwitchOperation] = []

        # Robot history
        self._robot_history: Dict[str, RobotSwitchHistory] = {}

        # Metrics
        self._total_switches = 0
        self._successful_switches = 0
        self._failed_switches = 0
        self._ping_pong_prevented = 0

    def _get_history(self, robot_id: str) -> RobotSwitchHistory:
        if robot_id not in self._robot_history:
            self._robot_history[robot_id] = RobotSwitchHistory(
                robot_id=robot_id,
                window_seconds=self.config.ping_pong_window_seconds
            )
        return self._robot_history[robot_id]

    def can_switch(self, robot_id: str, target_ap: str) -> tuple[bool, str]:
        """
        Check if a switch is allowed.

        Returns:
            (allowed, reason)
        """
        history = self._get_history(robot_id)

        # Check dwell time
        if history.time_since_last_switch() < self.config.min_dwell_time_seconds:
            return False, "Dwell time not elapsed"

        # Check switch rate limit
        recent_count = history.get_switch_count(60.0)
        if recent_count >= self.config.max_switches_per_robot_per_minute:
            return False, "Switch rate limit exceeded"

        # Check ping-pong
        if history.is_ping_pong(target_ap, self.config.ping_pong_threshold):
            return False, "Ping-pong detected"

        # Check concurrent operation
        if robot_id in self._operations:
            op = self._operations[robot_id]
            if not op.is_complete:
                return False, "Switch already in progress"

        # Check concurrent limit
        active_count = sum(1 for op in self._operations.values() if not op.is_complete)
        if active_count >= self.config.max_concurrent_operations:
            return False, "Max concurrent operations reached"

        return True, "OK"

    def arm_switch(
        self,
        robot_id: str,
        target_ap_ip: str,
        target_channel: int,
        original_ap_ip: str = "",
        original_channel: int = 0,
        priority: int = 0,
    ) -> Optional[SwitchOperation]:
        """
        Arm a switch for execution.

        Args:
            robot_id: Robot to switch
            target_ap_ip: Target AP IP
            target_channel: Target channel
            original_ap_ip: Current AP (for tracking)
            original_channel: Current channel (for tracking)
            priority: Switch priority

        Returns:
            SwitchOperation if armed, None if rejected
        """
        allowed, reason = self.can_switch(robot_id, target_ap_ip)
        if not allowed:
            logger.debug(f"Switch rejected for {robot_id}: {reason}")
            if "ping-pong" in reason.lower():
                self._ping_pong_prevented += 1
            return None

        target = SwitchTarget(
            robot_id=robot_id,
            target_ap_ip=target_ap_ip,
            target_channel=target_channel,
            priority=priority,
            original_ap_ip=original_ap_ip,
            original_channel=original_channel,
        )

        operation = SwitchOperation(target=target)
        operation.arm()

        self._operations[robot_id] = operation
        logger.debug(f"Armed switch for {robot_id} to {target_ap_ip}:{target_channel}")

        return operation

    async def execute_switch(self, operation: SwitchOperation) -> bool:
        """
        Execute an armed switch operation.

        Args:
            operation: Armed operation to execute

        Returns:
            True if execution sent successfully
        """
        if operation.state != SwitchState.ARMED:
            logger.warning(f"Cannot execute switch in state {operation.state}")
            return False

        operation.execute()
        self._total_switches += 1

        if self.execute_fn:
            try:
                success = await asyncio.wait_for(
                    self.execute_fn(operation.target),
                    timeout=self.config.execution_timeout_seconds
                )
                if not success:
                    operation.fail(SwitchResult.REJECTED, "RCS rejected switch")
                    self._failed_switches += 1
                    return False
            except asyncio.TimeoutError:
                operation.fail(SwitchResult.TIMEOUT, "Execution timeout")
                self._failed_switches += 1
                return False
            except Exception as e:
                operation.fail(SwitchResult.ERROR, str(e))
                self._failed_switches += 1
                return False

        operation.start_verification()
        return True

    async def verify_switch(self, operation: SwitchOperation) -> bool:
        """
        Verify that a switch completed successfully.

        Args:
            operation: Operation to verify

        Returns:
            True if verified successfully
        """
        if operation.state != SwitchState.VERIFYING:
            return False

        if not self.verify_fn:
            # No verify function - assume success
            operation.verify_success()
            self._record_switch_complete(operation)
            return True

        start_time = time.time()
        timeout = self.config.verification_timeout_seconds

        while (time.time() - start_time) < timeout:
            operation.verification_attempts += 1

            try:
                robot_info = await self.verify_fn(operation.robot_id)

                if robot_info:
                    current_ap = robot_info.get("ap", "")
                    current_channel = robot_info.get("channel", 0)

                    if (current_ap == operation.target.target_ap_ip and
                        current_channel == operation.target.target_channel):
                        operation.verify_success()
                        self._record_switch_complete(operation)
                        return True

            except Exception as e:
                logger.warning(f"Verification check failed: {e}")

            if operation.verification_attempts >= operation.max_verification_attempts:
                break

            await asyncio.sleep(self.config.verification_poll_interval_seconds)

        # Verification failed
        operation.fail(SwitchResult.TIMEOUT, "Verification timeout")
        self._failed_switches += 1
        return False

    def _record_switch_complete(self, operation: SwitchOperation) -> None:
        """Record successful switch completion."""
        self._successful_switches += 1

        history = self._get_history(operation.robot_id)
        history.record_switch(
            from_ap=operation.target.original_ap_ip,
            to_ap=operation.target.target_ap_ip,
            timestamp=time.time()
        )

        logger.info(
            f"Switch verified for {operation.robot_id}: "
            f"{operation.target.original_ap_ip} -> {operation.target.target_ap_ip} "
            f"(latency: {operation.latency_ms:.0f}ms)"
        )

    async def execute_and_verify(self, operation: SwitchOperation) -> SwitchResult:
        """
        Execute a switch and verify completion.

        Args:
            operation: Armed operation

        Returns:
            SwitchResult
        """
        if not await self.execute_switch(operation):
            return operation.result or SwitchResult.ERROR

        if not await self.verify_switch(operation):
            # Retry if possible
            if operation.retry_count < operation.max_retries:
                operation.retry_count += 1
                operation.state = SwitchState.ARMED
                logger.info(f"Retrying switch for {operation.robot_id} (attempt {operation.retry_count})")
                return await self.execute_and_verify(operation)

            return operation.result or SwitchResult.TIMEOUT

        return SwitchResult.SUCCESS

    def cancel_switch(self, robot_id: str) -> bool:
        """Cancel a pending switch."""
        if robot_id in self._operations:
            op = self._operations[robot_id]
            if not op.is_complete:
                op.fail(SwitchResult.CANCELLED, "Cancelled by user")
                return True
        return False

    def get_operation(self, robot_id: str) -> Optional[SwitchOperation]:
        """Get current operation for a robot."""
        return self._operations.get(robot_id)

    def get_active_operations(self) -> List[SwitchOperation]:
        """Get all active (non-complete) operations."""
        return [op for op in self._operations.values() if not op.is_complete]

    def cleanup_completed(self, max_age_seconds: float = 300.0) -> int:
        """Remove old completed operations."""
        now = time.time()
        to_remove = []

        for robot_id, op in self._operations.items():
            if op.is_complete and (now - op.completed_at) > max_age_seconds:
                to_remove.append(robot_id)

        for robot_id in to_remove:
            del self._operations[robot_id]

        return len(to_remove)

    def get_metrics(self) -> Dict[str, Any]:
        """Get FSM metrics."""
        active_ops = self.get_active_operations()
        return {
            "total_switches": self._total_switches,
            "successful_switches": self._successful_switches,
            "failed_switches": self._failed_switches,
            "ping_pong_prevented": self._ping_pong_prevented,
            "success_rate": (
                self._successful_switches / max(1, self._total_switches)
            ),
            "active_operations": len(active_ops),
            "pending_queue_size": len(self._pending_queue),
        }
