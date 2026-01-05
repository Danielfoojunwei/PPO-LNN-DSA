"""
Libiao RCS API Client

HTTP client for interacting with Libiao Robot Control System.
Implements the RCS API for querying and controlling robots.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from urllib.parse import urljoin

import aiohttp

from .auth import AuthProvider, NoAuth
from .schemas import (
    RobotInfo,
    RobotListResponse,
    SetRobotRequest,
    SetRobotResponse,
    FleetSnapshot,
    SwitchCommand
)

logger = logging.getLogger(__name__)


class RCSClientError(Exception):
    """Base exception for RCS client errors."""
    pass


class RCSConnectionError(RCSClientError):
    """Connection failed to RCS server."""
    pass


class RCSAuthError(RCSClientError):
    """Authentication failed."""
    pass


class RCSAPIError(RCSClientError):
    """API returned an error response."""
    def __init__(self, message: str, status_code: int = 0, response_body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


@dataclass
class RCSClientConfig:
    """Configuration for RCS client."""
    base_url: str
    auth: AuthProvider = field(default_factory=NoAuth)
    timeout_seconds: float = 10.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    verify_ssl: bool = True

    # Rate limiting
    min_request_interval_ms: float = 50.0  # Minimum ms between requests

    @classmethod
    def from_env(cls) -> "RCSClientConfig":
        """Create config from environment variables."""
        import os
        from .auth import AuthFactory

        base_url = os.environ.get("RCS_BASE_URL", "http://localhost:8080")
        auth = AuthFactory.from_env()

        return cls(
            base_url=base_url,
            auth=auth,
            timeout_seconds=float(os.environ.get("RCS_TIMEOUT", "10.0")),
            max_retries=int(os.environ.get("RCS_MAX_RETRIES", "3")),
            verify_ssl=os.environ.get("RCS_VERIFY_SSL", "true").lower() == "true"
        )


class RCSClient:
    """
    Async HTTP client for Libiao RCS API.

    Thread-safe and supports connection pooling.
    """

    QUERY_ROBOTS_PATH = "/com-api/query-robots"
    SET_ROBOT_PATH = "/com-api/set-robot"

    def __init__(self, config: RCSClientConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_request_time: float = 0
        self._lock = asyncio.Lock()

        # Metrics
        self._request_count = 0
        self._error_count = 0
        self._total_latency_ms = 0.0

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def connect(self) -> None:
        """Initialize the HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=20,
                ssl=self.config.verify_ssl if self.config.verify_ssl else False
            )
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _build_url(self, path: str) -> str:
        """Build full URL from path."""
        return urljoin(self.config.base_url, path)

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        async with self._lock:
            now = time.time() * 1000  # ms
            elapsed = now - self._last_request_time
            if elapsed < self.config.min_request_interval_ms:
                await asyncio.sleep((self.config.min_request_interval_ms - elapsed) / 1000)
            self._last_request_time = time.time() * 1000

    async def _request(
        self,
        method: str,
        path: str,
        json: Optional[Dict] = None,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic.

        Args:
            method: HTTP method
            path: URL path
            json: Request body
            retry_count: Current retry attempt

        Returns:
            Response JSON

        Raises:
            RCSConnectionError: Connection failed
            RCSAuthError: Authentication failed
            RCSAPIError: API error response
        """
        await self.connect()
        await self._rate_limit()

        url = self._build_url(path)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **self.config.auth.get_headers()
        }
        params = self.config.auth.get_params()
        cookies = self.config.auth.get_cookies()

        start_time = time.time()
        self._request_count += 1

        try:
            async with self._session.request(
                method,
                url,
                json=json,
                headers=headers,
                params=params,
                cookies=cookies
            ) as response:
                latency_ms = (time.time() - start_time) * 1000
                self._total_latency_ms += latency_ms

                # Log request
                logger.debug(
                    f"RCS {method} {path} -> {response.status} ({latency_ms:.1f}ms)"
                )

                # Handle auth errors
                if response.status in (401, 403):
                    raise RCSAuthError(
                        f"Authentication failed: {response.status}"
                    )

                # Handle server errors with retry
                if response.status >= 500:
                    body = await response.text()
                    if retry_count < self.config.max_retries:
                        await asyncio.sleep(self.config.retry_delay_seconds * (2 ** retry_count))
                        return await self._request(method, path, json, retry_count + 1)
                    raise RCSAPIError(
                        f"Server error: {response.status}",
                        status_code=response.status,
                        response_body=body
                    )

                # Handle client errors
                if response.status >= 400:
                    body = await response.text()
                    raise RCSAPIError(
                        f"Client error: {response.status}",
                        status_code=response.status,
                        response_body=body
                    )

                # Success
                return await response.json()

        except aiohttp.ClientError as e:
            self._error_count += 1
            if retry_count < self.config.max_retries:
                logger.warning(f"RCS connection error, retrying: {e}")
                await asyncio.sleep(self.config.retry_delay_seconds * (2 ** retry_count))
                return await self._request(method, path, json, retry_count + 1)
            raise RCSConnectionError(f"Connection failed: {e}") from e

    async def query_robots(self) -> RobotListResponse:
        """
        Query all robots from RCS.

        POST /com-api/query-robots (no body)

        Returns:
            RobotListResponse with list of all robots

        Raises:
            RCSClientError: On failure
        """
        response = await self._request("POST", self.QUERY_ROBOTS_PATH)
        return RobotListResponse.from_api_response(response)

    async def set_robot(self, request: SetRobotRequest) -> SetRobotResponse:
        """
        Set robot AP/channel assignment.

        POST /com-api/set-robot
        Body: {"robotId":"1","ap":"192.168.11.11","channel":41}

        Args:
            request: SetRobotRequest with target assignment

        Returns:
            SetRobotResponse with result

        Raises:
            RCSClientError: On failure
        """
        response = await self._request(
            "POST",
            self.SET_ROBOT_PATH,
            json=request.to_api_format()
        )
        return SetRobotResponse.from_api_response(response)

    async def execute_switch(self, command: SwitchCommand) -> SetRobotResponse:
        """
        Execute a switch command.

        Args:
            command: SwitchCommand to execute

        Returns:
            SetRobotResponse from RCS
        """
        command.mark_sent()
        request = command.to_request()
        response = await self.set_robot(request)

        if not response.success:
            command.mark_failed(response.message)

        return response

    async def verify_switch(
        self,
        command: SwitchCommand,
        timeout_seconds: float = 5.0,
        poll_interval_seconds: float = 0.5
    ) -> bool:
        """
        Verify that a switch command was executed successfully.

        Polls query-robots until robot is on target AP/channel or timeout.

        Args:
            command: SwitchCommand to verify
            timeout_seconds: Max time to wait
            poll_interval_seconds: Time between polls

        Returns:
            True if switch verified, False otherwise
        """
        start = time.time()

        while (time.time() - start) < timeout_seconds:
            response = await self.query_robots()
            robot = response.get_robot(command.robot_id)

            if robot:
                if (robot.ap_ip == command.target_ap_ip and
                    robot.channel == command.target_channel):
                    command.mark_verified()
                    return True

            await asyncio.sleep(poll_interval_seconds)

        command.mark_failed("Verification timeout")
        return False

    async def get_fleet_snapshot(self) -> FleetSnapshot:
        """
        Get complete fleet state snapshot.

        Returns:
            FleetSnapshot with aggregated statistics
        """
        response = await self.query_robots()
        return FleetSnapshot.from_robot_list(response)

    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics."""
        avg_latency = (
            self._total_latency_ms / self._request_count
            if self._request_count > 0 else 0
        )
        return {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(1, self._request_count),
            "avg_latency_ms": avg_latency
        }

    def reset_metrics(self) -> None:
        """Reset client metrics."""
        self._request_count = 0
        self._error_count = 0
        self._total_latency_ms = 0.0


class SyncRCSClient:
    """
    Synchronous wrapper around async RCS client.

    For use in non-async contexts.
    """

    def __init__(self, config: RCSClientConfig):
        self._config = config
        self._client: Optional[RCSClient] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop."""
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    def _get_client(self) -> RCSClient:
        """Get or create async client."""
        if self._client is None:
            self._client = RCSClient(self._config)
        return self._client

    def _run(self, coro):
        """Run coroutine in event loop."""
        return self._get_loop().run_until_complete(coro)

    def query_robots(self) -> RobotListResponse:
        """Query all robots."""
        return self._run(self._get_client().query_robots())

    def set_robot(self, request: SetRobotRequest) -> SetRobotResponse:
        """Set robot AP/channel."""
        return self._run(self._get_client().set_robot(request))

    def execute_switch(self, command: SwitchCommand) -> SetRobotResponse:
        """Execute switch command."""
        return self._run(self._get_client().execute_switch(command))

    def verify_switch(
        self,
        command: SwitchCommand,
        timeout_seconds: float = 5.0
    ) -> bool:
        """Verify switch command."""
        return self._run(
            self._get_client().verify_switch(command, timeout_seconds)
        )

    def get_fleet_snapshot(self) -> FleetSnapshot:
        """Get fleet snapshot."""
        return self._run(self._get_client().get_fleet_snapshot())

    def close(self) -> None:
        """Close client."""
        if self._client:
            self._run(self._client.close())
            self._client = None
