"""
LBAP UDP Client

Async UDP client for communicating with LBAP-102LU-900 devices.
Handles connection, message sending/receiving, and RTT tracking.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, Awaitable, Tuple
from collections import deque
import socket

from .codecs import (
    MessageCodec,
    CodecRegistry,
    LBAPMessage,
    EchoRequest,
    EchoResponse,
    StatusRequest,
    DiscoveryRequest,
    MessageType,
)

logger = logging.getLogger(__name__)


class UDPClientError(Exception):
    """Base exception for UDP client errors."""
    pass


class UDPConnectionError(UDPClientError):
    """Connection/network error."""
    pass


class UDPTimeoutError(UDPClientError):
    """Request timeout."""
    pass


@dataclass
class UDPClientConfig:
    """Configuration for UDP client."""
    # Target device
    device_ip: str = "192.168.0.200"  # Default factory IP
    device_port: int = 5000

    # Local binding
    local_ip: str = "0.0.0.0"
    local_port: int = 0  # Auto-assign

    # Timeouts
    connect_timeout_seconds: float = 5.0
    request_timeout_seconds: float = 2.0

    # Retries
    max_retries: int = 3
    retry_delay_seconds: float = 0.5

    # Protocol
    codec_version: str = "v0"

    # Metrics
    rtt_window_size: int = 100  # Number of RTT samples to keep


@dataclass
class UDPClientMetrics:
    """UDP client metrics."""
    packets_sent: int = 0
    packets_received: int = 0
    packets_lost: int = 0
    timeouts: int = 0
    errors: int = 0

    # RTT stats (ms)
    rtt_samples: deque = field(default_factory=lambda: deque(maxlen=100))
    min_rtt_ms: float = float('inf')
    max_rtt_ms: float = 0.0

    @property
    def loss_rate(self) -> float:
        """Packet loss rate."""
        total = self.packets_sent
        if total == 0:
            return 0.0
        return self.packets_lost / total

    @property
    def avg_rtt_ms(self) -> float:
        """Average RTT in milliseconds."""
        if not self.rtt_samples:
            return 0.0
        return sum(self.rtt_samples) / len(self.rtt_samples)

    @property
    def timeout_rate(self) -> float:
        """Timeout rate."""
        total = self.packets_sent
        if total == 0:
            return 0.0
        return self.timeouts / total

    def record_rtt(self, rtt_ms: float) -> None:
        """Record an RTT sample."""
        self.rtt_samples.append(rtt_ms)
        if rtt_ms < self.min_rtt_ms:
            self.min_rtt_ms = rtt_ms
        if rtt_ms > self.max_rtt_ms:
            self.max_rtt_ms = rtt_ms

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "packets_sent": self.packets_sent,
            "packets_received": self.packets_received,
            "packets_lost": self.packets_lost,
            "timeouts": self.timeouts,
            "errors": self.errors,
            "loss_rate": self.loss_rate,
            "timeout_rate": self.timeout_rate,
            "avg_rtt_ms": self.avg_rtt_ms,
            "min_rtt_ms": self.min_rtt_ms if self.min_rtt_ms != float('inf') else 0.0,
            "max_rtt_ms": self.max_rtt_ms,
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.packets_sent = 0
        self.packets_received = 0
        self.packets_lost = 0
        self.timeouts = 0
        self.errors = 0
        self.rtt_samples.clear()
        self.min_rtt_ms = float('inf')
        self.max_rtt_ms = 0.0


class LBAPUDPProtocol(asyncio.DatagramProtocol):
    """Asyncio datagram protocol for LBAP communication."""

    def __init__(
        self,
        codec: MessageCodec,
        on_message: Optional[Callable[[LBAPMessage, Tuple[str, int]], None]] = None
    ):
        self.codec = codec
        self.on_message = on_message
        self.transport: Optional[asyncio.DatagramTransport] = None
        self._pending: Dict[int, asyncio.Future] = {}
        self._closed = False

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:
        self.transport = transport

    def connection_lost(self, exc: Optional[Exception]) -> None:
        self._closed = True
        # Cancel all pending requests
        for future in self._pending.values():
            if not future.done():
                future.cancel()
        self._pending.clear()

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        """Handle received datagram."""
        try:
            message = self.codec.decode(data)
            message.source_ip = addr[0]
            message.source_port = addr[1]

            # Check for pending request
            if message.sequence_id in self._pending:
                future = self._pending.pop(message.sequence_id)
                if not future.done():
                    future.set_result(message)

            # Call callback if registered
            if self.on_message:
                self.on_message(message, addr)

        except Exception as e:
            logger.error(f"Error decoding datagram from {addr}: {e}")

    def error_received(self, exc: Exception) -> None:
        logger.error(f"UDP error: {exc}")

    def send_message(
        self,
        message: LBAPMessage,
        addr: Tuple[str, int]
    ) -> asyncio.Future:
        """Send message and return future for response."""
        if self._closed or not self.transport:
            future = asyncio.get_event_loop().create_future()
            future.set_exception(UDPConnectionError("Transport closed"))
            return future

        data = self.codec.encode(message)
        self.transport.sendto(data, addr)

        # Create future for response
        future = asyncio.get_event_loop().create_future()
        self._pending[message.sequence_id] = future
        return future

    def close(self) -> None:
        """Close the transport."""
        self._closed = True
        if self.transport:
            self.transport.close()


class LBAPUDPClient:
    """
    Async UDP client for LBAP device communication.

    Provides:
    - Echo/ping for connectivity testing
    - Status requests
    - Discovery broadcasts
    - RTT and loss tracking
    """

    def __init__(self, config: UDPClientConfig):
        self.config = config
        self.codec = CodecRegistry.get(config.codec_version) or CodecRegistry.default()
        self.metrics = UDPClientMetrics()

        self._protocol: Optional[LBAPUDPProtocol] = None
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._sequence_id = 0
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected and self._transport is not None

    @property
    def device_address(self) -> Tuple[str, int]:
        """Get target device address."""
        return (self.config.device_ip, self.config.device_port)

    async def connect(self) -> None:
        """
        Create UDP socket and bind.

        Note: UDP is connectionless, this just creates the socket.
        """
        if self._connected:
            return

        loop = asyncio.get_event_loop()

        self._protocol = LBAPUDPProtocol(self.codec)
        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: self._protocol,
            local_addr=(self.config.local_ip, self.config.local_port)
        )
        self._connected = True
        logger.info(f"LBAP UDP client connected to {self.device_address}")

    async def close(self) -> None:
        """Close the UDP socket."""
        if self._protocol:
            self._protocol.close()
        self._protocol = None
        self._transport = None
        self._connected = False
        logger.info("LBAP UDP client closed")

    def _next_sequence_id(self) -> int:
        """Get next sequence ID."""
        self._sequence_id = (self._sequence_id + 1) & 0xFFFF
        return self._sequence_id

    async def _send_and_wait(
        self,
        message: LBAPMessage,
        timeout: Optional[float] = None
    ) -> LBAPMessage:
        """
        Send message and wait for response.

        Args:
            message: Message to send
            timeout: Response timeout (uses config default if None)

        Returns:
            Response message

        Raises:
            UDPTimeoutError: On timeout
            UDPConnectionError: On connection error
        """
        if not self.is_connected:
            await self.connect()

        timeout = timeout or self.config.request_timeout_seconds
        message.sequence_id = self._next_sequence_id()
        message.timestamp = time.time()

        self.metrics.packets_sent += 1
        send_time = time.time()

        try:
            future = self._protocol.send_message(message, self.device_address)
            response = await asyncio.wait_for(future, timeout=timeout)

            # Calculate RTT
            rtt_ms = (time.time() - send_time) * 1000
            response.rtt_ms = rtt_ms
            self.metrics.record_rtt(rtt_ms)
            self.metrics.packets_received += 1

            return response

        except asyncio.TimeoutError:
            self.metrics.timeouts += 1
            self.metrics.packets_lost += 1
            raise UDPTimeoutError(
                f"Request timeout after {timeout}s to {self.device_address}"
            )
        except Exception as e:
            self.metrics.errors += 1
            raise UDPConnectionError(f"Request failed: {e}") from e

    async def ping(self, data: bytes = b"PING") -> Tuple[bool, float]:
        """
        Send echo request and measure RTT.

        Args:
            data: Echo payload

        Returns:
            Tuple of (success, rtt_ms)
        """
        request = EchoRequest(echo_data=data)

        for attempt in range(self.config.max_retries):
            try:
                response = await self._send_and_wait(request)
                if isinstance(response, EchoResponse):
                    return True, response.rtt_ms or 0.0
            except UDPTimeoutError:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay_seconds)
                continue
            except UDPConnectionError:
                break

        return False, 0.0

    async def get_status(self) -> Optional[Dict[str, Any]]:
        """
        Request device status.

        Returns:
            Status dict or None on failure
        """
        request = StatusRequest()

        try:
            response = await self._send_and_wait(request)
            if response.message_type == MessageType.STATUS_RESPONSE:
                return {
                    "uptime_seconds": getattr(response, 'uptime_seconds', 0),
                    "module1_active": getattr(response, 'module1_active', True),
                    "module2_active": getattr(response, 'module2_active', True),
                    "current_channel": getattr(response, 'current_channel', 0),
                    "tx_power_dbm": getattr(response, 'tx_power_dbm', 14),
                    "robot_count": getattr(response, 'robot_count', 0),
                    "error_count": getattr(response, 'error_count', 0),
                    "queue_depth": getattr(response, 'queue_depth', 0),
                    "rtt_ms": response.rtt_ms,
                }
        except (UDPTimeoutError, UDPConnectionError) as e:
            logger.warning(f"Status request failed: {e}")

        return None

    async def discover_devices(
        self,
        broadcast_ip: str = "255.255.255.255",
        port: int = 5000,
        timeout: float = 3.0
    ) -> list:
        """
        Broadcast discovery request and collect responses.

        Args:
            broadcast_ip: Broadcast address
            port: Discovery port
            timeout: Time to wait for responses

        Returns:
            List of discovered device info dicts
        """
        if not self.is_connected:
            await self.connect()

        # Enable broadcast
        sock = self._transport.get_extra_info('socket')
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        request = DiscoveryRequest()
        request.sequence_id = self._next_sequence_id()

        devices = []
        received_ids = set()

        def on_response(msg: LBAPMessage, addr: Tuple[str, int]):
            if msg.message_type == MessageType.DISCOVERY_RESPONSE:
                device_id = getattr(msg, 'device_id', addr[0])
                if device_id not in received_ids:
                    received_ids.add(device_id)
                    devices.append({
                        "device_id": device_id,
                        "ip_address": addr[0],
                        "port": addr[1],
                        "device_type": getattr(msg, 'device_type', 'unknown'),
                        "firmware_version": getattr(msg, 'firmware_version', ''),
                        "mac_address": getattr(msg, 'mac_address', ''),
                    })

        # Temporarily set callback
        old_callback = self._protocol.on_message
        self._protocol.on_message = on_response

        try:
            # Send broadcast
            data = self.codec.encode(request)
            self._transport.sendto(data, (broadcast_ip, port))
            self.metrics.packets_sent += 1

            # Wait for responses
            await asyncio.sleep(timeout)

        finally:
            self._protocol.on_message = old_callback
            # Disable broadcast
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 0)

        return devices

    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics."""
        return self.metrics.to_dict()

    def reset_metrics(self) -> None:
        """Reset client metrics."""
        self.metrics.reset()


async def quick_ping(
    device_ip: str,
    device_port: int = 5000,
    timeout: float = 2.0
) -> Tuple[bool, float]:
    """
    Quick connectivity test without creating persistent client.

    Args:
        device_ip: Target device IP
        device_port: Target port
        timeout: Request timeout

    Returns:
        Tuple of (success, rtt_ms)
    """
    config = UDPClientConfig(
        device_ip=device_ip,
        device_port=device_port,
        request_timeout_seconds=timeout
    )
    client = LBAPUDPClient(config)

    try:
        await client.connect()
        return await client.ping()
    finally:
        await client.close()
