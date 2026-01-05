"""
LBAP UDP Message Codecs

Pluggable message encoder/decoder for LBAP-102LU-900 UDP communication.
Implements abstraction layer for different protocol versions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import struct
import time


class MessageType(Enum):
    """LBAP message types."""
    # Discovery/Connectivity
    ECHO_REQUEST = 0x01
    ECHO_RESPONSE = 0x02
    DISCOVERY_REQUEST = 0x03
    DISCOVERY_RESPONSE = 0x04

    # Status/Health
    STATUS_REQUEST = 0x10
    STATUS_RESPONSE = 0x11
    HEALTH_REPORT = 0x12

    # Configuration
    CONFIG_GET = 0x20
    CONFIG_SET = 0x21
    CONFIG_ACK = 0x22

    # Control
    CHANNEL_SET = 0x30
    POWER_SET = 0x31
    RESET_CMD = 0x32

    # Robot association
    ROBOT_ASSOCIATE = 0x40
    ROBOT_DISASSOCIATE = 0x41
    ROBOT_LIST = 0x42

    # Unknown/Error
    ERROR = 0xFF
    UNKNOWN = 0x00


@dataclass
class LBAPMessage:
    """Base LBAP message structure."""
    message_type: MessageType
    sequence_id: int = 0
    timestamp: float = field(default_factory=time.time)
    payload: bytes = b""

    # Response fields (filled on receive)
    source_ip: Optional[str] = None
    source_port: Optional[int] = None
    rtt_ms: Optional[float] = None


@dataclass
class EchoRequest(LBAPMessage):
    """Echo/loopback request for connectivity testing."""
    message_type: MessageType = field(default=MessageType.ECHO_REQUEST, init=False)
    echo_data: bytes = b"PING"


@dataclass
class EchoResponse(LBAPMessage):
    """Echo response."""
    message_type: MessageType = field(default=MessageType.ECHO_RESPONSE, init=False)
    echo_data: bytes = b""


@dataclass
class DiscoveryRequest(LBAPMessage):
    """Device discovery broadcast."""
    message_type: MessageType = field(default=MessageType.DISCOVERY_REQUEST, init=False)


@dataclass
class DiscoveryResponse(LBAPMessage):
    """Device discovery response."""
    message_type: MessageType = field(default=MessageType.DISCOVERY_RESPONSE, init=False)
    device_id: str = ""
    device_type: str = "LBAP-102LU-900"
    firmware_version: str = ""
    mac_address: str = ""
    ip_address: str = ""


@dataclass
class StatusRequest(LBAPMessage):
    """Request device status."""
    message_type: MessageType = field(default=MessageType.STATUS_REQUEST, init=False)


@dataclass
class StatusResponse(LBAPMessage):
    """Device status response."""
    message_type: MessageType = field(default=MessageType.STATUS_RESPONSE, init=False)
    uptime_seconds: int = 0
    module1_active: bool = True
    module2_active: bool = True
    current_channel: int = 0
    tx_power_dbm: int = 14
    robot_count: int = 0
    error_count: int = 0
    queue_depth: int = 0


@dataclass
class HealthReport(LBAPMessage):
    """Periodic health report from device."""
    message_type: MessageType = field(default=MessageType.HEALTH_REPORT, init=False)
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    temperature_c: float = 0.0
    packet_loss_rate: float = 0.0
    avg_latency_ms: float = 0.0


@dataclass
class ErrorMessage(LBAPMessage):
    """Error response."""
    message_type: MessageType = field(default=MessageType.ERROR, init=False)
    error_code: int = 0
    error_message: str = ""


class MessageCodec(ABC):
    """Abstract base for LBAP message codecs."""

    @abstractmethod
    def encode(self, message: LBAPMessage) -> bytes:
        """Encode message to bytes for transmission."""
        pass

    @abstractmethod
    def decode(self, data: bytes) -> LBAPMessage:
        """Decode bytes to message."""
        pass

    @abstractmethod
    def get_version(self) -> str:
        """Get codec version identifier."""
        pass


class CodecV0(MessageCodec):
    """
    V0 Codec: Simple echo/loopback + connectivity test.

    Frame format:
    [1B: Type][2B: SeqID][4B: Timestamp][NB: Payload]

    Used for initial connectivity testing before real protocol is available.
    """

    HEADER_SIZE = 7

    def get_version(self) -> str:
        return "v0"

    def encode(self, message: LBAPMessage) -> bytes:
        """Encode message to bytes."""
        # Header: type (1B) + seq_id (2B) + timestamp (4B)
        header = struct.pack(
            ">BHI",
            message.message_type.value,
            message.sequence_id & 0xFFFF,
            int(message.timestamp * 1000) & 0xFFFFFFFF
        )

        # Payload based on message type
        if isinstance(message, EchoRequest):
            payload = message.echo_data
        elif isinstance(message, EchoResponse):
            payload = message.echo_data
        elif isinstance(message, DiscoveryRequest):
            payload = b"DISCOVER"
        else:
            payload = message.payload

        return header + payload

    def decode(self, data: bytes) -> LBAPMessage:
        """Decode bytes to message."""
        if len(data) < self.HEADER_SIZE:
            return ErrorMessage(error_code=1, error_message="Message too short")

        # Parse header
        msg_type_byte, seq_id, ts_ms = struct.unpack(">BHI", data[:self.HEADER_SIZE])
        payload = data[self.HEADER_SIZE:]

        try:
            msg_type = MessageType(msg_type_byte)
        except ValueError:
            msg_type = MessageType.UNKNOWN

        # Create appropriate message type
        if msg_type == MessageType.ECHO_RESPONSE:
            return EchoResponse(
                sequence_id=seq_id,
                timestamp=ts_ms / 1000.0,
                echo_data=payload
            )
        elif msg_type == MessageType.DISCOVERY_RESPONSE:
            return self._decode_discovery_response(seq_id, ts_ms, payload)
        elif msg_type == MessageType.STATUS_RESPONSE:
            return self._decode_status_response(seq_id, ts_ms, payload)
        elif msg_type == MessageType.ERROR:
            return ErrorMessage(
                sequence_id=seq_id,
                error_code=payload[0] if payload else 0,
                error_message=payload[1:].decode('utf-8', errors='ignore') if len(payload) > 1 else ""
            )
        else:
            return LBAPMessage(
                message_type=msg_type,
                sequence_id=seq_id,
                timestamp=ts_ms / 1000.0,
                payload=payload
            )

    def _decode_discovery_response(self, seq_id: int, ts_ms: int, payload: bytes) -> DiscoveryResponse:
        """Decode discovery response payload."""
        # Expected format: device_id|device_type|firmware|mac|ip
        try:
            parts = payload.decode('utf-8').split('|')
            return DiscoveryResponse(
                sequence_id=seq_id,
                timestamp=ts_ms / 1000.0,
                device_id=parts[0] if len(parts) > 0 else "",
                device_type=parts[1] if len(parts) > 1 else "",
                firmware_version=parts[2] if len(parts) > 2 else "",
                mac_address=parts[3] if len(parts) > 3 else "",
                ip_address=parts[4] if len(parts) > 4 else "",
            )
        except Exception:
            return DiscoveryResponse(sequence_id=seq_id)

    def _decode_status_response(self, seq_id: int, ts_ms: int, payload: bytes) -> StatusResponse:
        """Decode status response payload."""
        # Binary format: uptime(4B) + flags(1B) + channel(1B) + power(1B) + robots(2B) + errors(2B) + queue(2B)
        if len(payload) >= 13:
            uptime, flags, channel, power, robots, errors, queue = struct.unpack(
                ">IBBBHHH", payload[:13]
            )
            return StatusResponse(
                sequence_id=seq_id,
                timestamp=ts_ms / 1000.0,
                uptime_seconds=uptime,
                module1_active=bool(flags & 0x01),
                module2_active=bool(flags & 0x02),
                current_channel=channel,
                tx_power_dbm=power,
                robot_count=robots,
                error_count=errors,
                queue_depth=queue,
            )
        return StatusResponse(sequence_id=seq_id)


class CodecV1(MessageCodec):
    """
    V1 Codec: Placeholder for real Libiao command messages.

    Extended frame format with CRC and message priority:
    [1B: Magic][1B: Version][1B: Type][1B: Priority][2B: SeqID][4B: Timestamp][2B: PayloadLen][NB: Payload][2B: CRC16]

    To be implemented when Libiao provides actual protocol documentation.
    """

    MAGIC = 0x4C42  # "LB" in ASCII
    VERSION = 0x01
    HEADER_SIZE = 12
    CRC_SIZE = 2

    def get_version(self) -> str:
        return "v1"

    def encode(self, message: LBAPMessage) -> bytes:
        """Encode message with extended header and CRC."""
        payload = self._encode_payload(message)
        payload_len = len(payload)

        # Header
        header = struct.pack(
            ">BBBBHIH",
            0xAB,  # Magic byte (placeholder)
            self.VERSION,
            message.message_type.value,
            0,  # Priority (default)
            message.sequence_id & 0xFFFF,
            int(message.timestamp * 1000) & 0xFFFFFFFF,
            payload_len
        )

        # CRC16 over header + payload
        frame = header + payload
        crc = self._crc16(frame)

        return frame + struct.pack(">H", crc)

    def decode(self, data: bytes) -> LBAPMessage:
        """Decode message with CRC verification."""
        min_size = self.HEADER_SIZE + self.CRC_SIZE
        if len(data) < min_size:
            return ErrorMessage(error_code=1, error_message="Message too short")

        # Parse header
        magic, version, msg_type_byte, priority, seq_id, ts_ms, payload_len = struct.unpack(
            ">BBBBHIH", data[:self.HEADER_SIZE]
        )

        # Verify magic and version
        if magic != 0xAB:
            return ErrorMessage(error_code=2, error_message="Invalid magic byte")
        if version != self.VERSION:
            return ErrorMessage(error_code=3, error_message=f"Unsupported version: {version}")

        # Extract payload and CRC
        payload = data[self.HEADER_SIZE:self.HEADER_SIZE + payload_len]
        received_crc = struct.unpack(">H", data[-self.CRC_SIZE:])[0]

        # Verify CRC
        computed_crc = self._crc16(data[:-self.CRC_SIZE])
        if computed_crc != received_crc:
            return ErrorMessage(error_code=4, error_message="CRC mismatch")

        try:
            msg_type = MessageType(msg_type_byte)
        except ValueError:
            msg_type = MessageType.UNKNOWN

        return self._decode_payload(msg_type, seq_id, ts_ms, payload)

    def _encode_payload(self, message: LBAPMessage) -> bytes:
        """Encode message-specific payload."""
        if isinstance(message, EchoRequest):
            return message.echo_data
        elif isinstance(message, StatusRequest):
            return b""
        # Add more message types as protocol is documented
        return message.payload

    def _decode_payload(
        self,
        msg_type: MessageType,
        seq_id: int,
        ts_ms: int,
        payload: bytes
    ) -> LBAPMessage:
        """Decode message-specific payload."""
        if msg_type == MessageType.ECHO_RESPONSE:
            return EchoResponse(sequence_id=seq_id, timestamp=ts_ms/1000.0, echo_data=payload)
        elif msg_type == MessageType.STATUS_RESPONSE:
            return CodecV0()._decode_status_response(seq_id, ts_ms, payload)
        # Add more as needed
        return LBAPMessage(
            message_type=msg_type,
            sequence_id=seq_id,
            timestamp=ts_ms/1000.0,
            payload=payload
        )

    def _crc16(self, data: bytes) -> int:
        """Calculate CRC16-CCITT."""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ 0x1021
                else:
                    crc <<= 1
                crc &= 0xFFFF
        return crc


class CodecRegistry:
    """Registry of available message codecs."""

    _codecs: Dict[str, MessageCodec] = {}

    @classmethod
    def register(cls, codec: MessageCodec) -> None:
        """Register a codec."""
        cls._codecs[codec.get_version()] = codec

    @classmethod
    def get(cls, version: str) -> Optional[MessageCodec]:
        """Get codec by version."""
        return cls._codecs.get(version)

    @classmethod
    def list_versions(cls) -> List[str]:
        """List registered codec versions."""
        return list(cls._codecs.keys())

    @classmethod
    def default(cls) -> MessageCodec:
        """Get default codec (v0)."""
        if "v0" not in cls._codecs:
            cls._codecs["v0"] = CodecV0()
        return cls._codecs["v0"]


# Register default codecs
CodecRegistry.register(CodecV0())
CodecRegistry.register(CodecV1())
