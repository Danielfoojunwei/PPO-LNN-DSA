"""
LBAP Device Manager Module

Manages LBAP-102LU-900 wireless gateway devices for Libiao fleets.
Provides device inventory, discovery, health monitoring, and load estimation.
"""

from .radio_profile import (
    ModulationType,
    FrequencyBand,
    ChannelMapping,
    RadioProfile,
    WiFiRadioProfile,
    SiteRadioConfig,
)

from .codecs import (
    MessageType,
    LBAPMessage,
    EchoRequest,
    EchoResponse,
    DiscoveryRequest,
    DiscoveryResponse,
    StatusRequest,
    StatusResponse,
    HealthReport,
    ErrorMessage,
    MessageCodec,
    CodecV0,
    CodecV1,
    CodecRegistry,
)

from .udp_client import (
    UDPClientError,
    UDPConnectionError,
    UDPTimeoutError,
    UDPClientConfig,
    UDPClientMetrics,
    LBAPUDPProtocol,
    LBAPUDPClient,
    quick_ping,
)

from .inventory import (
    DeviceState,
    ModuleState,
    LBAPModule,
    LBAPDevice,
    InventoryConfig,
    LBAPInventory,
)

from .discovery import (
    DiscoveryResult,
    DiscoveryConfig,
    DeviceDiscovery,
    quick_discover,
)

from .health import (
    HealthLevel,
    AlertSeverity,
    HealthAlert,
    HealthSample,
    HealthHistory,
    HealthThresholds,
    DeviceHealthStatus,
    HealthMonitor,
)

from .load import (
    LoadLevel,
    LoadSample,
    LoadHistory,
    LoadThresholds,
    DeviceLoadStatus,
    ChannelLoadStatus,
    FleetLoadSummary,
    LoadEstimator,
)

__all__ = [
    # Radio Profile
    "ModulationType",
    "FrequencyBand",
    "ChannelMapping",
    "RadioProfile",
    "WiFiRadioProfile",
    "SiteRadioConfig",
    # Codecs
    "MessageType",
    "LBAPMessage",
    "EchoRequest",
    "EchoResponse",
    "DiscoveryRequest",
    "DiscoveryResponse",
    "StatusRequest",
    "StatusResponse",
    "HealthReport",
    "ErrorMessage",
    "MessageCodec",
    "CodecV0",
    "CodecV1",
    "CodecRegistry",
    # UDP Client
    "UDPClientError",
    "UDPConnectionError",
    "UDPTimeoutError",
    "UDPClientConfig",
    "UDPClientMetrics",
    "LBAPUDPProtocol",
    "LBAPUDPClient",
    "quick_ping",
    # Inventory
    "DeviceState",
    "ModuleState",
    "LBAPModule",
    "LBAPDevice",
    "InventoryConfig",
    "LBAPInventory",
    # Discovery
    "DiscoveryResult",
    "DiscoveryConfig",
    "DeviceDiscovery",
    "quick_discover",
    # Health
    "HealthLevel",
    "AlertSeverity",
    "HealthAlert",
    "HealthSample",
    "HealthHistory",
    "HealthThresholds",
    "DeviceHealthStatus",
    "HealthMonitor",
    # Load
    "LoadLevel",
    "LoadSample",
    "LoadHistory",
    "LoadThresholds",
    "DeviceLoadStatus",
    "ChannelLoadStatus",
    "FleetLoadSummary",
    "LoadEstimator",
]
