"""
Libiao RCS (Robot Control System) Integration Module

Provides HTTP client, data schemas, polling, and telemetry recording
for interfacing with Libiao RCS API.
"""

from .schemas import (
    RobotState,
    Coordinate,
    RobotInfo,
    RobotListResponse,
    SetRobotRequest,
    SetRobotResponse,
    APInfo,
    ChannelInfo,
    FleetSnapshot,
    SwitchCommand,
)

from .auth import (
    AuthProvider,
    NoAuth,
    BearerTokenAuth,
    ApiKeyAuth,
    CookieAuth,
    BasicAuth,
    CompositeAuth,
    AuthFactory,
)

from .client import (
    RCSClientError,
    RCSConnectionError,
    RCSAuthError,
    RCSAPIError,
    RCSClientConfig,
    RCSClient,
    SyncRCSClient,
)

from .poller import (
    PollerState,
    PollerConfig,
    RobotChangeEvent,
    RCSPoller,
)

from .recorder import (
    TelemetryRecord,
    RecordingBackend,
    JSONLBackend,
    MemoryBackend,
    RecorderConfig,
    TelemetryRecorder,
    load_recordings,
)

__all__ = [
    # Schemas
    "RobotState",
    "Coordinate",
    "RobotInfo",
    "RobotListResponse",
    "SetRobotRequest",
    "SetRobotResponse",
    "APInfo",
    "ChannelInfo",
    "FleetSnapshot",
    "SwitchCommand",
    # Auth
    "AuthProvider",
    "NoAuth",
    "BearerTokenAuth",
    "ApiKeyAuth",
    "CookieAuth",
    "BasicAuth",
    "CompositeAuth",
    "AuthFactory",
    # Client
    "RCSClientError",
    "RCSConnectionError",
    "RCSAuthError",
    "RCSAPIError",
    "RCSClientConfig",
    "RCSClient",
    "SyncRCSClient",
    # Poller
    "PollerState",
    "PollerConfig",
    "RobotChangeEvent",
    "RCSPoller",
    # Recorder
    "TelemetryRecord",
    "RecordingBackend",
    "JSONLBackend",
    "MemoryBackend",
    "RecorderConfig",
    "TelemetryRecorder",
    "load_recordings",
]
