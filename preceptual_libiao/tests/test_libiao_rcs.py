"""
Tests for Libiao RCS client module.
"""

import asyncio
import json
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preceptual.libiao_rcs.schemas import (
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
from preceptual.libiao_rcs.auth import (
    NoAuth,
    BearerTokenAuth,
    ApiKeyAuth,
    CookieAuth,
    BasicAuth,
    AuthFactory,
)
from preceptual.libiao_rcs.client import (
    RCSClientConfig,
    RCSClient,
)
from preceptual.libiao_rcs.recorder import (
    TelemetryRecord,
    MemoryBackend,
    RecorderConfig,
    TelemetryRecorder,
)


class TestCoordinate:
    """Tests for Coordinate parsing and methods."""

    def test_from_string_simple(self):
        coord = Coordinate.from_string("1,2")
        assert coord.x == 1.0
        assert coord.y == 2.0
        assert coord.zone_id is None

    def test_from_string_with_zone(self):
        coord = Coordinate.from_string("1.5,2.5,zone_a")
        assert coord.x == 1.5
        assert coord.y == 2.5
        assert coord.zone_id == "zone_a"

    def test_from_string_empty(self):
        coord = Coordinate.from_string("")
        assert coord.x == 0.0
        assert coord.y == 0.0

    def test_from_string_invalid(self):
        coord = Coordinate.from_string("invalid")
        assert coord.x == 0.0
        assert coord.y == 0.0

    def test_distance_to(self):
        coord1 = Coordinate(0, 0)
        coord2 = Coordinate(3, 4)
        assert coord1.distance_to(coord2) == 5.0


class TestRobotInfo:
    """Tests for RobotInfo parsing."""

    def test_from_api_response(self):
        data = {
            "robotId": "62",
            "ap": "192.168.11.62",
            "channel": 86,
            "rssi": 94,
            "coordinate": "1,2"
        }
        robot = RobotInfo.from_api_response(data)

        assert robot.robot_id == "62"
        assert robot.ap_ip == "192.168.11.62"
        assert robot.channel == 86
        assert robot.rssi == 94
        assert robot.coordinate.x == 1.0
        assert robot.coordinate.y == 2.0

    def test_to_api_format(self):
        robot = RobotInfo(
            robot_id="1",
            ap_ip="192.168.1.1",
            channel=10,
            rssi=80,
            coordinate=Coordinate(5.0, 10.0)
        )
        data = robot.to_api_format()

        assert data["robotId"] == "1"
        assert data["ap"] == "192.168.1.1"
        assert data["channel"] == 10
        assert data["rssi"] == 80
        assert data["coordinate"] == "5.0,10.0"


class TestRobotListResponse:
    """Tests for RobotListResponse parsing."""

    def test_from_api_response(self):
        data = {
            "robotList": [
                {"robotId": "1", "ap": "192.168.1.1", "channel": 10, "rssi": 80, "coordinate": "1,1"},
                {"robotId": "2", "ap": "192.168.1.2", "channel": 20, "rssi": 90, "coordinate": "2,2"},
            ]
        }
        response = RobotListResponse.from_api_response(data)

        assert len(response.robot_list) == 2
        assert response.robot_list[0].robot_id == "1"
        assert response.robot_list[1].robot_id == "2"

    def test_get_robot(self):
        data = {
            "robotList": [
                {"robotId": "1", "ap": "192.168.1.1", "channel": 10, "rssi": 80, "coordinate": "1,1"},
            ]
        }
        response = RobotListResponse.from_api_response(data)

        robot = response.get_robot("1")
        assert robot is not None
        assert robot.robot_id == "1"

        robot = response.get_robot("999")
        assert robot is None


class TestFleetSnapshot:
    """Tests for FleetSnapshot aggregation."""

    def test_from_robot_list(self):
        data = {
            "robotList": [
                {"robotId": "1", "ap": "192.168.1.1", "channel": 10, "rssi": 80, "coordinate": "1,1"},
                {"robotId": "2", "ap": "192.168.1.1", "channel": 10, "rssi": 90, "coordinate": "1,2"},
                {"robotId": "3", "ap": "192.168.1.2", "channel": 20, "rssi": 85, "coordinate": "2,1"},
            ]
        }
        response = RobotListResponse.from_api_response(data)
        snapshot = FleetSnapshot.from_robot_list(response)

        assert snapshot.robot_count == 3
        assert snapshot.ap_count == 2
        assert snapshot.channel_count == 2

        # AP aggregation
        assert "192.168.1.1" in snapshot.aps
        assert snapshot.aps["192.168.1.1"].robot_count == 2
        assert 10 in snapshot.aps["192.168.1.1"].channels_in_use

        # Channel aggregation
        assert 10 in snapshot.channels
        assert snapshot.channels[10].robot_count == 2

    def test_hotspot_detection(self):
        robots = []
        for i in range(25):
            robots.append({
                "robotId": str(i),
                "ap": "192.168.1.1",
                "channel": 10,
                "rssi": 80,
                "coordinate": f"{i},0"
            })
        data = {"robotList": robots}
        response = RobotListResponse.from_api_response(data)
        snapshot = FleetSnapshot.from_robot_list(response)

        hotspots = snapshot.get_hotspot_aps(threshold=20)
        assert "192.168.1.1" in hotspots


class TestSwitchCommand:
    """Tests for SwitchCommand lifecycle."""

    def test_lifecycle(self):
        cmd = SwitchCommand(
            robot_id="1",
            target_ap_ip="192.168.1.1",
            target_channel=20,
            original_ap_ip="192.168.1.2",
            original_channel=10
        )

        assert cmd.is_pending is False
        assert cmd.success is None

        cmd.mark_sent()
        assert cmd.sent_at is not None
        assert cmd.is_pending is True

        cmd.mark_verified()
        assert cmd.verified_at is not None
        assert cmd.success is True
        assert cmd.is_pending is False

    def test_to_request(self):
        cmd = SwitchCommand(
            robot_id="1",
            target_ap_ip="192.168.1.1",
            target_channel=20
        )
        request = cmd.to_request()

        assert request.robot_id == "1"
        assert request.ap_ip == "192.168.1.1"
        assert request.channel == 20


class TestAuth:
    """Tests for authentication providers."""

    def test_no_auth(self):
        auth = NoAuth()
        assert auth.get_headers() == {}
        assert auth.get_params() == {}
        assert auth.get_cookies() == {}

    def test_bearer_token(self):
        auth = BearerTokenAuth(token="test_token")
        assert auth.get_headers() == {"Authorization": "Bearer test_token"}
        assert auth.is_expired() is False

    def test_bearer_token_expired(self):
        auth = BearerTokenAuth(token="test_token", expires_at=time.time() - 100)
        assert auth.is_expired() is True

    def test_api_key_header(self):
        auth = ApiKeyAuth(api_key="my_key", header_name="X-Custom-Key")
        assert auth.get_headers() == {"X-Custom-Key": "my_key"}
        assert auth.get_params() == {}

    def test_api_key_param(self):
        auth = ApiKeyAuth(api_key="my_key", use_header=False, param_name="key")
        assert auth.get_headers() == {}
        assert auth.get_params() == {"key": "my_key"}

    def test_cookie_auth(self):
        auth = CookieAuth(session_cookie="session_123", cookie_name="sid")
        assert auth.get_cookies() == {"sid": "session_123"}

    def test_basic_auth(self):
        auth = BasicAuth(username="user", password="pass")
        import base64
        expected = base64.b64encode(b"user:pass").decode()
        assert auth.get_headers() == {"Authorization": f"Basic {expected}"}

    def test_factory_from_config(self):
        config = {"type": "bearer", "token": "my_token"}
        auth = AuthFactory.from_config(config)
        assert isinstance(auth, BearerTokenAuth)
        assert auth.token == "my_token"


class TestSetRobotResponse:
    """Tests for SetRobotResponse."""

    def test_success(self):
        data = {"result": 0, "message": "OK"}
        response = SetRobotResponse.from_api_response(data)
        assert response.success is True
        assert response.result == 0

    def test_failure(self):
        data = {"result": 1, "message": "Robot not found"}
        response = SetRobotResponse.from_api_response(data)
        assert response.success is False


class TestMemoryBackend:
    """Tests for in-memory recording backend."""

    def test_write_and_query(self):
        backend = MemoryBackend(max_records=100)

        record1 = TelemetryRecord(timestamp=1.0, record_type="snapshot", data={"count": 10})
        record2 = TelemetryRecord(timestamp=2.0, record_type="robot", data={"id": "1"})

        backend.write(record1)
        backend.write(record2)

        all_records = backend.get_records()
        assert len(all_records) == 2

        snapshots = backend.get_records(record_type="snapshot")
        assert len(snapshots) == 1

    def test_max_records(self):
        backend = MemoryBackend(max_records=5)

        for i in range(10):
            record = TelemetryRecord(timestamp=float(i), record_type="test", data={})
            backend.write(record)

        assert len(backend.records) == 5
        assert backend.records[0].timestamp == 5.0  # Oldest records removed


class TestTelemetryRecord:
    """Tests for TelemetryRecord creation."""

    def test_from_snapshot(self):
        data = {
            "robotList": [
                {"robotId": "1", "ap": "192.168.1.1", "channel": 10, "rssi": 80, "coordinate": "1,1"},
            ]
        }
        response = RobotListResponse.from_api_response(data)
        snapshot = FleetSnapshot.from_robot_list(response)

        record = TelemetryRecord.from_snapshot(snapshot)
        assert record.record_type == "snapshot"
        assert record.data["robot_count"] == 1

    def test_from_switch(self):
        cmd = SwitchCommand(
            robot_id="1",
            target_ap_ip="192.168.1.1",
            target_channel=20
        )
        cmd.mark_sent()
        cmd.mark_verified()

        record = TelemetryRecord.from_switch(cmd)
        assert record.record_type == "switch"
        assert record.data["robot_id"] == "1"
        assert record.data["success"] is True

    def test_serialization(self):
        record = TelemetryRecord(timestamp=123.456, record_type="test", data={"key": "value"})
        d = record.to_dict()
        restored = TelemetryRecord.from_dict(d)

        assert restored.timestamp == record.timestamp
        assert restored.record_type == record.record_type
        assert restored.data == record.data


# Integration tests with mocked aiohttp
class TestRCSClientIntegration:
    """Integration tests for RCS client with mocked HTTP."""

    @pytest.fixture
    def client_config(self):
        return RCSClientConfig(
            base_url="http://localhost:8080",
            auth=NoAuth(),
            timeout_seconds=5.0
        )

    @pytest.mark.asyncio
    async def test_query_robots(self, client_config):
        mock_response = {
            "robotList": [
                {"robotId": "62", "ap": "192.168.11.62", "channel": 86, "rssi": 94, "coordinate": "1,2"}
            ]
        }

        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session

            mock_response_obj = AsyncMock()
            mock_response_obj.status = 200
            mock_response_obj.json = AsyncMock(return_value=mock_response)
            mock_response_obj.__aenter__ = AsyncMock(return_value=mock_response_obj)
            mock_response_obj.__aexit__ = AsyncMock(return_value=None)

            mock_session.request.return_value = mock_response_obj
            mock_session.closed = False

            client = RCSClient(client_config)
            client._session = mock_session

            response = await client.query_robots()

            assert len(response.robot_list) == 1
            assert response.robot_list[0].robot_id == "62"
            assert response.robot_list[0].rssi == 94

    @pytest.mark.asyncio
    async def test_set_robot(self, client_config):
        mock_response = {"result": 0, "message": "OK"}

        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session

            mock_response_obj = AsyncMock()
            mock_response_obj.status = 200
            mock_response_obj.json = AsyncMock(return_value=mock_response)
            mock_response_obj.__aenter__ = AsyncMock(return_value=mock_response_obj)
            mock_response_obj.__aexit__ = AsyncMock(return_value=None)

            mock_session.request.return_value = mock_response_obj
            mock_session.closed = False

            client = RCSClient(client_config)
            client._session = mock_session

            request = SetRobotRequest(robot_id="1", ap_ip="192.168.11.11", channel=41)
            response = await client.set_robot(request)

            assert response.success is True
            assert response.result == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
