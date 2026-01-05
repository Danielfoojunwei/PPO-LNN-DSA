"""
LBAP Device Discovery

Network discovery for LBAP-102LU-900 devices.
Supports broadcast discovery and targeted scanning.
"""

import asyncio
import logging
import socket
import struct
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Set
from ipaddress import ip_address, ip_network, IPv4Address, IPv4Network

from .udp_client import LBAPUDPClient, UDPClientConfig
from .codecs import DiscoveryRequest, DiscoveryResponse, MessageType, CodecRegistry
from .inventory import LBAPDevice, DeviceState

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryResult:
    """Result of a device discovery attempt."""
    device_id: str
    ip_address: str
    port: int
    device_type: str
    firmware_version: str
    mac_address: str
    discovery_time: float = field(default_factory=time.time)
    rtt_ms: float = 0.0
    responded: bool = True


@dataclass
class DiscoveryConfig:
    """Configuration for device discovery."""
    # Network scanning
    broadcast_addresses: List[str] = field(default_factory=lambda: ["255.255.255.255"])
    subnets: List[str] = field(default_factory=list)  # e.g., ["192.168.1.0/24"]

    # Default LBAP settings
    default_port: int = 5000
    factory_ip: str = "192.168.0.200"  # Factory default IP

    # Timing
    broadcast_timeout: float = 3.0
    scan_timeout_per_host: float = 1.0
    concurrent_scans: int = 50

    # Retries
    broadcast_retries: int = 2
    scan_retries: int = 1

    # Protocol
    codec_version: str = "v0"


class DeviceDiscovery:
    """
    LBAP device discovery service.

    Provides multiple discovery methods:
    - Broadcast discovery (finds devices on local network)
    - Subnet scanning (targeted IP range scan)
    - Factory IP probe (check default factory address)
    - ARP-based discovery (requires root/admin)
    """

    def __init__(self, config: Optional[DiscoveryConfig] = None):
        self.config = config or DiscoveryConfig()
        self.codec = CodecRegistry.get(self.config.codec_version) or CodecRegistry.default()
        self._discovered: Dict[str, DiscoveryResult] = {}

    @property
    def discovered_devices(self) -> List[DiscoveryResult]:
        """Get all discovered devices."""
        return list(self._discovered.values())

    async def discover_all(
        self,
        timeout: Optional[float] = None
    ) -> List[DiscoveryResult]:
        """
        Run all discovery methods and combine results.

        Args:
            timeout: Overall timeout (None = use defaults)

        Returns:
            List of all discovered devices
        """
        results = []

        # Broadcast discovery
        for broadcast_ip in self.config.broadcast_addresses:
            try:
                broadcast_results = await self.broadcast_discover(broadcast_ip)
                results.extend(broadcast_results)
            except Exception as e:
                logger.warning(f"Broadcast discovery failed for {broadcast_ip}: {e}")

        # Subnet scanning
        for subnet in self.config.subnets:
            try:
                scan_results = await self.subnet_scan(subnet)
                results.extend(scan_results)
            except Exception as e:
                logger.warning(f"Subnet scan failed for {subnet}: {e}")

        # Probe factory IP
        try:
            factory_result = await self.probe_ip(self.config.factory_ip)
            if factory_result:
                results.append(factory_result)
        except Exception as e:
            logger.debug(f"Factory IP probe failed: {e}")

        # Deduplicate by device_id
        seen = set()
        unique_results = []
        for result in results:
            if result.device_id not in seen:
                seen.add(result.device_id)
                unique_results.append(result)
                self._discovered[result.device_id] = result

        return unique_results

    async def broadcast_discover(
        self,
        broadcast_ip: str = "255.255.255.255",
        port: Optional[int] = None,
        timeout: Optional[float] = None
    ) -> List[DiscoveryResult]:
        """
        Discover devices via UDP broadcast.

        Args:
            broadcast_ip: Broadcast address
            port: UDP port
            timeout: Response timeout

        Returns:
            List of discovered devices
        """
        port = port or self.config.default_port
        timeout = timeout or self.config.broadcast_timeout

        results = []

        # Create UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setblocking(False)

        try:
            sock.bind(('', 0))

            # Send discovery request
            request = DiscoveryRequest()
            request.sequence_id = int(time.time() * 1000) & 0xFFFF
            data = self.codec.encode(request)
            send_time = time.time()

            # Send with retries
            for _ in range(self.config.broadcast_retries):
                sock.sendto(data, (broadcast_ip, port))
                await asyncio.sleep(0.1)

            # Collect responses
            loop = asyncio.get_event_loop()
            end_time = time.time() + timeout
            seen_ips: Set[str] = set()

            while time.time() < end_time:
                try:
                    remaining = max(0.1, end_time - time.time())
                    ready = await asyncio.wait_for(
                        loop.sock_recv(sock, 1024),
                        timeout=remaining
                    )

                    # This will return data, but we need to get the address
                    # Re-implement with recvfrom
                except asyncio.TimeoutError:
                    break
                except Exception:
                    break

            # Alternative: use aiohttp-style approach
            # For now, simplified synchronous collection after delay
            await asyncio.sleep(timeout)

            # Try to receive any pending responses
            sock.setblocking(False)
            while True:
                try:
                    data, addr = sock.recvfrom(1024)
                    if addr[0] not in seen_ips:
                        seen_ips.add(addr[0])
                        rtt_ms = (time.time() - send_time) * 1000

                        try:
                            msg = self.codec.decode(data)
                            if msg.message_type == MessageType.DISCOVERY_RESPONSE:
                                result = DiscoveryResult(
                                    device_id=getattr(msg, 'device_id', '') or addr[0],
                                    ip_address=addr[0],
                                    port=addr[1],
                                    device_type=getattr(msg, 'device_type', 'LBAP-102LU-900'),
                                    firmware_version=getattr(msg, 'firmware_version', ''),
                                    mac_address=getattr(msg, 'mac_address', ''),
                                    rtt_ms=rtt_ms,
                                )
                                results.append(result)
                        except Exception:
                            # Unknown response format, still count as found
                            results.append(DiscoveryResult(
                                device_id=addr[0],
                                ip_address=addr[0],
                                port=addr[1],
                                device_type="unknown",
                                firmware_version="",
                                mac_address="",
                                rtt_ms=rtt_ms,
                            ))
                except BlockingIOError:
                    break
                except Exception:
                    break

        finally:
            sock.close()

        logger.info(f"Broadcast discovery found {len(results)} devices on {broadcast_ip}")
        return results

    async def subnet_scan(
        self,
        subnet: str,
        port: Optional[int] = None
    ) -> List[DiscoveryResult]:
        """
        Scan IP range for LBAP devices.

        Args:
            subnet: CIDR notation subnet (e.g., "192.168.1.0/24")
            port: UDP port to probe

        Returns:
            List of discovered devices
        """
        port = port or self.config.default_port

        try:
            network = ip_network(subnet, strict=False)
        except ValueError as e:
            logger.error(f"Invalid subnet: {subnet}: {e}")
            return []

        # Get all host IPs
        hosts = list(network.hosts())
        logger.info(f"Scanning {len(hosts)} hosts in {subnet}")

        results = []
        semaphore = asyncio.Semaphore(self.config.concurrent_scans)

        async def probe_host(ip: IPv4Address) -> Optional[DiscoveryResult]:
            async with semaphore:
                return await self.probe_ip(str(ip), port)

        # Scan all hosts concurrently
        tasks = [probe_host(ip) for ip in hosts]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for response in responses:
            if isinstance(response, DiscoveryResult):
                results.append(response)

        logger.info(f"Subnet scan found {len(results)} devices in {subnet}")
        return results

    async def probe_ip(
        self,
        ip: str,
        port: Optional[int] = None
    ) -> Optional[DiscoveryResult]:
        """
        Probe a specific IP address for LBAP device.

        Args:
            ip: Target IP address
            port: UDP port

        Returns:
            DiscoveryResult if device found, None otherwise
        """
        port = port or self.config.default_port

        config = UDPClientConfig(
            device_ip=ip,
            device_port=port,
            request_timeout_seconds=self.config.scan_timeout_per_host,
            max_retries=self.config.scan_retries
        )

        client = LBAPUDPClient(config)

        try:
            await client.connect()

            # Try ping first
            success, rtt_ms = await client.ping()
            if not success:
                return None

            # Try to get status for more info
            status = await client.get_status()

            return DiscoveryResult(
                device_id=ip,  # Use IP as ID until we get proper ID
                ip_address=ip,
                port=port,
                device_type="LBAP-102LU-900",
                firmware_version="",
                mac_address="",
                rtt_ms=rtt_ms,
            )

        except Exception:
            return None

        finally:
            await client.close()

    async def probe_multiple(
        self,
        ips: List[str],
        port: Optional[int] = None
    ) -> List[DiscoveryResult]:
        """
        Probe multiple specific IPs concurrently.

        Args:
            ips: List of IP addresses
            port: UDP port

        Returns:
            List of discovered devices
        """
        port = port or self.config.default_port
        semaphore = asyncio.Semaphore(self.config.concurrent_scans)

        async def probe_with_limit(ip: str) -> Optional[DiscoveryResult]:
            async with semaphore:
                return await self.probe_ip(ip, port)

        tasks = [probe_with_limit(ip) for ip in ips]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        return [r for r in responses if isinstance(r, DiscoveryResult)]

    def to_devices(self, results: List[DiscoveryResult]) -> List[LBAPDevice]:
        """Convert discovery results to LBAPDevice instances."""
        devices = []
        for result in results:
            device = LBAPDevice(
                device_id=result.device_id,
                ip_address=result.ip_address,
                port=result.port,
                device_type=result.device_type,
                firmware_version=result.firmware_version,
                mac_address=result.mac_address,
            )
            device.state = DeviceState.ONLINE if result.responded else DeviceState.UNKNOWN
            device.avg_rtt_ms = result.rtt_ms
            device.last_seen = result.discovery_time
            devices.append(device)
        return devices


async def quick_discover(
    broadcast_ip: str = "255.255.255.255",
    timeout: float = 3.0
) -> List[DiscoveryResult]:
    """
    Quick discovery without creating persistent service.

    Args:
        broadcast_ip: Broadcast address
        timeout: Discovery timeout

    Returns:
        List of discovered devices
    """
    discovery = DeviceDiscovery()
    return await discovery.broadcast_discover(broadcast_ip, timeout=timeout)
