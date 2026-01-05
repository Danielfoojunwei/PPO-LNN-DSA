"""
Libiao Radio Profile Configuration

Defines radio configuration parameters for LBAP-102LU-900 devices
and site-level radio profiles based on AirRob control bin parameters.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import json


class ModulationType(Enum):
    """Supported modulation types."""
    GFSK = "gfsk"
    FSK = "fsk"
    LORA = "lora"


class FrequencyBand(Enum):
    """Frequency bands."""
    BAND_900MHZ = "900mhz"  # 902-928 MHz
    WIFI_24GHZ = "2.4ghz"
    WIFI_5GHZ = "5ghz"


@dataclass
class ChannelMapping:
    """
    Mapping from logical channel index to physical frequency.

    Based on LBAP-102LU-900 docs: frequency range ~904.25-926.25 MHz
    """
    channel_index: int
    frequency_mhz: float
    bandwidth_khz: float = 500.0

    def __hash__(self):
        return hash(self.channel_index)


@dataclass
class RadioProfile:
    """
    Site-level radio configuration profile.

    Captures parameters exposed by AirRob control bin:
    - Broadcast address
    - HOME channel
    - Fundamental frequency
    - Frequency interval
    - Transmit power range

    Used by:
    - Simulator channel model
    - Policy observation vector (site constants)
    - Airtime budget calculations
    """
    # Identity
    profile_id: str
    profile_name: str

    # Band configuration
    band: FrequencyBand = FrequencyBand.BAND_900MHZ
    modulation: ModulationType = ModulationType.GFSK

    # Frequency parameters (900MHz band)
    fundamental_frequency_mhz: float = 904.25
    frequency_interval_mhz: float = 0.5
    max_frequency_mhz: float = 926.25
    num_channels: int = 44  # (926.25 - 904.25) / 0.5

    # Channel configuration
    home_channel: int = 0
    broadcast_address: int = 0xFF  # Default broadcast

    # Power configuration (dBm)
    min_tx_power_dbm: int = 0
    max_tx_power_dbm: int = 20
    default_tx_power_dbm: int = 14

    # Timing parameters
    slot_duration_ms: float = 10.0
    guard_time_ms: float = 1.0
    beacon_interval_ms: float = 100.0

    # Capacity limits (for airtime budgeting)
    max_robots_per_channel: int = 15
    max_robots_per_device: int = 50
    airtime_budget_per_slot_us: int = 9000  # microseconds

    # Additional site constants
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_channel_frequency(self, channel_index: int) -> float:
        """Get frequency for a logical channel index."""
        freq = self.fundamental_frequency_mhz + (channel_index * self.frequency_interval_mhz)
        if freq > self.max_frequency_mhz:
            raise ValueError(f"Channel {channel_index} exceeds max frequency")
        return freq

    def get_channel_mappings(self) -> List[ChannelMapping]:
        """Generate all channel mappings for this profile."""
        mappings = []
        for i in range(self.num_channels):
            freq = self.get_channel_frequency(i)
            mappings.append(ChannelMapping(
                channel_index=i,
                frequency_mhz=freq,
                bandwidth_khz=500.0
            ))
        return mappings

    def is_valid_channel(self, channel_index: int) -> bool:
        """Check if channel index is valid for this profile."""
        return 0 <= channel_index < self.num_channels

    def is_valid_power(self, power_dbm: int) -> bool:
        """Check if power level is within valid range."""
        return self.min_tx_power_dbm <= power_dbm <= self.max_tx_power_dbm

    def to_observation_vector(self) -> List[float]:
        """
        Convert profile to observation vector for RL policy.

        Returns normalized values suitable for neural network input.
        """
        return [
            self.fundamental_frequency_mhz / 1000.0,  # Normalized to GHz
            self.frequency_interval_mhz,
            self.num_channels / 100.0,
            self.home_channel / 100.0,
            self.max_tx_power_dbm / 30.0,
            self.slot_duration_ms / 100.0,
            self.max_robots_per_channel / 50.0,
            self.max_robots_per_device / 100.0,
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "profile_id": self.profile_id,
            "profile_name": self.profile_name,
            "band": self.band.value,
            "modulation": self.modulation.value,
            "fundamental_frequency_mhz": self.fundamental_frequency_mhz,
            "frequency_interval_mhz": self.frequency_interval_mhz,
            "max_frequency_mhz": self.max_frequency_mhz,
            "num_channels": self.num_channels,
            "home_channel": self.home_channel,
            "broadcast_address": self.broadcast_address,
            "min_tx_power_dbm": self.min_tx_power_dbm,
            "max_tx_power_dbm": self.max_tx_power_dbm,
            "default_tx_power_dbm": self.default_tx_power_dbm,
            "slot_duration_ms": self.slot_duration_ms,
            "guard_time_ms": self.guard_time_ms,
            "beacon_interval_ms": self.beacon_interval_ms,
            "max_robots_per_channel": self.max_robots_per_channel,
            "max_robots_per_device": self.max_robots_per_device,
            "airtime_budget_per_slot_us": self.airtime_budget_per_slot_us,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RadioProfile":
        """Deserialize from dictionary."""
        return cls(
            profile_id=d["profile_id"],
            profile_name=d["profile_name"],
            band=FrequencyBand(d.get("band", "900mhz")),
            modulation=ModulationType(d.get("modulation", "gfsk")),
            fundamental_frequency_mhz=d.get("fundamental_frequency_mhz", 904.25),
            frequency_interval_mhz=d.get("frequency_interval_mhz", 0.5),
            max_frequency_mhz=d.get("max_frequency_mhz", 926.25),
            num_channels=d.get("num_channels", 44),
            home_channel=d.get("home_channel", 0),
            broadcast_address=d.get("broadcast_address", 0xFF),
            min_tx_power_dbm=d.get("min_tx_power_dbm", 0),
            max_tx_power_dbm=d.get("max_tx_power_dbm", 20),
            default_tx_power_dbm=d.get("default_tx_power_dbm", 14),
            slot_duration_ms=d.get("slot_duration_ms", 10.0),
            guard_time_ms=d.get("guard_time_ms", 1.0),
            beacon_interval_ms=d.get("beacon_interval_ms", 100.0),
            max_robots_per_channel=d.get("max_robots_per_channel", 15),
            max_robots_per_device=d.get("max_robots_per_device", 50),
            airtime_budget_per_slot_us=d.get("airtime_budget_per_slot_us", 9000),
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def default_900mhz(cls) -> "RadioProfile":
        """Create default 900MHz profile for LBAP-102LU-900."""
        return cls(
            profile_id="default_900",
            profile_name="Default LBAP-102LU-900 Profile",
            band=FrequencyBand.BAND_900MHZ,
            modulation=ModulationType.GFSK,
        )

    @classmethod
    def from_json_file(cls, path: str) -> "RadioProfile":
        """Load profile from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_json(self, path: str) -> None:
        """Save profile to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class WiFiRadioProfile:
    """
    WiFi radio profile for optional Wi-Fi links.

    LinkType B in multi-link topology.
    """
    profile_id: str
    profile_name: str

    # Band
    band: FrequencyBand = FrequencyBand.WIFI_24GHZ

    # 2.4GHz channels
    channels_24ghz: List[int] = field(default_factory=lambda: [1, 6, 11])

    # 5GHz channels (DFS and non-DFS)
    channels_5ghz: List[int] = field(default_factory=lambda: [36, 40, 44, 48, 149, 153, 157, 161])

    # Power
    max_tx_power_dbm: int = 23

    # Capacity
    max_robots_per_ap: int = 30

    def to_observation_vector(self) -> List[float]:
        """Convert to observation vector."""
        return [
            1.0 if self.band == FrequencyBand.WIFI_24GHZ else 0.0,
            len(self.channels_24ghz) / 14.0,
            len(self.channels_5ghz) / 24.0,
            self.max_tx_power_dbm / 30.0,
            self.max_robots_per_ap / 50.0,
        ]


@dataclass
class SiteRadioConfig:
    """
    Complete site radio configuration.

    Combines LBAP 900MHz and optional WiFi profiles.
    """
    site_id: str
    site_name: str

    # Primary link (LBAP 900MHz)
    lbap_profile: RadioProfile

    # Secondary link (optional WiFi)
    wifi_profile: Optional[WiFiRadioProfile] = None

    # Multi-link preferences
    prefer_lbap: bool = True  # Prefer 900MHz for control traffic
    wifi_for_bulk: bool = True  # Use WiFi for bulk data if available

    def to_dict(self) -> Dict[str, Any]:
        return {
            "site_id": self.site_id,
            "site_name": self.site_name,
            "lbap_profile": self.lbap_profile.to_dict(),
            "wifi_profile": self.wifi_profile.to_observation_vector() if self.wifi_profile else None,
            "prefer_lbap": self.prefer_lbap,
            "wifi_for_bulk": self.wifi_for_bulk,
        }

    @classmethod
    def default(cls, site_id: str = "default", site_name: str = "Default Site") -> "SiteRadioConfig":
        """Create default site configuration."""
        return cls(
            site_id=site_id,
            site_name=site_name,
            lbap_profile=RadioProfile.default_900mhz(),
        )
