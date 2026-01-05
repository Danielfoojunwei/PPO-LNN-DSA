"""
Libiao RCS Telemetry Recorder

Records robot state telemetry for training and analysis.
Supports multiple storage backends.
"""

import asyncio
import gzip
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterator
from datetime import datetime
import threading
from queue import Queue, Empty

from .schemas import RobotInfo, FleetSnapshot, SwitchCommand

logger = logging.getLogger(__name__)


@dataclass
class TelemetryRecord:
    """Single telemetry record for training."""
    timestamp: float
    record_type: str  # "snapshot", "robot", "switch", "event"
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "record_type": self.record_type,
            "data": self.data
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TelemetryRecord":
        return cls(
            timestamp=d["timestamp"],
            record_type=d["record_type"],
            data=d["data"]
        )

    @classmethod
    def from_snapshot(cls, snapshot: FleetSnapshot) -> "TelemetryRecord":
        """Create record from fleet snapshot."""
        robots_data = [r.to_api_format() for r in snapshot.robots]
        return cls(
            timestamp=snapshot.timestamp,
            record_type="snapshot",
            data={
                "robot_count": snapshot.robot_count,
                "ap_count": snapshot.ap_count,
                "channel_count": snapshot.channel_count,
                "robots": robots_data,
                "ap_loads": {
                    ip: {"count": info.robot_count, "avg_rssi": info.avg_rssi}
                    for ip, info in snapshot.aps.items()
                },
                "channel_loads": {
                    str(ch): {"count": info.robot_count}
                    for ch, info in snapshot.channels.items()
                }
            }
        )

    @classmethod
    def from_robot(cls, robot: RobotInfo) -> "TelemetryRecord":
        """Create record from robot info."""
        return cls(
            timestamp=robot.timestamp,
            record_type="robot",
            data=robot.to_api_format()
        )

    @classmethod
    def from_switch(cls, switch: SwitchCommand) -> "TelemetryRecord":
        """Create record from switch command."""
        return cls(
            timestamp=switch.created_at,
            record_type="switch",
            data={
                "robot_id": switch.robot_id,
                "target_ap": switch.target_ap_ip,
                "target_channel": switch.target_channel,
                "original_ap": switch.original_ap_ip,
                "original_channel": switch.original_channel,
                "success": switch.success,
                "latency_ms": switch.latency_ms,
                "retry_count": switch.retry_count,
                "error": switch.error_message
            }
        )


class RecordingBackend(ABC):
    """Abstract base for telemetry storage backends."""

    @abstractmethod
    def write(self, record: TelemetryRecord) -> None:
        """Write a single record."""
        pass

    @abstractmethod
    def write_batch(self, records: List[TelemetryRecord]) -> None:
        """Write multiple records."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the backend."""
        pass


class JSONLBackend(RecordingBackend):
    """
    JSONL file backend.

    Writes records as newline-delimited JSON, optionally compressed.
    """

    def __init__(
        self,
        base_path: str,
        compress: bool = True,
        rotate_size_mb: float = 100.0,
        max_files: int = 10
    ):
        self.base_path = Path(base_path)
        self.compress = compress
        self.rotate_size_bytes = int(rotate_size_mb * 1024 * 1024)
        self.max_files = max_files

        self._current_file: Optional[Any] = None
        self._current_path: Optional[Path] = None
        self._current_size = 0
        self._file_count = 0
        self._lock = threading.Lock()

        # Create directory
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _open_new_file(self) -> None:
        """Open a new recording file."""
        if self._current_file:
            self._current_file.close()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = ".jsonl.gz" if self.compress else ".jsonl"
        filename = f"telemetry_{timestamp}_{self._file_count}{suffix}"
        self._current_path = self.base_path / filename
        self._file_count += 1

        if self.compress:
            self._current_file = gzip.open(self._current_path, "wt", encoding="utf-8")
        else:
            self._current_file = open(self._current_path, "w", encoding="utf-8")

        self._current_size = 0
        logger.info(f"Recording to {self._current_path}")

        # Cleanup old files
        self._cleanup_old_files()

    def _cleanup_old_files(self) -> None:
        """Remove oldest files if exceeding max_files."""
        files = sorted(self.base_path.glob("telemetry_*.jsonl*"))
        while len(files) > self.max_files:
            old_file = files.pop(0)
            try:
                old_file.unlink()
                logger.info(f"Cleaned up old recording: {old_file}")
            except Exception as e:
                logger.warning(f"Failed to remove old file {old_file}: {e}")

    def write(self, record: TelemetryRecord) -> None:
        """Write a single record."""
        with self._lock:
            if self._current_file is None or self._current_size >= self.rotate_size_bytes:
                self._open_new_file()

            line = json.dumps(record.to_dict()) + "\n"
            self._current_file.write(line)
            self._current_size += len(line.encode("utf-8"))

    def write_batch(self, records: List[TelemetryRecord]) -> None:
        """Write multiple records."""
        for record in records:
            self.write(record)

    def flush(self) -> None:
        """Flush current file."""
        with self._lock:
            if self._current_file:
                self._current_file.flush()

    def close(self) -> None:
        """Close current file."""
        with self._lock:
            if self._current_file:
                self._current_file.close()
                self._current_file = None


class MemoryBackend(RecordingBackend):
    """
    In-memory backend for testing.

    Stores records in a list with optional size limit.
    """

    def __init__(self, max_records: int = 100000):
        self.max_records = max_records
        self.records: List[TelemetryRecord] = []
        self._lock = threading.Lock()

    def write(self, record: TelemetryRecord) -> None:
        with self._lock:
            self.records.append(record)
            if len(self.records) > self.max_records:
                self.records.pop(0)

    def write_batch(self, records: List[TelemetryRecord]) -> None:
        with self._lock:
            self.records.extend(records)
            while len(self.records) > self.max_records:
                self.records.pop(0)

    def close(self) -> None:
        pass

    def get_records(
        self,
        record_type: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[TelemetryRecord]:
        """Query records with filters."""
        with self._lock:
            results = self.records.copy()

        if record_type:
            results = [r for r in results if r.record_type == record_type]
        if start_time:
            results = [r for r in results if r.timestamp >= start_time]
        if end_time:
            results = [r for r in results if r.timestamp <= end_time]

        return results


@dataclass
class RecorderConfig:
    """Configuration for telemetry recorder."""
    backend_type: str = "jsonl"  # "jsonl" or "memory"
    base_path: str = "./telemetry"
    compress: bool = True
    rotate_size_mb: float = 100.0
    max_files: int = 10

    # Batching
    batch_size: int = 100
    batch_interval_seconds: float = 1.0

    # Filters
    record_snapshots: bool = True
    record_robots: bool = False  # Per-robot records (verbose)
    record_switches: bool = True
    record_events: bool = True

    # Sampling
    snapshot_sample_rate: float = 1.0  # 1.0 = record all
    robot_sample_rate: float = 0.1  # Sample 10% of robot records


class TelemetryRecorder:
    """
    Asynchronous telemetry recorder.

    Batches records and writes to backend in background thread.
    """

    def __init__(self, config: Optional[RecorderConfig] = None):
        self.config = config or RecorderConfig()

        # Create backend
        if self.config.backend_type == "jsonl":
            self._backend = JSONLBackend(
                base_path=self.config.base_path,
                compress=self.config.compress,
                rotate_size_mb=self.config.rotate_size_mb,
                max_files=self.config.max_files
            )
        elif self.config.backend_type == "memory":
            self._backend = MemoryBackend()
        else:
            raise ValueError(f"Unknown backend type: {self.config.backend_type}")

        # Batching
        self._batch_queue: Queue = Queue()
        self._batch: List[TelemetryRecord] = []
        self._last_flush = time.time()

        # Background writer
        self._running = False
        self._writer_thread: Optional[threading.Thread] = None

        # Metrics
        self._record_count = 0
        self._batch_count = 0

    def start(self) -> None:
        """Start background writer thread."""
        if self._running:
            return

        self._running = True
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()
        logger.info("Telemetry recorder started")

    def stop(self) -> None:
        """Stop recorder and flush remaining records."""
        if not self._running:
            return

        self._running = False
        if self._writer_thread:
            self._writer_thread.join(timeout=5.0)
        self._flush_batch()
        self._backend.close()
        logger.info(f"Telemetry recorder stopped. Recorded {self._record_count} records")

    def _writer_loop(self) -> None:
        """Background writer loop."""
        while self._running:
            try:
                # Get records from queue with timeout
                try:
                    record = self._batch_queue.get(timeout=0.1)
                    self._batch.append(record)
                except Empty:
                    pass

                # Flush if batch is full or interval elapsed
                now = time.time()
                if (
                    len(self._batch) >= self.config.batch_size or
                    (self._batch and now - self._last_flush >= self.config.batch_interval_seconds)
                ):
                    self._flush_batch()

            except Exception as e:
                logger.error(f"Writer loop error: {e}")

    def _flush_batch(self) -> None:
        """Flush current batch to backend."""
        if not self._batch:
            return

        try:
            self._backend.write_batch(self._batch)
            self._batch_count += 1
        except Exception as e:
            logger.error(f"Failed to write batch: {e}")
        finally:
            self._batch = []
            self._last_flush = time.time()

    def _should_sample(self, rate: float) -> bool:
        """Probabilistic sampling."""
        import random
        return random.random() < rate

    def record_snapshot(self, snapshot: FleetSnapshot) -> None:
        """Record a fleet snapshot."""
        if not self.config.record_snapshots:
            return
        if not self._should_sample(self.config.snapshot_sample_rate):
            return

        record = TelemetryRecord.from_snapshot(snapshot)
        self._batch_queue.put(record)
        self._record_count += 1

    def record_robot(self, robot: RobotInfo) -> None:
        """Record individual robot state."""
        if not self.config.record_robots:
            return
        if not self._should_sample(self.config.robot_sample_rate):
            return

        record = TelemetryRecord.from_robot(robot)
        self._batch_queue.put(record)
        self._record_count += 1

    def record_switch(self, switch: SwitchCommand) -> None:
        """Record a switch command and its outcome."""
        if not self.config.record_switches:
            return

        record = TelemetryRecord.from_switch(switch)
        self._batch_queue.put(record)
        self._record_count += 1

    def record_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Record a custom event."""
        if not self.config.record_events:
            return

        record = TelemetryRecord(
            timestamp=time.time(),
            record_type="event",
            data={"event_type": event_type, **data}
        )
        self._batch_queue.put(record)
        self._record_count += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get recorder metrics."""
        return {
            "record_count": self._record_count,
            "batch_count": self._batch_count,
            "pending_records": self._batch_queue.qsize(),
            "current_batch_size": len(self._batch),
            "running": self._running
        }


def load_recordings(
    path: str,
    record_type: Optional[str] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None
) -> Iterator[TelemetryRecord]:
    """
    Load telemetry recordings from files.

    Args:
        path: Directory or file path
        record_type: Filter by record type
        start_time: Filter by minimum timestamp
        end_time: Filter by maximum timestamp

    Yields:
        TelemetryRecord instances
    """
    path = Path(path)

    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("telemetry_*.jsonl*"))

    for file_path in files:
        try:
            if file_path.suffix == ".gz":
                f = gzip.open(file_path, "rt", encoding="utf-8")
            else:
                f = open(file_path, "r", encoding="utf-8")

            with f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        record = TelemetryRecord.from_dict(data)

                        # Apply filters
                        if record_type and record.record_type != record_type:
                            continue
                        if start_time and record.timestamp < start_time:
                            continue
                        if end_time and record.timestamp > end_time:
                            continue

                        yield record
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
