"""
Codec Health Monitor – Multi-Point Media Pipeline Recovery.

Monitors the media pipeline (RTSP → codec → app) and performs staged
recovery when components stall.

Data flow monitored:
    RTSP camera
      → hobot_rtsp_client  (/rtsp_image_ch_0)
      → hobot_codec        (/nv12_images)
      → ConveyorCounterApp

Recovery stages escalate from targeted codec restart up through full
media stack restart.

Root Cause Context:
    The RDK's hobot_codec uses the hardware VPU for H.264 decoding. The VPU
    can enter a stalled state when:
    - It's restarted mid-stream without receiving an IDR (keyframe)
    - Memory pressure causes the VPU to close/reopen
    - Another process briefly claims the VPU hardware

    When stalled, the node remains registered in ROS2 and receives input
    frames, but produces no output — it's waiting for a valid decode
    sequence that never arrives. The only recovery is to restart the node.

Production Deployment:
    1. Run as part of the main application (recommended):
       - Import and start in main.py

    2. Run as standalone systemd service:
       - See docs/CODEC_HEALTH_MONITOR.md
"""

import glob as glob_module
import json
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Callable, List, Tuple

from src.utils.AppLogging import logger
from src.utils.platform import is_rdk_platform


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class HealthState(Enum):
    """Health monitor state."""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    CRITICAL = "critical"       # Requires restart
    RECOVERING = "recovering"   # Restart in progress


class HealthCheckpoint(Enum):
    """Individual health check points along the media pipeline."""
    RTSP_INGEST = "rtsp_ingest"         # /rtsp_image_ch_0
    CODEC_OUTPUT = "codec_output"       # /nv12_images


class RecoveryStage(Enum):
    """Staged recovery escalation levels."""
    CODEC_ONLY = 1          # Stage 1: restart both ROS2 pipelines (breadcount-ros2 + breadcount-container-ros2)
    MEDIA_STACK = 2         # Stage 2: restart breadcount-ros2 + breadcount-container-ros2
    BROAD_SERVICES = 3      # Stage 3: all services except uvicorn
    REBOOT_RECOMMENDED = 4  # Stage 4: log recommendation; don't actually reboot


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RecoveryEvent:
    """Record of a single recovery action."""
    timestamp: float
    stage: int
    action: str
    success: bool
    details: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "time_iso": datetime.fromtimestamp(self.timestamp).isoformat(),
            "stage": self.stage,
            "action": self.action,
            "success": self.success,
            "details": self.details,
        }


@dataclass
class MonitorConfig:
    """Configuration for codec health monitor."""

    # Topic to monitor (codec output)
    topic: str = "/nv12_images"

    # Additional topic for multi-point monitoring
    rtsp_topic: str = "/rtsp_image_ch_0"

    # How long to wait for a message before considering the topic stalled
    message_timeout_sec: float = 2.5

    # How often to check topic health
    check_interval_sec: float = 2.0

    # Number of consecutive failures before triggering restart
    failure_threshold: int = 1

    # Cooldown period after restart before checking again
    # Increased to 10s to allow ROS2/DDS time to stabilize after codec restart
    restart_cooldown_sec: float = 10.0

    # Maximum restarts per hour (circuit breaker)
    max_restarts_per_hour: int = 5

    # Process name pattern to kill
    process_pattern: str = "hobot_codec"

    # Whether to attempt restart (set False for monitoring-only mode)
    enable_restart: bool = True

    # Log level for health checks (reduce noise in production)
    verbose: bool = False

    # Optional command to start/restart codec pipeline when process is missing.
    # Example: "supervisorctl restart breadcount-ros2"
    restart_command: str = ""

    # How long to wait for process to come back after restart command
    process_start_timeout_sec: float = 10.0

    # Graceful termination: SIGTERM timeout before SIGKILL
    graceful_kill_timeout_sec: float = 5.0

    # Startup cleanup of stale artifacts
    enable_startup_cleanup: bool = True

    # Maximum recovery events to keep in history
    max_recovery_events: int = 20


@dataclass
class MonitorStats:
    """Statistics for the health monitor."""
    checks_total: int = 0
    checks_healthy: int = 0
    checks_failed: int = 0
    restarts_total: int = 0
    restarts_this_hour: int = 0
    last_healthy_time: Optional[float] = None
    last_restart_time: Optional[float] = None
    last_failure_reason: Optional[str] = None
    consecutive_failures: int = 0
    state: HealthState = HealthState.UNKNOWN
    # Track restart timestamps for rate limiting
    restart_timestamps: List[float] = field(default_factory=list)
    # Staged recovery fields
    current_recovery_stage: RecoveryStage = RecoveryStage.CODEC_ONLY
    escalation_count: int = 0
    recovery_events: List[dict] = field(default_factory=list)
    health_checkpoints: dict = field(default_factory=dict)
    # Timestamp-based detection
    last_frame_time: float = 0.0


# ---------------------------------------------------------------------------
# Shared status file (cross-process communication)
# ---------------------------------------------------------------------------

from src.config.paths import CODEC_HEALTH_STATUS_FILE


def read_codec_health_status() -> Optional[dict]:
    """
    Read codec health status from shared file.

    This allows other processes (e.g., FastAPI server) to read the
    health status written by the codec health monitor in main.py.

    Returns:
        dict with health status or None if not available
    """
    try:
        if os.path.exists(CODEC_HEALTH_STATUS_FILE):
            with open(CODEC_HEALTH_STATUS_FILE, 'r') as f:
                data = json.load(f)
                # Check if status is stale (older than 60 seconds)
                if data.get("timestamp"):
                    age = time.time() - data["timestamp"]
                    data["age_seconds"] = round(age, 1)
                    data["stale"] = age > 60
                return data
    except Exception as e:
        logger.debug(f"[CodecHealthMonitor] Error reading status file: {e}")
    return None


def _write_codec_health_status(stats: dict) -> None:
    """Write codec health status to shared file for cross-process access."""
    try:
        stats["timestamp"] = time.time()
        tmp_path = CODEC_HEALTH_STATUS_FILE + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(stats, f)
        os.replace(tmp_path, CODEC_HEALTH_STATUS_FILE)
    except Exception as e:
        logger.debug(f"[CodecHealthMonitor] Error writing status file: {e}")


# ---------------------------------------------------------------------------
# Startup cleanup
# ---------------------------------------------------------------------------

def perform_startup_cleanup() -> dict:
    """
    Remove stale artifacts from previous runs / unclean shutdown.

    Cleans:
      - /tmp/codec_health_status.json
      - /dev/shm/fastrtps_*  and  /dev/shm/fast_datasharing_*

    Returns:
        dict summarising what was cleaned
    """
    cleaned: dict = {"files_removed": [], "errors": []}

    # 1. Stale status file
    if os.path.exists(CODEC_HEALTH_STATUS_FILE):
        try:
            os.remove(CODEC_HEALTH_STATUS_FILE)
            cleaned["files_removed"].append(CODEC_HEALTH_STATUS_FILE)
            logger.info(f"[StartupCleanup] Removed stale {CODEC_HEALTH_STATUS_FILE}")
        except OSError as e:
            cleaned["errors"].append(str(e))
            logger.warning(f"[StartupCleanup] Failed to remove {CODEC_HEALTH_STATUS_FILE}: {e}")

    # 2. FastDDS shared memory artifacts
    for pattern in ("/dev/shm/fastrtps_*", "/dev/shm/fast_datasharing_*"):
        for path in glob_module.glob(pattern):
            try:
                os.remove(path)
                cleaned["files_removed"].append(path)
                logger.info(f"[StartupCleanup] Removed FastDDS shm: {path}")
            except OSError as e:
                cleaned["errors"].append(str(e))
                logger.warning(f"[StartupCleanup] Failed to remove {path}: {e}")

    total = len(cleaned["files_removed"])
    if total:
        logger.info(f"[StartupCleanup] Cleanup complete – removed {total} stale file(s)")
    else:
        logger.info("[StartupCleanup] Cleanup complete – nothing to remove")

    return cleaned


# ---------------------------------------------------------------------------
# CodecHealthMonitor
# ---------------------------------------------------------------------------

class CodecHealthMonitor:
    """
    Monitors the media pipeline health and performs staged recovery.

    Usage:
        monitor = CodecHealthMonitor()
        monitor.start()

        # Later...
        monitor.stop()
        print(monitor.get_stats())
    """

    def __init__(
        self,
        config: Optional[MonitorConfig] = None,
        on_restart_callback: Optional[Callable[[], None]] = None
    ):
        """
        Initialize the codec health monitor.

        Args:
            config: Monitor configuration
            on_restart_callback: Optional callback invoked after restart
        """
        self.config = config or MonitorConfig()
        self.on_restart_callback = on_restart_callback
        self.stats = MonitorStats()

        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Check if we're on RDK (monitoring only makes sense there)
        self._is_rdk = is_rdk_platform()

        if not self._is_rdk:
            logger.info("[CodecHealthMonitor] Not on RDK platform — monitor disabled")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Start the health monitor background thread."""
        if not self._is_rdk:
            return

        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            logger.warning("[CodecHealthMonitor] Already running")
            return

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="CodecHealthMonitor",
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(
            f"[CodecHealthMonitor] Started | "
            f"topic={self.config.topic} | "
            f"timeout={self.config.message_timeout_sec}s | "
            f"interval={self.config.check_interval_sec}s | "
            f"threshold={self.config.failure_threshold}"
        )

    def stop(self, timeout: float = 5.0):
        """Stop the health monitor."""
        if self._monitor_thread is None:
            return

        self._stop_event.set()
        self._monitor_thread.join(timeout=timeout)
        self._monitor_thread = None
        logger.info("[CodecHealthMonitor] Stopped")

    def get_stats(self) -> dict:
        """Get current monitor statistics."""
        with self._lock:
            data = {
                "enabled": True,
                "state": self.stats.state.value,
                "checks_total": self.stats.checks_total,
                "checks_healthy": self.stats.checks_healthy,
                "checks_failed": self.stats.checks_failed,
                "restarts_total": self.stats.restarts_total,
                "restarts_this_hour": self.stats.restarts_this_hour,
                "consecutive_failures": self.stats.consecutive_failures,
                "last_healthy": (
                    datetime.fromtimestamp(self.stats.last_healthy_time).isoformat()
                    if self.stats.last_healthy_time else None
                ),
                "last_restart": (
                    datetime.fromtimestamp(self.stats.last_restart_time).isoformat()
                    if self.stats.last_restart_time else None
                ),
                "last_failure_reason": self.stats.last_failure_reason,
                "current_recovery_stage": self.stats.current_recovery_stage.value,
                "escalation_count": self.stats.escalation_count,
                "recovery_events": list(self.stats.recovery_events),
                "health_checkpoints": dict(self.stats.health_checkpoints),
            }
            return data

    def is_healthy(self) -> bool:
        """Check if the codec is currently healthy."""
        with self._lock:
            return self.stats.state == HealthState.HEALTHY

    def update_frame_timestamp(self):
        """
        Update the last frame timestamp for timestamp-based stall detection.

        This method MUST be called on every frame received from the codec.
        It is the replacement for slow subprocess-based health checks.
        """
        with self._lock:
            self.stats.last_frame_time = time.time()

    def force_restart(self) -> bool:
        """
        Manually trigger a codec restart (bypasses rate limiting).

        Returns:
            True if restart was initiated
        """
        if not self._is_rdk:
            return False

        logger.warning("[CodecHealthMonitor] Manual restart requested")
        ok = self._restart_codec()
        self._persist_status()
        return ok

    def force_recovery_at_stage(self, stage: int) -> bool:
        """
        Trigger recovery at a specific stage (1-4).  Stage 5 only logs.

        Returns:
            True if recovery action was executed.
        """
        if not self._is_rdk:
            return False

        try:
            target = RecoveryStage(stage)
        except ValueError:
            logger.error(f"[CodecHealthMonitor] Invalid recovery stage: {stage}")
            return False

        logger.warning(f"[CodecHealthMonitor] Manual recovery at stage {target.name}")
        ok = self._execute_recovery(target)
        self._persist_status()
        return ok

    # ------------------------------------------------------------------
    # Health check helpers
    # ------------------------------------------------------------------

    def _check_topic_alive(self, topic: Optional[str] = None,
                           timeout: Optional[float] = None) -> Tuple[bool, str]:
        """
        Check if a ROS2 topic is receiving messages.

        Returns:
            Tuple of (is_alive, reason)
        """
        topic = topic or self.config.topic
        timeout = timeout or self.config.message_timeout_sec

        try:
            result = subprocess.run(
                [
                    "ros2", "topic", "echo",
                    topic,
                    "--once",
                ],
                timeout=timeout,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                return True, "message_received"
            else:
                return False, f"echo_failed: {result.stderr[:100]}"

        except subprocess.TimeoutExpired:
            return False, f"timeout_after_{timeout}s"
        except FileNotFoundError:
            return False, "ros2_command_not_found"
        except Exception as e:
            return False, f"exception: {str(e)[:100]}"

    def _get_codec_process_info(self) -> Optional[dict]:
        """Get information about the hobot_codec process."""
        try:
            result = subprocess.run(
                ["pgrep", "-f", self.config.process_pattern],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                return {
                    "running": True,
                    "pids": pids,
                    "count": len(pids)
                }
            else:
                return {"running": False, "pids": [], "count": 0}

        except Exception as e:
            logger.error(f"[CodecHealthMonitor] Error getting process info: {e}")
            return None

    def _check_vpu_status(self) -> Optional[str]:
        """Check VPU hardware status from dmesg."""
        try:
            result = subprocess.run(
                ["dmesg"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                lines = result.stdout.split('\n')
                vpu_lines = [l for l in lines if 'vpu' in l.lower()]
                if vpu_lines:
                    return '\n'.join(vpu_lines[-3:])
            return None

        except Exception:
            return None

    def _wait_for_process(self, timeout_sec: float) -> bool:
        """Wait until codec process appears."""
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            info = self._get_codec_process_info()
            if info and info.get("running"):
                return True
            time.sleep(0.5)
        return False

    def _run_restart_command(self) -> bool:
        """Run configured restart command when codec process is absent."""
        cmd = (self.config.restart_command or "").strip()
        if not cmd:
            logger.error("[CodecHealthMonitor] No restart_command configured and codec process is missing")
            return False

        logger.warning(f"[CodecHealthMonitor] Running restart_command: {cmd}")
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                executable="/bin/bash",
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.error(
                    f"[CodecHealthMonitor] restart_command failed rc={result.returncode}: {result.stderr.strip()}"
                )
                return False

            if self._wait_for_process(self.config.process_start_timeout_sec):
                logger.info("[CodecHealthMonitor] Codec process recovered after restart_command")
                return True

            logger.error("[CodecHealthMonitor] restart_command ran but codec process did not appear")
            return False
        except Exception as e:
            logger.error(f"[CodecHealthMonitor] restart_command exception: {e}")
            return False

    # ------------------------------------------------------------------
    # Multi-point health assessment
    # ------------------------------------------------------------------

    def _check_all_health_points(self) -> dict:
        """
        Check health at every checkpoint along the pipeline.

        Returns dict keyed by HealthCheckpoint name → {alive, reason}.
        """
        results = {}

        # Codec output (primary – always checked)
        alive, reason = self._check_topic_alive(self.config.topic)
        results[HealthCheckpoint.CODEC_OUTPUT.value] = {
            "alive": alive, "reason": reason
        }

        # RTSP ingest
        alive_r, reason_r = self._check_topic_alive(
            self.config.rtsp_topic, timeout=self.config.message_timeout_sec
        )
        results[HealthCheckpoint.RTSP_INGEST.value] = {
            "alive": alive_r, "reason": reason_r
        }

        return results

    def _detect_stalled_process(
        self,
        codec_alive: bool,
        proc_info: Optional[dict] = None
    ) -> bool:
        """
        Detect if hobot_codec process exists but the pipeline is stalled
        (process alive, no output).
        """
        if proc_info is None:
            proc_info = self._get_codec_process_info()
        if proc_info and proc_info.get("running"):
            if not codec_alive:
                logger.warning(
                    "[CodecHealthMonitor] Process exists but pipeline stalled "
                    f"(PIDs={proc_info.get('pids')})"
                )
                return True
        return False

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _can_restart(self) -> Tuple[bool, str]:
        """
        Check if a restart is allowed (rate limiting / circuit breaker).

        Returns:
            (can_restart, reason)
        """
        if not self.config.enable_restart:
            return False, "restart_disabled"

        now = time.time()
        one_hour_ago = now - 3600.0

        # Prune timestamps older than 1 hour
        self.stats.restart_timestamps = [
            ts for ts in self.stats.restart_timestamps if ts > one_hour_ago
        ]
        self.stats.restarts_this_hour = len(self.stats.restart_timestamps)

        if self.stats.restarts_this_hour >= self.config.max_restarts_per_hour:
            return False, (
                f"rate_limit_exceeded: {self.stats.restarts_this_hour}/"
                f"{self.config.max_restarts_per_hour} restarts this hour"
            )

        return True, "ok"

    # ------------------------------------------------------------------
    # Graceful process termination
    # ------------------------------------------------------------------

    def _kill_process_gracefully(self, pattern: str,
                                 timeout: Optional[float] = None) -> bool:
        """
        Terminate a process by name pattern: SIGTERM first, then SIGKILL.

        Args:
            pattern: Process name pattern for pgrep.
            timeout: Seconds to wait between SIGTERM and SIGKILL.
                     None means use config default.

        Returns:
            True if the process was terminated (or was not running).
        """
        if timeout is None:
            timeout = self.config.graceful_kill_timeout_sec

        # Find PIDs first
        try:
            result = subprocess.run(
                ["pgrep", "-f", pattern],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                logger.info(f"[CodecHealthMonitor] No process matching '{pattern}' to terminate")
                return True  # nothing to terminate is success
            pids = [p.strip() for p in result.stdout.strip().split('\n') if p.strip()]
        except Exception as e:
            logger.error(f"[CodecHealthMonitor] pgrep failed for '{pattern}': {e}")
            return False

        # SIGTERM
        logger.info(f"[CodecHealthMonitor] Sending SIGTERM to '{pattern}' (PIDs: {pids})")
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except (ProcessLookupError, ValueError):
                logger.debug(f"[CodecHealthMonitor] PID {pid} already gone during SIGTERM")

        # Wait for graceful exit
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                check = subprocess.run(
                    ["pgrep", "-f", pattern],
                    capture_output=True, text=True, timeout=5
                )
                if check.returncode != 0:
                    logger.info(f"[CodecHealthMonitor] '{pattern}' exited after SIGTERM")
                    return True
            except Exception:
                pass
            time.sleep(0.5)

        # SIGKILL remaining
        logger.warning(f"[CodecHealthMonitor] '{pattern}' did not exit after SIGTERM, sending SIGKILL")
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                logger.debug(f"[CodecHealthMonitor] PID {pid} already gone during SIGKILL")

        time.sleep(0.5)
        return True

    # ------------------------------------------------------------------
    # Recovery event tracking
    # ------------------------------------------------------------------

    def _record_recovery_event(self, stage: RecoveryStage, action: str,
                               success: bool, details: str = ""):
        """Append a recovery event to the rolling history."""
        evt = RecoveryEvent(
            timestamp=time.time(),
            stage=stage.value,
            action=action,
            success=success,
            details=details,
        )
        with self._lock:
            self.stats.recovery_events.append(evt.to_dict())
            max_events = self.config.max_recovery_events
            if len(self.stats.recovery_events) > max_events:
                self.stats.recovery_events = self.stats.recovery_events[-max_events:]

    # ------------------------------------------------------------------
    # Staged recovery
    # ------------------------------------------------------------------

    def _execute_recovery(self, stage: RecoveryStage) -> bool:
        """
        Execute recovery at the given stage.

        Returns True if the action itself succeeded (does not guarantee
        health is restored – that is checked on the next cycle).
        """
        with self._lock:
            self.stats.state = HealthState.RECOVERING
            self.stats.current_recovery_stage = stage

        success = False

        if stage == RecoveryStage.CODEC_ONLY:
            success = self._recovery_stage_1()

        elif stage == RecoveryStage.MEDIA_STACK:
            success = self._recovery_stage_2()

        elif stage == RecoveryStage.BROAD_SERVICES:
            success = self._recovery_stage_3()

        elif stage == RecoveryStage.REBOOT_RECOMMENDED:
            logger.critical(
                "[CodecHealthMonitor] REBOOT RECOMMENDED – all recovery "
                "stages exhausted. Manual intervention required."
            )
            self._record_recovery_event(
                stage, "reboot_recommended", False,
                "All automated recovery stages exhausted"
            )
            return False

        self._record_recovery_event(stage, stage.name, success)

        if success:
            with self._lock:
                self.stats.restarts_total += 1
                self.stats.last_restart_time = time.time()
                self.stats.restart_timestamps.append(time.time())
                self.stats.restarts_this_hour = len(self.stats.restart_timestamps)

            logger.info(
                f"[CodecHealthMonitor] Recovery stage {stage.name} succeeded | "
                f"total_restarts={self.stats.restarts_total}"
            )

            if self.on_restart_callback:
                try:
                    self.on_restart_callback()
                except Exception as e:
                    logger.error(f"[CodecHealthMonitor] Restart callback error: {e}")

        return success

    # -- Individual stage implementations --------------------------------

    def _recovery_stage_1(self) -> bool:
        """Stage 1: Restart both ROS2 services via supervisor (safest for ROS2 context)."""
        logger.warning("[CodecHealthMonitor] Stage 1: Restarting both ROS2 services via supervisor")

        try:
            # Use supervisor to cleanly restart BOTH the main and container ROS2 pipelines
            # This is safer than pkill because supervisor properly manages the service lifecycle
            result = subprocess.run(
                "sudo supervisorctl restart breadcount-ros2 breadcount-container-ros2",
                shell=True,
                executable="/bin/bash",
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                logger.info("[CodecHealthMonitor] Both ROS2 services restarted successfully via supervisor")
                return True
            else:
                logger.error(
                    f"[CodecHealthMonitor] Stage 1 supervisorctl failed: {result.stderr.strip()}"
                )
                return False

        except Exception as e:
            logger.error(f"[CodecHealthMonitor] Stage 1 failed: {e}")
            return False

    def _recovery_stage_2(self) -> bool:
        """Stage 2: Restart both ROS2 services (breadcount-ros2 + breadcount-container-ros2)."""
        logger.warning(
            "[CodecHealthMonitor] Stage 2: Restarting both ROS2 services"
        )
        try:
            result = subprocess.run(
                "sudo supervisorctl restart breadcount-ros2 breadcount-container-ros2",
                shell=True,
                executable="/bin/bash",
                capture_output=True, text=True, timeout=30,
            )
            ok = result.returncode == 0
            if not ok:
                logger.error(
                    f"[CodecHealthMonitor] Stage 2 supervisorctl failed: {result.stderr.strip()}"
                )
            return ok
        except Exception as e:
            logger.error(f"[CodecHealthMonitor] Stage 2 failed: {e}")
            return False

    def _recovery_stage_3(self) -> bool:
        """Stage 3: Restart broader services (all except uvicorn)."""
        logger.warning(
            "[CodecHealthMonitor] Stage 3: Restarting all services except uvicorn"
        )
        try:
            result = subprocess.run(
                "sudo supervisorctl restart breadcount-ros2 breadcount-main",
                shell=True,
                executable="/bin/bash",
                capture_output=True, text=True, timeout=30,
            )
            ok = result.returncode == 0
            if not ok:
                logger.error(
                    f"[CodecHealthMonitor] Stage 3 supervisorctl failed: {result.stderr.strip()}"
                )
            return ok
        except Exception as e:
            logger.error(f"[CodecHealthMonitor] Stage 3 failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Legacy restart wrapper (used by force_restart)
    # ------------------------------------------------------------------

    def _restart_codec(self) -> bool:
        """Restart hobot_codec (stage-1 shortcut for backward compat)."""
        return self._execute_recovery(RecoveryStage.CODEC_ONLY)

    # ------------------------------------------------------------------
    # Escalation logic
    # ------------------------------------------------------------------

    def _escalate(self):
        """Move to the next recovery stage after a failed recovery."""
        with self._lock:
            current = self.stats.current_recovery_stage.value
            next_val = current + 1
            self.stats.escalation_count += 1

            try:
                next_stage = RecoveryStage(next_val)
            except ValueError:
                next_stage = RecoveryStage.REBOOT_RECOMMENDED

            prev_name = self.stats.current_recovery_stage.name
            self.stats.current_recovery_stage = next_stage
            # Note: ESCALATING state removed for simplicity, just track escalation_count

        logger.warning(
            f"[CodecHealthMonitor] ESCALATING: {prev_name} → {next_stage.name} | "
            f"escalation_count={self.stats.escalation_count}"
        )

    def _reset_escalation(self):
        """Reset escalation back to stage 1 after a healthy cycle."""
        with self._lock:
            if self.stats.current_recovery_stage != RecoveryStage.CODEC_ONLY:
                logger.info(
                    f"[CodecHealthMonitor] De-escalated from "
                    f"{self.stats.current_recovery_stage.name} → CODEC_ONLY"
                )
                self.stats.current_recovery_stage = RecoveryStage.CODEC_ONLY

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_status(self):
        """Write current stats to shared file for cross-process access."""
        try:
            _write_codec_health_status(self.get_stats())
        except Exception as e:
            logger.debug(f"[CodecHealthMonitor] Failed to persist status: {e}")

    # ------------------------------------------------------------------
    # Main monitoring loop
    # ------------------------------------------------------------------

    def _monitor_loop(self):
        """Main monitoring loop."""
        logger.info("[CodecHealthMonitor] Monitor loop started")

        # Initial delay to let system stabilize after startup
        initial_delay = 30.0
        logger.info(f"[CodecHealthMonitor] Waiting {initial_delay}s for system startup...")
        if self._stop_event.wait(initial_delay):
            return

        while not self._stop_event.is_set():
            try:
                self._run_health_check()
            except Exception as e:
                logger.error(f"[CodecHealthMonitor] Health check exception: {e}")

            # Wait for next check interval
            self._stop_event.wait(self.config.check_interval_sec)

        logger.info("[CodecHealthMonitor] Monitor loop stopped")

    def _run_health_check(self):
        """Run a single health check cycle using timestamp-based stall detection."""
        with self._lock:
            self.stats.checks_total += 1
            now = time.time()
            last_frame = self.stats.last_frame_time

        # Calculate time since last frame
        if last_frame > 0:
            delta = now - last_frame
        else:
            # No frames received yet
            delta = 0.0

        if delta <= self.config.message_timeout_sec:
            # Healthy - frames are flowing
            with self._lock:
                self.stats.checks_healthy += 1
                self.stats.consecutive_failures = 0
                self.stats.last_healthy_time = now
                self.stats.state = HealthState.HEALTHY

            if self.config.verbose:
                logger.debug(f"[CodecHealthMonitor] Health check passed: frames active ({delta:.1f}s ago)")

            with self._lock:
                self.stats.current_recovery_stage = RecoveryStage.CODEC_ONLY
        else:
            # Stall detected - no frames received within timeout
            with self._lock:
                self.stats.checks_failed += 1
                self.stats.consecutive_failures += 1
                self.stats.last_failure_reason = f"no_frames_for_{delta:.1f}s"
                self.stats.state = HealthState.CRITICAL

            logger.warning(
                f"[CodecHealthMonitor] No frames for {delta:.1f}s "
                f"(timeout={self.config.message_timeout_sec}s) | "
                f"consecutive={self.stats.consecutive_failures}"
            )

            # Trigger recovery immediately when threshold is met
            if self.stats.consecutive_failures >= self.config.failure_threshold:
                can_restart, restart_reason = self._can_restart()

                if can_restart:
                    logger.error(
                        f"[CodecHealthMonitor] CRITICAL: triggering recovery | "
                        f"no_frames_for={delta:.1f}s | "
                        f"stage={self.stats.current_recovery_stage.name}"
                    )

                    success = self._execute_recovery(self.stats.current_recovery_stage)

                    if success:
                        with self._lock:
                            self.stats.consecutive_failures = 0

                        logger.info(
                            f"[CodecHealthMonitor] Recovery succeeded, waiting "
                            f"{self.config.restart_cooldown_sec}s for pipeline to reinitialize..."
                        )
                        self._stop_event.wait(self.config.restart_cooldown_sec)
                    else:
                        # Recovery failed - try escalation
                        self._escalate()
                else:
                    logger.error(
                        f"[CodecHealthMonitor] CRITICAL but cannot restart: {restart_reason}"
                    )

        # Persist status after every check
        self._persist_status()


# ---------------------------------------------------------------------------
# Standalone execution for systemd service
# ---------------------------------------------------------------------------

def main():
    """Main entry point for standalone service execution."""
    import sys

    logger.info("[CodecHealthMonitor] Standalone service starting...")

    # Startup cleanup
    if os.getenv("CODEC_MONITOR_ENABLE_STARTUP_CLEANUP", "true").lower() == "true":
        perform_startup_cleanup()

    # Use hardcoded defaults (no environment variable overrides)
    config = MonitorConfig()

    logger.info(f"[CodecHealthMonitor] Starting standalone service with config: {config}")

    monitor = CodecHealthMonitor(config=config)

    # Handle shutdown signals
    def signal_handler(signum, frame):
        logger.info(f"[CodecHealthMonitor] Received signal {signum}, shutting down...")
        monitor.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start monitor
    monitor.start()

    # Keep main thread alive
    try:
        while True:
            time.sleep(60)
            stats = monitor.get_stats()
            logger.info(f"[CodecHealthMonitor] Stats: {stats}")
    except KeyboardInterrupt:
        monitor.stop()


if __name__ == "__main__":
    main()
