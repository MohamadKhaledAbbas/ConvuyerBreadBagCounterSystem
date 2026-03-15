"""
Codec Health Monitor for hobot_codec VPU Decoder.

Monitors the /nv12_images topic and automatically restarts hobot_codec
when it becomes unresponsive. This handles VPU decoder stalls that occur
when the hardware decoder loses sync (e.g., after VPU close/reopen cycles
or when restarted mid-GOP without an IDR keyframe).

Production Deployment:
    1. Run as part of the main application (recommended):
       - Import and start in main.py

    2. Run as standalone systemd service:
       - See docs/CODEC_HEALTH_MONITOR.md

Root Cause Context:
    The RDK's hobot_codec uses the hardware VPU for H.264 decoding. The VPU
    can enter a stalled state when:
    - It's restarted mid-stream without receiving an IDR (keyframe)
    - Memory pressure causes the VPU to close/reopen
    - Another process briefly claims the VPU hardware

    When stalled, the node remains registered in ROS2 and receives input
    frames, but produces no output — it's waiting for a valid decode
    sequence that never arrives. The only recovery is to restart the node.
"""

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Callable, List

from src.utils.AppLogging import logger
from src.utils.platform import is_rdk_platform


class HealthState(Enum):
    """Health monitor state."""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Missed some checks but not yet critical
    CRITICAL = "critical"  # Requires restart
    RECOVERING = "recovering"  # Restart in progress


@dataclass
class MonitorConfig:
    """Configuration for codec health monitor."""

    # Topic to monitor
    topic: str = "/nv12_images"

    # How long to wait for a message before considering the topic stalled
    message_timeout_sec: float = 10.0

    # How often to check topic health
    check_interval_sec: float = 15.0

    # Number of consecutive failures before triggering restart
    failure_threshold: int = 2

    # Cooldown period after restart before checking again
    restart_cooldown_sec: float = 30.0

    # Maximum restarts per hour (circuit breaker)
    max_restarts_per_hour: int = 5

    # Process name pattern to kill
    process_pattern: str = "hobot_codec"

    # Whether to attempt restart (set False for monitoring-only mode)
    enable_restart: bool = True

    # Log level for health checks (reduce noise in production)
    verbose: bool = False


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


class CodecHealthMonitor:
    """
    Monitors hobot_codec health and restarts it when stalled.

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
            return {
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
            }

    def is_healthy(self) -> bool:
        """Check if the codec is currently healthy."""
        with self._lock:
            return self.stats.state == HealthState.HEALTHY

    def _check_topic_alive(self) -> tuple[bool, str]:
        """
        Check if the monitored topic is receiving messages.

        Returns:
            Tuple of (is_alive, reason)
        """
        try:
            # Use ros2 topic echo with timeout to check for messages
            result = subprocess.run(
                [
                    "ros2", "topic", "echo",
                    self.config.topic,
                    "--once",  # Exit after receiving one message
                ],
                timeout=self.config.message_timeout_sec,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                return True, "message_received"
            else:
                return False, f"echo_failed: {result.stderr[:100]}"

        except subprocess.TimeoutExpired:
            return False, f"timeout_after_{self.config.message_timeout_sec}s"
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
                    # Return last 3 VPU-related lines
                    return '\n'.join(vpu_lines[-3:])
            return None

        except Exception:
            return None

    def _prune_restart_timestamps(self):
        """Remove restart timestamps older than 1 hour."""
        one_hour_ago = time.time() - 3600
        self.stats.restart_timestamps = [
            ts for ts in self.stats.restart_timestamps
            if ts > one_hour_ago
        ]
        self.stats.restarts_this_hour = len(self.stats.restart_timestamps)

    def _can_restart(self) -> tuple[bool, str]:
        """Check if we're allowed to restart (rate limiting)."""
        self._prune_restart_timestamps()

        if not self.config.enable_restart:
            return False, "restart_disabled"

        if self.stats.restarts_this_hour >= self.config.max_restarts_per_hour:
            return False, f"rate_limit_exceeded ({self.stats.restarts_this_hour}/{self.config.max_restarts_per_hour} this hour)"

        if self.stats.last_restart_time:
            time_since_restart = time.time() - self.stats.last_restart_time
            if time_since_restart < self.config.restart_cooldown_sec:
                return False, f"cooldown_active ({time_since_restart:.0f}s < {self.config.restart_cooldown_sec}s)"

        return True, "allowed"

    def _restart_codec(self) -> bool:
        """
        Restart the hobot_codec process.

        Returns:
            True if restart was initiated successfully
        """
        with self._lock:
            self.stats.state = HealthState.RECOVERING

        logger.warning("[CodecHealthMonitor] Initiating hobot_codec restart...")

        # Get current process info for logging
        proc_info = self._get_codec_process_info()
        if proc_info:
            logger.info(f"[CodecHealthMonitor] Current process state: {proc_info}")

        # Check VPU status before restart
        vpu_status = self._check_vpu_status()
        if vpu_status:
            logger.info(f"[CodecHealthMonitor] VPU status before restart:\n{vpu_status}")

        try:
            # Kill hobot_codec processes
            # Using SIGKILL (-9) because the process is likely stuck in VPU wait
            result = subprocess.run(
                ["pkill", "-9", "-f", self.config.process_pattern],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                logger.info("[CodecHealthMonitor] Sent SIGKILL to hobot_codec")
            elif result.returncode == 1:
                # No processes matched — might have already died
                logger.warning("[CodecHealthMonitor] No hobot_codec process found to kill")
            else:
                logger.error(f"[CodecHealthMonitor] pkill failed: {result.stderr}")
                return False

            # Wait for process to actually die
            time.sleep(2)

            # Verify it's dead
            proc_info_after = self._get_codec_process_info()
            if proc_info_after and proc_info_after.get("running"):
                logger.warning(
                    f"[CodecHealthMonitor] Process still running after SIGKILL: "
                    f"{proc_info_after}"
                )
                # Try harder
                subprocess.run(["pkill", "-9", "-f", self.config.process_pattern], timeout=5)
                time.sleep(2)

            # Record restart
            with self._lock:
                self.stats.restarts_total += 1
                self.stats.last_restart_time = time.time()
                self.stats.restart_timestamps.append(time.time())
                self.stats.restarts_this_hour = len(self.stats.restart_timestamps)

            logger.info(
                f"[CodecHealthMonitor] Restart initiated | "
                f"total_restarts={self.stats.restarts_total} | "
                f"restarts_this_hour={self.stats.restarts_this_hour}"
            )

            # Call callback if provided
            if self.on_restart_callback:
                try:
                    self.on_restart_callback()
                except Exception as e:
                    logger.error(f"[CodecHealthMonitor] Restart callback error: {e}")

            return True

        except subprocess.TimeoutExpired:
            logger.error("[CodecHealthMonitor] Restart timed out")
            return False
        except Exception as e:
            logger.error(f"[CodecHealthMonitor] Restart failed: {e}")
            return False

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
        """Run a single health check cycle."""
        with self._lock:
            self.stats.checks_total += 1

        # Check if topic is alive
        is_alive, reason = self._check_topic_alive()

        if is_alive:
            # Healthy
            with self._lock:
                self.stats.checks_healthy += 1
                self.stats.consecutive_failures = 0
                self.stats.last_healthy_time = time.time()
                self.stats.state = HealthState.HEALTHY

            if self.config.verbose:
                logger.debug(f"[CodecHealthMonitor] Health check passed: {reason}")
        else:
            # Failed
            with self._lock:
                self.stats.checks_failed += 1
                self.stats.consecutive_failures += 1
                self.stats.last_failure_reason = reason

                if self.stats.consecutive_failures >= self.config.failure_threshold:
                    self.stats.state = HealthState.CRITICAL
                else:
                    self.stats.state = HealthState.DEGRADED

            logger.warning(
                f"[CodecHealthMonitor] Health check FAILED | "
                f"reason={reason} | "
                f"consecutive={self.stats.consecutive_failures}/{self.config.failure_threshold}"
            )

            # Check if we should restart
            if self.stats.consecutive_failures >= self.config.failure_threshold:
                can_restart, restart_reason = self._can_restart()

                if can_restart:
                    # Get additional diagnostics before restart
                    proc_info = self._get_codec_process_info()
                    logger.error(
                        f"[CodecHealthMonitor] CRITICAL | "
                        f"topic={self.config.topic} stalled | "
                        f"failures={self.stats.consecutive_failures} | "
                        f"process={proc_info} | "
                        f"action=RESTART"
                    )

                    if self._restart_codec():
                        with self._lock:
                            self.stats.consecutive_failures = 0

                        # Wait for cooldown before next check
                        logger.info(
                            f"[CodecHealthMonitor] Waiting {self.config.restart_cooldown_sec}s "
                            f"for codec to reinitialize..."
                        )
                        self._stop_event.wait(self.config.restart_cooldown_sec)
                else:
                    logger.error(
                        f"[CodecHealthMonitor] CRITICAL but cannot restart: {restart_reason}"
                    )

    def force_restart(self) -> bool:
        """
        Manually trigger a codec restart (bypasses rate limiting).

        Returns:
            True if restart was initiated
        """
        if not self._is_rdk:
            return False

        logger.warning("[CodecHealthMonitor] Manual restart requested")
        return self._restart_codec()


# ─────────────────────────────────────────────────────────────────────────────
# Standalone execution for systemd service
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Main entry point for standalone service execution."""
    import sys

    # Configure from environment
    config = MonitorConfig(
        topic=os.getenv("CODEC_MONITOR_TOPIC", "/nv12_images"),
        message_timeout_sec=float(os.getenv("CODEC_MONITOR_TIMEOUT", "10")),
        check_interval_sec=float(os.getenv("CODEC_MONITOR_INTERVAL", "15")),
        failure_threshold=int(os.getenv("CODEC_MONITOR_THRESHOLD", "2")),
        restart_cooldown_sec=float(os.getenv("CODEC_MONITOR_COOLDOWN", "30")),
        max_restarts_per_hour=int(os.getenv("CODEC_MONITOR_MAX_RESTARTS", "5")),
        enable_restart=os.getenv("CODEC_MONITOR_ENABLE_RESTART", "true").lower() == "true",
        verbose=os.getenv("CODEC_MONITOR_VERBOSE", "false").lower() == "true",
    )

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

