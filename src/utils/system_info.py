"""
System information utilities for the status page.

Collects hardware metrics:
- CPU / BPU / DDR temperatures (via ``hrut_somstatus`` on RDK)
- CPU load (via ``/proc/loadavg`` on Linux, ``psutil`` elsewhere)
- RAM usage (via ``/proc/meminfo`` on Linux, ``psutil`` elsewhere)
- Database file size
- Free disk space

All functions are non-blocking and return sensible defaults on error
so the health endpoint never crashes due to a missing metric.
"""

import os
import re
import shutil
import subprocess
from typing import Dict, Any, Optional

from src.utils.AppLogging import logger
from src.utils.platform import is_rdk_platform, IS_WINDOWS


def get_system_info(db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Collect all system metrics in a single call.

    Args:
        db_path: Path to the SQLite database file (for size calculation).

    Returns:
        Dict with temperature, cpu_load, db_size, and disk_space sections.
    """
    info: Dict[str, Any] = {}

    # ── Temperatures & board info ──
    if is_rdk_platform():
        info["temperatures"] = _get_rdk_temperatures()
        info["cpu_info"] = _get_rdk_cpu_info()
    else:
        info["temperatures"] = _get_generic_temperatures()
        info["cpu_info"] = _get_generic_cpu_info()

    # ── CPU load ──
    info["cpu_load"] = _get_cpu_load()

    # ── RAM usage ──
    info["ram"] = _get_ram_usage()

    # ── Database size ──
    info["db_size"] = _get_db_size(db_path)

    # ── Disk space ──
    info["disk"] = _get_disk_space()

    return info


# ─── RDK-specific (hrut_somstatus) ──────────────────────────────────────

def _get_rdk_temperatures() -> Dict[str, Any]:
    """
    Parse ``hrut_somstatus`` output for CPU, BPU, and DDR temperatures.

    Typical hrut_somstatus output includes lines like:
        CPU Temperature: 65.3°C
        BPU Temperature: 62.1°C
        DDR Temperature: 58.7°C
    """
    result: Dict[str, Any] = {
        "cpu_temp": None,
        "bpu_temp": None,
        "ddr_temp": None,
        "source": "hrut_somstatus",
    }
    try:
        proc = subprocess.run(
            ["hrut_somstatus"],
            capture_output=True, text=True, timeout=5,
        )
        output = proc.stdout + proc.stderr

        # Parse temperature lines (flexible pattern)
        for line in output.splitlines():
            line_lower = line.lower()
            temp_match = re.search(r'([\d.]+)\s*°?[cC]', line)
            if temp_match:
                temp_val = float(temp_match.group(1))
                if 'cpu' in line_lower and 'temp' in line_lower:
                    result["cpu_temp"] = temp_val
                elif 'bpu' in line_lower and 'temp' in line_lower:
                    result["bpu_temp"] = temp_val
                elif 'ddr' in line_lower and 'temp' in line_lower:
                    result["ddr_temp"] = temp_val

        # Fallback: try reading from sysfs thermal zones
        if result["cpu_temp"] is None:
            result.update(_read_thermal_zones())

    except FileNotFoundError:
        logger.debug("[SystemInfo] hrut_somstatus not found, trying sysfs")
        result.update(_read_thermal_zones())
    except Exception as e:
        logger.debug(f"[SystemInfo] Error getting RDK temps: {e}")
        result.update(_read_thermal_zones())

    return result


def _read_thermal_zones() -> Dict[str, Any]:
    """Fallback: read temperatures from Linux sysfs thermal zones."""
    temps: Dict[str, Any] = {}
    try:
        thermal_base = "/sys/class/thermal"
        if os.path.isdir(thermal_base):
            zones = sorted(os.listdir(thermal_base))
            for zone in zones:
                if not zone.startswith("thermal_zone"):
                    continue
                temp_path = os.path.join(thermal_base, zone, "temp")
                type_path = os.path.join(thermal_base, zone, "type")
                if os.path.isfile(temp_path):
                    with open(temp_path) as f:
                        raw = f.read().strip()
                    temp_c = float(raw) / 1000.0
                    zone_type = ""
                    if os.path.isfile(type_path):
                        with open(type_path) as f:
                            zone_type = f.read().strip().lower()
                    if "cpu" in zone_type and temps.get("cpu_temp") is None:
                        temps["cpu_temp"] = round(temp_c, 1)
                    elif "bpu" in zone_type and temps.get("bpu_temp") is None:
                        temps["bpu_temp"] = round(temp_c, 1)
                    elif "ddr" in zone_type and temps.get("ddr_temp") is None:
                        temps["ddr_temp"] = round(temp_c, 1)
                    elif temps.get("cpu_temp") is None:
                        # First zone without specific type → assume CPU
                        temps["cpu_temp"] = round(temp_c, 1)
    except Exception as e:
        logger.debug(f"[SystemInfo] Error reading thermal zones: {e}")
    return temps


def _get_rdk_cpu_info() -> Dict[str, Any]:
    """Get CPU info from RDK board via hrut_somstatus."""
    info: Dict[str, Any] = {"model": None, "cores": None}
    try:
        proc = subprocess.run(
            ["hrut_somstatus"],
            capture_output=True, text=True, timeout=5,
        )
        output = proc.stdout + proc.stderr
        for line in output.splitlines():
            line_lower = line.lower()
            if 'cpu' in line_lower and ('model' in line_lower or 'type' in line_lower):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    info["model"] = parts[1].strip()
            elif 'core' in line_lower and 'count' in line_lower:
                num_match = re.search(r'(\d+)', line)
                if num_match:
                    info["cores"] = int(num_match.group(1))
    except Exception:
        pass

    # Fallback to /proc/cpuinfo
    if info["model"] is None:
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        info["model"] = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

    if info["cores"] is None:
        info["cores"] = os.cpu_count()

    return info


# ─── Generic (non-RDK) ──────────────────────────────────────────────────

def _get_generic_temperatures() -> Dict[str, Any]:
    """Get temperatures on non-RDK systems (psutil or sysfs fallback)."""
    result: Dict[str, Any] = {
        "cpu_temp": None,
        "bpu_temp": None,
        "ddr_temp": None,
        "source": "psutil" if IS_WINDOWS else "sysfs",
    }

    # Try psutil first (works on Windows and some Linux)
    try:
        import psutil
        temps = psutil.sensors_temperatures()
        if temps:
            for name, entries in temps.items():
                if entries and result["cpu_temp"] is None:
                    result["cpu_temp"] = round(entries[0].current, 1)
                    break
            return result
    except (ImportError, AttributeError):
        pass

    # Linux sysfs fallback
    if not IS_WINDOWS:
        result.update(_read_thermal_zones())

    return result


def _get_generic_cpu_info() -> Dict[str, Any]:
    """Get CPU info on non-RDK systems."""
    import platform
    return {
        "model": platform.processor() or None,
        "cores": os.cpu_count(),
    }


# ─── CPU Load ───────────────────────────────────────────────────────────

def _get_cpu_load() -> Dict[str, Any]:
    """
    Get CPU load averages.

    Returns 1-min, 5-min, 15-min load averages on Linux,
    or a simple percentage estimate on Windows.
    """
    result: Dict[str, Any] = {
        "load_1m": None,
        "load_5m": None,
        "load_15m": None,
        "load_percent": None,
    }

    # Try /proc/loadavg (Linux)
    try:
        with open("/proc/loadavg") as f:
            parts = f.read().split()
            result["load_1m"] = float(parts[0])
            result["load_5m"] = float(parts[1])
            result["load_15m"] = float(parts[2])
            cores = os.cpu_count() or 1
            result["load_percent"] = round(result["load_1m"] / cores * 100, 1)
            return result
    except (FileNotFoundError, IndexError, ValueError):
        pass

    # Fallback: psutil
    try:
        import psutil
        result["load_percent"] = psutil.cpu_percent(interval=0.1)
    except ImportError:
        pass

    return result


# ─── RAM Usage ──────────────────────────────────────────────────────────

def _get_ram_usage() -> Dict[str, Any]:
    """
    Get RAM usage statistics.

    Reads from ``/proc/meminfo`` on Linux (no external dependencies),
    or falls back to ``psutil`` on Windows.

    Returns:
        Dict with total, used, free, available (bytes + display), and usage_percent.
    """
    result: Dict[str, Any] = {
        "total_bytes": 0, "used_bytes": 0, "free_bytes": 0, "available_bytes": 0,
        "total_display": "—", "used_display": "—", "free_display": "—",
        "available_display": "—",
        "usage_percent": 0.0,
    }

    # Try /proc/meminfo (Linux — works on RDK and any Linux)
    try:
        meminfo: Dict[str, int] = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    # Values in /proc/meminfo are in kB
                    meminfo[key] = int(parts[1]) * 1024

        total = meminfo.get("MemTotal", 0)
        free = meminfo.get("MemFree", 0)
        available = meminfo.get("MemAvailable", 0)
        buffers = meminfo.get("Buffers", 0)
        cached = meminfo.get("Cached", 0)

        # "used" in the practical sense = total - available
        # (available accounts for reclaimable buffers/cache)
        used = total - available if available else total - free - buffers - cached

        result["total_bytes"] = total
        result["used_bytes"] = max(0, used)
        result["free_bytes"] = free
        result["available_bytes"] = available
        result["total_display"] = _format_bytes(total)
        result["used_display"] = _format_bytes(max(0, used))
        result["free_display"] = _format_bytes(free)
        result["available_display"] = _format_bytes(available)
        result["usage_percent"] = round(used / total * 100, 1) if total > 0 else 0.0
        return result
    except (FileNotFoundError, ValueError):
        pass

    # Fallback: psutil (Windows and other platforms)
    try:
        import psutil  # noqa: F811
        mem = psutil.virtual_memory()
        result["total_bytes"] = mem.total
        result["used_bytes"] = mem.used
        result["free_bytes"] = mem.free
        result["available_bytes"] = mem.available
        result["total_display"] = _format_bytes(mem.total)
        result["used_display"] = _format_bytes(mem.used)
        result["free_display"] = _format_bytes(mem.free)
        result["available_display"] = _format_bytes(mem.available)
        result["usage_percent"] = mem.percent
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"[SystemInfo] Error getting RAM usage: {e}")

    return result


# ─── Database Size ──────────────────────────────────────────────────────

def _get_db_size(db_path: Optional[str]) -> Dict[str, Any]:
    """Get database file size in human-readable format."""
    result: Dict[str, Any] = {"bytes": 0, "display": "—", "path": db_path}
    if not db_path or not os.path.isfile(db_path):
        return result
    try:
        size = os.path.getsize(db_path)
        result["bytes"] = size
        result["display"] = _format_bytes(size)

        # Also check WAL file size
        wal_path = db_path + "-wal"
        if os.path.isfile(wal_path):
            wal_size = os.path.getsize(wal_path)
            result["wal_bytes"] = wal_size
            result["wal_display"] = _format_bytes(wal_size)
    except Exception as e:
        logger.debug(f"[SystemInfo] Error getting DB size: {e}")
    return result


# ─── Disk Space ─────────────────────────────────────────────────────────

def _get_disk_space() -> Dict[str, Any]:
    """Get free disk space for the root (or data) partition."""
    result: Dict[str, Any] = {
        "total_bytes": 0, "used_bytes": 0, "free_bytes": 0,
        "total_display": "—", "used_display": "—", "free_display": "—",
        "usage_percent": 0.0,
    }
    try:
        # Use the data directory to check the correct partition
        check_path = "data" if os.path.isdir("data") else "/"
        usage = shutil.disk_usage(check_path)
        result["total_bytes"] = usage.total
        result["used_bytes"] = usage.used
        result["free_bytes"] = usage.free
        result["total_display"] = _format_bytes(usage.total)
        result["used_display"] = _format_bytes(usage.used)
        result["free_display"] = _format_bytes(usage.free)
        result["usage_percent"] = round(usage.used / usage.total * 100, 1) if usage.total > 0 else 0.0
    except Exception as e:
        logger.debug(f"[SystemInfo] Error getting disk space: {e}")
    return result


# ─── Helpers ────────────────────────────────────────────────────────────

def _format_bytes(num_bytes: int) -> str:
    """Format byte count to human-readable string (e.g., '1.2 GB')."""
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"




