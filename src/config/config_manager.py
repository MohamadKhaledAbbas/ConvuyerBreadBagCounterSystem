"""
Configuration Manager for ConveyorBreadBagCounterSystem.

Centralized configuration with:
- Type-safe dataclass
- Default values
- Global instance
- Easy updates

All configuration in one place for maintainability.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AppConfig:
    """
    Application configuration with defaults.

    All system settings in one place for easy management.
    """

    # ==================== Timezone Configuration ====================
    timezone_offset_hours: int = 3  # Offset from UTC for production environment

    # ==================== Analytics Configuration ====================
    noise_threshold: int = 10  # Minimum bags in run to be valid (not noise)
    shift_start_hour: int = 16  # Daily shift start time (4 PM)
    shift_end_hour: int = 14    # Daily shift end time (2 PM next day)

    # ==================== Database Configuration ====================
    db_path: str = "data/db/bag_events.db"

    # ==================== Image Paths Configuration ====================
    # Filesystem paths
    known_classes_dir: str = "data/classes"      # Known bag type images
    unknown_classes_dir: str = "data/unknown"    # Unknown/unclassified images

    # Web serving paths (URL paths)
    web_known_classes_path: str = "known_classes"
    web_unknown_classes_path: str = "unknown_classes"

    # ==================== Analytics Display ====================
    max_events_per_query: int = 10000  # Max events to fetch per analytics query
    confidence_threshold_high: float = 0.8  # Threshold for "high confidence"


# ============================================================================
# Global Configuration Instance
# ============================================================================

_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """
    Get global configuration instance (singleton).

    Lazy initialization on first access.

    Returns:
        AppConfig: Global configuration instance
    """
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def update_config(**kwargs) -> None:
    """
    Update configuration values.

    Example:
        update_config(timezone_offset_hours=5, noise_threshold=15)

    Args:
        **kwargs: Configuration fields to update

    Raises:
        AttributeError: If invalid configuration key provided
    """
    config = get_config()
    for key, value in kwargs.items():
        if not hasattr(config, key):
            raise AttributeError(f"Invalid configuration key: {key}")
        setattr(config, key, value)


def reset_config() -> None:
    """Reset configuration to defaults."""
    global _config
    _config = AppConfig()
