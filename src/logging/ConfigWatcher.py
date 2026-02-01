"""
Configuration watcher for runtime config changes.

Monitors database configuration and triggers callbacks on changes.
"""

import threading
import time
from typing import Dict, Callable, Optional

from src.utils.AppLogging import logger


class ConfigWatcher:
    """
    Watch configuration values for changes and trigger callbacks.
    
    Polls the database at specified intervals and invokes callbacks
    when watched values change.
    """
    
    def __init__(self, db_path: str, poll_interval: float = 5.0):
        """
        Initialize config watcher.
        
        Args:
            db_path: Path to database file
            poll_interval: Seconds between polls
        """
        self.db_path = db_path
        self.poll_interval = poll_interval
        
        self._watches: Dict[str, Callable] = {}
        self._current_values: Dict[str, str] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Lazy import to avoid circular dependency
        self._db = None
    
    def _get_db(self):
        """Get database manager (lazy initialization)."""
        if self._db is None:
            from src.logging.Database import DatabaseManager
            self._db = DatabaseManager(self.db_path)
        return self._db
    
    def add_watch(self, key: str, callback: Callable[[str, str], None]):
        """
        Add a configuration key to watch.
        
        Args:
            key: Configuration key to watch
            callback: Function to call when value changes (old_value, new_value)
        """
        self._watches[key] = callback
        
        # Initialize current value
        db = self._get_db()
        self._current_values[key] = db.get_config_value(key, "")
        
        logger.debug(f"[ConfigWatcher] Watching: {key}")
    
    def remove_watch(self, key: str):
        """Remove a watch."""
        self._watches.pop(key, None)
        self._current_values.pop(key, None)
    
    def start(self):
        """Start watching in background thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("[ConfigWatcher] Started")
    
    def stop(self):
        """Stop watching."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("[ConfigWatcher] Stopped")
    
    def _poll_loop(self):
        """Polling loop for configuration changes."""
        while self._running:
            try:
                self._check_changes()
            except Exception as e:
                logger.error(f"[ConfigWatcher] Error checking changes: {e}")
            
            time.sleep(self.poll_interval)
    
    def _check_changes(self):
        """Check for configuration changes."""
        db = self._get_db()
        
        for key, callback in list(self._watches.items()):
            new_value = db.get_config_value(key, "")
            old_value = self._current_values.get(key, "")
            
            if new_value != old_value:
                logger.info(f"[ConfigWatcher] Config changed: {key}: {old_value} -> {new_value}")
                self._current_values[key] = new_value
                
                try:
                    callback(old_value, new_value)
                except Exception as e:
                    logger.error(f"[ConfigWatcher] Callback error for {key}: {e}")
