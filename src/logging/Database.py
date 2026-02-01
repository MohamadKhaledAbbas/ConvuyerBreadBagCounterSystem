"""
Database manager for ConveyerBreadBagCounterSystem.

SQLite-based storage for:
- Counting events
- Configuration values
- System metrics
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional, Tuple, List, Dict
from contextlib import contextmanager
import threading

from src.constants import CONFIG_KEYS
from src.utils.AppLogging import logger


class DatabaseManager:
    """
    Thread-safe SQLite database manager.
    
    Handles:
    - Event logging (bread bag counts)
    - Configuration storage
    - Metrics storage
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database schema
        self._init_database()
        logger.info(f"[DatabaseManager] Initialized: {db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    @contextmanager
    def _cursor(self):
        """Context manager for database cursor."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"[DatabaseManager] Database error: {e}")
            raise
        finally:
            cursor.close()
    
    def _init_database(self):
        """Initialize database schema."""
        with self._cursor() as cursor:
            # Events table for counting records
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER NOT NULL,
                    bag_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    phash TEXT,
                    image_path TEXT,
                    candidates_count INTEGER DEFAULT 1,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Configuration table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Metrics table for system monitoring
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_timestamp 
                ON events(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_bag_type 
                ON events(bag_type)
            """)
            
            # Initialize default config values
            self._init_default_config(cursor)
    
    def _init_default_config(self, cursor):
        """Initialize default configuration values."""
        defaults = {
            'show_ui_screen': '0',
            'is_development': '0',
            'is_recording': '0',
            'recording_dir': 'data/recordings',
            'recording_seconds': '60',
            'recording_fps': '25',
            'rtsp_username': 'admin',
            'rtsp_password': '',
            'rtsp_host': '192.168.1.100',
            'rtsp_port': '554',
            'is_profiler_enabled': '0',
        }
        
        for key, value in defaults.items():
            cursor.execute("""
                INSERT OR IGNORE INTO config (key, value) VALUES (?, ?)
            """, (key, value))
    
    # ==========================================================================
    # Event Methods
    # ==========================================================================
    
    def log_event(
        self,
        track_id: int,
        bag_type: str,
        confidence: float,
        phash: Optional[str] = None,
        image_path: Optional[str] = None,
        candidates_count: int = 1,
        metadata: Optional[str] = None
    ) -> int:
        """
        Log a counting event.
        
        Returns:
            Event ID
        """
        timestamp = datetime.now().isoformat()
        
        with self._cursor() as cursor:
            cursor.execute("""
                INSERT INTO events 
                (track_id, bag_type, confidence, timestamp, phash, image_path, 
                 candidates_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (track_id, bag_type, confidence, timestamp, phash, 
                  image_path, candidates_count, metadata))
            
            event_id = cursor.lastrowid
        
        logger.debug(
            f"[DatabaseManager] Logged event: id={event_id}, "
            f"track={track_id}, type={bag_type}, conf={confidence:.2f}"
        )
        return event_id
    
    def get_events(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        bag_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Query counting events.
        
        Returns:
            List of event dictionaries
        """
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        if bag_type:
            query += " AND bag_type = ?"
            params.append(bag_type)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with self._cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    def get_event_counts(self, date: Optional[str] = None) -> Dict[str, int]:
        """
        Get event counts by bag type.
        
        Args:
            date: Optional date filter (YYYY-MM-DD format)
            
        Returns:
            Dictionary of {bag_type: count}
        """
        if date:
            query = """
                SELECT bag_type, COUNT(*) as count 
                FROM events 
                WHERE DATE(timestamp) = ?
                GROUP BY bag_type
            """
            params = [date]
        else:
            query = """
                SELECT bag_type, COUNT(*) as count 
                FROM events 
                GROUP BY bag_type
            """
            params = []
        
        with self._cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
        
        return {row['bag_type']: row['count'] for row in rows}
    
    def get_total_count(self, date: Optional[str] = None) -> int:
        """Get total event count."""
        if date:
            query = "SELECT COUNT(*) FROM events WHERE DATE(timestamp) = ?"
            params = [date]
        else:
            query = "SELECT COUNT(*) FROM events"
            params = []
        
        with self._cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchone()[0]
    
    # ==========================================================================
    # Configuration Methods
    # ==========================================================================
    
    def get_config_value(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get configuration value."""
        with self._cursor() as cursor:
            cursor.execute("SELECT value FROM config WHERE key = ?", (key,))
            row = cursor.fetchone()
        
        if row:
            return row['value']
        return default
    
    def set_config_value(self, key: str, value: str):
        """Set configuration value."""
        timestamp = datetime.now().isoformat()
        
        with self._cursor() as cursor:
            cursor.execute("""
                INSERT OR REPLACE INTO config (key, value, updated_at)
                VALUES (?, ?, ?)
            """, (key, value, timestamp))
        
        logger.debug(f"[DatabaseManager] Config set: {key}={value}")
    
    def get_all_config(self) -> Dict[str, str]:
        """Get all configuration values."""
        with self._cursor() as cursor:
            cursor.execute("SELECT key, value FROM config")
            rows = cursor.fetchall()
        
        return {row['key']: row['value'] for row in rows}
    
    # ==========================================================================
    # Metrics Methods
    # ==========================================================================
    
    def log_metric(self, name: str, value: float):
        """Log a metric value."""
        with self._cursor() as cursor:
            cursor.execute("""
                INSERT INTO metrics (metric_name, metric_value)
                VALUES (?, ?)
            """, (name, value))
    
    def get_metrics(self, name: str, limit: int = 100) -> List[Dict]:
        """Get metric history."""
        with self._cursor() as cursor:
            cursor.execute("""
                SELECT metric_value, timestamp 
                FROM metrics 
                WHERE metric_name = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (name, limit))
            rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    # ==========================================================================
    # Utility Methods
    # ==========================================================================
    
    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
    
    def vacuum(self):
        """Optimize database by running VACUUM."""
        conn = self._get_connection()
        conn.execute("VACUUM")
        logger.info("[DatabaseManager] Database vacuumed")
