"""
Database Manager for ConveyorBreadBagCounterSystem V2.
Production-quality database layer with:
- Foreign key support
- bag_types table management  
- Repository-style methods
- Schema initialization from schema.sql
- Thread-safe connections
- Comprehensive error handling
"""
import sqlite3
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import threading
from src.utils.AppLogging import logger
class DatabaseManager:
    """Enhanced database manager with V2 schema support."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        self._ensure_db_exists()
        self._initialize_schema()
        logger.info(f"[DatabaseManager] Initialized with V2 schema: {db_path}")
    def _ensure_db_exists(self):
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        if not db_file.exists():
            db_file.touch()
            logger.info(f"[DatabaseManager] Created new database: {self.db_path}")
    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                self.db_path, check_same_thread=False, timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
            self._local.connection.execute("PRAGMA foreign_keys = ON")
        return self._local.connection
    @contextmanager
    def _cursor(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"[DatabaseManager] Database error: {e}", exc_info=True)
            raise
        finally:
            cursor.close()
    def _initialize_schema(self):
        schema_path = Path(__file__).parent / "schema.sql"
        if schema_path.exists():
            logger.info(f"[DatabaseManager] Loading schema from: {schema_path}")
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_sql = f.read()
            with self._get_connection() as conn:
                conn.executescript(schema_sql)
                conn.commit()
            logger.info("[DatabaseManager] Schema initialized from schema.sql")
        else:
            logger.warning("[DatabaseManager] schema.sql not found, using fallback")
            self._create_fallback_schema()
        
        # Initialize default config values
        self._initialize_default_config()
    def _create_fallback_schema(self):
        with self._get_connection() as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bag_types (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    arabic_name TEXT,
                    weight REAL DEFAULT 0,
                    thumb TEXT,
                    created_at TEXT DEFAULT (datetime('now', 'utc'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    bag_type_id INTEGER NOT NULL,
                    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
                    track_id INTEGER,
                    metadata TEXT,
                    FOREIGN KEY (bag_type_id) REFERENCES bag_types(id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT DEFAULT (datetime('now', 'utc'))
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_bag_type_id ON events(bag_type_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_confidence ON events(confidence)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_bag_types_name ON bag_types(name)")
            conn.commit()
            logger.info("[DatabaseManager] Fallback schema created")
    
    def _initialize_default_config(self):
        """Initialize default configuration values if they don't exist."""
        from src.constants import (
            show_ui_screen_key,
            is_development_key,
            rtsp_username,
            rtsp_password,
            rtsp_host,
            rtsp_port,
            is_profiler_enabled,
            enable_display_key,
            enable_recording_key
        )
        
        # Default values for config keys
        default_config = {
            show_ui_screen_key: '0',
            is_development_key: '0',
            rtsp_username: 'admin',
            rtsp_password: 'a1234567',
            rtsp_host: '192.168.2.108',
            rtsp_port: '554',
            is_profiler_enabled: '0',
            enable_display_key: '0',
            enable_recording_key: '0'
        }
        
        with self._cursor() as cursor:
            for key, default_value in default_config.items():
                # Check if key exists
                cursor.execute("SELECT value FROM config WHERE key = ?", (key,))
                result = cursor.fetchone()
                
                if result is None:
                    # Insert default value
                    cursor.execute(
                        "INSERT INTO config (key, value) VALUES (?, ?)",
                        (key, default_value)
                    )
                    logger.info(f"[DatabaseManager] Initialized config: {key} = {default_value}")
        
        logger.info("[DatabaseManager] Default config values initialized")
    def get_or_create_bag_type(self, name: str, arabic_name: Optional[str] = None,
                               weight: float = 0, thumb: Optional[str] = None) -> int:
        with self._cursor() as cursor:
            cursor.execute("SELECT id FROM bag_types WHERE name = ?", (name,))
            result = cursor.fetchone()
            if result:
                return result[0]
            if thumb is None:
                thumb = f"data/classes/{name}/{name}.jpg"
            cursor.execute(
                "INSERT INTO bag_types (name, arabic_name, weight, thumb) VALUES (?, ?, ?, ?)",
                (name, arabic_name or name, weight, thumb)
            )
            bag_type_id = cursor.lastrowid
            logger.info(f"[DatabaseManager] Created bag_type: {name} (ID: {bag_type_id})")
            return bag_type_id
    def get_all_bag_types(self) -> List[Dict[str, Any]]:
        with self._cursor() as cursor:
            cursor.execute("SELECT * FROM bag_types ORDER BY name")
            return [dict(row) for row in cursor.fetchall()]
    def get_bag_type_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        with self._cursor() as cursor:
            cursor.execute("SELECT * FROM bag_types WHERE name = ?", (name,))
            row = cursor.fetchone()
            return dict(row) if row else None
    def add_event(self, timestamp: str, bag_type_name: str, confidence: float,
                 track_id: Optional[int] = None,
                 metadata: Optional[str] = None, **bag_type_metadata) -> int:
        """
        Add a counting event to the database.

        Args:
            timestamp: ISO 8601 timestamp
            bag_type_name: Name of the bag type
            confidence: Classification confidence [0.0-1.0]
            track_id: Optional track ID for debugging
            metadata: Optional JSON metadata
            **bag_type_metadata: Optional bag type metadata (arabic_name, weight, thumb)

        Returns:
            Event ID
        """
        bag_type_id = self.get_or_create_bag_type(
            name=bag_type_name,
            arabic_name=bag_type_metadata.get('arabic_name'),
            weight=bag_type_metadata.get('weight', 0),
            thumb=bag_type_metadata.get('thumb')
        )
        with self._cursor() as cursor:
            cursor.execute(
                "INSERT INTO events (timestamp, bag_type_id, confidence, track_id, metadata) VALUES (?, ?, ?, ?, ?)",
                (timestamp, bag_type_id, confidence, track_id, metadata)
            )
            event_id = cursor.lastrowid
        logger.debug(f"[DatabaseManager] Event added: id={event_id}, bag_type={bag_type_name}")
        return event_id
    def get_events_with_bag_types(self, start_date: Optional[str] = None,
                                  end_date: Optional[str] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Get events joined with bag_types data.

        Args:
            start_date: Optional start date filter (ISO 8601)
            end_date: Optional end date filter (ISO 8601)
            limit: Maximum number of events to return

        Returns:
            List of event dictionaries with bag_type metadata
        """
        query = """
            SELECT e.id, e.timestamp, e.confidence, e.track_id, e.metadata,
                   bt.id as bag_type_id, bt.name as bag_type, bt.arabic_name, bt.weight, bt.thumb
            FROM events e JOIN bag_types bt ON e.bag_type_id = bt.id
        """
        params = []
        conditions = []
        if start_date:
            conditions.append("e.timestamp >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("e.timestamp <= ?")
            params.append(end_date)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY e.timestamp ASC LIMIT ?"
        params.append(limit)
        with self._cursor() as cursor:
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    def get_aggregated_stats(self, start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> Dict[str, Any]:
        query = """
            SELECT bt.id as bag_type_id, bt.name, bt.arabic_name, bt.weight, bt.thumb,
                   COUNT(*) as count,
                   SUM(CASE WHEN e.confidence >= 0.8 THEN 1 ELSE 0 END) as high_count,
                   SUM(CASE WHEN e.confidence < 0.8 THEN 1 ELSE 0 END) as low_count
            FROM events e JOIN bag_types bt ON e.bag_type_id = bt.id
        """
        params = []
        conditions = []
        if start_date:
            conditions.append("e.timestamp >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("e.timestamp <= ?")
            params.append(end_date)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " GROUP BY bt.id"
        with self._cursor() as cursor:
            cursor.execute(query, params)
            rows = [dict(row) for row in cursor.fetchall()]
        # Explicit type conversions for aggregation
        total_count = sum(int(r['count']) for r in rows)
        total_high = sum(int(r['high_count']) for r in rows)
        total_low = sum(int(r['low_count']) for r in rows)
        total_weight = sum(int(r['count']) * float(r['weight']) for r in rows)
        by_type = {row['bag_type_id']: row for row in rows}
        return {
            'total': {'count': total_count, 'high_count': total_high, 'low_count': total_low, 'weight': total_weight},
            'by_type': by_type
        }
    def get_config(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._cursor() as cursor:
            cursor.execute("SELECT value FROM config WHERE key = ?", (key,))
            result = cursor.fetchone()
            return result[0] if result else default
    def set_config(self, key: str, value: str) -> None:
        with self._cursor() as cursor:
            cursor.execute(
                "INSERT INTO config (key, value, updated_at) VALUES (?, ?, datetime('now', 'utc')) ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = datetime('now', 'utc')",
                (key, value)
            )
        logger.debug(f"[DatabaseManager] Config updated: {key} = {value}")
    def get_all_config(self) -> Dict[str, str]:
        with self._cursor() as cursor:
            cursor.execute("SELECT key, value FROM config")
            return {row['key']: row['value'] for row in cursor.fetchall()}
    def close(self):
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
            logger.debug("[DatabaseManager] Connection closed")
    def vacuum(self):
        conn = self._get_connection()
        conn.execute("VACUUM")
        conn.commit()
        logger.info("[DatabaseManager] Database vacuumed")
    def get_schema_version(self) -> str:
        return self.get_config('schema_version', '1.0')
