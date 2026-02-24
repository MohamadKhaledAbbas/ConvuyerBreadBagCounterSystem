"""
Database Manager for ConveyorBreadBagCounterSystem V2.
Production-quality database layer with:
- Foreign key support
- bag_types table management  
- Repository-style methods
- Schema initialization from schema.sql
- Thread-safe connections
- Comprehensive error handling
- Async write queue for non-blocking hot-path DB writes
"""
import queue
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from src.utils.AppLogging import logger


class DatabaseManager:
    """Enhanced database manager with V2 schema support."""

    # Write queue tuning constants
    WRITE_QUEUE_MAX_SIZE = 10000
    WRITE_BATCH_SIZE = 50
    WRITE_FLUSH_INTERVAL = 1.0  # seconds

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        self._ensure_db_exists()
        self._initialize_schema()
        # Async write queue for non-blocking hot-path DB writes
        self._write_queue: queue.Queue[Tuple[str, tuple]] = queue.Queue(maxsize=self.WRITE_QUEUE_MAX_SIZE)
        self._write_thread: Optional[threading.Thread] = None
        self._write_stop = threading.Event()
        self._start_write_thread()
        logger.info(f"[DatabaseManager] Initialized with V2 schema: {db_path}")
    def _start_write_thread(self):
        """Start background thread for batched async writes."""
        self._write_stop.clear()
        self._write_thread = threading.Thread(
            target=self._write_loop, name="DB-WriteQueue", daemon=True
        )
        self._write_thread.start()
        logger.info("[DatabaseManager] Async write queue started")
    def _write_loop(self):
        """Background loop: drain write queue in batches, commit periodically."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA synchronous = NORMAL")
        batch_size = self.WRITE_BATCH_SIZE
        flush_interval = self.WRITE_FLUSH_INTERVAL
        pending = 0
        while not self._write_stop.is_set():
            try:
                sql, params = self._write_queue.get(timeout=flush_interval)
                try:
                    conn.execute(sql, params)
                    pending += 1
                except Exception as e:
                    logger.error(f"[DatabaseManager] Async write error: {e}")
                # Drain remaining items up to batch_size
                drained = 0
                while drained < batch_size - 1:
                    try:
                        sql2, params2 = self._write_queue.get_nowait()
                        try:
                            conn.execute(sql2, params2)
                            pending += 1
                        except Exception as e:
                            logger.error(f"[DatabaseManager] Async write error: {e}")
                        drained += 1
                    except queue.Empty:
                        break
                # Commit immediately when queue is drained or batch is full
                if pending > 0:
                    conn.commit()
                    pending = 0
            except queue.Empty:
                # Timeout: flush any pending writes
                if pending > 0:
                    conn.commit()
                    pending = 0
        # Final flush on shutdown
        while not self._write_queue.empty():
            try:
                sql, params = self._write_queue.get_nowait()
                try:
                    conn.execute(sql, params)
                    pending += 1
                except Exception:
                    pass
            except queue.Empty:
                break
        if pending > 0:
            try:
                conn.commit()
            except Exception:
                pass
        conn.close()
        logger.info("[DatabaseManager] Async write queue stopped")
    def enqueue_write(self, sql: str, params: tuple = ()):
        """
        Enqueue a write operation for async background execution.

        Non-blocking. If queue is full, the write is dropped with a warning.
        Use this for hot-path analytics writes that should not block the main loop.

        Args:
            sql: SQL INSERT/UPDATE statement
            params: Query parameters
        """
        try:
            self._write_queue.put_nowait((sql, params))
        except queue.Full:
            logger.warning("[DatabaseManager] Write queue full, dropping write")
    def flush_write_queue(self, timeout: float = 5.0):
        """Wait for write queue to drain and commit (for testing/shutdown)."""
        start = time.monotonic()
        while not self._write_queue.empty() and time.monotonic() - start < timeout:
            time.sleep(0.05)
        # Allow time for the write thread to commit the last batch
        remaining = timeout - (time.monotonic() - start)
        if remaining > 0:
            time.sleep(min(remaining, 0.2))
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
            self._local.connection.execute("PRAGMA journal_mode = WAL")
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

        # Migrate existing tables to add new columns (safe for existing DBs)
        self._migrate_track_events_schema()

        # Initialize default config values
        self._initialize_default_config()

    def _migrate_track_events_schema(self):
        """Add missing columns to track_events table for existing databases.

        Uses ALTER TABLE which is safe to call even if the column already exists
        (the exception is caught and ignored).  This handles databases created
        before the enhanced lifecycle fields were added to schema.sql.
        """
        # NOTE: Column names/types are hardcoded constants — not user input.
        # f-string usage is safe here; no SQL injection risk.
        new_columns = [
            ("entry_type", "TEXT DEFAULT 'bottom_entry'"),
            ("suspected_duplicate", "INTEGER DEFAULT 0"),
            ("ghost_recovery_count", "INTEGER DEFAULT 0"),
            ("shadow_of", "INTEGER"),
            ("shadow_count", "INTEGER DEFAULT 0"),
            ("occlusion_events", "TEXT"),
            ("merge_events", "TEXT"),
            ("ghost_exit_promoted", "INTEGER DEFAULT 0"),
            ("concurrent_track_count", "INTEGER DEFAULT 0"),
        ]
        try:
            conn = self._get_connection()
            for col_name, col_type in new_columns:
                try:
                    conn.execute(
                        f"ALTER TABLE track_events ADD COLUMN {col_name} {col_type}"
                    )
                    logger.info(
                        f"[DatabaseManager] Migrated track_events: added column {col_name}"
                    )
                except sqlite3.OperationalError:
                    pass  # Column already exists
            conn.commit()
        except Exception as e:
            logger.error(f"[DatabaseManager] Migration error: {e}")
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
        self._seed_default_bag_types()
    
    def _initialize_default_config(self):
        """Initialize default configuration values if they don't exist."""
        from src.constants import (
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
    def _seed_default_bag_types(self):
        """Seed default bag types if they don't exist (fallback schema path)."""
        default_bag_types = [
            (1, 'Brown_Orange', 'Brown_Orange', 0, 'data/classes/Brown_Orange_Family/Brown_Orange_Family.jpg', '2026-02-03 03:31:05'),
            (2, 'Red_Yellow', 'Red_Yellow', 0, 'data/classes/Red_Yellow/Red_Yellow.jpg', '2026-02-03 03:31:05'),
            (3, 'Wheatberry', 'Wheatberry', 0, 'data/classes/Wheatberry/Wheatberry.jpg', '2026-02-03 10:52:21'),
            (4, 'Blue_Yellow', 'Blue_Yellow', 0, 'data/classes/Blue_Yellow/Blue_Yellow.jpg', '2026-02-06 01:39:32'),
            (5, 'Green_Yellow', 'Green_Yellow', 0, 'data/classes/Green_Yellow/Green_Yellow.jpg', '2026-02-06 01:55:36'),
            (6, 'Bran', 'Bran', 0, 'data/classes/Bran/Bran.jpg', '2026-02-08 18:16:49'),
            (7, 'Black_Orange', 'Black_Orange', 0, 'data/classes/Black_Orange/Black_Orange.jpg', '2026-02-10 00:00:00'),
            (8, 'Purple_Yellow', 'Purple_Yellow', 0, 'data/classes/Purple_Yellow/Purple_Yellow.jpg', '2026-02-10 00:00:00'),
            (9, 'Rejected', 'Rejected', 0, 'data/classes/Rejected/Rejected.jpg', '2026-02-06 01:51:49'),
        ]
        with self._cursor() as cursor:
            cursor.executemany(
                "INSERT OR IGNORE INTO bag_types (id, name, arabic_name, weight, thumb, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                default_bag_types
            )
        logger.info("[DatabaseManager] Default bag types seeded")
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
    def add_track_event(
        self,
        track_id: int,
        event_type: str,
        timestamp: str,
        created_at: str,
        entry_x: Optional[int] = None,
        entry_y: Optional[int] = None,
        exit_x: Optional[int] = None,
        exit_y: Optional[int] = None,
        exit_direction: Optional[str] = None,
        distance_pixels: Optional[float] = None,
        duration_seconds: Optional[float] = None,
        total_frames: Optional[int] = None,
        avg_confidence: Optional[float] = None,
        total_hits: Optional[int] = None,
        classification: Optional[str] = None,
        classification_confidence: Optional[float] = None,
        position_history: Optional[str] = None
    ) -> int:
        """
        Add a track lifecycle event for analytics.

        Args:
            track_id: Track ID from tracker
            event_type: 'track_completed', 'track_lost', 'track_invalid'
            timestamp: ISO 8601 timestamp when track ended
            created_at: ISO 8601 timestamp when track was first seen
            entry_x: Center X when track was created
            entry_y: Center Y when track was created
            exit_x: Center X when track ended
            exit_y: Center Y when track ended
            exit_direction: 'top', 'bottom', 'left', 'right', 'timeout'
            distance_pixels: Total Euclidean distance traveled
            duration_seconds: Track lifetime in seconds
            total_frames: Total frames the track existed
            avg_confidence: Average detection confidence
            total_hits: Frames where track was detected
            classification: Final class name (None if skipped)
            classification_confidence: Classification confidence
            position_history: JSON array of [x,y] points

        Returns:
            Track event ID
        """
        with self._cursor() as cursor:
            cursor.execute(
                """INSERT INTO track_events (
                    track_id, event_type, timestamp, created_at,
                    entry_x, entry_y, exit_x, exit_y,
                    exit_direction, distance_pixels, duration_seconds, total_frames,
                    avg_confidence, total_hits,
                    classification, classification_confidence,
                    position_history
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    track_id, event_type, timestamp, created_at,
                    entry_x, entry_y, exit_x, exit_y,
                    exit_direction, distance_pixels, duration_seconds, total_frames,
                    avg_confidence, total_hits,
                    classification, classification_confidence,
                    position_history
                )
            )
            event_id = cursor.lastrowid
        logger.debug(f"[DatabaseManager] Track event added: id={event_id}, track={track_id}, type={event_type}")
        return event_id
    def update_track_event_classification(
        self,
        track_id: int,
        classification: str,
        classification_confidence: float
    ) -> None:
        """
        Update classification result for a track event.

        Called after async classification completes for a track_completed event.

        Args:
            track_id: Track ID to update
            classification: Final class name
            classification_confidence: Classification confidence
        """
        with self._cursor() as cursor:
            cursor.execute(
                """UPDATE track_events
                   SET classification = ?, classification_confidence = ?
                   WHERE track_id = ? AND classification IS NULL
                   ORDER BY id DESC LIMIT 1""",
                (classification, classification_confidence, track_id)
            )
        logger.debug(
            f"[DatabaseManager] Track event classification updated: "
            f"track={track_id}, class={classification}"
        )
    def get_track_events(
        self,
        event_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get track events for analytics.

        Args:
            event_type: Optional filter by event type
            start_date: Optional start date filter (ISO 8601)
            end_date: Optional end date filter (ISO 8601)
            limit: Maximum number of events to return

        Returns:
            List of track event dictionaries
        """
        query = "SELECT * FROM track_events"
        params = []
        conditions = []
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)
        with self._cursor() as cursor:
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    def get_track_event_stats(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get aggregated track event statistics.

        Returns:
            Dictionary with counts by event type and summary metrics
        """
        query = """
            SELECT
                event_type,
                COUNT(*) as count,
                AVG(duration_seconds) as avg_duration,
                AVG(distance_pixels) as avg_distance,
                AVG(avg_confidence) as avg_confidence
            FROM track_events
        """
        params = []
        conditions = []
        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " GROUP BY event_type"
        with self._cursor() as cursor:
            cursor.execute(query, params)
            rows = [dict(row) for row in cursor.fetchall()]
        by_type = {row['event_type']: row for row in rows}
        total = sum(int(r['count']) for r in rows)
        return {
            'total': total,
            'by_type': by_type
        }
    def add_track_event_detail(
        self,
        track_id: int,
        timestamp: str,
        step_type: str,
        bbox_x1: Optional[int] = None,
        bbox_y1: Optional[int] = None,
        bbox_x2: Optional[int] = None,
        bbox_y2: Optional[int] = None,
        quality_score: Optional[float] = None,
        roi_index: Optional[int] = None,
        class_name: Optional[str] = None,
        confidence: Optional[float] = None,
        is_rejected: int = 0,
        vote_distribution: Optional[str] = None,
        total_rois: Optional[int] = None,
        valid_votes: Optional[int] = None,
        detail: Optional[str] = None
    ) -> int:
        """
        Add a detailed lifecycle step for a track event.

        Args:
            track_id: Track ID from tracker
            timestamp: ISO 8601 timestamp
            step_type: One of 'roi_collected', 'roi_rejected', 'roi_classified',
                       'voting_result', 'track_created', 'track_completed',
                       'track_lost', 'track_invalid'
            bbox_x1..bbox_y2: ROI bounding box coordinates
            quality_score: ROI quality score
            roi_index: ROI index in collection
            class_name: Classification result
            confidence: Classification confidence
            is_rejected: 1 if classified as Rejected
            vote_distribution: JSON vote distribution
            total_rois: Total ROIs for voting
            valid_votes: Non-rejected votes
            detail: Additional JSON context

        Returns:
            Detail event ID
        """
        with self._cursor() as cursor:
            cursor.execute(
                """INSERT INTO track_event_details (
                    track_id, timestamp, step_type,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    quality_score, roi_index,
                    class_name, confidence, is_rejected,
                    vote_distribution, total_rois, valid_votes,
                    detail
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    track_id, timestamp, step_type,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    quality_score, roi_index,
                    class_name, confidence, is_rejected,
                    vote_distribution, total_rois, valid_votes,
                    detail
                )
            )
            return cursor.lastrowid
    def enqueue_track_event_detail(
        self,
        track_id: int,
        timestamp: str,
        step_type: str,
        bbox_x1: Optional[int] = None,
        bbox_y1: Optional[int] = None,
        bbox_x2: Optional[int] = None,
        bbox_y2: Optional[int] = None,
        quality_score: Optional[float] = None,
        roi_index: Optional[int] = None,
        class_name: Optional[str] = None,
        confidence: Optional[float] = None,
        is_rejected: int = 0,
        vote_distribution: Optional[str] = None,
        total_rois: Optional[int] = None,
        valid_votes: Optional[int] = None,
        detail: Optional[str] = None
    ):
        """
        Non-blocking version of add_track_event_detail.

        Enqueues the write to background thread. Use this from hot-path code
        (per-frame ROI collection, per-ROI classification) to avoid blocking.
        """
        sql = """INSERT INTO track_event_details (
            track_id, timestamp, step_type,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            quality_score, roi_index,
            class_name, confidence, is_rejected,
            vote_distribution, total_rois, valid_votes,
            detail
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        params = (
            track_id, timestamp, step_type,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            quality_score, roi_index,
            class_name, confidence, is_rejected,
            vote_distribution, total_rois, valid_votes,
            detail
        )
        self.enqueue_write(sql, params)
    def get_track_event_details(
        self,
        track_id: Optional[int] = None,
        step_type: Optional[str] = None,
        limit: int = 5000
    ) -> List[Dict[str, Any]]:
        """
        Get track event detail records.

        Args:
            track_id: Optional filter by track ID
            step_type: Optional filter by step type
            limit: Maximum records to return

        Returns:
            List of detail dictionaries
        """
        query = "SELECT * FROM track_event_details"
        params = []
        conditions = []
        if track_id is not None:
            conditions.append("track_id = ?")
            params.append(track_id)
        if step_type:
            conditions.append("step_type = ?")
            params.append(step_type)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY id ASC LIMIT ?"
        params.append(limit)
        with self._cursor() as cursor:
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    def get_track_lifecycle(self, track_id: int) -> Dict[str, Any]:
        """
        Get the full lifecycle of a single track: summary + all detail steps.

        WARNING: track_id is NOT unique across sessions. Use get_track_lifecycle_by_event_id()
        for unambiguous lookups.

        Args:
            track_id: Track ID to look up

        Returns:
            Dictionary with 'summary' (from track_events) and 'details' (from track_event_details)
        """
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT * FROM track_events WHERE track_id = ? ORDER BY id DESC LIMIT 1",
                (track_id,)
            )
            row = cursor.fetchone()
            summary = dict(row) if row else None
        details = self.get_track_event_details(track_id=track_id)
        return {
            'summary': summary,
            'details': details
        }

    def get_track_lifecycle_by_event_id(self, event_id: int) -> Dict[str, Any]:
        """
        Get the full lifecycle of a track by its UNIQUE event_id.

        This is the recommended method for unambiguous track lookups since
        track_id resets on each app restart.

        Args:
            event_id: Unique ID from track_events.id

        Returns:
            Dictionary with 'summary' (from track_events) and 'details' (from track_event_details)
        """
        with self._cursor() as cursor:
            # Get the track event by unique ID
            cursor.execute(
                "SELECT * FROM track_events WHERE id = ?",
                (event_id,)
            )
            row = cursor.fetchone()
            if not row:
                return {'summary': None, 'details': []}

            summary = dict(row)
            track_id = summary['track_id']
            timestamp = summary['timestamp']

            # Get details within a narrow time window around this event
            # This ensures we only get details from this specific session
            cursor.execute("""
                SELECT * FROM track_event_details 
                WHERE track_id = ? 
                  AND timestamp >= datetime(?, '-1 minute')
                  AND timestamp <= datetime(?, '+1 minute')
                ORDER BY id ASC
            """, (track_id, timestamp, timestamp))
            details = [dict(r) for r in cursor.fetchall()]

        return {
            'summary': summary,
            'details': details
        }
    def purge_old_events(self, retention_days: int = 3) -> int:
        """
        Delete events older than retention_days from the events table.

        Args:
            retention_days: Number of days to retain (default: 3)

        Returns:
            Number of events rows deleted
        """
        with self._cursor() as cursor:
            cursor.execute(
                "DELETE FROM events WHERE timestamp < datetime('now', '-' || ? || ' days')",
                (retention_days,)
            )
            events_deleted = cursor.rowcount
        logger.info(
            f"[DatabaseManager] Purged old events: "
            f"{events_deleted} rows (retention={retention_days}d)"
        )
        return events_deleted
    def purge_old_track_events(self, retention_days: int = 3) -> int:
        """
        Delete track events and details older than retention_days.

        Args:
            retention_days: Number of days to retain (default: 7)

        Returns:
            Number of track_events rows deleted
        """
        with self._cursor() as cursor:
            cursor.execute(
                "DELETE FROM track_event_details WHERE timestamp < datetime('now', '-' || ? || ' days')",
                (retention_days,)
            )
            details_deleted = cursor.rowcount
            cursor.execute(
                "DELETE FROM track_events WHERE timestamp < datetime('now', '-' || ? || ' days')",
                (retention_days,)
            )
            events_deleted = cursor.rowcount
        logger.info(
            f"[DatabaseManager] Purged old track events: "
            f"{events_deleted} events, {details_deleted} details (retention={retention_days}d)"
        )
        return events_deleted
    def close(self):
        # Stop the write queue thread first
        if self._write_thread and self._write_thread.is_alive():
            self._write_stop.set()
            self._write_thread.join(timeout=5.0)
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
