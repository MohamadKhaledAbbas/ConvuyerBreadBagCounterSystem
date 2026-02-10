-- ============================================================================
-- Conveyor Bread Bag Counter System V2 - Database Schema
-- ============================================================================
--
-- Schema Version: 2.0
-- Created: 2026-02-03
--
-- This schema defines the database structure for the V2 counting system with:
-- - bag_types: Catalog of bread bag types with metadata
-- - events: Counting events with foreign key to bag_types
-- - config: System configuration key-value store
--
-- All tables use CREATE TABLE IF NOT EXISTS for safe initialization.
-- ============================================================================

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- ============================================================================
-- Table: bag_types
-- ============================================================================
-- Stores the catalog of different bread bag types with their metadata.
-- Each bag type has a name, optional Arabic name, weight, and thumbnail.
-- ============================================================================

CREATE TABLE IF NOT EXISTS bag_types (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,              -- English name (e.g., "Wheatberry", "Multigrain")
    arabic_name TEXT,                       -- Arabic/localized name (optional)
    weight REAL DEFAULT 0,                  -- Weight in grams (optional)
    thumb TEXT,                             -- Path to thumbnail image (optional)
    created_at TEXT DEFAULT (datetime('now', 'utc'))  -- Creation timestamp (UTC)
);

-- Index for fast lookup by name
CREATE INDEX IF NOT EXISTS idx_bag_types_name ON bag_types(name);

-- ============================================================================
-- Table: events
-- ============================================================================
-- Stores each counting event (detected and classified bag).
-- Uses foreign key to bag_types for referential integrity.
-- ============================================================================

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,                -- ISO 8601 timestamp (e.g., "2026-02-03T14:30:00")
    bag_type_id INTEGER NOT NULL,          -- Foreign key to bag_types.id
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),  -- Classification confidence [0.0-1.0]
    track_id INTEGER,                      -- Track ID from tracker (for debugging)
    metadata TEXT,                         -- JSON metadata (vote_ratio, smoothed, etc.)

    -- Foreign key constraint
    FOREIGN KEY (bag_type_id) REFERENCES bag_types(id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_bag_type_id ON events(bag_type_id);
CREATE INDEX IF NOT EXISTS idx_events_track_id ON events(track_id);

-- ============================================================================
-- Table: track_events
-- ============================================================================
-- Stores every track lifecycle event for analytics and debugging.
-- Records the full journey of each tracked object through the frame.
-- ============================================================================

CREATE TABLE IF NOT EXISTS track_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL,                 -- Track ID from tracker
    event_type TEXT NOT NULL,                   -- 'track_completed', 'track_lost', 'track_invalid'
    timestamp TEXT NOT NULL,                    -- ISO 8601 timestamp when track ended
    created_at TEXT NOT NULL,                   -- ISO 8601 timestamp when track was first seen

    -- Entry position (where the track first appeared)
    entry_x INTEGER,                           -- Center X when track was created
    entry_y INTEGER,                           -- Center Y when track was created

    -- Exit position (where the track was last seen)
    exit_x INTEGER,                            -- Center X when track ended
    exit_y INTEGER,                            -- Center Y when track ended

    -- Travel metrics
    exit_direction TEXT,                        -- 'top', 'bottom', 'left', 'right', 'timeout'
    distance_pixels REAL,                      -- Total Euclidean distance traveled (pixels)
    duration_seconds REAL,                     -- Track lifetime in seconds
    total_frames INTEGER,                      -- Total frames the track existed

    -- Quality metrics
    avg_confidence REAL,                       -- Average detection confidence
    total_hits INTEGER,                        -- Frames where track was detected

    -- Classification outcome (NULL if not classified)
    classification TEXT,                       -- Final class name (NULL if skipped)
    classification_confidence REAL,            -- Classification confidence (NULL if skipped)

    -- Position history as JSON (for trajectory visualization)
    position_history TEXT                       -- JSON array of [x,y] points
);

-- Indexes for analytics queries
CREATE INDEX IF NOT EXISTS idx_track_events_event_type ON track_events(event_type);
CREATE INDEX IF NOT EXISTS idx_track_events_timestamp ON track_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_track_events_track_id ON track_events(track_id);

-- ============================================================================
-- Table: track_event_details
-- ============================================================================
-- Stores detailed lifecycle steps for each track event.
-- Each row represents one step: ROI collection, per-ROI classification, voting, etc.
-- Linked to track_events via track_id for full lifecycle reconstruction.
-- ============================================================================

CREATE TABLE IF NOT EXISTS track_event_details (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL,                 -- Track ID (links to track_events.track_id)
    timestamp TEXT NOT NULL,                    -- ISO 8601 timestamp of this step
    step_type TEXT NOT NULL,                    -- Step type (see below)
    -- step_type values:
    --   'roi_collected'     - ROI passed quality check and was collected
    --   'roi_rejected'      - ROI failed quality check
    --   'roi_classified'    - Individual ROI classification result
    --   'voting_result'     - Final voting/classification result
    --   'track_created'     - Track was first created
    --   'track_completed'   - Track exited frame normally
    --   'track_lost'        - Track lost (disappeared)
    --   'track_invalid'     - Track had invalid travel path

    -- Position data (for ROI steps)
    bbox_x1 INTEGER,                           -- ROI bounding box x1
    bbox_y1 INTEGER,                           -- ROI bounding box y1
    bbox_x2 INTEGER,                           -- ROI bounding box x2
    bbox_y2 INTEGER,                           -- ROI bounding box y2

    -- Quality data (for ROI steps)
    quality_score REAL,                        -- ROI quality score
    roi_index INTEGER,                         -- ROI index in collection (0-based)

    -- Classification data (for classification steps)
    class_name TEXT,                           -- Predicted class name
    confidence REAL,                           -- Classification confidence
    is_rejected INTEGER DEFAULT 0,             -- 1 if classified as Rejected

    -- Voting data (for voting_result step)
    vote_distribution TEXT,                    -- JSON: {"ClassName": score, ...}
    total_rois INTEGER,                        -- Total ROIs used for voting
    valid_votes INTEGER,                       -- Non-rejected votes

    -- Extra context
    detail TEXT                                -- Additional JSON context
);

-- Indexes for fast lookup
CREATE INDEX IF NOT EXISTS idx_track_event_details_track_id ON track_event_details(track_id);
CREATE INDEX IF NOT EXISTS idx_track_event_details_step_type ON track_event_details(step_type);
CREATE INDEX IF NOT EXISTS idx_track_event_details_timestamp ON track_event_details(timestamp);

-- ============================================================================
-- Table: config
-- ============================================================================
-- Stores system configuration as key-value pairs.
-- Used for runtime configuration updates (e.g., via ConfigWatcher).
-- ============================================================================

CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,                   -- Configuration key (e.g., "detection_confidence")
    value TEXT NOT NULL,                    -- Configuration value (as string)
    updated_at TEXT DEFAULT (datetime('now', 'utc'))  -- Last update timestamp (UTC)
);

-- Index for fast config lookups
CREATE INDEX IF NOT EXISTS idx_config_key ON config(key);

-- ============================================================================
-- Initial Data (Optional)
-- ============================================================================
-- Pre-populate with default bag types if needed.
-- Uncomment and modify as needed for your deployment.
-- ============================================================================

-- Example: Insert default bag types
-- INSERT OR IGNORE INTO bag_types (name, arabic_name, weight, thumb) VALUES
--     ('Wheatberry', 'توت القمح', 500, 'data/classes/Wheatberry/Wheatberry.jpg'),
--     ('Multigrain', 'متعدد الحبوب', 450, 'data/classes/Multigrain/Multigrain.jpg'),
--     ('WholeWheat', 'قمح كامل', 500, 'data/classes/WholeWheat/WholeWheat.jpg');

-- Example: Insert default config values
-- INSERT OR IGNORE INTO config (key, value) VALUES
--     ('detection_confidence', '0.5'),
--     ('classification_confidence', '0.7'),
--     ('system_enabled', 'true');

-- ============================================================================
-- Schema Validation Queries
-- ============================================================================
-- Use these queries to verify the schema is correctly initialized:
--
-- Check tables exist:
--   SELECT name FROM sqlite_master WHERE type='table';
--
-- Check indexes exist:
--   SELECT name FROM sqlite_master WHERE type='index';
--
-- Check foreign keys enabled:
--   PRAGMA foreign_keys;  -- Should return 1
--
-- Check bag_types structure:
--   PRAGMA table_info(bag_types);
--
-- Check events structure:
--   PRAGMA table_info(events);
--
-- Check config structure:
--   PRAGMA table_info(config);
-- ============================================================================

-- ============================================================================
-- Migration Notes (V1 -> V2)
-- ============================================================================
-- If migrating from V1 schema:
--
-- V1 had:
--   - bag_events (different schema)
--   - bag_types (similar)
--   - configs (renamed to config)
--
-- V2 changes:
--   1. Renamed bag_events -> events
--   2. Added bag_type_id foreign key
--   3. Changed timestamp format to ISO 8601
--   4. Added confidence constraint
--   5. Renamed configs -> config
--   6. Added indexes for performance
--   7. Enabled foreign key constraints
--
-- No automatic migration provided - deploy fresh for V2.
-- ============================================================================

-- ============================================================================
-- End of Schema
-- ============================================================================
