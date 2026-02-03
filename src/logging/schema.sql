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
