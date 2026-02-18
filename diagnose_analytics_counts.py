"""
Diagnostic script to compare analytics/daily counts vs raw SQL.

Run this script on the production system to debug count mismatches.

Usage:
    python diagnose_analytics_counts.py
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

# Configuration (match your production config)
DB_PATH = "data/db/bag_events.db"
TIMEZONE_OFFSET_HOURS = 3
SHIFT_START_HOUR = 16
SHIFT_END_HOUR = 14


def calculate_shift_times():
    """Calculate shift times the same way analytics_service.py does (FIXED)."""
    # FIXED: Use local time directly, no offset needed when system is in local time
    time_now = datetime.now()
    print(f"[DEBUG] datetime.now() = {time_now}")

    if time_now.hour >= SHIFT_START_HOUR:
        start_time = time_now
        end_time = time_now + timedelta(days=1)
    else:
        start_time = time_now - timedelta(days=1)
        end_time = time_now

    start_time = start_time.replace(hour=SHIFT_START_HOUR, minute=0, second=0, microsecond=0)
    end_time = end_time.replace(hour=SHIFT_END_HOUR, minute=0, second=0, microsecond=0)

    print(f"[DEBUG] Shift start = {start_time}")
    print(f"[DEBUG] Shift end   = {end_time}")

    return start_time, end_time


def convert_to_db_time(local_time):
    """Convert local time to DB query time (FIXED: no conversion needed)."""
    # FIXED: Events are stored in local time, no conversion needed
    return local_time


def main():
    print("=" * 70)
    print("Analytics Count Diagnostic Tool")
    print("=" * 70)

    # Check database exists
    db_path = Path(DB_PATH)
    if not db_path.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        return

    # Calculate shift times
    print("\n[1] Calculating shift times...")
    shift_start, shift_end = calculate_shift_times()

    # Convert to DB times
    db_start = convert_to_db_time(shift_start)
    db_end = convert_to_db_time(shift_end)

    print(f"\n[2] DB query times (after -{TIMEZONE_OFFSET_HOURS}h adjustment):")
    print(f"    db_start = {db_start.isoformat()}")
    print(f"    db_end   = {db_end.isoformat()}")

    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Query 1: Raw count without time filter
    print("\n[3] Raw counts (no time filter):")
    cursor.execute("SELECT COUNT(*) as cnt FROM events")
    total_all = cursor.fetchone()['cnt']
    print(f"    Total events in DB: {total_all}")

    # Query 2: Count with time filter (what analytics endpoint does)
    print(f"\n[4] Filtered count (analytics endpoint logic):")
    cursor.execute(
        "SELECT COUNT(*) as cnt FROM events WHERE timestamp >= ? AND timestamp <= ?",
        (db_start.isoformat(), db_end.isoformat())
    )
    filtered_count = cursor.fetchone()['cnt']
    print(f"    Count with filter: {filtered_count}")

    # Query 3: Show sample timestamps from DB
    print("\n[5] Sample event timestamps from DB:")
    cursor.execute("SELECT timestamp FROM events ORDER BY timestamp DESC LIMIT 5")
    rows = cursor.fetchall()
    for row in rows:
        print(f"    {row['timestamp']}")

    # Query 4: Check timestamp boundaries
    cursor.execute("SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts FROM events")
    bounds = cursor.fetchone()
    print(f"\n[6] Event timestamp boundaries:")
    print(f"    Oldest event: {bounds['min_ts']}")
    print(f"    Newest event: {bounds['max_ts']}")

    # Query 5: Count by day (raw)
    print("\n[7] Events per day (raw, no timezone adjustment):")
    cursor.execute("""
        SELECT DATE(timestamp) as day, COUNT(*) as cnt 
        FROM events 
        GROUP BY DATE(timestamp) 
        ORDER BY day DESC 
        LIMIT 7
    """)
    for row in cursor.fetchall():
        print(f"    {row['day']}: {row['cnt']} events")

    # Query 6: Try different time ranges to find the mismatch
    print("\n[8] Testing different query ranges:")

    # Without timezone adjustment (treat shift times as DB times)
    cursor.execute(
        "SELECT COUNT(*) as cnt FROM events WHERE timestamp >= ? AND timestamp <= ?",
        (shift_start.isoformat(), shift_end.isoformat())
    )
    no_adjust_count = cursor.fetchone()['cnt']
    print(f"    Without TZ adjustment: {no_adjust_count} (query: {shift_start.isoformat()} to {shift_end.isoformat()})")

    # With timezone adjustment (what analytics does)
    print(f"    With TZ adjustment:    {filtered_count} (query: {db_start.isoformat()} to {db_end.isoformat()})")

    conn.close()

    print("\n" + "=" * 70)
    print("DIAGNOSIS:")
    if no_adjust_count != filtered_count:
        print(f"  Timezone adjustment is causing a difference of {abs(no_adjust_count - filtered_count)} events.")
        print(f"  Check if events are stored in UTC or local time.")
        print(f"  Check if your system time matches the expected timezone.")
    else:
        print("  No difference detected between adjusted and non-adjusted queries.")
        print("  The issue may be in how COUNT(*) is being run separately.")
    print("=" * 70)


if __name__ == "__main__":
    main()
