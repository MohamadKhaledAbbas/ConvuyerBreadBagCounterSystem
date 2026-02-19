import sqlite3
import json

output_lines = []

conn = sqlite3.connect('data/db/bag_events.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# Check table exists
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cur.fetchall()
output_lines.append(f"Tables: {[t[0] for t in tables]}")

# Check track_events
cur.execute('SELECT COUNT(*) FROM track_events')
count = cur.fetchone()[0]
output_lines.append(f"Total track_events: {count}")

# Get min/max coordinates
cur.execute('''
SELECT 
    MIN(entry_x) as min_entry_x, MAX(entry_x) as max_entry_x,
    MIN(entry_y) as min_entry_y, MAX(entry_y) as max_entry_y,
    MIN(exit_x) as min_exit_x, MAX(exit_x) as max_exit_x,
    MIN(exit_y) as min_exit_y, MAX(exit_y) as max_exit_y
FROM track_events
WHERE entry_x IS NOT NULL
''')
stats = cur.fetchone()
output_lines.append(f"\nCoordinate ranges:")
output_lines.append(f"  Entry X: {stats['min_entry_x']} - {stats['max_entry_x']}")
output_lines.append(f"  Entry Y: {stats['min_entry_y']} - {stats['max_entry_y']}")
output_lines.append(f"  Exit X:  {stats['min_exit_x']} - {stats['max_exit_x']}")
output_lines.append(f"  Exit Y:  {stats['min_exit_y']} - {stats['max_exit_y']}")

# Get sample data
cur.execute('SELECT track_id, entry_x, entry_y, exit_x, exit_y, position_history FROM track_events ORDER BY id DESC LIMIT 5')
rows = cur.fetchall()
output_lines.append(f"\nLast 5 track events:")
for r in rows:
    pos_hist = r['position_history']
    if pos_hist:
        try:
            positions = json.loads(pos_hist)
            first = positions[0] if positions else None
            last = positions[-1] if positions else None
            output_lines.append(f"  T{r['track_id']}: DB entry=({r['entry_x']}, {r['entry_y']}), DB exit=({r['exit_x']}, {r['exit_y']})")
            output_lines.append(f"           pos_hist: {len(positions)} points, first={first}, last={last}")
        except:
            output_lines.append(f"  T{r['track_id']}: entry=({r['entry_x']}, {r['entry_y']}), exit=({r['exit_x']}, {r['exit_y']}), pos_hist=PARSE_ERROR")
    else:
        output_lines.append(f"  T{r['track_id']}: entry=({r['entry_x']}, {r['entry_y']}), exit=({r['exit_x']}, {r['exit_y']}), pos_hist=NULL")

conn.close()

# Write to file
with open('db_check_output.txt', 'w') as f:
    f.write('\n'.join(output_lines))

print("Output written to db_check_output.txt")
