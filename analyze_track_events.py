#!/usr/bin/env python3
"""
Track Event Analytics Visualizer.

Standalone script to query and visualize track event data from the database.
Helps diagnose tracking issues by showing:

1. Summary statistics (counts by event type, avg duration, etc.)
2. Trajectory plot (all track paths on the frame)
3. Entry/exit heatmaps (where tracks appear and disappear)
4. Duration/distance distributions
5. Timeline of events

Usage:
    python analyze_track_events.py                      # Use default DB path
    python analyze_track_events.py --db path/to/db      # Custom DB path
    python analyze_track_events.py --type track_invalid  # Filter by event type
    python analyze_track_events.py --export report.png   # Export plots to file
    python analyze_track_events.py --list                # Print events as table
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.logging.Database import DatabaseManager

# Try to import matplotlib (optional - falls back to text output)
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server/headless
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import LineCollection
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# Frame dimensions
FRAME_W = 1280
FRAME_H = 720

# Color scheme for event types
EVENT_COLORS = {
    'track_completed': '#2ecc71',   # Green
    'track_lost': '#e74c3c',        # Red
    'track_invalid': '#f39c12',     # Orange
}

EVENT_LABELS = {
    'track_completed': 'Completed (valid)',
    'track_lost': 'Lost (disappeared)',
    'track_invalid': 'Invalid (bad path)',
}


def print_summary(events: List[Dict[str, Any]]):
    """Print text summary of track events."""
    if not events:
        print("No track events found.")
        return

    print("=" * 70)
    print("TRACK EVENT ANALYTICS SUMMARY")
    print("=" * 70)
    print(f"Total events: {len(events)}")
    print()

    # Count by type
    by_type = {}
    for e in events:
        t = e['event_type']
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(e)

    print("Events by Type:")
    print("-" * 50)
    for event_type, group in sorted(by_type.items()):
        pct = len(group) / len(events) * 100
        label = EVENT_LABELS.get(event_type, event_type)
        print(f"  {label:30s}  {len(group):5d}  ({pct:5.1f}%)")

    print()

    # Duration stats
    durations = [e['duration_seconds'] for e in events if e.get('duration_seconds') is not None]
    if durations:
        print("Duration (seconds):")
        print(f"  Min:  {min(durations):8.2f}")
        print(f"  Max:  {max(durations):8.2f}")
        print(f"  Mean: {sum(durations)/len(durations):8.2f}")
        print()

    # Distance stats
    distances = [e['distance_pixels'] for e in events if e.get('distance_pixels') is not None]
    if distances:
        print("Distance (pixels):")
        print(f"  Min:  {min(distances):8.1f}")
        print(f"  Max:  {max(distances):8.1f}")
        print(f"  Mean: {sum(distances)/len(distances):8.1f}")
        print()

    # Exit direction breakdown
    exit_dirs = {}
    for e in events:
        d = e.get('exit_direction', 'unknown')
        exit_dirs[d] = exit_dirs.get(d, 0) + 1

    print("Exit Directions:")
    print("-" * 50)
    for direction, count in sorted(exit_dirs.items(), key=lambda x: -x[1]):
        pct = count / len(events) * 100
        print(f"  {direction:15s}  {count:5d}  ({pct:5.1f}%)")
    print()

    # Classification stats (for completed tracks)
    classified = [e for e in events if e.get('classification') is not None]
    if classified:
        class_counts = {}
        for e in classified:
            c = e['classification']
            class_counts[c] = class_counts.get(c, 0) + 1

        print("Classifications (completed tracks):")
        print("-" * 50)
        for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            pct = count / len(classified) * 100
            print(f"  {cls:25s}  {count:5d}  ({pct:5.1f}%)")
        print()

    # Time range
    timestamps = [e['timestamp'] for e in events if e.get('timestamp')]
    if timestamps:
        print(f"Time range: {min(timestamps)} → {max(timestamps)}")

    print("=" * 70)


def print_event_table(events: List[Dict[str, Any]], limit: int = 50):
    """Print events as a table."""
    if not events:
        print("No track events found.")
        return

    shown = events[:limit]

    # Header
    print(f"{'ID':>5} {'Track':>6} {'Type':>16} {'Entry':>10} {'Exit':>10} "
          f"{'Dir':>8} {'Dist':>8} {'Dur(s)':>7} {'Conf':>6} {'Class':>15} {'Timestamp':>20}")
    print("-" * 130)

    for e in shown:
        entry = f"({e.get('entry_x', '?')},{e.get('entry_y', '?')})"
        exit_pos = f"({e.get('exit_x', '?')},{e.get('exit_y', '?')})"
        dist = f"{e['distance_pixels']:.0f}" if e.get('distance_pixels') is not None else "?"
        dur = f"{e['duration_seconds']:.2f}" if e.get('duration_seconds') is not None else "?"
        conf = f"{e['avg_confidence']:.2f}" if e.get('avg_confidence') is not None else "?"
        cls = e.get('classification') or '-'
        ts = e.get('timestamp', '?')[:19]

        print(f"{e['id']:>5} {e['track_id']:>6} {e['event_type']:>16} {entry:>10} {exit_pos:>10} "
              f"{e.get('exit_direction', '?'):>8} {dist:>8} {dur:>7} {conf:>6} {cls:>15} {ts:>20}")

    if len(events) > limit:
        print(f"... and {len(events) - limit} more events")


def plot_trajectories(events: List[Dict[str, Any]], output_path: Optional[str] = None):
    """Plot all track trajectories on a frame-sized canvas."""
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        print("matplotlib and numpy required for plotting. Install with: pip install matplotlib numpy")
        return

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, FRAME_W)
    ax.set_ylim(FRAME_H, 0)  # Invert Y axis (image coordinates)
    ax.set_aspect('equal')
    ax.set_title('Track Trajectories (all events)', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')

    # Draw frame border
    ax.add_patch(plt.Rectangle((0, 0), FRAME_W, FRAME_H, fill=False,
                                edgecolor='gray', linewidth=2, linestyle='--'))

    # Draw entry/exit zones
    entry_zone_y = FRAME_H * 0.75  # Bottom 25%
    exit_zone_y = FRAME_H * 0.15   # Top 15%

    ax.axhspan(entry_zone_y, FRAME_H, alpha=0.1, color='blue', label='Entry zone (bottom 25%)')
    ax.axhspan(0, exit_zone_y, alpha=0.1, color='green', label='Exit zone (top 15%)')

    plotted = 0
    for e in events:
        if not e.get('position_history'):
            continue

        try:
            positions = json.loads(e['position_history'])
        except (json.JSONDecodeError, TypeError):
            continue

        if len(positions) < 2:
            continue

        color = EVENT_COLORS.get(e['event_type'], '#95a5a6')
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        # Plot trajectory line
        ax.plot(xs, ys, color=color, alpha=0.4, linewidth=1.0)

        # Mark start and end points
        ax.plot(xs[0], ys[0], 'o', color=color, markersize=4, alpha=0.6)
        ax.plot(xs[-1], ys[-1], 's', color=color, markersize=4, alpha=0.6)
        plotted += 1

    # Legend
    patches = []
    for event_type, color in EVENT_COLORS.items():
        label = EVENT_LABELS.get(event_type, event_type)
        patches.append(mpatches.Patch(color=color, label=label))
    patches.append(mpatches.Patch(color='blue', alpha=0.2, label='Entry zone'))
    patches.append(mpatches.Patch(color='green', alpha=0.2, label='Exit zone'))
    ax.legend(handles=patches, loc='upper left', fontsize=9)

    ax.set_title(f'Track Trajectories ({plotted} tracks plotted)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Trajectory plot saved to: {output_path}")
    else:
        fig.savefig('track_trajectories.png', dpi=150, bbox_inches='tight')
        print("Trajectory plot saved to: track_trajectories.png")
    plt.close(fig)


def plot_heatmaps(events: List[Dict[str, Any]], output_path: Optional[str] = None):
    """Plot entry and exit position heatmaps."""
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        print("matplotlib and numpy required for plotting. Install with: pip install matplotlib numpy")
        return

    entry_points = []
    exit_points = []

    for e in events:
        if e.get('entry_x') is not None and e.get('entry_y') is not None:
            entry_points.append((e['entry_x'], e['entry_y']))
        if e.get('exit_x') is not None and e.get('exit_y') is not None:
            exit_points.append((e['exit_x'], e['exit_y']))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for idx, (points, title) in enumerate([
        (entry_points, 'Entry Positions (where tracks appear)'),
        (exit_points, 'Exit Positions (where tracks disappear)')
    ]):
        ax = axes[idx]
        ax.set_xlim(0, FRAME_W)
        ax.set_ylim(FRAME_H, 0)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')

        if not points:
            ax.text(FRAME_W/2, FRAME_H/2, 'No data', ha='center', va='center', fontsize=14)
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        # Scatter plot with density coloring
        hb = ax.hexbin(xs, ys, gridsize=20, cmap='YlOrRd', mincnt=1,
                        extent=[0, FRAME_W, 0, FRAME_H])
        plt.colorbar(hb, ax=ax, label='Count')

        # Also show individual points
        ax.scatter(xs, ys, s=8, color='blue', alpha=0.3, zorder=5)

    plt.tight_layout()
    out = output_path or 'track_heatmaps.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Heatmap plot saved to: {out}")
    plt.close(fig)


def plot_distributions(events: List[Dict[str, Any]], output_path: Optional[str] = None):
    """Plot duration and distance distributions by event type."""
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        print("matplotlib and numpy required for plotting. Install with: pip install matplotlib numpy")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Group by event type
    by_type = {}
    for e in events:
        t = e['event_type']
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(e)

    # 1. Duration histogram
    ax = axes[0, 0]
    for event_type, group in sorted(by_type.items()):
        durations = [e['duration_seconds'] for e in group if e.get('duration_seconds') is not None]
        if durations:
            color = EVENT_COLORS.get(event_type, '#95a5a6')
            label = EVENT_LABELS.get(event_type, event_type)
            ax.hist(durations, bins=20, alpha=0.6, color=color, label=label)
    ax.set_title('Duration Distribution', fontweight='bold')
    ax.set_xlabel('Duration (seconds)')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)

    # 2. Distance histogram
    ax = axes[0, 1]
    for event_type, group in sorted(by_type.items()):
        distances = [e['distance_pixels'] for e in group if e.get('distance_pixels') is not None]
        if distances:
            color = EVENT_COLORS.get(event_type, '#95a5a6')
            label = EVENT_LABELS.get(event_type, event_type)
            ax.hist(distances, bins=20, alpha=0.6, color=color, label=label)
    ax.set_title('Distance Distribution', fontweight='bold')
    ax.set_xlabel('Distance (pixels)')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)

    # 3. Event type pie chart
    ax = axes[1, 0]
    labels = []
    sizes = []
    colors = []
    for event_type, group in sorted(by_type.items()):
        labels.append(EVENT_LABELS.get(event_type, event_type))
        sizes.append(len(group))
        colors.append(EVENT_COLORS.get(event_type, '#95a5a6'))

    if sizes:
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Event Type Breakdown', fontweight='bold')

    # 4. Confidence distribution
    ax = axes[1, 1]
    for event_type, group in sorted(by_type.items()):
        confs = [e['avg_confidence'] for e in group if e.get('avg_confidence') is not None]
        if confs:
            color = EVENT_COLORS.get(event_type, '#95a5a6')
            label = EVENT_LABELS.get(event_type, event_type)
            ax.hist(confs, bins=20, alpha=0.6, color=color, label=label, range=(0, 1))
    ax.set_title('Confidence Distribution', fontweight='bold')
    ax.set_xlabel('Average Confidence')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = output_path or 'track_distributions.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Distribution plot saved to: {out}")
    plt.close(fig)


def plot_all(events: List[Dict[str, Any]], output_prefix: str = 'track_analytics'):
    """Generate all plots and save them."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available. Install with: pip install matplotlib")
        print("Falling back to text output only.")
        return

    plot_trajectories(events, f'{output_prefix}_trajectories.png')
    plot_heatmaps(events, f'{output_prefix}_heatmaps.png')
    plot_distributions(events, f'{output_prefix}_distributions.png')
    print()
    print(f"All plots saved with prefix: {output_prefix}_*.png")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze track events from the conveyor counter database.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_track_events.py                          # Summary + plots
  python analyze_track_events.py --db data/counter.db     # Custom DB path
  python analyze_track_events.py --type track_invalid     # Filter invalid tracks
  python analyze_track_events.py --list                   # Print event table
  python analyze_track_events.py --export report          # Export as report_*.png
  python analyze_track_events.py --start 2026-02-10       # Filter by date
        """
    )
    parser.add_argument('--db', default='data/conveyor_counter.db',
                        help='Path to SQLite database (default: data/conveyor_counter.db)')
    parser.add_argument('--type', choices=['track_completed', 'track_lost', 'track_invalid'],
                        help='Filter by event type')
    parser.add_argument('--start', help='Start date filter (ISO 8601)')
    parser.add_argument('--end', help='End date filter (ISO 8601)')
    parser.add_argument('--limit', type=int, default=10000,
                        help='Maximum events to load (default: 10000)')
    parser.add_argument('--list', action='store_true',
                        help='Print events as table')
    parser.add_argument('--export', default=None,
                        help='Export plots with this filename prefix')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip plot generation (text summary only)')

    args = parser.parse_args()

    # Check database exists
    if not os.path.exists(args.db):
        print(f"Database not found: {args.db}")
        print("Make sure the conveyor counter has been run at least once to create the database.")
        return 1

    # Connect and query
    db = DatabaseManager(args.db)

    try:
        events = db.get_track_events(
            event_type=args.type,
            start_date=args.start,
            end_date=args.end,
            limit=args.limit
        )

        # Print summary
        print_summary(events)

        # Print table if requested
        if args.list:
            print()
            print_event_table(events)

        # Generate plots
        if not args.no_plots and events:
            print()
            prefix = args.export or 'track_analytics'
            plot_all(events, prefix)

    finally:
        db.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())
