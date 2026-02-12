#!/bin/bash
# Quick deployment and restart script for RDK board
# Run this after syncing updated code

echo "================================================"
echo "Restarting Spool Recorder with Fixes"
echo "================================================"

# Stop existing spool recorder
echo "Stopping existing spool_recorder..."
pkill -f spool_recorder_node
sleep 1

# Clear old segments (optional - comment out if you want to keep old data)
# rm -f /tmp/spool/seg_*.bin
# rm -f /tmp/spool/seg_*.json

# Start spool recorder with logging
echo "Starting spool_recorder..."
cd ~/ConvuyerBreadCounting
python3 -m src.spool.spool_recorder_node > data/logs/spool-recorder.log 2>&1 &
RECORDER_PID=$!

echo "Spool recorder started with PID: $RECORDER_PID"
echo ""

# Wait a moment for startup
sleep 2

# Check if it's running
if ps -p $RECORDER_PID > /dev/null; then
    echo "✅ Spool recorder is running"
else
    echo "❌ Spool recorder failed to start"
    echo "Check logs: tail data/logs/spool-recorder.log"
    exit 1
fi

# Show recent log output
echo ""
echo "Recent log output:"
echo "------------------------------------------------"
tail -20 data/logs/spool-recorder.log
echo "------------------------------------------------"

# Check for errors
if grep -q "ERROR" data/logs/spool-recorder.log; then
    echo ""
    echo "⚠️  Errors detected in log!"
    echo "Review: tail -f data/logs/spool-recorder.log"
fi

# Check if segments are being created
echo ""
echo "Checking for segments (waiting 5 seconds)..."
sleep 5
SEGMENT_COUNT=$(ls /tmp/spool/seg_*.bin 2>/dev/null | wc -l)
echo "Segments found: $SEGMENT_COUNT"

if [ "$SEGMENT_COUNT" -gt 0 ]; then
    echo "✅ Spool recorder is writing segments!"
    ls -lh /tmp/spool/seg_*.bin | tail -5
else
    echo "⚠️  No segments yet - check if frames are being received"
    echo "Debug: tail -f data/logs/spool-recorder.log"
fi

echo ""
echo "================================================"
echo "Monitoring commands:"
echo "  tail -f data/logs/spool-recorder.log"
echo "  watch -n 1 'ls -lh /tmp/spool/*.bin | tail -5'"
echo "  ros2 topic info /rtsp_image_ch_0 -v"
echo "================================================"
