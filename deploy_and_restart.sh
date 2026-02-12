#!/bin/bash
# Full deployment and restart script for RDK board
# Run this after syncing code from Windows

set -e

echo "================================================"
echo "ConvuyerBreadCounting - Full Restart"
echo "================================================"

cd ~/ConvuyerBreadCounting

# Stop all running processes
echo ""
echo "Stopping existing processes..."
pkill -f main.py || true
pkill -f spool_processor_node || true
pkill -f spool_recorder_node || true
sleep 2

# Clear old segments (optional - uncomment if needed)
# echo "Clearing old segments..."
# rm -f /tmp/spool/seg_*.bin /tmp/spool/seg_*.json

# Start spool recorder
echo ""
echo "Starting spool recorder..."
python3 -m src.spool.spool_recorder_node > data/logs/spool-recorder.log 2>&1 &
RECORDER_PID=$!
echo "  Recorder PID: $RECORDER_PID"
sleep 2

# Start spool processor
echo ""
echo "Starting spool processor..."
python3 -m src.spool.spool_processor_node > data/logs/spool-processor.log 2>&1 &
PROCESSOR_PID=$!
echo "  Processor PID: $PROCESSOR_PID"
sleep 2

# Start main app
echo ""
echo "Starting main application..."
LOG_FILE="data/logs/convuyer_counter_$(date +%Y%m%d_%H%M%S).log"
python3 main.py > "$LOG_FILE" 2>&1 &
MAIN_PID=$!
echo "  Main app PID: $MAIN_PID"
echo "  Log file: $LOG_FILE"
sleep 3

# Check status
echo ""
echo "================================================"
echo "Process Status"
echo "================================================"

check_process() {
    if ps -p $1 > /dev/null 2>&1; then
        echo "  ✅ $2 is running (PID: $1)"
        return 0
    else
        echo "  ❌ $2 failed to start"
        return 1
    fi
}

check_process $RECORDER_PID "Spool Recorder"
check_process $PROCESSOR_PID "Spool Processor"
check_process $MAIN_PID "Main Application"

# Show recent logs
echo ""
echo "================================================"
echo "Recent Main App Logs:"
echo "================================================"
sleep 2
tail -20 "$LOG_FILE" || echo "No logs yet"

# Monitor commands
echo ""
echo "================================================"
echo "Monitoring Commands:"
echo "================================================"
echo "  tail -f $LOG_FILE"
echo "  tail -f data/logs/spool-processor.log"
echo "  ros2 topic hz /nv12_images"
echo "  watch -n 1 'ls /tmp/spool/*.bin | wc -l'"
echo "================================================"
