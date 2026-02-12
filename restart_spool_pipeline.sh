#!/bin/bash
# Complete spool pipeline restart script
# Restarts both recorder and processor with fixes

echo "================================================"
echo "Restarting Complete Spool Pipeline"
echo "================================================"

# Stop existing processes
echo "Stopping existing spool processes..."
pkill -f spool_recorder_node
pkill -f spool_processor_node
sleep 2

# Start spool recorder
echo ""
echo "Starting spool_recorder..."
cd ~/ConvuyerBreadCounting
python3 -m src.spool.spool_recorder_node > data/logs/spool-recorder.log 2>&1 &
RECORDER_PID=$!
echo "Spool recorder started with PID: $RECORDER_PID"

# Wait for recorder to start
sleep 2

# Start spool processor
echo ""
echo "Starting spool_processor..."
python3 -m src.spool.spool_processor_node > data/logs/spool-processor.log 2>&1 &
PROCESSOR_PID=$!
echo "Spool processor started with PID: $PROCESSOR_PID"

# Wait for startup
sleep 3

# Check if both are running
echo ""
echo "Checking process status..."
if ps -p $RECORDER_PID > /dev/null; then
    echo "✅ Spool recorder is running (PID: $RECORDER_PID)"
else
    echo "❌ Spool recorder failed"
fi

if ps -p $PROCESSOR_PID > /dev/null; then
    echo "✅ Spool processor is running (PID: $PROCESSOR_PID)"
else
    echo "❌ Spool processor failed"
fi

# Show recent logs
echo ""
echo "================================================"
echo "Recent Spool Recorder Logs:"
echo "================================================"
tail -10 data/logs/spool-recorder.log

echo ""
echo "================================================"
echo "Recent Spool Processor Logs:"
echo "================================================"
tail -10 data/logs/spool-processor.log

# Check for errors
echo ""
echo "================================================"
echo "Error Check:"
echo "================================================"
RECORDER_ERRORS=$(grep -c "ERROR" data/logs/spool-recorder.log 2>/dev/null || echo 0)
PROCESSOR_ERRORS=$(grep -c "ERROR" data/logs/spool-processor.log 2>/dev/null || echo 0)

if [ "$RECORDER_ERRORS" -gt 0 ]; then
    echo "⚠️  Recorder has $RECORDER_ERRORS errors - check: tail -f data/logs/spool-recorder.log"
else
    echo "✅ Recorder running clean"
fi

if [ "$PROCESSOR_ERRORS" -gt 0 ]; then
    echo "⚠️  Processor has $PROCESSOR_ERRORS errors - check: tail -f data/logs/spool-processor.log"
else
    echo "✅ Processor running clean"
fi

# Check segments and topics
echo ""
echo "================================================"
echo "Pipeline Status:"
echo "================================================"
sleep 2
SEGMENTS=$(ls /tmp/spool/seg_*.bin 2>/dev/null | wc -l)
echo "Segments on disk: $SEGMENTS"

echo ""
echo "Checking ROS2 topics..."
echo "  /rtsp_image_ch_0:    $(ros2 topic hz /rtsp_image_ch_0 --once 2>&1 | grep -o '[0-9.]*' | head -1) fps"
echo "  /spool_image_ch_0:   $(ros2 topic hz /spool_image_ch_0 --once 2>&1 | grep -o '[0-9.]*' | head -1) fps"
echo "  /nv12_images:        $(ros2 topic hz /nv12_images --once 2>&1 | grep -o '[0-9.]*' | head -1) fps"

echo ""
echo "================================================"
echo "Monitoring Commands:"
echo "  tail -f data/logs/spool-recorder.log"
echo "  tail -f data/logs/spool-processor.log"
echo "  watch -n 1 'ls -lh /tmp/spool/*.bin | tail -5'"
echo "  ros2 topic hz /spool_image_ch_0"
echo "================================================"
