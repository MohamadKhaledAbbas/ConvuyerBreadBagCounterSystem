#!/bin/bash
set -e

LOGFILE="/media/SSD_DRIVE/ConvuyerBreadCounting/data/logs/*.log"
STATUSFILE="/tmp/codec_health_status.json"
CAMERA_IP="192.168.2.108"   # ⚠️ change if needed

PASS_COUNT=0
FAIL_COUNT=0

log() {
    echo -e "\n==== $1 ===="
}

pass() {
    echo "✓ PASS: $1"
    PASS_COUNT=$((PASS_COUNT+1))
}

fail() {
    echo "✗ FAIL: $1"
    FAIL_COUNT=$((FAIL_COUNT+1))
}

wait_for_detection() {
    sleep 4
    if grep -q "CRITICAL: triggering recovery" $LOGFILE; then
        pass "Recovery triggered"
    else
        fail "Recovery NOT triggered"
    fi
}

wait_for_frames() {
    sleep 4
    if timeout 3 ros2 topic hz /nv12_images | grep -q "average rate"; then
        pass "Frames resumed"
    else
        fail "Frames did NOT resume"
    fi
}

clear_logs() {
    > $LOGFILE
}

# ===============================
# TEST 1: Kill codec
# ===============================
log "TEST 1: Kill hobot_codec"
clear_logs

START=$(date +%s.%N)
sudo pkill -f hobot_codec || true

wait_for_detection
wait_for_frames

END=$(date +%s.%N)
echo "Detection+Recovery Time: $(echo "$END - $START" | bc)s"

# ===============================
# TEST 2: Freeze codec (REAL stall)
# ===============================
log "TEST 2: Freeze hobot_codec (SIGSTOP)"
clear_logs

PID=$(pgrep -f hobot_codec | head -n1)
if [ -z "$PID" ]; then
    fail "Codec not running"
else
    sudo kill -STOP $PID
    wait_for_detection

    sudo kill -CONT $PID || true
    wait_for_frames
fi

# ===============================
# TEST 3: Kill RTSP client
# ===============================
log "TEST 3: Kill RTSP client"
clear_logs

sudo pkill -f hobot_rtsp_client || true

wait_for_detection
wait_for_frames

# ===============================
# TEST 4: Block camera network
# ===============================
log "TEST 4: Block RTSP network"
clear_logs

sudo iptables -A INPUT -s $CAMERA_IP -j DROP

sleep 5
wait_for_detection

sudo iptables -D INPUT -s $CAMERA_IP -j DROP
sleep 5

wait_for_frames

# ===============================
# TEST 5: Rapid repeated stalls
# ===============================
log "TEST 5: Rapid repeated stalls"
clear_logs

for i in 1 2 3; do
    echo "Iteration $i"
    sudo pkill -f hobot_codec || true
    sleep 3
done

wait_for_detection
wait_for_frames

# ===============================
# TEST 6: Verify no false positives
# ===============================
log "TEST 6: No false positives"
clear_logs

sleep 5

if grep -q "CRITICAL" $LOGFILE; then
    fail "False positive detected"
else
    pass "No false positives"
fi

# ===============================
# SUMMARY
# ===============================
log "FINAL RESULTS"

echo "PASSED: $PASS_COUNT"
echo "FAILED: $FAIL_COUNT"

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "🎉 ALL TESTS PASSED — SYSTEM IS STABLE"
else
    echo "⚠️ SOME TESTS FAILED — INVESTIGATE BEFORE PRODUCTION"
fi