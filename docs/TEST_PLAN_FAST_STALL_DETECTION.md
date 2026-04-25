# Test Plan: Fast Stall Detection & Recovery

## Objective
Validate that stall detection now occurs in ~2-3 seconds (down from ~30s) and that recovery is deterministic.

---

## Prerequisites

1. SSH into RDK:
   ```bash
   ssh sunrise@192.168.2.111
   ```

2. Activate environment:
   ```bash
   cd /home/sunrise/ConvuyerBreadCounting
   source /opt/tros/humble/setup.bash
   source .venv/bin/activate
   ```

3. Ensure services are running:
   ```bash
   sudo supervisorctl status
   ```
   Should show all services running.

---

## Test 1: Verify Startup Logging

**Purpose**: Confirm monitor starts with correct fast configuration.

**Steps**:
```bash
# View logs
tail -f /home/sunrise/ConvuyerBreadCounting/data/logs/app.log | grep -i "codec.*health\|health.*monitor"
```

**Expected output**:
```
[CodecHealthMonitor] Started | topic=/nv12_images | timeout=2.5s | interval=2.0s | threshold=1
[ConveyorCounterApp] Codec health monitor started
```

---

## Test 2: Simulate Stall (Kill hobot_codec)

**Purpose**: Verify stall is detected in ≤ 3 seconds and recovery triggers.

**Steps**:

1. **Terminal 1 - Monitor logs**:
   ```bash
   tail -f /home/sunrise/ConvuyerBreadCounting/data/logs/app.log | grep -i "monitor\|codec"
   ```

2. **Terminal 2 - Kill codec via supervisor** (NOT pkill - supervisor is safer):
   ```bash
   sudo supervisorctl restart breadcount-ros2
   ```

3. **Observe** - You should see within 2-5 seconds:
   ```
   [CodecHealthMonitor] No frames for X.Xs (timeout=2.5s)
   [CodecHealthMonitor] CRITICAL: triggering recovery
   [CodecHealthMonitor] Stage 1: Restarting breadcount-ros2 via supervisor
   [CodecHealthMonitor] breadcount-ros2 restarted successfully via supervisor
   [CodecHealthMonitor] Recovery stage CODEC_ONLY succeeded
   [CodecHealthMonitor] Recovery succeeded, waiting 10.0s for pipeline to reinitialize...
   ```

**Pass Criteria**:
- Stall detected in ≤ 5 seconds (2.5s timeout + ~1-2 check cycles)
- Recovery triggered automatically
- **Pipeline resumes and frames flow again** (key difference from previous attempts)
- 10s cooldown allows ROS2/DDS to stabilize

---

## Test 3: Verify Timestamp-Based Detection (No Subprocess)

**Purpose**: Confirm we're using timestamp-based detection, not `ros2 topic echo`.

**Steps**:

1. While pipeline is running normally:
   ```bash
   # Check that ros2 topic echo is NOT being called by the monitor
   ps aux | grep "ros2 topic echo"
   ```
   Should show nothing (except maybe manual grep).

2. Check monitor stats file:
   ```bash
   cat /tmp/codec_health_status.json
   ```
   Should show `state: "healthy"` and `checks_healthy` incrementing.

**Pass Criteria**:
- No `ros2 topic echo` subprocess spawned by monitor
- `last_frame_time` in stats is recent (within 2-3 seconds)

---

## Test 4: Rapid Stall Test (Multiple Cycles)

**Purpose**: Verify rate limiting prevents crash loops.

**Steps**:
```bash
# Trigger recovery 3 times
sudo supervisorctl restart breadcount-ros2
sleep 12  # Wait for cooldown
sudo supervisorctl restart breadcount-ros2
sleep 12
sudo supervisorctl restart breadcount-ros2
```

**Observe logs** - Should see recovery each time, with proper cooldown periods.

**Pass Criteria**:
- Each stall is detected and recovered
- No crash loops
- `restarts_this_hour` increments correctly

---

## Test 5: Stage 2 Escalation Test

**Purpose**: If Stage 1 restart fails repeatedly, verify Stage 2 restarts both services.

**Steps**:
```bash
# Simulate repeated failures by repeatedly triggering recovery
# The escalation will happen after first stage fails

# First trigger normal recovery
sudo supervisorctl restart breadcount-ros2
sleep 15

# If you want to force Stage 2, manually break the service so Stage 1 can't fix it
# Then trigger another recovery - it should escalate to Stage 2
```

**Pass Criteria**:
- Stage 2 restarts `breadcount-ros2 breadcount-container-ros2` when Stage 1 fails

---

## Test 6: Frame Processing Verification

**Purpose**: Confirm frames are still being processed after recovery.

**Steps**:
```bash
# Check the health status file
watch -n 1 'cat /tmp/codec_health_status.json | python3 -m json.tool'
```

**Look for**:
- `state: "healthy"`
- `checks_healthy` incrementing every ~2 seconds
- `last_frame_time` updating

---

## Test 7: Compare Before/After Timing

**Purpose**: Quantify improvement from ~30s to ~3s detection.

**Steps**:
```bash
# Record timestamp when you restart the service
date +%s.%N
sudo supervisorctl restart breadcount-ros2

# Find the "No frames for" message and calculate delta
grep "No frames for" /home/sunrise/ConvuyerBreadCounting/data/logs/app.log | tail -1
```

**Pass Criteria**:
- Delta should be ≤ 5 seconds (2.5s timeout + up to 2s check interval)
- Detection should be MUCH faster than the old 30+ second approach

---

## Automated Test Script

Save this as `test_stall_detection.sh` and run it:

```bash
#!/bin/bash
set -e

LOGFILE="/home/sunrise/ConvuyerBreadCounting/data/logs/app.log"
STATUSFILE="/tmp/codec_health_status.json"

echo "=== Test: Fast Stall Detection ==="

# Clear recent logs
tail -c 1000000 $LOGFILE > ${LOGFILE}.tmp && mv ${LOGFILE}.tmp $LOGFILE || true

echo "[Test] Restarting breadcount-ros2 via supervisor..."
sudo supervisorctl restart breadcount-ros2

echo "[Test] Waiting for stall detection (expect ~2-3s)..."
sleep 8

echo "[Test] Checking logs for stall detection..."
if grep -q "No frames for" $LOGFILE; then
    echo "✓ PASS: Stall detected"
    grep "No frames for" $LOGFILE | tail -1
else
    echo "✗ FAIL: Stall not detected"
    exit 1
fi

if grep -q "CRITICAL: triggering recovery" $LOGFILE; then
    echo "✓ PASS: Recovery triggered"
else
    echo "✗ FAIL: Recovery not triggered"
    exit 1
fi

sleep 12

if grep -q "Recovery stage CODEC_ONLY succeeded" $LOGFILE; then
    echo "✓ PASS: Recovery succeeded"
else
    echo "✗ FAIL: Recovery failed"
    exit 1
fi

if grep -q "Recovery succeeded, waiting" $LOGFILE; then
    echo "✓ PASS: System waiting for reinitialize"
else
    echo "⚠️  WARN: Cooldown period not logged"
fi

echo ""
echo "=== All tests passed! ==="
```

Run with:
```bash
chmod +x test_stall_detection.sh
./test_stall_detection.sh
```

---

## Troubleshooting

### Monitor not updating timestamp
```bash
# Verify update_frame_timestamp is being called
grep "update_frame_timestamp" /home/sunrise/ConvuyerBreadCounting/data/logs/app.log
```

### Recovery not triggering
```bash
# Check if monitor is running
ps aux | grep "CodecHealthMonitor"

# Check rate limiting
cat $STATUSFILE | python3 -m json.tool | grep restarts_this_hour
```

### Restart doesn't work
```bash
# Manually test supervisor restart
sudo supervisorctl restart breadcount-ros2 breadcount-container-ros2

# Verify both services started
sudo supervisorctl status
```

---

## Success Metrics

| Metric | Target | Verified |
|--------|--------|----------|
| Stall detection time | ≤ 3 seconds | ⬜ |
| Recovery success | 100% | ⬜ |
| No crash loops | True | ⬜ |
| Frames resume after recovery | True | ⬜ |
| Both services restart on Stage 2 | True | ⬜ |
