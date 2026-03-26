# Skip-N Removal — Deployment Checklist

## Pre-Deployment Verification ✅

- [x] All 52 tests passing (`test_adaptive_frame_throttle.py`)
- [x] No compile/lint errors in modified files
- [x] Import and instantiation smoke test passed
- [x] State dict has correct fields (11 fields, no skip_n/frames_skipped)
- [x] Health UI updated (DEGRADED mode shows sentinel info)
- [x] No references to removed fields in production code

## Files Modified

### Core Logic
- `src/app/adaptive_frame_throttle.py` — **Completely rewritten**
  - Removed: skip_n, should_process(), frames_skipped, detection_only_wakes
  - Kept: All mode transitions, two-signal wake, hysteresis, thread-safety

### Frontend
- `src/endpoint/templates/health.html` — **2 sections updated**
  - DEGRADED mode card: shows sentinel interval instead of skip_n
  - Components grid: removed noiseCount

### Tests
- `test_adaptive_frame_throttle.py` — **Comprehensive rewrite**
  - All skip pattern tests removed
  - All core behavior tests retained and passing

## Files Verified Clean (No Changes Needed)

- ✅ `src/app/ConveyorCounterApp.py`
- ✅ `src/config/tracking_config.py`
- ✅ `src/app/pipeline_throttle_state.py`
- ✅ `src/endpoint/routes/health.py`
- ✅ `src/endpoint/server.py`

## Deployment Steps

### 1. Backup Current State
```bash
# On RDK
cd ~/breadcount
git status
git stash  # if any uncommitted changes
```

### 2. Deploy Changes
```bash
# From development machine
rsync -avz --exclude='.venv' --exclude='__pycache__' \
  ~/0012_ConvuyerBreadBagCounterSystem/ \
  rdk:~/breadcount/
```

### 3. Restart Services
```bash
# On RDK
sudo systemctl restart breadcount-main
sudo systemctl restart breadcount-spool-processor
sudo systemctl restart breadcount-endpoint
```

### 4. Verify Operation

#### Check Logs
```bash
tail -f data/logs/convuyer_counter.log
# Expected: "[FrameThrottle] Initialized: ... Sentinel probe rate controlled by SpoolProcessorNode"
```

#### Check Health Endpoint
```bash
curl http://localhost:5010/health | jq '.power_save'
```

**Expected output (FULL mode):**
```json
{
  "enabled": true,
  "mode": "full",
  "idle_seconds": 12.3,
  "idle_timeout_s": 900.0,
  "idle_percent": 1.4,
  "time_until_degrade_s": 887.7,
  "degraded_since_seconds": null,
  "hysteresis_s": 60.0,
  "degraded_transitions": 0,
  "wake_transitions": 0,
  "last_wake_signal": ""
}
```

**Fields that should NOT exist:**
- ❌ `skip_n`
- ❌ `frames_skipped`
- ❌ `total_frames_seen`
- ❌ `detection_only_wakes`

#### Check Health UI
Open `http://<rdk-ip>:5010/health` and verify:
- ✅ Throttle card shows "معالجة كاملة" (FULL mode)
- ✅ No "إطار كل N" text visible
- ✅ After 15 min idle → DEGRADED mode shows "إطار حارس / ثانية"

### 5. Functional Test

#### Idle Degradation Test
1. Stop conveyor belt
2. Wait 15 minutes
3. Health endpoint should show `"mode": "degraded"`
4. Logs should show: `[FrameThrottle] DEGRADE → sentinel probe mode`

#### Wake Test (Signal A)
1. Place a bag on the stopped belt
2. Move bag into ROI
3. Should wake within ~1 second (not 5 seconds)
4. Logs: `[FrameThrottle] WAKE → FULL (Signal A: detection)`

#### Wake Test (Signal B)
1. Start conveyor with bags
2. System wakes + confirmed tracks form
3. Logs: `[FrameThrottle] WAKE → FULL (Signal B: confirmed track)`

## Rollback Plan (if needed)

```bash
# On RDK
cd ~/breadcount
git log --oneline -5  # find commit before changes
git reset --hard <commit-hash>
sudo systemctl restart breadcount-*
```

## Success Criteria

- ✅ System starts without errors
- ✅ Health endpoint returns correct state dict (11 fields)
- ✅ DEGRADED mode triggers after 15 min idle
- ✅ Detection in DEGRADED mode wakes within ~1 second
- ✅ Power consumption unchanged (~6% CPU/VPU in sentinel)
- ✅ No missed bags in production

## Post-Deployment Monitoring

### First 24 Hours
- Monitor `data/logs/convuyer_counter.log` for throttle messages
- Check `/health` endpoint every hour
- Verify no missed bags in `/analytics`

### First Week
- Compare detection latency (should be 5x faster in idle mode)
- Verify degraded_transitions counter increases during idle periods
- Check wake_transitions matches expected wake events

## Notes

- **No config changes required** — all parameters unchanged
- **No database migration** — state fields are runtime-only
- **Backward compatible logs** — identical format
- **No API breaking changes** — `/health` endpoint simply has fewer fields

---

**Deployment Date:** _________________  
**Deployed By:** _________________  
**Verification Status:** _________________

