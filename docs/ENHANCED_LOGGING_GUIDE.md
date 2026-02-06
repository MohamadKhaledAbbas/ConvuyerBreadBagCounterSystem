# Enhanced Logging Guide

## Overview
Comprehensive track-by-track logging has been added to trace each object's complete journey through the pipeline. All logs use consistent prefixes for easy filtering and analysis.

## Log Categories

### 1. Track Lifecycle (`[TRACK_LIFECYCLE]`)
Tracks the creation, updates, and completion of tracked objects.

#### Detection Filtering
Low-confidence detections (< `min_confidence_new_track`, default 0.7) are excluded from creating new tracks to avoid noise:
```
[ConveyorTracker] Skipping low-confidence detection (conf=0.44 < 0.70) - not creating track
```
**Note**: Existing tracks can still be updated with lower confidence detections. The threshold only applies to NEW track creation.

#### Track Creation
```
[TRACK_LIFECYCLE] T{id} CREATED | bbox=(x1,y1,x2,y2) center=(cx,cy) conf=0.XX
```
**When**: A new detection creates a new track
**Info**: Initial bounding box, center position, detection confidence

#### Track Completion
```
[TRACK_LIFECYCLE] T{id} COMPLETED | type={event_type} exit={direction} hits={N} missed={M} 
  duration=X.XXs distance=XXXpx vel=(vx,vy) positions={N} avg_conf=0.XX
```
**When**: Track exits frame or times out
**Info**: 
- `type`: track_completed or track_lost
- `exit`: left/right/top/bottom/timeout
- `hits`: successful detections
- `missed`: frames without detection
- `distance`: pixels traveled
- `vel`: velocity vector

### 2. ROI Collection (`[ROI_LIFECYCLE]`)
Tracks ROI extraction and quality filtering.

#### Collection Start
```
[ROI_LIFECYCLE] T{id} ROI_COLLECTION_START | max_rois={N}
```
**When**: First ROI collected for a track

#### ROI Collected
```
[ROI_LIFECYCLE] T{id} ROI_COLLECTED | quality=XXX.X count={N}/{max} 
  best_quality=XXX.X size={W}x{H}
```
**When**: Valid ROI extracted and stored
**Info**: Quality score, collection progress, best quality so far, ROI dimensions

#### ROI Rejected
```
[ROI_LIFECYCLE] T{id} ROI_REJECTED | reason={reason} total_collected={N} total_rejected={M}
```
**When**: ROI fails quality checks
**Reasons**: low_sharpness, too_dark, too_bright, wrong_aspect_ratio

### 3. Classification (`[CLASSIFICATION]`)
Detailed per-ROI classification and voting results.

#### Single ROI Classification
```
[CLASSIFICATION] T{id} SINGLE_ROI | result={class} conf=0.XXX
```
**When**: Track has only one ROI

#### Multi-ROI Classification Start
```
[CLASSIFICATION] T{id} MULTI_ROI_START | total_rois={N}
```

#### Individual ROI Results
```
[CLASSIFICATION] T{id} ROI_{N}/{total} | result={class} conf=0.XXX
```
**When**: Each ROI is classified
**Info**: ROI index, classification result, confidence

#### Vote Exclusion
```
[CLASSIFICATION] T{id} ROI_{N} VOTE_EXCLUDED | reason=Rejected conf=0.XXX
```
**When**: ROI classified as "Rejected" (poor quality indicator)

#### All ROIs Rejected
```
[CLASSIFICATION] T{id} ALL_REJECTED | using_primary={class} conf=0.XXX
```
**When**: All ROIs classified as Rejected (fallback to primary)

#### Voting Result
```
[CLASSIFICATION] T{id} VOTING_RESULT | winner={class} conf=0.XXX 
  valid_votes={N}/{total} distribution=[class1:score1 class2:score2 ...]
```
**When**: Multi-ROI voting completes
**Info**: Final class, average confidence, vote distribution

#### Classification Complete
```
[CLASSIFICATION] T{id} COMPLETE | final={class} conf=0.XXX time=XX.Xms
```
**When**: Classification finishes (before smoothing)

### 4. Smoothing (`[SMOOTHING]`)
Batch processing and smoothing decisions.

#### Batch Created
```
[SMOOTHING] Batch {id} CREATED
```

#### Added to Batch
```
[SMOOTHING] T{id} ADDED_TO_BATCH | batch={id} class={name} conf=0.XXX 
  batch_size={N}/{max}
```
**When**: Classification result added to smoother batch

#### Rejected Exclusion
```
[SMOOTHING] T{id} EXCLUDED | reason=Rejected conf=0.XXX
```
**When**: "Rejected" classification bypasses smoothing

#### Batch Finalizing
```
[SMOOTHING] Batch {id} FINALIZING | reason={size_reached|timeout}
```

#### Batch Analysis
```
[SMOOTHING] Batch {id} ANALYSIS | size={N} dominant={class} 
  dominance=0.XX threshold=0.XX
```
**When**: Analyzing batch for smoothing decisions
**Info**: Dominant class and its confidence-weighted ratio

#### No Smoothing
```
[SMOOTHING] Batch {id} | action=NO_SMOOTHING reason={too_small|no_dominant_class}
```

#### Track Smoothed
```
[SMOOTHING] T{id} SMOOTHED | {original}->{new_class} conf=0.XXX dominance=0.XX
```
**When**: Low-confidence outlier overridden by batch dominance

#### Batch Finalized
```
[SMOOTHING] Batch {id} FINALIZED | records={N} distribution=[class1:count1 class2:count2 ...]
```

### 5. Counting (`[COUNT]`)
Final counting stage after smoothing.

#### Rejected (Not Counted)
```
[COUNT] T{id} REJECTED | conf=0.XXX total_rejected={N}
```
**When**: "Rejected" classification (excluded from count)

#### Counted
```
[COUNT] T{id} COUNTED | class={name} conf=0.XXX smoothed={yes|no} 
  class_total={N} system_total={M}
```
**When**: Object successfully counted
**Info**: Class name, confidence, whether smoothed, class count, total count

If smoothed:
```
[COUNT] T{id} COUNTED | class={name} conf=0.XXX smoothed_from={original} 
  class_total={N} system_total={M}
```

### 6. Pipeline (`[PIPELINE]`)
High-level pipeline events.

#### No ROIs
```
[PIPELINE] T{id} NO_ROIS | type={event_type} exit={direction} frames={N}
```
**When**: Track completes but no ROIs collected

#### Submit for Classification
```
[PIPELINE] T{id} SUBMIT_CLASSIFY | total_rois={N} using={K} 
  quality_avg=XXX.X quality_range=[min-max]
```
**When**: Submitting track for classification
**Info**: Total ROIs collected, how many used for voting, quality statistics

## Example Track Flow

Here's a complete example of logs for a single track (T42):

```
# Track starts
[TRACK_LIFECYCLE] T42 CREATED | bbox=(120,50,200,150) center=(160,100) conf=0.87

# ROI collection begins
[ROI_LIFECYCLE] T42 ROI_COLLECTION_START | max_rois=10
[ROI_LIFECYCLE] T42 ROI_COLLECTED | quality=234.5 count=1/10 best_quality=234.5 size=82x102
[ROI_LIFECYCLE] T42 ROI_REJECTED | reason=low_sharpness total_collected=1 total_rejected=1
[ROI_LIFECYCLE] T42 ROI_COLLECTED | quality=267.8 count=2/10 best_quality=267.8 size=84x104
[ROI_LIFECYCLE] T42 ROI_COLLECTED | quality=251.2 count=3/10 best_quality=267.8 size=86x106

# Track completes
[TRACK_LIFECYCLE] T42 COMPLETED | type=track_completed exit=top hits=25 missed=2 
  duration=1.87s distance=245px vel=(5.2,12.3) positions=25 avg_conf=0.89

# Classification
[PIPELINE] T42 SUBMIT_CLASSIFY | total_rois=3 using=3 quality_avg=251.2 quality_range=[234.5-267.8]
[CLASSIFICATION] T42 MULTI_ROI_START | total_rois=3
[CLASSIFICATION] T42 ROI_1/3 | result=Red_Yellow conf=0.987
[CLASSIFICATION] T42 ROI_2/3 | result=Red_Yellow conf=0.945
[CLASSIFICATION] T42 ROI_3/3 | result=Red_Yellow conf=0.923
[CLASSIFICATION] T42 VOTING_RESULT | winner=Red_Yellow conf=0.952 valid_votes=3/3 
  distribution=[Red_Yellow:2.855]
[CLASSIFICATION] T42 COMPLETE | final=Red_Yellow conf=0.952 time=45.3ms

# Smoothing
[SMOOTHING] T42 ADDED_TO_BATCH | batch=5 class=Red_Yellow conf=0.952 batch_size=7/10
[SMOOTHING] Batch 5 FINALIZING | reason=size_reached
[SMOOTHING] Batch 5 ANALYSIS | size=10 dominant=Red_Yellow dominance=0.89 threshold=0.70
[SMOOTHING] Batch 5 FINALIZED | records=10 distribution=[Red_Yellow:8 Brown_Orange:1 Green_Yellow:1]

# Counting
[COUNT] T42 COUNTED | class=Red_Yellow conf=0.952 smoothed=no class_total=45 system_total=123
```

## Filtering Logs

### By Track ID
```bash
grep "T42" log_file.log
```

### By Stage
```bash
grep "\[TRACK_LIFECYCLE\]" log_file.log
grep "\[CLASSIFICATION\]" log_file.log
grep "\[SMOOTHING\]" log_file.log
grep "\[COUNT\]" log_file.log
```

### By Event Type
```bash
# All track completions
grep "COMPLETED" log_file.log

# All smoothing decisions
grep "SMOOTHED" log_file.log

# All rejected items
grep "REJECTED" log_file.log

# Voting results
grep "VOTING_RESULT" log_file.log
```

### Specific Issues
```bash
# Tracks with no ROIs
grep "NO_ROIS" log_file.log

# All rejected ROIs
grep "ROI_REJECTED" log_file.log

# Tracks that were smoothed
grep "smoothed_from=" log_file.log

# Low confidence classifications
grep "conf=0\.[0-4]" log_file.log
```

## Analysis Tips

### 1. Track Success Rate
Count tracks that completed vs. tracks that were lost:
```bash
grep "COMPLETED" log_file.log | wc -l
grep "track_lost" log_file.log | wc -l
```

### 2. ROI Quality Issues
```bash
# What's being rejected most?
grep "ROI_REJECTED" log_file.log | awk '{print $7}' | sort | uniq -c
```

### 3. Classification Confidence
```bash
# Average voting confidence
grep "VOTING_RESULT" log_file.log | awk -F'conf=' '{print $2}' | awk '{print $1}'
```

### 4. Smoothing Impact
```bash
# How many were smoothed?
grep "SMOOTHED" log_file.log | wc -l

# What classes are being smoothed?
grep "SMOOTHED" log_file.log
```

### 5. Processing Time
```bash
# Classification times
grep "COMPLETE" log_file.log | grep "time=" | awk -F'time=' '{print $2}' | awk '{print $1}'
```

## Troubleshooting

### Track Loss Issues
1. Check `[TRACK_LIFECYCLE]` logs for tracks that become "track_lost"
2. Look at `hits` vs `missed` ratio
3. Check velocity values for erratic movement

### Misclassification
1. Follow a specific track ID through all stages
2. Check ROI quality scores in `[ROI_LIFECYCLE]`
3. Review individual ROI classifications in `[CLASSIFICATION]`
4. See if smoothing changed the result in `[SMOOTHING]`

### Rejected Items
1. Check `[CLASSIFICATION]` for "Rejected" votes
2. Review `[ROI_LIFECYCLE]` for quality issues
3. Count how many tracks have all ROIs rejected

### Duplicate Counts
1. Search for a track ID in `[COUNT]` logs
2. Should appear exactly once with "COUNTED"
3. If appears multiple times, batch smoothing may be emitting duplicates
