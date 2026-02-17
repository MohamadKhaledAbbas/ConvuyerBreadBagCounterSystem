# ROI Collection Flow - Visual Guide

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    CONVEYOR BELT SYSTEM                          │
│                                                                  │
│   Object Movement: Bottom → Top (Y decreases)                   │
│                                                                  │
│   ┌──────────────────────────────────────────────────┐         │
│   │ Y=0   TOP    [EXIT ZONE - Objects leave here]   │ ◄─┐     │
│   │                                                  │   │     │
│   │         ╔════════════════════╗                  │   │     │
│   │         ║  HEAVY PENALTY     ║                  │   │     │
│   │         ║  30% quality       ║                  │   │     │
│   │         ╚════════════════════╝                  │   │     │
│   │ Y=72  Penalty Max (15%)                         │   │     │
│   │         ▲                                        │   │     │
│   │         │ Gradual quality increase              │   │     │
│   │         │ (Linear interpolation)                │   │     │
│   │         │                                        │   │     │
│   │ Y=240 CENTER (50%)                              │   │     │
│   │         ╔════════════════════╗                  │   │     │
│   │         ║  NO PENALTY        ║                  │   │     │
│   │         ║  100% quality      ║                  │   │     │
│   │         ╚════════════════════╝                  │   │     │
│   │                                                  │   │     │
│   │                                                  │   │     │
│   │ Y=480 BOTTOM [ENTRY - Objects enter here]      │   │     │
│   └──────────────────────────────────────────────────┘   │     │
│                                                           │     │
│   Frame Height: 480px                                     │     │
│   Frame Width: 640px                                      │     │
│                                                           │     │
│   Quality Gradient: Bottom (best) → Top (worst) ─────────┘     │
└──────────────────────────────────────────────────────────────────┘
```

## ROI Collection Decision Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  NEW DETECTION FOR TRACK T1                                    │
│  Frame #100, BBox: (150, 300, 250, 400)                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │ 1. Check Collection Limit     │
         │ collected_count < max_rois?   │
         └───────────┬───────────────────┘
                     │ YES
                     ▼
         ┌───────────────────────────────┐
         │ 2. Check Frame Spacing        │
         │ frame_gap >= 3 frames?        │
         └───────────┬───────────────────┘
                     │ YES (gap=5)
                     ▼
         ┌───────────────────────────────┐
         │ 3. Check Position Diversity   │
         │ movement >= 20 pixels?        │
         └───────────┬───────────────────┘
                     │ YES (moved 35px)
                     ▼
         ┌───────────────────────────────┐
         │ 4. Extract ROI from Frame     │
         │ With padding: +5px border     │
         └───────────┬───────────────────┘
                     │
                     ▼
         ┌───────────────────────────────┐
         │ 5. Check Quality Thresholds   │
         │ • Sharpness >= 50.0           │
         │ • Brightness: 30.0 - 225.0    │
         │ • Size: >= 20px               │
         └───────────┬───────────────────┘
                     │ PASS
                     ▼
         ┌───────────────────────────────┐
         │ 6. Apply Position Penalty     │
         │ Y=350 (ratio=0.73)            │
         │ → Below center → No penalty   │
         │ Quality: 1850.0 * 1.0 = 1850  │
         └───────────┬───────────────────┘
                     │
                     ▼
         ┌───────────────────────────────┐
         │ 7. Apply Temporal Weighting   │
         │ (if enabled)                  │
         │ Decay factor based on count   │
         └───────────┬───────────────────┘
                     │
                     ▼
         ┌───────────────────────────────┐
         │ 8. Add to Collection          │
         │ • Store ROI image             │
         │ • Store quality score         │
         │ • Store frame index           │
         │ • Store position              │
         │ • Update last_frame_index     │
         │ • Update last_position        │
         └───────────┬───────────────────┘
                     │
                     ▼
         ┌───────────────────────────────┐
         │ ✅ ROI COLLECTED              │
         │ Log: ROI_COLLECTED            │
         └───────────────────────────────┘
```

## Rejection Scenarios

```
┌──────────────────────────────────────────────────────────────────┐
│ SCENARIO 1: Frame Spacing Rejection                             │
├──────────────────────────────────────────────────────────────────┤
│ Frame #102 (2 frames after last collection at #100)             │
│ ❌ REJECT: frame_gap=2 < min_frame_spacing=3                    │
│ Log: ROI_SKIPPED_FRAME_SPACING | frame_gap=2 < min=3            │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ SCENARIO 2: Position Diversity Rejection                        │
├──────────────────────────────────────────────────────────────────┤
│ Last position: (200, 350)                                       │
│ New position:  (210, 355)                                       │
│ Movement: sqrt((10)² + (5)²) = 11.2px                          │
│ ❌ REJECT: movement=11.2px < min_position_change=20.0px         │
│ Log: ROI_SKIPPED_POSITION | change=11.2px < min=20.0px          │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ SCENARIO 3: Quality Rejection                                   │
├──────────────────────────────────────────────────────────────────┤
│ Sharpness: 25.3 (< 50.0 threshold)                             │
│ ❌ REJECT: blurry (sharpness=25.3)                              │
│ Log: ROI_REJECTED | reason=blurry (sharpness=25.3)              │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ SCENARIO 4: Position Penalty Applied                            │
├──────────────────────────────────────────────────────────────────┤
│ BBox center: Y=150 (ratio=0.31)                                │
│ Between penalty_max (0.15) and penalty_start (0.5)              │
│ Interpolation: (0.31 - 0.15) / (0.5 - 0.15) = 0.457            │
│ Penalty: 0.3 + (1.0 - 0.3) * 0.457 = 0.62                      │
│ Quality: 1500.0 * 0.62 = 930.0                                  │
│ ✅ COLLECT (with penalty)                                       │
│ Log: GRADUAL_POSITION_PENALTY | y_ratio=0.31 penalty=0.62       │
└──────────────────────────────────────────────────────────────────┘
```

## Track Lifecycle Example

```
Timeline: Object moving from bottom to top of frame

Frame 0-49: Object enters frame at bottom
─────────────────────────────────────────────────────────────
Frame 50:  Y=420 | ✅ ROI #1 collected (quality=1900, no penalty)
Frame 51:  Y=410 | ❌ Skipped (frame spacing)
Frame 52:  Y=400 | ❌ Skipped (frame spacing)
Frame 53:  Y=390 | ✅ ROI #2 collected (quality=1850, no penalty)
                   Movement: 30px ✓, Gap: 3 frames ✓

Frame 56:  Y=370 | ❌ Skipped (frame spacing)
Frame 57:  Y=360 | ✅ ROI #3 collected (quality=1820, no penalty)
                   Movement: 30px ✓, Gap: 4 frames ✓

Frame 60:  Y=340 | ✅ ROI #4 collected (quality=1800, no penalty)
Frame 63:  Y=320 | ✅ ROI #5 collected (quality=1790, no penalty)
Frame 66:  Y=300 | ✅ ROI #6 collected (quality=1780, no penalty)
Frame 69:  Y=280 | ✅ ROI #7 collected (quality=1760, no penalty)
Frame 72:  Y=260 | ✅ ROI #8 collected (quality=1750, no penalty)

Frame 75:  Y=240 | ✅ ROI #9 collected (quality=1740, center line)
                   Still no penalty at center

Frame 78:  Y=220 | ✅ ROI #10 collected (quality=1560, 10% penalty)
                   Y_ratio=0.46 → light penalty starts

Frame 81:  Y=200 | ❌ Max ROIs reached (10/10)
Frame 100: Track exits frame (Y < 0)

─────────────────────────────────────────────────────────────
RESULT: 10 diverse, high-quality ROIs collected
        9 with no penalty (bottom/center)
        1 with light penalty (upper-mid)
        Ready for classification with voting
─────────────────────────────────────────────────────────────
```

## Quality Score Breakdown

```
┌──────────────────────────────────────────────────────────────┐
│  FINAL ROI QUALITY CALCULATION                               │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. BASE QUALITY (from image analysis)                       │
│     ├─ Sharpness: 1850.0 (Laplacian variance)              │
│     └─ Base score: 1850.0                                   │
│                                                              │
│  2. POSITION PENALTY (new feature)                           │
│     ├─ Y position: 220px                                    │
│     ├─ Y ratio: 220/480 = 0.458                             │
│     ├─ Below center (0.5)? NO → Apply penalty               │
│     ├─ Interpolate: (0.458-0.15)/(0.5-0.15) = 0.88         │
│     ├─ Penalty: 0.3 + 0.7*0.88 = 0.916                     │
│     └─ After penalty: 1850.0 * 0.916 = 1694.6              │
│                                                              │
│  3. TEMPORAL WEIGHTING (optional)                            │
│     ├─ Collection index: 7/10                               │
│     ├─ Decay factor: 1.0 - (7/10)*0.15 = 0.895             │
│     └─ Final score: 1694.6 * 0.895 = 1516.7                │
│                                                              │
│  ✅ FINAL QUALITY SCORE: 1516.7                             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Classification Workflow

```
Track T1 completes (exits frame)
        │
        ▼
┌─────────────────────────┐
│ Get all collected ROIs  │
│ Count: 10 ROIs          │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Select top-K by quality │
│ K=8 (configurable)      │
│ Use: 8 best ROIs        │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Classify each ROI       │
│ • ROI 1: T13 (0.95)    │
│ • ROI 2: T13 (0.92)    │
│ • ROI 3: T13 (0.88)    │
│ • ROI 4: T14 (0.65)    │ ◄─ Outlier
│ • ROI 5: T13 (0.91)    │
│ • ROI 6: T13 (0.89)    │
│ • ROI 7: T13 (0.87)    │
│ • ROI 8: T13 (0.90)    │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Voting with evidence    │
│ T13: 7 votes (strong)   │
│ T14: 1 vote (weak)      │
│ Winner: T13 ✅          │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Result: T13 counted     │
│ Confidence: HIGH        │
└─────────────────────────┘
```

## Benefits Visualization

```
┌─────────────────────────────────────────────────────────────┐
│                    BEFORE ENHANCEMENTS                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Frame 100: ✅ Collect ROI (Y=350)                         │
│  Frame 101: ✅ Collect ROI (Y=348)  ◄─ Nearly identical!  │
│  Frame 102: ✅ Collect ROI (Y=346)  ◄─ Nearly identical!  │
│  Frame 103: ✅ Collect ROI (Y=344)  ◄─ Nearly identical!  │
│                                                             │
│  Result: 10 ROIs, mostly redundant                         │
│          High correlation between ROIs                      │
│          Waste of classification resources                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    AFTER ENHANCEMENTS                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Frame 100: ✅ Collect ROI (Y=350, pos=200)               │
│  Frame 101: ❌ Skip (frame spacing)                        │
│  Frame 102: ❌ Skip (frame spacing)                        │
│  Frame 103: ✅ Collect ROI (Y=330, pos=235)  ◄─ Different!│
│                                  ↑                          │
│                    Moved 35px, 3 frames later              │
│                                                             │
│  Result: 8-10 ROIs, highly diverse                         │
│          Different poses/angles                             │
│          Better classification accuracy                     │
│          Efficient resource usage                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Summary Diagram

```
╔═══════════════════════════════════════════════════════════════╗
║            ROI COLLECTOR - PRODUCTION GRADE                   ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  🕐 FRAME SPACING                                             ║
║     └─ Min 3 frames between collections                      ║
║                                                               ║
║  📍 POSITION DIVERSITY                                        ║
║     └─ Min 20px movement required                            ║
║                                                               ║
║  📉 GRADUAL POSITION PENALTY                                  ║
║     └─ Smooth quality: Bottom (100%) → Top (30%)             ║
║                                                               ║
║  ⏳ TEMPORAL WEIGHTING                                        ║
║     └─ Earlier ROIs preferred (closer to camera)             ║
║                                                               ║
║  🎯 QUALITY FILTERING                                         ║
║     └─ Sharpness, brightness, size checks                    ║
║                                                               ║
╠═══════════════════════════════════════════════════════════════╣
║                          RESULT                               ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  ✅ Diverse ROIs from different moments                      ║
║  ✅ Different poses and perspectives                         ║
║  ✅ Smooth quality gradients (no harsh cutoffs)              ║
║  ✅ Better classification accuracy                           ║
║  ✅ Production-ready and fully tested                        ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```
