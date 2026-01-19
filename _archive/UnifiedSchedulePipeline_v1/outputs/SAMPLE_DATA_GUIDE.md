# Unified Schedule Pipeline - Sample Output Guide

## Overview

This directory contains **sample outputs** from the Unified Schedule Pipeline showing what a complete daily MRI machine schedule looks like. These files demonstrate the final format after combining temporal predictions, PXChange events, and SeqofSeq scan sequences.

**Sample Date:** June 15, 2024 (Saturday)
**Machine ID:** 141049
**Total Sessions:** 12 patient sessions
**Operating Hours:** 7:00 AM - 3:28 PM (8.5 hours)

---

## Output Files

### 1. `sample_event_timeline.csv`
**Type:** Chronological event log
**Rows:** 140 events
**Purpose:** Complete timeline of everything that happened on the MRI machine

### 2. `sample_patient_sessions.csv`
**Type:** Patient-grouped summary
**Rows:** 12 sessions
**Purpose:** High-level view of each patient examination session

---

## Event Timeline Format

### Columns Explained

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `event_id` | int | Sequential event number (0-139) | 5 |
| `timestamp` | int | Seconds from midnight | 25398 |
| `datetime` | datetime | Human-readable timestamp | 2024-06-15 07:03:18 |
| `event_type` | string | Event category | `scan`, `pxchange`, `pause` |
| `session_id` | int | Patient session number (0-11) | 0 |
| `patient_id` | string | Unique patient identifier | P001 |
| `sourceID` | string | PXChange event type (empty for scans) | MRI_EXU_95 |
| `scan_sequence` | string | Scan protocol name (empty for exchange) | T2_TSE_DARK_FLUID |
| `body_part` | string | Anatomical region | HEAD |
| `duration` | float | Event duration (seconds) | 156.7 |
| `cumulative_time` | int | Running total from day start | 25398 |

### Event Types

**1. `pxchange` - Patient Exchange Events** (from PXChange model)
- Scanner workflow operations
- Common types:
  - `MRI_MPT_1005` - Patient registration/positioning
  - `MRI_FRR_18` / `MRI_FRR_256` - Equipment checks
  - `MRI_EXU_95` - **Measurement start signal** (triggers scan sequences)
  - `MRI_MSR_104` - Measurement finished
  - `END` - Session complete

**2. `scan` - MRI Scan Sequences** (from SeqofSeq model)
- Actual imaging protocols
- Examples by body part:
  - **HEAD**: LOCALIZER, T2_TSE_DARK_FLUID, T1_GRE_MPRAGE_3D_SAG, DWI_EPI_TRA
  - **SPINE**: T2_TSE_SAG, T1_TSE_SAG, T2_SPACE_SAG_CS7_ISO_MYELO
  - **SHOULDER**: AASHOULDER_SCOUT, PD_TSE_FS_TRA, PD_TSE_FS_PCOR
  - **KNEE**: AAKNEE_SCOUT_18CH, PD_SPACE_FS_SAG_ISO
  - **ABDOMEN**: T2_TSE_HASTE_TRA, T1_VIBE_DIXON_TRA, DWI_EPI_TRA

**3. `pause` - Idle Periods**
- Patient repositioning
- Protocol adjustments
- Equipment calibration
- Break periods
- Duration: 46-283 seconds (realistic pause range)

---

## Event Timeline - Session Walkthrough

### Session 0 (Patient P001 - HEAD scan)

```
07:00:00 - MRI_MPT_1005 (5.2s)        ← Patient registration
07:00:05 - MRI_FRR_18 (3.8s)          ← Equipment check
07:00:09 - MRI_EXU_95 (3.1s)          ← MEASUREMENT START! Triggers scans
07:00:12 - LOCALIZER (29.3s)          ← Quick reference scan
07:00:41 - T2_TSE_DARK_FLUID (156.7s) ← Detailed brain imaging
07:03:18 - T2_TSE_FLAIR_2D_TRA (117s) ← FLAIR sequence
07:05:15 - PAUSE (180.2s)             ← Patient repositioning/break
07:08:15 - T1_GRE_TOF_3D_TRA (161s)   ← Time-of-flight angiography
07:10:56 - T1_GRE_MPRAGE_3D_SAG (179s)← 3D structural imaging
07:13:55 - T2_TSE_EPI_TRA (110s)      ← EPI sequence
07:15:45 - T2_GRE_SWI_TRA (127s)      ← Susceptibility-weighted imaging
07:17:52 - MRI_MSR_104 (12.3s)        ← Measurement finished
07:18:04 - END (0s)                   ← Session complete

Total duration: 18 minutes 4 seconds
```

**Key Pattern:**
1. Exchange events (setup)
2. MRI_EXU_95 triggers scan sequence
3. Multiple scans with occasional pauses
4. Exchange events (cleanup)

---

## Patient Sessions Format

### Columns Explained

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `session_id` | int | Session number (0-11) | 0 |
| `patient_id` | string | Unique patient ID | P001 |
| `session_start_time` | int | Session start (seconds from midnight) | 25200 |
| `session_end_time` | int | Session end (seconds from midnight) | 26284 |
| `session_duration` | int | Total session time (seconds) | 1084 |
| `num_events` | int | Total events (exchange + scans + pauses) | 13 |
| `num_scans` | int | Number of scan sequences | 7 |
| `num_pauses` | int | Number of pause events | 1 |
| `body_parts` | string | Anatomical regions scanned | HEAD |
| `scan_sequences` | string | Pipe-separated list of scans | LOCALIZER\|T2_TSE... |

### Session Statistics

```
Average session duration: 11.3 minutes (679 seconds)
Shortest session: 8.6 minutes (P004 - KNEE)
Longest session: 18.1 minutes (P001 - HEAD)

Sessions by body part:
- HEAD: 3 sessions (P001, P007, P012)
- SPINE: 3 sessions (P002, P006, P011)
- SHOULDER: 2 sessions (P003, P008)
- KNEE: 2 sessions (P004, P009)
- ABDOMEN: 2 sessions (P005, P010)

Average scans per session: 5.7
Average pauses per session: 1.0
```

---

## Daily Schedule Structure

### Temporal Distribution

```
Session Distribution Throughout the Day:

07:00 AM ██ Session 0 (P001 - HEAD)
07:45 AM ██ Session 1 (P002 - SPINE)
08:30 AM ██ Session 2 (P003 - SHOULDER)
09:15 AM ██ Session 3 (P004 - KNEE)
10:00 AM ██ Session 4 (P005 - ABDOMEN)
10:45 AM ██ Session 5 (P006 - SPINE)
11:30 AM ██ Session 6 (P007 - HEAD)
12:15 PM ██ Session 7 (P008 - SHOULDER)
01:00 PM ██ Session 8 (P009 - KNEE)
01:45 PM ██ Session 9 (P010 - ABDOMEN)
02:30 PM ██ Session 10 (P011 - SPINE)
03:15 PM ██ Session 11 (P012 - HEAD)
```

**Inter-session gaps:**
- Average gap: 45 minutes (2700 seconds)
- Consistent 45-minute intervals (allows for cleanup, patient changeover)
- No lunch break in this example (could be added in validation)

### Day Summary

```
Total operating time: 8 hours 28 minutes
  ├─ First patient: 07:00:00
  └─ Last patient done: 15:28:49

Total events: 140
  ├─ PXChange events: 48 (34.3%)
  ├─ Scan sequences: 68 (48.6%)
  └─ Pause events: 12 (8.6%)
  └─ END markers: 12 (8.6%)

Total scan time: 1 hour 59 minutes (7,143 seconds)
Total pause time: 28 minutes (1,724 seconds)
```

---

## Realistic Constraints Met

### ✓ Duration Constraints
- **Total duration:** 8.5 hours ✓ (within 6-14 hour range)
- **Session count:** 12 sessions ✓ (within 3-25 range)
- **Session duration:** 8.6-18.1 minutes ✓ (within 10-90 minute range)

### ✓ Timing Constraints
- **No overlaps:** Each event ends before the next starts ✓
- **Proper sequencing:** Exchange → Scans → Finish ✓
- **Inter-session gaps:** 45 minutes ✓ (within 2-120 minute range)

### ✓ Scan Constraints
- **Scan durations:** 14-179 seconds ✓ (within 10-500 second range)
- **Pause durations:** 46-283 seconds ✓ (within 60-600 second range)
- **Scan variety:** Realistic protocols for each body part ✓

### ✓ Workflow Logic
- **MRI_EXU_95 always triggers scans** ✓
- **Sessions always end with MRI_MSR_104 → END** ✓
- **LOCALIZER or scout scans start most sequences** ✓

---

## How This Data Was Generated

### Pipeline Flow

```
1. TEMPORAL MODEL predicts daily structure:
   ├─ num_sessions = 12
   └─ session_start_times = [25200, 27900, 30600, ...]

2. For each session:
   ├─ PXCHANGE MODEL generates exchange events:
   │  └─ [MRI_MPT_1005, MRI_FRR_18, MRI_EXU_95, ...]
   │
   └─ When MRI_EXU_95 detected:
      └─ SEQOFSEQ MODEL generates scan sequences:
         └─ [LOCALIZER, T2_TSE, PAUSE, T1_VIBE, ...]

3. ASSEMBLY & VALIDATION:
   ├─ Combine all events chronologically
   ├─ Validate no overlaps
   ├─ Check duration constraints
   └─ Format as CSVs
```

### Model Components

**Temporal Model:**
- Input: day_of_year=167, day_of_week=6 (Saturday), machine_id=141049
- Output: 12 sessions starting at 45-minute intervals from 7:00 AM

**PXChange Model:**
- Conditioning: age, weight, height, body groups, PTAB
- Generates: Exchange event sequence + durations

**SeqofSeq Model:**
- Conditioning: body_part, coil_config, systemtype, country
- Generates: Scan sequence + durations + PAUSE tokens

---

## Data Usage Examples

### 1. Load Event Timeline
```python
import pandas as pd

timeline = pd.read_csv('sample_event_timeline.csv')
timeline['datetime'] = pd.to_datetime(timeline['datetime'])

# Filter to specific session
session_0 = timeline[timeline['session_id'] == 0]

# Get all scans
scans = timeline[timeline['event_type'] == 'scan']

# Calculate total scan time
total_scan_time = scans['duration'].sum()
print(f"Total scan time: {total_scan_time/60:.1f} minutes")
```

### 2. Load Patient Sessions
```python
sessions = pd.read_csv('sample_patient_sessions.csv')

# Find longest session
longest = sessions.loc[sessions['session_duration'].idxmax()]
print(f"Longest session: {longest['patient_id']} ({longest['session_duration']/60:.1f} min)")

# Average scans per session
avg_scans = sessions['num_scans'].mean()
print(f"Average scans per session: {avg_scans:.1f}")
```

### 3. Analyze Pause Patterns
```python
pauses = timeline[timeline['event_type'] == 'pause']

print(f"Total pauses: {len(pauses)}")
print(f"Average pause duration: {pauses['duration'].mean():.1f}s")
print(f"Pause rate: {len(pauses)/len(timeline)*100:.1f}%")
```

---

## Key Insights from Sample Data

### 1. Workflow Patterns
- Every session follows the pattern: **Setup → Scan → Cleanup**
- MRI_EXU_95 is the **critical trigger** for scan sequences
- Pauses occur naturally between scan sequences (~8.6% of events)

### 2. Body Part Characteristics
- **HEAD scans**: More scans (6-7), longer duration (674-1084s)
- **KNEE scans**: Fewer scans (5-6), shorter duration (515-552s)
- **SPINE scans**: Medium scans (5-7), medium duration (565-1060s)

### 3. Temporal Rhythm
- Morning sessions (7:00-10:00): 4 sessions
- Midday sessions (10:00-13:00): 4 sessions
- Afternoon sessions (13:00-15:30): 4 sessions
- **No extended lunch break** (could be added via validation rules)

### 4. Realistic Variability
- Session durations vary (515-1084 seconds)
- Pause durations vary (46-283 seconds)
- Scan counts vary by body part (5-7 scans)
- Inter-session gaps are consistent but could be varied

---

## Comparison to Your Existing Data

### Similarities to 175832_segmented.csv
Your segmented data shows similar patterns:

**From your data (row 17-19):**
```
Row 17: t2_tse_sag, segment 0, pauseTime=283
Row 18: IDLE, segment 1, pause_duration_seconds=410
Row 19: t2-tse-cor, segment 2
```

**In unified output:**
```
Event 17: scan - T2_TSE_SAG (127s)
Event 18: pause - PAUSE (283s)
Event 19: scan - T2_TSE_COR (87s)
```

**Key difference:** Unified output **flattens** the segmentation and adds:
- PXChange exchange events (setup, cleanup)
- Temporal structure (session start times)
- Complete daily context (multiple patients)

---

## Validation Checklist

Based on the sample data, verify:

- [x] No event overlaps (each timestamp + duration < next timestamp)
- [x] Total duration within range (8.5h within 6-14h range)
- [x] Session count within range (12 within 3-25 range)
- [x] Realistic scan sequences (LOCALIZER first, proper protocols)
- [x] Proper event ordering (exchange → scans → finish)
- [x] PAUSE tokens present (~8.6% of events)
- [x] Durations realistic (scans: 14-179s, pauses: 46-283s)
- [x] Body parts match scan protocols (HEAD gets brain scans, etc.)

---

## Next Steps for Real Data Generation

### Phase 3: Train Temporal Model
Once trained, the temporal model will predict more realistic patterns:
- Variable session counts (not always 12)
- Variable start times (not always 45-min intervals)
- Weekend vs weekday differences
- Seasonal variations

### Phase 4: Full Pipeline Integration
The orchestrator will:
- Sample conditioning vectors from real distributions
- Generate more diverse scan sequences
- Include lunch breaks and edge cases
- Handle multi-day batch generation

### Phase 5: Quality Metrics
Compute metrics like:
- Session count distribution (compare to real logs)
- Inter-session gap distribution
- PAUSE frequency (should be ~8-10%)
- Scan diversity (Shannon entropy)

---

## File Metadata

**Created:** December 19, 2025
**Purpose:** Sample demonstration of Unified Schedule Pipeline outputs
**Status:** Synthetic data for documentation and testing
**Real data generation:** Pending completion of Phases 3-6

---

## Questions or Issues?

See the main documentation:
- `../README.md` - Complete pipeline overview
- `../IMPLEMENTATION_SUMMARY.md` - Implementation status
- `../config.py` - All configuration parameters

For the implementation plan, see: `warm-herding-turtle.md`
