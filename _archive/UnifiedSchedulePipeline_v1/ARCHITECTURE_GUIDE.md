# Daily Schedule Generation - Architecture Guide

## What This System Does

Generates synthetic MRI facility schedules by predicting:
- **When** patients arrive (session start times)
- **What** happens during each session (MRI events and scans)
- **How long** each event takes (durations)

---

## Execution Flow (Step by Step)

### STEP 1: Load Patient Data
**File:** `generate_daily_schedule.py:311-319`

```
INPUT:  Patient CSV file (optional)
OUTPUT: DataFrame with patient_id, age, weight, height, bodygroup_from, bodygroup_to, ptab
```

| If CSV Provided | If No CSV |
|-----------------|-----------|
| Load from file | Generate random patient data |
| num_sessions = row count | num_sessions = 15 (default) |

**Patient CSV Format:**
```csv
patient_id,age,weight,height,bodygroup_from,bodygroup_to,ptab
PAT001,45,72,175,HEAD,HEAD,0
```

---

### STEP 2: Load All 5 Models
**File:** `generate_daily_schedule.py:329-341`

| Model | Purpose | File |
|-------|---------|------|
| Temporal | Predict session count & start times | `datetime_model/temporal_schedule_model.py` |
| PXChange Sequence | Generate MRI event tokens | `PXChange_Refactored/models/conditional_sequence_generator.py` |
| PXChange Duration | Predict duration per event | `PXChange_Refactored/models/conditional_counts_generator.py` |
| SeqofSeq Sequence | Generate scan tokens | `SeqofSeq_Pipeline/models/conditional_sequence_generator.py` |
| SeqofSeq Duration | Predict duration per scan | `SeqofSeq_Pipeline/models/conditional_duration_predictor.py` |

---

### STEP 3: Predict Daily Structure (Temporal Model)
**File:** `generate_daily_schedule.py:343-358`

```
INPUT:  Date features (day of week, month, machine_id, etc.)
OUTPUT: Session start times (seconds from midnight)
```

**What happens:**
1. Extract temporal features from date
2. Model predicts `lambda` (expected session count)
3. Model samples `num_sessions` start times from mixture of Gaussians
4. Sort times chronologically

**Current Issue:** Model outputs clustered times (e.g., all within 2 hours)

---

### STEP 4: Generate Events for Each Patient Session
**File:** `generate_daily_schedule.py:383-526`

For each patient:

#### 4a. Build Conditioning Array
```
INPUT:  Patient data (from CSV or generated)
OUTPUT: Conditioning tensor [age, weight, height, bodygroup_from, bodygroup_to, ptab, entity_type]
```

**Body Group Mapping:**
| ID | Name | ID | Name |
|----|------|----|------|
| 0 | HEAD | 5 | SPINE |
| 1 | NECK | 6 | ARM |
| 2 | CHEST | 7 | LEG |
| 3 | ABDOMEN | 8 | HAND |
| 4 | PELVIS | 9 | FOOT |
| | | 10 | UNKNOWN |

#### 4b. Generate PXChange Sequence (MRI Events)
```
INPUT:  Conditioning array
OUTPUT: Token sequence [MRI_FRR_2, MRI_CCS_11, MRI_EXU_95, ...]
```

**Token Vocabulary (sourceID):**
| ID | Token | ID | Token |
|----|-------|----|-------|
| 0 | PAD | 10 | MRI_MSR_100 |
| 1 | MRI_CCS_11 | 11 | START |
| 2 | MRI_EXU_95 | 12 | MRI_MSR_104 |
| 3 | MRI_FRR_18 | 13 | MRI_MSR_21 |
| 4 | MRI_FRR_257 | 14 | END |
| 5 | MRI_FRR_264 | 15 | MRI_MSR_34 |
| 6 | MRI_FRR_2 | 16 | MRI_FRR_256 |
| 7 | MRI_FRR_3 | 17 | UNK |
| 8 | MRI_FRR_34 | | |
| 9 | MRI_MPT_1005 | | |

#### 4c. Predict Durations (PXChange Duration Model)
```
INPUT:  Conditioning + token sequence
OUTPUT: Duration (seconds) for each token
```

#### 4d. Trigger Scan Sequence (When MRI_EXU_95 Appears)
```
TRIGGER: Token ID = 2 (MRI_EXU_95 = measurement start)
```

When triggered:
1. **SeqofSeq Sequence Model** generates scan tokens
2. **SeqofSeq Duration Model** predicts scan durations
3. Scan events inserted into timeline

---

### STEP 5: Assemble Final Schedule
**File:** `generate_daily_schedule.py:528-543`

```
OUTPUT: DataFrame with columns:
- event_id, timestamp, datetime, event_type
- session_id, patient_id
- sourceID, scan_sequence, body_part
- bodygroup_from, bodygroup_to
- duration, cumulative_time
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  INPUT                                                              │
│  ─────                                                              │
│  • Date (2026-01-10)                                                │
│  • Machine ID (141049)                                              │
│  • Patient CSV (optional)                                           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TEMPORAL MODEL                                                     │
│  ──────────────                                                     │
│  Input:  Date → temporal features (12 dims)                         │
│  Output: Session start times (seconds from midnight)                │
│                                                                     │
│  Example: [25200, 27900, 30600, ...]  (7:00, 7:45, 8:30, ...)       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │  For each session/patient...  │
                    └───────────────┬───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PATIENT CONDITIONING                                               │
│  ────────────────────                                               │
│  From CSV:      [age, weight, height, bodygroup_from/to, ptab]      │
│  Or Generated:  Random realistic values                             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PXCHANGE SEQUENCE MODEL                                            │
│  ───────────────────────                                            │
│  Input:  Conditioning (7 features)                                  │
│  Output: Token sequence                                             │
│                                                                     │
│  Example: [11, 6, 1, 1, 16, 4, 5, 9, 10, 2, ...]                    │
│           START→MRI_FRR_2→MRI_CCS_11→...→MRI_EXU_95→...             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PXCHANGE DURATION MODEL                                            │
│  ───────────────────────                                            │
│  Input:  Conditioning + token sequence                              │
│  Output: μ, σ for each token → sample from Gamma distribution       │
│                                                                     │
│  Example: [0.0, 17.5, 56.5, 11.0, ...]  (seconds per event)         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │  When MRI_EXU_95 (ID=2)       │
                    │  appears in sequence...       │
                    └───────────────┬───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SEQOFSEQ SEQUENCE MODEL                                            │
│  ───────────────────────                                            │
│  Input:  Conditioning (5 features)                                  │
│  Output: Scan token sequence                                        │
│                                                                     │
│  Example: [LOCALIZER, T2_TSE, T1_MPRAGE, DWI_EPI, ...]              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SEQOFSEQ DURATION MODEL                                            │
│  ───────────────────────                                            │
│  Input:  Conditioning + scan tokens                                 │
│  Output: Duration per scan                                          │
│                                                                     │
│  Example: [29.3, 156.7, 179.0, 98.0, ...]  (seconds per scan)       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  OUTPUT CSV                                                         │
│  ──────────                                                         │
│  Columns: event_id, timestamp, datetime, event_type, session_id,    │
│           patient_id, sourceID, scan_sequence, body_part,           │
│           bodygroup_from, bodygroup_to, duration, cumulative_time   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Current Status

### What Works
- Patient data loaded from CSV with body groups
- Body groups used in model conditioning
- Body groups and patient IDs appear in output
- All 5 models execute in sequence
- Scan sequences triggered by MRI_EXU_95

### What Needs Investigation
- Temporal model outputs narrow time window (sessions cluster instead of spanning full day)
- Session times depend entirely on temporal model's training and output distribution

---

## Open Questions

### 1. Temporal Model Time Distribution
**Problem:** Sessions cluster in ~2 hour window instead of spanning workday

**Questions:**
- What training data was used for the temporal model?
- What does the mixture of Gaussians parameters look like?
- Should the model be retrained with different data?
- Is this expected behavior for some customer/machine configurations?

### 2. MRI_EXU_95 Trigger Behavior
**Current:** Scan sequence generated once when first MRI_EXU_95 token appears

**Questions:**
- Should scan sequences be generated for EVERY MRI_EXU_95 occurrence?
- Are there sessions that should NOT have scan sequences?
- What determines when/if MRI_EXU_95 appears in the PXChange sequence?

### 3. Patient ID Format
**Current:** Uses IDs from CSV (e.g., PAT001, PAT002)

**Question:** Should output use 40-character hex hashes like real data?

### 4. Body Group → Scan Correlation
**Current:** Body group used in conditioning, but scan body_part can differ

**Questions:**
- Should scan body_part always match bodygroup_from?
- Does the SeqofSeq model learn body-part-specific scan sequences?
- Should bodygroup be passed to SeqofSeq conditioning?

### 5. SeqofSeq Conditioning
**Current:** Uses 5 random features for SeqofSeq conditioning

**Question:** What should the 5 SeqofSeq conditioning features actually be?

### 6. Entity Type
**Current:** Always set to 0 (real patient)

**Question:** What are the other entity types (1, 2, 3)? When should they be used?

---

## File Locations

| Component | Path |
|-----------|------|
| Main Script | `UnifiedSchedulePipeline/generation/generate_daily_schedule.py` |
| Temporal Model | `UnifiedSchedulePipeline/datetime_model/temporal_schedule_model.py` |
| PXChange Seq Model | `PXChange_Refactored/models/conditional_sequence_generator.py` |
| PXChange Dur Model | `PXChange_Refactored/models/conditional_counts_generator.py` |
| SeqofSeq Seq Model | `SeqofSeq_Pipeline/models/conditional_sequence_generator.py` |
| SeqofSeq Dur Model | `SeqofSeq_Pipeline/models/conditional_duration_predictor.py` |
| Sample Patient CSV | `UnifiedSchedulePipeline/data/sample_patients.csv` |
| Output Schedules | `UnifiedSchedulePipeline/outputs/generated_schedules/` |

---

## Usage

```bash
# With patient CSV
python generate_daily_schedule.py --patient-csv data/sample_patients.csv

# Without CSV (generates 15 random patients)
python generate_daily_schedule.py

# Specify number of sessions
python generate_daily_schedule.py --num-sessions 20

# Full options
python generate_daily_schedule.py \
    --date 2026-01-15 \
    --machine-id 141049 \
    --patient-csv data/sample_patients.csv \
    --output-dir outputs/custom \
    --seed 42
```
