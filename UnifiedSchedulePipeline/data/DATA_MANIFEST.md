# Data Manifest - Unified Schedule Pipeline

## Preprocessed Data Locations

### PXChange Data with PAUSE Tokens
**Location:** `../../PXChange_Refactored/data/preprocessed_with_pauses/`
**Files:** 40 CSV files
**Format:** Patient exchange events with PAUSE tokens inserted
**Total Events:** 141,201
**File Pattern:** `{machine_id}_with_pauses.csv`

**Example files:**
- `141049_with_pauses.csv`
- `155687_with_pauses.csv`
- `175693_with_pauses.csv`
- ... (37 more)

**Columns:**
- datetime, sourceID, text, timediff, Age, Weight, Height
- BodyGroup_from, BodyGroup_to, PTAB, Position, Direction
- PatientId, dataset_id
- **sourceID includes:** PAD, MRI_CCS_11, ..., UNK, **PAUSE**

---

### SeqofSeq Data with PAUSE Tokens
**Location:** `../../SeqofSeq_Pipeline/data/preprocessed_with_pauses/`
**Files:** 1 CSV file
**Format:** MRI scan sequences with PAUSE tokens inserted
**Total Scans:** 4,527
**PAUSE Tokens:** 374 (8.26%)

**File:** `176625_with_pauses.csv`

**Columns:**
- BodyPart, Sequence, Protocol, PatientID, SN
- startTime, endTime, duration
- Country, Systemtype, Group
- 88 coil configuration columns (#0_*, #1_*)
- dataset_id
- **Sequence includes:** PAUSE token

---

### Original Training Data (For Reference)

#### PXChange Original
**Location:** `../../PXChange_Refactored/data/`
**Files:** 40 CSV files (141049.csv, 155687.csv, ...)
**Status:** Original raw data (no PAUSE tokens)

#### SeqofSeq Original
**Location:** `../../SeqofSeq_Pipeline/data/`
**Files:** 1 CSV file (176625.csv)
**Status:** Original raw data (no PAUSE tokens)

---

## Temporal Training Data

### Status: TODO (Phase 3)

**Target Location:** `temporal_training_data/`

**Files to be created:**
1. `real_daily_summaries.csv` - Extracted from PXChange
2. `augmented_daily_summaries.csv` - 50x augmented
3. `temporal_training_dataset.pkl` - Model-ready format

**Schema for daily summaries:**
```
date, day_of_week, day_of_year, num_sessions, session_start_times, machine_id, is_synthetic
```

---

## Model Checkpoints

### Existing Models (Original - No PAUSE Support)
**PXChange:**
- `../../PXChange_Refactored/saved_models/sequence_model_best.pth`
- `../../PXChange_Refactored/saved_models/duration_model_best.pth`

**SeqofSeq:**
- `../../SeqofSeq_Pipeline/saved_models/sequence_model_best.pth`
- `../../SeqofSeq_Pipeline/saved_models/duration_model_best.pth`

### Fine-Tuned Models (With PAUSE Support)
**Target Location:** `../saved_models/`

**PXChange (TO BE CREATED):**
- `pxchange_models/sequence_model_with_pause.pth`
- `pxchange_models/duration_model_with_pause.pth`

**SeqofSeq (TO BE CREATED):**
- `seqofseq_models/sequence_model_with_pause.pth`
- `seqofseq_models/duration_model_with_pause.pth`

**Temporal Model (TO BE CREATED):**
- `temporal_schedule_model/temporal_model_best.pth`

---

## Data Loading Examples

### Load PXChange Preprocessed Data
```python
import pandas as pd
import glob

# Load all preprocessed PXChange files
data_dir = '../../PXChange_Refactored/data/preprocessed_with_pauses/'
all_files = glob.glob(f'{data_dir}/*_with_pauses.csv')

# Load into single dataframe
dfs = []
for file in all_files:
    df = pd.read_csv(file)
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
print(f"Total events: {len(combined_df):,}")
print(f"PAUSE events: {(combined_df['sourceID'] == 'PAUSE').sum():,}")
```

### Load SeqofSeq Preprocessed Data
```python
import pandas as pd

# Load SeqofSeq data
data_path = '../../SeqofSeq_Pipeline/data/preprocessed_with_pauses/176625_with_pauses.csv'
df = pd.read_csv(data_path)

print(f"Total scans: {len(df):,}")
print(f"PAUSE events: {(df['Sequence'] == 'PAUSE').sum():,}")
```

---

## Data Statistics

### PXChange
- **Total Files:** 40
- **Total Events:** 141,201
- **Average Events per File:** 3,530
- **Date Range:** 2024 (various dates)
- **Machines:** 40 unique machine IDs

### SeqofSeq
- **Total Files:** 1
- **Total Scans:** 4,527
- **PAUSE Events:** 374 (8.26%)
- **Unique Patients:** 362
- **Unique Sequences:** 359
- **Average Sequence Length:** 11.4 scans
- **Country:** KR (Korea)
- **System Type:** VIDA

---

## Preprocessing Parameters

### PAUSE Detection
- **Threshold:** 5 minutes (300 seconds)
- **Min Duration:** 60 seconds
- **Max Duration:** 600 seconds

### Configuration
Defined in: `../config.py`
```python
PAUSE_DETECTION_THRESHOLD_MINUTES = 5
PAUSE_DURATION_MIN_SECONDS = 60
PAUSE_DURATION_MAX_SECONDS = 600
```

---

## Data Integrity Checks

### PXChange
✓ All files have PAUSE tokens or no gaps > 5 minutes
✓ sourceID vocabulary includes PAUSE (ID 18)
✓ datetime columns properly formatted
✓ No missing critical columns

### SeqofSeq
✓ PAUSE tokens: 374 confirmed
✓ Pause rate: 8.26% (realistic)
✓ Sequence column includes PAUSE
✓ All scans have valid timestamps

---

Last Updated: December 5, 2025
