# Unified MR Machine Daily Schedule Prediction Pipeline

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Implementation Status](#implementation-status)
- [How It Works](#how-it-works)
- [Data Flow](#data-flow)
- [Phase Summaries](#phase-summaries)
- [Directory Structure](#directory-structure)
- [Usage Guide](#usage-guide)
- [Future Roadmap](#future-roadmap)

---

## Overview

The **Unified Schedule Pipeline** combines two independent MRI modeling systems into a single comprehensive daily schedule generator:

1. **PXChange_Refactored**: Models patient exchange events (scanner setup, table movements, measurement signals)
2. **SeqofSeq_Pipeline**: Models body region scan sequences (actual MRI protocols and timings)

### The Problem
Your existing systems are separate:
- PXChange data and SeqofSeq data come from different time periods
- No system predicts *when* patients arrive or *how many* sessions per day
- Models don't account for pause events (breaks, repositioning, equipment adjustments)

### The Solution
A three-tier architecture:
```
[Temporal Model] → Predicts daily structure (12 sessions, starting at 7:00, 7:45, 8:30...)
       ↓
[PXChange Models] → For each session: exchange events + measurement signals
       ↓
[SeqofSeq Models] → For each measurement: scan sequences + PAUSE events
       ↓
[Assembly] → Complete daily schedule as CSV files
```

---

## Architecture

### System Diagram
```
INPUT: Day metadata (day_of_year=165, day_of_week=3, machine_id=141049)
    │
    ├──→ [Temporal Feature Engineering]
    │    - Cyclical encoding (sin/cos for day of year, week)
    │    - Time of day features (morning/afternoon/evening)
    │    - Weekend detection
    │    → 12 temporal features
    │
    ├──→ [Temporal Schedule Model] (NEW - Transformer encoder)
    │    Input: 12 temporal features
    │    Output:
    │      - Session count (Poisson λ) → Sample: 12 sessions
    │      - Start times (Mixture of 3 Gaussians) → [25200s, 27900s, 30600s, ...]
    │    Architecture: 4-layer Transformer, d_model=128
    │
    └──→ For each session (1 to 12):
         │
         ├──→ [PXChange Adapter]
         │    ├─ ConditionalSequenceGenerator
         │    │  - Predicts: [START, MRI_MPT_1005, MRI_EXU_95, PAUSE, ...]
         │    │  - Vocab: 19 tokens (including PAUSE)
         │    └─ ConditionalCountsGenerator
         │       - Predicts: duration μ, σ for each event (Gamma distribution)
         │
         └──→ When MRI_EXU_95 detected (measurement start signal):
              │
              └──→ [SeqofSeq Adapter]
                   ├─ ConditionalSequenceGenerator
                   │  - Predicts: [START, LOCALIZER, T2_TSE, PAUSE, T1_VIBE, END]
                   │  - Vocab: ~35 tokens (including PAUSE)
                   └─ ConditionalDurationPredictor
                      - Predicts: scan duration μ, σ (Gamma distribution)
    │
    └──→ [Assembly & Validation]
         - Combine all events chronologically
         - Validate: no overlaps, realistic durations, proper sequencing
         - Format outputs
    │
    └──→ OUTPUTS:
         ├─ event_timeline.csv (chronological)
         └─ patient_sessions.csv (grouped by patient)
```

---

## Implementation Status

### ✅ Phase 1: Foundation & Setup (COMPLETE)

**Status:** 100% Complete

**What Was Built:**
1. **Directory Structure** - Full pipeline organization with 19 subdirectories
2. **Unified Configuration** (`config.py`) - Central management for all models
3. **PAUSE Token Integration** - Updated vocabularies in both PXChange and SeqofSeq
4. **Model Adapters** - Wrapper classes for consistent interfaces
5. **Pause Detection System** - Automatic PAUSE token injection (5-min threshold)
6. **Temporal Model Architecture** - Complete Transformer-based datetime predictor
7. **Temporal Feature Engineering** - 12-feature extraction system
8. **Data Utilities** - Temporal pattern extraction and 50x augmentation

**Key Files Created:**
- `config.py` (250 lines) - Unified configuration
- `pxchange_adapter/wrapper.py` (280 lines) - PXChange interface
- `seqofseq_adapter/wrapper.py` (270 lines) - SeqofSeq interface
- `preprocessing/pause_injection.py` (300 lines) - PAUSE detection
- `datetime_model/temporal_features.py` (300 lines) - Feature engineering
- `datetime_model/temporal_schedule_model.py` (350 lines) - Model architecture
- `datetime_model/temporal_data_utils.py` (400 lines) - Data extraction

**Technical Achievements:**
- Cyclical time encoding (day 1 and day 365 are 1 day apart, not 364!)
- Mixture of Gaussians for multi-modal start time distributions
- Smart PAUSE initialization (mean of existing embeddings)

---

### ✅ Phase 2: PAUSE Integration (60% COMPLETE)

**Status:** 60% Complete - Data preprocessing done, fine-tuning in progress

**What Was Built:**

#### 1. Fine-Tuning Infrastructure ✓
**File:** `training/retrain_with_pauses.py` (500 lines)

**Key Components:**
```python
class ModelExpander:
    """Expands model vocabularies for PAUSE token"""

    @staticmethod
    def expand_token_embedding(model, old_size, new_size):
        # Adds PAUSE token to embedding layer
        # Initializes as mean of existing embeddings
        # Prevents catastrophic forgetting

    @staticmethod
    def expand_output_projection(model, old_size, new_size):
        # Expands final layer for new vocabulary
```

**Functions:**
- `load_and_expand_pxchange_sequence_model()` - Loads + expands PXChange sequence model
- `load_and_expand_pxchange_duration_model()` - Loads + expands PXChange duration model
- `load_and_expand_seqofseq_sequence_model()` - Loads + expands SeqofSeq models
- `fine_tune_model()` - Fine-tuning loop with lower LR (1e-5), 20 epochs

**Why Fine-Tuning, Not Retraining?**
- Preserves existing knowledge (models already understand MRI workflows)
- Fast (20 epochs vs 100)
- Stable (10x lower learning rate)
- Just teaches PAUSE behavior

#### 2. Data Preprocessing ✓✓✓
**File:** `preprocessing/preprocess_all_data.py` (250 lines)

**Results:**
```
PXChange:
  ✓ Files processed: 40/40
  ✓ Total events: 141,201
  ✓ Output: PXChange_Refactored/data/preprocessed_with_pauses/

SeqofSeq:
  ✓ Files processed: 1/1 (176625.csv)
  ✓ Total scans: 4,527
  ✓ PAUSE tokens added: 374 (8.26% pause rate)
  ✓ Output: SeqofSeq_Pipeline/data/preprocessed_with_pauses/176625_with_pauses.csv
```

**PAUSE Detection Logic:**
- **Threshold:** 5 minutes (300 seconds)
- **Method:** Analyzes time gaps between consecutive events
- **Duration:** Actual gap time, capped at 600 seconds
- **Insertion:** Creates new row with sourceID='PAUSE' or Sequence='PAUSE'

**Example:**
```csv
# BEFORE:
datetime,sourceID,duration
2024-04-16 08:00:00,MRI_MSR_104,45
2024-04-16 08:08:30,MRI_FRR_256,12    ← 8.5 minute gap!

# AFTER:
datetime,sourceID,duration
2024-04-16 08:00:00,MRI_MSR_104,45
2024-04-16 08:00:45,PAUSE,465          ← PAUSE inserted (8.5 min = 510s)
2024-04-16 08:08:30,MRI_FRR_256,12
```

**Validation:**
```bash
$ grep -c "PAUSE" SeqofSeq_Pipeline/data/preprocessed_with_pauses/176625_with_pauses.csv
374  ✓ Confirmed

$ Pause rate: 374/4,527 = 8.26% ✓ Realistic
```

#### 3. Vocabulary Updates ✓
**PXChange config:**
```python
SOURCEID_VOCAB = {
    'PAD': 0,
    # ... existing 17 tokens ...
    'UNK': 17,
    'PAUSE': 18  # NEW
}
VOCAB_SIZE = 19  # Was 18
```

**SeqofSeq config:**
```python
SPECIAL_TOKENS = {
    'PAD': 0,
    'START': 1,
    'END': 2,
    'UNK': 3,
    'PAUSE': 4  # NEW
}
```

#### 4. Remaining Tasks (40%)
- ⏳ Fine-tune PXChange sequence model (infrastructure ready)
- ⏳ Fine-tune PXChange duration model (infrastructure ready)
- ⏳ Fine-tune SeqofSeq sequence model (infrastructure ready)
- ⏳ Fine-tune SeqofSeq duration model (infrastructure ready)
- ⏳ Validate models generate PAUSE tokens correctly

---

### ⏳ Phase 3: Temporal Model Training (PLANNED)

**Status:** 0% - Infrastructure built, ready to execute

**Goal:** Train the datetime prediction model to predict daily session structure

**Steps:**
1. **Extract temporal patterns from PXChange data**
   ```python
   # Analyze existing MRI logs
   daily_summaries = extract_temporal_patterns_from_pxchange('PXChange_Refactored/data/')

   # Output:
   # date       | day_of_week | num_sessions | session_start_times
   # 2024-04-16 | 2 (Wed)    | 8            | [25200, 27900, ...]
   # 2024-04-17 | 3 (Thu)    | 12           | [25800, 28200, ...]
   ```

2. **Augment data (50x multiplication)**
   ```python
   # Generate synthetic days based on real patterns
   augmented = augment_temporal_data(daily_summaries, augmentation_factor=50)

   # Real: 40 days → Synthetic: 2,000 days → Total: 2,040 samples
   ```

3. **Create training dataset**
   ```python
   dataset = create_temporal_training_dataset(augmented)

   # Features: [2040, 12] - temporal features
   # Targets:
   #   - num_sessions: [2040] - session counts
   #   - start_times: [2040, 20] - session start times (padded)
   ```

4. **Train TemporalScheduleModel**
   ```python
   model = TemporalScheduleModel(
       temporal_feature_dim=12,
       d_model=128,
       nhead=4,
       num_layers=4
   )

   # Training:
   # - Epochs: 100
   # - Loss: Poisson NLL (session count) + Mixture Gaussian NLL (timings)
   # - Early stopping: patience=15
   # - Learning rate: 0.001
   ```

5. **Validate predictions**
   ```python
   daily_structure = model.predict_daily_structure(temporal_features)

   # Output:
   # {
   #     'num_sessions': 12,
   #     'session_start_times': [25200, 27900, 30600, ...],  # seconds from midnight
   #     'session_gaps': [2700, 2700, ...]  # seconds between sessions
   # }
   ```

**Expected Deliverables:**
- Trained model: `saved_models/temporal_schedule_model/temporal_model_best.pth`
- Training data: `data/temporal_training_data/temporal_training_dataset.pkl`
- Validation metrics: MAE on session counts, KL divergence on timings

---

### ⏳ Phase 4: Pipeline Integration (PLANNED)

**Status:** 0% - Design complete, ready to implement

**Goal:** Build the orchestrator that combines all models

**Components to Build:**

#### 1. UnifiedScheduleOrchestrator
**File:** `unified_generation/orchestrator.py`

```python
class UnifiedScheduleOrchestrator:
    """Coordinates all models to generate complete daily schedules"""

    def __init__(self, temporal_model, pxchange_adapter, seqofseq_adapter):
        self.temporal_model = temporal_model
        self.pxchange = pxchange_adapter
        self.seqofseq = seqofseq_adapter

    def generate_daily_schedule(self, day_of_year, day_of_week, machine_id, seed=None):
        """
        Main generation pipeline

        Steps:
        1. Extract temporal features
        2. Predict daily structure (num_sessions, start_times)
        3. For each session:
           a. Generate PXChange event sequence
           b. For each MRI_EXU_95 (measurement start):
              - Generate SeqofSeq scan sequence
        4. Assemble all events chronologically
        5. Validate and adjust (no overlaps, realistic durations)
        6. Format outputs

        Returns:
            schedule_df: Complete daily schedule
        """
```

**Key Methods:**
- `_build_temporal_features()` - Extract features from date
- `_generate_pxchange_session()` - Generate one patient exchange session
- `_generate_seqofseq_scans()` - Generate scan sequence for measurement
- `_assemble_schedule()` - Combine all events chronologically
- `_validate_and_adjust()` - Fix overlaps, check constraints

#### 2. Output Formatters
**Files:** `output_formatters/event_timeline.py`, `output_formatters/patient_sessions.py`

**Event Timeline Format:**
```csv
event_id,timestamp,datetime,event_type,session_id,patient_id,sourceID,scan_sequence,body_part,duration,cumulative_time
0,25200,2024-01-15 07:00:00,pxchange,0,P001,MRI_MPT_1005,,,5.2,25200
1,25205,2024-01-15 07:00:05,pxchange,0,P001,MRI_EXU_95,,,3.1,25205
2,25208,2024-01-15 07:00:08,scan,0,P001,,LOCALIZER,HEAD,29.3,25208
3,25237,2024-01-15 07:00:37,scan,0,P001,,T2_TSE_DARK_FLUID,HEAD,156.7,25237
4,25394,2024-01-15 07:03:14,pause,0,P001,,PAUSE,,180.2,25394
```

**Patient Sessions Format:**
```csv
session_id,patient_id,session_start,session_end,duration,num_scans,num_pauses,body_parts
0,P001,25200,27895,2695,12,2,HEAD
1,P002,28800,31234,2434,10,1,ABDOMEN
2,P003,32400,35678,3278,15,3,SPINE
```

#### 3. Validation & Constraint Checking
**File:** `unified_generation/validation.py`

**Constraints:**
- ✓ Total duration: 6-14 hours
- ✓ Session count: 3-25 per day
- ✓ Session duration: 10-90 minutes
- ✓ Inter-session gaps: 2-120 minutes
- ✓ No event overlaps
- ✓ Proper sequencing (exchange before scans)
- ✓ Realistic pause durations (60-600 seconds)

**Validation Flow:**
```python
def validate_schedule(schedule_df):
    violations = []

    # Check duration
    total_duration = (schedule_df['timestamp'].max() - schedule_df['timestamp'].min()) / 3600
    if not (6 <= total_duration <= 14):
        violations.append(f"Total duration {total_duration:.1f}h outside 6-14h range")

    # Check overlaps
    for i in range(len(schedule_df) - 1):
        current_end = schedule_df.iloc[i]['timestamp'] + schedule_df.iloc[i]['duration']
        next_start = schedule_df.iloc[i+1]['timestamp']
        if current_end > next_start:
            violations.append(f"Overlap at event {i}")
            # Auto-fix: adjust next_start = current_end + 1

    return violations, adjusted_schedule
```

---

### ⏳ Phase 5: Testing & Validation (PLANNED)

**Goal:** Ensure generated schedules are realistic and meet quality metrics

**Tasks:**
1. **Generate test schedules** for various scenarios:
   - Weekday vs weekend
   - Different machines
   - Different seasons (day of year)

2. **Compute quality metrics:**
   - Total duration distribution
   - Session count distribution
   - Inter-session gap distribution
   - PAUSE token frequency
   - Scan type diversity

3. **Manual validation:**
   - Review sample schedules with domain experts
   - Verify realistic workflows
   - Check for impossible sequences

4. **Hyperparameter tuning:**
   - Temporal model sampling temperature
   - Sequence model top-k, top-p values
   - PAUSE detection threshold

5. **Constraint satisfaction testing:**
   - Measure violation rate
   - Test auto-adjustment logic
   - Stress test edge cases

**Deliverables:**
- Validation metrics report
- Sample schedules (5-10 days)
- Quality benchmark suite

---

### ⏳ Phase 6: Production Readiness (PLANNED)

**Goal:** Package for end-user deployment

**Tasks:**
1. **Command-Line Interface**
   ```bash
   python generate_schedule.py \
       --day-of-year 165 \
       --day-of-week 3 \
       --machine-id 141049 \
       --output-dir outputs/schedules/ \
       --seed 42
   ```

2. **Documentation:**
   - User guide
   - API documentation
   - Example notebooks
   - Troubleshooting guide

3. **Example notebooks:**
   - `01_generating_single_day.ipynb`
   - `02_batch_generation.ipynb`
   - `03_analyzing_schedules.ipynb`
   - `04_custom_constraints.ipynb`

4. **Performance optimization:**
   - Model quantization for faster inference
   - Batch generation
   - GPU utilization
   - Caching common conditioning vectors

5. **Final testing:**
   - Integration tests
   - End-to-end workflow tests
   - Performance benchmarks

---

## How It Works

### 1. Temporal Feature Engineering

**Input:** Date (e.g., "June 15, 2024")

**Processing:**
```python
# Cyclical encoding for day of year
day_of_year = 167
angle = 2π * 167 / 365
day_of_year_sin = sin(angle)  # 0.78
day_of_year_cos = cos(angle)  # 0.62

# Why? Day 1 and Day 365 are 1 day apart, not 364!
# Linear: |1 - 365| = 364 ❌
# Cyclical: |sin(2π*1/365) - sin(2π*365/365)| ≈ 0 ✓
```

**Output:** 12 features
```python
{
    'day_of_year_sin': 0.78,
    'day_of_year_cos': 0.62,
    'day_of_week_sin': -0.43,
    'day_of_week_cos': -0.90,
    'day_of_month': 15,
    'week_of_year': 24,
    'is_weekend': 0,
    'is_morning': 0,
    'is_afternoon': 1,
    'is_evening': 0,
    'machine_id_encoded': 141049,
    'typical_daily_load': 12.0
}
```

---

### 2. Temporal Schedule Prediction

**Model:** Transformer encoder with dual output heads

**Input:** 12 temporal features → **[1, 12]** tensor

**Processing:**
```
Feature Projection (12 → 128)
    ↓
Transformer Encoder (4 layers, 128-dim)
    ↓
    ├─→ Session Count Head → Poisson λ
    │   Output: λ = 12.3 → Sample: 12 sessions
    │
    └─→ Timing Mixture Head → 3 Gaussians
        Output:
          Component 1: μ=27000s (7:30 AM), σ=1800s, w=0.4
          Component 2: μ=39600s (11:00 AM), σ=2400s, w=0.35
          Component 3: μ=54000s (3:00 PM), σ=3000s, w=0.25

        Sample 12 times:
          [25200, 27900, 30600, 33300, 36000, 38700, 41400, 44100, 46800, 49500, 52200, 54900]
          [7:00,  7:45,  8:30,  9:15,  10:00, 10:45, 11:30, 12:15, 1:00,  1:45,  2:30,  3:15 ]
```

**Why Mixture of Gaussians?**
- Single Gaussian → assumes uniform spread
- Mixture → captures morning rush, lunch lull, afternoon sessions
- Real MRI facilities have multi-modal patterns

---

### 3. Session Generation (PXChange)

For each predicted session start time:

**Input:** Conditioning features
```python
conditioning = {
    'age': 60,
    'weight': 80,
    'height': 1.75,
    'bodygroup_from': 'HEAD',
    'bodygroup_to': 'HEAD',
    'ptab': -500000
}
```

**PXChange Sequence Model:**
```
Conditioning → Transformer Encoder → Memory
    ↓
Auto-regressive Generation:
  t=0: START → predict next token
  t=1: MRI_MPT_1005 (patient registration)
  t=2: MRI_FRR_18 (equipment check)
  t=3: MRI_EXU_95 (MEASUREMENT START! ← triggers scan sequence)
  t=4: PAUSE (patient repositioning)
  t=5: MRI_MSR_104 (measurement finished)
  t=6: END
```

**PXChange Duration Model:**
```
Conditioning + Sequence Tokens → Cross-Attention Transformer
    ↓
For each token, predict:
  μ (mean duration), σ (std dev)

Sample from Gamma(α, β):
  α = (μ/σ)²
  β = μ/σ²

Example:
  MRI_MPT_1005: μ=5.2s, σ=1.1s → sample: 5.7s
  MRI_EXU_95: μ=3.1s, σ=0.8s → sample: 2.9s
  PAUSE: μ=180s, σ=45s → sample: 165s
```

---

### 4. Scan Sequence Generation (SeqofSeq)

When MRI_EXU_95 detected → trigger scan sequence:

**Input:** Conditioning
```python
conditioning = {
    'bodypart': 'HEAD',
    'systemtype': 'VIDA',
    'country': 'KR',
    'group': 'Brain',
    'coil_config': {...}  # 88 coil settings
}
```

**SeqofSeq Sequence Model:**
```
Auto-regressive Generation:
  t=0: START
  t=1: LOCALIZER (quick reference scan)
  t=2: T2_TSE_DARK_FLUID (detailed brain imaging)
  t=3: PAUSE (patient break)
  t=4: T1_VIBE_DIXON (contrast imaging)
  t=5: DWI_EPI (diffusion scan)
  t=6: END
```

**SeqofSeq Duration Model:**
```
Predict durations:
  LOCALIZER: μ=29.3s
  T2_TSE_DARK_FLUID: μ=156.7s
  PAUSE: μ=180.2s
  T1_VIBE_DIXON: μ=95.4s
  DWI_EPI: μ=112.8s
```

---

### 5. Assembly & Validation

**Combine all events chronologically:**
```python
all_events = []

for session_idx, session_start in enumerate(session_start_times):
    # Generate PXChange events
    pxchange_events = pxchange_adapter.generate_complete_session(...)

    for event in pxchange_events:
        if event['sourceID'] == 'MRI_EXU_95':
            # Trigger scan sequence
            scan_events = seqofseq_adapter.generate_complete_sequence(...)
            all_events.extend(scan_events)
        else:
            all_events.append(event)

# Sort by timestamp
schedule = sort_by_timestamp(all_events)

# Validate
validate_no_overlaps(schedule)
validate_duration_constraints(schedule)
validate_session_counts(schedule)

# Format outputs
event_timeline_csv = format_event_timeline(schedule)
patient_sessions_csv = format_patient_sessions(schedule)
```

---

## Data Flow

### Training Data Flow
```
PXChange Raw CSV (40 files)
  ├─ 141049.csv (4,100 events)
  ├─ 155687.csv
  └─ ... (38 more)
    │
    └─→ [PAUSE Injection] (5-min threshold)
        │
        ├─ preprocessed_with_pauses/
        │   ├─ 141049_with_pauses.csv (4,102 events, +2 PAUSE)
        │   └─ ... (40 files total)
        │
        └─→ [Sequence Preprocessing]
            └─ Group by PatientID → Sequences (min=3, max=128)

SeqofSeq Raw CSV (1 file)
  └─ 176625.csv (4,153 scans)
    │
    └─→ [PAUSE Injection]
        │
        └─ 176625_with_pauses.csv (4,527 scans, +374 PAUSE = 8.26%)
        │
        └─→ [Sequence Preprocessing]
            └─ Group by PatientID → Sequences (min=3, max=64)

Temporal Training Data
  │
  ├─→ [Extract from PXChange logs]
  │   - Identify session starts (MRI_EXU_95)
  │   - Group by date
  │   - Calculate: num_sessions, start_times per day
  │   → ~40 real daily summaries
  │
  ├─→ [Augment 50x]
  │   - Fit Poisson to session counts
  │   - Fit Gaussian Mixture to start times
  │   - Generate 2,000 synthetic days
  │   → 2,040 total samples
  │
  └─→ temporal_training_dataset.pkl
      Features: [2040, 12]
      Targets: {num_sessions: [2040], start_times: [2040, 20]}
```

### Inference Data Flow
```
User Input: day_of_year=165, day_of_week=3, machine_id=141049
    │
    ├─→ [Temporal Features] → [12 features]
    │
    ├─→ [Temporal Model] → {num_sessions: 12, start_times: [...]}
    │
    └─→ For each session (12 iterations):
        │
        ├─→ [Sample Conditioning]
        │   - age, weight, height (random from distributions)
        │   - bodygroup_from/to (HEAD, ABDOMEN, SPINE, etc.)
        │   - ptab (random)
        │
        ├─→ [PXChange Adapter]
        │   Input: conditioning
        │   Output:
        │     - tokens: [START, MRI_MPT_1005, MRI_EXU_95, PAUSE, MRI_MSR_104, END]
        │     - durations: [0, 5.2, 3.1, 165, 12.3, 0]
        │
        └─→ For each MRI_EXU_95:
            │
            └─→ [SeqofSeq Adapter]
                Input: conditioning (bodypart, coils, etc.)
                Output:
                  - tokens: [START, LOCALIZER, T2_TSE, PAUSE, T1_VIBE, END]
                  - durations: [0, 29.3, 156.7, 180.2, 95.4, 0]
    │
    ├─→ [Assembly]
    │   - Flatten all events
    │   - Add timestamps (cumulative from session_start)
    │   - Sort chronologically
    │
    ├─→ [Validation]
    │   - Check overlaps → adjust if needed
    │   - Verify constraints → reject/retry if violated
    │
    └─→ [Format Outputs]
        ├─ event_timeline.csv (chronological)
        └─ patient_sessions.csv (grouped)
```

---

## Directory Structure

```
UnifiedSchedulePipeline/
│
├── README.md                          ← You are here
├── PHASE1_SUMMARY.md                  ← Phase 1 details (Foundation)
├── PHASE2_SUMMARY.md                  ← Phase 2 details (PAUSE integration)
├── config.py                          ← Central configuration (250 lines)
│
├── datetime_model/                    ← Temporal prediction system
│   ├── __init__.py
│   ├── temporal_features.py           ← Feature engineering (300 lines)
│   ├── temporal_schedule_model.py     ← Model architecture (350 lines)
│   ├── temporal_data_utils.py         ← Data extraction (400 lines)
│   └── train_temporal_model.py        ← Training script (TODO)
│
├── pxchange_adapter/                  ← PXChange model interface
│   ├── __init__.py
│   └── wrapper.py                     ← Adapter class (280 lines)
│
├── seqofseq_adapter/                  ← SeqofSeq model interface
│   ├── __init__.py
│   └── wrapper.py                     ← Adapter class (270 lines)
│
├── preprocessing/                     ← Data preprocessing
│   ├── __init__.py
│   ├── pause_injection.py             ← PAUSE detection (300 lines)
│   └── preprocess_all_data.py         ← Batch processing (250 lines)
│
├── unified_generation/                ← Pipeline orchestration (TODO)
│   ├── __init__.py
│   ├── orchestrator.py                ← Main coordinator (TODO)
│   ├── session_generator.py           ← Per-session generation (TODO)
│   ├── assembly.py                    ← Event assembly (TODO)
│   └── validation.py                  ← Constraint checking (TODO)
│
├── output_formatters/                 ← CSV formatting (TODO)
│   ├── __init__.py
│   ├── event_timeline.py              ← Chronological format (TODO)
│   └── patient_sessions.py            ← Patient-grouped format (TODO)
│
├── training/                          ← Model training scripts
│   ├── __init__.py
│   └── retrain_with_pauses.py         ← Fine-tuning with PAUSE (500 lines)
│
├── evaluation/                        ← Quality metrics (TODO)
│   ├── __init__.py
│   ├── temporal_metrics.py            ← Datetime accuracy (TODO)
│   └── schedule_metrics.py            ← Schedule quality (TODO)
│
├── data/                              ← Training data
│   ├── temporal_training_data/
│   │   ├── real_daily_summaries.csv   ← Extracted patterns (TODO)
│   │   ├── augmented_daily_summaries.csv  ← 50x augmented (TODO)
│   │   └── temporal_training_dataset.pkl  ← Model-ready (TODO)
│   │
│   ├── pxchange_preprocessed/         ← Links to preprocessed data
│   └── seqofseq_preprocessed/         ← Links to preprocessed data
│
├── saved_models/                      ← Model checkpoints
│   ├── temporal_schedule_model/
│   │   └── temporal_model_best.pth    ← Trained temporal model (TODO)
│   │
│   ├── pxchange_models/               ← Fine-tuned models (IN PROGRESS)
│   │   ├── sequence_model_with_pause.pth
│   │   └── duration_model_with_pause.pth
│   │
│   └── seqofseq_models/               ← Fine-tuned models (IN PROGRESS)
│       ├── sequence_model_with_pause.pth
│       └── duration_model_with_pause.pth
│
└── outputs/                           ← Generated schedules
    ├── generated_schedules/           ← Complete schedules (TODO)
    ├── event_timelines/               ← Chronological CSVs (TODO)
    └── patient_sessions/              ← Grouped CSVs (TODO)
```

---

## Usage Guide

### Current Usage (Phase 1-2)

#### 1. Preprocess Data with PAUSE Tokens
```bash
cd UnifiedSchedulePipeline/preprocessing
python preprocess_all_data.py
```

**Output:**
- `PXChange_Refactored/data/preprocessed_with_pauses/` (40 files)
- `SeqofSeq_Pipeline/data/preprocessed_with_pauses/` (1 file)

#### 2. Fine-Tune Models (In Progress)
```bash
cd UnifiedSchedulePipeline/training
python retrain_with_pauses.py
```

**What it does:**
- Loads existing models
- Expands vocabularies for PAUSE token
- Fine-tunes with lower learning rate
- Saves updated models

---

### Future Usage (Phase 3-6)

#### 3. Train Temporal Model
```bash
python datetime_model/train_temporal_model.py \
    --augmentation-factor 50 \
    --epochs 100
```

#### 4. Generate Daily Schedule
```bash
python generate_schedule.py \
    --day-of-year 165 \
    --day-of-week 3 \
    --machine-id 141049 \
    --output-dir outputs/schedules/ \
    --seed 42
```

**Output:**
```
outputs/schedules/
├── event_timeline_day165.csv          ← Chronological events
└── patient_sessions_day165.csv        ← Patient-grouped sessions
```

#### 5. Batch Generation
```bash
python generate_schedule.py \
    --batch \
    --start-day 1 \
    --end-day 365 \
    --machine-id 141049
```

---

## Future Roadmap

### Phase 3: Temporal Model (Week 3)
- [ ] Extract temporal patterns from PXChange data
- [ ] Generate augmented training data (50x)
- [ ] Train TemporalScheduleModel
- [ ] Validate session count and timing predictions
- [ ] Save trained model

### Phase 4: Pipeline Integration (Week 4)
- [ ] Implement UnifiedScheduleOrchestrator
- [ ] Build session generation logic
- [ ] Create assembly and validation modules
- [ ] Implement output formatters
- [ ] End-to-end integration test

### Phase 5: Testing & Validation (Week 5)
- [ ] Generate test schedules
- [ ] Compute quality metrics
- [ ] Manual domain expert review
- [ ] Hyperparameter tuning
- [ ] Stress testing

### Phase 6: Production Readiness (Week 6)
- [ ] CLI implementation
- [ ] User documentation
- [ ] Example notebooks
- [ ] Performance optimization
- [ ] Final deployment testing

---

## Technical Deep Dives

### Why Transformers?
- **Self-attention:** Captures long-range dependencies (scan at position 1 influences scan at position 50)
- **Parallel processing:** Faster than RNNs
- **State-of-the-art:** Your existing models already use Transformers
- **Proven:** Works for sequence modeling (NLP, time series, protein folding)

### Why Gamma Distribution for Durations?
- **Positive values only:** Durations can't be negative
- **Right-skewed:** Most scans are quick, some are long (Gamma models this)
- **Two parameters:** μ (mean) and σ (std dev) capture uncertainty
- **Better than:** Normal (allows negatives), Poisson (integer only), Exponential (memoryless)

### Why Mixture of Gaussians for Start Times?
- **Multi-modal:** MRI facilities have multiple "rush hours"
  - Morning: 7-10 AM (early appointments)
  - Midday: 11 AM-1 PM (pre-lunch)
  - Afternoon: 2-5 PM (post-lunch)
- **Single Gaussian:** Would predict average (noon) ← unrealistic
- **Mixture:** Learns natural clustering

### Why 50x Augmentation?
- **Deep learning needs data:** 40 real days → 2,000 synthetic days
- **Preserves distributions:** Fits statistical models to real data
- **Controlled variation:** Doesn't create impossible scenarios
- **Standard practice:** GANs, data augmentation in CV, transfer learning

---

## Contributing

### Code Style
- Follow PEP 8
- Type hints for function signatures
- Docstrings for all public methods
- Comments for complex logic

### Testing
- Unit tests for each module
- Integration tests for pipeline
- Validation tests for generated schedules

### Documentation
- Update README for new features
- Add docstrings
- Create example notebooks

---

## References

### Papers
- "Attention is All You Need" (Vaswani et al., 2017) - Transformer architecture
- "Gamma Distribution for Duration Modeling" (Various)
- "Mixture Models for Multi-Modal Distributions" (Various)

### Your Existing Codebases
- `PXChange_Refactored/` - Patient exchange modeling
- `SeqofSeq_Pipeline/` - Scan sequence modeling

---

## License & Contact

**Project:** Unified MR Machine Schedule Prediction
**Author:** Luke (with Claude Code assistance)
**Date:** December 2025
**Status:** Phase 2 (60% complete)

For questions or issues, see the implementation plan in `warm-herding-turtle.md`.

---

**Last Updated:** December 5, 2025
**Version:** 0.2.0 (Phase 2 in progress)
