# MRI Digital Twin - Alternating Pipeline

## Overview

This project implements a **digital twin** for MRI facility workflows, generating synthetic examination schedules that match real-world patterns. The core approach uses a **sequential alternating model** that mimics how MRI examinations actually occur:

```
Exchange → Examination → Exchange → Examination → ... → Exchange (final)
```

## Architecture

### Key Insight (from Meeting Transcript)

> "You can just think about body from→to buckets. Produce 1000 samples per bucket.
> To generate a day, you don't need to rerun models - just pick random samples
> from already generated buckets."

### Two Main Models

1. **Exchange Model**: Predicts events during body region transitions (patient setup/breakdown, coil changes, table movements)
   - Input: Patient conditioning (age, weight, height, PTAB, direction) + current body region
   - Output: Event sequence for the transition phase

2. **Examination Model**: Generates MRI scan sequences for a specific body region
   - Input: Patient conditioning + body region being examined
   - Output: Scan event sequence (MRI_EXU_95 markers, measurements, etc.)

### Bucket-Based Generation

Instead of running models on-the-fly, we pre-generate **1000 samples per bucket**:
- **Exchange buckets**: One bucket per body region transition (e.g., HEAD→CHEST)
- **Examination buckets**: One bucket per body region (e.g., HEAD examinations)

Day simulation then simply **samples randomly from these buckets**, making generation instant.

## Project Structure

```
Time-Series-Models/
├── AlternatingPipeline/           # Main implementation
│   ├── config.py                  # Central configuration
│   ├── models/
│   │   ├── exchange_model.py      # Body region transition model
│   │   └── examination_model.py   # Scan sequence generator
│   ├── data/
│   │   └── preprocessing.py       # Extract exchange/examination events
│   ├── training/
│   │   ├── train_exchange.py      # Train exchange model
│   │   └── train_examination.py   # Train examination model
│   ├── generation/
│   │   ├── bucket_generator.py    # Pre-generate 1000 samples/bucket
│   │   └── day_simulator.py       # Sample from buckets for day
│   ├── validation/
│   │   └── metrics.py             # Real vs predicted comparison
│   ├── buckets/                   # Pre-generated samples
│   ├── saved_models/              # Trained models
│   └── outputs/                   # Generated schedules
│
├── PXChange_Refactored/           # Raw data & legacy models
│   └── data/*.csv                 # 135 raw MRI event log CSVs
│
├── _archive/                      # Archived old implementations
│
└── docs/                          # Documentation
    ├── Transcript of Meeting.md   # Requirements source
    ├── Context_PatientWorkflow.md # Workflow context
    └── Instructions_New_MeetingNotes.md
```

## Quick Start

### 1. Preprocess Data

```bash
cd AlternatingPipeline
python -c "from data.preprocessing import preprocess_all_data; preprocess_all_data()"
```

### 2. Train Models

```bash
# Train Exchange Model
python training/train_exchange.py

# Train Examination Model
python training/train_examination.py
```

### 3. Generate Buckets

```python
from models import create_exchange_model, create_examination_model
from generation import BucketGenerator

# Load trained models
exchange_model = create_exchange_model()
exchange_model.load_state_dict(torch.load('saved_models/exchange/exchange_model_best.pt'))

examination_model = create_examination_model()
examination_model.load_state_dict(torch.load('saved_models/examination/examination_model_best.pt'))

# Generate buckets (1000 samples each)
generator = BucketGenerator(exchange_model, examination_model)
generator.generate_all_buckets()
generator.save_buckets()
```

### 4. Simulate a Day

```python
from generation import DaySimulator

# Define ground truth patient sequence
patients = [
    {'patient_id': 'PAT001', 'body_region': 'HEAD'},
    {'patient_id': 'PAT002', 'body_region': 'CHEST'},
    {'patient_id': 'PAT003', 'body_region': 'SPINE'},
]

# Simulate
simulator = DaySimulator(buckets_dir='buckets')
schedule = simulator.simulate_day(patients)
simulator.save_schedule(schedule, 'generated_day.csv')
```

## Data Format

### Input: Raw Event Log CSVs

Each CSV contains MRI machine events with columns:
- `datetime`, `sourceID`, `timediff`
- `BodyGroup_from`, `BodyGroup_to` (body region transitions)
- `PatientId`, `Age`, `Weight`, `Height`, `Direction`
- Coil elements: `BC`, `SP1-8`, `HE1-4`, etc.
- Hardware signals: `ZAxisInPossible`, `YAxisDownPossible`, etc.

### Output: Generated Schedule CSV

```csv
event_id,timestamp,datetime,event_type,patient_id,sourceID,body_region,body_from,body_to,duration
0,0.0,2024-04-16T07:00:00,exchange,,MRI_FRR_2,,START,HEAD,5.2
1,5.2,2024-04-16T07:00:05,exchange,,MRI_CCS_11,,START,HEAD,3.1
2,8.3,2024-04-16T07:00:08,examination,PAT001,MRI_EXU_95,HEAD,,,29.3
...
```

## Key Configuration

Edit `AlternatingPipeline/config.py`:

```python
# Bucket configuration
BUCKET_SIZE = 1000  # Samples per bucket

# Body regions
BODY_REGIONS = ['HEAD', 'NECK', 'CHEST', 'ABDOMEN', 'PELVIS',
                'SPINE', 'ARM', 'LEG', 'HAND', 'FOOT', 'UNKNOWN']

# Model architectures
EXCHANGE_MODEL_CONFIG = {...}
EXAMINATION_MODEL_CONFIG = {...}
```

## Validation

Compare real vs predicted schedules:

```python
from validation import compare_real_vs_predicted, print_comparison_report

metrics = compare_real_vs_predicted(real_schedule, predicted_schedule)
print_comparison_report(metrics)
```

Outputs metrics for dashboard integration:
- Total duration (real vs predicted)
- Per-body-region examination times
- Per-exchange-type transition times
- Event type distributions

## Customer-Specific Models

Currently focused on **customer-specific models** (one model per MRI machine). Future work may expand to multi-customer modeling.

## References

- `Transcript of Meeting.md` - Original requirements discussion
- `Context_PatientWorkflow.md` - Patient workflow context
- `Instructions_New_MeetingNotes.md` - Implementation notes

## License

Internal use only.
