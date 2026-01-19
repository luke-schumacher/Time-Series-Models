# Training & Generation Guide

## Current Training Status

Training is running in the background for all 5 models:
1. **Temporal Model** - Session count and timing predictions
2. **PXChange Sequence Model** - Scan sequence generation for PXChange data
3. **PXChange Duration Model** - Duration predictions for PXChange scans
4. **SeqofSeq Sequence Model** - Scan sequence generation for SeqofSeq data
5. **SeqofSeq Duration Model** - Duration predictions for SeqofSeq scans

**Estimated completion time:** 6-8 hours (training 100 epochs per model)

---

## Monitoring Training Progress

### Quick Status Check
To quickly check the current training status:

```bash
cd UnifiedSchedulePipeline/training
python check_status.py
```

This will show:
- Overall progress (X/5 models completed)
- Currently training model
- Current epoch and validation loss
- Status of each model

### Continuous Monitoring
To monitor training continuously with updates every 5 minutes:

```bash
cd UnifiedSchedulePipeline/training
python monitor_training.py
```

Or with custom update interval (in seconds):

```bash
python monitor_training.py 600  # Update every 10 minutes
```

The monitor will automatically notify when all training is complete.

### Check Raw Logs
To view the full training log:

```bash
cd UnifiedSchedulePipeline/training
tail -100 training_log_new.txt
```

---

## When Training Completes

Once all 5 models are trained, you can generate complete daily MRI schedules.

### Step 1: Check Trained Models

Verify all models are saved:

```bash
ls saved_models/*/
```

You should see:
- `temporal_schedule_model/temporal_model_best.pth`
- `pxchange_models/sequence_model_best.pth`
- `pxchange_models/duration_model_best.pth`
- `seqofseq_models/sequence_model_best.pth`
- `seqofseq_models/duration_model_best.pth`

### Step 2: Generate Prototype Schedule

The complete pipeline generates a full daily schedule:

```bash
cd UnifiedSchedulePipeline
python generate_complete_schedule.py
```

This will:
1. Use the Temporal Model to predict number and timing of patient sessions
2. For each session, choose PXChange or SeqofSeq based on session timing
3. Generate scan sequences using the appropriate sequence model
4. Predict durations for each scan using the duration model
5. Output a complete schedule with all scans and timings

### Step 3: Review Generated Results

Generated schedules will be saved to:

```
UnifiedSchedulePipeline/outputs/
├── generated_schedules/        # Full daily schedules (CSV)
│   └── schedule_YYYYMMDD_HHMMSS.csv
├── event_timelines/           # Event-by-event timelines
│   └── events_YYYYMMDD_HHMMSS.csv
└── patient_sessions/          # Patient session summaries
    └── sessions_YYYYMMDD_HHMMSS.csv
```

### Step 4: Visualize Results (Optional)

If visualization scripts are available:

```bash
cd UnifiedSchedulePipeline/visualization
python visualize_schedule.py ../outputs/generated_schedules/schedule_*.csv
```

---

## Customizing Generation

### Generate Multiple Schedules

To generate multiple schedules for different days/machines:

```python
from generate_complete_schedule import generate_daily_schedule

# Wednesday, Machine 141049
schedule1 = generate_daily_schedule(day_of_week=3, machine_id=141049, seed=42)

# Friday, Machine 175832
schedule2 = generate_daily_schedule(day_of_week=5, machine_id=175832, seed=43)
```

### Adjust Generation Parameters

Edit `config.py` to modify:

```python
GENERATION_CONFIG = {
    'temperature_temporal': 0.8,     # Higher = more random session counts
    'temperature_sequences': 1.0,    # Higher = more diverse sequences
    'enable_validation': True,       # Enforce constraints
    'enable_adjustment': True,       # Auto-fix violations
    'max_retries': 3                # Retry on generation failures
}
```

---

## Troubleshooting

### Training Failed or Stopped

If training fails or is interrupted:

1. Check the error in `training_log_new.txt`
2. Resume training:
   ```bash
   cd UnifiedSchedulePipeline/training
   python train_all_models.py  # Will skip already-trained models
   ```

### Retrain Specific Model

To force retraining of a specific model:

```bash
python train_all_models.py --force --models temporal
python train_all_models.py --force --models pxchange_sequence
python train_all_models.py --force --models pxchange_duration
python train_all_models.py --force --models seqofseq_sequence
python train_all_models.py --force --models seqofseq_duration
```

### Generation Errors

If generation fails:

1. Verify all models are trained (check Step 1 above)
2. Check config paths in `config.py` under `MODEL_PATHS`
3. Review error messages for missing dependencies

---

## Data Overview

### Preprocessed Data Statistics

From `outputs/preprocessing_summary.txt`:

- **PXChange**: 17 files, 2,413 segments (2,218 real, 195 pseudo-patient)
- **SeqofSeq**: 2 files, 1,339 segments (930 real, 409 pseudo-patient)
- **Total**: 3,752 segments with 16.1% pseudo-patient sequences

Pseudo-patient sequences represent machine idle periods between patient sessions.

---

## Next Steps After Generation

1. **Validate Results**: Check generated schedules for realism
2. **Compare with Real Data**: If available, compare to actual schedules
3. **Tune Models**: Adjust hyperparameters if needed
4. **Deploy**: Integrate into scheduling system or dashboard

---

## Need Help?

- Check `training_log_new.txt` for detailed training logs
- Review `README.md` for project overview
- Check individual model configs in `config.py`
