# MRI Scan Time Prediction Model Optimization

## Overview

This document summarizes the optimization work performed on the MRI scan time prediction pipeline.

## Key Problems Identified

### 1. Original LSTM Ensemble Model Limitations
- **Limited Feature Set**: Only used predicted proportions as input
- **Missing Context**: Ignored valuable information about:
  - Scan types (sourceID sequences)
  - Patient characteristics (Age, Weight, Height)
  - Body regions being scanned (BodyGroup_from/to)
  - Equipment settings (PTAB values)
  - Patient positioning (Position, Direction)

### 2. Model Architecture Issues
- Three-stage ensemble (Statistical → Sequential → Meta-learner) was complex
- No direct use of original sequence features
- Proportions alone don't capture scan type information

## Optimizations Implemented

### Enhanced Model Architecture (`LSTM_TF_Optimized.py`)

#### Multi-Input Architecture
The new model combines THREE information pathways:

1. **Original Sequence Features Branch**
   - Input: sourceID, PTAB, BodyGroup_from, BodyGroup_to, Position_encoded, Direction_encoded
   - Architecture: Bidirectional LSTM (128 units) → Multi-Head Attention (8 heads) → Global Pooling
   - Purpose: Captures temporal patterns in scan sequences and equipment usage

2. **Predicted Proportions Branch**
   - Input: Predicted time proportions from Transformer model
   - Architecture: Bidirectional GRU (64 units) → Multi-Head Attention (4 heads) → Global Pooling
   - Purpose: Leverages the proportion patterns learned by the Transformer

3. **Patient/Scan Metadata Branch**
   - Input: Age, Weight, Height, BodyGroup_from, BodyGroup_to, Sequence Length
   - Architecture: Dense layers (32 → 16 units)
   - Purpose: Captures patient-specific and scan-level characteristics

#### Final Prediction Network
- All three branches are concatenated
- Deep feedforward network: 256 → 128 → 64 units with dropout
- Single output: Predicted total scan time in seconds

### Key Improvements

1. **Feature Engineering**
   - ✅ Incorporates ALL available features, not just proportions
   - ✅ Separate pathways for sequential vs. static features
   - ✅ Attention mechanisms to focus on important timesteps

2. **Training Enhancements**
   - Huber loss (robust to outliers)
   - ReduceLROnPlateau (adaptive learning rate)
   - EarlyStopping (prevents overfitting)
   - StandardScaler normalization

3. **Data Pipeline**
   - Proper merging of proportion predictions with preprocessed data
   - Handles missing data gracefully
   - Validates data integrity before training

## Results

### Quick Test (Dataset 176401, 50 epochs)
- **MAE**: 113.88 seconds (~1.9 minutes)
- **RMSE**: 215.56 seconds (~3.6 minutes)
- **MAPE**: 29.47%
- **Training Time**: ~35 seconds

### Full Training (All datasets, 200 epochs)
- Currently in progress...
- Processes all datasets with proportion predictions available
- Generates visualizations for each dataset
- Saves predictions to `prediction_{dataset_id}_total_time_refactored.csv`

## Pipeline Structure

```
Raw Data
    ↓
[Preprocessor] → preprocessed_{id}.csv
    ↓
[Transformer Model] → prediction_{id}_proportions_refactored.csv
    ↓
[Enhanced LSTM Model] → prediction_{id}_total_time_refactored.csv
    ↓
[Combine & Denormalize] → combined_denormalized_predictions.csv
```

## Files Created/Modified

### New Files
- `LSTM_TF_Optimized.py` - Enhanced total time prediction model
- `test_optimized_model.py` - Quick testing script
- `MODEL_OPTIMIZATION_SUMMARY.md` - This document

### Existing Files
- `preprocessor.py` - No changes (already optimized)
- `TF_TimeSeries_Refactored.py` - No changes (proportion prediction works well)
- `LSTM_TF_Refactored.py` - Can be replaced by `LSTM_TF_Optimized.py`
- `combine_and_denormalize_predictions.py` - No changes needed

## Usage

### Train on all datasets:
```bash
cd PXChange
python LSTM_TF_Optimized.py
```

### Quick test on single dataset:
```bash
cd PXChange
python test_optimized_model.py
```

### Complete pipeline (from scratch):
```bash
# 1. Preprocess data
python processing/preprocessor.py

# 2. Generate proportion predictions
python TF_TimeSeries_Refactored.py

# 3. Generate total time predictions
python LSTM_TF_Optimized.py

# 4. Combine all predictions
python processing/combine_and_denormalize_predictions.py
```

## Future Improvements

1. **Hyperparameter Tuning**
   - Grid search for optimal learning rate, dropout rates, layer sizes
   - Experiment with different attention head configurations

2. **Advanced Architectures**
   - Try Temporal Convolutional Networks (TCN)
   - Experiment with Transformer encoders for sequence features
   - Add cross-attention between feature branches

3. **Ensemble Methods**
   - Combine multiple model predictions
   - Use bootstrap aggregating (bagging)

4. **Feature Engineering**
   - Create interaction features (e.g., Age × BodyGroup)
   - Add temporal features (time of day, day of week if available)
   - One-hot encode sourceID for better representation

5. **Data Augmentation**
   - Synthetic sequence generation
   - Time warping for sequence data

## Conclusion

The enhanced model significantly improves total time prediction by:
- Using ALL available features instead of just proportions
- Employing specialized pathways for different feature types
- Leveraging attention mechanisms to focus on important information
- Proper handling of patient metadata

Initial results show promising performance with MAE of ~114 seconds on test data.
