# SeqofSeq Pipeline: MRI Scan Sequence and Duration Prediction

A deep learning pipeline for predicting MRI scan sequences and their durations using transformer-based models. This project uses sequence-of-sequences modeling to predict both the order of MRI scans and the time each scan will take.

## Overview

This pipeline processes MRI scan data from the 176625.csv dataset and trains two models:
1. **Sequence Generator**: Predicts the sequence of MRI scans (sequence types)
2. **Duration Predictor**: Predicts the duration of each scan in the sequence

## Features

- **Data Preprocessing**: Automatic preprocessing of raw MRI scan data
- **Transformer Models**: State-of-the-art transformer architecture for sequence prediction
- **Conditional Generation**: Generates sequences conditioned on patient/scan context
- **Docker Support**: Full containerization with Docker and docker-compose
- **Scalable**: Designed to handle large datasets with room for expansion

## Project Structure

```
SeqofSeq_Pipeline/
├── data/                      # Data directory
│   ├── 176625.csv            # Raw data file
│   └── preprocessed/         # Preprocessed data
├── preprocessing/            # Data preprocessing modules
│   ├── __init__.py
│   ├── preprocess_raw_data.py
│   └── data_loader.py
├── models/                   # Model architectures
│   ├── __init__.py
│   ├── layers.py
│   ├── conditional_sequence_generator.py
│   └── conditional_duration_predictor.py
├── training/                 # Training modules
│   ├── __init__.py
│   ├── train_sequence_model.py
│   └── train_duration_model.py
├── generation/               # Generation modules
│   ├── __init__.py
│   └── generate_pipeline.py
├── outputs/                  # Generated results
├── saved_models/            # Trained model checkpoints
├── visualizations/          # Plots and visualizations
├── config.py                # Configuration file
├── main_pipeline.py         # Main pipeline script
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
└── README.md               # This file
```

## Quick Start

### Option 1: Using Docker (Recommended)

1. **Build and start the container:**
```bash
docker-compose up -d seqofseq
```

2. **Enter the container:**
```bash
docker exec -it seqofseq_pipeline bash
```

3. **Run preprocessing:**
```bash
python main_pipeline.py preprocess
```

4. **Train models:**
```bash
python main_pipeline.py train
```

5. **Generate sequences:**
```bash
python main_pipeline.py generate
```

### Option 2: Local Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run preprocessing:**
```bash
python main_pipeline.py preprocess
```

3. **Train models:**
```bash
python main_pipeline.py train --batch-size 32 --epochs 100
```

4. **Generate sequences:**
```bash
python main_pipeline.py generate --num-samples 10
```

## Docker Commands

### Build the Docker image:
```bash
docker-compose build
```

### Start the main service:
```bash
docker-compose up -d seqofseq
```

### Start Jupyter Lab (for interactive development):
```bash
docker-compose up -d jupyter
```
Access Jupyter Lab at: http://localhost:8888

### Stop all services:
```bash
docker-compose down
```

### View logs:
```bash
docker-compose logs -f seqofseq
```

## Pipeline Commands

### Preprocessing
Converts raw CSV data into training-ready format:
```bash
python main_pipeline.py preprocess --data-file 176625.csv
```

### Training
Trains both the sequence generator and duration predictor:
```bash
python main_pipeline.py train \
    --batch-size 32 \
    --val-split 0.2 \
    --epochs 100 \
    --lr 0.0001
```

### Generation
Generates new sequences using trained models:
```bash
python main_pipeline.py generate \
    --num-samples 10 \
    --output-file generated_sequences.csv
```

## Data Format

The input data (176625.csv) contains MRI scan information with:
- **Coil configurations**: Boolean columns indicating which coils are connected
- **Scan metadata**: BodyPart, Sequence, Protocol, etc.
- **Timing information**: startTime, endTime, duration
- **Patient/System info**: PatientID, SN (System Number), Country, etc.

## Model Architecture

### Sequence Generator
- Transformer encoder-decoder architecture
- Predicts sequence of scan types conditioned on patient/scan context
- Uses attention mechanisms for sequential dependencies

### Duration Predictor
- Transformer encoder with cross-attention
- Predicts duration for each scan in the sequence
- Outputs mean (μ) and uncertainty (σ) using Gamma distribution

## Configuration

Edit `config.py` to customize:
- Model hyperparameters
- Training settings
- Data preprocessing options
- Paths and directories

## GPU Support

To enable GPU support with Docker:

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

2. Uncomment the GPU section in `docker-compose.yml`:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

3. Rebuild and restart:
```bash
docker-compose down
docker-compose build
docker-compose up -d seqofseq
```

## Extending the Pipeline

This pipeline is designed to be easily extended:

1. **Add more data**: Place additional CSV files in the `data/` directory
2. **Modify features**: Edit `CONDITIONING_FEATURES` in `config.py`
3. **Adjust model size**: Change `d_model`, `nhead`, etc. in `config.py`
4. **Custom preprocessing**: Modify `preprocessing/preprocess_raw_data.py`

## Troubleshooting

### Issue: "Data file not found"
**Solution**: Make sure 176625.csv is in the `data/` directory

### Issue: "No module named 'torch'"
**Solution**: Install requirements: `pip install -r requirements.txt`

### Issue: Docker container exits immediately
**Solution**: Check logs with `docker-compose logs seqofseq`

### Issue: Out of memory during training
**Solution**: Reduce batch size in config.py or training command

## References

Based on the PXChange_Refactored pipeline architecture with adaptations for MRI scan sequence prediction.

## License

MIT License

## Author

Created for MRI scan sequence and duration prediction research.
