# Quick Start Guide

## Prerequisites

- Docker and Docker Compose installed (recommended)
- OR Python 3.10+ with pip (for local installation)
- WSL2 enabled (for Windows users using Docker)

## Option 1: Docker Setup (Recommended)

### Step 1: Build the Docker Container

```bash
cd SeqofSeq_Pipeline
docker-compose build
```

### Step 2: Start the Container

```bash
docker-compose up -d seqofseq
```

### Step 3: Enter the Container

```bash
docker exec -it seqofseq_pipeline bash
```

### Step 4: Run Preprocessing

```bash
python main_pipeline.py preprocess
```

**Expected Output:**
- Processes 4,091 MRI scans
- Creates 359 sequences from 362 patients
- Average sequence length: ~11.4 scans
- Vocabulary size: 33 unique sequence types
- 92 conditioning features (88 coils + 4 context features)

### Step 5: Train Models (Coming Soon)

```bash
python main_pipeline.py train --batch-size 32 --epochs 100
```

### Step 6: Generate Sequences (Coming Soon)

```bash
python main_pipeline.py generate --num-samples 10
```

## Option 2: Local Installation

### Step 1: Install Dependencies

```bash
cd SeqofSeq_Pipeline
pip install -r requirements.txt
```

### Step 2: Run Preprocessing

```bash
python main_pipeline.py preprocess
```

### Step 3: Train Models (Coming Soon)

```bash
python main_pipeline.py train
```

## Docker Commands Reference

### View Container Logs
```bash
docker-compose logs -f seqofseq
```

### Stop Container
```bash
docker-compose down
```

### Restart Container
```bash
docker-compose restart seqofseq
```

### Start Jupyter Lab (Optional)
```bash
docker-compose up -d jupyter
# Access at http://localhost:8888
```

## Jupyter Lab for Interactive Development

For interactive development and experimentation:

```bash
docker-compose up -d jupyter
```

Then open your browser to `http://localhost:8888` and create notebooks in the container.

## Data Structure

After preprocessing, you'll have:
- `data/preprocessed/preprocessed_data.csv` - Processed sequences
- `data/preprocessed/metadata.pkl` - Vocabularies and encoders

## Next Steps

1. **Preprocessing Complete** ✓ - You've successfully preprocessed the data
2. **Model Training** - Train the sequence and duration models
3. **Generation** - Generate new MRI scan sequences
4. **Evaluation** - Evaluate model performance

## Troubleshooting

### Issue: Docker container won't start
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Issue: Permission denied in WSL
```bash
sudo chmod -R 755 SeqofSeq_Pipeline
```

### Issue: Out of disk space
```bash
# Clean up Docker images
docker system prune -a
```

## File Locations in Container

- **Code**: `/app/`
- **Data**: `/app/data/`
- **Models**: `/app/saved_models/`
- **Outputs**: `/app/outputs/`

## Tips

1. Use `docker exec -it seqofseq_pipeline bash` to run commands in the container
2. All changes to files are persisted through volume mounts
3. You can edit code on your host machine and run it in the container
4. Use Jupyter Lab for experimentation and visualization

## Getting Help

- Check `README.md` for detailed documentation
- Review `config.py` to understand configuration options
- Examine the PXChange_Refactored project for reference implementations
