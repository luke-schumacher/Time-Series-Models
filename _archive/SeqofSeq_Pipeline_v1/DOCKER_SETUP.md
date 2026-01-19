# Docker Setup Guide

Complete guide for setting up and using the SeqofSeq Pipeline with Docker.

## System Requirements

- **Windows**: Windows 10/11 with WSL2
- **Docker Desktop**: Latest version
- **RAM**: 8GB minimum, 16GB recommended
- **Disk Space**: 10GB free space

## Installation

### 1. Install Docker Desktop

Download and install from: https://www.docker.com/products/docker-desktop/

### 2. Enable WSL2 (Windows Only)

```powershell
# Run in PowerShell as Administrator
wsl --install
wsl --set-default-version 2
```

Restart your computer after installation.

### 3. Verify Docker Installation

```bash
docker --version
docker-compose --version
```

## Building the Container

### Basic Build

```bash
cd SeqofSeq_Pipeline
docker-compose build
```

### Build with No Cache (if you encounter issues)

```bash
docker-compose build --no-cache
```

## Running the Container

### Start in Background

```bash
docker-compose up -d seqofseq
```

### Start with Logs

```bash
docker-compose up seqofseq
```

### Start Multiple Services

```bash
# Start main pipeline and Jupyter
docker-compose up -d seqofseq jupyter
```

## Working Inside the Container

### Enter the Container Shell

```bash
docker exec -it seqofseq_pipeline bash
```

### Run Commands in Container

```bash
# One-off command
docker exec -it seqofseq_pipeline python main_pipeline.py preprocess

# Interactive session
docker exec -it seqofseq_pipeline bash
>>> python main_pipeline.py preprocess
>>> python main_pipeline.py train
```

## GPU Support (Optional)

### Prerequisites

1. Install NVIDIA GPU drivers on host
2. Install NVIDIA Container Toolkit:

```bash
# Ubuntu/WSL2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Enable GPU in docker-compose.yml

Uncomment the GPU section:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Rebuild and Test

```bash
docker-compose down
docker-compose build
docker-compose up -d

# Test GPU access
docker exec -it seqofseq_pipeline python -c "import torch; print(torch.cuda.is_available())"
```

## Volume Mounts

The following directories are mounted between host and container:

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./data` | `/app/data` | Dataset storage |
| `./outputs` | `/app/outputs` | Generated results |
| `./saved_models` | `/app/saved_models` | Model checkpoints |
| `./visualizations` | `/app/visualizations` | Plots and charts |

Changes in these directories are immediately reflected on both host and container.

## Jupyter Lab

### Start Jupyter

```bash
docker-compose up -d jupyter
```

### Access Jupyter

1. Check logs for the token:
```bash
docker-compose logs jupyter
```

2. Look for a line like:
```
http://127.0.0.1:8888/?token=abc123...
```

3. Open in browser: `http://localhost:8888`

### Stop Jupyter

```bash
docker-compose stop jupyter
```

## Container Management

### View Running Containers

```bash
docker-compose ps
```

### View Logs

```bash
# All logs
docker-compose logs

# Follow logs
docker-compose logs -f

# Specific service
docker-compose logs -f seqofseq
```

### Stop Containers

```bash
# Stop all
docker-compose down

# Stop specific service
docker-compose stop seqofseq
```

### Restart Containers

```bash
# Restart all
docker-compose restart

# Restart specific
docker-compose restart seqofseq
```

### Remove Containers and Images

```bash
# Remove containers
docker-compose down

# Remove containers and volumes
docker-compose down -v

# Remove everything including images
docker-compose down --rmi all
```

## Troubleshooting

### Container Exits Immediately

Check logs:
```bash
docker-compose logs seqofseq
```

### Permission Errors

```bash
# Fix permissions
sudo chown -R $USER:$USER .

# In WSL2
sudo chmod -R 755 .
```

### Port Already in Use (Jupyter)

Change port in `docker-compose.yml`:
```yaml
ports:
  - "8889:8888"  # Change 8888 to 8889
```

### Out of Disk Space

```bash
# Clean up Docker
docker system prune -a

# Remove unused volumes
docker volume prune
```

### Slow Performance on Windows

1. Move project to WSL2 filesystem:
```bash
# In WSL2
cd ~
git clone <repo>
```

2. Use WSL2-based Docker backend in Docker Desktop settings

### Network Issues

```bash
# Restart Docker
sudo systemctl restart docker

# Or in Docker Desktop: Settings > Reset > Restart Docker
```

## Best Practices

1. **Always use volume mounts** for data and models
2. **Don't store large files in the image** - use volumes instead
3. **Use .dockerignore** to exclude unnecessary files
4. **Regularly clean up** unused containers and images
5. **Monitor resource usage** in Docker Desktop

## Advanced Usage

### Custom Build Arguments

```bash
docker-compose build --build-arg PYTHON_VERSION=3.11
```

### Override Container Command

```bash
docker-compose run seqofseq python custom_script.py
```

### Copy Files To/From Container

```bash
# Copy to container
docker cp local_file.txt seqofseq_pipeline:/app/

# Copy from container
docker cp seqofseq_pipeline:/app/outputs/results.csv ./
```

### Execute Multiple Commands

```bash
docker exec -it seqofseq_pipeline bash -c "
python main_pipeline.py preprocess && \
python main_pipeline.py train --epochs 50
"
```

## Resource Limits

Edit `docker-compose.yml` to set resource limits:

```yaml
services:
  seqofseq:
    # ... other config ...
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
```

## Production Deployment

For production deployment:

1. Use specific image tags (not `latest`)
2. Set resource limits
3. Configure health checks
4. Use secrets for sensitive data
5. Set up logging and monitoring
6. Use orchestration (Kubernetes, Docker Swarm)

## Getting Help

- Docker Documentation: https://docs.docker.com/
- Docker Compose Reference: https://docs.docker.com/compose/
- WSL2 Setup: https://docs.microsoft.com/en-us/windows/wsl/
