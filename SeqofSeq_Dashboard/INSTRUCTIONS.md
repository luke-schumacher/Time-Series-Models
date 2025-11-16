# 📋 Complete Instructions for Running SeqofSeq Dashboard

## 🎯 Overview

This document provides complete, step-by-step instructions for running the SeqofSeq Dashboard with Docker.

---

## 📦 What You Need

1. **Docker Desktop** installed on your system
2. **Data from SeqofSeq Pipeline** (generated sequences CSV file)
3. **Terminal/Command Prompt** access

---

## 🚀 Running with Docker (3 Steps)

### Step 1: Open Terminal

**Windows:**
- Press `Win + R`
- Type `cmd` or `powershell`
- Click OK

**Mac/Linux:**
- Press `Cmd + Space` (Mac) or `Ctrl + Alt + T` (Linux)
- Type `terminal`
- Press Enter

### Step 2: Navigate to Dashboard Directory

```bash
cd SeqofSeq_Dashboard
```

Or use the full path:

**Windows:**
```powershell
cd C:\Users\lukis\Documents\GitHub\Time-Series-Models\SeqofSeq_Dashboard
```

**Mac/Linux:**
```bash
cd /path/to/Time-Series-Models/SeqofSeq_Dashboard
```

### Step 3: Start the Dashboard

```bash
docker-compose up -d --build
```

**What this does:**
- `docker-compose` - Uses Docker Compose tool
- `up` - Starts the services
- `-d` - Runs in background (detached mode)
- `--build` - Builds the image first

**Expected output:**
```
[+] Building ...
[+] Running 2/2
 ✔ Network seqofseq-network     Created
 ✔ Container seqofseq_dashboard  Started
```

### Step 4: Open Dashboard in Browser

Open your web browser and go to:

```
http://localhost:8050
```

---

## 🎨 Using the Dashboard

### Navigation

The dashboard has 3 main tabs:

1. **📊 Overview Tab**
   - View key metrics (samples, sequences, patients)
   - See sequence type distributions
   - Analyze duration statistics

2. **🔍 Sequence Analysis Tab**
   - Select specific samples to view
   - See timeline visualizations
   - Compare multiple samples
   - View sequence transitions

3. **⏱️ Duration Analysis Tab**
   - Filter by sequence type
   - View duration distributions
   - See statistical breakdowns
   - Compare durations across types

### Interactive Features

- **Hover** over charts for details
- **Click and drag** to zoom
- **Double-click** to reset zoom
- **Click legend items** to show/hide
- **Use dropdowns** to filter data
- **Click refresh button** to reload data

---

## 🛠️ Common Operations

### Viewing Logs

```bash
docker-compose logs -f dashboard
```

Press `Ctrl + C` to exit log view.

### Stopping the Dashboard

```bash
docker-compose down
```

### Restarting the Dashboard

```bash
docker-compose restart dashboard
```

### Rebuilding After Changes

```bash
docker-compose up -d --build
```

### Checking if Dashboard is Running

```bash
docker-compose ps
```

Look for `seqofseq_dashboard` with status "Up".

---

## 🔄 Updating Data

### Method 1: Generate New Data in Pipeline

1. **Navigate to pipeline directory:**
   ```bash
   cd ../SeqofSeq_Pipeline
   ```

2. **Generate new sequences:**
   ```bash
   docker exec -it seqofseq_pipeline python main_pipeline.py generate
   ```

3. **Data automatically updates** (mounted volume)

4. **Click "Refresh Data" button** in dashboard header

### Method 2: Replace Data File

1. **Place new CSV file** in:
   ```
   ../SeqofSeq_Pipeline/outputs/generated_sequences.csv
   ```

2. **Click "Refresh Data" button** in dashboard

---

## ❓ Troubleshooting

### Problem: "Data Not Available" Error

**Symptoms:**
- Dashboard shows warning message
- No visualizations displayed

**Solutions:**

1. **Check if data file exists:**
   ```bash
   ls ../SeqofSeq_Pipeline/outputs/generated_sequences.csv
   ```

2. **Generate data if missing:**
   ```bash
   cd ../SeqofSeq_Pipeline
   docker exec -it seqofseq_pipeline python main_pipeline.py generate
   ```

3. **Restart dashboard:**
   ```bash
   cd ../SeqofSeq_Dashboard
   docker-compose restart dashboard
   ```

---

### Problem: Port Already in Use

**Symptoms:**
- Error: "Port 8050 is already allocated"
- Container fails to start

**Solution:**

1. **Edit docker-compose.yml:**
   ```yaml
   ports:
     - "8051:8050"  # Change to any available port
   ```

2. **Restart:**
   ```bash
   docker-compose down
   docker-compose up -d
   ```

3. **Access dashboard at:**
   ```
   http://localhost:8051
   ```

---

### Problem: Container Not Starting

**Symptoms:**
- Container exits immediately
- Dashboard not accessible

**Solutions:**

1. **Check logs:**
   ```bash
   docker-compose logs dashboard
   ```

2. **Check Docker is running:**
   - Windows: Look for Docker icon in system tray
   - Mac: Look for Docker icon in menu bar

3. **Verify data path:**
   ```bash
   docker exec seqofseq_dashboard ls /app/data
   ```

4. **Rebuild container:**
   ```bash
   docker-compose down
   docker-compose up -d --build
   ```

---

### Problem: Charts Not Loading

**Symptoms:**
- Empty charts or spinning loaders
- JavaScript errors

**Solutions:**

1. **Clear browser cache:**
   - Chrome: `Ctrl + Shift + Delete` (Windows) or `Cmd + Shift + Delete` (Mac)
   - Select "Cached images and files"
   - Click "Clear data"

2. **Try different browser:**
   - Recommended: Google Chrome, Firefox, Edge

3. **Check browser console:**
   - Press `F12`
   - Look for errors in Console tab

---

## 🎓 Advanced Usage

### Custom Configuration

**Edit environment variables in docker-compose.yml:**

```yaml
environment:
  - DATA_PATH=/app/data/generated_sequences.csv
  - PORT=8050
  - DEBUG=false
```

### Using Different Data Source

1. **Edit docker-compose.yml:**
   ```yaml
   volumes:
     - /path/to/your/data:/app/data:ro
   ```

2. **Update DATA_PATH:**
   ```yaml
   environment:
     - DATA_PATH=/app/data/your_file.csv
   ```

3. **Restart:**
   ```bash
   docker-compose up -d --build
   ```

---

## 📞 Getting Help

### Documentation Files

- **README.md** - Complete documentation
- **QUICKSTART.md** - Fast setup guide
- **INSTRUCTIONS.md** - This file
- **Makefile** - Command shortcuts

### Command Cheat Sheet

| What you want | Command |
|---------------|---------|
| Start dashboard | `docker-compose up -d` |
| Stop dashboard | `docker-compose down` |
| View logs | `docker-compose logs -f` |
| Restart | `docker-compose restart` |
| Rebuild | `docker-compose up -d --build` |
| Check status | `docker-compose ps` |

### Using Makefile (Shortcuts)

If you have `make` installed:

```bash
make help      # Show all commands
make up        # Start dashboard
make down      # Stop dashboard
make logs      # View logs
make restart   # Restart dashboard
make clean     # Clean everything
```

---

## ✅ Verification Checklist

Before reporting issues, verify:

- [ ] Docker Desktop is running
- [ ] You're in the SeqofSeq_Dashboard directory
- [ ] Data file exists at ../SeqofSeq_Pipeline/outputs/generated_sequences.csv
- [ ] Port 8050 is not used by another application
- [ ] Container is running: `docker-compose ps`
- [ ] No errors in logs: `docker-compose logs dashboard`

---

## 🎉 Success!

If you see the dashboard with data and visualizations, you're all set!

**Key URLs:**
- **Dashboard**: http://localhost:8050
- **Container Logs**: `docker-compose logs -f dashboard`

**Enjoy analyzing your MRI sequence predictions!** 📊✨

---

*For more detailed information, see README.md*
