# 📦 Export & Share Guide

This guide explains how to export and share the SeqofSeq Dashboard as a standalone package.

---

## ✅ **What's Been Done**

Your dashboard is now **STANDALONE** and ready to export!

### Changes Made:
1. ✓ Data copied to `SeqofSeq_Dashboard/data/`
2. ✓ Docker configuration updated to use local data
3. ✓ No dependency on SeqofSeq_Pipeline folder

---

## 📦 **To Export the Dashboard**

### **Option 1: Export as Folder (Recommended)**

Simply **zip or share** the entire `SeqofSeq_Dashboard` folder:

```bash
# Navigate to parent directory
cd ..

# Create a zip archive
tar -czf SeqofSeq_Dashboard.tar.gz SeqofSeq_Dashboard/

# Or on Windows (PowerShell):
Compress-Archive -Path SeqofSeq_Dashboard -DestinationPath SeqofSeq_Dashboard.zip
```

**Share the zip file** - it contains everything needed!

---

### **Option 2: Export as Docker Image**

Save the Docker image for distribution:

```bash
# Save the image
docker save seqofseq_dashboard-dashboard:latest | gzip > seqofseq_dashboard.tar.gz

# On another machine, load it:
docker load < seqofseq_dashboard.tar.gz
```

---

## 🚀 **Running on Another Machine**

### **Prerequisites:**
- Docker Desktop installed
- Docker Compose installed

### **Steps:**

1. **Extract the folder** (if zipped)

2. **Navigate to folder:**
   ```bash
   cd SeqofSeq_Dashboard
   ```

3. **Start the dashboard:**
   ```bash
   docker-compose up -d
   ```

4. **Open in browser:**
   ```
   http://localhost:8050
   ```

That's it! 🎉

---

## 📁 **What's Included**

The exported package contains:

```
SeqofSeq_Dashboard/
├── app/                    # Dashboard application
│   ├── app.py
│   ├── components/
│   ├── utils/
│   └── assets/
├── data/                   # Your sequence data
│   └── generated_sequences.csv
├── config/                 # Configuration
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker orchestration
├── requirements.txt        # Python dependencies
└── README.md              # Documentation
```

---

## 🔄 **Updating Data**

To update the data in the dashboard:

1. **Replace the CSV file:**
   ```bash
   cp new_data.csv SeqofSeq_Dashboard/data/generated_sequences.csv
   ```

2. **Restart the dashboard:**
   ```bash
   docker-compose restart
   ```

3. **Refresh browser** at http://localhost:8050

---

## 🌐 **Deploying to Cloud**

### **Deploy to Cloud Run (Google Cloud):**

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/seqofseq-dashboard
gcloud run deploy seqofseq-dashboard \
  --image gcr.io/PROJECT_ID/seqofseq-dashboard \
  --platform managed \
  --port 8050
```

### **Deploy to AWS ECS:**

```bash
# Build and push to ECR
docker tag seqofseq_dashboard-dashboard:latest \
  ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/seqofseq-dashboard
docker push ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/seqofseq-dashboard
```

### **Deploy to Azure Container Instances:**

```bash
az container create \
  --resource-group myResourceGroup \
  --name seqofseq-dashboard \
  --image seqofseq_dashboard-dashboard:latest \
  --dns-name-label seqofseq \
  --ports 8050
```

---

## 📊 **Data Privacy**

⚠️ **Important:** The exported package includes your actual sequence data.

If sharing publicly:
- Remove or anonymize sensitive data first
- Or share without the `data/` folder and provide instructions to add data separately

---

## 🛠️ **Troubleshooting**

### Dashboard won't start:
```bash
# Check logs
docker-compose logs -f dashboard

# Rebuild from scratch
docker-compose down
docker-compose up -d --build
```

### Port already in use:
Edit `docker-compose.yml`:
```yaml
ports:
  - "8051:8050"  # Change 8051 to any free port
```

### Data not loading:
Verify the file exists:
```bash
ls -lh data/generated_sequences.csv
```

---

## 📝 **System Requirements**

**Minimum:**
- Docker Desktop 20.10+
- 2 GB RAM
- 1 GB disk space

**Recommended:**
- Docker Desktop 24.0+
- 4 GB RAM
- 2 GB disk space

---

## 📧 **Support**

For issues or questions:
- Check `README.md` for detailed documentation
- Check `TROUBLESHOOTING.md` for common issues
- Review Docker logs: `docker-compose logs`

---

**Your dashboard is ready to share! 🚀**
