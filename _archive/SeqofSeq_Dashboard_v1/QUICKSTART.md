# 🚀 Quick Start Guide

Get the SeqofSeq Dashboard running in under 2 minutes!

## ⚡ TL;DR

```bash
cd SeqofSeq_Dashboard
docker-compose up -d --build
```

Then open: **http://localhost:8050**

---

## 📋 Step-by-Step

### 1️⃣ Prerequisites Check

Make sure you have:
- ✅ Docker installed
- ✅ SeqofSeq Pipeline data generated

### 2️⃣ Generate Data (if needed)

```bash
cd SeqofSeq_Pipeline
docker exec -it seqofseq_pipeline python main_pipeline.py generate
```

### 3️⃣ Start Dashboard

```bash
cd SeqofSeq_Dashboard
docker-compose up -d --build
```

### 4️⃣ Open Browser

Navigate to: **http://localhost:8050**

---

## 🎯 What You'll See

### Overview Tab 📊
- **Key metrics** at a glance
- **Sequence distributions**
- **Duration statistics**

### Sequence Analysis Tab 🔍
- **Interactive timelines**
- **Sample comparisons**
- **Transition flows**

### Duration Analysis Tab ⏱️
- **Duration distributions**
- **Statistical analysis**
- **Comparative visualizations**

---

## 🛠️ Common Commands

| Task | Command |
|------|---------|
| **Start Dashboard** | `docker-compose up -d` |
| **Stop Dashboard** | `docker-compose down` |
| **View Logs** | `docker-compose logs -f` |
| **Restart** | `docker-compose restart` |
| **Rebuild** | `docker-compose up -d --build` |

---

## ❓ Troubleshooting

### "Data Not Available" Error

1. Check if data exists:
   ```bash
   ls ../SeqofSeq_Pipeline/outputs/generated_sequences.csv
   ```

2. If missing, generate it:
   ```bash
   cd ../SeqofSeq_Pipeline
   docker exec -it seqofseq_pipeline python main_pipeline.py generate
   ```

3. Restart dashboard:
   ```bash
   docker-compose restart
   ```

### Port Already in Use

Edit `docker-compose.yml`:
```yaml
ports:
  - "8051:8050"  # Change 8051 to any available port
```

Then restart:
```bash
docker-compose down
docker-compose up -d
```

---

## 📖 Need More Help?

Check out the full [README.md](README.md) for:
- Detailed architecture
- Development guide
- Advanced configuration
- API reference

---

**Happy Analyzing! 📊✨**
