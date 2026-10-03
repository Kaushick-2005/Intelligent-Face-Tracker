# Hackathon Demonstration & Submission Details

This document provides demonstration guidelines, verification steps, and project submission references for the Katomaran Technologies Hackathon.

---

## 1. Project Demonstration Overview
The **Intelligent Face Tracker with Auto-Registration and Visitor Counting** solution implements an end-to-end edge-AI vision system that:
1. Detects faces dynamically in video feeds using a YOLOv8-face neural model.
2. Tracks faces continuously using the ByteTrack object association tracker.
3. Automatically computes 512-D ArcFace deep facial representations using InsightFace.
4. Performs dynamic visitor registration upon novel encounters.
5. Re-identifies returning visitors across subsequent frames and videos to prevent duplicate visitor inflation.
6. Dispatches real-time ENTRY and EXIT events with date-partitioned timestamped face crops.
7. Logs all transactions consistently across SQLite (offline resilience) and MongoDB Atlas (cloud scalability).
8. Supports both pre-recorded video batches and live camera streams (RTSP / webcam).

---

## 2. Video Demonstration Link

**Loom video link:** [Watch the Loom demonstration](https://www.loom.com/share/f593fa6f79344ee4a489f765e8e4e974)

Copy the video URL:

```text
https://www.loom.com/share/f593fa6f79344ee4a489f765e8e4e974
```

---

## 3. Quick Run Instructions for Evaluators

### Step 1: Activate Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### Step 2: Configure Environment
Copy the example configuration if needed:
```powershell
cp config.example.json config.json
```

### Step 3: Run Full Pipeline
```powershell
python main.py
```

### Step 4: Verify Output Logs & Database
Check `logs/events.log` and verify database records:
```powershell
python -c "from src.database.db_manager import DatabaseManager; db = DatabaseManager(); print('Registered Unique Visitors:', db.get_visitor_count())"
```

---

## 4. Testing Live RTSP Stream
To run with a live RTSP camera feed during the live interview:
1. Open `config.json`.
2. Change `"source_type"` to `"rtsp"`.
3. Provide the RTSP stream URL under `"rtsp_url"`, e.g.:
   ```json
   "input": {
     "source_type": "rtsp",
     "rtsp_url": "rtsp://username:password@camera_ip:554/live"
   }
   ```
4. Run `python main.py`.

---

## 5. Hackathon Disclaimer
This project is a part of a hackathon run by https://katomaran.com
