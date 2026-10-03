# Intelligent Face Tracker with Auto-Registration and Visitor Counting

An AI-driven real-time video surveillance and visitor analytics system built with **Python**, **YOLOv8-Face**, **InsightFace (ArcFace)**, **ByteTrack**, **SQLite**, and **MongoDB Atlas**.

The system automatically registers new faces upon first detection, re-identifies them across subsequent frames and videos to prevent duplicate visitor counts, tracks their trajectory continuously, and logs every entry and exit event along with timestamped facial crops in both local date-structured directories and persistent databases.

---

## Key Features

- **Real-Time Face Detection**: Powered by YOLOv8-Face with CUDA acceleration support.
- **Deep Face Recognition**: 512-dimensional facial embedding extraction via InsightFace (`buffalo_l` ArcFace backbone).
- **Multi-Object Tracking**: Smooth trajectory tracking across occlusions via ByteTrack.
- **Dynamic Auto-Registration**: Zero-shot enrollment of unknown faces with automatic unique visitor ID generation (`VISITOR_XXXX`).
- **Visitor Re-Identification (Re-ID)**: Cosine similarity matching against registered visitor embeddings ensures returning visitors never inflate the unique visitor count.
- **Dual Persistent Logging**:
  - **Local SQLite3** (`database/intelligent_face_tracker.sqlite3`) for offline reliability.
  - **MongoDB Atlas** for scalable cloud storage.
- **Entry & Exit Lifecycle Tracking**:
  - Automatically triggers an `ENTRY` event on first track appearance.
  - Triggers an `EXIT` event after track inactivity exceeds `exit_timeout_frames` or stream conclusion.
  - Saves cropped face images under structured date directories: `logs/entries/YYYY-MM-DD/` and `logs/exits/YYYY-MM-DD/`.
- **RTSP & Video Input Flexibility**: Supports local surveillance video folders, live RTSP IP cameras, and local webcams.
- **Configurable Architecture**: Easily adjust confidence, frame skip, exit timeout, similarity threshold, and data sources via `config.json`.

---

## System Architecture

```mermaid
flowchart TD
    subgraph InputSource ["Input Layer"]
        V["Video File (*.mp4)"] --> Cap["OpenCV VideoCapture"]
        R["Live RTSP Camera Stream"] --> Cap
        W["Webcam Device 0"] --> Cap
    end

    subgraph DetectionTracking ["Detection & Tracking Engine"]
        Cap --> Frame["Video Frame"]
        Frame --> YOLO["YOLOv8-Face Detector"]
        YOLO --> BBoxes["Face Bounding Boxes"]
        BBoxes --> Tracker["ByteTrack Object Tracker"]
        Tracker --> Tracks["Active Track IDs"]
    end

    subgraph RecognitionEngine ["InsightFace Recognition"]
        Tracks --> Crop["Padded Face Crop Extractor"]
        Crop --> ArcFace["InsightFace buffalo_l - 512D Embedding"]
        ArcFace --> Matcher{"Cosine Similarity >= 0.40?"}
        Matcher -- Yes --> KnownVisitor["Assign Known Visitor ID"]
        Matcher -- No --> RegisterNew["Auto-Register New Visitor ID"]
    end

    subgraph EventManager ["Event Lifecycle State Machine"]
        RegisterNew --> EntryEvent["Log ENTRY Event and Face Crop"]
        KnownVisitor --> EntryEvent
        Tracks --> InactivityCheck{"Frames Absent > 30 frames?"}
        InactivityCheck -- Yes --> ExitEvent["Log EXIT Event and Face Crop"]
    end

    subgraph StorageLayer ["Dual Storage & Archival"]
        EntryEvent & ExitEvent --> SQLite[("Local SQLite3 Database")]
        EntryEvent & ExitEvent --> MongoDB[("MongoDB Atlas Cloud")]
        EntryEvent & ExitEvent --> FileLogs["Structured Image Storage and events.log"]
    end
```

---

## Project Structure

```
Intelligent-Face-Tracker/
├── config.json                     # Active runtime configuration
├── config.example.json             # Example configuration template
├── main.py                         # Unified execution pipeline
├── dashboard.py                    # Read-only Streamlit analytics dashboard
├── requirements.txt                # Python dependencies
├── requirements-dashboard.txt      # Dashboard dependencies
├── database/
│   └── intelligent_face_tracker.sqlite3  # SQLite database
├── docs/
│   ├── ai_planning_and_architecture.md   # System architecture and dataflow
│   ├── compute_load_and_performance.md   # CPU/GPU compute estimations
│   ├── sample_output_and_verification.md # Sample DB schema and logs
│   └── demonstration_and_submission.md   # Demo instructions and submission details
├── logs/
│   ├── events.log                  # System and visitor event log
│   ├── entries/YYYY-MM-DD/         # Timestamped entry face crops
│   ├── exits/YYYY-MM-DD/           # Timestamped exit face crops
│   └── registrations/YYYY-MM-DD/   # First-seen registration face crops
├── models/
│   ├── yolo/yolov8n-face.pt        # YOLO face detector weights
│   └── insightface/                # InsightFace buffalo_l model files
├── output/
│   └── tracking/                   # Annotated video outputs
├── src/
│   ├── database/db_manager.py      # Dual SQLite & MongoDB database manager
│   └── recognition/face_recognizer.py # InsightFace embedding extraction
└── video/                          # Input surveillance video files
```

---

## Configuration (`config.json`)

```json
{
  "detection": {
    "confidence": 0.5,
    "frame_skip": 1,
    "imgsz": 1280
  },
  "input": {
    "source_type": "folder",
    "source": "video",
    "rtsp_url": "rtsp://username:password@camera_ip:554/stream1"
  },
  "models": {
    "yolo_face": "models/yolo/yolov8n-face.pt",
    "insightface_root": "models/insightface"
  },
  "tracking": {
    "tracker": "bytetrack.yaml",
    "exit_timeout_frames": 30
  },
  "recognition": {
    "similarity_threshold": 0.40,
    "retry_frames": 5,
    "crop_padding_ratio": 0.25
  },
  "output": {
    "directory": "output/tracking"
  },
  "logging": {
    "directory": "logs"
  },
  "database": {
    "sqlite_path": "database/intelligent_face_tracker.sqlite3",
    "mongodb_uri": "mongodb+srv://USERNAME:PASSWORD@CLUSTER.mongodb.net/?appName=Cluster01",
    "database_name": "intelligent_face_tracker"
  }
}
```

---

## Setup & Installation

### 1. Prerequisites

- Python 3.10 or 3.11
- NVIDIA GPU with CUDA 11.8+ / 12.x (Recommended for real-time inference) or multi-core CPU.

### 2. Environment Setup

```powershell
# Clone repository
git clone https://github.com/Kaushick-2005/Intelligent-Face-Tracker.git
cd Intelligent-Face-Tracker

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Pipeline

Run the main processing pipeline:

```powershell
python main.py
```

### 4. Running with Live RTSP Stream

To stream from an IP Camera / RTSP stream:

1. Edit `config.json`:
   ```json
   "input": {
     "source_type": "rtsp",
     "rtsp_url": "rtsp://YOUR_RTSP_STREAM_URL"
   }
   ```
2. Run `python main.py`.

### 5. Optional Verification Dashboard

The hackathon does not require a frontend, but this repository includes a
read-only Streamlit dashboard for demonstrating database-backed visitor
analytics and queryable entry/exit event records:

```powershell
pip install -r requirements-dashboard.txt
python -m streamlit run dashboard.py
```

The dashboard uses MongoDB as its live data source when the configured
connection is available and contains data. If MongoDB is unavailable or its
collections are empty while local data exists, it automatically uses the local
SQLite event store and shows a fallback warning. It supports date and
video filters, registered visitors, unique visitors in the selection,
ENTRY/EXIT totals, event balance, database rows, visitor ID search, and event-type
filters.

Use the same interpreter for installation and startup. This prevents a global
Streamlit installation from using a different Python environment without
`pymongo`:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-dashboard.txt
.\venv\Scripts\python.exe -m streamlit run dashboard.py
```

Event crop filenames use the stable visitor/event format
`VISITOR_0001_entry.jpg` and `VISITOR_0001_exit.jpg`.

The required database design remains the two durable collections/tables
`visitors` and `events`.

---

## Fresh Clone and Log Cleanup

The repository includes sample logs and face-crop files under `logs/` so the
sample output can be inspected after cloning. These files are not automatically
deleted or reset when the project is cloned.

If you want to run a clean local verification after cloning, stop any running
pipeline first and remove only the generated contents of `logs/`:

```powershell
Remove-Item .\logs\events.log -Force -ErrorAction SilentlyContinue
Remove-Item .\logs\entries\* -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item .\logs\exits\* -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item .\logs\registrations\* -Recurse -Force -ErrorAction SilentlyContinue
```

The pipeline recreates the required date folders and `events.log` when it
starts. `events.log` is appended to during processing. A same-day
`<VISITOR_ID>_entry.jpg`, `<VISITOR_ID>_exit.jpg`, or
`<VISITOR_ID>_registered.jpg` file can be replaced by a later run for the same
visitor and event type. Existing visitor records in MongoDB are persistent and
are not removed by this local log cleanup. Review the files before deleting
them if you need to preserve the checked-in sample output.

---

## Assumptions Made

1. **Camera Quality & Resolution**: Assumes camera stream resolution of 720p or 1080p at 15 to 30 FPS.
2. **Face Size & Pose**: Detectable faces must have a minimum resolution of approximately 20x20 pixels with frontal or semi-profile head orientations ($\le 45^\circ$ yaw/pitch).
3. **Lighting Conditions**: Standard indoor/outdoor lighting. Night vision or IR streams may slightly impact color embeddings.
4. **Disappearance Timeout**: A visitor is assumed to have exited after 30 consecutive frames without detection.
5. **Database Availability**: If MongoDB Atlas is unreachable, the system automatically falls back to local SQLite3 without dropping any events.

---

## Compute Load & Performance

| Mode           | Device            | Average Inference Speed                                     | VRAM / RAM Usage   |
| -------------- | ----------------- | ----------------------------------------------------------- | ------------------ |
| **GPU (CUDA)** | NVIDIA RTX GPU    | **30 - 40 FPS** (Real-Time)                                 | ~865 MB VRAM       |
| **CPU**        | Multi-Core x86_64 | **6 - 8 FPS** (Default) / **18 - 24 FPS** (`frame_skip: 3`) | ~950 MB System RAM |

Detailed hardware profiling can be found in [docs/compute_load_and_performance.md](docs/compute_load_and_performance.md).

---

## Video Demonstration

> [!NOTE]
> **Demonstration Video Link**: [Watch the Loom walkthrough](https://www.loom.com/share/f593fa6f79344ee4a489f765e8e4e974)

---

This project is a part of a hackathon run by https://katomaran.com
