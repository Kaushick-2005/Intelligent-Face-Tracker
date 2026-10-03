# AI Planning Document & Application Architecture

## 1. Problem Overview
The **Intelligent Face Tracker with Auto-Registration and Visitor Counting** system is built to solve continuous real-time video surveillance challenges:
- High accuracy real-time face detection under varying illumination, head poses, and motion blur.
- Consistent cross-frame object association (tracking across frames).
- Face feature extraction using state-of-the-art Deep Face Recognition representations (512-D ArcFace / InsightFace embeddings).
- Dynamic auto-registration of unknown visitors without manual enrollment.
- Robust cross-session and cross-camera Re-Identification (Re-ID) to ensure unique visitor tallies never increment upon return visits.
- Event logging for Entry and Exit with date-partitioned timestamped face crops.
- Persistent structured data logging across dual storage engines: **SQLite** (local guaranteed offline persistence) and **MongoDB Atlas** (cloud synchronization).
- Seamless switching between pre-recorded surveillance videos and live RTSP camera streams.

---

## 2. System Architecture

```mermaid
flowchart TD
    subgraph Input ["Video & Stream Ingestion"]
        V["Video File (*.mp4, *.avi)"] --> Cap["OpenCV VideoCapture / RTSP Client"]
        R["Live RTSP Camera Stream"] --> Cap
        W["Webcam Device 0"] --> Cap
    end

    subgraph DetectionTracking ["Face Detection & Tracking Engine"]
        Cap --> Frame["Video Frame (BGR)"]
        Frame --> YOLO["YOLOv8-Face Model (ONNX/PyTorch CUDA)"]
        YOLO --> BBox["Face Bounding Boxes and Confidence"]
        BBox --> ByteTrack["ByteTrack Multi-Object Tracker"]
        ByteTrack --> Tracks["Persistent Track IDs"]
    end

    subgraph FaceRecognition ["InsightFace ArcFace Recognition"]
        Tracks --> Crop["Padded Face Crop Extractor"]
        Crop --> BuffaloL["InsightFace buffalo_l (512-D Embedding)"]
        BuffaloL --> SimCheck{"Cosine Similarity >= 0.40?"}
        SimCheck -- "Yes: Known Visitor" --> ReID["Assign Known Visitor ID"]
        SimCheck -- "No: Unknown" --> Register["Auto-Register New Visitor ID"]
    end

    subgraph EventLifecycle ["Entry / Exit Lifecycle State Machine"]
        Register --> LogEntry["Generate ENTRY Event and Face Crop"]
        ReID --> LogEntry
        Tracks --> ActiveMap["Active Tracks Hash Map"]
        ActiveMap --> Inactivity{"Frames Since Last Seen > Timeout (30)?"}
        Inactivity -- Yes --> LogExit["Generate EXIT Event and Face Crop"]
    end

    subgraph Storage ["Dual Storage & Logging Engine"]
        LogEntry --> SQLite[("Local SQLite Database")]
        LogEntry --> Mongo[("MongoDB Atlas Cloud DB")]
        LogEntry --> FS["Date-Structured Filesystem: logs/entries/YYYY-MM-DD/"]
        LogExit --> SQLite
        LogExit --> Mongo
        LogExit --> FS2["Date-Structured Filesystem: logs/exits/YYYY-MM-DD/"]
        LogEntry & LogExit --> SysLog["System Log File: logs/events.log"]
    end

    subgraph OutputVisual ["Output Visualization & Telemetry"]
        ActiveMap --> Overlay["Draw Bounding Boxes, ID and HUD Telemetry"]
        Overlay --> VideoWriter["Annotated Output Video: output/tracking/"]
    end
```

---

## 3. Modular Pipeline Design

1. **`src/detection/face_detector.py`**:
   - High-throughput batch inference pipeline for face localization.
   - Evaluates frame-skip strategies and detection confidence thresholds.

2. **`src/tracking/face_tracker.py`**:
   - Integrates ByteTrack with YOLO bounding box representations.
   - Maintains continuous motion trajectories and trajectory CSV reports.

3. **`src/recognition/face_recognizer.py`**:
   - Implements InsightFace `buffalo_l` (ResNet50 / ArcFace backbone).
   - Generates normalized 512-dimensional facial feature vectors.
   - Computes Cosine Similarity metric:
     $$\text{Cosine Similarity} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

4. **`src/database/db_manager.py`**:
   - Dual-engine transactional logging for SQLite and MongoDB Atlas.
   - Automatic table schema creation (`visitors`, `events`).
   - In-memory visitor pre-loading on startup for immediate cross-run re-identification.

5. **`main.py`**:
   - Unified real-time orchestration engine.
   - Handles video folder ingestion, live RTSP streams, and webcams.
   - Coordinates dynamic auto-registration, entry/exit logging, crop archiving, and video rendering.

---

## 4. State Machine: Visitor Lifecycle

- **STATE 1: NEW DETECTION**
  - Track ID assigned by ByteTrack.
  - Bounding box padded by 25% to capture full facial context.
  - 512-D embedding extracted via InsightFace.

- **STATE 2: RE-IDENTIFICATION CHECK**
  - Comparison against loaded visitor database.
  - If $\max(\text{similarity}) \ge 0.40$: Identified as existing visitor.
  - If $\max(\text{similarity}) < 0.40$: New visitor ID allocated (`VISITOR_XXXX`).

- **STATE 3: ENTRY EVENT DISPATCH**
  - Face crop saved to `logs/entries/YYYY-MM-DD/<visitor_id>_entry.jpg`.
  - Transaction written to `events` table (Type: `ENTRY`).
  - Recorded in `logs/events.log`.

- **STATE 4: CONTINUOUS TRACKING**
  - Position updated per frame; real-time telemetry HUD rendered.

- **STATE 5: EXIT EVENT DISPATCH**
  - Triggered when track is missing for $\ge 30$ consecutive frames or video reaches EOF.
  - Face crop saved to `logs/exits/YYYY-MM-DD/<visitor_id>_exit.jpg`.
  - Transaction written to `events` table (Type: `EXIT`).
  - Memory cleanup of active track metadata.
