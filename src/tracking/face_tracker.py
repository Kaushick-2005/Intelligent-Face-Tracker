import os
import json
import csv
import cv2
import torch
from ultralytics import YOLO


# ============================================================
# INTELLIGENT FACE TRACKER
# ============================================================

# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = json.load(file)


MODEL_PATH = os.path.join(
    ROOT_DIR,
    config["models"]["yolo_face"]
)

VIDEO_DIR = os.path.join(
    ROOT_DIR,
    config["input"]["source"]
)

OUTPUT_DIR = os.path.join(
    ROOT_DIR,
    "output",
    "tracking"
)

LOG_DIR = os.path.join(
    ROOT_DIR,
    "logs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

CONFIDENCE = config["detection"]["confidence"]

# Keep this at 1 for tracking.
# Tracking needs consecutive frames.
FRAME_SKIP = config["detection"].get("frame_skip", 1)

TRACKER = config.get(
    "tracking",
    {}
).get(
    "tracker",
    "bytetrack.yaml"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------
# Information
# ------------------------------------------------------------

print("=" * 50)
print("INTELLIGENT FACE TRACKER")
print("FACE TRACKING MODULE")
print("=" * 50)

print("YOLO model:", MODEL_PATH)
print("Device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

print("Video directory:", VIDEO_DIR)
print("Confidence:", CONFIDENCE)
print("Frame skip:", FRAME_SKIP)
print("Tracker:", TRACKER)

print("=" * 50)


# ------------------------------------------------------------
# Load YOLO model
# ------------------------------------------------------------

if not os.path.exists(MODEL_PATH):
    print("ERROR: YOLO model not found.")
    print(MODEL_PATH)
    raise SystemExit(1)

model = YOLO(MODEL_PATH)


# ------------------------------------------------------------
# Find all videos
# ------------------------------------------------------------

video_files = [
    file
    for file in os.listdir(VIDEO_DIR)
    if file.lower().endswith(
        (".mp4", ".avi", ".mov", ".mkv")
    )
]

video_files.sort()

print("Videos found:", len(video_files))

if not video_files:
    print("ERROR: No videos found.")
    raise SystemExit(1)


# ------------------------------------------------------------
# CSV output
# ------------------------------------------------------------

csv_path = os.path.join(
    LOG_DIR,
    "face_tracking.csv"
)

csv_file = open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "video",
    "frame",
    "timestamp",
    "track_id",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2"
])


# ------------------------------------------------------------
# Overall statistics
# ------------------------------------------------------------

total_videos_processed = 0
total_tracking_records = 0
total_valid_tracks = 0


# ============================================================
# PROCESS EVERY VIDEO
# ============================================================

for video_index, video_name in enumerate(video_files, start=1):

    print()
    print("=" * 50)
    print(
        f"VIDEO {video_index}/{len(video_files)}: "
        f"{video_name}"
    )
    print("=" * 50)

    video_path = os.path.join(
        VIDEO_DIR,
        video_name
    )

    output_name = (
        os.path.splitext(video_name)[0]
        + "_tracked.mp4"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )


    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("ERROR: Could not open video.")
        continue


    # --------------------------------------------------------
    # Video information
    # --------------------------------------------------------

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )


    # --------------------------------------------------------
    # Video writer
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        print("ERROR: Could not create output video.")
        cap.release()
        continue


    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    frame_number = 0
    detection_count = 0
    valid_track_count = 0

    unique_ids = set()


    # ========================================================
    # FRAME PROCESSING
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1


        # ----------------------------------------------------
        # Tracking
        #
        # IMPORTANT:
        # Run YOLO tracking on consecutive frames.
        # Do NOT skip frames before tracker.update().
        # ----------------------------------------------------

        if FRAME_SKIP > 1:

            # Optional processing interval.
            #
            # However, we still send frames to the tracker
            # so that object identities remain stable.
            process_this_frame = (
                frame_number % FRAME_SKIP == 0
            )

        else:
            process_this_frame = True


        # ----------------------------------------------------
        # Run tracking
        # ----------------------------------------------------

        results = model.track(
            frame,
            conf=CONFIDENCE,
            device=DEVICE,
            persist=True,
            tracker=TRACKER,
            verbose=False
        )

        result = results[0]

        boxes = result.boxes


        # ----------------------------------------------------
        # Draw detections
        # ----------------------------------------------------

        annotated_frame = result.plot()


        # ----------------------------------------------------
        # Record detections
        # ----------------------------------------------------

        if (
            boxes is not None
            and len(boxes) > 0
            and process_this_frame
        ):

            for i in range(len(boxes)):

                # --------------------------------------------
                # Bounding box
                # --------------------------------------------

                xyxy = (
                    boxes.xyxy[i]
                    .cpu()
                    .numpy()
                )

                x1, y1, x2, y2 = [
                    int(value)
                    for value in xyxy
                ]


                # --------------------------------------------
                # Confidence
                # --------------------------------------------

                confidence = float(
                    boxes.conf[i].item()
                )


                # --------------------------------------------
                # Track ID
                # --------------------------------------------

                track_id = -1

                if boxes.id is not None:

                    track_id = int(
                        boxes.id[i].item()
                    )

                    unique_ids.add(
                        track_id
                    )

                    valid_track_count += 1


                # --------------------------------------------
                # Timestamp
                # --------------------------------------------

                timestamp = (
                    frame_number / fps
                )


                # --------------------------------------------
                # Save CSV record
                # --------------------------------------------

                csv_writer.writerow([
                    video_name,
                    frame_number,
                    round(timestamp, 3),
                    track_id,
                    round(confidence, 4),
                    x1,
                    y1,
                    x2,
                    y2
                ])

                detection_count += 1


        # ----------------------------------------------------
        # Write output frame
        # ----------------------------------------------------

        writer.write(
            annotated_frame
        )


        # ----------------------------------------------------
        # Progress display
        # ----------------------------------------------------

        if frame_number % 300 == 0:

            percentage = (
                frame_number
                / total_frames
                * 100
                if total_frames > 0
                else 0
            )

            print(
                f"Progress: "
                f"{frame_number}/{total_frames} "
                f"({percentage:.1f}%) | "
                f"Detections: {detection_count} | "
                f"Tracks: {len(unique_ids)}"
            )


    # --------------------------------------------------------
    # Release resources
    # --------------------------------------------------------

    cap.release()
    writer.release()


    # --------------------------------------------------------
    # Video summary
    # --------------------------------------------------------

    print()
    print("Frames:", total_frames)
    print("Detections:", detection_count)
    print("Valid track assignments:", valid_track_count)
    print("Unique track IDs:", len(unique_ids))
    print("Output:", output_path)


    total_videos_processed += 1
    total_tracking_records += detection_count
    total_valid_tracks += valid_track_count


# ============================================================
# CLOSE CSV
# ============================================================

csv_file.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 50)
print("FACE TRACKING COMPLETED")
print("=" * 50)

print(
    "Videos processed:",
    total_videos_processed
)

print(
    "Total tracking records:",
    total_tracking_records
)

print(
    "Total valid track assignments:",
    total_valid_tracks
)

print(
    "Tracking CSV:",
    csv_path
)

print(
    "Tracking outputs:",
    OUTPUT_DIR
)

print("=" * 50)