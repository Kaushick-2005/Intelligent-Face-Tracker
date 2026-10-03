import os
import json
import cv2
import torch
from ultralytics import YOLO


# Project root
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# Load configuration
CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")

with open(CONFIG_PATH, "r") as file:
    config = json.load(file)

# Configuration
CONFIDENCE = config["detection"]["confidence"]
FRAME_SKIP = config["detection"]["frame_skip"]

VIDEO_DIR = os.path.join(
    ROOT_DIR,
    config["input"]["source"]
)

OUTPUT_DIR = os.path.join(
    ROOT_DIR,
    config["output"]["directory"]
)

MODEL_PATH = os.path.join(
    ROOT_DIR,
    config["models"]["yolo_face"]
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("========================================")
print("INTELLIGENT FACE TRACKER")
print("========================================")
print("YOLO model:", MODEL_PATH)
print("Device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

print("Video directory:", VIDEO_DIR)
print("Confidence:", CONFIDENCE)
print("Frame skip:", FRAME_SKIP)
print("========================================")


# Load YOLO model
model = YOLO(MODEL_PATH)


# Find all videos
video_files = [
    file
    for file in os.listdir(VIDEO_DIR)
    if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
]

video_files.sort()

print("Videos found:", len(video_files))

if not video_files:
    print("No video files found.")
    exit()


# Process every video
for video_name in video_files:

    video_path = os.path.join(VIDEO_DIR, video_name)

    output_name = os.path.splitext(video_name)[0] + "_detected.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    print()
    print("----------------------------------------")
    print("Processing:", video_name)
    print("----------------------------------------")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("ERROR: Could not open video.")
        continue

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    frame_number = 0
    detected_faces = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # Run detection only on selected frames
        if frame_number % FRAME_SKIP == 0:

            results = model.predict(
                frame,
                conf=CONFIDENCE,
                device=DEVICE,
                verbose=False
            )

            annotated_frame = results[0].plot()

            boxes = results[0].boxes

            if boxes is not None:
                detected_faces += len(boxes)

        else:
            annotated_frame = frame

        writer.write(annotated_frame)

    cap.release()
    writer.release()

    print("Frames:", total_frames)
    print("Face detections:", detected_faces)
    print("Output:", output_path)

print()
print("========================================")
print("ALL VIDEOS PROCESSED")
print("========================================")