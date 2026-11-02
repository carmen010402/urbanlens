"""
detect_combined_visual.py

Same as detect_combined.py, but draws boxes on every frame and saves the
result as a new video file you can watch afterward -- so you can actually
SEE what each model is detecting, not just read numbers.

Color coding:
  GREEN boxes = your original COCO model (person, chair, backpack, etc.)
  RED boxes   = your NEW custom model (hole, dump, weather_collision)

HOW TO RUN:
  python3 detect_combined_visual.py shot1.mp4

This creates a new file: shot1_annotated.mp4 (same name + "_annotated"),
in the same folder. Open that file directly in Windows (double-click it
in File Explorer, or drag it into VLC/Windows Media Player) to watch it.
"""

import sys
import os
import cv2
from ultralytics import YOLO

coco_model = YOLO('yolov8s.pt')
PERSON_CLASS = 0
RELEVANT_COCO_CLASSES = {56, 24, 28, 26, 13, 58, 41}

CUSTOM_MODEL_PATH = "runs/detect/hole_dump_weather_model-5/weights/best.pt"
custom_model = YOLO(CUSTOM_MODEL_PATH)
CUSTOM_CLASS_NAMES = ["hole", "dump", "weather_collision"]

MIN_HITS_TO_COUNT = 5

# FIX: reject any custom-model detection whose box covers an unreasonably
# large fraction of the frame -- see detect_combined.py for the full
# explanation. Real litter/potholes/branches are never that large.
MAX_BOX_AREA_FRACTION = 0.25


def box_area_fraction(box, frame_width, frame_height):
    x1, y1, x2, y2 = box
    box_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_width * frame_height
    return box_area / frame_area

GREEN = (0, 255, 0)
RED = (0, 0, 255)


def draw_box(frame, box, label, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 detect_combined_visual.py <video_file>")
        sys.exit(1)

    source = sys.argv[1]
    cap = cv2.VideoCapture(source)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base, ext = os.path.splitext(source)
    output_path = f"{base}_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    track_hit_counts = {}
    frame_num = 0

    print(f"Processing {source} -> {output_path} ...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        coco_results = coco_model.track(
            frame, persist=True, conf=0.35, tracker="botsort.yaml",
            classes=None, verbose=False,
        )
        if coco_results[0].boxes.id is not None:
            for box, cls_id, track_id in zip(coco_results[0].boxes.xyxy,
                                              coco_results[0].boxes.cls,
                                              coco_results[0].boxes.id):
                cls_id = int(cls_id)
                track_id = int(track_id)
                key = f"coco_{track_id}"
                track_hit_counts[key] = track_hit_counts.get(key, 0) + 1
                if track_hit_counts[key] < MIN_HITS_TO_COUNT:
                    continue
                if cls_id == PERSON_CLASS:
                    draw_box(frame, box.tolist(), "person", GREEN)
                elif cls_id in RELEVANT_COCO_CLASSES:
                    draw_box(frame, box.tolist(), coco_model.names[cls_id], GREEN)

        custom_results = custom_model.track(
            frame, persist=True, conf=0.35, tracker="botsort.yaml",
            verbose=False,
        )
        if custom_results[0].boxes.id is not None:
            for box, cls_id, track_id in zip(custom_results[0].boxes.xyxy,
                                              custom_results[0].boxes.cls,
                                              custom_results[0].boxes.id):
                cls_id = int(cls_id)
                track_id = int(track_id)
                key = f"custom_{track_id}"
                track_hit_counts[key] = track_hit_counts.get(key, 0) + 1
                if track_hit_counts[key] < MIN_HITS_TO_COUNT:
                    continue
                if box_area_fraction(box.tolist(), frame.shape[1], frame.shape[0]) > MAX_BOX_AREA_FRACTION:
                    continue  # box too large to be real -- likely a false positive
                draw_box(frame, box.tolist(), CUSTOM_CLASS_NAMES[cls_id], RED)

        writer.write(frame)

        if frame_num % 50 == 0:
            print(f"  ...processed {frame_num} frames")

    cap.release()
    writer.release()
    print(f"\nDone. Saved annotated video to: {os.path.abspath(output_path)}")
    print("Open that file directly in Windows (double-click in File Explorer) to watch it.")


if __name__ == "__main__":
    main()