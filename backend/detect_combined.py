"""
detect_combined.py

Runs BOTH your models on the same video/camera feed at once:
  1. yolov8s.pt      -- the original COCO model (person, chair, backpack, etc.)
  2. best.pt         -- your NEW custom model (hole, dump, weather_collision)

This is the "two-model" approach from train_custom_classes.py's notes --
each model only knows its own classes, and we just combine both sets of
detections into one findings list per frame. Nothing about your existing
detect.py logic needs to change; this sits alongside it.

HOW TO RUN:
  python3 detect_combined.py <video_file_or_0_for_webcam>

Examples:
  python3 detect_combined.py shot1.mp4
  python3 detect_combined.py 0          # 0 = live webcam
"""

import sys
import cv2
from ultralytics import YOLO

# --- Model 1: your original, general-purpose detector ---
coco_model = YOLO('yolov8s.pt')
PERSON_CLASS = 0
RELEVANT_COCO_CLASSES = {56, 24, 28, 26, 13, 58, 41}  # chair, backpack, suitcase, handbag, bench, potted plant, cup

# --- Model 2: your NEW custom-trained damage-type detector ---
# UPDATE THIS PATH to match your real trained model's location:
CUSTOM_MODEL_PATH = "runs/detect/hole_dump_weather_model-5/weights/best.pt" 
custom_model = YOLO(CUSTOM_MODEL_PATH)
CUSTOM_CLASS_NAMES = ["hole", "dump", "weather_collision"]

MIN_HITS_TO_COUNT = 5

# FIX: reject any custom-model detection whose box covers an unreasonably
# large fraction of the frame. Real potholes/litter/fallen branches never
# span half the image -- when this happens it's the model mistakenly boxing
# a whole wall/doorway/bed area as "dump", which we saw repeatedly in testing.
MAX_BOX_AREA_FRACTION = 0.25


def box_area_fraction(box, frame_width, frame_height):
    x1, y1, x2, y2 = box
    box_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_width * frame_height
    return box_area / frame_area


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 detect_combined.py <video_file_or_0_for_webcam>")
        sys.exit(1)

    source = sys.argv[1]
    cap = cv2.VideoCapture(int(source) if source == "0" else source)

    seen_person_ids = set()
    seen_obstacle_ids = set()      # from the COCO model
    seen_damage_ids = set()        # from the custom model
    track_hit_counts = {}

    obstacle_log = []
    damage_log = []

    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        frame_width = frame.shape[1]

        # --- Run Model 1: general objects/people ---
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
                    seen_person_ids.add(track_id)
                elif cls_id in RELEVANT_COCO_CLASSES:
                    if track_id not in seen_obstacle_ids:
                        seen_obstacle_ids.add(track_id)
                        obstacle_log.append({
                            "frame": frame_num,
                            "type": coco_model.names[cls_id],
                            "source": "coco_model",
                        })

        # --- Run Model 2: your custom damage types ---
        # NOTE: your custom model wasn't trained with a tracker in mind, so
        # we track it separately with its own BoT-SORT instance.
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

                if track_id not in seen_damage_ids:
                    if box_area_fraction(box.tolist(), frame_width, frame.shape[0]) > MAX_BOX_AREA_FRACTION:
                        continue  # box too large to be real -- likely a false positive
                    seen_damage_ids.add(track_id)
                    damage_log.append({
                        "frame": frame_num,
                        "type": CUSTOM_CLASS_NAMES[cls_id],
                        "source": "custom_model",
                    })

    cap.release()

    print(f"\n=== Results from {source} ===")
    print(f"Unique people: {len(seen_person_ids)}")
    print(f"General obstacles (COCO model): {len(obstacle_log)}")
    for o in obstacle_log:
        print(f"  - {o['type']} (frame {o['frame']})")
    print(f"Damage findings (custom model): {len(damage_log)}")
    for d in damage_log:
        print(f"  - {d['type']} (frame {d['frame']})")


if __name__ == "__main__":
    main()