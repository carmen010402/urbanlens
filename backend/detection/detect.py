import cv2
from ultralytics import YOLO
import json
from datetime import datetime

model = YOLO('yolov8s.pt')
cap = cv2.VideoCapture("shot3.mp4")

seen_person_ids = set()
seen_obstacle_ids = set()
obstacle_log = []

PERSON_CLASS = 0

# COCO class IDs relevant to a ground-survey rover.
# person(0) is handled separately above -- everything else here counts as "obstacle".
RELEVANT_OBSTACLE_CLASSES = {
    56,  # chair
    24,  # backpack
    28,  # suitcase (stand-in for "box")
    26,  # handbag
    13,  # bench
    58,  # potted plant
    41,  # cup / bottle-type litter, optional
}

track_hit_counts = {}
MIN_HITS_TO_COUNT = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        conf=0.35,
        tracker="botsort_custom.yaml",
        classes=None,
    )

    annotated = results[0].plot()

    if results[0].boxes.id is not None:
        for box, cls_id, track_id in zip(results[0].boxes.xyxy,
                                          results[0].boxes.cls,
                                          results[0].boxes.id):
            cls_id = int(cls_id)
            track_id = int(track_id)
            x1, y1, x2, y2 = box.tolist()
            center_x = (x1 + x2) / 2
            frame_width = frame.shape[1]

            track_hit_counts[track_id] = track_hit_counts.get(track_id, 0) + 1
            if track_hit_counts[track_id] < MIN_HITS_TO_COUNT:
                continue

            if cls_id == PERSON_CLASS:
                seen_person_ids.add(track_id)
            elif cls_id in RELEVANT_OBSTACLE_CLASSES:
                if track_id not in seen_obstacle_ids:
                    seen_obstacle_ids.add(track_id)

                    if center_x < frame_width / 3:
                        position = "left"
                    elif center_x < 2 * frame_width / 3:
                        position = "center"
                    else:
                        position = "right"

                    obstacle_log.append({
                        "track_id": track_id,
                        "type": model.names[cls_id],
                        "position_in_frame": position,
                        "first_seen": datetime.now().isoformat(),
                    })

    cv2.putText(annotated, f"People: {len(seen_person_ids)}  Obstacles: {len(obstacle_log)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Detection - press q to quit", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

session_data = {
    "timestamp": datetime.now().isoformat(),
    "unique_people_count": len(seen_person_ids),
    "obstacles": obstacle_log
}

with open("session_results.json", "w") as f:
    json.dump(session_data, f, indent=2)

print(f"\nSaved results to session_results.json")
print(f"Final unique people: {len(seen_person_ids)}")
print(f"Obstacle sightings logged: {len(obstacle_log)}")