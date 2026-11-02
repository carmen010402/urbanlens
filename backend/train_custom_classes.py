"""
train_custom_classes.py

Fine-tunes a YOLOv8 model on just your 3 new classes (hole, dump,
weather_collision), using the merged dataset from merge_yolo_datasets.py.

WHY A SEPARATE MODEL INSTEAD OF RETRAINING YOUR EXISTING ONE:
Your current detect.py uses yolov8s.pt, which already knows 80 COCO classes
(person, chair, backpack, etc). Retraining THAT model to also know your 3
new classes would need you to also re-provide COCO's full image set so it
doesn't forget those 80 classes (a huge amount of data you don't have).

Instead, this trains a SEPARATE, small model that only knows your 3 new
classes. At runtime, you run BOTH models on each camera frame and combine
their results -- the "two-model" approach. This needs far less data, far
less training time, and can't break anything that already works.

HOW TO RUN:
  pip3 install ultralytics --break-system-packages
  python3 train_custom_classes.py

This produces a file: runs/detect/train/weights/best.pt
That's your new hole/dump/weather_collision detector.
"""

from ultralytics import YOLO

DATA_YAML = "./combined_dataset/data.yaml"   # from merge_yolo_datasets.py
EPOCHS = 100            # how many passes over the data; 50 is a reasonable start
IMAGE_SIZE = 640
BASE_MODEL = "yolov8n.pt"   # "nano" -- smallest/fastest, good for a 3-class model


def main():
    model = YOLO(BASE_MODEL)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        patience=15,      # stop early if it stops improving for 15 epochs
        name="hole_dump_weather_model",
        batch=8,          # reduced from default 16 -- lowers memory use
        workers=2,        # reduced from default 8 -- lowers RAM use (main fix for "Killed")
        cache=False,      # don't hold all images in RAM at once
    )

    print("\nTraining complete.")
    print("Your new model is at: runs/detect/hole_dump_weather_model/weights/best.pt")
    print("Copy that file next to detect.py and load it as a second model.")


if __name__ == "__main__":
    main()