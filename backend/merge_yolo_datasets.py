"""
merge_yolo_datasets.py

Combines several downloaded YOLO-format datasets (each with their own,
different class lists) into ONE unified 3-class dataset:
    0 = hole
    1 = dump
    2 = weather_collision

WHY THIS IS NEEDED:
Each downloaded dataset has its own classes.txt / data.yaml with its own
class numbering (e.g. Mendeley's "Pothole" might be class 1, TACO's trash
categories are 60 different classes, Fallen_Trees has "tree"/"trunk" as
0/1). This script re-labels every dataset's annotation files so they all
point at the SAME 3 target classes, then copies everything into one
combined folder ready for training.

HOW TO USE:
1. Download and unzip each source dataset somewhere on your machine.
2. Edit the SOURCES list below: for each dataset, point "path" at its
   folder, and fill "class_map" with {original_class_id: target_class_id}
   for every original class you want to KEEP (leave others out entirely --
   they'll be skipped, not merged in as junk).
3. Run: python3 merge_yolo_datasets.py
4. A new folder "combined_dataset/" appears, ready for training.

EXPECTED INPUT FORMAT (standard YOLO):
  some_dataset/
    images/
      img1.jpg
      img2.jpg
    labels/
      img1.txt   (each line: class_id x_center y_center width height, normalized 0-1)
      img2.txt
(Some datasets use train/valid/test subfolders instead -- see NOTE below.)
"""

import os
import shutil
import random

TARGET_CLASSES = ["hole", "dump", "weather_collision"]

# ---------------------------------------------------------------------------
# EDIT THIS SECTION to match your actual downloaded folders.
# "class_map" keys are the ORIGINAL dataset's class IDs (check its
# classes.txt or data.yaml to find these numbers), values are indices into
# TARGET_CLASSES above (0=hole, 1=dump, 2=weather_collision).
# ---------------------------------------------------------------------------
SOURCES = [
    {
        "name": "mendeley_road_damage",
        "path": "./Data Y12 Final",             # unzipped Road_Damage_Annotated_Data.zip
        "class_map": {1: 0},                    # their class 1 = "Pothole" -> hole (0)
    },
    {
        "name": "taco",
        "path": "./TACO/TACO_yolo",             # after running convert_taco_to_yolo.py
        "class_map": "ALL_TO:1",                # all TACO litter categories -> dump (1)
    },
    {
        "name": "fallen_trees",
        "path": "./fallen_trees_dataset",       # unzipped Fallen_Trees_v6i_yolov8.zip
        "class_map": {0: 2},                    # their only class (unnamed, "fallen tree") -> weather_collision (2)
    },
]

OUTPUT_DIR = "./combined_dataset"
VAL_SPLIT = 0.15  # 15% of merged images held out for validation


def find_image_label_pairs(dataset_path: str):
    """
    Handles both common layouts:
      dataset/images/*.jpg + dataset/labels/*.txt
      dataset/train/images/*.jpg + dataset/train/labels/*.txt (+ valid/, test/)
    Returns a list of (image_path, label_path) tuples.
    """
    pairs = []
    for root, dirs, files in os.walk(dataset_path):
        if os.path.basename(root) != "images":
            continue
        labels_dir = os.path.join(os.path.dirname(root), "labels")
        if not os.path.isdir(labels_dir):
            continue
        for f in files:
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(root, f)
            label_name = os.path.splitext(f)[0] + ".txt"
            label_path = os.path.join(labels_dir, label_name)
            if os.path.exists(label_path):
                pairs.append((img_path, label_path))
    return pairs


def remap_label_file(label_path: str, class_map) -> list:
    """
    Reads a YOLO .txt label file, remaps class ids per class_map, and
    returns the new lines (skipping any class not in class_map, unless
    class_map is the "ALL_TO:<id>" special string).
    """
    new_lines = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            orig_class = int(parts[0])
            coords = parts[1:]

            if isinstance(class_map, str) and class_map.startswith("ALL_TO:"):
                new_class = int(class_map.split(":")[1])
            elif orig_class in class_map:
                new_class = class_map[orig_class]
            else:
                continue  # this class isn't one we want -- drop the line

            new_lines.append(f"{new_class} {' '.join(coords)}")
    return new_lines


def main():
    # FIX: always start from a clean output folder (stops train/val overlap)
    if os.path.exists(OUTPUT_DIR):
        print(f"Removing old {OUTPUT_DIR} so the split starts clean...")
        shutil.rmtree(OUTPUT_DIR)

    all_images_dir = os.path.join(OUTPUT_DIR, "all_images")
    all_labels_dir = os.path.join(OUTPUT_DIR, "all_labels")
    os.makedirs(all_images_dir, exist_ok=True)
    os.makedirs(all_labels_dir, exist_ok=True)

    kept_count = 0
    skipped_count = 0

    for source in SOURCES:
        if not os.path.isdir(source["path"]):
            print(f"SKIPPING '{source['name']}': folder not found at {source['path']}")
            continue

        pairs = find_image_label_pairs(source["path"])
        print(f"{source['name']}: found {len(pairs)} image/label pairs")

        for img_path, label_path in pairs:
            new_lines = remap_label_file(label_path, source["class_map"])
            if not new_lines:
                skipped_count += 1
                continue  # no relevant objects in this image -- skip it entirely

            unique_name = f"{source['name']}_{os.path.basename(img_path)}"
            unique_name = unique_name.replace(" ", "_")
            new_img_path = os.path.join(all_images_dir, unique_name)
            new_label_path = os.path.join(
                all_labels_dir, os.path.splitext(unique_name)[0] + ".txt"
            )

            shutil.copy(img_path, new_img_path)
            with open(new_label_path, "w") as f:
                f.write("\n".join(new_lines) + "\n")
            kept_count += 1

    print(f"\nKept {kept_count} images with relevant objects, skipped {skipped_count} with none.")

    # Split into train/val
    all_images = os.listdir(all_images_dir)
    random.seed(42)
    random.shuffle(all_images)
    val_count = int(len(all_images) * VAL_SPLIT)
    val_images = set(all_images[:val_count])

    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

    for img_name in all_images:
        split = "val" if img_name in val_images else "train"
        label_name = os.path.splitext(img_name)[0] + ".txt"
        shutil.move(os.path.join(all_images_dir, img_name),
                    os.path.join(OUTPUT_DIR, "images", split, img_name))
        shutil.move(os.path.join(all_labels_dir, label_name),
                    os.path.join(OUTPUT_DIR, "labels", split, label_name))

    shutil.rmtree(all_images_dir)
    shutil.rmtree(all_labels_dir)

    # Write the data.yaml the training script needs
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"train: {os.path.abspath(os.path.join(OUTPUT_DIR, 'images/train'))}\n")
        f.write(f"val: {os.path.abspath(os.path.join(OUTPUT_DIR, 'images/val'))}\n")
        f.write(f"nc: {len(TARGET_CLASSES)}\n")
        f.write(f"names: {TARGET_CLASSES}\n")

    print(f"\nDone. {len(all_images) - val_count} train images, {val_count} val images.")
    print(f"Dataset ready at: {os.path.abspath(OUTPUT_DIR)}")
    print(f"data.yaml written -- use this path when training.")


if __name__ == "__main__":
    main()