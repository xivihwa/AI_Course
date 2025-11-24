import os
import glob
import random
from pathlib import Path
import cv2
import numpy as np
import albumentations as A
import yaml
from sklearn.model_selection import train_test_split

INPUT_DIR = Path("/kaggle/input/dataset-yolo")
OUTPUT_DIR = Path("/kaggle/working/dataset-aug")
N_AUG = 7
RANDOM_SEED = 42
MIN_BOX_AREA = 0.0005
IMG_EXTS = [".jpg", ".jpeg", ".png"]
TRAIN_SPLIT = 0.8

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

images_dir = INPUT_DIR / "images"
labels_dir = INPUT_DIR / "labels"

if not images_dir.exists():
    raise FileNotFoundError(f"Images directory not found: {images_dir}")
if not labels_dir.exists():
    raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

for split in ['train', 'val']:
    (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

augmenter = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=25,
                       border_mode=cv2.BORDER_REFLECT_101, p=0.7),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.MotionBlur(blur_limit=7, p=0.3),
    A.CLAHE(clip_limit=4.0, p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['labels'], min_area=1, min_visibility=0.3))

def read_yolo_txt(path: Path):
    boxes, labels = [], []
    if not path.exists():
        return boxes, labels
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = map(float, parts)
            boxes.append([x, y, w, h])
            labels.append(int(cls))
    return boxes, labels

def write_yolo_txt(path: Path, boxes, labels):
    with open(path, "w") as f:
        for cls, (x, y, w, h) in zip(labels, boxes):
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def validate_bbox(bbox):
    x, y, w, h = bbox
    if not (0 <= x <= 1 and 0 <= y <= 1):
        return False
    if not (0 < w <= 1 and 0 < h <= 1):
        return False
    if x - w / 2 < 0 or x + w / 2 > 1:
        return False
    if y - h / 2 < 0 or y + h / 2 > 1:
        return False
    return True

def bbox_area(b):
    return b[2] * b[3]

image_files = [p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
print(f"Found {len(image_files)} images.")

train_files, val_files = train_test_split(
    image_files,
    train_size=TRAIN_SPLIT,
    random_state=RANDOM_SEED
)

print(f"Train: {len(train_files)}, Val: {len(val_files)}")
print("Starting augmentation...")

def process_images(img_list, split_name, apply_aug=True):
    total_saved = 0
    out_img_dir = OUTPUT_DIR / split_name / "images"
    out_lbl_dir = OUTPUT_DIR / split_name / "labels"
    
    for img_path in img_list:
        base = img_path.stem
        label_path = labels_dir / f"{base}.txt"
        
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"Error reading: {img_path}")
            continue
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        boxes, labels = read_yolo_txt(label_path)
        
        if len(boxes) == 0:
            print(f"Skipped (no boxes): {base}")
            continue
        
        cv2.imwrite(str(out_img_dir / f"{base}.jpg"), img_bgr)
        write_yolo_txt(out_lbl_dir / f"{base}.txt", boxes, labels)
        total_saved += 1
        
        if apply_aug and split_name == 'train':
            for i in range(N_AUG):
                try:
                    aug = augmenter(image=img_rgb, bboxes=boxes, labels=labels)
                    aug_img = aug["image"]
                    aug_boxes = aug["bboxes"]
                    aug_labels = aug["labels"]
                    
                    new_boxes, new_labels = [], []
                    for bb, lb in zip(aug_boxes, aug_labels):
                        if validate_bbox(bb) and bbox_area(bb) >= MIN_BOX_AREA:
                            new_boxes.append(bb)
                            new_labels.append(lb)
                    
                    if len(new_boxes) == 0:
                        continue
                    
                    out_name = f"{base}_aug{i}.jpg"
                    cv2.imwrite(str(out_img_dir / out_name),
                                cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                    write_yolo_txt(out_lbl_dir / f"{base}_aug{i}.txt", new_boxes, new_labels)
                    total_saved += 1
                except Exception as e:
                    print(f"Augmentation error {base}_aug{i}: {e}")
                    continue
    
    return total_saved

train_saved = process_images(train_files, 'train', apply_aug=True)
print(f"Train: saved {train_saved} images")

val_saved = process_images(val_files, 'val', apply_aug=False)
print(f"Val: saved {val_saved} images")

print(f"Total saved {train_saved + val_saved} images")

class_ids = set()
for split in ['train', 'val']:
    lbl_dir = OUTPUT_DIR / split / "labels"
    for lbl in lbl_dir.glob("*.txt"):
        with open(lbl, "r") as f:
            for line in f:
                cls = int(float(line.split()[0]))
                class_ids.add(cls)

names = [f"class{c}" for c in sorted(class_ids)]

data_yaml = {
    "path": str(OUTPUT_DIR.absolute()),
    "train": "train/images",
    "val": "val/images",
    "nc": len(names),
    "names": names
}

with open(OUTPUT_DIR / "data.yaml", "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"\nCreated data.yaml with {len(names)} classes")
print(f"Dataset structure:\n{OUTPUT_DIR}/\n  train/\n    images/\n    labels/\n  val/\n    images/\n    labels/\n  data.yaml")