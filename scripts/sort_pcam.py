"""
sort_pcam.py
Sorts images from the Kaggle "histopathologic-cancer-detection" dataset
(train/ folder of .tif images + train_labels.csv with id,label columns)
into data/processed/benign/ and data/processed/malignant/ as .png files.
"""

import argparse
import csv
import os
import random

from PIL import Image


def sort_images(raw_dir: str, out_dir: str, limit: int, seed: int):
    train_dir = os.path.join(raw_dir, "train")
    labels_csv = os.path.join(raw_dir, "train_labels.csv")

    if not os.path.isdir(train_dir):
        print(f"ERROR: train/ folder not found at {train_dir}")
        return
    if not os.path.isfile(labels_csv):
        print(f"ERROR: train_labels.csv not found at {labels_csv}")
        return

    benign_dir = os.path.join(out_dir, "benign")
    malignant_dir = os.path.join(out_dir, "malignant")
    os.makedirs(benign_dir, exist_ok=True)
    os.makedirs(malignant_dir, exist_ok=True)

    benign_ids, malignant_ids = [], []
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["label"] == "1":
                malignant_ids.append(row["id"])
            else:
                benign_ids.append(row["id"])

    print(f"Found in CSV: benign={len(benign_ids)}  malignant={len(malignant_ids)}")

    random.seed(seed)
    random.shuffle(benign_ids)
    random.shuffle(malignant_ids)

    per_class = limit // 2
    benign_ids = benign_ids[:per_class]
    malignant_ids = malignant_ids[:per_class]

    def convert_and_copy(ids, dest_dir, label_name):
        ok, missing = 0, 0
        for img_id in ids:
            src = os.path.join(train_dir, img_id + ".tif")
            if not os.path.isfile(src):
                missing += 1
                continue
            dst = os.path.join(dest_dir, img_id + ".png")
            try:
                with Image.open(src) as im:
                    im.convert("RGB").save(dst, "PNG")
                ok += 1
            except Exception as e:
                print(f"  skip {img_id}: {e}")
        print(f"{label_name}: converted={ok}, missing_source={missing}")

    convert_and_copy(benign_ids, benign_dir, "benign")
    convert_and_copy(malignant_ids, malignant_dir, "malignant")

    print(f"Done. Total images ready in {out_dir}: "
          f"{len(os.listdir(benign_dir))} benign, {len(os.listdir(malignant_dir))} malignant")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default=os.path.expanduser(
        "~/.cache/kagglehub/competitions/histopathologic-cancer-detection"))
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--limit", type=int, default=4000,
                         help="Total images to use (split evenly benign/malignant).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sort_images(args.raw_dir, args.out_dir, args.limit, args.seed)