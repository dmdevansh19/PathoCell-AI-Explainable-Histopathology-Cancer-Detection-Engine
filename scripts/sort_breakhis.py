"""
Sorts BreakHis images from data/raw/ (recursive) into
data/processed/benign and data/processed/malignant using the filename label
(SOB_B_... = benign, SOB_M_... = malignant).

Usage:  python scripts/sort_breakhis.py
"""
import argparse
import os
import re
import shutil

PATTERN = re.compile(r"SOB_([BM])_", re.IGNORECASE)


def sort_images(raw_dir, out_dir):
    benign_dir = os.path.join(out_dir, "benign")
    malignant_dir = os.path.join(out_dir, "malignant")
    os.makedirs(benign_dir, exist_ok=True)
    os.makedirs(malignant_dir, exist_ok=True)
    nb = nm = ns = 0
    for root, _, files in os.walk(raw_dir):
        for fname in files:
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            m = PATTERN.search(fname)
            if not m:
                ns += 1
                continue
            dest = benign_dir if m.group(1).upper() == "B" else malignant_dir
            shutil.copy2(os.path.join(root, fname), os.path.join(dest, fname))
            if dest == benign_dir:
                nb += 1
            else:
                nm += 1
    print(f"Done. benign={nb}, malignant={nm}, skipped={ns}")
    if nb + nm == 0:
        print("No BreakHis images found. Make sure the dataset is extracted inside", raw_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="data/raw")
    p.add_argument("--out_dir", default="data/processed")
    a = p.parse_args()
    sort_images(a.raw_dir, a.out_dir)
