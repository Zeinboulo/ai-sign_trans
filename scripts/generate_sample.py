#!/usr/bin/env python3
"""Generate a tiny, anonymized sample keypoints file and a manifest for CI/tests.

This script creates:
- samples/sample_keypoints.npy  (random float32 array)
- splits/sample_manifest.csv    (CSV with columns: id,keypoints_path,label)
"""
import os
import csv
import numpy as np


ROOT = os.path.dirname(os.path.dirname(__file__))
SAMPLES_DIR = os.path.join(ROOT, "samples")
SPLITS_DIR = os.path.join(ROOT, "splits")


def generate_keypoints(path: str, frames=30, features=63):
    # Small deterministic pseudo-random tensor for tests
    rng = np.random.RandomState(42)
    data = rng.rand(frames, features).astype("float32")
    np.save(path, data)
    return data.shape


def write_manifest(manifest_path: str, keypoints_relpath: str):
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "keypoints_path", "label"])
        writer.writerow(["sample-000", keypoints_relpath, "sample_label"])


def main():
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)

    keypoints_path = os.path.join(SAMPLES_DIR, "sample_keypoints.npy")
    shape = generate_keypoints(keypoints_path)

    manifest_path = os.path.join(SPLITS_DIR, "sample_manifest.csv")
    # Store manifest with repo-relative path
    relpath = os.path.relpath(keypoints_path, ROOT)
    write_manifest(manifest_path, relpath)

    print(f"Wrote sample keypoints: {keypoints_path} (shape={shape})")
    print(f"Wrote sample manifest: {manifest_path}")


if __name__ == "__main__":
    main()
