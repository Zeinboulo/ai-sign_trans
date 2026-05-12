import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

KEYPOINTS_ROOT = Path("./keypoints_output")
NORMALIZED_ROOT = Path("./keypoints_normalized")
FINAL_ROOT = Path("./keypoints_final")
REPORT_DIR = Path("./analysis_report")


def make_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for i in range(1, 10000):
        candidate = path.with_name(f"{stem}__dup{i}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Too many duplicates for {path}")


def normalize_keypoints(kp: np.ndarray, log_prefix: str) -> np.ndarray:
    t = kp.shape[0]
    pose = kp[:, 126:225].reshape(t, 33, 3)
    left_shoulder = pose[:, 11, :]
    right_shoulder = pose[:, 12, :]
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=1)

    invalid = ~np.isfinite(shoulder_width) | (shoulder_width < 1e-6)
    if np.all(invalid):
        print(f"WARNING: {log_prefix} shoulder width invalid for all frames, using 1.0")
    shoulder_width = np.where(invalid, 1.0, shoulder_width)

    kp_norm = kp / shoulder_width[:, np.newaxis]
    kp_norm = np.clip(kp_norm, -5.0, 5.0)
    kp_norm = np.nan_to_num(kp_norm, nan=0.0).astype(np.float32)
    return kp_norm


def compute_max_len(lengths: List[int]) -> int:
    if not lengths:
        return 0
    p90 = np.percentile(lengths, 90)
    return int(math.ceil(p90 / 10.0) * 10)


def load_quality_report() -> pd.DataFrame:
    report_path = REPORT_DIR / "video_quality_report.csv"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing report: {report_path}")
    return pd.read_csv(report_path)


def main() -> int:
    start = time.time()
    df = load_quality_report()

    total = len(df)
    passed = df[(df["flag_as_bad"] == False) & (df["quality_score"] >= 60)]
    passed = passed[passed["total_frames"] >= 5]
    print(f"Videos passing filter: {len(passed)} / {total}")

    if total == 0:
        print("No videos in report. Exiting.")
        return 1

    filtered_words = set(passed["word"].unique())
    all_words = set(df["word"].unique())
    empty_words = sorted([w for w in all_words if w not in filtered_words])
    for word in empty_words:
        print(f"WARNING: word folder has 0 usable videos after filtering: {word}")

    NORMALIZED_ROOT.mkdir(parents=True, exist_ok=True)
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)

    normalized_records: List[Dict] = []
    for _, row in passed.iterrows():
        rel_path = Path(row["npy_path"])
        src_path = KEYPOINTS_ROOT / rel_path
        if not src_path.exists():
            print(f"MISSING: {src_path}")
            continue

        try:
            kp = np.load(src_path)
            if kp.ndim != 2 or kp.shape[1] != 225:
                print(f"INVALID SHAPE: {src_path} -> {kp.shape}")
                continue
            if kp.shape[0] < 5:
                print(f"SKIP SHORT: {src_path}")
                continue

            kp_norm = normalize_keypoints(kp, str(src_path))

            out_path = NORMALIZED_ROOT / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path = make_unique_path(out_path)
            np.save(out_path, kp_norm)

            normalized_records.append(
                {
                    "word": row["word"],
                    "npy_path": str(rel_path).replace("\\", "/"),
                    "norm_path": str(out_path.relative_to(NORMALIZED_ROOT)).replace("\\", "/"),
                    "quality_score": float(row["quality_score"]),
                    "original_frames": int(kp.shape[0]),
                }
            )
        except Exception as exc:
            print(f"FAILED: {src_path} -> {exc}")
            continue

    if not normalized_records:
        print("No normalized files created. Exiting.")
        return 1

    lengths = [r["original_frames"] for r in normalized_records]
    max_len = compute_max_len(lengths)
    print(f"Computed MAX_LEN: {max_len}")

    final_records: List[Dict] = []
    for rec in normalized_records:
        norm_path = NORMALIZED_ROOT / rec["norm_path"]
        if not norm_path.exists():
            print(f"MISSING: {norm_path}")
            continue

        try:
            kp_norm = np.load(norm_path)
            t = kp_norm.shape[0]
            if max_len == 0:
                print("MAX_LEN is 0, skipping finalization")
                break

            if t > max_len:
                start_idx = (t - max_len) // 2
                kp_final = kp_norm[start_idx : start_idx + max_len]
            elif t < max_len:
                pad = np.zeros((max_len - t, 225), dtype=np.float32)
                kp_final = np.concatenate([kp_norm, pad], axis=0)
            else:
                kp_final = kp_norm

            final_path = FINAL_ROOT / rec["norm_path"]
            final_path.parent.mkdir(parents=True, exist_ok=True)
            final_path = make_unique_path(final_path)
            np.save(final_path, kp_final.astype(np.float32))

            final_records.append(
                {
                    "word": rec["word"],
                    "video_path": str(final_path.relative_to(FINAL_ROOT)).replace("\\", "/"),
                    "quality_score": rec["quality_score"],
                    "original_frames": rec["original_frames"],
                    "final_frames": int(max_len),
                }
            )
        except Exception as exc:
            print(f"FAILED: {norm_path} -> {exc}")
            continue

    if not final_records:
        print("No final files created. Exiting.")
        return 1

    words_sorted = sorted({r["word"] for r in final_records})
    label_map = {word: idx for idx, word in enumerate(words_sorted)}
    label_map_path = Path("./label_map.json")
    label_map_path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    manifest_rows: List[Dict] = []
    for rec in final_records:
        rec_out = dict(rec)
        rec_out["label_index"] = label_map[rec["word"]]
        manifest_rows.append(rec_out)

    manifest_df = pd.DataFrame(
        manifest_rows,
        columns=[
            "word",
            "video_path",
            "quality_score",
            "original_frames",
            "final_frames",
            "label_index",
        ],
    )
    manifest_path = Path("./dataset_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)

    # Sanity checks
    sample_files = list(FINAL_ROOT.rglob("*.npy"))
    random.seed(42)
    sample_files = random.sample(sample_files, k=min(5, len(sample_files)))

    for path in sample_files:
        arr = np.load(path)
        checks = {
            "shape": arr.shape == (max_len, 225),
            "no_nan": not np.isnan(arr).any(),
            "range": arr.min() >= -5.0 and arr.max() <= 5.0,
            "dtype": arr.dtype == np.float32,
        }
        for name, ok in checks.items():
            status = "PASS" if ok else "FAIL"
            print(f"{status}: {name} -> {path}")
            if not ok:
                if name == "shape":
                    print(f"  shape={arr.shape}, expected=({max_len}, 225)")
                if name == "dtype":
                    print(f"  dtype={arr.dtype}, expected=float32")
                if name == "range":
                    print(f"  range=({arr.min()}, {arr.max()})")
                if name == "no_nan":
                    print("  contains NaN values")

    # Final summary report
    report_df = df
    total_videos = len(report_df)
    passed_videos = len(passed)
    discarded = total_videos - passed_videos
    passed_pct = (passed_videos / total_videos * 100.0) if total_videos else 0.0
    discarded_pct = (discarded / total_videos * 100.0) if total_videos else 0.0
    avg_quality = float(report_df["quality_score"].mean()) if total_videos else 0.0

    summary = (
        "+---------------------------------------------+\n"
        "|  PIPELINE COMPLETE                          |\n"
        "+---------------------------------------------+\n"
        f"|  Word classes         : {len(words_sorted):<19}|\n"
        f"|  Total videos         : {total_videos:<19}|\n"
        f"|  Passed quality filter: {passed_videos:<3} ({passed_pct:>5.1f}%)       |\n"
        f"|  Discarded            : {discarded:<3} ({discarded_pct:>5.1f}%)       |\n"
        f"|  Avg quality score    : {avg_quality:>6.2f}/100          |\n"
        f"|  MAX_LEN (frames)     : {max_len:<19}|\n"
        f"|  Final array shape    : ({max_len}, 225)    |\n"
        f"|  Manifest rows        : {len(manifest_df):<19}|\n"
        "+---------------------------------------------+\n"
        "|  OUTPUT LOCATIONS                           |\n"
        "|  Raw keypoints    -> ./keypoints_output/     |\n"
        "|  Normalized       -> ./keypoints_normalized/ |\n"
        "|  Final (padded)   -> ./keypoints_final/      |\n"
        "|  Quality report   -> ./analysis_report/      |\n"
        "|  Manifest CSV     -> ./dataset_manifest.csv  |\n"
        "|  Label map        -> ./label_map.json        |\n"
        "+---------------------------------------------+"
    )

    print(summary)
    print(f"Preprocessing done in {time.time() - start:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
