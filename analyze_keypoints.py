import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

VIDEO_EXTS = {".npy"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze keypoint quality.")
    parser.add_argument("--keypoints", default="./keypoints_output")
    parser.add_argument("--report_dir", default="./analysis_report")
    parser.add_argument("--bad_threshold", type=float, default=0.40)
    parser.add_argument("--sample_plots", type=int, default=15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    keypoints_root = Path(args.keypoints).resolve()
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    rows: List[Dict] = []
    failed = 0

    if not keypoints_root.exists():
        print(f"Keypoints root not found: {keypoints_root}")
        return 1

    npy_files = sorted([p for p in keypoints_root.rglob("*.npy") if p.is_file()])
    for npy_path in npy_files:
        try:
            rel = npy_path.relative_to(keypoints_root)
            word = rel.parts[0] if len(rel.parts) > 1 else "unknown"
            arr = np.load(npy_path)
            if arr.ndim != 2 or arr.shape[1] != 225:
                raise RuntimeError(f"Unexpected shape {arr.shape}")
            total_frames = arr.shape[0]

            valid_mask = np.isfinite(arr)
            valid_ratio = float(valid_mask.sum() / arr.size) if arr.size else 0.0
            quality_score = valid_ratio * 100.0

            threshold = args.bad_threshold
            if threshold > 1.0:
                bad_cutoff = threshold
            else:
                bad_cutoff = threshold * 100.0

            too_short = total_frames < 5
            flag_as_bad = bool(quality_score < bad_cutoff or too_short)
            reason = "too_short" if too_short else "low_quality" if quality_score < bad_cutoff else ""

            rows.append(
                {
                    "word": word,
                    "npy_path": str(rel).replace("\\", "/"),
                    "total_frames": total_frames,
                    "valid_ratio": valid_ratio,
                    "quality_score": quality_score,
                    "flag_as_bad": flag_as_bad,
                    "reason": reason,
                }
            )
        except Exception as exc:
            failed += 1
            print(f"FAILED: {npy_path} -> {exc}")

    df = pd.DataFrame(rows)
    csv_path = report_dir / "video_quality_report.csv"
    df.to_csv(csv_path, index=False)

    total_videos = len(df)
    usable = int((~df["flag_as_bad"]).sum()) if total_videos else 0
    flagged = int(df["flag_as_bad"].sum()) if total_videos else 0
    avg_quality = float(df["quality_score"].mean()) if total_videos else 0.0

    frame_stats = {
        "min": int(df["total_frames"].min()) if total_videos else 0,
        "median": float(df["total_frames"].median()) if total_videos else 0.0,
        "max": int(df["total_frames"].max()) if total_videos else 0,
    }

    word_scores = (
        df.groupby("word")["quality_score"].mean().sort_values(ascending=True)
        if total_videos
        else pd.Series(dtype=float)
    )
    worst_words = (
        word_scores.head(5).to_dict() if not word_scores.empty else {}
    )

    summary = {
        "total_videos": total_videos,
        "usable_videos": usable,
        "flagged_videos": flagged,
        "avg_quality_score": avg_quality,
        "frame_stats": frame_stats,
        "worst_words": worst_words,
        "failed_files": failed,
        "elapsed_seconds": time.time() - start,
    }

    summary_path = report_dir / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if total_videos:
        sns.set(style="whitegrid")

        plt.figure(figsize=(8, 4))
        sns.histplot(df["quality_score"], bins=20, kde=True)
        plt.title("Quality Score Distribution")
        plt.xlabel("Quality Score (0-100)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(report_dir / "quality_hist.png")
        plt.close()

        plt.figure(figsize=(8, 4))
        sns.histplot(df["total_frames"], bins=20, kde=False)
        plt.title("Frame Count Distribution")
        plt.xlabel("Frames")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(report_dir / "frame_count_hist.png")
        plt.close()

        sample_n = min(args.sample_plots, total_videos)
        random.seed(42)
        sample_df = df.sample(n=sample_n, random_state=42) if sample_n > 0 else df
        for _, row in sample_df.iterrows():
            npy_path = keypoints_root / row["npy_path"]
            try:
                arr = np.load(npy_path)
                valid_per_frame = np.isfinite(arr).mean(axis=1)
                plt.figure(figsize=(8, 3))
                plt.plot(valid_per_frame)
                plt.ylim(0.0, 1.0)
                plt.title(f"Valid Ratio per Frame: {row['word']}")
                plt.xlabel("Frame")
                plt.ylabel("Valid Ratio")
                out_name = f"valid_ratio_{Path(row['npy_path']).stem}.png"
                plt.tight_layout()
                plt.savefig(report_dir / out_name)
                plt.close()
            except Exception:
                plt.close()
                continue

    print("Analysis complete")
    print(f"Total videos analyzed: {total_videos}")
    print(f"Usable videos: {usable}")
    print(f"Flagged videos: {flagged}")
    print(f"Average quality score: {avg_quality:.2f}")
    print(
        f"Average frame count (min/median/max): {frame_stats['min']}/"
        f"{frame_stats['median']}/{frame_stats['max']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
