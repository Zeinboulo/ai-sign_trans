import argparse
import atexit
import hashlib
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core import base_options as base_options_lib
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.vision.core import vision_task_running_mode

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MODEL_DIR = Path("./mediapipe_models")
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
EXCLUDE_DIRS = {
    "keypoints_output",
    "keypoints_normalized",
    "keypoints_final",
    "analysis_report",
    "__pycache__",
    ".venv",
}

_HAND_LANDMARKER = None
_POSE_LANDMARKER = None


def ensure_model(path: Path, url: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    print(f"Downloading model: {url}")
    import urllib.request

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp_path)
    tmp_path.replace(path)
    return path


def _close_landmarkers() -> None:
    global _HAND_LANDMARKER, _POSE_LANDMARKER
    if _HAND_LANDMARKER is not None:
        _HAND_LANDMARKER.close()
        _HAND_LANDMARKER = None
    if _POSE_LANDMARKER is not None:
        _POSE_LANDMARKER.close()
        _POSE_LANDMARKER = None


def init_worker(hand_model_path: str, pose_model_path: str, min_det: float, min_track: float) -> None:
    global _HAND_LANDMARKER, _POSE_LANDMARKER
    running_mode = vision_task_running_mode.VisionTaskRunningMode.IMAGE
    hand_options = HandLandmarkerOptions(
        base_options=base_options_lib.BaseOptions(model_asset_path=hand_model_path),
        running_mode=running_mode,
        num_hands=2,
        min_hand_detection_confidence=min_det,
        min_hand_presence_confidence=min_det,
        min_tracking_confidence=min_track,
    )
    pose_options = PoseLandmarkerOptions(
        base_options=base_options_lib.BaseOptions(model_asset_path=pose_model_path),
        running_mode=running_mode,
        num_poses=1,
        min_pose_detection_confidence=min_det,
        min_pose_presence_confidence=min_det,
        min_tracking_confidence=min_track,
        output_segmentation_masks=False,
    )
    _HAND_LANDMARKER = HandLandmarker.create_from_options(hand_options)
    _POSE_LANDMARKER = PoseLandmarker.create_from_options(pose_options)
    atexit.register(_close_landmarkers)


def has_direct_videos(folder: Path) -> bool:
    if not folder.exists() or not folder.is_dir():
        return False
    for item in folder.iterdir():
        if item.is_file() and item.suffix.lower() in VIDEO_EXTS:
            return True
    return False


def is_dataset_root(folder: Path) -> bool:
    if not folder.exists() or not folder.is_dir():
        return False
    for sub in folder.iterdir():
        if sub.is_dir() and has_direct_videos(sub):
            return True
    return False


def find_dataset_root(base_dir: Path) -> Optional[Path]:
    candidates: List[Path] = []
    for root, dirs, _ in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        root_path = Path(root)
        if root_path.name in EXCLUDE_DIRS:
            continue
        for d in sorted(dirs):
            sub = root_path / d
            if has_direct_videos(sub):
                candidates.append(root_path)
                break
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: str(p).lower())[0]


def iter_videos(word_dir: Path) -> Iterable[Path]:
    for path in word_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            yield path


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


def output_name_for(video_path: Path) -> str:
    digest = hashlib.md5(str(video_path).encode("utf-8")).hexdigest()[:8]
    return f"{video_path.stem}__{digest}.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MediaPipe keypoints.")
    parser.add_argument("--dataset", default="./dataset", help="Dataset root folder")
    parser.add_argument("--output", default="./keypoints_output", help="Output folder")
    parser.add_argument("--min_detection", type=float, default=0.5)
    parser.add_argument("--min_tracking", type=float, default=0.5)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker processes to use (0 = auto)",
    )
    return parser.parse_args()


def process_video(task: Tuple[str, str]) -> Tuple[bool, str, str]:
    video_path = Path(task[0])
    out_word_dir = Path(task[1])

    if _HAND_LANDMARKER is None or _POSE_LANDMARKER is None:
        return False, str(video_path), "Landmarkers not initialized"

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return False, str(video_path), "Failed to open video"

        frames: List[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hands_res = _HAND_LANDMARKER.detect(image)
            pose_res = _POSE_LANDMARKER.detect(image)

            left = np.full((21, 3), np.nan, dtype=np.float32)
            right = np.full((21, 3), np.nan, dtype=np.float32)
            pose_arr = np.full((33, 3), np.nan, dtype=np.float32)

            if hands_res.hand_landmarks and hands_res.handedness:
                for hand_lm, hand_info in zip(
                    hands_res.hand_landmarks,
                    hands_res.handedness,
                ):
                    label = hand_info[0].category_name or ""
                    label = label.lower()
                    target = left if label == "left" else right
                    for i, lm in enumerate(hand_lm):
                        target[i, 0] = lm.x
                        target[i, 1] = lm.y
                        target[i, 2] = lm.z

            if pose_res.pose_landmarks:
                pose_lms = pose_res.pose_landmarks[0]
                for i, lm in enumerate(pose_lms):
                    pose_arr[i, 0] = lm.x
                    pose_arr[i, 1] = lm.y
                    pose_arr[i, 2] = lm.z

            frame_kp = np.concatenate(
                [left.reshape(-1), right.reshape(-1), pose_arr.reshape(-1)]
            ).astype(np.float32)
            frames.append(frame_kp)

        if not frames:
            return False, str(video_path), "No frames read"

        kp = np.stack(frames, axis=0).astype(np.float32)
        out_word_dir.mkdir(parents=True, exist_ok=True)
        out_name = output_name_for(video_path)
        out_path = out_word_dir / out_name
        if out_path.exists():
            out_path = make_unique_path(out_path)
        np.save(out_path, kp)
        return True, str(video_path), ""
    except Exception as exc:
        return False, str(video_path), str(exc)
    finally:
        cap.release()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset).resolve()
    if not is_dataset_root(dataset_root):
        auto_root = find_dataset_root(Path.cwd())
        if auto_root is None:
            print("No dataset root found. Expected a folder with subfolders of videos.")
            return 1
        print(f"Dataset not found at {dataset_root}. Using auto-detected: {auto_root}")
        dataset_root = auto_root

    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    succeeded = 0
    failed = 0
    start_time = time.time()

    model_dir = MODEL_DIR.resolve()
    hand_model = ensure_model(model_dir / "hand_landmarker.task", HAND_MODEL_URL)
    pose_model = ensure_model(model_dir / "pose_landmarker_full.task", POSE_MODEL_URL)

    word_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
    tasks: List[Tuple[str, str]] = []
    for word_dir in word_dirs:
        videos = list(iter_videos(word_dir))
        if not videos:
            continue
        word = word_dir.name
        out_word_dir = output_root / word
        for video_path in sorted(videos):
            tasks.append((str(video_path), str(out_word_dir)))

    total = len(tasks)
    if total == 0:
        print("No videos found in dataset.")
        return 1

    workers = args.workers
    if workers <= 0:
        cpu_count = os.cpu_count() or 1
        workers = min(4, cpu_count)
    workers = max(1, min(workers, total))

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=init_worker,
        initargs=(str(hand_model), str(pose_model), args.min_detection, args.min_tracking),
    ) as executor:
        for ok, video_path, error in executor.map(process_video, tasks, chunksize=1):
            if ok:
                succeeded += 1
            else:
                failed += 1
                print(f"FAILED: {video_path} -> {error}")

    elapsed = time.time() - start_time
    print("Extraction complete")
    print(f"Total videos attempted: {total}")
    print(f"Total succeeded: {succeeded}")
    print(f"Total failed: {failed}")
    print(f"Total time taken (s): {elapsed:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
