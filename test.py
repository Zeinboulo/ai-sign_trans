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

MODEL_DIR = Path("./mediapipe_models")
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"


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



def process_video(video_path):
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

            frame_video_representation = np.concatenate(
                [left.reshape(-1), right.reshape(-1), pose_arr.reshape(-1)]
            ).astype(np.float32)
            frames.append(frame_video_representation)

        if not frames:
            return False, str(video_path), "No frames read"

        video_representation = np.stack(frames, axis=0).astype(np.float32)
    except Exception as e:
        return False, str(video_path), f"Error processing video: {e}"
    
    return video_representation

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


def videos_per_word(folder_path):
    results = {}
    for video_path in os.listdir(folder_path):
        video_representation = run(os.path.join(folder_path, video_path))
        results[video_path] = video_representation
        # Process the video representation as needed
    return results

def all_videos(root_folder):
    overall_results = {}
    for word_folder in root_folder.iterdir():
        if word_folder.is_dir():
            word = word_folder.name
            overall_results[word] = videos_per_word(word_folder)
    return overall_results


from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean

def _fill_nan(rep: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN values per feature column (missing landmarks)."""
    rep = rep.copy()
    for col in range(rep.shape[1]):
        col_data = rep[:, col]
        nan_mask = np.isnan(col_data)
        if nan_mask.all():
            rep[:, col] = 0.0  # entire column missing → zero out
        elif nan_mask.any():
            valid_idx = np.where(~nan_mask)[0]
            rep[:, col] = np.interp(np.arange(len(col_data)), valid_idx, col_data[valid_idx])
    return rep


def _resample(rep: np.ndarray, n_frames: int) -> np.ndarray:
    """Resample a (T, F) array to (n_frames, F) using linear interpolation."""
    T = len(rep)
    if T == n_frames:
        return rep
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, n_frames)
    f = interp1d(x_old, rep, axis=0, kind='linear')
    return f(x_new)


def _normalize(rep: np.ndarray) -> np.ndarray:
    """
    Normalize hand landmarks per frame:
    - Center each hand on its wrist (landmark 0)
    - Scale by wrist-to-middle-MCP distance (landmark 9)
    rep shape: (T, 225)  →  left(63) | right(63) | pose(99)
    """
    rep = rep.copy()
    for t in range(len(rep)):
        for offset in [0, 63]:                        # left hand at 0, right at 63
            hand = rep[t, offset:offset+63].reshape(21, 3)
            wrist = hand[0].copy()
            hand -= wrist                             # center on wrist
            scale = np.linalg.norm(hand[9])
            if scale > 1e-6:
                hand /= scale
            rep[t, offset:offset+63] = hand.reshape(-1)
    return rep


# ─────────────────────────────────────────────
# Approach 1 — Resample + Cosine (fast)
# ─────────────────────────────────────────────
def compute_similarity(rep1: np.ndarray, rep2: np.ndarray, n_frames: int = 30) -> float:
    """
    Similarity between two variable-length gesture sequences.
    rep1, rep2: shape (T, 225)  —  T can differ
    Returns cosine similarity in [0, 1].
    """
    # 1. Fill NaN (missing detections)
    r1 = _fill_nan(rep1)
    r2 = _fill_nan(rep2)

    # 2. Normalize landmarks
    r1 = _normalize(r1)
    r2 = _normalize(r2)

    # 3. Resample both to the same fixed length
    r1 = _resample(r1, n_frames)
    r2 = _resample(r2, n_frames)

    # 4. Flatten → 1D vectors
    v1 = r1.flatten()
    v2 = r2.flatten()

    # 5. Cosine similarity
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0

    sim = np.dot(v1, v2) / (norm1 * norm2)
    return float(np.clip((sim + 1) / 2, 0.0, 1.0))  # rescale [-1,1] → [0,1]


# ─────────────────────────────────────────────
# Approach 2 — DTW (more accurate, slower)
# ─────────────────────────────────────────────
def compute_similarity(rep1: np.ndarray, rep2: np.ndarray) -> float:
    """
    DTW-based similarity — handles different speeds & lengths natively.
    Returns similarity in [0, 1] (1 = identical).
    """
    r1 = _normalize(_fill_nan(rep1))   # (T1, 225)
    r2 = _normalize(_fill_nan(rep2))   # (T2, 225)

    T1, T2 = len(r1), len(r2)
    # DTW cost matrix
    dtw_matrix = np.full((T1 + 1, T2 + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            cost = np.linalg.norm(r1[i-1] - r2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],    # insertion
                dtw_matrix[i, j-1],    # deletion
                dtw_matrix[i-1, j-1],  # match
            )

    dtw_distance = dtw_matrix[T1, T2]
    # Normalize by path length and convert to similarity
    similarity = 1 / (1 + dtw_distance / (T1 + T2))
    return float(similarity)

def compute_similarity_per_word(results):
    similiarities = []
    for key,value in results.items():
        for key2, value2 in results.items():
            if key != key2:
                sim = compute_similarity(value, value2)
                similiarities.append(sim)
    return np.mean(similiarities) if similiarities else 0

def run(video_path):
    model_dir = MODEL_DIR.resolve()
    hand_model = ensure_model(model_dir / "hand_landmarker.task", HAND_MODEL_URL)
    pose_model = ensure_model(model_dir / "pose_landmarker_full.task", POSE_MODEL_URL)
    init_worker(str(hand_model), str(pose_model), 0.5, 0.5)
    video_representation = process_video(video_path)
    return video_representation

def load_npy_word(folder_path):
    results = {}
    for npy_file in os.listdir(folder_path):
        if npy_file.endswith(".npy"):
            word = npy_file[:-4]  # remove .npy extension
            video_representation = np.load(os.path.join(folder_path, npy_file))
            results[word] = video_representation
    return results

def load_all_words(root_folder):
    overall_results = {}
    for word_folder in root_folder.iterdir():
        if word_folder.is_dir():
            word = word_folder.name
            overall_results[word] = load_npy_word(word_folder)
    return overall_results

def compute_all_similarities(root_folder):
    all_results = load_all_words(root_folder)
    similarities = {}
    for word, results in all_results.items():
        sim = compute_similarity_per_word(results)
        similarities[word] = sim
    return similarities

similarities = compute_all_similarities(Path("keypoints_output"))


