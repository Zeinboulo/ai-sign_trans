import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from model import CNNBiLSTM


def _pad_or_crop(arr: np.ndarray, max_len: int) -> np.ndarray:
    t = arr.shape[0]
    if t == max_len:
        return arr
    if t > max_len:
        start = (t - max_len) // 2
        return arr[start : start + max_len]
    pad = np.zeros((max_len - t, arr.shape[1]), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


def _load_export(model_path: Path) -> Dict:
    return torch.load(model_path, map_location="cpu")


def predict(npy_path: str, model_path: str) -> str:
    """
    Load a single .npy file and return the predicted Arabic word.
    """
    model_path = Path(model_path)
    payload = _load_export(model_path)

    label_map = payload["label_map"]
    inv_label_map = {v: k for k, v in label_map.items()}

    max_len = int(payload["max_len"])
    input_dim = int(payload.get("input_dim", 225))
    use_projection = bool(payload.get("use_projection", True))
    projection_dim = int(payload.get("projection_dim", 512))

    arr = np.load(npy_path).astype(np.float32)
    if arr.shape[0] != max_len:
        arr = _pad_or_crop(arr, max_len)
    if arr.shape[1] != input_dim:
        raise ValueError(f"Expected feature dim {input_dim}, got {arr.shape[1]}")

    mask = np.any(arr != 0, axis=1)
    length = int(mask.sum())
    if length <= 0:
        length = arr.shape[0]

    x = torch.from_numpy(arr).unsqueeze(0)
    lengths = torch.tensor([length], dtype=torch.long)

    model = CNNBiLSTM(
        num_classes=len(label_map),
        input_dim=input_dim,
        use_projection=use_projection,
        projection_dim=projection_dim,
    )
    model.load_state_dict(payload["model_state"])
    model.eval()

    with torch.no_grad():
        logits = model(x, lengths)
        pred_idx = int(torch.argmax(logits, dim=1).item())

    return inv_label_map[pred_idx]
