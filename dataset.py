import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_label_map(path: Path) -> Dict[str, int]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    return pd.read_csv(path)


def _allocate_split_counts(n: int) -> Tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0

    # Ensure at least one sample per split when possible.
    remaining = n - 3
    base = [1, 1, 1]
    if remaining <= 0:
        return base[0], base[1], base[2]

    proportions = [0.70, 0.15, 0.15]
    raw = [p * remaining for p in proportions]
    add = [int(math.floor(x)) for x in raw]
    leftover = remaining - sum(add)
    frac = [raw[i] - add[i] for i in range(3)]
    order = sorted(range(3), key=lambda i: frac[i], reverse=True)
    for i in range(leftover):
        add[order[i % 3]] += 1

    return base[0] + add[0], base[1] + add[1], base[2] + add[2]


def split_manifest_stratified(
    df: pd.DataFrame,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    train_rows: List[pd.DataFrame] = []
    val_rows: List[pd.DataFrame] = []
    test_rows: List[pd.DataFrame] = []

    for label, group in df.groupby("label_index"):
        indices = list(group.index)
        rng.shuffle(indices)
        n = len(indices)
        n_train, n_val, n_test = _allocate_split_counts(n)
        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val : n_train + n_val + n_test]

        train_rows.append(df.loc[train_idx])
        if val_idx:
            val_rows.append(df.loc[val_idx])
        if test_idx:
            test_rows.append(df.loc[test_idx])

    train_df = pd.concat(train_rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = (
        pd.concat(val_rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        if val_rows
        else pd.DataFrame(columns=df.columns)
    )
    test_df = (
        pd.concat(test_rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        if test_rows
        else pd.DataFrame(columns=df.columns)
    )

    return train_df, val_df, test_df


def create_splits(
    manifest_path: Path,
    splits_dir: Path,
    seed: int = 42,
    force: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_path = splits_dir / "train_manifest.csv"
    val_path = splits_dir / "val_manifest.csv"
    test_path = splits_dir / "test_manifest.csv"

    if not force and train_path.exists() and val_path.exists() and test_path.exists():
        return (
            pd.read_csv(train_path),
            pd.read_csv(val_path),
            pd.read_csv(test_path),
        )

    df = load_manifest(manifest_path)
    counts = df["label_index"].value_counts()
    too_small = counts[counts < 3]
    if not too_small.empty:
        print(f"WARNING: {len(too_small)} classes have < 3 samples; they cannot appear in all splits.")
    train_df, val_df, test_df = split_manifest_stratified(df, seed=seed)

    train_labels = set(train_df["label_index"].unique())
    val_labels = set(val_df["label_index"].unique())
    test_labels = set(test_df["label_index"].unique())
    missing_val = sorted(train_labels - val_labels)
    missing_test = sorted(train_labels - test_labels)
    if missing_val:
        print(f"WARNING: {len(missing_val)} classes are missing from the val split.")
    if missing_test:
        print(f"WARNING: {len(missing_test)} classes are missing from the test split.")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_df, val_df, test_df


def _pad_or_crop(arr: np.ndarray, max_len: int) -> np.ndarray:
    t = arr.shape[0]
    if t == max_len:
        return arr
    if t > max_len:
        start = (t - max_len) // 2
        return arr[start : start + max_len]
    pad = np.zeros((max_len - t, arr.shape[1]), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


def _temporal_jitter(arr: np.ndarray, max_len: int, scale: float = 1.0) -> np.ndarray:
    max_shift = int(0.10 * max_len * scale)
    if max_shift < 1:
        return arr
    shift = random.randint(-max_shift, max_shift)
    if shift == 0:
        return arr
    out = np.zeros_like(arr)
    if shift > 0:
        out[shift:] = arr[:-shift]
    else:
        out[:shift] = arr[-shift:]
    return out


def _gaussian_noise(arr: np.ndarray, std: float = 0.01) -> np.ndarray:
    mask = np.any(arr != 0, axis=1)
    if not np.any(mask):
        return arr
    noise = np.random.normal(0.0, std, size=arr.shape).astype(arr.dtype)
    out = arr.copy()
    out[mask] = out[mask] + noise[mask]
    return out


def _frame_dropout(arr: np.ndarray, max_len: int, drop_range: Tuple[float, float]) -> np.ndarray:
    min_drop = max(1, int(drop_range[0] * max_len))
    max_drop = max(1, int(drop_range[1] * max_len))
    num_drop = random.randint(min_drop, max_drop)
    frame_indices = np.where(np.any(arr != 0, axis=1))[0]
    if len(frame_indices) == 0:
        return arr
    candidates = frame_indices if len(frame_indices) >= num_drop else np.arange(max_len)
    drop_idx = random.sample(list(candidates), k=min(num_drop, len(candidates)))
    out = arr.copy()
    out[drop_idx] = 0
    return out


def _horizontal_flip(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    left = out[:, 0:63].copy()
    right = out[:, 63:126].copy()
    out[:, 0:63] = right
    out[:, 63:126] = left
    return out


class ArSLDataset(Dataset):
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        root_dir: Path,
        max_len: int,
        augment: bool = False,
        class_counts: Optional[Dict[int, int]] = None,
    ) -> None:
        self.df = manifest_df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.max_len = max_len
        self.augment = augment
        self.class_counts = class_counts or {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rel_path = row.get("video_path", row.get("npy_path"))
        path = self.root_dir / rel_path
        if not path.exists():
            fallback = Path("keypoints_normalized") / rel_path
            path = fallback

        arr = np.load(path).astype(np.float32)
        if arr.shape[0] != self.max_len:
            arr = _pad_or_crop(arr, self.max_len)

        label = int(row["label_index"])
        if self.augment:
            count = int(self.class_counts.get(label, 0))
            prob_scale = 1.0
            jitter_scale = 1.0
            noise_std = 0.01
            drop_range = (0.05, 0.10)

            if count and count < 5:
                prob_scale = 0.35
                jitter_scale = 0.5
                noise_std = 0.005
                drop_range = (0.03, 0.06)
            elif count and count < 10:
                prob_scale = 0.5
                jitter_scale = 0.75
                noise_std = 0.008
                drop_range = (0.04, 0.08)

            if random.random() < 0.5 * prob_scale:
                arr = _temporal_jitter(arr, self.max_len, scale=jitter_scale)
            if random.random() < 0.5 * prob_scale:
                arr = _gaussian_noise(arr, std=noise_std)
            if random.random() < 0.5 * prob_scale:
                arr = _frame_dropout(arr, self.max_len, drop_range=drop_range)
            if random.random() < 0.5 * prob_scale:
                arr = _horizontal_flip(arr)

        x = torch.from_numpy(arr).float()
        y = torch.tensor(label, dtype=torch.long)

        mask = np.any(arr != 0, axis=1)
        length = int(mask.sum())
        if length <= 0:
            length = arr.shape[0]

        return x, y, torch.tensor(length, dtype=torch.long)


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_len: int,
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = True,
    class_counts: Optional[Dict[int, int]] = None,
    weighted_sampler: Optional[Sampler] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    root_dir = Path("keypoints_final") if Path("keypoints_final").exists() else Path("keypoints_normalized")

    train_ds = ArSLDataset(train_df, root_dir, max_len, augment=True, class_counts=class_counts)
    val_ds = ArSLDataset(val_df, root_dir, max_len, augment=False)
    test_ds = ArSLDataset(test_df, root_dir, max_len, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=weighted_sampler is None,
        sampler=weighted_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader
