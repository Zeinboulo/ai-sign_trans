import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import WeightedRandomSampler

from dataset import build_dataloaders, create_splits, load_manifest, set_seed
from model import CNNBiLSTM, count_trainable_params

EPOCHS = 80
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42

MANIFEST_PATH = Path("./dataset_manifest.csv")
LABEL_MAP_PATH = Path("./label_map.json")
FILTERED_MANIFEST_PATH = Path("./filtered_dataset_manifest.csv")
FILTERED_LABEL_MAP_PATH = Path("./filtered_label_map.json")
SPLITS_DIR = Path("./splits")
CHECKPOINT_DIR = Path("./checkpoints")
RESULTS_DIR = Path("./results")


def _load_label_map(path: Path) -> Dict[str, int]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_paths() -> Tuple[Path, Path, Path, Path, Path, Path]:
    if FILTERED_MANIFEST_PATH.exists() and FILTERED_LABEL_MAP_PATH.exists():
        return (
            FILTERED_MANIFEST_PATH,
            FILTERED_LABEL_MAP_PATH,
            Path("./splits_filtered"),
            Path("./checkpoints_filtered"),
            Path("./results_filtered"),
            Path("./runs_filtered"),
        )
    return (
        MANIFEST_PATH,
        LABEL_MAP_PATH,
        SPLITS_DIR,
        CHECKPOINT_DIR,
        RESULTS_DIR,
        Path("./runs"),
    )


def _dataset_stats(df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    num_classes = df["label_index"].nunique()
    total = len(df)
    max_len = int(df["final_frames"].max()) if "final_frames" in df.columns else int(df["original_frames"].max())

    print(f"Classes (label_map): {num_classes}")
    print(f"Total samples: {total}")
    print(f"Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    print(f"MAX_LEN: {max_len}, feature dim: 225")

    counts = df["label_index"].value_counts()
    low_counts = counts[counts < 5]
    if not low_counts.empty:
        print(f"WARNING: {len(low_counts)} classes have < 5 samples.")


def _compute_class_weights(train_df: pd.DataFrame, num_classes: int) -> torch.Tensor:
    counts = train_df["label_index"].value_counts()
    total = counts.sum()
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for idx in range(num_classes):
        count = int(counts.get(idx, 0))
        if count > 0:
            weights[idx] = total / (num_classes * count)
        else:
            weights[idx] = 0.0
    return weights


def _train_once(batch_size: int) -> Tuple[Dict[str, List[float]], Dict, float, int]:
    set_seed(RANDOM_SEED)

    manifest_path, label_map_path, splits_dir, ckpt_dir, results_dir, runs_dir = _resolve_paths()
    df = load_manifest(manifest_path)
    if "final_frames" in df.columns and df["final_frames"].nunique() > 1:
        print("WARNING: final_frames has multiple values; using the maximum.")

    max_len = int(df["final_frames"].max()) if "final_frames" in df.columns else int(df["original_frames"].max())

    label_map = _load_label_map(label_map_path)
    num_classes = len(label_map)
    if num_classes < 2:
        raise RuntimeError("Need at least 2 classes to train a classifier.")

    train_df, val_df, test_df = create_splits(manifest_path, splits_dir, seed=RANDOM_SEED, force=True)
    _dataset_stats(df, train_df, val_df, test_df)

    class_counts = train_df["label_index"].value_counts().to_dict()
    sample_weights = [1.0 / class_counts[int(label)] for label in train_df["label_index"].tolist()]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader, val_loader, _ = build_dataloaders(
        train_df,
        val_df,
        test_df,
        max_len,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        class_counts=class_counts,
        weighted_sampler=sampler,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNBiLSTM(num_classes=num_classes, input_dim=225, use_projection=True, projection_dim=512)
    model.to(device)

    total_params = count_trainable_params(model)
    print(f"Total trainable parameters: {total_params:,}")

    counts = train_df["label_index"].value_counts()
    use_weights = (counts < 5).any()
    if use_weights:
        class_weights = _compute_class_weights(train_df, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=8,
        min_lr=1e-6,
    )

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(runs_dir))

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    early_stop_patience = 15

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y, lengths in train_loader:
            x = x.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y, lengths in val_loader:
                x = x.to(device)
                y = y.to(device)
                lengths = lengths.to(device)
                logits = model(x, lengths)
                loss = criterion(logits, y)

                val_loss_total += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_loss = val_loss_total / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        scheduler.step(val_acc)
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        writer.add_scalar("learning_rate", lr, epoch)

        print(
            f"Epoch [{epoch}/{EPOCHS}]  Train Loss: {train_loss:.4f}  Train Acc: {train_acc*100:.2f}%"
        )
        print(
            f"                 Val   Loss: {val_loss:.4f}  Val   Acc: {val_acc*100:.2f}%"
        )
        print(f"                 LR: {lr:.6f}")

        if epoch == 20 and val_acc < 0.5:
            print("WARNING: Model may not be learning. Check data loading and normalization.")

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "label_map": label_map,
                "max_len": max_len,
                "input_dim": 225,
                "use_projection": True,
                "projection_dim": 512,
            }
            torch.save(ckpt, ckpt_dir / "best_model.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    writer.close()
    return history, {"best_val": best_val, "best_epoch": best_epoch, "max_len": max_len, "results_dir": results_dir}, best_val, best_epoch


def main() -> int:
    batch_size = BATCH_SIZE
    while batch_size >= 1:
        try:
            history, meta, _, _ = _train_once(batch_size)
            _save_training_curves(history, meta["best_epoch"], meta["results_dir"])
            return 0
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and batch_size > 1:
                print("OOM detected. Reducing batch size and retrying.")
                batch_size = max(1, batch_size // 2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise

    return 1


def _save_training_curves(history: Dict[str, List[float]], best_epoch: int, results_dir: Path) -> None:
    import matplotlib.pyplot as plt

    results_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.axvline(best_epoch, color="gray", linestyle="--")
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.axvline(best_epoch, color="gray", linestyle="--")
    plt.title("Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(results_dir / "training_curves.png")
    plt.close()


if __name__ == "__main__":
    raise SystemExit(main())
