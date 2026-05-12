import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from dataset import build_dataloaders, create_splits, load_manifest, set_seed
from hybrid_model import HybridGraphTransformer

EPOCHS = 120
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42

CONTRASTIVE_WEIGHT = 0.2
CONTRASTIVE_TEMPERATURE = 0.1
LABEL_SMOOTHING = 0.1

MANIFEST_PATH = Path("./dataset_manifest.csv")
LABEL_MAP_PATH = Path("./label_map.json")
FILTERED_MANIFEST_PATH = Path("./filtered_dataset_manifest.csv")
FILTERED_LABEL_MAP_PATH = Path("./filtered_label_map.json")
SPLITS_DIR = Path("./splits")
CHECKPOINT_DIR = Path("./hybrid_checkpoints")
RESULTS_DIR = Path("./hybrid_results")
RUNS_DIR = Path("./runs_hybrid")
USE_FILTERED = True


def _load_label_map(path: Path) -> Dict[str, int]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_paths() -> Tuple[Path, Path, Path]:
    if USE_FILTERED and FILTERED_MANIFEST_PATH.exists() and FILTERED_LABEL_MAP_PATH.exists():
        return FILTERED_MANIFEST_PATH, FILTERED_LABEL_MAP_PATH, Path("./splits_filtered")
    return MANIFEST_PATH, LABEL_MAP_PATH, SPLITS_DIR


def _dataset_stats(df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> int:
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

    return max_len


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


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    if features.size(0) < 2:
        return torch.tensor(0.0, device=features.device)

    feats = F.normalize(features, dim=1)
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(features.device)

    logits = torch.matmul(feats, feats.T) / temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=mask.device)
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    positives = mask.sum(dim=1)
    valid = positives > 0
    if not valid.any():
        return torch.tensor(0.0, device=features.device)

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (positives + 1e-12)
    return -mean_log_prob_pos[valid].mean()


def _train_once(batch_size: int) -> Tuple[Dict[str, List[float]], Dict, float, int]:
    set_seed(RANDOM_SEED)

    manifest_path, label_map_path, splits_dir = _resolve_paths()
    df = load_manifest(manifest_path)
    if "final_frames" in df.columns and df["final_frames"].nunique() > 1:
        print("WARNING: final_frames has multiple values; using the maximum.")

    label_map = _load_label_map(label_map_path)
    num_classes = len(label_map)
    if num_classes < 2:
        raise RuntimeError("Need at least 2 classes to train a classifier.")

    train_df, val_df, test_df = create_splits(manifest_path, splits_dir, seed=RANDOM_SEED, force=True)
    max_len = _dataset_stats(df, train_df, val_df, test_df)

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
    model = HybridGraphTransformer(
        num_classes=num_classes,
        max_len=max_len,
        num_nodes=75,
        in_channels=3,
        gcn_hidden=64,
        gcn_out=128,
        tf_layers=4,
        tf_heads=8,
        tf_ff=256,
        dropout=0.2,
        proj_dim=256,
    )
    model.to(device)

    counts = train_df["label_index"].value_counts()
    use_weights = (counts < 5).any()
    if use_weights:
        class_weights = _compute_class_weights(train_df, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(RUNS_DIR))

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
    early_stop_patience = 20

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

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
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, proj = model(x, lengths, return_embedding=True)
                ce_loss = criterion(logits, y)
                con_loss = supervised_contrastive_loss(proj, y, temperature=CONTRASTIVE_TEMPERATURE)
                loss = ce_loss + CONTRASTIVE_WEIGHT * con_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

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
                logits, proj = model(x, lengths, return_embedding=True)
                ce_loss = criterion(logits, y)
                con_loss = supervised_contrastive_loss(proj, y, temperature=CONTRASTIVE_TEMPERATURE)
                loss = ce_loss + CONTRASTIVE_WEIGHT * con_loss

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
                "model_config": {
                    "num_classes": num_classes,
                    "max_len": max_len,
                    "num_nodes": 75,
                    "in_channels": 3,
                    "gcn_hidden": 64,
                    "gcn_out": 128,
                    "tf_layers": 4,
                    "tf_heads": 8,
                    "tf_ff": 256,
                    "dropout": 0.2,
                    "proj_dim": 256,
                },
            }
            torch.save(ckpt, CHECKPOINT_DIR / "best_model.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    writer.close()
    return history, {"best_val": best_val, "best_epoch": best_epoch, "max_len": max_len}, best_val, best_epoch


def main() -> int:
    start = time.time()
    history, summary, best_val, best_epoch = _train_once(BATCH_SIZE)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "train_summary.json").write_text(
        json.dumps(
            {
                "best_val": best_val,
                "best_epoch": best_epoch,
                "max_len": summary["max_len"],
                "epochs": len(history["train_loss"]),
                "elapsed_minutes": (time.time() - start) / 60.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Best val acc: {best_val*100:.2f}% at epoch {best_epoch}")
    print(f"Training summary saved to {RESULTS_DIR / 'train_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
