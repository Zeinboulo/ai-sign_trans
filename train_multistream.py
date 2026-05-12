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
from torch.utils.data import WeightedRandomSampler

from dataset import build_dataloaders, create_splits, load_manifest, set_seed
from multistream_model import MultiStreamClassifier

EPOCHS = 80
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 2e-4
RANDOM_SEED = 42

LABEL_SMOOTHING = 0.1
CONTRASTIVE_WEIGHT = 0.2
CONTRASTIVE_TEMP = 0.1
GRAD_CLIP = 1.0
USE_COSINE_CLASSIFIER = True
COSINE_SCALE = 30.0

MANIFEST_PATH = Path("./filtered_dataset_manifest.csv")
LABEL_MAP_PATH = Path("./filtered_label_map.json")
SPLITS_DIR = Path("./splits_multistream")
CHECKPOINT_DIR = Path("./multistream_checkpoints")
RESULTS_DIR = Path("./multistream_results")
PRETRAIN_PATH = Path("./pretrain_checkpoints/best_pretrain.pt")


def _load_label_map(path: Path) -> Dict[str, int]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
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


def main() -> int:
    set_seed(RANDOM_SEED)

    if not MANIFEST_PATH.exists() or not LABEL_MAP_PATH.exists():
        raise FileNotFoundError("Missing filtered manifest/label map. Run prepare_filtered_dataset.py first.")

    df = load_manifest(MANIFEST_PATH)
    max_len = int(df["final_frames"].max()) if "final_frames" in df.columns else int(df["original_frames"].max())

    label_map = _load_label_map(LABEL_MAP_PATH)
    num_classes = len(label_map)
    if num_classes < 2:
        raise RuntimeError("Need at least 2 classes to train a classifier.")

    train_df, val_df, test_df = create_splits(MANIFEST_PATH, SPLITS_DIR, seed=RANDOM_SEED, force=True)

    class_counts = train_df["label_index"].value_counts().to_dict()
    sample_weights = [1.0 / class_counts[int(label)] for label in train_df["label_index"].tolist()]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader, val_loader, _ = build_dataloaders(
        train_df,
        val_df,
        test_df,
        max_len,
        batch_size=BATCH_SIZE,
        num_workers=2,
        pin_memory=True,
        class_counts=class_counts,
        weighted_sampler=sampler,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiStreamClassifier(
        num_classes=num_classes,
        max_len=max_len,
        gcn_hidden=64,
        gcn_out=96,
        d_model=192,
        conformer_layers=4,
        conformer_heads=4,
        conformer_ff=256,
        dropout=0.2,
        proj_dim=128,
        use_cosine=USE_COSINE_CLASSIFIER,
        cosine_scale=COSINE_SCALE,
    ).to(device)

    if PRETRAIN_PATH.exists():
        ckpt = torch.load(PRETRAIN_PATH, map_location="cpu")
        state_key = "encoder_state" if "encoder_state" in ckpt else "model_state"
        state = {k: v for k, v in ckpt[state_key].items() if not k.startswith("recon_head")}
        missing, unexpected = model.encoder.load_state_dict(state, strict=False)
        if missing:
            print(f"WARNING: Missing pretrain keys: {len(missing)}")
        if unexpected:
            print(f"WARNING: Unexpected pretrain keys: {len(unexpected)}")
    else:
        print("WARNING: Pretrain checkpoint not found; training from scratch.")

    counts = train_df["label_index"].value_counts()
    if (counts < 5).any():
        class_weights = _compute_class_weights(train_df, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    best_val = 0.0
    best_epoch = 0
    no_improve = 0
    patience = 15

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start = time.time()

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
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, proj = model(x, lengths, return_embedding=True)
                ce_loss = criterion(logits, y)
                con_loss = supervised_contrastive_loss(proj, y, CONTRASTIVE_TEMP)
                loss = ce_loss + CONTRASTIVE_WEIGHT * con_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
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
                con_loss = supervised_contrastive_loss(proj, y, CONTRASTIVE_TEMP)
                loss = ce_loss + CONTRASTIVE_WEIGHT * con_loss

                val_loss_total += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_loss = val_loss_total / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        print(f"Epoch [{epoch}/{EPOCHS}]  Train Loss: {train_loss:.4f}  Train Acc: {train_acc*100:.2f}%")
        print(f"                 Val   Loss: {val_loss:.4f}  Val   Acc: {val_acc*100:.2f}%")

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            no_improve = 0
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "label_map": label_map,
                "max_len": max_len,
                "model_config": {
                    "gcn_hidden": 64,
                    "gcn_out": 96,
                    "d_model": 192,
                    "conformer_layers": 4,
                    "conformer_heads": 4,
                    "conformer_ff": 256,
                    "dropout": 0.2,
                    "proj_dim": 128,
                    "use_cosine": USE_COSINE_CLASSIFIER,
                    "cosine_scale": COSINE_SCALE,
                },
            }
            torch.save(ckpt, CHECKPOINT_DIR / "best_model.pt")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    (RESULTS_DIR / "train_summary.json").write_text(
        json.dumps(
            {
                "best_val": best_val,
                "best_epoch": best_epoch,
                "max_len": max_len,
                "epochs": epoch,
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
