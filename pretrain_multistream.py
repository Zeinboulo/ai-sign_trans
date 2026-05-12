import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset import ArSLDataset, create_splits, load_manifest, set_seed
from multistream_model import MultiStreamEncoder

PRETRAIN_EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42
PROJ_DIM = 128
CONTRASTIVE_TEMP = 0.2
AUG_PROB = 0.5
AUG_NOISE_STD = 0.01
AUG_DROP_RANGE = (0.05, 0.10)

MANIFEST_PATH = Path("./filtered_dataset_manifest.csv")
LABEL_MAP_PATH = Path("./filtered_label_map.json")
SPLITS_DIR = Path("./splits_multistream_pretrain")
CHECKPOINT_DIR = Path("./pretrain_checkpoints")
RESULTS_DIR = Path("./pretrain_results")


def _temporal_jitter(x: torch.Tensor, length: int, max_len: int) -> torch.Tensor:
    max_shift = int(0.10 * max_len)
    if max_shift < 1:
        return x
    shift = random.randint(-max_shift, max_shift)
    if length > 1:
        shift = max(-(length - 1), min(shift, length - 1))
    if shift == 0:
        return x
    out = torch.zeros_like(x)
    if shift > 0:
        out[shift:length] = x[: length - shift]
    else:
        out[: length + shift] = x[-shift:length]
    return out


def _frame_dropout(x: torch.Tensor, length: int, max_len: int, drop_range: Tuple[float, float]) -> torch.Tensor:
    min_drop = max(1, int(drop_range[0] * max_len))
    max_drop = max(1, int(drop_range[1] * max_len))
    num_drop = random.randint(min_drop, max_drop)
    valid = torch.arange(max_len, device=x.device) < length
    valid_idx = torch.where(valid)[0]
    if valid_idx.numel() == 0:
        return x
    if valid_idx.numel() < num_drop:
        drop_idx = torch.randperm(max_len, device=x.device)[:num_drop]
    else:
        drop_idx = valid_idx[torch.randperm(valid_idx.numel(), device=x.device)[:num_drop]]
    out = x.clone()
    out[drop_idx] = 0.0
    return out


def _horizontal_flip(x: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    left = out[:, 0:63].clone()
    right = out[:, 63:126].clone()
    out[:, 0:63] = right
    out[:, 63:126] = left
    return out


def _augment_tensor(x: torch.Tensor, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    b, t, f = x.shape
    out = x.clone()
    for i in range(b):
        length = int(lengths[i].item())
        length = max(1, min(length, t))
        sample = out[i]

        if random.random() < AUG_PROB:
            sample = _temporal_jitter(sample, length, max_len)
        if random.random() < AUG_PROB:
            noise = torch.randn_like(sample) * AUG_NOISE_STD
            sample = sample + noise
        if random.random() < AUG_PROB:
            sample = _frame_dropout(sample, length, max_len, AUG_DROP_RANGE)
        if random.random() < AUG_PROB:
            sample = _horizontal_flip(sample)

        out[i] = sample
    return out


def _nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    n = z1.size(0)
    if n == 0:
        return torch.tensor(0.0, device=z1.device)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    mask = torch.eye(2 * n, device=z.device, dtype=torch.bool)
    neg_inf = torch.finfo(sim.dtype).min
    sim = sim.masked_fill(mask, neg_inf)
    pos = torch.arange(n, device=z.device)
    pos = torch.cat([pos + n, pos])
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -log_prob[torch.arange(2 * n, device=z.device), pos].mean()


def _run_epoch(
    encoder: MultiStreamEncoder,
    projector: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_len: int,
    optimizer: torch.optim.Optimizer = None,
    scaler: torch.amp.GradScaler = None,
) -> float:
    total_loss = 0.0
    total_count = 0
    use_amp = device.type == "cuda"

    for x, _, lengths in loader:
        x = x.to(device)
        lengths = lengths.to(device)

        view1 = _augment_tensor(x, lengths, max_len)
        view2 = _augment_tensor(x, lengths, max_len)

        if optimizer is not None:
            optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            z1 = projector(encoder(view1, lengths))
            z2 = projector(encoder(view2, lengths))
            loss = _nt_xent_loss(z1, z2, CONTRASTIVE_TEMP)

        if optimizer is not None:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)

    return total_loss / max(total_count, 1)


def main() -> int:
    set_seed(RANDOM_SEED)

    if not MANIFEST_PATH.exists() or not LABEL_MAP_PATH.exists():
        raise FileNotFoundError("Missing filtered manifest/label map. Run prepare_filtered_dataset.py first.")

    df = load_manifest(MANIFEST_PATH)
    max_len = int(df["final_frames"].max()) if "final_frames" in df.columns else int(df["original_frames"].max())

    train_df, val_df, _ = create_splits(MANIFEST_PATH, SPLITS_DIR, seed=RANDOM_SEED, force=True)
    root_dir = Path("keypoints_final") if Path("keypoints_final").exists() else Path("keypoints_normalized")

    train_ds = ArSLDataset(train_df, root_dir, max_len, augment=False)
    val_ds = ArSLDataset(val_df, root_dir, max_len, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = MultiStreamEncoder(
        max_len=max_len,
        gcn_hidden=64,
        gcn_out=96,
        d_model=192,
        conformer_layers=4,
        conformer_heads=4,
        conformer_ff=256,
        dropout=0.2,
    ).to(device)

    projector = nn.Sequential(
        nn.Linear(192, PROJ_DIM),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(PROJ_DIM, PROJ_DIM),
    ).to(device)

    optimizer = AdamW(list(encoder.parameters()) + list(projector.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_epoch = 0
    patience = 8
    no_improve = 0

    start = time.time()

    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        encoder.train()
        projector.train()
        train_loss = _run_epoch(
            encoder,
            projector,
            train_loader,
            device,
            max_len,
            optimizer=optimizer,
            scaler=scaler,
        )

        encoder.eval()
        projector.eval()
        with torch.no_grad():
            val_loss = _run_epoch(
                encoder,
                projector,
                val_loader,
                device,
                max_len,
                optimizer=None,
                scaler=None,
            )

        print(f"Epoch [{epoch}/{PRETRAIN_EPOCHS}]  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            ckpt = {
                "epoch": epoch,
                "encoder_state": encoder.state_dict(),
                "projector_state": projector.state_dict(),
                "max_len": max_len,
                "config": {
                    "gcn_hidden": 64,
                    "gcn_out": 96,
                    "d_model": 192,
                    "conformer_layers": 4,
                    "conformer_heads": 4,
                    "conformer_ff": 256,
                    "dropout": 0.2,
                    "proj_dim": PROJ_DIM,
                    "temperature": CONTRASTIVE_TEMP,
                },
            }
            torch.save(ckpt, CHECKPOINT_DIR / "best_pretrain.pt")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    (RESULTS_DIR / "pretrain_summary.json").write_text(
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

    print(f"Best val loss: {best_val:.4f} at epoch {best_epoch}")
    print(f"Pretrain summary saved to {RESULTS_DIR / 'pretrain_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
