import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam

from dataset import ArSLDataset, create_splits, load_manifest, set_seed
from fewshot_model import ProtoNet, STGCNEncoder

EPOCHS = 80
EPISODES_PER_EPOCH = 200
VAL_EPISODES = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42

N_WAY = 5
K_SHOT = 1
Q_QUERY = 1

MANIFEST_PATH = Path("./fewshot_4_19_manifest.csv")
LABEL_MAP_PATH = Path("./fewshot_4_19_label_map.json")
SPLITS_DIR = Path("./splits_fewshot_4_19")
CHECKPOINT_DIR = Path("./fewshot_checkpoints_4_19")
RESULTS_DIR = Path("./fewshot_results_4_19")


def _load_label_map() -> Dict[str, int]:
    return json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))


def _build_class_index(df: pd.DataFrame) -> Dict[int, List[int]]:
    class_to_indices: Dict[int, List[int]] = {}
    for idx, row in df.iterrows():
        label = int(row["label_index"])
        class_to_indices.setdefault(label, []).append(idx)
    return class_to_indices


def _eligible_classes(class_to_indices: Dict[int, List[int]], k_shot: int, q_query: int) -> List[int]:
    return [label for label, idxs in class_to_indices.items() if len(idxs) >= (k_shot + q_query)]


def _sample_episode(
    ds: ArSLDataset,
    class_to_indices: Dict[int, List[int]],
    n_way: int,
    k_shot: int,
    q_query: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    eligible = _eligible_classes(class_to_indices, k_shot, q_query)
    if len(eligible) < n_way:
        n_way = len(eligible)
    if n_way == 0:
        raise RuntimeError("No eligible classes for the requested episode configuration.")

    episode_classes = random.sample(eligible, n_way)

    support_x, support_len, query_x, query_len, query_y = [], [], [], [], []

    for episode_label, class_label in enumerate(episode_classes):
        indices = class_to_indices[class_label]
        picks = random.sample(indices, k_shot + q_query)
        support_idx = picks[:k_shot]
        query_idx = picks[k_shot:]

        for idx in support_idx:
            x, _, length = ds[idx]
            support_x.append(x)
            support_len.append(length)

        for idx in query_idx:
            x, _, length = ds[idx]
            query_x.append(x)
            query_len.append(length)
            query_y.append(episode_label)

    support_x = torch.stack(support_x, dim=0)
    support_len = torch.stack(support_len, dim=0)
    query_x = torch.stack(query_x, dim=0)
    query_len = torch.stack(query_len, dim=0)
    query_y = torch.tensor(query_y, dtype=torch.long)

    return support_x, support_len, query_x, query_len, query_y


def _run_episodes(
    model: ProtoNet,
    ds: ArSLDataset,
    class_to_indices: Dict[int, List[int]],
    n_way: int,
    k_shot: int,
    q_query: int,
    episodes: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
    criterion: nn.Module = None,
) -> Tuple[float, float]:
    total_loss = 0.0
    total_acc = 0.0

    for _ in range(episodes):
        support_x, support_len, query_x, query_len, query_y = _sample_episode(
            ds, class_to_indices, n_way, k_shot, q_query
        )
        support_x = support_x.to(device)
        support_len = support_len.to(device)
        query_x = query_x.to(device)
        query_len = query_len.to(device)
        query_y = query_y.to(device)

        episode_n_way = support_x.size(0) // k_shot
        logits = model(support_x, support_len, query_x, query_len, episode_n_way, k_shot)
        loss = criterion(logits, query_y)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        total_acc += (preds == query_y).float().mean().item()

    return total_loss / episodes, total_acc / episodes


def _save_training_curves(history: Dict[str, List[float]], best_epoch: int) -> None:
    import matplotlib.pyplot as plt

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"][1:], label="Val Loss")
    plt.axvline(best_epoch, color="gray", linestyle="--")
    plt.title("Few-shot Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"][1:], label="Val Acc")
    plt.axvline(best_epoch, color="gray", linestyle="--")
    plt.title("Few-shot Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_curves.png")
    plt.close()


def main() -> int:
    set_seed(RANDOM_SEED)

    if not MANIFEST_PATH.exists() or not LABEL_MAP_PATH.exists():
        raise FileNotFoundError("Missing few-shot manifest/label map. Run prepare_fewshot_4_19.py first.")

    df = load_manifest(MANIFEST_PATH)
    label_map = _load_label_map()
    num_classes = len(label_map)
    if num_classes < 2:
        raise RuntimeError("Need at least 2 classes to run few-shot training.")

    train_df, val_df, test_df = create_splits(MANIFEST_PATH, SPLITS_DIR, seed=RANDOM_SEED, force=True)
    max_len = int(df["final_frames"].max()) if "final_frames" in df.columns else int(df["original_frames"].max())

    print(f"Classes: {num_classes}")
    print(f"Total samples: {len(df)}")
    print(f"Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    print(f"MAX_LEN: {max_len}, feature dim: 225")
    print(f"Episode config: {N_WAY}-way {K_SHOT}-shot {Q_QUERY}-query")

    root_dir = Path("keypoints_final") if Path("keypoints_final").exists() else Path("keypoints_normalized")
    train_ds = ArSLDataset(train_df, root_dir, max_len, augment=True)
    val_ds = ArSLDataset(val_df, root_dir, max_len, augment=False)

    train_classes = _build_class_index(train_df)
    val_classes = _build_class_index(val_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = STGCNEncoder(num_nodes=75, in_channels=3, hidden=64, out_dim=128)
    model = ProtoNet(encoder).to(device)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [0.0], "val_acc": [0.0]}

    best_val = 0.0
    best_epoch = 0
    patience = 15
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_acc = _run_episodes(
            model,
            train_ds,
            train_classes,
            N_WAY,
            K_SHOT,
            Q_QUERY,
            EPISODES_PER_EPOCH,
            device,
            optimizer=optimizer,
            criterion=criterion,
        )

        model.eval()
        with torch.no_grad():
            val_loss, val_acc = _run_episodes(
                model,
                val_ds,
                val_classes,
                N_WAY,
                K_SHOT,
                Q_QUERY,
                VAL_EPISODES,
                device,
                optimizer=None,
                criterion=criterion,
            )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch}/{EPOCHS}]  Train Loss: {train_loss:.4f}  Train Acc: {train_acc*100:.2f}%"
        )
        print(
            f"                 Val   Loss: {val_loss:.4f}  Val   Acc: {val_acc*100:.2f}%"
        )

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
                "input_dim": 225,
                "n_way": N_WAY,
                "k_shot": K_SHOT,
                "q_query": Q_QUERY,
            }
            torch.save(ckpt, CHECKPOINT_DIR / "best_fewshot.pt")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    _save_training_curves(history, best_epoch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
