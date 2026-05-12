import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW

from dataset import ArSLDataset, create_splits, load_manifest, set_seed
from hybrid_fewshot_model import HybridEncoder, ProtoNet

EPOCHS = 100
EPISODES_PER_EPOCH = 300
VAL_EPISODES = 100
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 2e-4
LABEL_SMOOTHING = 0.1
GRAD_CLIP = 1.0
RANDOM_SEED = 42

N_WAY_START = 5
N_WAY_FINAL = 5
N_WAY_SWITCH_EPOCH = 0
LR_SWITCH_FACTOR = 1.0
K_SHOT = 1
Q_QUERY = 1

DATASET = "filtered_4plus"

if DATASET == "fewshot_4_19":
    MANIFEST_PATH = Path("./fewshot_4_19_manifest.csv")
    LABEL_MAP_PATH = Path("./fewshot_4_19_label_map.json")
    SPLITS_DIR = Path("./splits_fewshot_4_19")
    CHECKPOINT_DIR = Path("./hybrid_fewshot_checkpoints_4_19")
    RESULTS_DIR = Path("./hybrid_fewshot_results_4_19")
elif DATASET == "filtered_4plus":
    MANIFEST_PATH = Path("./filtered_4plus_manifest.csv")
    LABEL_MAP_PATH = Path("./filtered_4plus_label_map.json")
    SPLITS_DIR = Path("./splits_filtered_4plus")
    CHECKPOINT_DIR = Path("./hybrid_fewshot_checkpoints_4plus")
    RESULTS_DIR = Path("./hybrid_fewshot_results_4plus")
elif DATASET == "filtered_ge5":
    MANIFEST_PATH = Path("./filtered_dataset_manifest.csv")
    LABEL_MAP_PATH = Path("./filtered_label_map.json")
    SPLITS_DIR = Path("./splits_filtered")
    CHECKPOINT_DIR = Path("./hybrid_fewshot_checkpoints_filtered")
    RESULTS_DIR = Path("./hybrid_fewshot_results_filtered")
else:
    raise ValueError(f"Unknown DATASET setting: {DATASET}")


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
    scaler: torch.amp.GradScaler = None,
    use_amp: bool = False,
    amp_device: str = "cuda",
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

        if optimizer is None:
            logits = model(support_x, support_len, query_x, query_len, episode_n_way, k_shot)
            loss = criterion(logits, query_y)
        else:
            optimizer.zero_grad()
            with torch.amp.autocast(amp_device, enabled=use_amp):
                logits = model(support_x, support_len, query_x, query_len, episode_n_way, k_shot)
                loss = criterion(logits, query_y)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
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
    plt.title("Hybrid Few-shot Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"][1:], label="Val Acc")
    plt.axvline(best_epoch, color="gray", linestyle="--")
    plt.title("Hybrid Few-shot Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_curves.png")
    plt.close()


def main() -> int:
    set_seed(RANDOM_SEED)

    if not MANIFEST_PATH.exists() or not LABEL_MAP_PATH.exists():
        raise FileNotFoundError("Missing few-shot manifest/label map.")

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
    if N_WAY_START == N_WAY_FINAL:
        print(
            "Episode config: "
            f"{N_WAY_START}-way, {K_SHOT}-shot {Q_QUERY}-query"
        )
    else:
        print(
            "Episode config: "
            f"{N_WAY_START}-way to {N_WAY_FINAL}-way (switch @ epoch {N_WAY_SWITCH_EPOCH}), "
            f"{K_SHOT}-shot {Q_QUERY}-query"
        )

    root_dir = Path("keypoints_final") if Path("keypoints_final").exists() else Path("keypoints_normalized")
    train_ds = ArSLDataset(train_df, root_dir, max_len, augment=True)
    val_ds = ArSLDataset(val_df, root_dir, max_len, augment=False)

    train_classes = _build_class_index(train_df)
    val_classes = _build_class_index(val_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = HybridEncoder(
        max_len=max_len,
        num_nodes=75,
        in_channels=3,
        gcn_hidden=64,
        gcn_out=96,
        tf_layers=3,
        tf_heads=4,
        tf_ff=192,
        dropout=0.35,
        proj_dim=128,
    )
    model = ProtoNet(encoder).to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [0.0], "val_acc": [0.0]}

    best_val = 0.0
    best_epoch = 0
    patience = 20
    no_improve = 0

    amp_device = "cuda" if device.type == "cuda" else "cpu"
    use_amp = amp_device == "cuda"
    scaler = torch.amp.GradScaler(amp_device, enabled=use_amp)

    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        n_way = N_WAY_START if epoch <= N_WAY_SWITCH_EPOCH else N_WAY_FINAL
        if N_WAY_START != N_WAY_FINAL and epoch == N_WAY_SWITCH_EPOCH + 1:
            for group in optimizer.param_groups:
                group["lr"] = group["lr"] * LR_SWITCH_FACTOR
            print(f"LR reduced by {LR_SWITCH_FACTOR} at epoch {epoch}")

        model.train()
        train_loss, train_acc = _run_episodes(
            model,
            train_ds,
            train_classes,
            n_way,
            K_SHOT,
            Q_QUERY,
            EPISODES_PER_EPOCH,
            device,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            use_amp=use_amp,
            amp_device=amp_device,
        )

        model.eval()
        with torch.no_grad():
            val_loss, val_acc = _run_episodes(
                model,
                val_ds,
                val_classes,
                n_way,
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
        print(f"                 Episode way: {n_way}")

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
                "n_way": n_way,
                "k_shot": K_SHOT,
                "q_query": Q_QUERY,
                "model_config": {
                    "max_len": max_len,
                    "num_nodes": 75,
                    "in_channels": 3,
                    "gcn_hidden": 64,
                    "gcn_out": 96,
                    "tf_layers": 3,
                    "tf_heads": 4,
                    "tf_ff": 192,
                    "dropout": 0.35,
                    "proj_dim": 128,
                },
            }
            torch.save(ckpt, CHECKPOINT_DIR / "best_fewshot.pt")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    _save_training_curves(history, best_epoch)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "train_summary.json").write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "max_len": max_len,
                "epochs": len(history["train_loss"]),
                "episodes_per_epoch": EPISODES_PER_EPOCH,
                "val_episodes": VAL_EPISODES,
                "n_way_start": N_WAY_START,
                "n_way_final": N_WAY_FINAL,
                "n_way_switch_epoch": N_WAY_SWITCH_EPOCH,
                "k_shot": K_SHOT,
                "q_query": Q_QUERY,
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
