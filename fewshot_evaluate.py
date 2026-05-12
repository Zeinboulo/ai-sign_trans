import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from dataset import ArSLDataset, create_splits, load_manifest, set_seed
from fewshot_model import ProtoNet, STGCNEncoder

MANIFEST_PATH = Path("./fewshot_4_19_manifest.csv")
LABEL_MAP_PATH = Path("./fewshot_4_19_label_map.json")
SPLITS_DIR = Path("./splits_fewshot_4_19")
CHECKPOINT_PATH = Path("./fewshot_checkpoints_4_19/best_fewshot.pt")
RESULTS_DIR = Path("./fewshot_results_4_19")

TEST_EPISODES = 200


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


def main() -> int:
    set_seed(42)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT_PATH}")

    df = load_manifest(MANIFEST_PATH)
    _ = _load_label_map()

    train_df, val_df, test_df = create_splits(MANIFEST_PATH, SPLITS_DIR, seed=42, force=False)
    max_len = int(df["final_frames"].max()) if "final_frames" in df.columns else int(df["original_frames"].max())

    root_dir = Path("keypoints_final") if Path("keypoints_final").exists() else Path("keypoints_normalized")
    test_ds = ArSLDataset(test_df, root_dir, max_len, augment=False)
    test_classes = _build_class_index(test_df)

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    n_way = int(ckpt.get("n_way", 5))
    k_shot = int(ckpt.get("k_shot", 1))
    q_query = int(ckpt.get("q_query", 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = STGCNEncoder(num_nodes=75, in_channels=3, hidden=64, out_dim=128)
    model = ProtoNet(encoder).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    accuracies: List[float] = []
    criterion = nn.CrossEntropyLoss()
    losses: List[float] = []

    with torch.no_grad():
        for _ in range(TEST_EPISODES):
            support_x, support_len, query_x, query_len, query_y = _sample_episode(
                test_ds, test_classes, n_way, k_shot, q_query
            )
            support_x = support_x.to(device)
            support_len = support_len.to(device)
            query_x = query_x.to(device)
            query_len = query_len.to(device)
            query_y = query_y.to(device)

            episode_n_way = support_x.size(0) // k_shot
            logits = model(support_x, support_len, query_x, query_len, episode_n_way, k_shot)
            loss = criterion(logits, query_y)
            losses.append(loss.item())

            preds = logits.argmax(dim=1)
            acc = (preds == query_y).float().mean().item()
            accuracies.append(acc)

    mean_acc = float(np.mean(accuracies))
    std_acc = float(np.std(accuracies))
    mean_loss = float(np.mean(losses))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "fewshot_eval_summary.json"
    report_path.write_text(
        json.dumps(
            {
                "test_episodes": TEST_EPISODES,
                "n_way": n_way,
                "k_shot": k_shot,
                "q_query": q_query,
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "mean_loss": mean_loss,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Few-shot test accuracy: {mean_acc*100:.2f}% (std {std_acc*100:.2f}%)")
    print(f"Few-shot test loss: {mean_loss:.4f}")
    print(f"Report saved to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
