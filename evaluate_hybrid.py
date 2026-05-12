import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from dataset import ArSLDataset, load_manifest
from hybrid_model import HybridGraphTransformer

MANIFEST_PATH = Path("./dataset_manifest.csv")
LABEL_MAP_PATH = Path("./label_map.json")
FILTERED_MANIFEST_PATH = Path("./filtered_dataset_manifest.csv")
FILTERED_LABEL_MAP_PATH = Path("./filtered_label_map.json")
SPLITS_DIR = Path("./splits")
CHECKPOINT_PATH = Path("./hybrid_checkpoints/best_model.pt")
RESULTS_DIR = Path("./hybrid_results")
MODEL_EXPORT_DIR = Path("./hybrid_model_export")
USE_FILTERED = True


def _load_label_map(path: Path) -> Dict[str, int]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_paths() -> Tuple[Path, Path, Path, Path, Path, Path]:
    if USE_FILTERED and FILTERED_MANIFEST_PATH.exists() and FILTERED_LABEL_MAP_PATH.exists():
        return (
            FILTERED_MANIFEST_PATH,
            FILTERED_LABEL_MAP_PATH,
            Path("./splits_filtered"),
            CHECKPOINT_PATH,
            RESULTS_DIR,
            MODEL_EXPORT_DIR,
        )
    return (
        MANIFEST_PATH,
        LABEL_MAP_PATH,
        SPLITS_DIR,
        CHECKPOINT_PATH,
        RESULTS_DIR,
        MODEL_EXPORT_DIR,
    )


def _load_checkpoint(path: Path, device: torch.device) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return torch.load(path, map_location=device)


def _prepare_model(ckpt: Dict, label_map: Dict[str, int], device: torch.device) -> HybridGraphTransformer:
    label_map = ckpt.get("label_map", label_map)
    cfg = ckpt.get("model_config", {})
    model = HybridGraphTransformer(
        num_classes=len(label_map),
        max_len=int(cfg.get("max_len", ckpt.get("max_len", 1))),
        num_nodes=int(cfg.get("num_nodes", 75)),
        in_channels=int(cfg.get("in_channels", 3)),
        gcn_hidden=int(cfg.get("gcn_hidden", 64)),
        gcn_out=int(cfg.get("gcn_out", 128)),
        tf_layers=int(cfg.get("tf_layers", 4)),
        tf_heads=int(cfg.get("tf_heads", 8)),
        tf_ff=int(cfg.get("tf_ff", 256)),
        dropout=float(cfg.get("dropout", 0.2)),
        proj_dim=int(cfg.get("proj_dim", 256)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def _get_max_len(df: pd.DataFrame, ckpt: Dict) -> int:
    if "max_len" in ckpt:
        return int(ckpt["max_len"])
    if "final_frames" in df.columns:
        return int(df["final_frames"].max())
    return int(df["original_frames"].max())


def _save_label_index_map(label_map: Dict[str, int], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    inv = {v: k for k, v in label_map.items()}
    lines = [f"{idx}: {word}" for idx, word in sorted(inv.items())]
    (results_dir / "label_index_map.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path, label_map_path, splits_dir, ckpt_path, results_dir, export_dir = _resolve_paths()
    df = load_manifest(manifest_path)

    test_manifest = pd.read_csv(splits_dir / "test_manifest.csv")
    if test_manifest.empty:
        raise RuntimeError("Test split is empty. Cannot evaluate.")

    ckpt = _load_checkpoint(ckpt_path, device)
    label_map = ckpt.get("label_map", _load_label_map(label_map_path))
    inv_label_map = {v: k for k, v in label_map.items()}

    max_len = _get_max_len(df, ckpt)
    root_dir = Path("keypoints_final") if Path("keypoints_final").exists() else Path("keypoints_normalized")
    test_ds = ArSLDataset(test_manifest, root_dir, max_len, augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = _prepare_model(ckpt, label_map, device)

    all_logits: List[np.ndarray] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for x, y, lengths in test_loader:
            x = x.to(device)
            lengths = lengths.to(device)
            logits, _ = model(x, lengths, return_embedding=False)
            all_logits.append(logits.cpu().numpy())
            all_labels.extend(y.numpy().tolist())

    logits = np.concatenate(all_logits, axis=0)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    y_true = np.array(all_labels, dtype=int)
    y_pred = probs.argmax(axis=1)

    accuracy = float((y_pred == y_true).mean())
    top3 = np.argsort(probs, axis=1)[:, -3:]
    top3_acc = float(np.mean([y_true[i] in top3[i] for i in range(len(y_true))]))

    present_labels = sorted(set(y_true.tolist()))
    missing_labels = [idx for idx in range(len(label_map)) if idx not in present_labels]
    if missing_labels:
        print(f"WARNING: {len(missing_labels)} classes have 0 test samples and will be skipped in per-class metrics.")

    target_names = [inv_label_map[idx] for idx in present_labels]
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=present_labels,
            target_names=target_names,
            zero_division=0,
        )
    )

    macro_f1 = f1_score(y_true, y_pred, labels=present_labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=present_labels, average="weighted", zero_division=0)

    try:
        y_true_bin = label_binarize(y_true, classes=present_labels)
        y_prob_present = probs[:, present_labels]
        roc_auc = roc_auc_score(y_true_bin, y_prob_present, average="macro", multi_class="ovr")
        print(f"ROC-AUC (macro OvR): {roc_auc:.4f}")
    except Exception as exc:
        roc_auc = float("nan")
        print(f"ROC-AUC (macro OvR) could not be computed: {exc}")

    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"MCC: {mcc:.4f}")

    results_dir.mkdir(parents=True, exist_ok=True)
    _save_label_index_map(label_map, results_dir)

    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    fig_size = (20, 20) if len(present_labels) > 20 else (12, 12)
    plt.figure(figsize=fig_size)

    annot = len(present_labels) <= 50
    if not annot:
        print("WARNING: Confusion matrix is large; cell annotations are disabled to avoid rendering issues.")

    ax = sns.heatmap(
        cm,
        cmap="Blues",
        annot=annot,
        fmt="d",
        xticklabels=[str(idx) for idx in present_labels] if len(present_labels) > 20 else target_names,
        yticklabels=[str(idx) for idx in present_labels] if len(present_labels) > 20 else target_names,
        cbar=True,
    )
    if len(present_labels) > 20:
        ax.tick_params(axis="both", labelsize=7)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png")
    plt.close()

    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    flat_indices = np.dstack(np.unravel_index(np.argsort(cm_no_diag.ravel())[::-1], cm_no_diag.shape))[0]
    print("Most confused pairs:")
    shown = 0
    for i, j in flat_indices:
        if cm_no_diag[i, j] == 0:
            break
        true_label = present_labels[i]
        pred_label = present_labels[j]
        print(f"{inv_label_map[true_label]} -> predicted as {inv_label_map[pred_label]} : {cm_no_diag[i, j]} times")
        shown += 1
        if shown >= 10:
            break

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=present_labels,
        zero_division=0,
    )
    f1_pairs = list(zip(present_labels, prec, rec, f1))
    f1_pairs.sort(key=lambda x: x[3])
    print("Bottom-5 classes by F1:")
    for label_idx, p, r, f in f1_pairs[:5]:
        print(f"{inv_label_map[label_idx]} -> precision={p:.3f}, recall={r:.3f}, f1={f:.3f}")

    conf = probs.max(axis=1)
    correct_mask = y_pred == y_true
    mean_conf_correct = float(conf[correct_mask].mean()) if np.any(correct_mask) else 0.0
    mean_conf_incorrect = float(conf[~correct_mask].mean()) if np.any(~correct_mask) else 0.0
    print(f"Mean confidence on correct predictions  : {mean_conf_correct*100:.2f}%")
    print(f"Mean confidence on incorrect predictions: {mean_conf_incorrect*100:.2f}%")

    plt.figure(figsize=(8, 4))
    plt.hist(conf[correct_mask], bins=20, alpha=0.7, color="green", label="Correct")
    plt.hist(conf[~correct_mask], bins=20, alpha=0.7, color="red", label="Incorrect")
    plt.title("Confidence Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "confidence_dist.png")
    plt.close()

    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / "arsl_hybrid_graph_transformer.pt"
    export_payload = {
        "model_state": ckpt["model_state"],
        "label_map": label_map,
        "max_len": max_len,
        "model_config": ckpt.get("model_config", {}),
    }
    torch.save(export_payload, export_path)

    config = {
        "architecture": "HybridGraphTransformer",
        "max_len": max_len,
        "num_classes": len(label_map),
        "model_config": ckpt.get("model_config", {}),
        "label_map": label_map,
    }
    (export_dir / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"Top-3 Accuracy: {top3_acc*100:.2f}%")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
