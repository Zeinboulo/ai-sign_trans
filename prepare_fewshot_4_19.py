import json
from pathlib import Path

import pandas as pd

MIN_SAMPLES = 4
MAX_SAMPLES = 19

MANIFEST_PATH = Path("./dataset_manifest.csv")
OUT_MANIFEST = Path("./fewshot_4_19_manifest.csv")
OUT_LABEL_MAP = Path("./fewshot_4_19_label_map.json")


def main() -> int:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST_PATH}")

    df = pd.read_csv(MANIFEST_PATH)
    counts = df.groupby("word").size()
    keep_words = set(counts[(counts >= MIN_SAMPLES) & (counts <= MAX_SAMPLES)].index)

    filtered = df[df["word"].isin(keep_words)].copy()
    words_sorted = sorted(filtered["word"].unique())
    label_map = {word: idx for idx, word in enumerate(words_sorted)}
    filtered["label_index"] = filtered["word"].map(label_map)

    filtered.to_csv(OUT_MANIFEST, index=False)
    OUT_LABEL_MAP.write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")

    removed_classes = len(counts) - len(words_sorted)
    removed_samples = len(df) - len(filtered)

    print(f"Sample range per class: {MIN_SAMPLES}-{MAX_SAMPLES}")
    print(f"Classes kept: {len(words_sorted)}")
    print(f"Samples kept: {len(filtered)}")
    print(f"Classes removed: {removed_classes}")
    print(f"Samples removed: {removed_samples}")
    print(f"Filtered manifest: {OUT_MANIFEST}")
    print(f"Filtered label map: {OUT_LABEL_MAP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
