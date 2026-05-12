import argparse
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, Tuple

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

ALEF_MAP = {
    "\u0622": "\u0627",  # ALEF WITH MADDA ABOVE
    "\u0623": "\u0627",  # ALEF WITH HAMZA ABOVE
    "\u0625": "\u0627",  # ALEF WITH HAMZA BELOW
    "\u0671": "\u0627",  # ALEF WASLA
}

DASHES = ("\u2013", "\u2014")  # EN DASH, EM DASH
TATWEEL = "\u0640"
SUFFIX_RE = re.compile(r"(?:\s*[_-]?\s*\(\d+\)|\s*[_-]?\s*\d+)\s*$")


def normalize_label(label: str) -> str:
    s = label.strip()
    s = s.rstrip(".")
    for dash in DASHES:
        s = s.replace(dash, "-")

    # Strip duplicate-style suffixes like _1, (2), -3.
    while True:
        new = SUFFIX_RE.sub("", s)
        if new == s:
            break
        s = new.strip().rstrip(".")

    s = re.sub(r"\s+", " ", s).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.replace(TATWEEL, "")
    s = "".join(ALEF_MAP.get(ch, ch) for ch in s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.lower()
    return s or "unknown"


def iter_video_files(input_root: Path, output_root: Path) -> Iterable[Path]:
    for path in input_root.rglob("*"):
        if not path.is_file():
            continue
        if output_root in path.parents:
            continue
        if path.suffix.lower() in VIDEO_EXTS:
            yield path


def make_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for i in range(1, 10000):
        candidate = path.with_name(f"{stem}__dup{i}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Too many duplicates for {path}")


def build_default_paths() -> Tuple[Path, Path]:
    base = Path(__file__).resolve().parent
    input_root = base / "grouped_output"
    output_root = base / "grouped_by_word"
    return input_root, output_root


def parse_args() -> argparse.Namespace:
    default_input, default_output = build_default_paths()
    parser = argparse.ArgumentParser(
        description="Group video files into folders by normalized label."
    )
    parser.add_argument(
        "--input",
        default=str(default_input),
        help="Input folder with video files (default: grouped_output beside this script)",
    )
    parser.add_argument(
        "--output",
        default=str(default_output),
        help="Output folder for grouped videos (default: grouped_by_word beside this script)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without copying or moving",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each file action",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()

    if not input_root.exists():
        print(f"Input folder not found: {input_root}")
        return 1

    output_root.mkdir(parents=True, exist_ok=True)

    total = 0
    copied = 0
    duplicates = 0
    groups: Dict[str, int] = {}

    for src in iter_video_files(input_root, output_root):
        total += 1
        label = normalize_label(src.stem)
        group_dir = output_root / label
        if not args.dry_run:
            group_dir.mkdir(parents=True, exist_ok=True)

        dest = group_dir / src.name
        if dest.exists():
            dest = make_unique_path(dest)
            duplicates += 1

        if args.verbose or args.dry_run:
            action = "MOVE" if args.move else "COPY"
            if args.dry_run:
                action = f"DRY-{action}"
            print(f"{action}: {src} -> {dest}")

        if not args.dry_run:
            if args.move:
                shutil.move(str(src), str(dest))
            else:
                shutil.copy2(str(src), str(dest))

        copied += 1
        groups[label] = groups.get(label, 0) + 1

    print("Done.")
    print(f"Videos processed: {total}")
    print(f"Videos grouped:   {copied}")
    print(f"Groups created:   {len(groups)}")
    print(f"Name collisions:  {duplicates}")
    print(f"Output folder:    {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
