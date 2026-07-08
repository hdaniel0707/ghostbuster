"""Count .txt files in a raw-data folder and flag empty ones.

Walks a folder recursively (e.g. one of the ghostbuster-data datasets) and
reports, to both the terminal and a log file:
  - total number of .txt files
  - number of empty .txt files (0 bytes, or whitespace-only) and *which* ones
  - a per-top-level-subfolder breakdown (e.g. human / claude / gpt / ...)

Any file living under a ``logprobs/`` directory is skipped (model artifacts,
not raw text).

Output log saved to results/txt_check/<folder_name>/check.log

Usage:
    uv run python analyse_txt_folder.py data/essay
    uv run python analyse_txt_folder.py data/reuter
    uv run python analyse_txt_folder.py data/wp
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results2" / "txt_check"

# Cap on how many empty-file paths to list before truncating.
MAX_EMPTY_SHOWN = 200

# Directory names to skip entirely (matched against any path component).
EXCLUDE_DIRS = {"logprobs", "headlines"}


class Tee:
    """Collect printed lines so they can be echoed to stdout and a log file."""

    def __init__(self) -> None:
        self._lines: list[str] = []

    def print(self, line: str = "") -> None:
        print(line)
        self._lines.append(line)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self._lines) + "\n")


def _emptiness(path: Path) -> str | None:
    """Classify a file: 'zero-byte', 'whitespace-only', or None if non-empty."""
    if path.stat().st_size == 0:
        return "zero-byte"
    text = path.read_text(encoding="utf-8", errors="replace")
    if text.strip() == "":
        return "whitespace-only"
    return None


def _category(path: Path, root: Path) -> str:
    """Top-level subfolder under root that a file belongs to ('.' if directly in root)."""
    rel = path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else "."


def analyse(root: Path, out: Tee) -> None:
    out.print("=" * 70)
    out.print(f"Raw .txt check: {root}")
    out.print(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    out.print("=" * 70)

    txt_files = sorted(
        f for f in root.rglob("*.txt")
        if EXCLUDE_DIRS.isdisjoint(f.relative_to(root).parts)
    )
    total = len(txt_files)

    # Per-category tallies + collected empties.
    per_cat_total: dict[str, int] = {}
    per_cat_empty: dict[str, int] = {}
    empties: list[tuple[Path, str]] = []

    for f in txt_files:
        cat = _category(f, root)
        per_cat_total[cat] = per_cat_total.get(cat, 0) + 1
        kind = _emptiness(f)
        if kind is not None:
            per_cat_empty[cat] = per_cat_empty.get(cat, 0) + 1
            empties.append((f, kind))

    out.print(f"\nTotal .txt files: {total}")
    out.print(f"Empty .txt files: {len(empties)}")

    out.print("\nPer-subfolder breakdown (files | empty):")
    for cat in sorted(per_cat_total):
        n = per_cat_total[cat]
        e = per_cat_empty.get(cat, 0)
        flag = "  <-- has empty" if e else ""
        out.print(f"  {cat:<20} {n:>8} | {e:>6}{flag}")

    out.print("\nEmpty files:")
    if empties:
        for f, kind in empties[:MAX_EMPTY_SHOWN]:
            out.print(f"  - {f.relative_to(root)}  ({kind})")
        if len(empties) > MAX_EMPTY_SHOWN:
            out.print(f"  ... (+{len(empties) - MAX_EMPTY_SHOWN} more)")
    else:
        out.print("  (none — no empty .txt files)")

    out.print("\n" + "=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", help="Folder to scan recursively for .txt files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.folder)
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    out = Tee()
    analyse(root, out)

    log_path = RESULTS_DIR / root.name / "check.log"
    out.save(log_path)
    print(f"\nLog saved to {log_path}")


if __name__ == "__main__":
    main()
