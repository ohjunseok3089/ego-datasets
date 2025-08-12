"""
Add `speaker_id` to face-tracking CSVs using manual mappings per conversation and wearer.

Input CSV format (expected):
- frame_number, person_id, x1, y1, x2, y2 [, ...]

We detect which mapping to use from the filename, e.g.:
  vid_001__day_1__con_1__person_1_part1_global_gallery.csv
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Parse the key: day_1__con_1__person_1 (part index is ignored)

Mappings are derived from `notes.sh` and embedded below. Some entries include
rows to delete (e.g., "delete person_1, person_2").

Usage:
  python face_tracking_speaker_id_mapping.py INPUT_PATH [--output-dir DIR | --inplace]

  - INPUT_PATH: a CSV file or a directory containing CSVs
  - --output-dir: directory to write updated CSVs (defaults to same dir)
  - --inplace: overwrite input files (mutually exclusive with --output-dir)
  - --suffix: filename suffix before .csv when not using --inplace (default: _with_speaker)
  - --dry-run: don't write files; just print summary
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ------------------------------
# Mapping data from notes.sh
# ------------------------------

@dataclass(frozen=True)
class MappingSpec:
    person_to_speaker: Dict[str, int]
    delete_persons: Tuple[str, ...] = ()


def _p(ids: List[int], speaker: int) -> Dict[str, int]:
    return {f"person_{i}": speaker for i in ids}


# Key format: "day_X__con_Y__person_Z"
MAPPINGS: Dict[str, MappingSpec] = {
    # day_1__con_1
    "day_1__con_1__person_1": MappingSpec(
        person_to_speaker={
            **_p([1], 2),
            **_p([2], 3),
            **_p([3], 1),
        }
    ),
    "day_1__con_1__person_2": MappingSpec(
        person_to_speaker={
            **_p([1, 2], 1),
            **_p([3], 3),
        }
    ),
    "day_1__con_1__person_3": MappingSpec(
        person_to_speaker={
            **_p([4, 5], 1),
            **_p([3], 2),
        },
        delete_persons=("person_1", "person_2"),
    ),

    # day_1__con_2
    "day_1__con_2__person_1": MappingSpec(person_to_speaker={**_p([1], 2), **_p([2], 3)}),
    "day_1__con_2__person_2": MappingSpec(person_to_speaker={**_p([1], 1), **_p([2], 3)}),
    "day_1__con_2__person_3": MappingSpec(
        person_to_speaker={**_p([1], 3), **_p([2], 1), **_p([3], 2)}
    ),

    # day_1__con_3
    "day_1__con_3__person_1": MappingSpec(
        person_to_speaker={**_p([1], 1), **_p([2], 3), **_p([3], 2)}
    ),
    "day_1__con_3__person_2": MappingSpec(
        person_to_speaker={**_p([2, 3], 1), **_p([1], 3), **_p([4], 2)}
    ),
    "day_1__con_3__person_3": MappingSpec(person_to_speaker={**_p([1], 1), **_p([2], 3)}),

    # day_1__con_4
    "day_1__con_4__person_1": MappingSpec(person_to_speaker={**_p([1], 2), **_p([2, 3], 3)}),
    "day_1__con_4__person_2": MappingSpec(person_to_speaker={**_p([1], 1), **_p([2], 3)}),
    "day_1__con_4__person_3": MappingSpec(person_to_speaker={**_p([1], 1), **_p([2], 2)}),

    # day_1__con_5
    "day_1__con_5__person_1": MappingSpec(
        person_to_speaker={**_p([1], 1), **_p([2], 3), **_p([3], 2)}
    ),
    "day_1__con_5__person_2": MappingSpec(person_to_speaker={**_p([1], 3), **_p([2], 1)}),

    # day_2__con_1
    "day_2__con_1__person_1": MappingSpec(
        person_to_speaker={**_p([1], 1), **_p([2], 2), **_p([3], 3)}
    ),
    "day_2__con_1__person_2": MappingSpec(person_to_speaker={**_p([1], 1), **_p([2], 3)}),
    "day_2__con_1__person_3": MappingSpec(person_to_speaker={**_p([1], 1), **_p([2], 2)}),

    # day_2__con_2
    "day_2__con_2__person_1": MappingSpec(
        person_to_speaker={**_p([4, 5], 1), **_p([1, 3], 2)},
        delete_persons=("person_2",),
    ),
    "day_2__con_2__person_2": MappingSpec(person_to_speaker={**_p([1], 1), **_p([2], 3)}),
}


def parse_key_from_filename(filename: str) -> Optional[str]:
    """Extract key like 'day_1__con_1__person_1' from filename.

    Works for both CSV or MP4 bases. Returns None if not found.
    """
    base = os.path.basename(filename)
    stem, _ = os.path.splitext(base)
    # Example stems:
    #   vid_001__day_1__con_1__person_1_part1_global_gallery
    #   vid_001__day_1__con_1__person_1_part1 (if not the global_gallery CSV)
    m = re.search(r"(day_\d+__con_\d+__person_\d+)", stem)
    return m.group(1) if m else None


def normalize_person_id(value) -> Optional[str]:
    """Normalize person_id to 'person_<int>' string form.

    Accepts ints, strings like '1', 'person_1', etc. Returns None if invalid.
    """
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return f"person_{int(value)}"
    s = str(value).strip()
    if not s:
        return None
    # Already person_<n>
    if re.fullmatch(r"person_\d+", s):
        return s
    # Pure integer string
    if s.isdigit():
        return f"person_{int(s)}"
    # Try to extract trailing digits
    m = re.search(r"(\d+)$", s)
    if m:
        return f"person_{int(m.group(1))}"
    return None


def apply_mapping(df: pd.DataFrame, key: str) -> Tuple[pd.DataFrame, dict]:
    """Apply mapping for given key; return updated df and stats.

    Stats keys: deleted_rows, mapped_rows, unmapped_rows
    """
    stats = {"deleted_rows": 0, "mapped_rows": 0, "unmapped_rows": 0}
    if key not in MAPPINGS:
        df = df.copy()
        df["speaker_id"] = np.nan
        stats["unmapped_rows"] = len(df)
        return df, stats

    spec = MAPPINGS[key]

    # Drop rows to be deleted (by person_id label)
    df = df.copy()
    if spec.delete_persons:
        norm_ids = df["person_id"].apply(normalize_person_id)
        delete_mask = norm_ids.isin(set(spec.delete_persons))
        stats["deleted_rows"] = int(delete_mask.sum())
        df = df.loc[~delete_mask].reset_index(drop=True)

    # Map person_id -> speaker_id
    norm_ids_after = df["person_id"].apply(normalize_person_id)
    mapped = norm_ids_after.map(spec.person_to_speaker)
    df["speaker_id"] = mapped

    stats["mapped_rows"] = int(mapped.notna().sum())
    stats["unmapped_rows"] = int(mapped.isna().sum())
    return df, stats


def process_csv_file(csv_path: str, output_dir: Optional[str], inplace: bool, suffix: str, dry_run: bool) -> None:
    if not os.path.isfile(csv_path) or not csv_path.lower().endswith(".csv"):
        print(f"Skipping non-CSV: {csv_path}")
        return

    key = parse_key_from_filename(csv_path)
    if not key:
        print(f"[WARN] Could not parse mapping key from filename: {os.path.basename(csv_path)}. Writing speaker_id as NaN.")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV '{csv_path}': {e}")
        return

    required_cols = {"frame_number", "person_id", "x1", "y1", "x2", "y2"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[WARN] '{os.path.basename(csv_path)}' missing columns {missing}; continuing with available columns.")

    df_out, stats = apply_mapping(df, key or "")

    # Decide output path
    if inplace:
        out_path = csv_path
    else:
        out_dir = output_dir or os.path.dirname(csv_path)
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.basename(csv_path)
        stem, ext = os.path.splitext(base)
        out_path = os.path.join(out_dir, f"{stem}{suffix}{ext}")

    # Report
    print(f"Processed: {os.path.basename(csv_path)} | key={key or 'N/A'} | mapped={stats['mapped_rows']} | unmapped={stats['unmapped_rows']} | deleted={stats['deleted_rows']}")

    if dry_run:
        return

    try:
        df_out.to_csv(out_path, index=False)
        if not inplace and out_path != csv_path:
            print(f"  -> Wrote: {out_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write '{out_path}': {e}")


def process_path(input_path: str, output_dir: Optional[str], inplace: bool, suffix: str, dry_run: bool) -> None:
    if os.path.isfile(input_path):
        process_csv_file(input_path, output_dir, inplace, suffix, dry_run)
        return
    if os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for name in files:
                if name.lower().endswith(".csv"):
                    process_csv_file(os.path.join(root, name), output_dir, inplace, suffix, dry_run)
        return
    print(f"[ERROR] Input path is not a file or directory: {input_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add speaker_id to face-tracking CSVs using manual mappings.")
    p.add_argument("input_path", type=str, help="CSV file or directory containing CSVs")
    p.add_argument("--output-dir", type=str, default=None, help="Directory to write outputs (default: same as input)")
    p.add_argument("--inplace", action="store_true", help="Overwrite input files in place")
    p.add_argument("--suffix", type=str, default="_with_speaker", help="Suffix for output filenames when not using --inplace")
    p.add_argument("--dry-run", action="store_true", help="Don't write files; print what would happen")
    args = p.parse_args()

    if args.inplace and args.output_dir:
        p.error("--inplace and --output-dir are mutually exclusive")
    return args


def main() -> None:
    args = parse_args()
    process_path(args.input_path, args.output_dir, args.inplace, args.suffix, args.dry_run)


if __name__ == "__main__":
    main()


