"""
Add frame index lists to a transcript CSV.

Input CSV is expected to have columns:
- conversation_id
- startTime (seconds)
- endTime (seconds)
- speaker_id
- word

This script appends a column `frame` containing an inclusive list of frame
indices that cover the interval [startTime, endTime) at a given FPS
(default 30). The convention follows existing code in this repo:
- start_frame = floor(startTime * fps)
- end_frame_inclusive = max(start_frame, ceil(endTime * fps) - 1)

If `endTime` is missing, we treat the span as one frame at `start_frame`.

By default, frames are 0-based. You can opt-in to 1-based via `--one-based`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


def _to_int(value: float) -> int:
    return int(value)


def compute_frame_range(start_time_s: float, end_time_s: Optional[float], fps: float) -> List[int]:
    """Compute an inclusive list of frame indices covering [start, end).

    If `end_time_s` is None or not finite, returns a single-frame list at the
    start frame.
    """
    if not np.isfinite(start_time_s):
        return []

    start_frame: int = math.floor(start_time_s * fps)

    if end_time_s is None or not np.isfinite(end_time_s):
        return [start_frame]

    # Inclusive end frame covers up to but not including end_time_s
    end_frame_inclusive: int = max(start_frame, math.ceil(end_time_s * fps) - 1)

    # Guard against pathological spans
    if end_frame_inclusive < start_frame:
        end_frame_inclusive = start_frame

    # Generate consecutive frames [start_frame, ..., end_frame_inclusive]
    return list(range(start_frame, end_frame_inclusive + 1))


def add_frame_column(
    df: pd.DataFrame,
    fps: float = 30.0,
    one_based: bool = False,
    frame_column: str = "frame",
) -> pd.DataFrame:
    """Return a copy of df with a new `frame` column containing JSON arrays.

    - Ensures required columns exist
    - Coerces numeric types for start/end
    - Applies rounding policy (floor start, ceil-1 end)
    - Optionally converts to 1-based frame indices
    """
    required_cols = {"startTime", "word"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    # Ensure numeric types; allow NaNs in endTime
    df = df.copy()
    df["startTime"] = pd.to_numeric(df["startTime"], errors="coerce")
    if "endTime" in df.columns:
        df["endTime"] = pd.to_numeric(df["endTime"], errors="coerce")
    else:
        df["endTime"] = np.nan

    def _compute_row_frames(row) -> str:
        frames = compute_frame_range(
            start_time_s=float(row["startTime"]),
            end_time_s=float(row["endTime"]) if np.isfinite(row["endTime"]) else None,
            fps=fps,
        )
        if one_based:
            frames = [f + 1 for f in frames]
        # Store as compact JSON array string (e.g., "[5,6,7]")
        return json.dumps(frames, separators=(",", ":"))

    # Drop rows without valid startTime
    valid_mask = np.isfinite(df["startTime"].astype(float))
    df = df.loc[valid_mask].reset_index(drop=True)

    df[frame_column] = df.apply(_compute_row_frames, axis=1)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add a frame index list column to a transcript CSV",
    )
    parser.add_argument("input_csv", type=Path, help="Path to input transcript CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (defaults to input path with _with_frames suffix)",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second")
    parser.add_argument(
        "--one-based",
        action="store_true",
        help="Use 1-based frame indices instead of 0-based",
    )
    parser.add_argument(
        "--frame-column",
        type=str,
        default="frame",
        help="Name of the output frame column",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv: Path = args.input_csv
    output_csv: Path = (
        args.output
        if args.output is not None
        else input_csv.with_name(input_csv.stem + "_with_frames" + input_csv.suffix)
    )

    df = pd.read_csv(input_csv)
    df_out = add_frame_column(
        df,
        fps=float(args.fps),
        one_based=bool(args.one_based),
        frame_column=str(args.frame_column),
    )

    df_out.to_csv(output_csv, index=False)
    print(f"Wrote: {output_csv}")


if __name__ == "__main__":
    main()


