import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def convert_to_json_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_json_serializable(v) for v in obj]
    if isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float_)):
        return float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def infer_fps_from_frames(frames: List[Dict[str, Any]]) -> float:
    if not frames or len(frames) < 2:
        return 30.0
    frames_sorted = sorted(frames, key=lambda f: int(f.get("frame_index", 0)))
    ts = np.array([float(f.get("timestamp", 0.0)) for f in frames_sorted], dtype=float)
    dts = np.diff(ts)
    dts = dts[dts > 0]
    if dts.size == 0:
        return 30.0
    fps = 1.0 / float(np.median(dts))
    return float(fps)


def extract_conversation_id_from_group_id(group_id: str) -> Optional[str]:
    # e.g., "vid_001__day_1__con_1__person_1_part1(0_1920_social_interaction)" -> "day_1__con_1"
    m = re.search(r"(day_\d+__con_\d+)", group_id)
    return m.group(1) if m else None


def map_transcriptions_to_frames(transcriptions_csv: Path, frames: List[Dict[str, Any]], conversation_hint: Optional[str]) -> Dict[int, List[Dict[str, Any]]]:
    df = pd.read_csv(transcriptions_csv)
    # Optional filter by conversation id substring
    if conversation_hint and "conversation_id" in df.columns:
        df = df[df["conversation_id"].astype(str).str.contains(conversation_hint, na=False)]

    # Determine FPS from frames to map seconds → frames when needed
    fps = infer_fps_from_frames(frames)

    # Normalize column names we need
    # Expect either frame_number column OR startTime (seconds) and speaker_id, word
    if "frame_number" not in df.columns and "startTime" in df.columns:
        df = df.dropna(subset=["startTime"])  # keep only with startTime
        df["frame_number"] = (df["startTime"] * fps).apply(lambda x: int(math.floor(x)))

    speaker_events_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    if {"frame_number", "speaker_id", "word"}.issubset(df.columns):
        tmp = df[["frame_number", "speaker_id", "word"]].copy()
        tmp["speaker_id"] = tmp["speaker_id"].astype(int)
        for _, row in tmp.iterrows():
            frame_idx = int(row["frame_number"])
            event = {"id": int(row["speaker_id"]), "word": str(row["word"]) }
            speaker_events_by_frame.setdefault(frame_idx, []).append(event)
    return speaker_events_by_frame


def index_face_gallery(face_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(face_csv)
    # Expect columns: frame_number, person_id, x1,y1,x2,y2
    required = {"frame_number", "person_id", "x1", "y1", "x2", "y2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Face gallery CSV missing required columns: {sorted(missing)}")
    df = df.copy()
    df["person_id"] = df["person_id"].astype(int)
    # If multiple boxes exist for the same (frame, person), keep the largest area
    df["area"] = (df["x2"] - df["x1"]).clip(lower=0) * (df["y2"] - df["y1"]).clip(lower=0)
    df.sort_values(["frame_number", "person_id", "area"], ascending=[True, True, False], inplace=True)
    df = df.drop_duplicates(subset=["frame_number", "person_id"], keep="first")
    df = df[["frame_number", "person_id", "x1", "y1", "x2", "y2"]]
    df.set_index(["frame_number", "person_id"], inplace=True)
    df.sort_index(inplace=True)
    return df


def join_ground_truth(transcriptions_csv: Path, analysis_json_in: Path, face_gallery_csv: Path, output_json: Path) -> None:
    with open(analysis_json_in, "r") as f:
        analysis = json.load(f)

    frames: List[Dict[str, Any]] = analysis.get("frames", [])
    metadata: Dict[str, Any] = analysis.get("metadata", {})

    conversation_hint = None
    if isinstance(metadata.get("group_id"), str):
        conversation_hint = extract_conversation_id_from_group_id(metadata["group_id"])

    speaker_events_by_frame = map_transcriptions_to_frames(transcriptions_csv, frames, conversation_hint)
    face_df = index_face_gallery(face_gallery_csv)

    total = len(frames)
    for i, frame in enumerate(frames):
        if (i + 1) % 100 == 0 or i == total - 1:
            print(f"Processing frame {i+1}/{total}")

        frame_index = int(frame.get("frame_index", i))

        # Initialize/clear fields
        frame.setdefault("speaker_id", None)
        frame.setdefault("speaker_words", [])
        frame.setdefault("speaker_location", {})

        # Attach speaker info from transcriptions
        events = speaker_events_by_frame.get(frame_index)
        if events:
            # Choose first speaker event in the frame (simple policy)
            speaker_id = int(events[0]["id"]) if "id" in events[0] else None
            frame["speaker_id"] = speaker_id
            frame["speaker_words"] = [evt["word"] for evt in events if evt.get("id") == speaker_id]

            # Try exact face box lookup for (frame, person_id)
            try:
                loc = face_df.loc[(frame_index, speaker_id)]
                # loc can be a Series (unique index) or DataFrame (if duplicates slipped in)
                if isinstance(loc, pd.Series):
                    vals = loc
                else:
                    vals = loc.iloc[0]
                frame["speaker_location"] = {
                    "x1": int(vals["x1"]),
                    "y1": int(vals["y1"]),
                    "x2": int(vals["x2"]),
                    "y2": int(vals["y2"]),
                }
            except KeyError:
                # No box for this exact frame/person; leave empty
                pass

    # Save to output
    out = {
        "metadata": metadata,
        "analysis_summary": analysis.get("analysis_summary", {}),
        "frames": frames,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(convert_to_json_serializable(out), f, indent=2)
    print(f"Saved merged JSON to {output_json}")


def main():
    parser = argparse.ArgumentParser(description="Join ground-truth transcriptions and face gallery into an analysis JSON.")
    parser.add_argument("--transcriptions_csv", type=str, required=True)
    parser.add_argument("--analysis_json_in", type=str, required=True)
    parser.add_argument("--face_gallery_csv", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    join_ground_truth(
        transcriptions_csv=Path(args.transcriptions_csv),
        analysis_json_in=Path(args.analysis_json_in),
        face_gallery_csv=Path(args.face_gallery_csv),
        output_json=Path(args.output_json),
    )


if __name__ == "__main__":
    main()


