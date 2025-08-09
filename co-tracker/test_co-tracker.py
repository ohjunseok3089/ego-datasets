import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2

# Reuse the same ROI computation used by the analyzer
from track_prediction_past_frame import compute_content_roi
from track_red import detect_red_circle


def load_overlays_from_json(json_path: Path) -> Dict[int, Dict[str, Optional[Tuple[int, int, int, int]]]]:
    """
    Load per-frame overlays from analysis JSON: ROI and red circle (full-frame coords).

    Returns mapping: frame_index -> { 'roi': (x,y,w,h) or None, 'circle': (cx, cy, r) or None }
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    metadata = data.get("metadata", {})
    # Prefer metadata-level ROI, fallback to per-clip map if present
    default_roi = None
    if metadata.get("roi"):
        r = metadata["roi"]
        default_roi = (int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))
    roi_by_clip = metadata.get("roi_by_clip") or {}
    frame_size_default = None
    if metadata.get("frame_size_full"):
        s = metadata["frame_size_full"]
        frame_size_default = (int(s.get("width", 0)), int(s.get("height", 0)))
    frame_size_by_clip = metadata.get("frame_size_full_by_clip") or {}
    mapping: Dict[int, Dict[str, Optional[Tuple[int, int, int, int]]]] = {}
    for frame in frames:
        # Choose the most stable global index if available
        if "source_frame_index" in frame:
            frame_index = int(frame.get("source_frame_index"))
        elif "global_frame_index" in frame:
            frame_index = int(frame.get("global_frame_index"))
        else:
            frame_index = int(frame.get("frame_index"))
        # Try metadata-level ROI first, then per-clip, then per-frame (backward compat)
        roi_dict = {}
        clip_name = frame.get("source_clip") or frame.get("clip_name")
        if default_roi is not None:
            roi = default_roi
        elif clip_name and clip_name in roi_by_clip and roi_by_clip[clip_name] is not None:
            r = roi_by_clip[clip_name]
            roi = (int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))
        else:
            roi_dict = frame.get("roi") or {}
            roi: Optional[Tuple[int, int, int, int]] = None
            if all(k in roi_dict for k in ("x", "y", "w", "h")):
                roi = (
                    int(roi_dict["x"]),
                    int(roi_dict["y"]),
                    int(roi_dict["w"]),
                    int(roi_dict["h"]),
                )

        circle_dict = frame.get("red_circle") or {}
        circle_full: Optional[Tuple[int, int, int]] = None
        circle_content: Optional[Tuple[int, int, int]] = None
        if circle_dict.get("detected"):
            pos_full = circle_dict.get("position_full")
            radius = circle_dict.get("radius")
            if pos_full and len(pos_full) == 2 and radius is not None:
                circle_full = (int(pos_full[0]), int(pos_full[1]), int(radius))
            else:
                # Fallback: if only content position present, remap using ROI
                pos_content = circle_dict.get("position_content")
                if pos_content and len(pos_content) == 2 and radius is not None and roi is not None:
                    cx = int(pos_content[0]) + int(roi[0])
                    cy = int(pos_content[1]) + int(roi[1])
                    circle_full = (cx, cy, int(radius))
                    circle_content = (int(pos_content[0]), int(pos_content[1]), int(radius))

        # read original frame size if present
        # frame size from metadata, per-clip, or per-frame
        frame_size_full = None
        if frame_size_default is not None:
            frame_size_full = frame_size_default
        elif clip_name and clip_name in frame_size_by_clip and frame_size_by_clip[clip_name]:
            s = frame_size_by_clip[clip_name]
            frame_size_full = (int(s.get("width", 0)), int(s.get("height", 0)))
        else:
            fs = frame.get("frame_size_full") or {}
            if "width" in fs and "height" in fs:
                frame_size_full = (int(fs["width"]), int(fs["height"]))

        mapping[frame_index] = {
            "roi": roi,
            "circle_full": circle_full,
            "circle_content": circle_content,
            "frame_size_full": frame_size_full,
        }
    return mapping


def draw_roi_overlay(
    frame,
    roi: Optional[Tuple[int, int, int, int]],
    frame_index: int,
    color=(0, 255, 0),
    thickness: int = 2,
):
    """Draw ROI rectangle and annotation on the frame in-place."""
    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        label = f"frame={frame_index} roi=({x},{y},{w},{h})"
    else:
        label = f"frame={frame_index} roi=None"
    cv2.putText(
        frame,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )


def draw_circle_overlay(
    frame,
    circle: Optional[Tuple[int, int, int]],
    color=(0, 0, 255),
    thickness: int = 2,
):
    """Draw the detected red circle (center and radius) on the frame."""
    if circle is None:
        return
    cx, cy, r = circle
    cv2.circle(frame, (cx, cy), r, color, thickness)
    # draw center point
    cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)


def main():
    parser = argparse.ArgumentParser(
        description="Overlay ROI on the original video to verify correctness."
    )
    parser.add_argument(
        "video",
        type=str,
        help="Path to the original MP4 video.",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional path to analysis JSON. If provided, ROI per frame is read from JSON.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save an annotated video (e.g., /path/out.mp4).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the annotated video in a window.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally limit the number of frames processed for a quick check.",
    )
    parser.add_argument(
        "--recompute-each-frame",
        dest="recompute_each_frame",
        action="store_true",
        help="When no JSON is provided, recompute ROI for each frame instead of only first frame.",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    overlays_from_json: Dict[int, Dict[str, Optional[Tuple[int, int, int, int]]]] = {}
    if args.json:
        json_path = Path(args.json)
        if not json_path.is_file():
            raise FileNotFoundError(f"JSON not found: {json_path}")
        overlays_from_json = load_overlays_from_json(json_path)
        print(f"Loaded {len(overlays_from_json)} frame entries from JSON")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps if fps > 0 else 30.0, (width, height))

    print(f"Video: {video_path} | size=({width}x{height}) fps={fps:.2f}")

    frame_index = 0
    cached_roi: Optional[Tuple[int, int, int, int]] = None

    while True:
        if args.limit is not None and frame_index >= args.limit:
            break

        ok, frame = cap.read()
        if not ok:
            break

        roi: Optional[Tuple[int, int, int, int]] = None
        circle: Optional[Tuple[int, int, int]] = None

        if overlays_from_json:
            entry = overlays_from_json.get(frame_index, {})
            roi = entry.get("roi")
            circle = entry.get("circle_full")
            frame_size_full = entry.get("frame_size_full")
            # If the JSON was generated on a padded resolution and we're viewing the original content-only video,
            # remap ROI and circle to the current frame size when obvious.
            if frame_size_full is not None and (width, height) != frame_size_full and roi is not None:
                rx, ry, rw, rh = roi
                # If our current frame exactly matches the content size stored in ROI, use content coords
                if (width, height) == (rw, rh):
                    # shift ROI to origin and reuse size
                    roi = (0, 0, rw, rh)
                    if circle is not None:
                        cx, cy, r = circle
                        circle = (cx - rx, cy - ry, r)
                    else:
                        circle_content = entry.get("circle_content")
                        if circle_content is not None:
                            circle = circle_content
        else:
            # Compute ROI either once or at every frame
            if cached_roi is None or args.recompute_each_frame:
                roi = compute_content_roi(frame)
                if not args.recompute_each_frame:
                    cached_roi = roi
            else:
                roi = cached_roi

            # Recompute circle using the same detector on the content area
            if roi is not None:
                rx, ry, rw, rh = roi
                proc_frame = frame[ry : ry + rh, rx : rx + rw]
            else:
                rx = ry = 0
                rw, rh = frame.shape[1], frame.shape[0]
                proc_frame = frame
            detected = detect_red_circle(proc_frame)
            if detected:
                cx_full = int(detected[0]) + int(rx)
                cy_full = int(detected[1]) + int(ry)
                r_full = int(detected[2])
                circle = (cx_full, cy_full, r_full)

        draw_roi_overlay(frame, roi, frame_index)
        draw_circle_overlay(frame, circle)

        if writer is not None:
            writer.write(frame)
        if args.show:
            cv2.imshow("ROI Overlay", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print("Done.")


if __name__ == "__main__":
    main()


