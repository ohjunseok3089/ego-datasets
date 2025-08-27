import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List, Optional


def compute_frame_range(start_time_s: float, end_time_s: Optional[float], fps: float) -> List[int]:
    if not (start_time_s is not None and math.isfinite(start_time_s)):
        return []
    start_frame = math.floor(start_time_s * fps)
    if end_time_s is None or not math.isfinite(end_time_s):
        return [start_frame]
    end_inclusive = max(start_frame, math.ceil(end_time_s * fps) - 1)
    if end_inclusive < start_frame:
        end_inclusive = start_frame
    return list(range(start_frame, end_inclusive + 1))


def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run_whisperx(
    input_path: str,
    output_base_dir: str,
    model_size: str = "large-v3",
    language: str = "en",
    device: str = "cuda",
    compute_type: str = "float16",
    batch_size: int = 16,
    hf_token: Optional[str] = None,
    vad_threshold: Optional[float] = None,
    fps: float = 30.0,
) -> Dict[str, Any]:
    import whisperx  # type: ignore

    os.makedirs(output_base_dir, exist_ok=True)

    audio = whisperx.load_audio(input_path)
    model = whisperx.load_model(model_size, device=device, compute_type=compute_type, language=language or "en")
    result = model.transcribe(audio, batch_size=batch_size, language=language)

    lang_code = language or result.get("language") or "en"
    align_model, metadata = whisperx.load_align_model(language_code=lang_code, device=device)
    result_aligned = whisperx.align(result.get("segments", []), align_model, metadata, audio, device, vad_threshold=vad_threshold)

    hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    diar = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = diar(audio)
    result_spk, _ = whisperx.assign_word_speakers(diarize_segments, result_aligned)

    # Build rows in target schema with speaker_id mapped to person_1..N
    conversation_id = os.path.splitext(os.path.basename(input_path))[0]
    rows: List[Dict[str, Any]] = []

    words = result_spk.get("word_segments") or []
    speaker_map: Dict[str, str] = {}
    next_id = 1

    def map_speaker(raw: str) -> str:
        nonlocal next_id
        if raw not in speaker_map:
            speaker_map[raw] = f"person_{next_id}"
            next_id += 1
        return speaker_map[raw]

    if words:
        for w in sorted(words, key=lambda x: float(x.get("start", 0.0))):
            map_speaker(w.get("speaker") or "SPEAKER_00")
    else:
        for s in sorted(result_spk.get("segments", []), key=lambda x: float(x.get("start", 0.0))):
            map_speaker(s.get("speaker") or "SPEAKER_00")

    if words:
        for w in words:
            start = float(w.get("start", 0.0))
            end = float(w.get("end", start))
            spk = map_speaker(w.get("speaker") or "SPEAKER_00")
            token = (w.get("text") or w.get("word") or "").strip()
            frames = compute_frame_range(start, end, fps)
            rows.append({
                "conversation_id": conversation_id,
                "endTime": round(end, 2),
                "speaker_id": spk,
                "startTime": round(start, 2),
                "word": token,
                "frame": json.dumps(frames, separators=(",", ":")),
            })
    else:
        for seg in result_spk.get("segments", []):
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            spk = map_speaker(seg.get("speaker") or "SPEAKER_00")
            text = (seg.get("text") or "").strip()
            toks = text.split() or [text]
            n = len(toks)
            dur = max(0.0, end - start)
            step = dur / n if n else 0.0
            for i, tok in enumerate(toks):
                ws = start + i * step
                we = start + (i + 1) * step if dur > 0 else start
                frames = compute_frame_range(ws, we, fps)
                rows.append({
                    "conversation_id": conversation_id,
                    "endTime": round(we, 2),
                    "speaker_id": spk,
                    "startTime": round(ws, 2),
                    "word": tok,
                    "frame": json.dumps(frames, separators=(",", ":")),
                })

    # Append to shared transcript CSV
    transcript_path = os.path.join(output_base_dir, "transcript", "ground_truth_transcriptions_with_frames.csv")
    os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
    file_exists = os.path.exists(transcript_path)
    with open(transcript_path, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
        fieldnames = ["conversation_id", "endTime", "speaker_id", "startTime", "word", "frame"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    # Optional debug dump per input
    try:
        debug_dir = os.path.join(output_base_dir, "transcript_debug")
        save_json({
            "language": lang_code,
            "segments": result_spk.get("segments", []),
            "word_segments": result_spk.get("word_segments", []),
        }, os.path.join(debug_dir, f"{conversation_id}_whisperx.json"))
    except Exception:
        pass

    return {"rows": len(rows)}


def main() -> None:
    p = argparse.ArgumentParser(description="Aria WhisperX diarization to CSV append (ego schema)")
    p.add_argument("--input_path", required=True, type=str, help="Path to input audio/video (wav, mp4)")
    p.add_argument("--output_base_dir", required=True, type=str, help="Base dir where transcript/ CSV lives")
    p.add_argument("--model_size", default="large-v3", type=str)
    p.add_argument("--language", default="en", type=str)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--compute_type", default="float16", type=str)
    p.add_argument("--batch_size", default=16, type=int)
    p.add_argument("--hf_token", default=None, type=str)
    p.add_argument("--vad_threshold", default=None, type=float)
    p.add_argument("--fps", default=30.0, type=float, help="FPS used to convert times to frames")
    args = p.parse_args()

    run_whisperx(
        input_path=args.input_path,
        output_base_dir=args.output_base_dir,
        model_size=args.model_size,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        hf_token=args.hf_token,
        vad_threshold=args.vad_threshold,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()

