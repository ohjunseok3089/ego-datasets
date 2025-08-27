import argparse
import json
import os
import sys
import math
from typing import Any, Dict, List, Optional


def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def compute_frame_range(start_time_s: float, end_time_s: Optional[float], fps: float = 30.0) -> List[int]:
    if not (start_time_s is not None and math.isfinite(start_time_s)):
        return []
    start_frame = math.floor(start_time_s * fps)
    if end_time_s is None or not math.isfinite(end_time_s):
        return [start_frame]
    end_inclusive = max(start_frame, math.ceil(end_time_s * fps) - 1)
    if end_inclusive < start_frame:
        end_inclusive = start_frame
    return list(range(start_frame, end_inclusive + 1))


def run_whisperx(
    input_path: str,
    output_base_dir: str,
    model_size: str = "large-v3",
    language: Optional[str] = "en",
    device: str = "cuda",
    compute_type: str = "float16",
    batch_size: int = 16,
    hf_token: Optional[str] = None,
    vad_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    import whisperx  # type: ignore

    print(f"[WhisperX] Loading audio: {input_path}")
    audio = whisperx.load_audio(input_path)

    print(f"[WhisperX] Loading ASR model: {model_size} on {device} ({compute_type})")
    model = whisperx.load_model(model_size, device=device, compute_type=compute_type, language=language or "en")

    print(f"[WhisperX] Transcribing (batch_size={batch_size}, language={language or 'auto'})")
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    segments = result.get("segments", [])
    print(f"[WhisperX] Transcribed segments: {len(segments)}")

    # Alignment
    lang_code = language or result.get("language") or "en"
    print(f"[WhisperX] Loading alignment model for language: {lang_code}")
    align_model, metadata = whisperx.load_align_model(language_code=lang_code, device=device)
    result_aligned = whisperx.align(segments, align_model, metadata, audio, device, vad_threshold=vad_threshold)

    # Diarization
    hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("[WARN] No HF token found in --hf_token/HF_TOKEN; diarization model may fail to download.")
    print("[WhisperX] Running speaker diarization")
    diarize_pipeline = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = diarize_pipeline(audio)

    print("[WhisperX] Assigning speakers to words/segments")
    result_spk, _ = whisperx.assign_word_speakers(diarize_segments, result_aligned)

    # Prepare transcript rows matching annotations_into_face_and_transcript.py
    conversation_id = os.path.splitext(os.path.basename(input_path))[0]
    rows: List[Dict[str, Any]] = []

    # Prefer word-level segments if available
    words = result_spk.get("word_segments") or []
    # Build a stable mapping from WhisperX speakers to person_1..N by first appearance in time
    speaker_map: Dict[str, str] = {}
    order_counter = 1
    def map_speaker(spk_raw: str) -> str:
        nonlocal order_counter
        if spk_raw not in speaker_map:
            speaker_map[spk_raw] = f"person_{order_counter}"
            order_counter += 1
        return speaker_map[spk_raw]
    # Prime map using time-ordered words if available, else segments
    if words:
        for w in sorted(words, key=lambda x: float(x.get("start", 0.0))):
            spk0 = w.get("speaker") or "SPEAKER_00"
            map_speaker(spk0)
    else:
        for s in sorted(result_spk.get("segments", []), key=lambda x: float(x.get("start", 0.0))):
            spk0 = s.get("speaker") or "SPEAKER_00"
            map_speaker(spk0)
    if words:
        for w in words:
            start = float(w.get("start", 0.0))
            end = float(w.get("end", start))
            speaker = map_speaker(w.get("speaker") or "SPEAKER_00")
            token = w.get("text") or w.get("word") or ""
            frames = compute_frame_range(start, end, fps=30.0)
            rows.append({
                "conversation_id": conversation_id,
                "endTime": round(end, 2),
                "speaker_id": speaker,
                "startTime": round(start, 2),
                "word": token.strip(),
                "frame": json.dumps(frames, separators=(",", ":")),
            })
    else:
        # Fallback: split segment text evenly if word segments unavailable
        for seg in result_spk.get("segments", []):
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            speaker = map_speaker(seg.get("speaker") or "SPEAKER_00")
            text = (seg.get("text") or "").strip()
            toks = text.split()
            n = max(1, len(toks))
            dur = max(0.0, end - start)
            step = dur / n if n else 0.0
            for i, tok in enumerate(toks or [text]):
                ws = start + i * step
                we = start + (i + 1) * step if dur > 0 else start
                frames = compute_frame_range(ws, we, fps=30.0)
                rows.append({
                    "conversation_id": conversation_id,
                    "endTime": round(we, 2),
                    "speaker_id": speaker,
                    "startTime": round(ws, 2),
                    "word": tok,
                    "frame": json.dumps(frames, separators=(",", ":")),
                })

    # Append to unified transcript path
    transcript_path = os.path.join(output_base_dir, 'transcript', 'ground_truth_transcriptions_with_frames.csv')
    os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
    import csv
    file_exists = os.path.exists(transcript_path)
    with open(transcript_path, 'a' if file_exists else 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['conversation_id', 'endTime', 'speaker_id', 'startTime', 'word', 'frame']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
    print(f"[WhisperX] {'Appended to' if file_exists else 'Created'} transcript: {transcript_path} (+{len(rows)} words)")

    return {
        "language": lang_code,
        "segments": result_spk.get("segments", []),
        "word_segments": result_spk.get("word_segments", []),
        "meta": {
            "model_size": model_size,
            "device": device,
            "compute_type": compute_type,
            "batch_size": batch_size,
            "vad_threshold": vad_threshold,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="WhisperX diarization for a single media file (ego4d)")
    parser.add_argument("--input_path", required=True, type=str, help="Path to input video/audio")
    parser.add_argument("--output_base_dir", required=True, type=str, help="Base directory with 'transcript/' to append CSV like annotations_into_face_and_transcript.py")
    parser.add_argument("--model_size", default="large-v3", type=str, help="WhisperX model size")
    parser.add_argument("--language", default="en", type=str, help="Language code (default: en)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to run")
    parser.add_argument("--compute_type", default="float16", type=str, help="Compute type (float16/int8_float16/etc)")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size for ASR")
    parser.add_argument("--hf_token", default=None, type=str, help="HuggingFace token for diarization models")
    parser.add_argument("--vad_threshold", default=None, type=float, help="Optional VAD threshold for alignment")

    args = parser.parse_args()

    result = run_whisperx(
        input_path=args.input_path,
        output_base_dir=args.output_base_dir,
        model_size=args.model_size,
        language=args.language or "en",
        device=args.device,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        hf_token=args.hf_token,
        vad_threshold=args.vad_threshold,
    )
    # Optional debug JSON alongside appends (can be disabled if undesired)
    try:
        debug_dir = os.path.join(args.output_base_dir, 'transcript_debug')
        base = os.path.splitext(os.path.basename(args.input_path))[0]
        save_json(result, os.path.join(debug_dir, f"{base}_whisperx.json"))
    except Exception:
        pass


if __name__ == "__main__":
    main()
