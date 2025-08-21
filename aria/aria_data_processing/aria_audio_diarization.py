#!/usr/bin/env python3
"""
Aria Audio Diarization and Speaker Labeling Script

This script processes Aria dataset speech.csv files to add speaker diarization labels.
It uses pyannote.audio for speaker diarization and adds speaker_label column to existing CSV files.

Usage:
    python aria_audio_diarization.py --input_dir /path/to/aria/raw --output_dir /path/to/output
    
Requirements:
    pip install pyannote.audio torch torchaudio pandas numpy librosa soundfile
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torchaudio
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import json
import csv
import math

try:
    from pyannote.audio import Pipeline
    from pyannote.core import Segment, Annotation
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("Warning: pyannote.audio not available. Install with: pip install pyannote.audio")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AriaSpeakerDiarization:
    """
    Speaker diarization for Aria dataset speech CSV files
    """
    
    def __init__(self, use_auth_token: Optional[str] = None):
        """
        Initialize the diarization pipeline
        
        Args:
            use_auth_token: HuggingFace auth token for pyannote models (optional)
        """
        self.pipeline = None
        self.use_auth_token = use_auth_token
        
        if PYANNOTE_AVAILABLE:
            self._initialize_pipeline()
        else:
            logger.warning("Pyannote.audio not available. Will use fallback speaker assignment.")
    
    def _initialize_pipeline(self):
        """Initialize the pyannote speaker diarization pipeline"""
        try:
            # Initialize the speaker diarization pipeline
            # You may need to authenticate with HuggingFace for some models
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.use_auth_token
            )
            logger.info("Pyannote speaker diarization pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pyannote pipeline: {e}")
            logger.info("Using fallback speaker assignment method")
            self.pipeline = None
    
    def nanoseconds_to_seconds(self, ns: int) -> float:
        """Convert nanoseconds to seconds"""
        return ns / 1_000_000_000
    
    def seconds_to_nanoseconds(self, s: float) -> int:
        """Convert seconds to nanoseconds"""
        return int(s * 1_000_000_000)
    
    def compute_frame_range(self, start_time_s: float, end_time_s: float = None, fps: float = 30.0):
        """Compute an inclusive list of frame indices covering [start, end).
        
        If `end_time_s` is None or not finite, returns a single-frame list at the
        start frame. Based on Ego4D annotation processing.
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
    
    def load_speech_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load speech.csv file and parse timestamps
        
        Args:
            csv_path: Path to speech.csv file
            
        Returns:
            DataFrame with parsed speech data
        """
        try:
            df = pd.read_csv(csv_path)
            
            # Validate required columns
            required_cols = ['startTime_ns', 'endTime_ns', 'written', 'confidence']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV missing required columns: {required_cols}")
            
            # Convert nanoseconds to seconds for processing
            df['start_sec'] = df['startTime_ns'].apply(self.nanoseconds_to_seconds)
            df['end_sec'] = df['endTime_ns'].apply(self.nanoseconds_to_seconds)
            
            logger.info(f"Loaded {len(df)} speech segments from {csv_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV {csv_path}: {e}")
            return None
    
    def extract_audio_for_diarization(self, recording_dir: str, output_audio_path: str) -> bool:
        """
        Extract audio from recording.vrs file for diarization
        
        Args:
            recording_dir: Directory containing recording.vrs
            output_audio_path: Output path for extracted audio
            
        Returns:
            True if successful, False otherwise
        """
        vrs_path = os.path.join(recording_dir, "recording.vrs")
        
        if not os.path.exists(vrs_path):
            logger.error(f"VRS file not found: {vrs_path}")
            return False
        
        try:
            # Use ffmpeg to extract audio from VRS file
            import subprocess
            
            cmd = [
                "ffmpeg", "-i", vrs_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM 16-bit
                "-ar", "16000",  # 16kHz sample rate
                "-ac", "1",  # Mono
                "-y",  # Overwrite output
                output_audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Audio extracted successfully: {output_audio_path}")
                return True
            else:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return False
    
    def perform_diarization(self, audio_path: str) -> Optional[Annotation]:
        """
        Perform speaker diarization on audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Pyannote Annotation object with speaker segments
        """
        if not self.pipeline:
            logger.warning("No diarization pipeline available")
            return None
        
        try:
            # Load audio and perform diarization
            diarization = self.pipeline(audio_path)
            logger.info(f"Diarization completed for {audio_path}")
            return diarization
            
        except Exception as e:
            logger.error(f"Error during diarization: {e}")
            return None
    
    def assign_speakers_to_segments(self, df: pd.DataFrame, diarization: Optional[Annotation]) -> pd.DataFrame:
        """
        Assign speaker labels to speech segments
        
        Args:
            df: DataFrame with speech segments
            diarization: Pyannote diarization results
            
        Returns:
            DataFrame with speaker_label column added
        """
        if diarization is None:
            # Fallback: assign speakers based on simple heuristics
            return self._fallback_speaker_assignment(df)
        
        # Create speaker_label column
        df['speaker_label'] = 'unknown'
        
        for idx, row in df.iterrows():
            start_time = row['start_sec']
            end_time = row['end_sec']
            
            # Find overlapping speakers in diarization
            segment = Segment(start_time, end_time)
            speakers = []
            
            for speech_segment, _, speaker in diarization.itertracks(yield_label=True):
                if segment.overlaps(speech_segment):
                    overlap_duration = segment & speech_segment
                    if overlap_duration.duration > 0.1:  # At least 100ms overlap
                        speakers.append((speaker, overlap_duration.duration))
            
            if speakers:
                # Assign speaker with longest overlap
                best_speaker = max(speakers, key=lambda x: x[1])[0]
                df.at[idx, 'speaker_label'] = best_speaker
            else:
                df.at[idx, 'speaker_label'] = 'speaker_unknown'
        
        return df
    
    def _fallback_speaker_assignment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fallback speaker assignment using simple heuristics
        
        Args:
            df: DataFrame with speech segments
            
        Returns:
            DataFrame with speaker_label column added
        """
        logger.info("Using fallback speaker assignment")
        
        # Simple heuristic: assign speakers based on time gaps
        df['speaker_label'] = 'speaker_A'
        
        if len(df) > 1:
            # Calculate time gaps between segments
            df['time_gap'] = df['start_sec'] - df['end_sec'].shift(1)
            
            current_speaker = 'speaker_A'
            speaker_map = {'speaker_A': 'speaker_B', 'speaker_B': 'speaker_A'}
            
            for idx, row in df.iterrows():
                if idx == 0:
                    df.at[idx, 'speaker_label'] = current_speaker
                else:
                    # If there's a significant gap (>2 seconds), potentially switch speaker
                    if row['time_gap'] > 2.0:
                        # Switch speaker with some probability based on gap duration
                        if row['time_gap'] > 5.0:
                            current_speaker = speaker_map[current_speaker]
                    
                    df.at[idx, 'speaker_label'] = current_speaker
            
            # Clean up temporary column
            df.drop('time_gap', axis=1, inplace=True)
        
        return df
    
    def save_updated_csv(self, df: pd.DataFrame, output_path: str):
        """
        Save updated CSV with speaker labels
        
        Args:
            df: DataFrame with speaker labels
            output_path: Output CSV path
        """
        try:
            # Remove temporary columns
            columns_to_save = ['startTime_ns', 'endTime_ns', 'written', 'confidence', 'speaker_label']
            df_output = df[columns_to_save].copy()
            
            df_output.to_csv(output_path, index=False)
            logger.info(f"Updated CSV saved: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving CSV: {e}")
    
    def generate_transcript_csv(self, df: pd.DataFrame, recording_name: str, transcript_output_dir: str):
        """
        Generate transcript CSV in Ego4D format
        
        Args:
            df: DataFrame with speech segments and speaker labels
            recording_name: Name of the recording
            transcript_output_dir: Output directory for transcript files
        """
        try:
            transcript_path = os.path.join(transcript_output_dir, 'ground_truth_transcriptions_with_frames.csv')
            os.makedirs(transcript_output_dir, exist_ok=True)
            
            transcription_data = []
            
            for idx, row in df.iterrows():
                # Split transcription into words - distribute time evenly across words
                transcription_text = row['written']
                start_time = row['start_sec']
                end_time = row['end_sec']
                speaker_id = row['speaker_label']
                
                words = transcription_text.split()
                if words:
                    time_per_word = (end_time - start_time) / len(words)
                    
                    for i, word in enumerate(words):
                        word_start = start_time + (i * time_per_word)
                        word_end = start_time + ((i + 1) * time_per_word)
                        
                        # Compute frame range for this word (assuming 30 fps)
                        frame_list = self.compute_frame_range(word_start, word_end, fps=30.0)
                        
                        transcription_data.append({
                            'conversation_id': recording_name,  # Using recording name as conversation_id
                            'endTime': round(word_end, 2),
                            'speaker_id': speaker_id,
                            'startTime': round(word_start, 2),
                            'word': word,
                            'frame': json.dumps(frame_list, separators=(",", ":"))  # Store as compact JSON array
                        })
            
            # Sort transcription data by frame (use the first frame in the frame list for sorting)
            if transcription_data:
                def get_first_frame(item):
                    frame_list = json.loads(item['frame'])
                    return frame_list[0] if frame_list else 0
                
                transcription_data.sort(key=get_first_frame)
            
            # Write or append to CSV
            file_exists = os.path.exists(transcript_path)
            
            if transcription_data:
                with open(transcript_path, 'a' if file_exists else 'w', newline='') as f:
                    fieldnames = ['conversation_id', 'endTime', 'speaker_id', 'startTime', 'word', 'frame']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    
                    if not file_exists:
                        writer.writeheader()
                    
                    writer.writerows(transcription_data)
                
                logger.info(f"{'Appended to' if file_exists else 'Created'} transcript file: {transcript_path}")
                return len(transcription_data)
            
        except Exception as e:
            logger.error(f"Error generating transcript CSV: {e}")
            return 0
        
        return 0
    
    def process_single_recording(self, recording_dir: str, output_dir: str) -> bool:
        """
        Process a single recording directory
        
        Args:
            recording_dir: Path to recording directory (contains speech.csv and recording.vrs)
            output_dir: Output directory for processed files
            
        Returns:
            True if successful, False otherwise
        """
        recording_name = os.path.basename(recording_dir)
        logger.info(f"Processing recording: {recording_name}")
        
        # Paths
        speech_csv_path = os.path.join(recording_dir, "speech.csv")
        output_recording_dir = os.path.join(output_dir, recording_name)
        output_csv_path = os.path.join(output_recording_dir, "speech_with_speakers.csv")
        temp_audio_path = os.path.join(output_recording_dir, "temp_audio.wav")
        
        # Create output directory
        os.makedirs(output_recording_dir, exist_ok=True)
        
        # Check if speech.csv exists
        if not os.path.exists(speech_csv_path):
            logger.warning(f"No speech.csv found in {recording_dir}")
            return False
        
        # Load speech data
        df = self.load_speech_csv(speech_csv_path)
        if df is None:
            return False
        
        # Extract audio for diarization
        diarization = None
        if self.pipeline:
            if self.extract_audio_for_diarization(recording_dir, temp_audio_path):
                diarization = self.perform_diarization(temp_audio_path)
                
                # Clean up temporary audio file
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
        
        # Assign speakers
        df_with_speakers = self.assign_speakers_to_segments(df, diarization)
        
        # Save updated CSV
        self.save_updated_csv(df_with_speakers, output_csv_path)
        
        # Generate transcript CSV in Ego4D format
        transcript_output_dir = "/mas/robots/prg-aria/transcript"
        transcript_count = self.generate_transcript_csv(df_with_speakers, recording_name, transcript_output_dir)
        logger.info(f"Generated {transcript_count} transcript word entries for {recording_name}")
        
        return True
    
    def process_aria_dataset(self, input_dir: str, output_dir: str):
        """
        Process entire Aria dataset directory structure
        
        Args:
            input_dir: Root directory containing Aria recordings
            output_dir: Output directory for processed files
        """
        logger.info(f"Processing Aria dataset from: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        processed_count = 0
        failed_count = 0
        
        # Find all recording directories
        for item in os.listdir(input_dir):
            item_path = os.path.join(input_dir, item)
            
            # Skip files and non-recording directories
            if not os.path.isdir(item_path):
                continue
            
            if not item.startswith('loc'):
                continue
            
            # Check if this is a recording directory (contains speech.csv)
            speech_csv = os.path.join(item_path, "speech.csv")
            if os.path.exists(speech_csv):
                try:
                    success = self.process_single_recording(item_path, output_dir)
                    if success:
                        processed_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.error(f"Error processing {item}: {e}")
                    failed_count += 1
        
        logger.info(f"Processing complete. Processed: {processed_count}, Failed: {failed_count}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Aria Audio Diarization and Speaker Labeling")
    parser.add_argument("--input_dir", required=True, help="Input directory containing Aria recordings")
    parser.add_argument("--output_dir", required=True, help="Output directory for processed files")
    parser.add_argument("--auth_token", help="HuggingFace auth token for pyannote models")
    parser.add_argument("--single_recording", help="Process single recording directory instead of entire dataset")
    
    args = parser.parse_args()
    
    # Initialize diarization system
    diarizer = AriaSpeakerDiarization(use_auth_token=args.auth_token)
    
    if args.single_recording:
        # Process single recording
        success = diarizer.process_single_recording(args.single_recording, args.output_dir)
        if success:
            logger.info("Single recording processed successfully")
        else:
            logger.error("Failed to process single recording")
            sys.exit(1)
    else:
        # Process entire dataset
        diarizer.process_aria_dataset(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
