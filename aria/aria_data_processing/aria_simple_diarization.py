#!/usr/bin/env python3
"""
Simple Audio Diarization for Aria Dataset
Updated version with latest libraries and fixed errors
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, List
import json
import csv
import math
import subprocess
from pathlib import Path

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Pyannote imports
try:
    print("Loading torch...")
    import torch
    print("Loading pyannote...")
    from pyannote.audio import Pipeline
    from pyannote.core import Segment, Annotation
    PYANNOTE_AVAILABLE = True
    print("All imports successful")
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("Error: pyannote.audio not available.")
    print("Install with: pip install pyannote.audio torch torchaudio")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AriaDiarization:
    """Simple speaker diarization for Aria dataset"""
    
    def __init__(self, auth_token: str):
        """
        Initialize with HuggingFace auth token
        
        Args:
            auth_token: HuggingFace access token for pyannote models
        """
        self.auth_token = auth_token
        self.pipeline = None
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the pyannote speaker diarization pipeline"""
        try:
            logger.info("Initializing pyannote.audio pipeline...")
            logger.info("This may take a few minutes on first run (downloading models)...")
            
            # Use the latest pipeline version
            logger.info("Loading pipeline from HuggingFace...")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.auth_token
            )
            logger.info("Pipeline loaded successfully")
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Moving pipeline to {device}...")
            self.pipeline.to(device)
            logger.info(f"Using {device} for diarization")
            logger.info("Pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def nanoseconds_to_seconds(self, ns: int) -> float:
        """Convert nanoseconds to seconds"""
        return ns / 1_000_000_000
    
    def compute_frame_range(self, start_time_s: float, end_time_s: float = None, fps: float = 20.0) -> List[int]:
        """
        Compute frame range for Aria (20 fps)
        
        Args:
            start_time_s: Start time in seconds
            end_time_s: End time in seconds (optional)
            fps: Frames per second (default 20 for Aria)
            
        Returns:
            List of frame indices
        """
        # Check for NaN or invalid values using numpy
        if pd.isna(start_time_s) or not np.isfinite(start_time_s):
            return []
        
        start_frame = math.floor(start_time_s * fps)
        
        if end_time_s is None or pd.isna(end_time_s) or not np.isfinite(end_time_s):
            return [start_frame]
        
        end_frame_inclusive = max(start_frame, math.ceil(end_time_s * fps) - 1)
        return list(range(start_frame, end_frame_inclusive + 1))
    
    def load_speech_csv(self, csv_path: str) -> Optional[pd.DataFrame]:
        """Load and process speech.csv with timestamps"""
        try:
            df = pd.read_csv(csv_path)
            
            # Validate required columns
            required_cols = ['startTime_ns', 'endTime_ns', 'written', 'confidence']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"CSV missing required columns: {missing_cols}")
            
            # Convert nanoseconds to seconds
            df['start_sec'] = df['startTime_ns'].apply(self.nanoseconds_to_seconds)
            df['end_sec'] = df['endTime_ns'].apply(self.nanoseconds_to_seconds)
            
            # Normalize timestamps to start from 0
            if len(df) > 0:
                min_start = df['start_sec'].min()
                df['start_sec_normalized'] = df['start_sec'] - min_start
                df['end_sec_normalized'] = df['end_sec'] - min_start
                
                logger.info(f"Loaded {len(df)} segments from {csv_path}")
                logger.info(f"Time range: 0.0s to {df['end_sec_normalized'].max():.2f}s")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV {csv_path}: {e}")
            return None
    
    def extract_audio_from_mp4(self, mp4_path: str, output_audio_path: str) -> bool:
        """
        Extract audio from MP4 file using ffmpeg
        
        Args:
            mp4_path: Path to input MP4 file
            output_audio_path: Path for output audio file
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(mp4_path):
            logger.error(f"MP4 file not found: {mp4_path}")
            return False
        
        try:
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
            
            # FFmpeg command for audio extraction
            cmd = [
                "ffmpeg", "-i", mp4_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM 16-bit
                "-ar", "16000",  # 16kHz sample rate
                "-ac", "1",  # Mono
                "-y",  # Overwrite output
                output_audio_path
            ]
            
            # Run ffmpeg
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Audio extracted successfully to {output_audio_path}")
                return True
            else:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return False
    
    def run_diarization(self, audio_path: str) -> Optional[Annotation]:
        """
        Run speaker diarization on audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Diarization annotation object or None if failed
        """
        try:
            logger.info(f"Running diarization on {audio_path}")
            
            # Run the pipeline with speaker constraints
            diarization = self.pipeline(audio_path, min_speakers=1, max_speakers=2)
            
            # Get statistics
            num_speakers = len(diarization.labels())
            
            # Calculate total duration using itertracks
            total_speech_duration = 0
            num_segments = 0
            for segment, _, _ in diarization.itertracks(yield_label=True):
                total_speech_duration += segment.duration
                num_segments += 1
            
            logger.info(f"Diarization complete:")
            logger.info(f"  - Speakers detected: {num_speakers}")
            logger.info(f"  - Total speech duration: {total_speech_duration:.2f}s")
            logger.info(f"  - Number of segments: {num_segments}")
            
            return diarization
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return None
    
    def assign_speakers_to_segments(self, df: pd.DataFrame, diarization: Annotation) -> pd.DataFrame:
        """
        Assign speaker labels to transcript segments based on diarization
        
        Args:
            df: DataFrame with transcript segments
            diarization: Pyannote diarization results
            
        Returns:
            DataFrame with added speaker_label column
        """
        # Initialize speaker column
        df['speaker_label'] = 'person_unknown'
        
        if not diarization or len(diarization.labels()) == 0:
            logger.warning("No speakers detected in diarization")
            return df
        
        logger.info(f"Assigning {len(diarization.labels())} speakers to {len(df)} segments")
        
        # Process each transcript segment
        for idx, row in df.iterrows():
            start_time = row['start_sec_normalized']
            end_time = row['end_sec_normalized']
            segment = Segment(start_time, end_time)
            
            # Find overlapping speakers
            overlaps = []
            
            for speaker_segment, _, speaker in diarization.itertracks(yield_label=True):
                # Calculate overlap
                overlap_start = max(segment.start, speaker_segment.start)
                overlap_end = min(segment.end, speaker_segment.end)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                # Minimum 20ms overlap required (more lenient)
                if overlap_duration > 0.02:
                    overlaps.append((speaker, overlap_duration))
            
            # Assign speaker with maximum overlap
            if overlaps:
                best_speaker = max(overlaps, key=lambda x: x[1])[0]
                # Convert speaker ID to readable format
                if isinstance(best_speaker, str) and 'SPEAKER_' in best_speaker:
                    speaker_num = int(best_speaker.replace('SPEAKER_', '')) + 1
                    df.at[idx, 'speaker_label'] = f'person_{speaker_num}'
                else:
                    df.at[idx, 'speaker_label'] = f'person_{best_speaker}'
        
        # Log speaker distribution
        speaker_counts = df['speaker_label'].value_counts()
        logger.info("Speaker assignment complete:")
        for speaker, count in speaker_counts.items():
            logger.info(f"  - {speaker}: {count} segments")
        
        return df
    
    def generate_transcript_csv(self, df: pd.DataFrame, recording_name: str, output_dir: str, 
                               ground_truth_path: str = None) -> int:
        """
        Generate transcript CSV in Aria format
        
        Args:
            df: DataFrame with speaker-labeled segments
            recording_name: Name of the recording
            output_dir: Output directory for transcript
            
        Returns:
            Number of transcript entries generated
        """
        transcript_path = os.path.join(output_dir, 'transcripts.csv')
        os.makedirs(output_dir, exist_ok=True)
        
        transcription_data = []
        
        for _, row in df.iterrows():
            text = row['written']
            start_time = row['start_sec_normalized']
            end_time = row['end_sec_normalized']
            speaker_id = row['speaker_label']
            
            # Split into words and assign timestamps
            words = text.split()
            if words:
                time_per_word = (end_time - start_time) / len(words)
                
                for i, word in enumerate(words):
                    word_start = start_time + (i * time_per_word)
                    word_end = start_time + ((i + 1) * time_per_word)
                    
                    # Compute frame range
                    frames = self.compute_frame_range(word_start, word_end, fps=20.0)
                    
                    transcription_data.append({
                        'conversation_id': recording_name,
                        'startTime': round(word_start, 3),
                        'endTime': round(word_end, 3),
                        'speaker_id': speaker_id,
                        'word': word,
                        'frames': json.dumps(frames) if frames else '[]'
                    })
        
        # Sort by start time
        transcription_data.sort(key=lambda x: x['startTime'])
        
        # Write to CSV
        if transcription_data:
            file_exists = os.path.exists(transcript_path)
            with open(transcript_path, 'a' if file_exists else 'w', newline='') as f:
                fieldnames = ['conversation_id', 'startTime', 'endTime', 'speaker_id', 'word', 'frames']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerows(transcription_data)
            
            logger.info(f"Generated {len(transcription_data)} transcript entries")
            
            # Also append to ground truth CSV if path provided
            if ground_truth_path:
                try:
                    os.makedirs(os.path.dirname(ground_truth_path), exist_ok=True)
                    file_exists = os.path.exists(ground_truth_path)
                    
                    with open(ground_truth_path, 'a' if file_exists else 'w', newline='') as f:
                        fieldnames = ['conversation_id', 'startTime', 'endTime', 'speaker_id', 'word', 'frames']
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        
                        if not file_exists:
                            writer.writeheader()
                        
                        writer.writerows(transcription_data)
                    
                    logger.info(f"Appended {len(transcription_data)} entries to ground truth CSV: {ground_truth_path}")
                    
                except Exception as e:
                    logger.warning(f"Failed to write to ground truth CSV: {e}")
        
        return len(transcription_data)
    
    def process_recording(self, 
                         recording_dir: str, 
                         mp4_path: str,
                         output_dir: str) -> bool:
        """
        Process a single recording
        
        Args:
            recording_dir: Directory containing speech.csv
            mp4_path: Path to MP4 file
            output_dir: Output directory
            
        Returns:
            True if successful, False otherwise
        """
        recording_name = os.path.basename(recording_dir)
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {recording_name}")
        logger.info(f"{'='*60}")
        
        # Setup paths
        speech_csv_path = os.path.join(recording_dir, "speech.csv")
        output_recording_dir = os.path.join(output_dir, recording_name)
        output_csv_path = os.path.join(output_recording_dir, "speech_with_speakers.csv")
        temp_audio_path = os.path.join(output_recording_dir, "audio.wav")
        
        # Check if speech.csv exists
        if not os.path.exists(speech_csv_path):
            logger.warning(f"No speech.csv found in {recording_dir}")
            return False
        
        try:
            # Step 1: Load speech data
            df = self.load_speech_csv(speech_csv_path)
            if df is None or len(df) == 0:
                logger.error("Failed to load speech data")
                return False
            
            # Step 2: Extract audio
            if not self.extract_audio_from_mp4(mp4_path, temp_audio_path):
                logger.error("Failed to extract audio")
                return False
            
            # Step 3: Run diarization
            diarization = self.run_diarization(temp_audio_path)
            if diarization is None:
                logger.error("Diarization failed")
                return False
            
            # Step 4: Assign speakers
            df_with_speakers = self.assign_speakers_to_segments(df, diarization)
            
            # Step 5: Save results
            os.makedirs(output_recording_dir, exist_ok=True)
            df_with_speakers.to_csv(output_csv_path, index=False)
            logger.info(f"Saved results to {output_csv_path}")
            
            # Step 6: Generate transcript
            transcript_dir = os.path.join(output_dir, "transcripts")
            ground_truth_path = "/mas/robots/prg-aria/transcript/ground_truth_transciptions.csv"
            self.generate_transcript_csv(df_with_speakers, recording_name, transcript_dir, ground_truth_path)
            
            # Clean up temp audio
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                logger.info("Cleaned up temporary files")
            
            logger.info(f"✓ Successfully processed {recording_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {recording_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_dataset(self, 
                       input_dir: str, 
                       mp4_dir: str,
                       output_dir: str):
        """
        Process entire dataset
        
        Args:
            input_dir: Directory containing recording folders with speech.csv
            mp4_dir: Directory containing MP4 files
            output_dir: Output directory for results
        """
        logger.info(f"\nStarting batch processing")
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"MP4 directory: {mp4_dir}")
        logger.info(f"Output directory: {output_dir}")
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all recordings
        recordings = []
        for item in sorted(os.listdir(input_dir)):
            item_path = os.path.join(input_dir, item)
            if os.path.isdir(item_path):
                speech_csv = os.path.join(item_path, "speech.csv")
                if os.path.exists(speech_csv):
                    mp4_path = os.path.join(mp4_dir, f"{item}.mp4")
                    if os.path.exists(mp4_path):
                        recordings.append((item_path, mp4_path))
                    else:
                        logger.warning(f"MP4 not found for {item}: {mp4_path}")
        
        logger.info(f"Found {len(recordings)} recordings to process")
        
        # Process recordings
        successful = 0
        failed = 0
        
        for i, (recording_dir, mp4_path) in enumerate(recordings, 1):
            logger.info(f"\n[{i}/{len(recordings)}] Processing...")
            
            try:
                if self.process_recording(recording_dir, mp4_path, output_dir):
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                failed += 1
        
        # Final summary
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total: {len(recordings)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {successful/len(recordings)*100:.1f}%")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Simple Audio Diarization for Aria Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--input_dir", 
        required=True, 
        help="Directory containing recording folders with speech.csv files"
    )
    
    parser.add_argument(
        "--mp4_dir", 
        required=True,
        help="Directory containing MP4 files"
    )
    
    parser.add_argument(
        "--output_dir", 
        required=True, 
        help="Output directory for processed files"
    )
    
    parser.add_argument(
        "--auth_token", 
        required=True,
        help="HuggingFace auth token for pyannote models"
    )
    
    parser.add_argument(
        "--single", 
        help="Process only a single recording (provide recording name)"
    )
    
    args = parser.parse_args()
    
    # Initialize diarization system
    try:
        diarizer = AriaDiarization(auth_token=args.auth_token)
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        sys.exit(1)
    
    # Process single or batch
    if args.single:
        recording_dir = os.path.join(args.input_dir, args.single)
        mp4_path = os.path.join(args.mp4_dir, f"{args.single}.mp4")
        
        if not os.path.exists(recording_dir):
            logger.error(f"Recording directory not found: {recording_dir}")
            sys.exit(1)
        
        if not os.path.exists(mp4_path):
            logger.error(f"MP4 file not found: {mp4_path}")
            sys.exit(1)
        
        success = diarizer.process_recording(recording_dir, mp4_path, args.output_dir)
        sys.exit(0 if success else 1)
    else:
        diarizer.process_dataset(args.input_dir, args.mp4_dir, args.output_dir)


if __name__ == "__main__":
    main()