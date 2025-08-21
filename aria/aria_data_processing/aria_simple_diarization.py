#!/usr/bin/env python3
"""
Simple Aria Audio Diarization using pyannote.audio

Just uses pyannote.audio pipeline directly - no complex voice analysis needed.
"""

import os
import sys
import argparse
import pandas as pd
import logging
from typing import Optional
import json
import csv
import math
import subprocess

try:
    from pyannote.audio import Pipeline
    from pyannote.core import Segment, Annotation
    import torch
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("Error: pyannote.audio not available. Install with: pip install pyannote.audio")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleAriaDiarization:
    """
    Simple speaker diarization for Aria dataset using only pyannote.audio
    """
    
    def __init__(self, auth_token: str):
        """
        Initialize with HuggingFace auth token
        
        Args:
            auth_token: HuggingFace access token
        """
        self.auth_token = auth_token
        self.pipeline = None
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the pyannote speaker diarization pipeline"""
        try:
            logger.info("Initializing pyannote.audio pipeline...")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.auth_token
            )
            
            # Send to GPU if available
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.pipeline.to(device)
                logger.info("Using GPU for diarization")
            else:
                logger.info("Using CPU for diarization")
                
            logger.info("Pyannote pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pyannote pipeline: {e}")
            raise
    
    def nanoseconds_to_seconds(self, ns: int) -> float:
        """Convert nanoseconds to seconds"""
        return ns / 1_000_000_000
    
    def compute_frame_range(self, start_time_s: float, end_time_s: float = None, fps: float = 20.0):
        """Compute frame range for Aria (20 fps)"""
        if not pd.isna(start_time_s) and pd.isfinite(start_time_s):
            start_frame = math.floor(start_time_s * fps)
        else:
            return []

        if end_time_s is None or not pd.isfinite(end_time_s):
            return [start_frame]

        end_frame_inclusive = max(start_frame, math.ceil(end_time_s * fps) - 1)
        if end_frame_inclusive < start_frame:
            end_frame_inclusive = start_frame

        return list(range(start_frame, end_frame_inclusive + 1))
    
    def load_speech_csv(self, csv_path: str) -> pd.DataFrame:
        """Load and normalize speech.csv timestamps"""
        try:
            df = pd.read_csv(csv_path)
            
            # Validate required columns
            required_cols = ['startTime_ns', 'endTime_ns', 'written', 'confidence']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV missing required columns: {required_cols}")
            
            # Convert nanoseconds to seconds
            df['start_sec'] = df['startTime_ns'].apply(self.nanoseconds_to_seconds)
            df['end_sec'] = df['endTime_ns'].apply(self.nanoseconds_to_seconds)
            
            # Normalize timestamps to start from 0
            if len(df) > 0:
                min_start_time = df['start_sec'].min()
                df['start_sec_normalized'] = df['start_sec'] - min_start_time
                df['end_sec_normalized'] = df['end_sec'] - min_start_time
                
                logger.info(f"Loaded {len(df)} segments. Normalized timestamps to start from 0.0s")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV {csv_path}: {e}")
            return None
    
    def extract_audio_from_mp4(self, recording_dir: str, output_audio_path: str) -> bool:
        """Extract audio from MP4 file"""
        recording_name = os.path.basename(recording_dir)
        dataset_dir = "/mas/robots/prg-aria/dataset"
        mp4_path = os.path.join(dataset_dir, f"{recording_name}.mp4")
        
        if not os.path.exists(mp4_path):
            logger.error(f"MP4 file not found: {mp4_path}")
            return False
        
        try:
            cmd = [
                "ffmpeg", "-i", mp4_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                "-y", output_audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Audio extracted from MP4: {mp4_path}")
                return True
            else:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return False
    
    def run_diarization(self, audio_path: str) -> Optional[Annotation]:
        """Run pyannote diarization on audio file"""
        try:
            logger.info(f"Running pyannote diarization on {audio_path}")
            diarization = self.pipeline(audio_path)
            
            # Log results
            num_speakers = len(diarization.labels())
            total_duration = sum(segment.duration for segment, _, _ in diarization.itertracks())
            logger.info(f"Detected {num_speakers} speakers, {total_duration:.2f}s total speech")
            
            return diarization
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return None
    
    def assign_speakers_from_diarization(self, df: pd.DataFrame, diarization: Annotation) -> pd.DataFrame:
        """Assign speaker labels based on pyannote diarization results"""
        df['speaker_label'] = 'person_unknown'
        
        for idx, row in df.iterrows():
            start_time = row['start_sec_normalized']
            end_time = row['end_sec_normalized']
            
            # Find overlapping speakers
            segment = Segment(start_time, end_time)
            speakers = []
            
            for speech_segment, _, speaker in diarization.itertracks(yield_label=True):
                if segment.overlaps(speech_segment):
                    overlap_duration = segment & speech_segment
                    if overlap_duration.duration > 0.05:  # At least 50ms overlap
                        speakers.append((speaker, overlap_duration.duration))
            
            if speakers:
                # Assign speaker with longest overlap
                best_speaker = max(speakers, key=lambda x: x[1])[0]
                # Convert pyannote speaker names to person_X format
                speaker_num = best_speaker.replace('SPEAKER_', '')
                df.at[idx, 'speaker_label'] = f'person_{int(speaker_num) + 1}'
            
        # Log speaker distribution
        speaker_counts = df['speaker_label'].value_counts()
        logger.info(f"Speaker assignment result: {dict(speaker_counts)}")
        
        return df
    
    def generate_transcript_csv(self, df: pd.DataFrame, recording_name: str, transcript_dir: str):
        """Generate transcript CSV in Ego4D format"""
        transcript_path = os.path.join(transcript_dir, 'ground_truth_transcriptions_with_frames.csv')
        os.makedirs(transcript_dir, exist_ok=True)
        
        transcription_data = []
        
        for idx, row in df.iterrows():
            transcription_text = row['written']
            start_time = row['start_sec_normalized']
            end_time = row['end_sec_normalized']
            speaker_id = row['speaker_label']
            
            words = transcription_text.split()
            if words:
                time_per_word = (end_time - start_time) / len(words)
                
                for i, word in enumerate(words):
                    word_start = start_time + (i * time_per_word)
                    word_end = start_time + ((i + 1) * time_per_word)
                    
                    # Compute frame range (20 fps for Aria)
                    frame_list = self.compute_frame_range(word_start, word_end, fps=20.0)
                    
                    transcription_data.append({
                        'conversation_id': recording_name,
                        'endTime': round(word_end, 2),
                        'speaker_id': speaker_id,
                        'startTime': round(word_start, 2),
                        'word': word,
                        'frame': json.dumps(frame_list, separators=(",", ":"))
                    })
        
        # Sort by start time
        transcription_data.sort(key=lambda x: x['startTime'])
        
        # Write or append to CSV
        file_exists = os.path.exists(transcript_path)
        
        if transcription_data:
            with open(transcript_path, 'a' if file_exists else 'w', newline='') as f:
                fieldnames = ['conversation_id', 'endTime', 'speaker_id', 'startTime', 'word', 'frame']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerows(transcription_data)
            
            logger.info(f"Generated {len(transcription_data)} transcript entries for {recording_name}")
        
        return len(transcription_data)
    
    def process_single_recording(self, recording_dir: str, output_dir: str) -> bool:
        """Process a single recording using pyannote.audio"""
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
        
        # Extract audio from MP4
        if not self.extract_audio_from_mp4(recording_dir, temp_audio_path):
            logger.error(f"Failed to extract audio for {recording_name}")
            return False
        
        # Run pyannote diarization
        diarization = self.run_diarization(temp_audio_path)
        if diarization is None:
            logger.error(f"Diarization failed for {recording_name}")
            return False
        
        # Assign speakers based on diarization
        df_with_speakers = self.assign_speakers_from_diarization(df, diarization)
        
        # Save updated CSV
        df_with_speakers.to_csv(output_csv_path, index=False)
        logger.info(f"Saved speaker-labeled CSV: {output_csv_path}")
        
        # Generate transcript
        transcript_dir = "/mas/robots/prg-aria/transcript"
        transcript_count = self.generate_transcript_csv(df_with_speakers, recording_name, transcript_dir)
        
        # Clean up temp audio
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        return True
    
    def process_aria_dataset(self, input_dir: str, output_dir: str):
        """Process entire Aria dataset"""
        logger.info(f"Processing Aria dataset from: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs("/mas/robots/prg-aria/transcript", exist_ok=True)
        
        processed_count = 0
        failed_count = 0
        
        # Process all recording directories
        for item in os.listdir(input_dir):
            item_path = os.path.join(input_dir, item)
            
            if not os.path.isdir(item_path) or not item.startswith('loc'):
                continue
            
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
    parser = argparse.ArgumentParser(description="Simple Aria Diarization using pyannote.audio")
    parser.add_argument("--input_dir", required=True, help="Input directory containing Aria recordings")
    parser.add_argument("--output_dir", required=True, help="Output directory for processed files")
    parser.add_argument("--auth_token", required=True, help="HuggingFace auth token for pyannote models")
    parser.add_argument("--single_recording", help="Process single recording directory")
    
    args = parser.parse_args()
    
    # Initialize diarization system
    try:
        diarizer = SimpleAriaDiarization(auth_token=args.auth_token)
    except Exception as e:
        logger.error(f"Failed to initialize diarizer: {e}")
        sys.exit(1)
    
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
