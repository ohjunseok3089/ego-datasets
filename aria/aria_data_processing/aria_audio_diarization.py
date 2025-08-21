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
import librosa
from scipy import signal
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
    
    def compute_frame_range(self, start_time_s: float, end_time_s: float = None, fps: float = 20.0):
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
    
    def extract_voice_features(self, audio_path: str, start_time: float, end_time: float) -> Dict:
        """
        Extract voice characteristics (pitch, tone, timbre) from audio segment
        
        Args:
            audio_path: Path to audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Dictionary of voice features
        """
        try:
            # Load audio segment
            y, sr = librosa.load(audio_path, sr=16000, offset=start_time, duration=end_time-start_time)
            
            if len(y) < 100:  # Too short segment
                return None
            
            # Extract features
            features = {}
            
            # 1. Pitch (F0) features
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
                features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
                features['pitch_range'] = 0
            
            # 2. Spectral features (timbre)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfccs, axis=1)
            features['mfcc_std'] = np.std(mfccs, axis=1)
            
            # 3. Spectral centroid (brightness)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            # 4. Zero crossing rate (voice texture)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # 5. RMS energy (loudness)
            rms = librosa.feature.rms(y=y)[0]
            features['rms_mean'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
            
            return features
            
        except Exception as e:
            logger.warning(f"Failed to extract voice features: {e}")
            return None
    
    def cluster_speakers_by_voice(self, df: pd.DataFrame, audio_path: str) -> pd.DataFrame:
        """
        Cluster speakers based on voice characteristics instead of time gaps
        
        Args:
            df: DataFrame with speech segments and normalized timestamps
            audio_path: Path to audio file for feature extraction
            
        Returns:
            DataFrame with improved speaker labels based on voice clustering
        """
        logger.info("Clustering speakers based on voice characteristics...")
        
        if len(df) <= 1:
            df['speaker_label'] = 'person_1'
            return df
        
        # Extract features for each segment
        features_list = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            features = self.extract_voice_features(
                audio_path, 
                row['start_sec_normalized'], 
                row['end_sec_normalized']
            )
            
            if features is not None:
                # Flatten MFCC features
                feature_vector = [
                    features['pitch_mean'],
                    features['pitch_std'], 
                    features['pitch_range'],
                    features['spectral_centroid_mean'],
                    features['spectral_centroid_std'],
                    features['zcr_mean'],
                    features['zcr_std'],
                    features['rms_mean'],
                    features['rms_std']
                ]
                # Add MFCC means and stds
                feature_vector.extend(features['mfcc_mean'])
                feature_vector.extend(features['mfcc_std'])
                
                features_list.append(feature_vector)
                valid_indices.append(idx)
        
        if len(features_list) < 2:
            logger.warning("Not enough valid voice features for clustering")
            logger.warning("Assigning all segments to person_1 (single speaker detected)")
            df['speaker_label'] = 'person_1'
            return df
        
        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_list)
        
        # Determine optimal number of clusters (speakers)
        max_clusters = min(4, len(features_list))  # Max 4 speakers
        best_score = -1
        best_n_clusters = 2
        
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_normalized)
            
            # Use silhouette score to evaluate clustering quality
            try:
                from sklearn.metrics import silhouette_score
                score = silhouette_score(features_normalized, cluster_labels)
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
            except:
                # Fallback if silhouette_score not available
                best_n_clusters = 2
                break
        
        # Final clustering with best number of clusters
        kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_normalized)
        
        # Assign speaker labels
        df['speaker_label'] = 'person_unknown'
        for i, idx in enumerate(valid_indices):
            speaker_num = cluster_labels[i] + 1  # Start from person_1
            df.at[idx, 'speaker_label'] = f'person_{speaker_num}'
        
        # For segments without valid features, assign based on nearest neighbor
        for idx, row in df.iterrows():
            if df.at[idx, 'speaker_label'] == 'person_unknown':
                # Find closest segment with valid label
                closest_idx = None
                min_time_diff = float('inf')
                
                for valid_idx in valid_indices:
                    time_diff = abs(row['start_sec_normalized'] - df.at[valid_idx, 'start_sec_normalized'])
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        closest_idx = valid_idx
                
                if closest_idx is not None:
                    df.at[idx, 'speaker_label'] = df.at[closest_idx, 'speaker_label']
                else:
                    df.at[idx, 'speaker_label'] = 'person_1'
        
        # Log clustering results
        speaker_counts = df['speaker_label'].value_counts()
        logger.info(f"Voice-based clustering result: {dict(speaker_counts)}")
        logger.info(f"Clustering quality score: {best_score:.3f}")
        
        return df
    
    def load_speech_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load speech.csv file and parse timestamps
        
        Args:
            csv_path: Path to speech.csv file
            
        Returns:
            DataFrame with parsed speech data and normalized timestamps
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
            
            # Normalize timestamps to start from 0
            if len(df) > 0:
                min_start_time = df['start_sec'].min()
                df['start_sec_normalized'] = df['start_sec'] - min_start_time
                df['end_sec_normalized'] = df['end_sec'] - min_start_time
                
                logger.info(f"Normalized timestamps. Original range: {df['start_sec'].min():.2f}-{df['end_sec'].max():.2f}s")
                logger.info(f"Normalized range: {df['start_sec_normalized'].min():.2f}-{df['end_sec_normalized'].max():.2f}s")
            
            logger.info(f"Loaded {len(df)} speech segments from {csv_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV {csv_path}: {e}")
            return None
    
    def extract_audio_for_diarization(self, recording_dir: str, output_audio_path: str) -> bool:
        """
        Extract audio from corresponding MP4 file for diarization
        
        Args:
            recording_dir: Directory containing recording data
            output_audio_path: Output path for extracted audio
            
        Returns:
            True if successful, False otherwise
        """
        # Get recording name from directory
        recording_name = os.path.basename(recording_dir)
        
        # Look for MP4 file in dataset directory
        dataset_dir = "/mas/robots/prg-aria/dataset"
        mp4_path = os.path.join(dataset_dir, f"{recording_name}.mp4")
        
        if not os.path.exists(mp4_path):
            logger.error(f"MP4 file not found: {mp4_path}")
            return False
        
        try:
            # Use ffmpeg to extract audio from MP4 file
            import subprocess
            
            cmd = [
                "ffmpeg", "-i", mp4_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM 16-bit
                "-ar", "16000",  # 16kHz sample rate (good for speech recognition)
                "-ac", "1",  # Mono (better for diarization)
                "-af", "highpass=f=200,lowpass=f=3400",  # Filter for speech frequency range
                "-y",  # Overwrite output
                output_audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Audio extracted successfully from MP4: {output_audio_path}")
                return True
            else:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error extracting audio from MP4: {e}")
            return False
    
    def perform_diarization(self, audio_path: str) -> Optional[Annotation]:
        """
        Perform speaker diarization on audio file with optimized parameters
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Pyannote Annotation object with speaker segments
        """
        if not self.pipeline:
            logger.warning("No diarization pipeline available")
            return None
        
        try:
            # Configure diarization parameters for better performance
            if hasattr(self.pipeline, '_segmentation'):
                # Reduce minimum segment duration for better precision
                self.pipeline._segmentation.min_duration_on = 0.1
                self.pipeline._segmentation.min_duration_off = 0.1
            
            # Load audio and perform diarization
            logger.info(f"Starting diarization for {audio_path}")
            diarization = self.pipeline(audio_path)
            
            # Log diarization results
            num_speakers = len(diarization.labels())
            total_duration = sum(segment.duration for segment, _, _ in diarization.itertracks())
            logger.info(f"Diarization completed: {num_speakers} speakers detected, {total_duration:.2f}s total speech")
            
            return diarization
            
        except Exception as e:
            logger.error(f"Error during diarization: {e}")
            return None
    
    def assign_speakers_to_segments(self, df: pd.DataFrame, diarization: Optional[Annotation], audio_path: str = None) -> pd.DataFrame:
        """
        Assign speaker labels using voice-based clustering as primary method
        
        Args:
            df: DataFrame with speech segments (with normalized timestamps)
            diarization: Pyannote diarization results (optional)
            audio_path: Path to audio file for voice feature extraction
            
        Returns:
            DataFrame with speaker_label column added
        """
        # Primary and ONLY method: Voice-based clustering
        if audio_path and os.path.exists(audio_path):
            logger.info("Using voice-based speaker clustering)")
            try:
                return self.cluster_speakers_by_voice(df, audio_path)
            except Exception as e:
                logger.error(f"Voice-based clustering failed: {e}")
                logger.error("Cannot proceed without voice analysis - audio extraction required!")
                # Return with all segments as person_1 rather than using gap-based methods
                df['speaker_label'] = 'person_1'
                logger.warning("Assigning all segments to person_1 due to voice analysis failure")
                return df
        
        # Secondary method: Pyannote diarization (still voice-based)
        if diarization is not None:
            logger.info("Using pyannote diarization (음성 기반 다이어리제이션)")
            # Create speaker_label column
            df['speaker_label'] = 'unknown'
            
            for idx, row in df.iterrows():
                # Use normalized timestamps for diarization
                start_time = row['start_sec_normalized']
                end_time = row['end_sec_normalized']
                
                # Find overlapping speakers in diarization
                segment = Segment(start_time, end_time)
                speakers = []
                
                for speech_segment, _, speaker in diarization.itertracks(yield_label=True):
                    if segment.overlaps(speech_segment):
                        overlap_duration = segment & speech_segment
                        if overlap_duration.duration > 0.05:  # Reduced threshold to 50ms
                            speakers.append((speaker, overlap_duration.duration))
                
                if speakers:
                    # Assign speaker with longest overlap
                    best_speaker = max(speakers, key=lambda x: x[1])[0]
                    df.at[idx, 'speaker_label'] = f"person_{best_speaker.split('_')[-1]}"
                else:
                    df.at[idx, 'speaker_label'] = 'person_unknown'
            
            return df
        
        # No voice-based methods available - refuse to use gap-based methods
        logger.error("No voice-based speaker separation method available!")
        logger.error("Audio extraction failed and pyannote not available.")
        logger.error("Gap-based methods are disabled - all segments assigned to person_1")
        df['speaker_label'] = 'person_1'
        return df
    
    def _improved_fallback_speaker_assignment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Improved fallback speaker assignment using multiple heuristics
        
        Args:
            df: DataFrame with speech segments (with normalized timestamps)
            
        Returns:
            DataFrame with speaker_label column added
        """
        logger.info("Using improved fallback speaker assignment")
        
        if len(df) == 0:
            return df
        
        # Initialize speaker labels
        df['speaker_label'] = 'person_1'
        
        if len(df) == 1:
            return df
        
        # Use normalized timestamps
        df['time_gap'] = df['start_sec_normalized'] - df['end_sec_normalized'].shift(1)
        df['segment_duration'] = df['end_sec_normalized'] - df['start_sec_normalized']
        
        # Analyze text patterns for speaker changes
        df['text_length'] = df['written'].str.len()
        df['word_count'] = df['written'].str.split().str.len()
        
        current_speaker = 'person_1'
        speaker_map = {'person_1': 'person_2', 'person_2': 'person_1'}
        
        # Parameters for speaker switching
        GAP_THRESHOLD_SHORT = 1.0    # Short gap threshold
        GAP_THRESHOLD_LONG = 3.0     # Long gap threshold
        CONFIDENCE_THRESHOLD = 0.7   # Low confidence might indicate speaker change
        
        for idx, row in df.iterrows():
            if idx == 0:
                df.at[idx, 'speaker_label'] = current_speaker
                continue
            
            prev_row = df.iloc[idx - 1]
            should_switch = False
            
            # Rule 1: Long gaps usually indicate speaker change
            if row['time_gap'] > GAP_THRESHOLD_LONG:
                should_switch = True
                logger.debug(f"Speaker switch at idx {idx}: Long gap ({row['time_gap']:.2f}s)")
            
            # Rule 2: Medium gaps with confidence drop
            elif (row['time_gap'] > GAP_THRESHOLD_SHORT and 
                  (row['confidence'] < CONFIDENCE_THRESHOLD or prev_row['confidence'] < CONFIDENCE_THRESHOLD)):
                should_switch = True
                logger.debug(f"Speaker switch at idx {idx}: Medium gap + low confidence")
            
            # Rule 3: Very short segments might be interjections
            elif (row['segment_duration'] < 0.5 and row['word_count'] <= 2 and 
                  row['time_gap'] > 0.5):
                should_switch = True
                logger.debug(f"Speaker switch at idx {idx}: Short interjection")
            
            # Rule 4: Pattern-based switching (every few segments if no clear indicators)
            elif idx > 3 and idx % 4 == 0 and row['time_gap'] > 0.8:
                # Periodic switching as fallback
                should_switch = True
                logger.debug(f"Speaker switch at idx {idx}: Periodic pattern")
            
            if should_switch:
                current_speaker = speaker_map[current_speaker]
            
            df.at[idx, 'speaker_label'] = current_speaker
        
        # Clean up temporary columns
        df.drop(['time_gap', 'segment_duration', 'text_length', 'word_count'], 
                axis=1, inplace=True, errors='ignore')
        
        # Log speaker distribution
        speaker_counts = df['speaker_label'].value_counts()
        logger.info(f"Speaker distribution: {dict(speaker_counts)}")
        
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
                # Use normalized timestamps starting from 0
                start_time = row['start_sec_normalized']
                end_time = row['end_sec_normalized']
                speaker_id = row['speaker_label']
                
                words = transcription_text.split()
                if words:
                    time_per_word = (end_time - start_time) / len(words)
                    
                    for i, word in enumerate(words):
                        word_start = start_time + (i * time_per_word)
                        word_end = start_time + ((i + 1) * time_per_word)
                        
                        # Compute frame range for this word (Aria uses 20 fps)
                        frame_list = self.compute_frame_range(word_start, word_end, fps=20.0)
                        
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
        
        # Extract audio for diarization and voice analysis
        diarization = None
        audio_extracted = False
        
        if self.extract_audio_for_diarization(recording_dir, temp_audio_path):
            audio_extracted = True
            
            # Run pyannote diarization if available
            if self.pipeline:
                diarization = self.perform_diarization(temp_audio_path)
        
        # Assign speakers using voice-based clustering (uses temp_audio_path)
        audio_path_for_clustering = temp_audio_path if audio_extracted else None
        df_with_speakers = self.assign_speakers_to_segments(df, diarization, audio_path_for_clustering)
        
        # Clean up temporary audio file after voice analysis
        if audio_extracted and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
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
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    
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
