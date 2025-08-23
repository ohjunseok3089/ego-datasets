#!/usr/bin/env python3
"""
Simple Audio Diarization Script
Just prints speaker segments from audio file using pyannote.audio
"""

import os
import sys
import argparse
import logging
import subprocess
import tempfile

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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


class SimpleAudioDiarizer:
    """Simple audio diarization using pyannote.audio"""
    
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
    
    def extract_audio_from_video(self, video_path: str) -> str:
        """
        Extract audio from video file using ffmpeg
        
        Args:
            video_path: Path to video file
            
        Returns:
            Path to extracted audio file
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None
        
        try:
            # Create temporary audio file
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio_path = temp_audio.name
            temp_audio.close()
            
            # FFmpeg command for audio extraction
            cmd = [
                "ffmpeg", "-i", video_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM 16-bit
                "-ar", "16000",  # 16kHz sample rate
                "-ac", "1",  # Mono
                "-y",  # Overwrite output
                temp_audio_path
            ]
            
            # Run ffmpeg
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Audio extracted successfully to {temp_audio_path}")
                return temp_audio_path
            else:
                logger.error(f"FFmpeg error: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return None
    
    def diarize_audio(self, audio_path: str, min_speakers: int = 1, max_speakers: int = 2):
        """
        Run diarization on audio file and print results
        
        Args:
            audio_path: Path to audio/video file
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
        """
        if not os.path.exists(audio_path):
            logger.error(f"Audio/video file not found: {audio_path}")
            return
        
        # Check if it's a video file and extract audio if needed
        temp_audio_path = None
        try:
            file_ext = os.path.splitext(audio_path)[1].lower()
            if file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
                logger.info(f"Detected video file: {audio_path}")
                temp_audio_path = self.extract_audio_from_video(audio_path)
                if temp_audio_path is None:
                    logger.error("Failed to extract audio from video")
                    return
                actual_audio_path = temp_audio_path
            else:
                actual_audio_path = audio_path
            
            logger.info(f"Running diarization on {actual_audio_path}")
            logger.info(f"Speaker constraints: {min_speakers}-{max_speakers} speakers")
            
            # Run the pipeline with speaker constraints
            diarization = self.pipeline(actual_audio_path, min_speakers=min_speakers, max_speakers=max_speakers)
            
            # Get statistics
            num_speakers = len(diarization.labels())
            logger.info(f"Detected {num_speakers} speakers")
            
            # Print all speaker segments
            print(f"\n=== Speaker Segments ===")
            print(f"Format: start=XX.Xs stop=XX.Xs speaker_X")
            print(f"{'='*50}")
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
            
            # Summary
            total_duration = sum(turn.duration for turn, _, _ in diarization.itertracks(yield_label=True))
            print(f"\n=== Summary ===")
            print(f"Total speech duration: {total_duration:.2f}s")
            print(f"Number of segments: {len(list(diarization.itertracks(yield_label=True)))}")
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up temporary audio file
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                    logger.info("Cleaned up temporary audio file")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Simple Audio Diarization - Print speaker segments",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "audio_file", 
        help="Path to audio file (WAV, MP3, etc.)"
    )
    
    parser.add_argument(
        "--auth_token", 
        required=True,
        help="HuggingFace auth token for pyannote models"
    )
    
    parser.add_argument(
        "--min_speakers", 
        type=int, 
        default=1,
        help="Minimum number of speakers (default: 1)"
    )
    
    parser.add_argument(
        "--max_speakers", 
        type=int, 
        default=2,
        help="Maximum number of speakers (default: 2)"
    )
    
    args = parser.parse_args()
    
    # Initialize diarization system
    try:
        diarizer = SimpleAudioDiarizer(auth_token=args.auth_token)
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        sys.exit(1)
    
    # Run diarization
    diarizer.diarize_audio(args.audio_file, args.min_speakers, args.max_speakers)


if __name__ == "__main__":
    main()
