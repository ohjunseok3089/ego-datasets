#!/usr/bin/env python3
"""
VRS Timestamp Extractor for Aria Dataset
Extracts recording start/end timestamps from VRS files for proper time synchronization
"""

import os
import logging
from typing import Optional, Tuple

try:
    import pyvrs as vrs
    VRS_AVAILABLE = True
except ImportError:
    VRS_AVAILABLE = False
    logging.warning("pyvrs not available. Install with: pip install pyvrs")

logger = logging.getLogger(__name__)


class VRSTimestampExtractor:
    """Extract timestamps from VRS files for Aria recordings"""
    
    def __init__(self):
        """Initialize the VRS timestamp extractor"""
        if not VRS_AVAILABLE:
            raise ImportError("pyvrs not available. Install with: pip install pyvrs")
    
    def extract_recording_timestamps(self, vrs_path: str) -> Optional[Tuple[int, int, float]]:
        """
        Extract recording start and end timestamps from VRS file
        
        Args:
            vrs_path: Path to VRS file
            
        Returns:
            Tuple of (start_timestamp_ns, end_timestamp_ns, duration_seconds) or None if failed
        """
        if not os.path.exists(vrs_path):
            logger.error(f"VRS file not found: {vrs_path}")
            return None
        
        try:
            logger.info(f"Extracting timestamps from VRS file: {vrs_path}")
            
            # Open VRS file
            reader = vrs.SyncVRSReader(vrs_path)
            
            # Get available streams
            stream_ids = reader.stream_ids
            
            # Find actual data timestamps (not configuration/state records)
            actual_start_timestamp = None
            actual_end_timestamp = None
            
            for stream_id in stream_ids:
                stream_info = reader.get_stream_info(stream_id)
                if stream_info['data_records_count'] > 0:
                    data_start = stream_info['first_data_record_timestamp']
                    data_end = stream_info['last_data_record_timestamp']
                    
                    if actual_start_timestamp is None or data_start < actual_start_timestamp:
                        actual_start_timestamp = data_start
                    if actual_end_timestamp is None or data_end > actual_end_timestamp:
                        actual_end_timestamp = data_end
            
            if actual_start_timestamp is None or actual_end_timestamp is None:
                logger.error("No valid data timestamps found in VRS file")
                return None
            
            # Convert to nanoseconds for consistency with speech.csv
            start_timestamp_ns = int(actual_start_timestamp * 1e9)
            end_timestamp_ns = int(actual_end_timestamp * 1e9)
            duration_seconds = actual_end_timestamp - actual_start_timestamp
            
            logger.info(f"VRS timestamps extracted:")
            logger.info(f"  - Start: {start_timestamp_ns} ns ({actual_start_timestamp:.3f}s)")
            logger.info(f"  - End: {end_timestamp_ns} ns ({actual_end_timestamp:.3f}s)")
            logger.info(f"  - Duration: {duration_seconds:.2f} seconds")
            
            return start_timestamp_ns, end_timestamp_ns, duration_seconds
            
        except Exception as e:
            logger.error(f"Error extracting VRS timestamps: {e}")
            return None
    
    def convert_speech_timestamps_to_relative(self, 
                                            speech_df, 
                                            vrs_start_timestamp_ns: int) -> Optional:
        """
        Convert speech.csv timestamps to relative seconds from VRS start
        
        Args:
            speech_df: DataFrame with speech.csv data
            vrs_start_timestamp_ns: VRS recording start timestamp in nanoseconds
            
        Returns:
            DataFrame with added relative timestamp columns
        """
        try:
            # Create a copy to avoid modifying original
            df = speech_df.copy()
            
            # Convert to relative seconds
            df['start_sec_relative'] = (df['startTime_ns'] - vrs_start_timestamp_ns) / 1e9
            df['end_sec_relative'] = (df['endTime_ns'] - vrs_start_timestamp_ns) / 1e9
            
            # Validate timestamps
            negative_times = df[df['start_sec_relative'] < 0]
            if len(negative_times) > 0:
                logger.warning(f"Found {len(negative_times)} segments with negative relative start times")
                logger.warning("This may indicate timestamp synchronization issues")
            
            logger.info(f"Converted {len(df)} speech segments to relative timestamps")
            logger.info(f"Relative time range: {df['start_sec_relative'].min():.3f}s - {df['end_sec_relative'].max():.3f}s")
            
            return df
            
        except Exception as e:
            logger.error(f"Error converting speech timestamps: {e}")
            return None
