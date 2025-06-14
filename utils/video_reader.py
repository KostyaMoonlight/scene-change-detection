import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FrameData:
    """Container for frame data with position information."""
    frame: np.ndarray
    position: int
    timestamp: float


class VideoReader:
    """
    Efficient video reading interface for shot boundary detection training.
    
    Supports extracting frames from positions in multiple formats:
    - [pos1, pos2, pos3] - individual frame positions
    - [[pos1, pos2], [pos3, pos4]] - position ranges or pairs
    - Mixed formats with any nesting level
    """
    
    def __init__(self, video_path: Union[str, Path]):
        """
        Initialize video reader.
        
        Args:
            video_path: Path to video file
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self._cap = None
        self._total_frames = None
        self._fps = None
        self._frame_cache = {}
        
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        
    def open(self):
        """Open video capture."""
        if self._cap is None:
            self._cap = cv2.VideoCapture(str(self.video_path))
            if not self._cap.isOpened():
                raise RuntimeError(f"Failed to open video: {self.video_path}")
            
            self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._fps = self._cap.get(cv2.CAP_PROP_FPS)
            
    def close(self):
        """Close video capture and clear cache."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._frame_cache.clear()
        
    @property
    def total_frames(self) -> int:
        """Get total number of frames in video."""
        if self._total_frames is None:
            self.open()
        return self._total_frames
    
    @property
    def fps(self) -> float:
        """Get frames per second."""
        if self._fps is None:
            self.open()
        return self._fps
    
    def get_frame_at_position(self, position: int, use_cache: bool = True) -> FrameData:
        """
        Extract single frame at specified position.
        
        Args:
            position: Frame number (0-indexed)
            use_cache: Whether to use frame caching
            
        Returns:
            FrameData object containing frame, position, and timestamp
        """
        if not (0 <= position < self.total_frames):
            raise ValueError(f"Position {position} out of range [0, {self.total_frames})")
        
        if use_cache and position in self._frame_cache:
            return self._frame_cache[position]
        
        if self._cap is None:
            self.open()
            
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        ret, frame = self._cap.read()
        
        if not ret:
            raise RuntimeError(f"Failed to read frame at position {position}")
        
        timestamp = position / self.fps
        frame_data = FrameData(frame=frame, position=position, timestamp=timestamp)
        
        if use_cache:
            self._frame_cache[position] = frame_data
            
        return frame_data
    
    def get_frames(self, positions: Union[List[int], List[List[int]]], 
                   use_cache: bool = True) -> List[FrameData]:
        """
        Extract frames from specified positions with flexible input format.
        
        Args:
            positions: Frame positions in various formats:
                      - [pos1, pos2, pos3] - individual positions
                      - [[pos1, pos2], [pos3, pos4]] - position pairs/ranges
                      - Mixed formats with any nesting
            use_cache: Whether to use frame caching
            
        Returns:
            List of FrameData objects
        """
        flattened_positions = self._flatten_positions(positions)
        unique_positions = sorted(set(flattened_positions))
        
        frames = []
        for position in unique_positions:
            frame_data = self.get_frame_at_position(position, use_cache)
            frames.append(frame_data)
            
        return frames
    
    def get_frames_from_ranges(self, position_ranges: List[List[int]], 
                              step: int = 1, use_cache: bool = True) -> List[FrameData]:
        """
        Extract frames from position ranges.
        
        Args:
            position_ranges: List of [start, end] pairs
            step: Step size for frame extraction within ranges
            use_cache: Whether to use frame caching
            
        Returns:
            List of FrameData objects
        """
        positions = []
        for start, end in position_ranges:
            if start > end:
                start, end = end, start
            positions.extend(range(start, end + 1, step))
            
        return self.get_frames(positions, use_cache)
    
    def get_frames_around_positions(self, positions: List[int], 
                                   window_size: int = 5, 
                                   use_cache: bool = True) -> List[FrameData]:
        """
        Extract frames around specified positions with a window.
        
        Args:
            positions: Center positions for frame extraction
            window_size: Number of frames to extract around each position
            use_cache: Whether to use frame caching
            
        Returns:
            List of FrameData objects
        """
        half_window = window_size // 2
        all_positions = []
        
        for pos in positions:
            start = max(0, pos - half_window)
            end = min(self.total_frames - 1, pos + half_window)
            all_positions.extend(range(start, end + 1))
            
        return self.get_frames(all_positions, use_cache)
    
    def extract_frames_batch(self, positions: Union[List[int], List[List[int]]], 
                           output_dir: Optional[Path] = None,
                           filename_template: str = "frame_{position:06d}.jpg") -> List[FrameData]:
        """
        Extract and optionally save frames in batch.
        
        Args:
            positions: Frame positions in flexible format
            output_dir: Directory to save frames (optional)
            filename_template: Template for saved filenames
            
        Returns:
            List of FrameData objects
        """
        frames = self.get_frames(positions)
        
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for frame_data in frames:
                filename = filename_template.format(position=frame_data.position)
                filepath = output_dir / filename
                cv2.imwrite(str(filepath), frame_data.frame)
                
        return frames
    
    def _flatten_positions(self, positions: Union[List[int], List[List[int]]]) -> List[int]:
        """
        Flatten nested position lists to a single list of integers.
        
        Args:
            positions: Nested position structure
            
        Returns:
            Flattened list of positions
        """
        flattened = []
        
        for item in positions:
            if isinstance(item, (list, tuple)):
                flattened.extend(self._flatten_positions(item))
            else:
                flattened.append(int(item))
                
        return flattened
    
    def get_video_info(self) -> dict:
        """
        Get video metadata information.
        
        Returns:
            Dictionary with video information
        """
        if self._cap is None:
            self.open()
            
        return {
            'path': str(self.video_path),
            'total_frames': self.total_frames,
            'fps': self.fps,
            'duration_seconds': self.total_frames / self.fps,
            'width': int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        } 