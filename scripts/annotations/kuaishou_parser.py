"""
Kuaishou Annotation Parser Module

Common parsing logic for kuaishou_v2.txt annotation files supporting multiple formats.
This module provides a centralized, reusable parser for all AutoShot annotation parsing needs.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class KuaishouAnnotationParser:
    """
    Centralized parser for kuaishou_v2.txt annotation files.
    
    Supports multiple annotation formats:
    1. Original: "filename.mp4 total_frames" followed by "start,end" 
    2. Bracket: "filename.mp4" followed by "[start,end]"
    
    Handles edge cases like extra commas, typos, and false positives.
    """
    
    def __init__(self, annotation_file_path: Path):
        """
        Initialize the parser.
        
        Args:
            annotation_file_path: Path to the kuaishou_v2.txt file
        """
        self.annotation_file_path = annotation_file_path
        self.format_stats = {'original': 0, 'bracket': 0, 'errors': 0}
        self._reset_stats()
    
    def _reset_stats(self):
        """Reset parsing statistics."""
        self.format_stats = {'original': 0, 'bracket': 0, 'errors': 0}
    
    def _parse_video_header(self, line: str, line_num: int) -> Tuple[Optional[str], Optional[int]]:
        """
        Parse a video header line.
        
        Args:
            line: Line to parse
            line_num: Line number for debugging
            
        Returns:
            Tuple of (video_filename, total_frames)
        """
        if '.mp4' not in line or line.endswith('.png'):
            return None, None
            
        parts = line.split()
        
        if len(parts) >= 2 and parts[0].endswith('.mp4'):
            # Format: "filename.mp4 total_frames"
            video_name = parts[0]
            try:
                total_frames = int(parts[1])
                self.format_stats['original'] += 1
                return video_name, total_frames
            except ValueError:
                logger.debug(f"Invalid frame count in line {line_num}: {line}")
                self.format_stats['errors'] += 1
                return video_name, None
                
        elif len(parts) == 1 and parts[0].endswith('.mp4'):
            # Format: "filename.mp4" (bracket format, no frame count)
            video_name = parts[0]
            self.format_stats['bracket'] += 1
            return video_name, None
            
        else:
            logger.debug(f"Skipping invalid video header in line {line_num}: {line}")
            self.format_stats['errors'] += 1
            return None, None
    
    def _parse_shot_boundary(self, line: str, line_num: int) -> List[Tuple[int, int]]:
        """
        Parse a shot boundary line.
        
        Args:
            line: Line to parse
            line_num: Line number for debugging
            
        Returns:
            List of (start_frame, end_frame) tuples
        """
        if not (',' in line or ('[' in line and ']' in line)):
            return []
        
        shot_boundaries = []
        
        try:
            if line.startswith('[') and line.endswith(']'):
                # Format: "[start_frame,end_frame]"
                content = line[1:-1]  # Remove brackets
                # Handle typos like [1174.1175] (period instead of comma)
                content = content.replace('.', ',')
                parts = content.split(',')
            else:
                # Format: "start_frame,end_frame"
                parts = line.split(',')
            
            # Clean up parts (remove empty strings and whitespace)
            parts = [part.strip() for part in parts if part.strip()]
            
            if len(parts) == 2:
                start_frame = int(parts[0])
                end_frame = int(parts[1])
                shot_boundaries.append((start_frame, end_frame))
                
            elif len(parts) > 2:
                # Handle multiple boundaries on one line or malformed entries
                # Try to parse pairs
                for i in range(0, len(parts) - 1, 2):
                    if i + 1 < len(parts):
                        try:
                            start_frame = int(parts[i])
                            end_frame = int(parts[i + 1])
                            shot_boundaries.append((start_frame, end_frame))
                        except ValueError:
                            continue
            else:
                logger.debug(f"Skipping invalid shot boundary format in line {line_num}: {line}")
                
        except ValueError:
            logger.debug(f"Skipping invalid shot boundary values in line {line_num}: {line}")
        
        return shot_boundaries
    
    def parse(self) -> Dict[str, Dict]:
        """
        Parse the annotation file and return structured data.
        
        Returns:
            Dictionary with video names as keys and annotation data as values:
            {
                'video.mp4': {
                    'total_frames': int or None,
                    'shot_boundaries': [(start, end), ...],
                    'num_shots': int
                }
            }
        """
        logger.info("📋 Parsing kuaishou annotations...")
        
        if not self.annotation_file_path.exists():
            logger.error(f"❌ Annotation file not found: {self.annotation_file_path}")
            return {}
        
        self._reset_stats()
        annotations = {}
        current_video = None
        current_total_frames = None
        current_shots = []
        
        with open(self.annotation_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            if not line:
                # Empty line - save current video data if exists
                if current_video:
                    annotations[current_video] = {
                        'total_frames': current_total_frames,
                        'shot_boundaries': current_shots.copy(),
                        'num_shots': len(current_shots)
                    }
                
                # Reset for next video
                current_video = None
                current_total_frames = None
                current_shots = []
                
            else:
                # Try to parse as video header
                video_name, total_frames = self._parse_video_header(line, line_num)
                if video_name:
                    current_video = video_name
                    current_total_frames = total_frames
                    continue
                
                # Try to parse as shot boundaries
                shot_boundaries = self._parse_shot_boundary(line, line_num)
                if shot_boundaries:
                    current_shots.extend(shot_boundaries)
        
        # Handle last video if file doesn't end with empty line
        if current_video:
            annotations[current_video] = {
                'total_frames': current_total_frames,
                'shot_boundaries': current_shots.copy(),
                'num_shots': len(current_shots)
            }
        
        self._log_parsing_results(annotations)
        return annotations
    
    def _log_parsing_results(self, annotations: Dict[str, Dict]):
        """Log parsing statistics."""
        total_videos = len(annotations)
        total_shots = sum(len(data['shot_boundaries']) for data in annotations.values())
        
        logger.info(f"   📊 Parsing completed:")
        logger.info(f"      📁 Total videos: {total_videos}")
        logger.info(f"      🎬 Total shot boundaries: {total_shots}")
        logger.info(f"      📋 Format distribution:")
        logger.info(f"         Original format: {self.format_stats['original']} videos")
        logger.info(f"         Bracket format: {self.format_stats['bracket']} videos")
        if self.format_stats['errors'] > 0:
            logger.info(f"         Parsing errors: {self.format_stats['errors']}")
    
    def get_format_statistics(self) -> Dict[str, int]:
        """
        Get parsing format statistics.
        
        Returns:
            Dictionary with format statistics
        """
        return self.format_stats.copy()
    
    def get_video_names(self) -> List[str]:
        """
        Get list of all video names from annotations.
        
        Returns:
            List of video filenames
        """
        annotations = self.parse()
        return list(annotations.keys())
    
    def get_video_count(self) -> int:
        """
        Get total number of annotated videos.
        
        Returns:
            Number of videos
        """
        return len(self.get_video_names())
    
    def get_shot_boundary_count(self) -> int:
        """
        Get total number of shot boundaries across all videos.
        
        Returns:
            Total shot boundaries
        """
        annotations = self.parse()
        return sum(len(data['shot_boundaries']) for data in annotations.values())


def parse_kuaishou_annotations(annotation_file_path: Path) -> Dict[str, Dict]:
    """
    Convenience function to parse kuaishou annotations.
    
    Args:
        annotation_file_path: Path to the kuaishou_v2.txt file
        
    Returns:
        Parsed annotation data
    """
    parser = KuaishouAnnotationParser(annotation_file_path)
    return parser.parse()


def get_annotation_statistics(annotation_file_path: Path) -> Dict[str, any]:
    """
    Get comprehensive statistics about the annotation file.
    
    Args:
        annotation_file_path: Path to the kuaishou_v2.txt file
        
    Returns:
        Dictionary with statistics
    """
    parser = KuaishouAnnotationParser(annotation_file_path)
    annotations = parser.parse()
    
    return {
        'total_videos': len(annotations),
        'total_shot_boundaries': sum(len(data['shot_boundaries']) for data in annotations.values()),
        'format_statistics': parser.get_format_statistics(),
        'avg_shots_per_video': sum(len(data['shot_boundaries']) for data in annotations.values()) / len(annotations) if annotations else 0
    } 