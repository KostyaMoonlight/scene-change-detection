"""
Annotation Parser Module

Handles parsing of AutoShot annotation files including kuaishou_v2.txt
and other annotation formats.

This module now uses the centralized kuaishou_parser for consistent parsing.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import logging

from .kuaishou_parser import KuaishouAnnotationParser

logger = logging.getLogger(__name__)


class AnnotationParser:
    """Parser for AutoShot annotation files using centralized parsing logic."""
    
    def __init__(self, data_dir: str = ".data"):
        """Initialize annotation parser."""
        self.data_dir = Path(data_dir)
        self.autoshot_dir = self.data_dir / "autoshot"
    
    def parse_kuaishou_annotations(self) -> Dict[str, Dict]:
        """
        Parse kuaishou_v2.txt annotation file using the centralized parser.
        
        Returns:
            Dictionary with parsed annotation data
        """
        annotations_file = self.autoshot_dir / "kuaishou_v2.txt"
        parser = KuaishouAnnotationParser(annotations_file)
        return parser.parse()
    
    def get_annotation_statistics(self) -> Dict[str, any]:
        """
        Get comprehensive statistics about the annotation file.
        
        Returns:
            Dictionary with statistics
        """
        annotations_file = self.autoshot_dir / "kuaishou_v2.txt"
        parser = KuaishouAnnotationParser(annotations_file)
        annotations = parser.parse()
        
        return {
            'total_videos': len(annotations),
            'total_shot_boundaries': sum(len(data['shot_boundaries']) for data in annotations.values()),
            'format_statistics': parser.get_format_statistics(),
            'avg_shots_per_video': sum(len(data['shot_boundaries']) for data in annotations.values()) / len(annotations) if annotations else 0
        } 