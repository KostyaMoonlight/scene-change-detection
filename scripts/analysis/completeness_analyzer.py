"""
Dataset Completeness Analyzer Module

Analyzes the completeness of our downloaded dataset by comparing it with
AutoShot annotations to identify missing videos and calculate coverage metrics.
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple
import logging
from collections import defaultdict

# Import common parser
from scripts.annotations.kuaishou_parser import KuaishouAnnotationParser
from scripts.dataset.constants import (
    BASE_DATA_DIR, PROCESSED_DIR, AUTOSHOT_DIR, 
    ADS_GAME_VIDEOS_FOLDER, VIDEO_DOWNLOAD_FOLDER, VIDEO_DATASET_FOLDERS,
    KUAISHOU_ANNOTATION_FILE, get_video_extensions_glob
)

logger = logging.getLogger(__name__)


class CompletenessAnalyzer:
    """Comprehensive analysis of dataset completeness against AutoShot annotations."""
    
    def __init__(self, data_dir: str = BASE_DATA_DIR):
        """
        Initialize completeness analyzer.
        
        Args:
            data_dir: Base directory containing the dataset
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / PROCESSED_DIR
        self.annotations_file = self.data_dir / AUTOSHOT_DIR / KUAISHOU_ANNOTATION_FILE
        
        # Video folder paths
        self.ads_game_videos_dir = self.processed_dir / ADS_GAME_VIDEOS_FOLDER
        self.video_download_dir = self.processed_dir / VIDEO_DOWNLOAD_FOLDER
        
        # Analysis results
        self.our_videos = set()
        self.annotation_videos = set()
        self.annotation_data = {}
        self.missing_videos = set()
        self.extra_videos = set()
    
    def count_our_videos(self) -> Dict[str, int]:
        """
        Count video files in our dataset.
        
        Returns:
            Dictionary with folder names and video counts
        """
        logger.info("📊 Counting videos in our dataset...")
        
        results = {}
        total_videos = 0
        
        for folder_name, folder_path in [
            (ADS_GAME_VIDEOS_FOLDER, self.ads_game_videos_dir),
            (VIDEO_DOWNLOAD_FOLDER, self.video_download_dir)
        ]:
            if folder_path.exists():
                # Count video files with common extensions
                video_files = []
                
                for pattern in get_video_extensions_glob():
                    video_files.extend(list(folder_path.glob(pattern)))
                
                # Remove duplicates and get just filenames
                video_filenames = set(f.name for f in video_files)
                self.our_videos.update(video_filenames)
                
                count = len(video_filenames)
                results[folder_name] = count
                total_videos += count
                
                logger.info(f"   📁 {folder_name}: {count} videos")
                
                # Log first few examples
                for i, filename in enumerate(sorted(video_filenames)[:3]):
                    logger.info(f"      {i+1}. {filename}")
                if count > 3:
                    logger.info(f"      ... and {count - 3} more")
            else:
                results[folder_name] = 0
                logger.warning(f"   ❌ {folder_name}: folder not found")
        
        results["total"] = total_videos
        logger.info(f"   🎯 Total videos in our dataset: {total_videos}")
        
        return results
    
    def parse_annotation_file(self) -> Dict[str, Dict]:
        """
        Parse the AutoShot annotation file (kuaishou_v2.txt) using common parser.
        
        Returns:
            Dictionary with video names and their annotation data
        """
        parser = KuaishouAnnotationParser(self.annotations_file)
        annotation_data = parser.parse()
        
        # Update our internal state with the parsed data
        self.annotation_data = annotation_data
        self.annotation_videos = set(annotation_data.keys())
        
        return annotation_data
    
    def analyze_completeness(self) -> Dict[str, any]:
        """
        Analyze dataset completeness by comparing our videos with annotations.
        
        Returns:
            Dictionary with completeness analysis results
        """
        logger.info("🔍 Analyzing dataset completeness...")
        
        # Count our videos and parse annotations
        our_counts = self.count_our_videos()
        annotation_data = self.parse_annotation_file()
        
        if not annotation_data:
            logger.error("❌ No annotation data available for comparison")
            return {}
        
        # Find missing and extra videos
        self.missing_videos = self.annotation_videos - self.our_videos
        self.extra_videos = self.our_videos - self.annotation_videos
        common_videos = self.annotation_videos & self.our_videos
        
        # Calculate metrics
        total_annotated = len(self.annotation_videos)
        total_available = len(common_videos)
        coverage_percentage = (total_available / total_annotated) * 100 if total_annotated > 0 else 0
        
        results = {
            'our_video_counts': our_counts,
            'annotation_counts': {
                'total_annotated_videos': total_annotated,
                'total_shot_boundaries': sum(len(data['shot_boundaries']) for data in annotation_data.values()),
                'avg_shots_per_video': sum(len(data['shot_boundaries']) for data in annotation_data.values()) / total_annotated if total_annotated > 0 else 0
            },
            'completeness_metrics': {
                'total_annotated_videos': total_annotated,
                'available_videos': total_available,
                'missing_videos': len(self.missing_videos),
                'extra_videos': len(self.extra_videos),
                'coverage_percentage': coverage_percentage
            },
            'missing_videos_list': sorted(list(self.missing_videos)),
            'extra_videos_list': sorted(list(self.extra_videos)),
            'common_videos_list': sorted(list(common_videos))
        }
        
        # Log summary
        logger.info(f"   📊 Completeness Analysis Results:")
        logger.info(f"      🎯 Total annotated videos: {total_annotated}")
        logger.info(f"      ✅ Available in our dataset: {total_available}")
        logger.info(f"      ❌ Missing from our dataset: {len(self.missing_videos)}")
        logger.info(f"      ➕ Extra in our dataset: {len(self.extra_videos)}")
        logger.info(f"      📈 Coverage: {coverage_percentage:.1f}%")
        
        return results
    
    def get_missing_videos_by_pattern(self) -> Dict[str, List[str]]:
        """
        Categorize missing videos by filename patterns.
        
        Returns:
            Dictionary categorizing missing videos by patterns
        """
        if not self.missing_videos:
            return {}
        
        patterns = {
            'numeric_only': [],
            'contains_underscore': [],
            'short_names': [],
            'long_names': [],
            'other': []
        }
        
        for video in self.missing_videos:
            filename = video.replace('.mp4', '')
            
            if filename.isdigit():
                patterns['numeric_only'].append(video)
            elif '_' in filename:
                patterns['contains_underscore'].append(video)
            elif len(filename) < 10:
                patterns['short_names'].append(video)
            elif len(filename) > 20:
                patterns['long_names'].append(video)
            else:
                patterns['other'].append(video)
        
        return patterns 