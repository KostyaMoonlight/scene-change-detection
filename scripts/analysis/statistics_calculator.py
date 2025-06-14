"""
Statistics Calculator Module

Calculates comprehensive statistics combining all analysis results
into unified metrics and insights.
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class StatisticsCalculator:
    """Calculates comprehensive dataset statistics."""
    
    def __init__(self, data_dir: str = ".data"):
        """Initialize statistics calculator."""
        self.data_dir = data_dir
    
    def calculate_all_statistics(self, 
                                completeness_results: Dict[str, Any],
                                annotation_results: Dict[str, Any], 
                                pickle_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics from all analysis results.
        
        Args:
            completeness_results: Results from completeness analysis
            annotation_results: Results from annotation analysis
            pickle_results: Results from pickle analysis
            
        Returns:
            Dictionary with comprehensive statistics
        """
        logger.info("📈 Calculating comprehensive statistics...")
        
        # Extract key metrics
        total_videos = completeness_results.get('our_video_counts', {}).get('total', 0)
        annotated_videos = completeness_results.get('completeness_metrics', {}).get('total_annotated_videos', 0)
        coverage = completeness_results.get('completeness_metrics', {}).get('coverage_percentage', 0)
        missing_count = completeness_results.get('completeness_metrics', {}).get('missing_videos', 0)
        
        # Calculate derived statistics
        total_shot_boundaries = completeness_results.get('annotation_counts', {}).get('total_shot_boundaries', 0)
        
        stats = {
            'total_videos': total_videos,
            'annotated_videos': annotated_videos,
            'coverage_percentage': coverage,
            'missing_videos_count': missing_count,
            'total_shot_boundaries': total_shot_boundaries,
            'analysis_timestamp': 'placeholder',
            'dataset_quality': self._assess_dataset_quality(coverage, total_videos)
        }
        
        return stats
    
    def _assess_dataset_quality(self, coverage: float, total_videos: int) -> str:
        """Assess overall dataset quality based on metrics."""
        if coverage >= 80 and total_videos >= 500:
            return 'Excellent'
        elif coverage >= 60 and total_videos >= 300:
            return 'Good'
        elif coverage >= 40:
            return 'Fair'
        else:
            return 'Poor' 