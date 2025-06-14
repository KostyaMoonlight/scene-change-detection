#!/usr/bin/env python3
"""
Dataset Completeness Analysis Script

This script analyzes our downloaded dataset and compares it with the AutoShot 
annotation file to determine dataset completeness and missing videos.

Based on the AutoShot paper: "AutoShot: A Short Video Dataset and State-of-the-Art 
Shot Boundary Detection" (CVPR 2023) by Wentao Zhu et al.

References:
- AutoShot Repository: https://github.com/wentaozhu/AutoShot
- Paper: https://github.com/wentaozhu/AutoShot/blob/main/CVPR23_AutoShot.pdf
- Annotations: https://github.com/wentaozhu/AutoShot/blob/main/kuaishou_v2.txt
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import logging
from collections import defaultdict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import common parser and constants
from scripts.annotations.kuaishou_parser import KuaishouAnnotationParser
from scripts.dataset.constants import (
    BASE_DATA_DIR, PROCESSED_DIR, ADS_GAME_VIDEOS_FOLDER, VIDEO_DOWNLOAD_FOLDER, 
    KUAISHOU_ANNOTATION_FILE, get_video_extensions_glob
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """Comprehensive analysis of dataset completeness against AutoShot annotations."""
    
    def __init__(self, data_dir: str = BASE_DATA_DIR, annotations_file: str = None):
        """
        Initialize dataset analyzer.
        
        Args:
            data_dir: Base directory containing the dataset
            annotations_file: Path to kuaishou_v2.txt annotation file
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / PROCESSED_DIR
        
        if annotations_file is None:
            self.annotations_file = self.data_dir / KUAISHOU_ANNOTATION_FILE
        else:
            self.annotations_file = Path(annotations_file)
        
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
            Comprehensive analysis results
        """
        logger.info("🔍 Analyzing dataset completeness...")
        
        # Find missing and extra videos
        self.missing_videos = self.annotation_videos - self.our_videos
        self.extra_videos = self.our_videos - self.annotation_videos
        
        # Calculate statistics
        total_annotated = len(self.annotation_videos)
        total_available = len(self.our_videos)
        total_missing = len(self.missing_videos)
        total_extra = len(self.extra_videos)
        
        completion_rate = ((total_annotated - total_missing) / total_annotated * 100) if total_annotated > 0 else 0
        
        results = {
            'total_annotated_videos': total_annotated,
            'total_available_videos': total_available,
            'total_missing_videos': total_missing,
            'total_extra_videos': total_extra,
            'completion_rate': completion_rate,
            'missing_videos': sorted(list(self.missing_videos)),
            'extra_videos': sorted(list(self.extra_videos))
        }
        
        # Analyze shot statistics for available videos
        available_videos = self.annotation_videos.intersection(self.our_videos)
        if available_videos:
            shot_counts = [self.annotation_data[video]['num_shots'] for video in available_videos]
            total_frames = [self.annotation_data[video]['total_frames'] for video in available_videos if self.annotation_data[video]['total_frames']]
            
            results.update({
                'avg_shots_per_video': sum(shot_counts) / len(shot_counts),
                'total_shots_available': sum(shot_counts),
                'avg_frames_per_video': sum(total_frames) / len(total_frames) if total_frames else 0,
                'min_shots': min(shot_counts),
                'max_shots': max(shot_counts)
            })
        
        return results
    
    def generate_detailed_report(self) -> str:
        """Generate a detailed analysis report."""
        
        logger.info("📝 Generating detailed analysis report...")
        
        # Count our videos
        our_counts = self.count_our_videos()
        
        # Parse annotations
        annotation_data = self.parse_annotation_file()
        
        # Analyze completeness
        analysis = self.analyze_completeness()
        
        # Generate report
        report = []
        report.append("🎯 DATASET COMPLETENESS ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Dataset Overview
        report.append("📊 DATASET OVERVIEW")
        report.append("-" * 30)
        report.append(f"📁 Local Dataset Location: {self.processed_dir}")
        report.append(f"📋 Annotation File: {self.annotations_file}")
        report.append(f"📄 Based on: AutoShot - A Short Video Dataset (CVPR 2023)")
        report.append(f"🔗 Reference: https://github.com/wentaozhu/AutoShot")
        report.append("")
        
        # Our Dataset Statistics
        report.append("📈 OUR DATASET STATISTICS")
        report.append("-" * 30)
        for folder, count in our_counts.items():
            if folder != "total":
                report.append(f"📁 {folder}: {count:,} videos")
        report.append(f"🎯 Total Videos Available: {our_counts['total']:,}")
        report.append("")
        
        # AutoShot Annotation Statistics
        report.append("📋 AUTOSHOT ANNOTATION STATISTICS")
        report.append("-" * 30)
        report.append(f"🎬 Total Annotated Videos: {analysis['total_annotated_videos']:,}")
        report.append(f"📊 Total Shot Boundaries: {sum(len(data['shot_boundaries']) for data in annotation_data.values()):,}")
        
        if 'avg_shots_per_video' in analysis:
            report.append(f"📈 Average Shots per Video: {analysis['avg_shots_per_video']:.1f}")
            report.append(f"📊 Total Available Shots: {analysis['total_shots_available']:,}")
            report.append(f"🎞️  Shot Range: {analysis['min_shots']}-{analysis['max_shots']} per video")
            if analysis['avg_frames_per_video'] > 0:
                report.append(f"🎬 Average Frames per Video: {analysis['avg_frames_per_video']:.0f}")
        report.append("")
        
        # Completeness Analysis
        report.append("🔍 COMPLETENESS ANALYSIS")
        report.append("-" * 30)
        report.append(f"✅ Videos Available: {analysis['total_annotated_videos'] - analysis['total_missing_videos']:,}")
        report.append(f"❌ Videos Missing: {analysis['total_missing_videos']:,}")
        report.append(f"➕ Extra Videos (not in annotations): {analysis['total_extra_videos']:,}")
        report.append(f"📊 Completion Rate: {analysis['completion_rate']:.1f}%")
        report.append("")
        
        # Missing Videos Analysis
        if analysis['total_missing_videos'] > 0:
            report.append("❌ MISSING VIDEOS DETAILS")
            report.append("-" * 30)
            
            # Group missing videos by pattern or show top missing
            missing_sample = analysis['missing_videos'][:20]  # Show first 20
            for i, video in enumerate(missing_sample, 1):
                # Show shot info if available
                shot_info = ""
                if video in annotation_data:
                    shots = annotation_data[video]['num_shots']
                    frames = annotation_data[video]['total_frames']
                    shot_info = f" ({shots} shots, {frames} frames)"
                report.append(f"   {i:2d}. {video}{shot_info}")
            
            if len(analysis['missing_videos']) > 20:
                report.append(f"   ... and {len(analysis['missing_videos']) - 20} more missing videos")
            report.append("")
            
            # Missing video statistics
            missing_videos_data = [annotation_data[v] for v in analysis['missing_videos'] if v in annotation_data]
            if missing_videos_data:
                missing_shots = sum(d['num_shots'] for d in missing_videos_data)
                report.append(f"📊 Missing shot boundaries: {missing_shots:,}")
                report.append("")
        
        # Extra Videos Analysis
        if analysis['total_extra_videos'] > 0:
            report.append("➕ EXTRA VIDEOS (NOT IN ANNOTATIONS)")
            report.append("-" * 30)
            extra_sample = analysis['extra_videos'][:10]  # Show first 10
            for i, video in enumerate(extra_sample, 1):
                report.append(f"   {i:2d}. {video}")
            
            if len(analysis['extra_videos']) > 10:
                report.append(f"   ... and {len(analysis['extra_videos']) - 10} more extra videos")
            report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 30)
        
        if analysis['completion_rate'] >= 95:
            report.append("🎉 Excellent! Your dataset is nearly complete.")
        elif analysis['completion_rate'] >= 80:
            report.append("✅ Good dataset coverage. Consider downloading missing videos for completeness.")
        else:
            report.append("⚠️  Significant videos are missing. Consider re-downloading the dataset.")
        
        if analysis['total_missing_videos'] > 0:
            report.append(f"📥 Download missing {analysis['total_missing_videos']} videos to improve completeness")
            
        if analysis['total_extra_videos'] > 0:
            report.append(f"🔍 {analysis['total_extra_videos']} extra videos found - these might be useful for additional training")
        
        report.append("")
        report.append("🔗 USEFUL LINKS")
        report.append("-" * 30)
        report.append("📄 AutoShot Paper: https://github.com/wentaozhu/AutoShot/blob/main/CVPR23_AutoShot.pdf")
        report.append("📋 Annotations: https://github.com/wentaozhu/AutoShot/blob/main/kuaishou_v2.txt")
        report.append("📦 Dataset: https://drive.google.com/drive/folders/1xZN6tvefXXmpZlIZ6GoSUUxpDQQOSNfJ")
        report.append("🏆 Papers with Code: https://paperswithcode.com/paper/autoshot-a-short-video-dataset-and-state-of")
        
        return "\n".join(report)
    
    def save_missing_videos_list(self, output_file: str = "missing_videos.txt") -> None:
        """Save list of missing videos to a file."""
        output_path = self.data_dir / output_file
        
        with open(output_path, 'w') as f:
            f.write("# Missing Videos from AutoShot Dataset\n")
            f.write(f"# Total missing: {len(self.missing_videos)}\n")
            f.write("# Format: video_filename (shots, total_frames)\n\n")
            
            for video in sorted(self.missing_videos):
                if video in self.annotation_data:
                    data = self.annotation_data[video]
                    f.write(f"{video}  # {data['num_shots']} shots, {data['total_frames']} frames\n")
                else:
                    f.write(f"{video}\n")
        
        logger.info(f"💾 Missing videos list saved to: {output_path}")
    
    def run_full_analysis(self) -> Dict[str, any]:
        """Run complete dataset analysis."""
        logger.info("🚀 Starting comprehensive dataset analysis...")
        
        try:
            # Generate detailed report
            report = self.generate_detailed_report()
            
            # Print report
            print("\n" + report)
            
            # Save missing videos list
            if self.missing_videos:
                self.save_missing_videos_list()
            
            # Save full report
            report_path = self.data_dir / "dataset_analysis_report.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"📄 Full report saved to: {report_path}")
            
            logger.info("✅ Dataset analysis completed successfully!")
            
            return {
                'success': True,
                'completion_rate': self.analyze_completeness()['completion_rate'],
                'missing_count': len(self.missing_videos),
                'total_available': len(self.our_videos)
            }
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {str(e)}")
            return {'success': False, 'error': str(e)}


def main():
    """Main function to run dataset analysis."""
    
    print("🎯 AutoShot Dataset Completeness Analyzer")
    print("=" * 50)
    print()
    
    # Initialize analyzer
    analyzer = DatasetAnalyzer(data_dir=BASE_DATA_DIR)
    
    # Check if dataset exists
    if not analyzer.processed_dir.exists():
        print("❌ Dataset not found. Please run the download script first:")
        print("   python download_dataset.py")
        return False
    
    # Check if annotation file exists
    if not analyzer.annotations_file.exists():
        print("❌ Annotation file not found. Downloading...")
        print(f"   Expected location: {analyzer.annotations_file}")
        
        # Try to download it
        import subprocess
        try:
            subprocess.run([
                'curl', '-o', str(analyzer.annotations_file),
                'https://raw.githubusercontent.com/wentaozhu/AutoShot/main/kuaishou_v2.txt'
            ], check=True, capture_output=True)
            print("✅ Annotation file downloaded successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to download annotation file")
            return False
    
    # Run analysis
    results = analyzer.run_full_analysis()
    
    if results['success']:
        print(f"\n🎉 Analysis completed successfully!")
        print(f"📊 Dataset completion rate: {results['completion_rate']:.1f}%")
        if results['missing_count'] > 0:
            print(f"📋 Missing videos: {results['missing_count']}")
            print("💡 Check 'missing_videos.txt' for detailed list")
    else:
        print(f"\n❌ Analysis failed: {results['error']}")
    
    return results['success']


if __name__ == "__main__":
    main() 