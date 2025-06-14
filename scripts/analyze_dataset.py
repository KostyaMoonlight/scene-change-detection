#!/usr/bin/env python3
"""
Dataset Analysis Script

Main script for calculating comprehensive statistics and analysis of the AutoShot dataset.

This script provides detailed analysis including:
1. Dataset completeness analysis
2. Annotation structure analysis  
3. Video file statistics
4. Cross-referencing with AutoShot annotations
5. Research viability assessment

Usage:
    python scripts/analyze_dataset.py [OPTIONS]
    
Options:
    --data-dir PATH         Base directory for dataset (default: .data)
    --output-dir PATH       Output directory for reports (default: .data/analysis)
    --detailed             Generate detailed analysis reports
    --export-missing       Export list of missing videos
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.analysis.completeness_analyzer import CompletenessAnalyzer
from scripts.analysis.annotation_analyzer import AnnotationAnalyzer
from scripts.analysis.pickle_analyzer import PickleAnalyzer
from scripts.analysis.statistics_calculator import StatisticsCalculator
from scripts.analysis.report_generator import ReportGenerator
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('.data/dataset_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main dataset analysis pipeline."""
    parser = argparse.ArgumentParser(description='Analyze AutoShot dataset comprehensively')
    parser.add_argument('--data-dir', default='.data', help='Base directory for dataset')
    parser.add_argument('--output-dir', default='.data/analysis', help='Output directory for reports')
    parser.add_argument('--detailed', action='store_true', help='Generate detailed analysis reports')
    parser.add_argument('--export-missing', action='store_true', help='Export list of missing videos')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🔍 Starting comprehensive dataset analysis...")
    
    try:
        # Initialize analyzers
        completeness_analyzer = CompletenessAnalyzer(data_dir=args.data_dir)
        annotation_analyzer = AnnotationAnalyzer(data_dir=args.data_dir)
        pickle_analyzer = PickleAnalyzer(data_dir=args.data_dir)
        stats_calculator = StatisticsCalculator(data_dir=args.data_dir)
        report_generator = ReportGenerator(output_dir=output_dir)
        
        # Step 1: Analyze dataset completeness
        logger.info("📊 Step 1: Analyzing dataset completeness...")
        completeness_results = completeness_analyzer.analyze_completeness()
        
        # Step 2: Analyze annotation structure
        logger.info("📋 Step 2: Analyzing annotation structure...")
        annotation_results = annotation_analyzer.analyze_annotations()
        
        # Step 3: Analyze pickle files
        logger.info("🔍 Step 3: Analyzing pickle annotations...")
        pickle_results = pickle_analyzer.analyze_pickle_files()
        
        # Step 4: Calculate comprehensive statistics
        logger.info("📈 Step 4: Calculating comprehensive statistics...")
        stats_results = stats_calculator.calculate_all_statistics(
            completeness_results, annotation_results, pickle_results
        )
        
        # Step 5: Generate reports
        logger.info("📄 Step 5: Generating analysis reports...")
        
        # Generate summary report
        summary_report = report_generator.generate_summary_report(stats_results)
        summary_path = output_dir / f"dataset_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(summary_path, 'w') as f:
            f.write(summary_report)
        logger.info(f"   ✅ Summary report saved: {summary_path}")
        
        if args.detailed:
            # Generate detailed report
            detailed_report = report_generator.generate_detailed_report(stats_results)
            detailed_path = output_dir / f"dataset_analysis_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(detailed_path, 'w') as f:
                f.write(detailed_report)
            logger.info(f"   ✅ Detailed report saved: {detailed_path}")
        
        if args.export_missing:
            # Export missing videos list
            missing_videos = completeness_results.get('missing_videos_list', [])
            missing_path = output_dir / "missing_videos.txt"
            with open(missing_path, 'w') as f:
                for video in missing_videos:
                    f.write(f"{video}\n")
            logger.info(f"   ✅ Missing videos list saved: {missing_path}")
        
        # Step 6: Print key metrics
        logger.info("📊 Analysis Summary:")
        logger.info(f"   📁 Total videos in dataset: {stats_results.get('total_videos', 0)}")
        logger.info(f"   📋 Annotated videos: {stats_results.get('annotated_videos', 0)}")
        logger.info(f"   ✅ Coverage: {stats_results.get('coverage_percentage', 0):.1f}%")
        logger.info(f"   ❌ Missing videos: {stats_results.get('missing_videos_count', 0)}")
        logger.info(f"   🎬 Total shot boundaries: {stats_results.get('total_shot_boundaries', 0)}")
        
        logger.info("🎉 Dataset analysis completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset analysis failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 