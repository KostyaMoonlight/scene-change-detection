#!/usr/bin/env python3
"""
AutoShot Dataset Preparation

Orchestrates the 4-step dataset preparation pipeline:
1. Download dataset files from Google Drive
2. Extract and organize video files  
3. Download AutoShot annotations
4. Unify annotations and videos into standardized format

Usage:
    python scripts/prepare_dataset.py [OPTIONS]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.core.logger import setup_logger
from scripts.processors.downloader import DatasetDownloader
from scripts.processors.organizer import DatasetOrganizer  
from scripts.processors.annotation_downloader import AnnotationDownloader
from scripts.processors.unifier import DatasetUnifier


class DatasetPreparationPipeline:
    """Main pipeline orchestrator for dataset preparation."""
    
    def __init__(self, data_dir: str = ".data"):
        self.data_dir = data_dir
        self.logger = setup_logger("Pipeline", data_dir)
        
        # Initialize processors
        self.downloader = DatasetDownloader(data_dir)
        self.organizer = DatasetOrganizer(data_dir)
        self.annotation_downloader = AnnotationDownloader(data_dir)
        self.unifier = DatasetUnifier(data_dir)
    
    def run(self, force_steps: dict = None) -> bool:
        """Run the complete preparation pipeline."""
        force_steps = force_steps or {}
        
        self.logger.info("🚀 Starting AutoShot dataset preparation")
        
        # Check current status
        status = self._check_status()
        self._log_status(status)
        
        try:
            # Step 1: Download dataset
            if force_steps.get('download') or not status['download']:
                self.logger.info("📥 Step 1: Downloading AutoShot dataset")
                result = self.downloader.process()
                if not result['success']:
                    if result.get('manual_required'):
                        self.logger.warning("Manual download required - continuing with other steps")
                        # Continue pipeline - user can organize existing files
                    else:
                        self.logger.error(f"Download failed: {result.get('error', 'Unknown')}")
                        return False
                else:
                    self.logger.info(f"✅ Downloaded {result['downloaded']} files")
            else:
                self.logger.info("✅ Step 1: Download complete")
            
            # Step 2: Extract and organize  
            if force_steps.get('organize') or not status['organize']:
                self.logger.info("📦 Step 2: Organizing videos")
                result = self.organizer.process()
                if not result['success']:
                    self.logger.error("Organization failed")
                    return False
                self.logger.info(f"✅ Organized {result['organized']} videos")
            else:
                self.logger.info("✅ Step 2: Organization complete")
            
            # Step 3: Download annotations
            if force_steps.get('annotations') or not status['annotations']:
                self.logger.info("📋 Step 3: Downloading annotations")
                result = self.annotation_downloader.process()
                if not result['success']:
                    self.logger.warning(f"Some annotations failed: {result['failed']}")
                self.logger.info(f"✅ Downloaded {result['downloaded']} annotation files")
            else:
                self.logger.info("✅ Step 3: Annotations complete")
            
            # Step 4: Unify dataset
            if force_steps.get('unify') or not status['unify']:
                self.logger.info("🔄 Step 4: Unifying dataset")
                result = self.unifier.process()
                if not result['success']:
                    self.logger.error("Unification failed")
                    return False
                self.logger.info(f"✅ Unified {result['unified']} videos")
            else:
                self.logger.info("✅ Step 4: Unification complete")
            
            # Final verification
            final_status = self._check_status()
            self.logger.info("🎉 Dataset preparation complete!")
            self._log_status(final_status, prefix="Final")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            return False
    
    def _check_status(self) -> dict:
        """Check completion status of all steps."""
        return {
            'download': self.downloader.is_complete(),
            'organize': self.organizer.is_complete(), 
            'annotations': self.annotation_downloader.is_complete(),
            'unify': self.unifier.is_complete()
        }
    
    def _log_status(self, status: dict, prefix: str = "Current") -> None:
        """Log pipeline status."""
        self.logger.info(f"📊 {prefix} status:")
        self.logger.info(f"   📥 Download: {'✅' if status['download'] else '❌'}")
        self.logger.info(f"   📦 Organize: {'✅' if status['organize'] else '❌'}")
        self.logger.info(f"   📋 Annotations: {'✅' if status['annotations'] else '❌'}")
        self.logger.info(f"   🔄 Unify: {'✅' if status['unify'] else '❌'}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Prepare AutoShot dataset')
    parser.add_argument('--data-dir', default='.data', 
                       help='Base directory for dataset')
    parser.add_argument('--force-download', action='store_true',
                       help='Force re-download')
    parser.add_argument('--force-organize', action='store_true',
                       help='Force re-organization')
    parser.add_argument('--force-annotations', action='store_true',
                       help='Force re-download annotations')
    parser.add_argument('--force-unify', action='store_true',
                       help='Force re-unification')
    
    args = parser.parse_args()
    
    # Setup force options
    force_steps = {
        'download': args.force_download,
        'organize': args.force_organize,
        'annotations': args.force_annotations,
        'unify': args.force_unify
    }
    
    # Run pipeline
    pipeline = DatasetPreparationPipeline(args.data_dir)
    success = pipeline.run(force_steps)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()