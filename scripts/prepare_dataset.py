#!/usr/bin/env python3
"""
Dataset Preparation Script

Main script for downloading, extracting, and transforming the AutoShot dataset
for shot boundary detection research.

This script intelligently orchestrates the complete dataset preparation pipeline:
1. Download dataset from Google Drive (skipped if already complete)
2. Extract and organize video files (skipped if already complete)
3. Download AutoShot annotations (skipped if already complete)
4. Unify annotations into JSON format and create flat video structure (skipped if already complete)
5. Verify dataset integrity

The script automatically detects which steps are already completed and skips
unnecessary work, making it safe to run multiple times.

Usage:
    python scripts/prepare_dataset.py [OPTIONS]
    
Options:
    --data-dir PATH           Base directory for dataset (default: .data)
    --google-drive-url URL    Google Drive folder URL for dataset
    --cleanup                 Remove downloaded zip files after extraction
    --force-download          Force re-download even if files exist
    --force-organize          Force re-organization even if already done
    --force-annotations       Force re-download annotations even if exist
    --force-unify             Force re-unification even if already done
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.dataset.downloader import DatasetDownloader
from scripts.dataset.organizer import DatasetOrganizer
from scripts.dataset.verifier import DatasetVerifier
from scripts.dataset.annotation_unifier import AnnotationUnifier
from scripts.annotations.autoshot_downloader import AutoShotDownloader
from scripts.dataset.constants import (
    BASE_DATA_DIR, DOWNLOADS_DIR, PROCESSED_DIR, AUTOSHOT_DIR,
    ALL_ZIP_FILES, REQUIRED_ANNOTATION_FILES,
    ADS_GAME_VIDEOS_FOLDER, VIDEO_DOWNLOAD_FOLDER,
    UNIFIED_VIDEOS_FOLDER, UNIFIED_ANNOTATIONS_FILE,
    DATASET_PREPARATION_LOG, get_video_extensions_glob
)
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(DATASET_PREPARATION_LOG),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_step_completion(data_dir: Path) -> dict:
    """
    Check which preparation steps are already completed.
    
    Args:
        data_dir: Base data directory
        
    Returns:
        Dictionary indicating completion status of each step
    """
    results = {
        'download_complete': False,
        'organization_complete': False,
        'annotations_complete': False,
        'unification_complete': False
    }
    
    # Check Step 1: Downloaded zip files exist
    downloads_dir = data_dir / DOWNLOADS_DIR
    
    if downloads_dir.exists():
        existing_zips = [f.name for f in downloads_dir.glob("*.zip")]
        # Consider download complete if we have at least some key zip files
        results['download_complete'] = all(zip_name in existing_zips for zip_name in ALL_ZIP_FILES)
    
    # Check Step 2: Organized dataset exists
    processed_dir = data_dir / PROCESSED_DIR
    ads_videos_dir = processed_dir / ADS_GAME_VIDEOS_FOLDER
    video_download_dir = processed_dir / VIDEO_DOWNLOAD_FOLDER
    
    if (ads_videos_dir.exists() and video_download_dir.exists()):
        # Check if directories contain video files
        video_count = 0
        for pattern in get_video_extensions_glob():
            video_count += len(list(ads_videos_dir.glob(pattern)))
            video_count += len(list(video_download_dir.glob(pattern)))
        results['organization_complete'] = video_count > 0
    
    # Check Step 3: AutoShot annotations exist
    autoshot_dir = data_dir / AUTOSHOT_DIR
    
    if autoshot_dir.exists():
        existing_files = [f.name for f in autoshot_dir.iterdir()]
        results['annotations_complete'] = all(file_name in existing_files for file_name in REQUIRED_ANNOTATION_FILES)
    
    # Check Step 4: Unified annotations and videos exist
    unified_json = data_dir / UNIFIED_ANNOTATIONS_FILE
    unified_videos_dir = data_dir / UNIFIED_VIDEOS_FOLDER
    
    if unified_json.exists() and unified_videos_dir.exists():
        # Check if unified folder contains videos
        video_count = 0
        for pattern in get_video_extensions_glob():
            video_count += len(list(unified_videos_dir.glob(pattern)))
        results['unification_complete'] = video_count > 0
    
    return results


def main():
    """Main dataset preparation pipeline."""
    parser = argparse.ArgumentParser(description='Prepare AutoShot dataset for research')
    parser.add_argument('--data-dir', default=BASE_DATA_DIR, help='Base directory for dataset')
    parser.add_argument('--google-drive-url', help='Google Drive folder URL for dataset')
    parser.add_argument('--cleanup', action='store_true', help='Remove zip files after extraction')
    parser.add_argument('--force-download', action='store_true', help='Force re-download even if files exist')
    parser.add_argument('--force-organize', action='store_true', help='Force re-organization even if already done')
    parser.add_argument('--force-annotations', action='store_true', help='Force re-download annotations even if exist')
    parser.add_argument('--force-unify', action='store_true', help='Force re-unification even if already done')
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    
    logger.info("🚀 Starting dataset preparation pipeline...")
    
    # Check current completion status
    logger.info("🔍 Checking current preparation status...")
    completion_status = check_step_completion(data_dir)
    
    logger.info("📊 Current status:")
    logger.info(f"   📥 Downloads: {'✅ Complete' if completion_status['download_complete'] else '❌ Needed'}")
    logger.info(f"   📦 Organization: {'✅ Complete' if completion_status['organization_complete'] else '❌ Needed'}")
    logger.info(f"   📋 Annotations: {'✅ Complete' if completion_status['annotations_complete'] else '❌ Needed'}")
    logger.info(f"   🔄 Unification: {'✅ Complete' if completion_status['unification_complete'] else '❌ Needed'}")
    
    try:
        # Initialize components
        downloader = DatasetDownloader(base_dir=args.data_dir)
        organizer = DatasetOrganizer(base_dir=args.data_dir)
        verifier = DatasetVerifier(base_dir=args.data_dir)
        autoshot_downloader = AutoShotDownloader(target_dir=data_dir / "autoshot")
        annotation_unifier = AnnotationUnifier(base_dir=args.data_dir)
        
        # Step 1: Download dataset
        if args.force_download or not completion_status['download_complete']:
            if args.google_drive_url:
                logger.info("📥 Step 1: Downloading dataset...")
                if not downloader.download_via_gdown(args.google_drive_url):
                    logger.error("❌ Dataset download failed")
                    return False
            else:
                logger.warning("⚠️ Google Drive URL not provided, skipping download")
                logger.info("💡 Provide --google-drive-url to download dataset")
        else:
            logger.info("⏭️ Step 1: Skipping download (already complete)")
        
        # Step 2: Extract and organize
        if args.force_organize or not completion_status['organization_complete']:
            logger.info("📦 Step 2: Extracting and organizing dataset...")
            if not organizer.extract_and_organize():
                logger.error("❌ Dataset organization failed")
                return False
        else:
            logger.info("⏭️ Step 2: Skipping organization (already complete)")
        
        # Step 3: Download AutoShot annotations
        if args.force_annotations or not completion_status['annotations_complete']:
            logger.info("📋 Step 3: Downloading AutoShot annotations...")
            autoshot_results = autoshot_downloader.download_all_files()
            if not all(autoshot_results.values()):
                logger.warning("⚠️ Some AutoShot files failed to download")
        else:
            logger.info("⏭️ Step 3: Skipping annotations download (already complete)")
        
        # Step 4: Unify annotations and create flat video structure
        if args.force_unify or not completion_status['unification_complete']:
            logger.info("🔄 Step 4: Creating unified annotations and flat video structure...")
            unification_results = annotation_unifier.unify_annotations_and_videos()
            if not unification_results['success']:
                logger.error("❌ Annotation unification failed")
                return False
        else:
            logger.info("⏭️ Step 4: Skipping unification (already complete)")
        
        # Step 5: Verify dataset
        logger.info("✅ Step 5: Verifying dataset integrity...")
        verification_results = verifier.verify_complete_dataset()
        
        # Step 6: Cleanup if requested
        if args.cleanup:
            logger.info("🧹 Step 6: Cleaning up temporary files...")
            downloader.cleanup_downloads()
        
        # Final status check
        final_status = check_step_completion(data_dir)
        
        # Summary
        logger.info("🎉 Dataset preparation completed successfully!")
        logger.info("📊 Final status:")
        logger.info(f"   📥 Downloads: {'✅ Complete' if final_status['download_complete'] else '❌ Incomplete'}")
        logger.info(f"   📦 Organization: {'✅ Complete' if final_status['organization_complete'] else '❌ Incomplete'}")
        logger.info(f"   📋 Annotations: {'✅ Complete' if final_status['annotations_complete'] else '❌ Incomplete'}")
        logger.info(f"   🔄 Unification: {'✅ Complete' if final_status['unification_complete'] else '❌ Incomplete'}")
        logger.info(f"📈 Verification results: {verification_results}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset preparation failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 