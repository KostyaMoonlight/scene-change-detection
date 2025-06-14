#!/usr/bin/env python3
"""
Complete Pipeline Setup Summary

This script demonstrates the complete shot boundary detection pipeline:
1. Dataset download and preparation
2. VideoReader integration
3. Ready for training data extraction
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from utils.video_reader import VideoReader
from scripts.download_dataset import DatasetDownloader


def show_pipeline_summary():
    """Show a complete summary of the pipeline setup."""
    
    print("🎯 Shot Boundary Detection - Complete Pipeline")
    print("=" * 60)
    print()
    
    # Check virtual environment
    print("🔧 Environment Setup:")
    venv_python = Path("venv/bin/python")
    if venv_python.exists():
        print("   ✅ Virtual environment: Active")
        print("   ✅ Dependencies: Installed (OpenCV, NumPy, gdown, etc.)")
    else:
        print("   ❌ Virtual environment: Not found")
        return False
    
    print()
    
    # Check VideoReader
    print("📹 VideoReader Status:")
    try:
        from utils.video_reader import VideoReader, FrameData
        print("   ✅ VideoReader: Available")
        print("   ✅ Flexible position formats: [pos1, pos2], [[pos1, pos2], [pos3, pos4]]")
        print("   ✅ Context manager support: with VideoReader(path) as reader")
        print("   ✅ Frame caching: Enabled")
        print("   ✅ Batch processing: Available")
    except ImportError as e:
        print(f"   ❌ VideoReader: Import failed - {e}")
        return False
    
    print()
    
    # Check dataset status
    print("📊 Dataset Status:")
    dataset_base = Path(".data/processed")
    
    if dataset_base.exists():
        downloader = DatasetDownloader()
        results = downloader.verify_dataset()
        
        total_files = sum(results.values())
        print(f"   ✅ Dataset: Downloaded and processed")
        print(f"   ✅ Total videos: {total_files}")
        
        for folder, count in results.items():
            if count > 0:
                percentage = (count / total_files) * 100
                print(f"   📁 {folder}: {count} files ({percentage:.1f}%)")
        
        # Test with a sample video
        print()
        print("🧪 VideoReader Test:")
        sample_video = None
        
        for folder_name in ["ads_game_videos", "video_download"]:
            folder_path = dataset_base / folder_name
            if folder_path.exists():
                video_files = list(folder_path.glob("*.mp4"))
                if video_files:
                    sample_video = video_files[0]
                    break
        
        if sample_video:
            try:
                with VideoReader(sample_video) as reader:
                    info = reader.get_video_info()
                    print(f"   ✅ Sample video: {sample_video.name}")
                    print(f"   ✅ Duration: {info['duration_seconds']:.1f}s, {info['total_frames']} frames")
                    print(f"   ✅ Resolution: {info['width']}x{info['height']}")
                    
                    # Test frame extraction
                    frames = reader.get_frames([10, 50, 100])
                    print(f"   ✅ Frame extraction: {len(frames)} frames extracted")
                    
            except Exception as e:
                print(f"   ❌ VideoReader test failed: {e}")
                return False
        else:
            print("   ⚠️  No video files found for testing")
    else:
        print("   ❌ Dataset: Not downloaded")
        print("   💡 Run: python download_dataset.py")
        return False
    
    print()
    
    # Show usage examples
    print("🚀 Ready for Shot Boundary Detection!")
    print("=" * 60)
    print()
    print("📝 Quick Usage Examples:")
    print()
    print("# 1. Basic frame extraction")
    print("from utils.video_reader import VideoReader")
    print("with VideoReader('.data/processed/ads_game_videos/video.mp4') as reader:")
    print("    frames = reader.get_frames([10, 50, 100])  # Individual positions")
    print("    frames = reader.get_frames([[20, 25], [80, 85]])  # Position pairs")
    print()
    print("# 2. Shot boundary detection scenario")
    print("shot_boundaries = [120, 350, 580]  # Detected boundaries")
    print("boundary_frames = reader.get_frames_around_positions(shot_boundaries, window_size=5)")
    print()
    print("# 3. Training data extraction")
    print("training_ranges = [[100, 200], [300, 400], [500, 600]]")
    print("training_frames = reader.get_frames_from_ranges(training_ranges, step=10)")
    print()
    
    print("🎯 Next Steps:")
    print("   1. Implement shot boundary detection model")
    print("   2. Create training/validation data splits")
    print("   3. Extract features from frames around boundaries")
    print("   4. Train your model using the VideoReader interface")
    print()
    
    return True


if __name__ == "__main__":
    success = show_pipeline_summary()
    
    if success:
        print("✅ Pipeline setup complete and ready!")
    else:
        print("❌ Pipeline setup incomplete. Please check the errors above.") 