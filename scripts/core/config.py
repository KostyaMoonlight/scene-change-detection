"""Configuration constants for AutoShot dataset preparation."""

from pathlib import Path
from typing import List, Dict


# Base directories
BASE_DATA_DIR = ".data"
DOWNLOADS_DIR = "downloads"
EXTRACTED_DIR = "extracted"
PROCESSED_DIR = "processed"
AUTOSHOT_DIR = "autoshot"
UNIFIED_VIDEOS_DIR = "autoshot_videos"

# Video directories
ADS_GAME_VIDEOS_DIR = "ads_game_videos"
VIDEO_DOWNLOAD_DIR = "video_download"

# Files
UNIFIED_ANNOTATIONS_FILE = "autoshot.json"
LOG_FILE = "dataset_preparation.log"

# AutoShot repository files
AUTOSHOT_FILES = {
    "kuaishou_v2.txt": "https://raw.githubusercontent.com/wentaozhu/AutoShot/main/kuaishou_v2.txt",
    "gt_scenes_dict_baseline_v2.pickle": "https://github.com/wentaozhu/AutoShot/raw/main/gt_scenes_dict_baseline_v2.pickle",
}

# AutoShot Google Drive configuration
AUTOSHOT_GOOGLE_DRIVE_FOLDER_ID = "1xZN6tvefXXmpZlIZ6GoSUUxpDQQOSNfJ"

# Required dataset files (as mentioned in AutoShot README)
REQUIRED_DATASET_FILES = [
    "ads_game_videos.zip",
    "ads_game_videos_2.zip", 
    "video_download.zip",
    "video_download_2.zip",
    "video_download_3.zip",
    "video_download_4.zip", 
    "video_download_5.zip",
    "original_videos.zip",
]

# Optional model/prediction files (not needed for basic dataset preparation)
OPTIONAL_MODEL_FILES = [
    "ckpt_0_200_0.pth",
    "baseline_one_hot_pred_dict_baseline.pickle",
]

# All dataset files for compatibility
DATASET_ZIPS = REQUIRED_DATASET_FILES

# Optional: Specific file IDs (fill in if known)
# Users can manually add Google Drive file IDs here for automatic download
GOOGLE_DRIVE_FILE_IDS = {
    "video_download.zip": "1dNdRse85_m_UzEW9pTImXUfO1aLwP9bF",
    "video_download_2.zip": "1S2hMlfzJt7FJun1tlmk6tjcF_Ir4GVZq",
    "video_download_3.zip": "1yXVDzCT4Pzb8ehdgEhG456FS52Maz6on",
    "video_download_4.zip": "1-Sh4V-bnQqvGSKoi2T-4Buyb9I7qCE1g",
    "video_download_5.zip": "1toySPvtDO6hLVKA3CGea-Gdi0CVHDqj2",
    "ads_game_videos.zip": "1F1T_qt2UdIakXL2FyWi2GAQTNifZ_nBk",
    "ads_game_videos_2.zip": "1IeF3PQySgHXKDN8I8GjjcVrUdyv812iU",
    "original_videos.zip": "1Unge1xMKMl7E18SsxFpG0yxlw44egRsh",
}

# Video file extensions
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]


def get_data_path(data_dir: str = BASE_DATA_DIR) -> Path:
    """Get base data directory path."""
    return Path(data_dir)


def get_downloads_path(data_dir: str = BASE_DATA_DIR) -> Path:
    """Get downloads directory path."""
    return get_data_path(data_dir) / DOWNLOADS_DIR


def get_processed_path(data_dir: str = BASE_DATA_DIR) -> Path:
    """Get processed directory path."""
    return get_data_path(data_dir) / PROCESSED_DIR


def get_autoshot_path(data_dir: str = BASE_DATA_DIR) -> Path:
    """Get AutoShot directory path."""
    return get_data_path(data_dir) / AUTOSHOT_DIR


def get_unified_videos_path(data_dir: str = BASE_DATA_DIR) -> Path:
    """Get unified videos directory path."""
    return get_data_path(data_dir) / UNIFIED_VIDEOS_DIR


def get_unified_annotations_path(data_dir: str = BASE_DATA_DIR) -> Path:
    """Get unified annotations file path."""
    return get_data_path(data_dir) / UNIFIED_ANNOTATIONS_FILE