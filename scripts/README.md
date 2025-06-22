# Scripts Directory

Clean, modular AutoShot dataset preparation pipeline following SOLID principles.

## Usage

### Basic Dataset Preparation
```bash
# Download annotations only
python scripts/prepare_dataset.py --force-annotations

# Complete pipeline (will show manual download instructions)
python scripts/prepare_dataset.py

# Download dataset files (shows manual instructions)
python scripts/prepare_dataset.py --force-download
```

### Automatic Download Setup (Optional)

To enable automatic download, add Google Drive file IDs to `scripts/core/config.py`:

1. Visit: https://drive.google.com/drive/folders/1xZN6tvefXXmpZlIZ6GoSUUxpDQQOSNfJ
2. For each required file, right-click → Get Link → copy file ID from URL
3. Add to `GOOGLE_DRIVE_FILE_IDS` in `scripts/core/config.py`:

```python
GOOGLE_DRIVE_FILE_IDS = {
    "ads_game_videos.zip": "1ABC123...",
    "ads_game_videos_2.zip": "1XYZ789...",
    # ... etc
}
```

## Architecture

### Core Components (`core/`)
- **`config.py`**: Centralized configuration constants
- **`logger.py`**: Logging setup with file + console output  
- **`base.py`**: Abstract base classes for processors

### Processors (`processors/`)
- **`downloader.py`**: Google Drive dataset download
- **`organizer.py`**: Zip extraction and video organization
- **`annotation_downloader.py`**: AutoShot annotation download
- **`unifier.py`**: Dataset unification into standardized format

## 4-Step Pipeline

1. **Download** → Automatically download zip files from AutoShot Google Drive
2. **Organize** → Extract videos into `processed/ads_game_videos/` and `processed/video_download/`
3. **Annotations** → Download `kuaishou_v2.txt` and `gt_scenes_dict_baseline_v2.pickle`
4. **Unify** → Create `autoshot.json` and flat `autoshot_videos/` structure

## Design Principles

✅ **Single Responsibility**: Each processor handles one task  
✅ **Open/Closed**: Easy to extend with new processors  
✅ **Dependency Inversion**: Abstract base classes define interfaces  
✅ **DRY**: Shared configuration and logging  
✅ **Clean Code**: Minimal, focused methods with clear names

## Output Structure

```
.data/
├── downloads/              # Zip files
├── processed/
│   ├── ads_game_videos/   # Extracted videos
│   └── video_download/    # Extracted videos
├── autoshot/              # Annotations
│   ├── kuaishou_v2.txt
│   └── gt_scenes_dict_baseline_v2.pickle
├── autoshot.json          # Unified annotations
├── autoshot_videos/       # Flat video structure
└── dataset_preparation.log
```

## Legacy Files Removed

- `dataset_analysis.py` (19KB) → Replaced by modular components
- `analyze_dataset.py` → Replaced by processors
- `annotations/` → Integrated into processors
- `analysis/` → Replaced by unifier component