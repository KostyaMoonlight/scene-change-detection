# Scripts Directory

This directory contains the main entry point scripts and organized submodules for AutoShot dataset preparation and analysis.

## Main Scripts

### 🚀 `prepare_dataset.py`
Main script for downloading, extracting, and transforming the AutoShot dataset.

**✨ Smart Features:**
- **Intelligent Step Detection**: Automatically detects which steps are already completed
- **Safe Re-runs**: Can be run multiple times without duplicating work
- **Selective Force Options**: Force re-execution of specific steps when needed

**Usage:**
```bash
python scripts/prepare_dataset.py [OPTIONS]
```

**Options:**
- `--data-dir PATH`: Base directory for dataset (default: .data)
- `--google-drive-url URL`: Google Drive folder URL for dataset
- `--cleanup`: Remove downloaded zip files after extraction
- `--force-download`: Force re-download even if files exist
- `--force-organize`: Force re-organization even if already done
- `--force-annotations`: Force re-download annotations even if exist
- `--force-unify`: Force re-unification even if already done

**Pipeline Steps:**
1. **Download dataset** from Google Drive (⏭️ skipped if already complete)
2. **Extract and organize** video files (⏭️ skipped if already complete)
3. **Download AutoShot annotations** (⏭️ skipped if already complete)
4. **Unify annotations and videos** into standardized format (⏭️ skipped if already complete)
5. **Verify dataset integrity** (always runs)

### 📊 `analyze_dataset.py`
Main script for calculating comprehensive statistics and analysis of the AutoShot dataset.

**Usage:**
```bash
python scripts/analyze_dataset.py [OPTIONS]
```

**Options:**
- `--data-dir PATH`: Base directory for dataset (default: .data)
- `--output-dir PATH`: Output directory for reports (default: .data/analysis)
- `--detailed`: Generate detailed analysis reports
- `--export-missing`: Export list of missing videos

**Analysis Components:**
1. Dataset completeness analysis
2. Annotation structure analysis
3. Video file statistics
4. Cross-referencing with AutoShot annotations
5. Research viability assessment

## Module Structure

### 📦 `dataset/` - Dataset Preparation Modules
- **`constants.py`**: **Centralized constants** - All dataset paths, filenames, and configurations
- **`downloader.py`**: Handles downloading from Google Drive
- **`organizer.py`**: Extracts and organizes video files
- **`annotation_unifier.py`**: **Unifies annotations** - Creates standardized JSON format and flat video structure
- **`verifier.py`**: Verifies dataset integrity and completeness

### 📋 `annotations/` - Annotation Handling Modules
- **`kuaishou_parser.py`**: **Common parsing module** - Centralized, reusable parser for kuaishou_v2.txt supporting multiple formats
- **`autoshot_downloader.py`**: Downloads AutoShot repository files
- **`parser.py`**: Legacy annotation parser (now uses kuaishou_parser.py)
- **`pickle_handler.py`**: Handles pickle files with ground truth data

### 📈 `analysis/` - Analysis and Statistics Modules
- **`completeness_analyzer.py`**: Analyzes dataset completeness vs annotations
- **`annotation_analyzer.py`**: Analyzes annotation structure and patterns
- **`pickle_analyzer.py`**: Analyzes pickle files for test set data
- **`statistics_calculator.py`**: Calculates comprehensive statistics
- **`report_generator.py`**: Generates analysis reports

## Example Usage

### Complete Dataset Preparation
```bash
# First run: Download and prepare the complete dataset
python scripts/prepare_dataset.py \
    --google-drive-url "https://drive.google.com/drive/folders/YOUR_FOLDER_ID" \
    --cleanup

# Second run: Will automatically skip completed steps
python scripts/prepare_dataset.py
# Output: 
# 📊 Current status:
#    📥 Downloads: ✅ Complete  
#    📦 Organization: ✅ Complete
#    📋 Annotations: ✅ Complete
# ⏭️ Step 1: Skipping download (already complete)
# ⏭️ Step 2: Skipping organization (already complete) 
# ⏭️ Step 3: Skipping annotations download (already complete)

# Force re-download annotations only
python scripts/prepare_dataset.py --force-annotations

# Analyze the prepared dataset
python scripts/analyze_dataset.py \
    --detailed \
    --export-missing
```

### Analysis Only (if dataset already prepared)
```bash
# Run comprehensive analysis
python scripts/analyze_dataset.py \
    --output-dir .data/reports \
    --detailed
```

## 📋 Unified Annotation Format

The unification step creates a standardized JSON format that combines video metadata with shot boundary annotations from **multiple sources**:

```json
{
  "filename": "video.mp4",
  "total_frames": 674,           // Extracted from actual video file
  "fps": 25.0,                   // Extracted from actual video file  
  "original_path": "processed/ads_game_videos/video.mp4",
  "current_path": "autoshot_videos/video.mp4",
  "original_folder": "ads_game_videos",
  "frame_boundaries": [          // Shot boundaries in 'from'/'to' format
    {"from": 0, "to": 68},
    {"from": 69, "to": 196}
  ],
  "num_shots": 2,
  "original_total_frames": 674,  // From annotation file (for comparison)
  "annotation_source": "gt_scenes_dict_baseline_v2.pickle (priority)",
  "kuaishou_total_frames": 674,  // Original kuaishou data (if merged)
  "kuaishou_shot_count": 3       // Original kuaishou shot count (if merged)
}
```

**Key Features:**
- **Multiple annotation sources**: Combines `kuaishou_v2.txt` and `gt_scenes_dict_baseline_v2.pickle`
- **Intelligent merging**: Pickle annotations take priority (higher quality ground truth)
- **Real video metadata**: `total_frames` and `fps` extracted from actual video files using OpenCV
- **Standardized paths**: Both original and current video locations tracked
- **Consistent format**: All shot boundaries use `{"from": X, "to": Y}` structure
- **Flat video structure**: All videos moved to `autoshot_videos/` for easy access
- **Complete coverage**: Includes videos both with and without annotations
- **Source tracking**: Clear indication of which annotation source was used

**Annotation Priority:**
1. **📦 Pickle files**: `gt_scenes_dict_baseline_v2.pickle` (200 videos) - Highest quality ground truth
   - Contains **shot segments**: `[[start1, end1], [start2, end2], ...]`
   - Converted to **shot boundaries**: Cut points between shots
2. **📄 Kuaishou text**: `kuaishou_v2.txt` (647 videos) - Fallback annotations
   - Contains **shot boundaries**: Direct cut point annotations
3. **🔄 Merged**: When both exist, pickle takes priority but kuaishou metadata is preserved

**Format Conversion:**
- **Pickle segments** → **Boundaries**: `[[0,68], [69,143], [144,170]]` → `[(68,69), (143,144)]`
- **Kuaishou boundaries** → **Boundaries**: Direct mapping with format normalization

## Output Structure

```
.data/
├── downloads/          # Downloaded zip files (removed after --cleanup)
├── extracted/          # Extracted content (temporary)
├── processed/          # Final organized dataset
│   ├── ads_game_videos/
│   └── video_download/
├── autoshot/           # AutoShot repository files
│   ├── kuaishou_v2.txt
│   ├── gt_scenes_dict_baseline_v2.pickle
│   └── supernet_best_f1.pickle
├── autoshot.json       # 🆕 Unified annotation format (JSON)
├── autoshot_videos/    # 🆕 Flat video structure (all videos)
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── analysis/           # Analysis reports and outputs
    ├── dataset_analysis_summary_YYYYMMDD_HHMMSS.md
    ├── dataset_analysis_detailed_YYYYMMDD_HHMMSS.md
    └── missing_videos.txt
```

## Architecture Benefits

1. **Modularity**: Each component has a single responsibility
2. **Reusability**: Modules can be imported and used independently
3. **Maintainability**: Clear separation of concerns
4. **Testability**: Each module can be tested in isolation
5. **Extensibility**: Easy to add new analysis components

### Common Parser Design

The `kuaishou_parser.py` module implements the **DRY (Don't Repeat Yourself)** principle by providing:

- **Single source of truth**: One place to maintain annotation parsing logic
- **Consistent parsing**: All components use exactly the same robust parsing algorithm
- **Format support**: Handles both original and bracket annotation formats
- **Error resilience**: Robust handling of malformed entries and edge cases
- **Statistics tracking**: Built-in format distribution and parsing statistics
- **Easy maintenance**: Bug fixes and improvements only need to be made once

**Usage Example:**
```python
from scripts.annotations.kuaishou_parser import KuaishouAnnotationParser

parser = KuaishouAnnotationParser(Path('.data/autoshot/kuaishou_v2.txt'))
annotations = parser.parse()  # Returns standardized format
stats = parser.get_format_statistics()  # Get parsing insights
```

### Constants Module Design

The `constants.py` module eliminates **magic strings** throughout the codebase by providing:

- **Centralized configuration**: All paths, filenames, and URLs in one place
- **Type safety**: Consistent naming and structure across modules
- **Easy maintenance**: Change paths/names once, update everywhere
- **Helper functions**: Utilities for path building and validation
- **Documentation**: Clear organization of all dataset-related constants

**Usage Example:**
```python
from scripts.dataset.constants import (
    BASE_DATA_DIR, ADS_GAME_VIDEOS_FOLDER, KUAISHOU_ANNOTATION_FILE,
    get_processed_video_dir, get_video_extensions_glob
)

# Use constants instead of magic strings
video_dir = get_processed_video_dir(BASE_DATA_DIR, ADS_GAME_VIDEOS_FOLDER)
for pattern in get_video_extensions_glob():
    videos.extend(Path(video_dir).glob(pattern))
```

## Legacy Files

The following files contain the original monolithic implementations and will be removed once the refactored modules are fully implemented:
- `comprehensive_dataset_analysis.py` (38KB) - Being replaced by analysis modules
- `dataset_analysis.py` (19KB) - Being replaced by completeness_analyzer.py

## Dependencies

The scripts require the following Python packages:
- `requests` - For downloading files
- `tqdm` - For progress bars
- `gdown` - For Google Drive downloads
- `numpy` - For data analysis (in pickle files)

Install with:
```bash
pip install requests tqdm gdown numpy
``` 