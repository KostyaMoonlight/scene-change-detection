# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a scene change detection research project focused on shot boundary detection in videos. The codebase provides:

- **VideoReader**: Efficient video reading interface with flexible frame extraction
- **AutoShot Dataset Integration**: PyTorch DataLoader for the AutoShot scene change detection dataset  
- **Transformation Pipeline**: Modular image processing pipeline for feature extraction
- **Dataset Analysis Tools**: Comprehensive analysis and preparation utilities

## Development Commands

### Environment Setup
```bash
# Activate virtual environment (required for all operations)
source activate_env.sh

# Install dependencies
pip install -r requirements.txt
```

### Dataset Operations
```bash
# Download AutoShot annotations only
python scripts/prepare_dataset.py --force-annotations

# Complete dataset preparation (automatic download)
python scripts/prepare_dataset.py

# Download dataset files automatically
python scripts/prepare_dataset.py --force-download
```

### Example Usage
```bash
# Test VideoReader functionality
python example_usage.py

# Test AutoShot DataLoader
python autoshot_dataloader.py
```

## Architecture Overview

### Core Components

**VideoReader (`utils/video_reader.py`)**
- Context manager for efficient video reading
- Supports flexible position formats: `[pos1, pos2]`, `[[pos1, pos2], [pos3, pos4]]`
- Frame caching for performance optimization
- Batch processing and range extraction capabilities

**AutoShot DataLoader (`autoshot_dataloader.py`)**
- PyTorch Dataset for scene change detection training
- Generates sequences centered around shot boundaries
- 3-class labeling: previous/transition/next shots
- Configurable sequence length and data augmentation

**Transformation Pipeline (`shot/transfromations/`)**
- Modular image processing pipeline
- Components: color manipulation, histogram calculation, HDR filtering, pooling
- Chainable transformations for feature extraction

### Key Modules

- `scripts/core/`: Core configuration, logging, and base classes
- `scripts/processors/`: Dataset preparation processors (download, organize, annotate, unify)
- `shot/dataset_preperation/`: Dataset preparation utilities
- `shot/transfromations/`: Image processing pipeline components

## Working with VideoReader

The VideoReader is the central component for video processing:

```python
from utils.video_reader import VideoReader

# Context manager (recommended)
with VideoReader("path/to/video.mp4") as reader:
    # Individual positions
    frames = reader.get_frames([10, 50, 100])
    
    # Position pairs/ranges
    frames = reader.get_frames([[20, 25], [80, 85]])
    
    # Around shot boundaries
    frames = reader.get_frames_around_positions([120, 350], window_size=5)
```

## Working with AutoShot Dataset

```python
from autoshot_dataloader import create_autoshot_dataloader

dataloader = create_autoshot_dataloader(
    sequence_length=16,
    batch_size=4,
    random_offset_range=3
)

for inputs, labels in dataloader:
    # inputs: [batch_size, sequence_length, 3, height, width]  
    # labels: [batch_size, sequence_length] (0=previous, 1=transition, 2=next)
```

## Dataset Structure

Expected structure after setup:
```
.data/processed/
├── ads_game_videos/     # Advertising/gaming videos
└── video_download/      # Downloaded video files
```

Annotation files:
- `.data/kuaishou_v2.txt`: Training/validation annotations (344 videos)
- `.data/autoshot/gt_scenes_dict_baseline_v2.pickle`: Test set annotations (200 videos)

## Common Development Patterns

**Video Processing**: Use VideoReader context manager with flexible position formats for efficient frame extraction

**Dataset Integration**: Extend AutoshotDataset for custom training data preparation

**Transformation Pipeline**: Chain transformations using TransformationPipeline for feature extraction

## Code style

- Functions and methods around 50 lines
- Leave comments only the code cannot be understood without them
- SOLID and DRY principles, emphasis on single responsibility, code extensibility 
