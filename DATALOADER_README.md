# Autoshot DataLoader

A PyTorch DataLoader for the Autoshot scene change detection dataset.

## Overview

The `AutoshotDataset` reads video sequences and provides frame-level scene change annotations. Each sample contains a sequence of frames with corresponding scene labels that increment when a scene boundary is detected.

## Quick Start

```python
from autoshot_dataloader import create_autoshot_dataloader

# Create dataloader
dataloader = create_autoshot_dataloader(
    sequence_length=16,     # 16 frames per sequence (window around transition)
    batch_size=4,           # Batch size
    random_offset_range=3,  # Random offset ±3 frames for data augmentation
    min_shots=2             # Only videos with >= 2 shots
)

# Iterate through data
for inputs, labels in dataloader:
    # inputs: [batch_size, sequence_length, channels, height, width]
    # labels: [batch_size, sequence_length] 
    frames = inputs    # torch.Size([4, 16, 3, 224, 224])
    scene_labels = labels  # torch.Size([4, 16])
    
    # Your training code here...
```

## Features

- **Flexible sequence length**: Configure how many consecutive frames to extract
- **Configurable stride**: Control overlap between sequences  
- **Data augmentation**: Random stride offset for better generalization
- **Frame preprocessing**: Automatic resizing, normalization, BGR→RGB conversion
- **Scene labeling**: 3-class labeling (previous/transition/next) based on shot boundaries
- **Video filtering**: Only include videos with sufficient scene changes
- **Memory efficient**: Uses existing VideoReader utility for frame extraction

## Dataset Structure

The dataloader expects:
- `autoshot.json`: Annotation file with shot boundaries
- `autoshot_videos/`: Directory containing MP4 video files

## Parameters

### AutoshotDataset
- `json_path`: Path to autoshot.json (default: ".data/autoshot.json")
- `videos_dir`: Video directory (default: ".data/autoshot_videos") 
- `sequence_length`: Frames per sequence (default: 5)
- `frame_size`: Target frame size (default: (224, 224))
- `stride`: Stride between sequences (default: 1)
- `min_shots`: Minimum shots per video (default: 1)
- `random_offset_range`: Range for random offset around transition center (default: 5)
- `transform`: Optional transform function

### create_autoshot_dataloader()
Additional parameters:
- `batch_size`: Batch size (default: 4)
- `shuffle`: Whether to shuffle (default: True)
- `num_workers`: Data loading workers (default: 2)

## Output Format

Each batch contains:
- **inputs**: Frame tensor `[batch_size, sequence_length, 3, height, width]`
- **labels**: Scene labels `[batch_size, sequence_length]`

The dataloader generates sequences centered around shot boundaries, ensuring each sequence contains frames from previous shot, potentially transition frames, and next shot.

Scene labels follow the shot boundary format:
- **0**: Previous shot (frames ≤ from_frame)
- **1**: Transition frames (from_frame < frame < to_frame)  
- **2**: Next shot (frames ≥ to_frame)

```python
# Example 1: Instant cut (boundary from=131, to=132)
frames = [129, 130, 131, 132, 133]  
labels = [0, 0, 0, 2, 2]  # No transition frames

# Example 2: Fade transition (boundary from=131, to=134)  
frames = [129, 130, 131, 132, 133, 134, 135]
labels = [0, 0, 0, 1, 1, 2, 2]  # Frames 132,133 are transition
```

## Example Usage

See `example_usage.py` for a complete example that demonstrates:
- Creating the dataloader
- Iterating through batches
- Accessing video metadata
- Detecting scene changes within sequences

## Requirements

- PyTorch
- OpenCV (cv2)  
- NumPy
- Existing utils/video_reader.py module

## Testing

Run the built-in test:
```bash
python autoshot_dataloader.py
```

Or run the full example:
```bash
python example_usage.py
``` 