import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import cv2
from collections import OrderedDict
import threading
import weakref
from utils.video_reader import VideoReader, FrameData


class AutoshotDataset(Dataset):
    """
    Dataset for Autoshot scene change detection data.
    
    Returns sequences of frames with corresponding scene change labels.
    Each sample contains a sequence of frames and their scene labels.
    """
    
    def __init__(
        self,
        json_path: str = ".data/autoshot.json",
        videos_dir: str = ".data/autoshot_videos",
        sequence_length: int = 16,
        frame_size: Tuple[int, int] = (224, 224),
        min_shots: int = 1,
        random_offset_range: int = 5,
        transform = None,
        cache_size: int = 10
    ):
        """
        Initialize the Autoshot dataset.
        
        Args:
            json_path: Path to the autoshot.json annotation file
            videos_dir: Directory containing the video files
            sequence_length: Number of frames per sequence (window around transition)
            frame_size: Target size for frames (height, width)
            min_shots: Minimum number of shots required for a video to be included
            random_offset_range: Range for random offset around transition center (data augmentation)
            transform: Optional transform to apply to frames
            cache_size: Maximum number of video files to keep open simultaneously
        """
        self.json_path = Path(json_path)
        self.videos_dir = Path(videos_dir)
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.min_shots = min_shots
        self.random_offset_range = random_offset_range
        self.transform = transform
        self.cache_size = cache_size
        
        # Video reader cache (LRU cache)
        self._video_cache = OrderedDict()
        self._cache_lock = threading.Lock()
        
        # Register cleanup when object is deleted
        weakref.finalize(self, self._cleanup_cache, self._video_cache)
        
        self.annotations = self._load_annotations()
        self.valid_videos = self._filter_valid_videos()
        self.sequences = self._generate_sequences()
        
    def _load_annotations(self) -> List[Dict]:
        """Load and return annotations from JSON file."""
        with open(self.json_path, 'r') as f:
            return json.load(f)
    
    def _filter_valid_videos(self) -> List[Dict]:
        """Filter videos that have enough shots and exist on disk."""
        valid_videos = []
        
        for video_data in self.annotations:
            # Check if video has enough shots
            if video_data['num_shots'] < self.min_shots:
                continue
                
            # Check if video file exists
            video_path = self.videos_dir / video_data['filename']
            if not video_path.exists():
                continue
                
            # Check if we have transitions (only annotated videos have transitions)
            if not video_data.get('transitions', []):
                continue
                
            valid_videos.append(video_data)
            
        return valid_videos
    
    def _generate_sequences(self) -> List[Dict]:
        """Generate sequences centered around transitions."""
        sequences = []
        
        for video_data in self.valid_videos:
            total_frames = video_data['total_frames']
            transitions = video_data['transitions']
            
            # Process each transition to create training sequences
            for transition in transitions:
                transition_frame = transition['frame']
                transition_type = transition['type']
                
                # Calculate base sequence start (center the sequence around transition)
                half_length = self.sequence_length // 2
                base_start = transition_frame - half_length
                
                # Generate sequences with random offset for data augmentation
                if self.random_offset_range > 0:
                    import random
                    # Create multiple sequences with different offsets
                    for _ in range(3):  # Generate 3 variations per transition
                        offset = random.randint(-self.random_offset_range, self.random_offset_range)
                        start_frame = base_start + offset
                        
                        # Ensure sequence is within video bounds
                        start_frame = max(0, start_frame)
                        start_frame = min(start_frame, total_frames - self.sequence_length)
                        
                        if start_frame + self.sequence_length <= total_frames:
                            sequence_info = {
                                'video_data': video_data,
                                'start_frame': start_frame,
                                'end_frame': start_frame + self.sequence_length - 1,
                                'frame_positions': list(range(start_frame, start_frame + self.sequence_length)),
                                'transition': transition
                            }
                            sequences.append(sequence_info)
                else:
                    # No random offset, just use centered sequence
                    start_frame = max(0, base_start)
                    start_frame = min(start_frame, total_frames - self.sequence_length)
                    
                    if start_frame + self.sequence_length <= total_frames:
                        sequence_info = {
                            'video_data': video_data,
                            'start_frame': start_frame,
                            'end_frame': start_frame + self.sequence_length - 1,
                            'frame_positions': list(range(start_frame, start_frame + self.sequence_length)),
                            'transition': transition
                        }
                        sequences.append(sequence_info)
                
        return sequences
    
    def _get_cached_video_reader(self, video_path: Path) -> VideoReader:
        """Get a cached video reader or create a new one."""
        video_path_str = str(video_path)
        
        with self._cache_lock:
            # Check if video is already in cache
            if video_path_str in self._video_cache:
                # Move to end (most recently used)
                reader = self._video_cache.pop(video_path_str)
                self._video_cache[video_path_str] = reader
                return reader
            
            # Create new reader
            reader = VideoReader(video_path)
            reader.__enter__()  # Open the video file
            
            # Add to cache
            self._video_cache[video_path_str] = reader
            
            # Evict oldest if cache is full
            while len(self._video_cache) > self.cache_size:
                oldest_path, oldest_reader = self._video_cache.popitem(last=False)
                try:
                    oldest_reader.__exit__(None, None, None)
                except:
                    pass  # Ignore cleanup errors
            
            return reader
    
    @staticmethod
    def _cleanup_cache(cache: OrderedDict):
        """Clean up video readers when dataset is destroyed."""
        for reader in cache.values():
            try:
                reader.__exit__(None, None, None)
            except:
                pass  # Ignore cleanup errors
        cache.clear()
    
    def _get_scene_labels(self, sequence_info: Dict) -> List[int]:
        """
        Generate scene labels for a sequence of frames based on the transition.
        
        Label scheme:
        - 0: Previous shot
        - 1: Gradual transition frames (only for gradual transitions)
        - 2: Next shot
        
        For instant cuts: only use labels 0 and 2.
        For gradual transitions: use 0, 1, 2.
        """
        labels = []
        transition = sequence_info['transition']
        frame_positions = sequence_info['frame_positions']
        
        transition_frame = transition['frame']
        transition_type = transition['type']
        
        for frame_pos in frame_positions:
            if transition_type == 'instant':
                # Instant cut: only previous (0) and next (2) shots
                if frame_pos < transition_frame:
                    labels.append(0)  # Previous shot
                elif frame_pos == transition_frame:
                    labels.append(2)  # Transition frame becomes part of next shot
                else:
                    labels.append(2)  # Next shot
            else:
                # Gradual transition
                duration = transition.get('duration', 0)
                transition_start = transition_frame - duration
                
                if frame_pos < transition_start:
                    labels.append(0)  # Previous shot
                elif frame_pos < transition_frame:
                    labels.append(1)  # Gradual transition frames
                else:
                    labels.append(2)  # Next shot
                
        return labels
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a single frame (resize, normalize, etc.)."""
        # Resize frame
        if frame.shape[:2] != self.frame_size:
            frame = cv2.resize(frame, (self.frame_size[1], self.frame_size[0]))
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # Apply transform if provided
        if self.transform:
            frame = self.transform(frame)
        
        return frame
    
    def __len__(self) -> int:
        """Return the total number of sequences."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence of frames and their corresponding scene labels.
        
        Returns:
            frames: Tensor of shape (sequence_length, height, width, channels)
            labels: Tensor of scene labels for each frame
        """
        sequence_info = self.sequences[idx]
        video_data = sequence_info['video_data']
        frame_positions = sequence_info['frame_positions']
        
        # Get cached video reader
        video_path = self.videos_dir / video_data['filename']
        reader = self._get_cached_video_reader(video_path)
        
        frames = []
        frame_data_list = reader.get_frames(frame_positions)
        
        for frame_data in frame_data_list:
            processed_frame = self._preprocess_frame(frame_data.frame)
            frames.append(processed_frame)
        
        # Get scene labels
        labels = self._get_scene_labels(sequence_info)
        
        # Convert to tensors
        frames_tensor = torch.stack([torch.from_numpy(frame).permute(2, 0, 1) for frame in frames])
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return frames_tensor, labels_tensor
    
    def get_video_info(self, idx: int) -> Dict:
        """Get information about the video for a given sequence index."""
        sequence_info = self.sequences[idx]
        video_data = sequence_info['video_data']
        transition = sequence_info['transition']
        
        return {
            'filename': video_data['filename'],
            'total_frames': video_data['total_frames'],
            'fps': video_data['fps'],
            'num_shots': video_data['num_shots'],
            'sequence_start': sequence_info['start_frame'],
            'sequence_end': sequence_info['end_frame'],
            'transition_frame': transition['frame'],
            'transition_type': transition['type'],
            'transition_duration': transition.get('duration', 0)
        }


def create_autoshot_dataloader(
    json_path: str = ".data/autoshot.json",
    videos_dir: str = ".data/autoshot_videos",
    sequence_length: int = 16,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 2,
    frame_size: Tuple[int, int] = (224, 224),
    min_shots: int = 1,
    random_offset_range: int = 5,
    transform = None,
    cache_size: int = 10
) -> DataLoader:
    """
    Create a DataLoader for the Autoshot dataset.
    
    Args:
        json_path: Path to the autoshot.json annotation file
        videos_dir: Directory containing the video files
        sequence_length: Number of frames per sequence (window around transition)
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        frame_size: Target size for frames (height, width)
        min_shots: Minimum number of shots required for a video to be included
        random_offset_range: Range for random offset around transition center (data augmentation)
        transform: Optional transform to apply to frames
        cache_size: Maximum number of video files to keep open simultaneously per worker
        
    Returns:
        DataLoader for the Autoshot dataset
    """
    dataset = AutoshotDataset(
        json_path=json_path,
        videos_dir=videos_dir,
        sequence_length=sequence_length,
        frame_size=frame_size,
        min_shots=min_shots,
        random_offset_range=random_offset_range,
        transform=transform,
        cache_size=cache_size
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


# Example usage and testing functions
def test_dataloader():
    """Test the dataloader with a small example."""
    print("Testing Autoshot DataLoader...")
    
    # Create dataloader with small batch size for testing
    dataloader = create_autoshot_dataloader(
        sequence_length=16,
        batch_size=2,
        shuffle=False,
        num_workers=0,                # Single process for debugging
        random_offset_range=3,        # Test data augmentation
        min_shots=2,
        cache_size=5                  # Small cache for testing
    )
    
    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Test a few batches
    for i, (frames, labels) in enumerate(dataloader):
        print(f"\nBatch {i+1}:")
        print(f"  Frames shape: {frames.shape}")  # [batch_size, sequence_length, channels, height, width]
        print(f"  Labels shape: {labels.shape}")  # [batch_size, sequence_length]
        print(f"  Labels example: {labels[0].tolist()}")  # First sequence labels
        
        # Show some video info
        video_info = dataloader.dataset.get_video_info(i * dataloader.batch_size)
        print(f"  Video: {video_info['filename']}")
        print(f"  Sequence frames: {video_info['sequence_start']}-{video_info['sequence_end']}")
        print(f"  Transition: frame {video_info['transition_frame']} ({video_info['transition_type']})")
        if video_info['transition_duration'] > 0:
            print(f"  Duration: {video_info['transition_duration']} frames")
        
        if i >= 2:  # Only test first few batches
            break
    
    print("\nDataLoader test completed successfully!")


if __name__ == "__main__":
    test_dataloader() 