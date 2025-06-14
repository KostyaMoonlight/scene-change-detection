import pytest
import cv2
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.video_reader import VideoReader, FrameData


class TestVideoReader:
    
    @pytest.fixture
    def mock_video_path(self):
        """Create a mock video file path."""
        return Path("test_video.mp4")
    
    @pytest.fixture
    def mock_video_capture(self):
        """Create a mock VideoCapture object."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080
        }.get(prop, 0)
        mock_cap.read.return_value = (True, np.zeros((1080, 1920, 3), dtype=np.uint8))
        return mock_cap
    
    @pytest.fixture
    def video_reader(self, mock_video_path, mock_video_capture):
        """Create a VideoReader instance with mocked dependencies."""
        with patch('utils.video_reader.Path.exists', return_value=True), \
             patch('utils.video_reader.cv2.VideoCapture', return_value=mock_video_capture):
            reader = VideoReader(mock_video_path)
            reader.open()
            return reader
    
    def test_init_existing_file(self, mock_video_path):
        """Test initialization with existing video file."""
        with patch('utils.video_reader.Path.exists', return_value=True):
            reader = VideoReader(mock_video_path)
            assert reader.video_path == mock_video_path
            assert reader._cap is None
            assert reader._total_frames is None
            assert reader._fps is None
            assert reader._frame_cache == {}
    
    def test_init_nonexistent_file(self):
        """Test initialization with non-existent video file."""
        with patch('utils.video_reader.Path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                VideoReader("nonexistent.mp4")
    
    def test_context_manager(self, mock_video_path, mock_video_capture):
        """Test context manager functionality."""
        with patch('utils.video_reader.Path.exists', return_value=True), \
             patch('utils.video_reader.cv2.VideoCapture', return_value=mock_video_capture):
            
            with VideoReader(mock_video_path) as reader:
                assert reader._cap is not None
                assert reader.total_frames == 100
                assert reader.fps == 30.0
            
            mock_video_capture.release.assert_called_once()
    
    def test_open_video(self, video_reader):
        """Test opening video capture."""
        assert video_reader._cap is not None
        assert video_reader.total_frames == 100
        assert video_reader.fps == 30.0
    
    def test_open_video_failure(self, mock_video_path):
        """Test video open failure."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        
        with patch('utils.video_reader.Path.exists', return_value=True), \
             patch('utils.video_reader.cv2.VideoCapture', return_value=mock_cap):
            
            reader = VideoReader(mock_video_path)
            with pytest.raises(RuntimeError):
                reader.open()
    
    def test_close_video(self, video_reader):
        """Test closing video capture."""
        video_reader._frame_cache[10] = FrameData(np.zeros((100, 100, 3)), 10, 0.33)
        
        # Store reference to mock before closing
        mock_cap = video_reader._cap
        
        video_reader.close()
        
        mock_cap.release.assert_called()
        assert video_reader._cap is None
        assert len(video_reader._frame_cache) == 0
    
    def test_get_frame_at_position_valid(self, video_reader):
        """Test getting frame at valid position."""
        frame_data = video_reader.get_frame_at_position(50)
        
        assert isinstance(frame_data, FrameData)
        assert frame_data.position == 50
        assert frame_data.timestamp == pytest.approx(50 / 30.0)
        assert frame_data.frame.shape == (1080, 1920, 3)
        
        video_reader._cap.set.assert_called_with(cv2.CAP_PROP_POS_FRAMES, 50)
    
    def test_get_frame_at_position_invalid(self, video_reader):
        """Test getting frame at invalid position."""
        with pytest.raises(ValueError):
            video_reader.get_frame_at_position(-1)
        
        with pytest.raises(ValueError):
            video_reader.get_frame_at_position(100)
    
    def test_get_frame_with_cache(self, video_reader):
        """Test frame caching functionality."""
        cached_frame = FrameData(np.ones((100, 100, 3)), 25, 0.83)
        video_reader._frame_cache[25] = cached_frame
        
        result = video_reader.get_frame_at_position(25, use_cache=True)
        
        assert result is cached_frame
        video_reader._cap.set.assert_not_called()
    
    def test_get_frame_without_cache(self, video_reader):
        """Test frame extraction without caching."""
        cached_frame = FrameData(np.ones((100, 100, 3)), 25, 0.83)
        video_reader._frame_cache[25] = cached_frame
        
        result = video_reader.get_frame_at_position(25, use_cache=False)
        
        assert result is not cached_frame
        video_reader._cap.set.assert_called_with(cv2.CAP_PROP_POS_FRAMES, 25)
    
    def test_get_frame_read_failure(self, video_reader):
        """Test frame read failure."""
        video_reader._cap.read.return_value = (False, None)
        
        with pytest.raises(RuntimeError):
            video_reader.get_frame_at_position(10)
    
    def test_flatten_positions_simple_list(self, video_reader):
        """Test flattening simple position list."""
        positions = [10, 20, 30]
        result = video_reader._flatten_positions(positions)
        assert result == [10, 20, 30]
    
    def test_flatten_positions_nested_list(self, video_reader):
        """Test flattening nested position list."""
        positions = [[10, 20], [30, 40], 50]
        result = video_reader._flatten_positions(positions)
        assert result == [10, 20, 30, 40, 50]
    
    def test_flatten_positions_deeply_nested(self, video_reader):
        """Test flattening deeply nested position list."""
        positions = [[[10, 20], [30]], [[40, 50]], 60]
        result = video_reader._flatten_positions(positions)
        assert result == [10, 20, 30, 40, 50, 60]
    
    def test_get_frames_simple_positions(self, video_reader):
        """Test getting frames from simple position list."""
        positions = [10, 20, 30]
        frames = video_reader.get_frames(positions)
        
        assert len(frames) == 3
        assert all(isinstance(f, FrameData) for f in frames)
        assert [f.position for f in frames] == [10, 20, 30]
    
    def test_get_frames_nested_positions(self, video_reader):
        """Test getting frames from nested position list."""
        positions = [[10, 20], [30, 40]]
        frames = video_reader.get_frames(positions)
        
        assert len(frames) == 4
        assert [f.position for f in frames] == [10, 20, 30, 40]
    
    def test_get_frames_duplicate_positions(self, video_reader):
        """Test getting frames with duplicate positions."""
        positions = [10, 20, 10, 30, 20]
        frames = video_reader.get_frames(positions)
        
        assert len(frames) == 3
        assert [f.position for f in frames] == [10, 20, 30]
    
    def test_get_frames_from_ranges(self, video_reader):
        """Test getting frames from position ranges."""
        ranges = [[10, 15], [20, 25]]
        frames = video_reader.get_frames_from_ranges(ranges)
        
        expected_positions = list(range(10, 16)) + list(range(20, 26))
        assert len(frames) == len(expected_positions)
        assert [f.position for f in frames] == sorted(expected_positions)
    
    def test_get_frames_from_ranges_with_step(self, video_reader):
        """Test getting frames from ranges with step."""
        ranges = [[10, 20]]
        frames = video_reader.get_frames_from_ranges(ranges, step=2)
        
        expected_positions = list(range(10, 21, 2))
        assert [f.position for f in frames] == expected_positions
    
    def test_get_frames_from_ranges_reversed(self, video_reader):
        """Test getting frames from reversed ranges."""
        ranges = [[15, 10], [25, 20]]
        frames = video_reader.get_frames_from_ranges(ranges)
        
        expected_positions = list(range(10, 16)) + list(range(20, 26))
        assert [f.position for f in frames] == sorted(expected_positions)
    
    def test_get_frames_around_positions(self, video_reader):
        """Test getting frames around positions with window."""
        positions = [50]
        frames = video_reader.get_frames_around_positions(positions, window_size=5)
        
        expected_positions = list(range(48, 53))
        assert [f.position for f in frames] == expected_positions
    
    def test_get_frames_around_positions_edge_cases(self, video_reader):
        """Test getting frames around positions at video edges."""
        positions = [2, 97]
        frames = video_reader.get_frames_around_positions(positions, window_size=5)
        
        expected_positions = list(range(0, 5)) + list(range(95, 100))
        assert [f.position for f in frames] == sorted(set(expected_positions))
    
    def test_extract_frames_batch_no_output(self, video_reader):
        """Test batch frame extraction without saving."""
        positions = [10, 20, 30]
        frames = video_reader.extract_frames_batch(positions)
        
        assert len(frames) == 3
        assert [f.position for f in frames] == [10, 20, 30]
    
    def test_extract_frames_batch_with_output(self, video_reader):
        """Test batch frame extraction with saving."""
        positions = [10, 20]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            with patch('utils.video_reader.cv2.imwrite') as mock_imwrite:
                frames = video_reader.extract_frames_batch(positions, output_dir)
                
                assert len(frames) == 2
                assert mock_imwrite.call_count == 2
                
                expected_files = [
                    str(output_dir / "frame_000010.jpg"),
                    str(output_dir / "frame_000020.jpg")
                ]
                actual_files = [call[0][0] for call in mock_imwrite.call_args_list]
                assert sorted(actual_files) == sorted(expected_files)
    
    def test_get_video_info(self, video_reader):
        """Test getting video information."""
        info = video_reader.get_video_info()
        
        expected_info = {
            'path': str(video_reader.video_path),
            'total_frames': 100,
            'fps': 30.0,
            'duration_seconds': 100 / 30.0,
            'width': 1920,
            'height': 1080,
        }
        
        assert info == expected_info


class TestFrameData:
    """Test the FrameData dataclass."""
    
    def test_frame_data_creation(self):
        """Test creating FrameData instance."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_data = FrameData(frame=frame, position=10, timestamp=0.33)
        
        assert np.array_equal(frame_data.frame, frame)
        assert frame_data.position == 10
        assert frame_data.timestamp == 0.33


if __name__ == "__main__":
    pytest.main([__file__]) 