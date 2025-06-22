"""Dataset unification component."""

import json
import shutil
import pickle
import cv2
import os
import contextlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from ..core.base import BaseProcessor
from ..core.config import (
    get_processed_path, get_autoshot_path, get_unified_videos_path,
    get_unified_annotations_path, ADS_GAME_VIDEOS_DIR, VIDEO_DOWNLOAD_DIR,
    VIDEO_EXTENSIONS
)


class DatasetUnifier(BaseProcessor):
    """Unifies videos and annotations into standardized format."""
    

    
    def process(self) -> Dict[str, Any]:
        """Create unified dataset structure."""
        if self.is_complete():
            self.logger.info("Unification already complete")
            return {"success": True, "unified": 0}
        
        self.logger.info("Starting dataset unification")
        
        # Create unified structure
        unified_videos_path = self.ensure_dir(get_unified_videos_path(self.data_dir))
        annotations = self._load_annotations()
        
        # Collect and copy videos
        videos_data = []
        copied_count = 0
        annotated_count = 0
        
        processed_path = get_processed_path(self.data_dir)
        
        # Process videos from both directories
        for subdir in [ADS_GAME_VIDEOS_DIR, VIDEO_DOWNLOAD_DIR]:
            video_dir = processed_path / subdir
            if not video_dir.exists():
                continue
            
            for video_file in self._get_video_files(video_dir):
                video_name = video_file.name
                target_path = unified_videos_path / video_name
                
                # Validate video file before processing
                if not self._is_valid_video(video_file):
                    self.logger.warning(f"Skipping invalid video: {video_name}")
                    continue
                
                # Copy video if not exists
                if not target_path.exists():
                    shutil.copy2(video_file, target_path)
                    copied_count += 1
                
                # Create video metadata
                video_data = self._create_video_metadata(
                    video_file, target_path, subdir, annotations
                )
                if video_data:  # Only add if metadata creation succeeded
                    videos_data.append(video_data)
                    # Track if video has annotations
                    if video_data.get('annotation_source', 'none') != 'none':
                        annotated_count += 1
        
        # Save unified annotations
        unified_file = get_unified_annotations_path(self.data_dir)
        with open(unified_file, 'w') as f:
            json.dump(videos_data, f, indent=2)
        
        # Calculate annotation statistics
        kuaishou_count = sum(1 for v in videos_data if v.get('annotation_source') == 'kuaishou_v2.txt')
        pickle_count = sum(1 for v in videos_data if v.get('annotation_source') == 'gt_scenes_dict_baseline_v2.pickle')
        no_annotation_count = len(videos_data) - annotated_count
        
        # Calculate total shots and transitions across all annotated videos
        total_shots = sum(v.get('num_shots', 0) for v in videos_data if v.get('annotation_source', 'none') != 'none')
        total_transitions = sum(v.get('num_transitions', 0) for v in videos_data if v.get('annotation_source', 'none') != 'none')
        kuaishou_shots = sum(v.get('num_shots', 0) for v in videos_data if v.get('annotation_source') == 'kuaishou_v2.txt')
        kuaishou_transitions = sum(v.get('num_transitions', 0) for v in videos_data if v.get('annotation_source') == 'kuaishou_v2.txt')
        pickle_shots = sum(v.get('num_shots', 0) for v in videos_data if v.get('annotation_source') == 'gt_scenes_dict_baseline_v2.pickle')
        pickle_transitions = sum(v.get('num_transitions', 0) for v in videos_data if v.get('annotation_source') == 'gt_scenes_dict_baseline_v2.pickle')
        
        self.logger.info(f"Unified {len(videos_data)} videos, copied {copied_count}")
        self.logger.info(f"Videos with annotations: {annotated_count}/{len(videos_data)} ({annotated_count/len(videos_data)*100:.1f}%)")
        self.logger.info(f"  - Kuaishou annotations: {kuaishou_count} videos")
        self.logger.info(f"  - Pickle annotations: {pickle_count} videos")
        self.logger.info(f"  - No annotations: {no_annotation_count} videos")
        self.logger.info(f"Total shots in annotated videos: {total_shots}")
        self.logger.info(f"  - Kuaishou shots: {kuaishou_shots}")
        self.logger.info(f"  - Pickle shots: {pickle_shots}")
        self.logger.info(f"Total transitions in annotated videos: {total_transitions}")
        self.logger.info(f"  - Kuaishou transitions: {kuaishou_transitions}")
        self.logger.info(f"  - Pickle transitions: {pickle_transitions}")
        
        return {
            "success": True,
            "unified": len(videos_data),
            "copied": copied_count,
            "annotated": annotated_count,
            "kuaishou_annotations": kuaishou_count,
            "pickle_annotations": pickle_count,
            "no_annotations": no_annotation_count,
            "total_shots": total_shots,
            "kuaishou_shots": kuaishou_shots,
            "pickle_shots": pickle_shots,
            "total_transitions": total_transitions,
            "kuaishou_transitions": kuaishou_transitions,
            "pickle_transitions": pickle_transitions
        }
    
    def is_complete(self) -> bool:
        """Check if unification is complete."""
        unified_videos_path = get_unified_videos_path(self.data_dir)
        unified_file = get_unified_annotations_path(self.data_dir)
        
        if not unified_file.exists() or not unified_videos_path.exists():
            return False
        
        # Check if unified videos directory has content
        video_count = sum(
            len(list(unified_videos_path.glob(f"*{ext}"))) 
            for ext in VIDEO_EXTENSIONS
        )
        
        return video_count > 0
    
    def _load_annotations(self) -> Dict[str, Any]:
        """Load all available annotations."""
        autoshot_path = get_autoshot_path(self.data_dir)
        annotations = {"kuaishou": {}, "pickle": {}}
        
        kuaishou_file = autoshot_path / "kuaishou_v2.txt"
        if kuaishou_file.exists():
            annotations["kuaishou"] = self._parse_kuaishou(kuaishou_file)
            self.logger.info(f"Loaded {len(annotations['kuaishou'])} Kuaishou annotations")
        
        pickle_file = autoshot_path / "gt_scenes_dict_baseline_v2.pickle"
        if pickle_file.exists():
            with open(pickle_file, 'rb') as f:
                annotations["pickle"] = pickle.load(f)
            self.logger.info(f"Loaded {len(annotations['pickle'])} pickle annotations")
            
            # Debug: show some pickle keys vs video names
            pickle_keys = list(annotations["pickle"].keys())[:5]
            self.logger.debug(f"Sample pickle keys: {pickle_keys}")
        
        return annotations
    
    def _parse_kuaishou(self, file_path: Path) -> Dict[str, Any]:
        """Parse kuaishou annotation file."""
        annotations = {}
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            try:
                parts = line.split()
                # Check if this line contains video name and total frames
                if len(parts) == 2 and parts[0].endswith('.mp4'):
                    video_name = parts[0]
                    total_frames = int(parts[1])
                    boundaries = []
                    
                    # Read subsequent lines for boundaries until next video or end of file
                    i += 1
                    current_boundary_line = ""
                    
                    while i < len(lines):
                        line = lines[i].strip()
                        if not line:
                            i += 1
                            continue
                        
                        # Check if this is a new video entry
                        parts = line.split()
                        if len(parts) == 2 and parts[0].endswith('.mp4'):
                            # This is a new video entry, don't increment i
                            break
                        
                        # Handle multi-line boundaries by checking indentation
                        if line.startswith(' ' * 40) or line.startswith('\t' * 5):  # Continuation line
                            current_boundary_line += " " + line.strip()
                        else:
                            # Process previous boundary line if exists
                            if current_boundary_line:
                                try:
                                    parsed_boundaries = self._parse_boundary_line(current_boundary_line)
                                    # Validate boundary pairs are in ascending order
                                    if self._validate_boundary_pairs(parsed_boundaries, current_boundary_line):
                                        boundaries.extend(parsed_boundaries)
                                except ValueError:
                                    self.logger.warning(f"Failed to parse boundary: {current_boundary_line}")
                            
                            # Start new boundary line
                            current_boundary_line = line
                        
                        i += 1
                    
                    # Process final boundary line
                    if current_boundary_line:
                        try:
                            parsed_boundaries = self._parse_boundary_line(current_boundary_line)
                            # Validate boundary pairs are in ascending order
                            if self._validate_boundary_pairs(parsed_boundaries, current_boundary_line):
                                boundaries.extend(parsed_boundaries)
                        except ValueError:
                            self.logger.warning(f"Failed to parse boundary: {current_boundary_line}")
                    
                    annotations[video_name] = {
                        "total_frames": total_frames,
                        "boundaries": boundaries
                    }
                else:
                    i += 1
                    
            except (ValueError, IndexError) as e:
                self.logger.warning(f"Failed to parse line: {line}")
                i += 1
        
        return annotations
    
    def _parse_boundary_line(self, line: str) -> List[int]:
        """Parse a single boundary line handling various formats."""
        boundaries = []
        
        # Remove brackets if present: [73,82] -> 73,82
        line = line.strip('[]')
        
        # Handle comma-separated values with potential trailing commas
        if ',' in line:
            # Split by comma and clean up
            parts = [part.strip() for part in line.split(',') if part.strip()]
            for part in parts:
                if part.isdigit():
                    boundaries.append(int(part))
                elif ' ' in part:
                    # Handle space-separated numbers within comma-separated parts
                    sub_parts = part.split()
                    for sub_part in sub_parts:
                        if sub_part.isdigit():
                            boundaries.append(int(sub_part))
        else:
            # Handle space-separated values
            parts = line.split()
            for part in parts:
                if part.isdigit():
                    boundaries.append(int(part))
        
        return boundaries
    
    def _validate_boundary_pairs(self, boundaries: List[int], original_line: str) -> bool:
        """Validate that boundary pairs are in ascending order within the line.
        
        For pairs like [100, 101, 150, 54], we should skip the entire line
        because 150 > 54 (not ascending within the pair).
        """
        if len(boundaries) < 2:
            return True
        
        # Check pairs: each pair should be [start, end] where start <= end
        for i in range(0, len(boundaries), 2):
            if i + 1 < len(boundaries):
                start, end = boundaries[i], boundaries[i + 1]
                if start > end:
                    self.logger.warning(f"Skipping boundary line with non-ascending pair {start},{end}: {original_line}")
                    return False
        
        return True
    
    def _get_video_files(self, directory: Path) -> List[Path]:
        """Get all video files from directory."""
        video_files = []
        for ext in VIDEO_EXTENSIONS:
            video_files.extend(directory.glob(f"*{ext}"))
        return video_files
    
    def _is_valid_video(self, video_path: Path) -> bool:
        """Check if video file is valid and readable."""
        if not video_path.exists() or video_path.stat().st_size == 0:
            return False
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            is_valid = cap.isOpened()
            if is_valid:
                # Quick check if we can read frame count
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                is_valid = frame_count > 0
            cap.release()
            return is_valid
        except Exception as e:
            self.logger.warning(f"Error checking video {video_path}: {e}")
            return False
    
    def _create_video_metadata(
        self, 
        original_path: Path, 
        unified_path: Path,
        source_dir: str,
        annotations: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Create metadata for a video file."""
        video_name = original_path.name
        
        # Get video info using OpenCV
        video_info = self._get_video_info(unified_path)
        
        # Skip if video is invalid (total_frames = 0)
        if video_info["total_frames"] == 0:
            self.logger.warning(f"Skipping video with no frames: {video_name}")
            return None
        
        # Find annotations
        annotation_data = self._find_annotations(video_name, annotations)
        
        metadata = {
            "filename": video_name,
            "total_frames": video_info["total_frames"],
            "fps": video_info["fps"],
            "original_path": f"processed/{source_dir}/{video_name}",
            "current_path": f"autoshot_videos/{video_name}",
            "original_folder": source_dir,
            "annotation_source": annotation_data["source"],
        }
        
        if annotation_data["data"]:
            metadata.update(annotation_data["data"])
        else:
            metadata["transitions"] = []
            metadata["num_shots"] = 1
            metadata["num_transitions"] = 0
        
        return metadata
    
    def _get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get video information using OpenCV."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.logger.warning(f"Failed to open video: {video_path}")
                return {"total_frames": 0, "fps": 25.0}
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            # Validate values
            if total_frames <= 0 or fps <= 0:
                self.logger.warning(f"Invalid video properties for {video_path}: frames={total_frames}, fps={fps}")
                return {"total_frames": 0, "fps": 25.0}
            
            return {
                "total_frames": total_frames,
                "fps": fps
            }
        except Exception as e:
            self.logger.warning(f"Error reading video {video_path}: {e}")
            return {"total_frames": 0, "fps": 25.0}
    
    def _find_annotations(self, video_name: str, annotations: Dict[str, Any]) -> Dict[str, Any]:
        """Find annotations for a video."""
        # Priority: pickle > kuaishou
        
        # Check pickle annotations first
        if video_name in annotations["pickle"]:
            segments = annotations["pickle"][video_name]
            transitions = self._segments_to_transitions(segments)
            # Safe calculation of num_shots
            try:
                num_segments = len(segments) if segments is not None else 0
            except:
                num_segments = 0
            self.logger.debug(f"Found pickle annotation for {video_name}: {num_segments} segments")
            return {
                "source": "gt_scenes_dict_baseline_v2.pickle",
                "data": {
                    "transitions": transitions,
                    "num_shots": num_segments,
                    "num_transitions": len(transitions)
                }
            }
        
        # Try matching video name without extension for pickle
        video_name_no_ext = video_name.rsplit('.', 1)[0] if '.' in video_name else video_name
        if video_name_no_ext in annotations["pickle"]:
            segments = annotations["pickle"][video_name_no_ext]
            transitions = self._segments_to_transitions(segments)
            # Safe calculation of num_shots
            try:
                num_segments = len(segments) if segments is not None else 0
            except:
                num_segments = 0
            self.logger.debug(f"Found pickle annotation for {video_name} using name without extension {video_name_no_ext}: {num_segments} segments")
            return {
                "source": "gt_scenes_dict_baseline_v2.pickle",
                "data": {
                    "transitions": transitions,
                    "num_shots": num_segments,
                    "num_transitions": len(transitions)
                }
            }
        
        # Check kuaishou annotations
        if video_name in annotations["kuaishou"]:
            kuaishou_data = annotations["kuaishou"][video_name]
            transitions = self._boundaries_to_transitions(kuaishou_data["boundaries"])
            self.logger.debug(f"Found kuaishou annotation for {video_name}: {len(transitions)} transitions")
            return {
                "source": "kuaishou_v2.txt",
                "data": {
                    "transitions": transitions,
                    "num_shots": len(transitions) + 1 if transitions else 1,
                    "num_transitions": len(transitions),
                    "original_total_frames": kuaishou_data["total_frames"]
                }
            }
        
        # Debug: log when no annotation is found
        self.logger.debug(f"No annotation found for {video_name}")
        return {"source": "none", "data": None}
    
    def _segments_to_transitions(self, segments) -> List[Dict[str, Any]]:
        """Convert pickle segments to transitions."""
        # Handle numpy arrays and other array-like objects
        try:
            # Convert to list if it's a numpy array or similar
            if hasattr(segments, 'tolist'):
                segments = segments.tolist()
            elif hasattr(segments, '__len__') and hasattr(segments, '__getitem__'):
                # It's array-like, convert to list
                segments = list(segments)
            
            # Check if empty
            if len(segments) == 0:
                return []
            
            transitions = []
            for i in range(len(segments) - 1):
                # Ensure each segment is also converted from numpy if needed
                seg_i = segments[i].tolist() if hasattr(segments[i], 'tolist') else segments[i]
                seg_next = segments[i + 1].tolist() if hasattr(segments[i + 1], 'tolist') else segments[i + 1]
                
                # Transition occurs at the end of current shot
                transition_frame = int(seg_i[1])
                next_shot_start = int(seg_next[0])
                
                # Determine transition type
                if next_shot_start <= transition_frame + 1:
                    # Instant cut
                    transitions.append({
                        "frame": transition_frame,
                        "type": "instant"
                    })
                else:
                    # Gradual transition
                    duration = next_shot_start - transition_frame - 1
                    transitions.append({
                        "frame": transition_frame,
                        "type": "gradual",
                        "duration": duration
                    })
            
            return transitions
            
        except Exception as e:
            self.logger.warning(f"Error processing pickle segments: {e}")
            return []
    
    def _boundaries_to_transitions(self, boundaries: List[int]) -> List[Dict[str, Any]]:
        """Convert boundary list to transitions.
        
        Kuaishou format has boundary pairs like [130, 131, 254, 255] where:
        - 130,131 represents a shot boundary transition
        - 254,255 represents another shot boundary transition
        
        These should be converted to transition events.
        """
        if not boundaries:
            return []
        
        transitions = []
        
        # Process pairs of boundaries
        for i in range(0, len(boundaries), 2):
            if i + 1 < len(boundaries):
                start_frame = boundaries[i]
                end_frame = boundaries[i + 1]
                
                # Determine transition type based on duration
                if end_frame <= start_frame + 1:
                    # Instant cut
                    transitions.append({
                        "frame": end_frame,
                        "type": "instant"
                    })
                else:
                    # Gradual transition
                    duration = end_frame - start_frame - 1
                    transitions.append({
                        "frame": end_frame,
                        "type": "gradual",
                        "duration": duration
                    })
        
        return transitions