"""Dataset organizer component."""

import shutil
import zipfile
from pathlib import Path
from typing import Dict, Any, List

from ..core.base import BaseProcessor
from ..core.config import (
    get_downloads_path, get_processed_path,
    ADS_GAME_VIDEOS_DIR, VIDEO_DOWNLOAD_DIR,
    DATASET_ZIPS, VIDEO_EXTENSIONS
)


class DatasetOrganizer(BaseProcessor):
    """Extracts and organizes video files from zip archives."""
    
    def process(self) -> Dict[str, Any]:
        """Extract and organize all dataset files."""
        if self.is_complete():
            self.logger.info("Organization already complete")
            return {"success": True, "organized": 0}
        
        self.logger.info("Starting dataset organization")
        
        # Create directories
        processed_path = self.ensure_dir(get_processed_path(self.data_dir))
        ads_path = self.ensure_dir(processed_path / ADS_GAME_VIDEOS_DIR)
        video_path = self.ensure_dir(processed_path / VIDEO_DOWNLOAD_DIR)
        
        downloads_path = get_downloads_path(self.data_dir)
        
        organized_count = 0
        failed = []
        
        # Extract and organize zip files
        for zip_name in DATASET_ZIPS:
            zip_file = downloads_path / zip_name
            
            if not zip_file.exists():
                self.logger.warning(f"Zip file not found: {zip_name}")
                continue
            
            try:
                if "ads_game" in zip_name:
                    target_dir = ads_path
                else:
                    target_dir = video_path
                
                count = self._extract_videos(zip_file, target_dir)
                organized_count += count
                self.logger.info(f"Extracted {count} videos from {zip_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to extract {zip_name}: {e}")
                failed.append(zip_name)
        
        return {
            "success": len(failed) == 0,
            "organized": organized_count,
            "failed": failed
        }
    
    def is_complete(self) -> bool:
        """Check if organization is complete."""
        processed_path = get_processed_path(self.data_dir)
        ads_path = processed_path / ADS_GAME_VIDEOS_DIR
        video_path = processed_path / VIDEO_DOWNLOAD_DIR
        
        if not all(p.exists() for p in [ads_path, video_path]):
            return False
        
        # Check if directories contain video files
        video_count = 0
        for ext in VIDEO_EXTENSIONS:
            video_count += len(list(ads_path.glob(f"*{ext}")))
            video_count += len(list(video_path.glob(f"*{ext}")))
        
        return video_count > 0
    
    def _extract_videos(self, zip_file: Path, target_dir: Path) -> int:
        """Extract video files from zip to target directory."""
        count = 0
        
        with zipfile.ZipFile(zip_file, 'r') as zf:
            for member in zf.namelist():
                # Check if it's a video file
                if any(member.lower().endswith(ext) for ext in VIDEO_EXTENSIONS) and not self.is_macos_internal(member):
                    # Extract to target directory with just filename
                    filename = Path(member).name
                    target_path = target_dir / filename
                    
                    # Skip if already exists
                    if target_path.exists():
                        continue
                    
                    # Extract file
                    with zf.open(member) as source, open(target_path, 'wb') as target:
                        shutil.copyfileobj(source, target)
                    
                    count += 1
        
        return count

    def is_macos_internal(self, member):
        filename = Path(member).name
        return filename.startswith("._")