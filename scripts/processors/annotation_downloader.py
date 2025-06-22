"""AutoShot annotation downloader component."""

from pathlib import Path
from typing import Dict, Any

from ..core.base import BaseDownloader
from ..core.config import get_autoshot_path, AUTOSHOT_FILES


class AnnotationDownloader(BaseDownloader):
    """Downloads AutoShot annotation files."""
    
    def process(self) -> Dict[str, Any]:
        """Download all annotation files."""
        if self.is_complete():
            self.logger.info("Annotations already downloaded")
            return {"success": True, "downloaded": 0}
        
        self.logger.info("Starting annotation download")
        autoshot_path = self.ensure_dir(get_autoshot_path(self.data_dir))
        
        downloaded = 0
        failed = []
        
        for filename, url in AUTOSHOT_FILES.items():
            filepath = autoshot_path / filename
            
            if filepath.exists():
                self.logger.info(f"Skipping existing file: {filename}")
                continue
            
            if self.download_file(url, filepath):
                downloaded += 1
            else:
                failed.append(filename)
        
        return {
            "success": len(failed) == 0,
            "downloaded": downloaded,
            "failed": failed
        }
    
    def is_complete(self) -> bool:
        """Check if all annotation files are downloaded."""
        autoshot_path = get_autoshot_path(self.data_dir)
        if not autoshot_path.exists():
            return False
        
        for filename in AUTOSHOT_FILES.keys():
            if not (autoshot_path / filename).exists():
                return False
        
        return True