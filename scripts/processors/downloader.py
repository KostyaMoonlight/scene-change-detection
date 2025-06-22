"""Dataset downloader component."""

import gdown
import os
import requests
from pathlib import Path
from typing import Dict, Any, List

from ..core.base import BaseDownloader
from ..core.config import (
    get_downloads_path, REQUIRED_DATASET_FILES, 
    AUTOSHOT_GOOGLE_DRIVE_FOLDER_ID, GOOGLE_DRIVE_FILE_IDS
)


class DatasetDownloader(BaseDownloader):
    """Downloads specific dataset files from Google Drive."""
    
    def process(self) -> Dict[str, Any]:
        """Download required dataset files from AutoShot Google Drive."""
        if self.is_complete():
            self.logger.info("Download already complete")
            return {"success": True, "downloaded": 0, "skipped": len(REQUIRED_DATASET_FILES)}
        
        self.logger.info("Starting selective AutoShot dataset download")
        downloads_path = self.ensure_dir(get_downloads_path(self.data_dir))
        
        # First, try to download specific files if we can discover them
        downloaded_count = 0
        failed_files = []
        
        # Method 1: Use configured file IDs if available
        if GOOGLE_DRIVE_FILE_IDS:
            self.logger.info(f"Using configured file IDs for {len(GOOGLE_DRIVE_FILE_IDS)} files")
            downloaded_count, failed_files = self._download_individual_files(
                GOOGLE_DRIVE_FILE_IDS, downloads_path
            )
        else:
            # Method 2: Try to discover files (placeholder)
            try:
                discovered_files = self._discover_drive_files()
                if discovered_files:
                    self.logger.info(f"Discovered {len(discovered_files)} files")
                    downloaded_count, failed_files = self._download_individual_files(
                        discovered_files, downloads_path
                    )
                else:
                    # Method 3: Fallback to manual instructions
                    self.logger.warning("No file IDs configured and auto-discovery unavailable")
                    return self._manual_download_instructions(downloads_path)
                    
            except Exception as e:
                self.logger.error(f"Auto-discovery failed: {e}")
                return self._manual_download_instructions(downloads_path)
        
        return {
            "success": len(failed_files) == 0,
            "downloaded": downloaded_count,
            "failed": failed_files,
            "files": [f for f in REQUIRED_DATASET_FILES if f not in failed_files]
        }
    
    def is_complete(self) -> bool:
        """Check if all required files are downloaded and complete."""
        downloads_path = get_downloads_path(self.data_dir)
        if not downloads_path.exists():
            return False
        
        # Get file IDs for validation if available
        file_ids = GOOGLE_DRIVE_FILE_IDS or {}
        
        for filename in REQUIRED_DATASET_FILES:
            filepath = downloads_path / filename
            file_id = file_ids.get(filename, "")
            
            # Use smart check if we have file ID, otherwise basic existence check
            if file_id:
                if not self._is_file_complete(filepath, file_id):
                    return False
            else:
                if not filepath.exists():
                    return False
        
        return True
    
    def _discover_drive_files(self) -> Dict[str, str]:
        """
        Attempt to discover file IDs from Google Drive folder.
        This is a placeholder - actual implementation would require
        Google Drive API or web scraping.
        """
        # Placeholder for file discovery
        # In practice, this would use Google Drive API or parse the folder page
        return {}
    
    def _is_file_complete(self, filepath: Path, file_id: str) -> bool:
        """
        Check if a file is completely downloaded by validating size against remote.
        """
        if not filepath.exists():
            return False
        
        try:
            # Get local file size
            local_size = filepath.stat().st_size
            
            # If file is very small (< 1KB), likely incomplete or corrupted
            if local_size < 1024:
                self.logger.warning(f"File {filepath.name} is suspiciously small ({local_size} bytes), re-downloading")
                return False
            
            # Try to get remote file size for comparison
            remote_size = self._get_remote_file_size(file_id)
            if remote_size and remote_size > 0:
                if local_size != remote_size:
                    self.logger.warning(f"File {filepath.name} size mismatch: local={local_size}, remote={remote_size}")
                    return False
                else:
                    self.logger.info(f"File {filepath.name} verified: {local_size} bytes")
                    return True
            
            # If we can't get remote size, assume large files are complete
            # This is a reasonable assumption for multi-GB files
            if local_size > 100 * 1024 * 1024:  # > 100MB
                self.logger.info(f"File {filepath.name} appears complete: {local_size} bytes")
                return True
            
            # For smaller files without size verification, be conservative
            self.logger.warning(f"Cannot verify {filepath.name} completeness, re-downloading")
            return False
            
        except Exception as e:
            self.logger.warning(f"Error checking file {filepath.name}: {e}")
            return False
    
    def _get_remote_file_size(self, file_id: str) -> int:
        """
        Attempt to get the remote file size from Google Drive.
        """
        try:
            url = f"https://drive.google.com/uc?id={file_id}"
            response = requests.head(url, allow_redirects=True, timeout=10)
            
            if 'content-length' in response.headers:
                return int(response.headers['content-length'])
            
            # Try alternative method with export parameter
            url = f"https://drive.google.com/uc?id={file_id}&export=download"
            response = requests.head(url, allow_redirects=True, timeout=10)
            
            if 'content-length' in response.headers:
                return int(response.headers['content-length'])
                
        except Exception as e:
            self.logger.debug(f"Could not get remote size for {file_id}: {e}")
        
        return 0
    
    def _download_individual_files(self, file_map: Dict[str, str], 
                                 downloads_path: Path) -> tuple[int, List[str]]:
        """Download individual files using their IDs."""
        downloaded = 0
        failed = []
        
        for filename, file_id in file_map.items():
            if filename not in REQUIRED_DATASET_FILES:
                continue
                
            filepath = downloads_path / filename
            
            # Smart file existence check
            if self._is_file_complete(filepath, file_id):
                self.logger.info(f"Skipping complete file: {filename}")
                continue
            
            try:
                self.logger.info(f"Downloading: {filename}")
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, str(filepath), quiet=False)
                downloaded += 1
                
            except Exception as e:
                self.logger.error(f"Failed to download {filename}: {e}")
                failed.append(filename)
        
        return downloaded, failed
    
    def _manual_download_instructions(self, downloads_path: Path) -> Dict[str, Any]:
        """Provide manual download instructions."""
        folder_url = f"https://drive.google.com/drive/folders/{AUTOSHOT_GOOGLE_DRIVE_FOLDER_ID}"
        
        self.logger.warning("⚠️ Automatic download not available")
        self.logger.info("📥 Manual download required:")
        self.logger.info(f"   1. Visit: {folder_url}")
        self.logger.info("   2. Download these files:")
        for filename in REQUIRED_DATASET_FILES:
            self.logger.info(f"      - {filename}")
        self.logger.info(f"   3. Place files in: {downloads_path}")
        self.logger.info("   4. Re-run the script")
        
        return {
            "success": False,
            "downloaded": 0,
            "failed": REQUIRED_DATASET_FILES,
            "manual_url": folder_url,
            "manual_required": True,
            "instructions": f"Download {len(REQUIRED_DATASET_FILES)} files manually"
        }
    
    def download_with_file_ids(self, file_ids: Dict[str, str]) -> Dict[str, Any]:
        """Download files using provided file IDs."""
        downloads_path = self.ensure_dir(get_downloads_path(self.data_dir))
        
        self.logger.info("Downloading with provided file IDs")
        downloaded, failed = self._download_individual_files(file_ids, downloads_path)
        
        return {
            "success": len(failed) == 0,
            "downloaded": downloaded,
            "failed": failed
        }