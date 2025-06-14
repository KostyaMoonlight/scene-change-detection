"""
AutoShot Repository Downloader Module

Handles downloading of AutoShot repository files including annotations,
pickle files, and evaluation scripts.
"""

import requests
from pathlib import Path
from typing import Dict
import logging

from scripts.dataset.constants import AUTOSHOT_FILE_URLS

logger = logging.getLogger(__name__)


class AutoShotDownloader:
    """
    Handles downloading of AutoShot repository files with retry logic and error handling.
    """
    
    def __init__(self, target_dir: Path, timeout: int = 30):
        """
        Initialize the downloader.
        
        Args:
            target_dir: Directory to save downloaded files
            timeout: Request timeout in seconds
        """
        self.target_dir = target_dir
        self.timeout = timeout
        self.target_dir.mkdir(exist_ok=True)
        
        # Repository URLs from constants
        self.repo_files = AUTOSHOT_FILE_URLS
    
    def download_file(self, filename: str, url: str) -> bool:
        """
        Download a single file from URL.
        
        Args:
            filename: Name of the file to save
            url: URL to download from
            
        Returns:
            True if download successful, False otherwise
        """
        file_path = self.target_dir / filename
        
        # Skip if file already exists
        if file_path.exists():
            logger.info(f"   ✅ {filename}: Already exists")
            return True
        
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Handle binary vs text files
            if filename.endswith('.pickle'):
                with open(file_path, 'wb') as f:
                    f.write(response.content)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
            
            logger.info(f"   ✅ {filename}: Downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"   ❌ {filename}: Download failed - {str(e)}")
            return False
    
    def download_all_files(self) -> Dict[str, bool]:
        """
        Download all AutoShot repository files.
        
        Returns:
            Dictionary indicating success/failure for each file download
        """
        logger.info("📥 Downloading AutoShot repository files for analysis...")
        
        results = {}
        for filename, url in self.repo_files.items():
            results[filename] = self.download_file(filename, url)
        
        return results
    
    def verify_downloads(self) -> Dict[str, bool]:
        """
        Verify that all expected files were downloaded successfully.
        
        Returns:
            Dictionary indicating which files exist locally
        """
        results = {}
        for filename in self.repo_files.keys():
            file_path = self.target_dir / filename
            results[filename] = file_path.exists()
        
        return results
    
    def get_download_status(self) -> Dict[str, any]:
        """
        Get comprehensive download status information.
        
        Returns:
            Dictionary with download statistics and status
        """
        verification = self.verify_downloads()
        
        return {
            'files_downloaded': sum(verification.values()),
            'files_expected': len(self.repo_files),
            'success_rate': sum(verification.values()) / len(self.repo_files),
            'file_status': verification
        } 