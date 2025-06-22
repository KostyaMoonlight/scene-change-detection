"""Base classes for dataset preparation components."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from .config import get_data_path
from .logger import setup_logger


class BaseProcessor(ABC):
    """Base class for dataset processing components."""
    
    def __init__(self, data_dir: str = ".data"):
        self.data_dir = data_dir
        self.data_path = get_data_path(data_dir)
        self.logger = setup_logger(self.__class__.__name__, data_dir)
        
    @abstractmethod
    def process(self) -> Dict[str, Any]:
        """Process the component. Returns result dictionary."""
        pass
    
    @abstractmethod
    def is_complete(self) -> bool:
        """Check if processing is already complete."""
        pass
    
    def ensure_dir(self, path: Path) -> Path:
        """Ensure directory exists."""
        path.mkdir(parents=True, exist_ok=True)
        return path


class BaseDownloader(BaseProcessor):
    """Base class for file downloaders."""
    
    def __init__(self, data_dir: str = ".data", timeout: int = 30):
        super().__init__(data_dir)
        self.timeout = timeout
        
    def download_file(self, url: str, filepath: Path, 
                     show_progress: bool = True) -> bool:
        """Download a single file."""
        import requests
        from tqdm import tqdm
        
        try:
            response = requests.get(url, stream=True, timeout=self.timeout)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                if show_progress and total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True, 
                            desc=filepath.name) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            self.logger.info(f"Downloaded: {filepath.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download {url}: {e}")
            return False