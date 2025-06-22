"""Logging configuration for dataset preparation."""

import logging
from pathlib import Path
from typing import Optional

from .config import get_data_path, LOG_FILE


def setup_logger(
    name: str, 
    data_dir: str = ".data",
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """Setup logger with file and console output."""
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler
    log_file = get_data_path(data_dir) / LOG_FILE
    log_file.parent.mkdir(exist_ok=True)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger