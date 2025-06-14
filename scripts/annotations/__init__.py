"""
Annotations module for AutoShot dataset analysis.

This module contains components for downloading, parsing, and analyzing
AutoShot annotations and related repository files.
"""

from .autoshot_downloader import AutoShotDownloader
from .parser import AnnotationParser
from .pickle_handler import PickleHandler

__all__ = ['AutoShotDownloader', 'AnnotationParser', 'PickleHandler'] 