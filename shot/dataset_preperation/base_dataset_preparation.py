import json
from pathlib import Path


class BaseDatasetPreparation:
    def __init__(self, metadata_src:Path, video_srcs:list[Path]):
        self.metadata = json.load(metadata_src.read_text())
        self.videos = self.__get_videos_paths(video_srcs)

    def __get_videos_paths(self, video_srcs:list[Path]):
        videos_paths = []
        for video_src in video_srcs:
            videos_paths.extend(list(video_src.iterdir()))
        return videos_paths

    def iterate_videos(self):
        for video_name, shot_changes in self.metadata.items():
            if video_name in self.videos:
                self._process(video_name, shot_changes)

    def _process(self, video_name, shot_changes):
        raise NotImplementedError()