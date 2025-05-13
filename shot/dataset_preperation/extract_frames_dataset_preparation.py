from pathlib import Path

from shot.dataset_preperation.base_dataset_preparation import BaseDatasetPreparation


class ExtractFramesDatasetPreparation(BaseDatasetPreparation):

    def __init__(self, metadata_src:Path, video_srcs:list[Path], frame_count:int = 8, random_shift:int = 6):
        super().__init__(metadata_src, video_srcs)
        self.frame_count = frame_count
        self.random_shift = random_shift

    def prepare(self, video_name, shot_changes):
        shot_ends = self.__get_shot_ends(shot_changes)
        false_zones = [0, shot_ends[0]]



    def __get_shot_ends(self, shot_changes):
        return [int(shot_change[1]) for shot_change in shot_changes]
