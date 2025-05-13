import json
import os
import pathlib
from pathlib import Path
VIDEO_FOLDERS = [
    '1_video_download',
    '2_ads_game_videos',
    '3_original_videos'
]

VIDEO_SRC = Path('dataset')
METADATA_SRC = Path('dataset/drive-download/kuaishou_v2.txt')

def read_metadata(src, dst):
    with open(src, 'r') as f:
        data = f.read()
    videos = [d.strip() for d in data.split('\n\n') if d.strip()]
    results = {}
    for video in videos:
        lines = [d.replace('[','').replace(']','').strip()
                 for d in video.split('\n') if d.strip()]
        title = lines[0].split('.mp4')[0]+'.mp4'
        metadata = [(line.split(',') if ',' in line else line.split(' ')) for line in lines[1:]]
        results[title] = metadata
    with open(dst, 'w') as f:
        json.dump(results, f, indent=3)
    print('Video metadata count:',len(results))

def read_video_count(base_src, srcs):
    total = 0
    for src in srcs:
        total += len(list(Path(f'{base_src}/{src}').iterdir()))
    print('Video count:',total)

def main():
    read_metadata(METADATA_SRC, Path('dataset/metadata.json'))
    read_video_count(VIDEO_SRC, VIDEO_FOLDERS)

if __name__=='__main__':
    main()