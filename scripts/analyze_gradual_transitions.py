#!/usr/bin/env python3
"""
Script to analyze gradual transitions in the unified AutoShot dataset.

A gradual transition is when there are actual frames between the end of one shot
and the beginning of the next shot (to_frame > from_frame + 1).
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_unified_annotations(json_path: str = ".data/autoshot.json") -> List[Dict]:
    """Load the unified annotations."""
    json_file = Path(json_path)
    if not json_file.exists():
        raise FileNotFoundError(f"Unified annotations not found at {json_path}")
    
    with open(json_file, 'r') as f:
        return json.load(f)


def analyze_transitions(annotations: List[Dict]) -> Dict:
    """Analyze transitions in the dataset."""
    stats = {
        'total_videos': 0,
        'annotated_videos': 0,
        'total_transitions': 0,
        'instant_cuts': 0,
        'gradual_transitions': 0,
        'gradual_transition_lengths': [],
        'videos_with_gradual': 0,
        'by_source': {
            'kuaishou_v2.txt': {'total': 0, 'instant': 0, 'gradual': 0},
            'gt_scenes_dict_baseline_v2.pickle': {'total': 0, 'instant': 0, 'gradual': 0}
        }
    }
    
    for video_data in annotations:
        stats['total_videos'] += 1
        
        # Skip videos without annotations
        if video_data.get('annotation_source', 'none') == 'none':
            continue
            
        stats['annotated_videos'] += 1
        source = video_data['annotation_source']
        transitions = video_data.get('transitions', [])
        
        if not transitions:
            continue
        
        video_has_gradual = False
        
        for transition in transitions:
            transition_type = transition['type']
            
            stats['total_transitions'] += 1
            stats['by_source'][source]['total'] += 1
            
            if transition_type == 'gradual':
                # Gradual transition
                duration = transition.get('duration', 0)
                stats['gradual_transitions'] += 1
                stats['gradual_transition_lengths'].append(duration)
                stats['by_source'][source]['gradual'] += 1
                video_has_gradual = True
            else:
                # Instant cut
                stats['instant_cuts'] += 1
                stats['by_source'][source]['instant'] += 1
        
        if video_has_gradual:
            stats['videos_with_gradual'] += 1
    
    return stats


def print_analysis(stats: Dict):
    """Print the analysis results."""
    print("=== AutoShot Dataset Transition Analysis ===\n")
    
    print(f"📊 Dataset Overview:")
    print(f"  Total videos: {stats['total_videos']}")
    print(f"  Annotated videos: {stats['annotated_videos']}")
    print(f"  Videos with gradual transitions: {stats['videos_with_gradual']}")
    print(f"  Total transitions: {stats['total_transitions']}")
    print()
    
    if stats['total_transitions'] > 0:
        print(f"🎬 Transition Types:")
        print(f"  Instant cuts: {stats['instant_cuts']} ({stats['instant_cuts']/stats['total_transitions']*100:.1f}%)")
        print(f"  Gradual transitions: {stats['gradual_transitions']} ({stats['gradual_transitions']/stats['total_transitions']*100:.1f}%)")
        print()
    else:
        print(f"🎬 No transitions found in dataset")
        print()
    
    if stats['gradual_transitions'] > 0:
        lengths = stats['gradual_transition_lengths']
        print(f"📏 Gradual Transition Lengths:")
        print(f"  Min length: {min(lengths)} frames")
        print(f"  Max length: {max(lengths)} frames")
        print(f"  Average length: {sum(lengths)/len(lengths):.1f} frames")
        print(f"  Median length: {sorted(lengths)[len(lengths)//2]} frames")
        print()
        
        # Distribution of transition lengths
        length_dist = {}
        for length in lengths:
            length_dist[length] = length_dist.get(length, 0) + 1
        
        print(f"🔍 Length Distribution (top 10):")
        for length, count in sorted(length_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {length} frames: {count} transitions")
        print()
    
    print(f"📋 By Annotation Source:")
    for source, source_stats in stats['by_source'].items():
        if source_stats['total'] > 0:
            print(f"  {source}:")
            print(f"    Total boundaries: {source_stats['total']}")
            print(f"    Instant cuts: {source_stats['instant']} ({source_stats['instant']/source_stats['total']*100:.1f}%)")
            print(f"    Gradual transitions: {source_stats['gradual']} ({source_stats['gradual']/source_stats['total']*100:.1f}%)")
    print()


def find_example_gradual_transitions(annotations: List[Dict], num_examples: int = 5) -> List[Dict]:
    """Find example videos with gradual transitions."""
    examples = []
    
    for video_data in annotations:
        transitions = video_data.get('transitions', [])
        
        for transition in transitions:
            if transition['type'] == 'gradual':
                duration = transition.get('duration', 0)
                examples.append({
                    'video': video_data['filename'],
                    'frame': transition['frame'],
                    'duration': duration,
                    'source': video_data['annotation_source']
                })
                
                if len(examples) >= num_examples:
                    break
        
        if len(examples) >= num_examples:
            break
    
    return examples


def main():
    """Main analysis function."""
    try:
        print("Loading unified annotations...")
        annotations = load_unified_annotations()
        
        print("Analyzing transitions...")
        stats = analyze_transitions(annotations)
        
        print_analysis(stats)
        
        # Show examples
        examples = find_example_gradual_transitions(annotations)
        if examples:
            print(f"🎯 Example Gradual Transitions:")
            for i, example in enumerate(examples, 1):
                print(f"  {i}. {example['video']}")
                print(f"     Transition at frame {example['frame']} ({example['duration']} frames duration)")
                print(f"     Source: {example['source']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()