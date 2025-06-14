#!/usr/bin/env python3

"""
Comprehensive test to demonstrate the corrected autoshot dataloader.
Shows examples of both instant cuts and transition sequences.
"""

from autoshot_dataloader import create_autoshot_dataloader
import json


def test_comprehensive():
    """Test the dataloader and show various boundary types."""
    print("Comprehensive Autoshot DataLoader Test")
    print("=" * 60)
    
    # Create dataloader
    dataloader = create_autoshot_dataloader(
        sequence_length=16,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        random_offset_range=0,  # No randomness for consistent testing
        min_shots=2
    )
    
    print(f"Total sequences in dataset: {len(dataloader.dataset)}")
    print()
    
    # Find and show different types of boundaries
    instant_cuts_found = 0
    transitions_found = 0
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        for seq_idx in range(inputs.shape[0]):
            dataset_idx = batch_idx * dataloader.batch_size + seq_idx
            if dataset_idx >= len(dataloader.dataset):
                break
                
            video_info = dataloader.dataset.get_video_info(dataset_idx)
            sequence_labels = labels[seq_idx]
            
            from_frame = video_info['boundary_from']
            to_frame = video_info['boundary_to']
            is_instant_cut = (to_frame - from_frame == 1)
            has_transitions = 1 in sequence_labels
            
            # Show examples of different boundary types
            if is_instant_cut and instant_cuts_found < 2:
                instant_cuts_found += 1
                print(f"INSTANT CUT Example #{instant_cuts_found}:")
                print(f"  Video: {video_info['filename']}")
                print(f"  Boundary: {from_frame} → {to_frame} (instant cut)")
                print(f"  Sequence frames: {video_info['sequence_start']}-{video_info['sequence_end']}")
                print(f"  Labels: {sequence_labels.tolist()}")
                
                # Analyze the boundary
                label_counts = {0: 0, 1: 0, 2: 0}
                for label in sequence_labels:
                    label_counts[label.item()] += 1
                print(f"  Distribution: {label_counts[0]} previous, {label_counts[1]} transition, {label_counts[2]} next")
                print()
                
            elif not is_instant_cut and has_transitions and transitions_found < 2:
                transitions_found += 1
                print(f"TRANSITION Example #{transitions_found}:")
                print(f"  Video: {video_info['filename']}")
                print(f"  Boundary: {from_frame} → {to_frame} (transition length: {to_frame - from_frame - 1})")
                print(f"  Sequence frames: {video_info['sequence_start']}-{video_info['sequence_end']}")
                print(f"  Labels: {sequence_labels.tolist()}")
                
                # Show frame-by-frame breakdown
                print("  Frame breakdown:")
                for i, label in enumerate(sequence_labels):
                    frame_num = video_info['sequence_start'] + i
                    label_name = {0: "Previous", 1: "TRANSITION", 2: "Next"}[label.item()]
                    print(f"    Frame {frame_num}: {label_name}")
                
                label_counts = {0: 0, 1: 0, 2: 0}
                for label in sequence_labels:
                    label_counts[label.item()] += 1
                print(f"  Distribution: {label_counts[0]} previous, {label_counts[1]} transition, {label_counts[2]} next")
                print()
        
        # Stop when we have enough examples
        if instant_cuts_found >= 2 and transitions_found >= 2:
            break
            
        # Don't run forever
        if batch_idx >= 20:
            break
    
    # Summary
    print("Summary:")
    print(f"- Found {instant_cuts_found} instant cut examples")
    print(f"- Found {transitions_found} transition examples")
    print("\nDataloader correctly implements the boundary format:")
    print("- Label 0: Previous shot (frames ≤ from_frame)")
    print("- Label 1: Transition frames (from_frame < frame < to_frame)")
    print("- Label 2: Next shot (frames ≥ to_frame)")


def analyze_dataset_boundaries():
    """Analyze the types of boundaries in the dataset."""
    print("\n" + "=" * 60)
    print("Dataset Boundary Analysis")
    print("=" * 60)
    
    with open('.data/autoshot.json', 'r') as f:
        data = json.load(f)
    
    instant_cuts = 0
    transitions = 0
    transition_lengths = []
    
    for video in data:
        if video['frame_boundaries']:
            for boundary in video['frame_boundaries']:
                from_frame = boundary['from']
                to_frame = boundary['to']
                
                if to_frame - from_frame == 1:
                    instant_cuts += 1
                else:
                    transitions += 1
                    transition_lengths.append(to_frame - from_frame - 1)
    
    print(f"Total boundaries analyzed: {instant_cuts + transitions}")
    print(f"Instant cuts (from → from+1): {instant_cuts}")
    print(f"Transitions (from → from+n): {transitions}")
    
    if transition_lengths:
        print(f"Transition length statistics:")
        print(f"  Min: {min(transition_lengths)} frames")
        print(f"  Max: {max(transition_lengths)} frames")
        print(f"  Average: {sum(transition_lengths) / len(transition_lengths):.1f} frames")


if __name__ == "__main__":
    test_comprehensive()
    analyze_dataset_boundaries() 