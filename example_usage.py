#!/usr/bin/env python3

"""
Example usage of the Autoshot DataLoader for scene change detection.

This script demonstrates how to use the autoshot_dataloader to iterate through
video sequences and their corresponding scene change labels.
"""

from autoshot_dataloader import create_autoshot_dataloader
import torch


def main():
    """Main example demonstrating dataloader usage."""
    print("Autoshot DataLoader Example")
    print("=" * 50)
    
    # Create the dataloader
    dataloader = create_autoshot_dataloader(
        sequence_length=16,           # 16 frames per sequence (window around transition)
        batch_size=2,                 # 2 sequences per batch
        shuffle=True,                 # Shuffle the data
        num_workers=0,                # Single process for this example
        min_shots=2,                  # Only include videos with at least 2 shots
        frame_size=(224, 224),        # Resize frames to 224x224
        random_offset_range=3         # Random offset ±3 frames for data augmentation
    )
    
    print(f"Total sequences in dataset: {len(dataloader.dataset)}")
    print(f"Total batches: {len(dataloader)}")
    print()
    
    # Iterate through the first few batches
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Input frames shape: {inputs.shape}")
        print(f"  Labels shape: {labels.shape}")
        
        # Process each sequence in the batch
        for seq_idx in range(inputs.shape[0]):
            frames = inputs[seq_idx]  # Shape: [16, 3, 224, 224]
            sequence_labels = labels[seq_idx]  # Shape: [16]
            
            # Get video information for this sequence
            dataset_idx = batch_idx * dataloader.batch_size + seq_idx
            if dataset_idx < len(dataloader.dataset):
                video_info = dataloader.dataset.get_video_info(dataset_idx)
                
                print(f"    Sequence {seq_idx + 1}:")
                print(f"      Video: {video_info['filename']}")
                print(f"      Frames: {video_info['sequence_start']}-{video_info['sequence_end']}")
                print(f"      Boundary: {video_info['boundary_from']}-{video_info['boundary_to']}")
                print(f"      Scene labels: {sequence_labels.tolist()}")
                
                # Analyze label distribution
                label_counts = {0: 0, 1: 0, 2: 0}
                for label in sequence_labels:
                    label_counts[label.item()] += 1
                
                print(f"      Label distribution: Previous={label_counts[0]}, Transition={label_counts[1]}, Next={label_counts[2]}")
                
                # Show frame-by-frame analysis (first few and last few frames)
                label_names = {0: "Prev", 1: "Trans", 2: "Next"}
                first_frames = [f"F{i}:{label_names[sequence_labels[i].item()]}" for i in range(min(5, len(sequence_labels)))]
                last_frames = [f"F{i}:{label_names[sequence_labels[i].item()]}" for i in range(max(0, len(sequence_labels)-5), len(sequence_labels))]
                print(f"      Frame analysis: {' '.join(first_frames)} ... {' '.join(last_frames)}")
                print()
        
        # Only show first 3 batches for this example
        if batch_idx >= 2:
            break
    
    print("Example completed!")
    
    # Demonstrate the expected format from your request
    print("\nDemonstrating the requested format:")
    print("-" * 40)
    
    # Take first item from dataloader
    for item in dataloader:
        inputs, labels = item
        
        # Show the format you requested
        print("for item in dataloader:")
        print("  inputs, labels = item")
        print(f"  frames = {[f'frame_{i+1}' for i in range(min(5, inputs.shape[1]))]}" + 
              f" ... {[f'frame_{inputs.shape[1]-4+i}' for i in range(4)]}")
        print(f"  labels = {labels[0][:5].tolist()} ... {labels[0][-4:].tolist()}")  # First sequence in batch
        print(f"  # Label meanings: 0=Previous shot, 1=Transition, 2=Next shot")
        
        # Show actual tensor shapes
        print(f"\nActual shapes:")
        print(f"  inputs: {inputs.shape} (batch_size, sequence_length, channels, height, width)")
        print(f"  labels: {labels.shape} (batch_size, sequence_length)")
        break


if __name__ == "__main__":
    main() 