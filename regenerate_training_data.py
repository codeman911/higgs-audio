#!/usr/bin/env python3
"""
Regenerate Training Data with Fixed ChatML Structure
Run this to fix the zero-shot voice cloning training data
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.data_processing.zero_shot_processor import ZeroShotDataProcessor

def main():
    """Regenerate training data with corrected ChatML structure"""
    
    parser = argparse.ArgumentParser(description="Regenerate training data with corrected ChatML structure")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to dataset directory containing metadata.json")
    parser.add_argument("--output_dir", type=str, default="lora_training_data_zr/chatml_fixed",
                       help="Output directory for corrected ChatML data")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process (for testing)")
    parser.add_argument("--train_split", type=float, default=0.8,
                       help="Fraction of data for training")
    parser.add_argument("--target_sample_rate", type=int, default=24000,
                       help="Target sample rate for audio processing")
    parser.add_argument("--max_duration", type=float, default=30.0,
                       help="Maximum audio duration in seconds")
    parser.add_argument("--min_duration", type=float, default=0.5,
                       help="Minimum audio duration in seconds")
    
    args = parser.parse_args()
    
    print(" REGENERATING TRAINING DATA WITH CORRECTED CHATML STRUCTURE")
    print("=" * 80)
    
    # Handle both directory and direct metadata.json file paths
    dataset_path = Path(args.dataset_path)
    if dataset_path.is_file() and dataset_path.name == "metadata.json":
        # User passed direct path to metadata.json - use parent directory
        dataset_dir = dataset_path.parent
        print(f" Using directory: {dataset_dir} (from metadata.json path)")
    else:
        # User passed directory path
        dataset_dir = dataset_path
        print(f" Using directory: {dataset_dir}")
    
    if not dataset_dir.exists():
        print(f" Dataset directory not found: {dataset_dir}")
        return
    
    metadata_file = dataset_dir / "metadata.json"
    if not metadata_file.exists():
        print(f" metadata.json not found in: {dataset_dir}")
        print(f"   Expected: {metadata_file}")
        return
    
    # Create processor with fixed ChatML structure
    processor = ZeroShotDataProcessor(
        dataset_path=str(dataset_dir),
        output_dir=args.output_dir,
        target_sample_rate=args.target_sample_rate,
        max_duration=args.max_duration,
        min_duration=args.min_duration
    )
    
    print(f" Dataset directory: {dataset_dir}")
    print(f" Output directory: {args.output_dir}")
    print(f" Max samples: {args.max_samples or 'All'}")
    
    # Process dataset with corrected structure
    print("\n Processing dataset with CORRECTED ChatML structure...")
    try:
        chatml_samples = processor.process_dataset(max_samples=args.max_samples)
        
        if not chatml_samples:
            print(" No samples were processed successfully!")
            return
            
    except Exception as e:
        print(f" Error processing dataset: {str(e)}")
        return
    
    # Create train/val split
    print(f"\n Creating train/validation split ({args.train_split:.1%} train)...")
    train_size = int(args.train_split * len(chatml_samples))
    train_samples = chatml_samples[:train_size]
    val_samples = chatml_samples[train_size:]
    
    # Save corrected samples
    print(f"\n Saving corrected ChatML samples...")
    try:
        processor.save_chatml_samples(train_samples, "train_chatml_samples.json")
        processor.save_chatml_samples(val_samples, "val_chatml_samples.json")
        
        print("\n DATA REGENERATION COMPLETE!")
        print(f" Training samples: {len(train_samples)}")
        print(f" Validation samples: {len(val_samples)}")
        print(f" Output directory: {args.output_dir}")
        
        print("\n NEXT STEPS:")
        print("1. Update --dataset_path to use the new corrected data:")
        print(f"   --dataset_path {args.output_dir}")
        print("2. Run training with debug samples to verify:")
        print(f"   --debug_samples 100 --debug_val_samples 10")
        print("3. Check logs for corrected ChatML structure!")
        
    except Exception as e:
        print(f" Error saving samples: {str(e)}")

if __name__ == "__main__":
    main()
