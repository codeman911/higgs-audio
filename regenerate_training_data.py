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
    parser.add_argument("--dataset_paths", nargs='+', required=True,
                       help="Paths to dataset directories containing metadata.json (can specify multiple)")
    parser.add_argument("--output_dir", type=str, default="lora_training_data_zr/chatml_fixed",
                       help="Output directory for corrected ChatML data")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process per dataset (for testing)")
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
    
    # Process multiple dataset directories
    all_chatml_samples = []
    processing_stats = {}
    
    for dataset_path_str in args.dataset_paths:
        print(f"\n Processing dataset: {dataset_path_str}")
        print("-" * 60)
        
        # Handle both directory and direct metadata.json file paths
        dataset_path = Path(dataset_path_str)
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
            continue
        
        metadata_file = dataset_dir / "metadata.json"
        if not metadata_file.exists():
            print(f" metadata.json not found in: {dataset_dir}")
            print(f"   Expected: {metadata_file}")
            continue
        
        # Create processor for this dataset
        processor = ZeroShotDataProcessor(
            dataset_path=str(dataset_dir),
            output_dir=args.output_dir,
            target_sample_rate=args.target_sample_rate,
            max_duration=args.max_duration,
            min_duration=args.min_duration
        )
        
        # Process this dataset
        try:
            dataset_samples = processor.process_dataset(max_samples=args.max_samples)
            
            if dataset_samples:
                all_chatml_samples.extend(dataset_samples)
                processing_stats[str(dataset_dir)] = len(dataset_samples)
                print(f" Processed {len(dataset_samples)} samples from {dataset_dir}")
            else:
                print(f"  No samples processed from {dataset_dir}")
                processing_stats[str(dataset_dir)] = 0
                
        except Exception as e:
            print(f" Error processing {dataset_dir}: {str(e)}")
            processing_stats[str(dataset_dir)] = 0
            continue
    
    # Check if any samples were processed
    if not all_chatml_samples:
        print("\n No samples were processed successfully from any dataset!")
        return
    
    print(f"\n COMBINED RESULTS:")
    print(f" Total samples across all datasets: {len(all_chatml_samples)}")
    for dataset_dir, count in processing_stats.items():
        print(f"   {Path(dataset_dir).name}: {count} samples")
    
    # Create train/val split across all combined data
    print(f"\n Creating train/validation split ({args.train_split:.1%} train)...")
    train_size = int(args.train_split * len(all_chatml_samples))
    train_samples = all_chatml_samples[:train_size]
    val_samples = all_chatml_samples[train_size:]
    
    # Save combined corrected samples
    print(f"\n Saving combined corrected ChatML samples...")
    try:
        # Use the last processor for saving (all have same output settings)
        processor.save_chatml_samples(train_samples, "train_chatml_samples.json")
        processor.save_chatml_samples(val_samples, "val_chatml_samples.json")
        
        print("\n DATA REGENERATION COMPLETE!")
        print(f" Training samples: {len(train_samples)}")
        print(f" Validation samples: {len(val_samples)}")
        print(f" Output directory: {args.output_dir}")
        print(f" Datasets processed: {len([v for v in processing_stats.values() if v > 0])}")
        
        print("\n NEXT STEPS:")
        print("1. Update --dataset_path to use the new corrected data:")
        print(f"   --dataset_path {args.output_dir}")
        print("2. Run training with debug samples to verify:")
        print(f"   --debug_samples 100 --debug_val_samples 10")
        print("3. Check logs for corrected ChatML structure!")
        
    except Exception as e:
        print(f" Error saving combined samples: {str(e)}")

if __name__ == "__main__":
    main()
