#!/usr/bin/env python3
"""
Regenerate Training Data with Fixed ChatML Structure
Run this to fix the zero-shot voice cloning training data
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.data_processing.zero_shot_processor import ZeroShotDataProcessor

def main():
    """Regenerate training data with corrected ChatML structure"""
    
    print("🔧 REGENERATING TRAINING DATA WITH CORRECTED CHATML STRUCTURE")
    print("=" * 80)
    
    # Configuration
    dataset_path = "path_to_your_original_dataset"  # UPDATE THIS PATH
    output_dir = "lora_training_data_zr/chatml_fixed"
    
    # Create processor with fixed ChatML structure
    processor = ZeroShotDataProcessor(
        dataset_path=dataset_path,
        output_dir=output_dir,
        target_sample_rate=24000,
        max_duration=30.0,
        min_duration=0.5
    )
    
    print(f"📁 Dataset path: {dataset_path}")
    print(f"📁 Output directory: {output_dir}")
    
    # Process dataset with corrected structure
    print("\n🔄 Processing dataset with CORRECTED ChatML structure...")
    chatml_samples = processor.process_dataset()
    
    # Save corrected samples
    print(f"\n💾 Saving corrected ChatML samples...")
    processor.save_chatml_samples(chatml_samples, "train_chatml_samples.json")
    
    # Create train/val split (80/20)
    print(f"\n📊 Creating train/validation split...")
    train_size = int(0.8 * len(chatml_samples))
    train_samples = chatml_samples[:train_size]
    val_samples = chatml_samples[train_size:]
    
    processor.save_chatml_samples(train_samples, "train_chatml_samples.json")
    processor.save_chatml_samples(val_samples, "val_chatml_samples.json")
    
    print("\n✅ DATA REGENERATION COMPLETE!")
    print(f"📊 Training samples: {len(train_samples)}")
    print(f"📊 Validation samples: {len(val_samples)}")
    print(f"📁 Output directory: {output_dir}")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Update --dataset_path to use the new corrected data:")
    print(f"   --dataset_path {output_dir}")
    print("2. Run training with debug samples to verify:")
    print(f"   --debug_samples 100 --debug_val_samples 10")
    print("3. Check logs for corrected ChatML structure!")

if __name__ == "__main__":
    main()
