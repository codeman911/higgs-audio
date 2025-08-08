#!/usr/bin/env python3
"""
Test script for zero-shot voice cloning data processing
Run this to validate your dataset processing before full training
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Robust import handling for both CLI and module usage
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from scripts.data_processing.zero_shot_processor import ZeroShotDataProcessor
except ImportError:
    # Fallback for different project structures
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)
    from scripts.data_processing.zero_shot_processor import ZeroShotDataProcessor


def test_data_processing(
    dataset_path: str,
    output_dir: str = "/workspace/data/test_processed",
    max_samples: int = 10,
    target_sample_rate: int = 24000,
    max_duration: float = 30.0,
    min_duration: float = 0.5
):
    """Test data processing with configurable parameters"""
    
    print("🧪 Testing Zero-Shot Voice Cloning Data Processing")
    print("=" * 50)
    
    print(f"📁 Dataset path: {dataset_path}")
    print(f"📁 Output path: {output_dir}")
    print(f"🔢 Test samples: {max_samples}")
    print(f"🎵 Sample rate: {target_sample_rate} Hz")
    print(f"⏱️  Duration range: {min_duration}-{max_duration} seconds")
    
    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        print("Please update the dataset_path variable in this script")
        return False
    
    # Check metadata file
    metadata_file = Path(dataset_path) / "metadata.json"
    if not metadata_file.exists():
        print(f"❌ metadata.json not found in {dataset_path}")
        return False
    
    print("✅ Dataset path and metadata.json found")
    
    try:
        # Create processor
        processor = ZeroShotDataProcessor(
            dataset_path=dataset_path,
            output_dir=output_dir,
            target_sample_rate=target_sample_rate,
            max_duration=max_duration,
            min_duration=min_duration
        )
        
        print("\n🔄 Processing samples...")
        
        # Process small sample
        chatml_samples = processor.process_dataset(max_samples=max_samples)
        
        if not chatml_samples:
            print("❌ No samples were successfully processed")
            return False
        
        # Save results
        processor.save_chatml_samples(chatml_samples, "test_chatml_samples.json")
        
        print(f"\n✅ Successfully processed {len(chatml_samples)} samples")
        
        # Show sample structure
        print("\n📋 Sample ChatML structure:")
        if chatml_samples:
            sample = chatml_samples[0]
            print(json.dumps(sample, indent=2, ensure_ascii=False)[:500] + "...")
        
        # Validation checks
        print("\n🔍 Validation checks:")
        
        # Check message structure
        valid_samples = 0
        for sample in chatml_samples:
            if (len(sample['messages']) == 3 and 
                sample['messages'][0]['role'] == 'system' and
                sample['messages'][1]['role'] == 'user' and
                sample['messages'][2]['role'] == 'assistant'):
                valid_samples += 1
        
        print(f"✅ Valid ChatML structure: {valid_samples}/{len(chatml_samples)} samples")
        
        # Check audio files exist
        audio_files_exist = 0
        for sample in chatml_samples:
            user_content = sample['messages'][1]['content']
            assistant_content = sample['messages'][2]['content']
            
            # Find audio URLs
            ref_audio = None
            target_audio = None
            
            for content in user_content:
                if content['type'] == 'audio':
                    ref_audio = content['audio_url']
            
            for content in assistant_content:
                if content['type'] == 'audio':
                    target_audio = content['audio_url']
            
            if ref_audio and target_audio and Path(ref_audio).exists() and Path(target_audio).exists():
                audio_files_exist += 1
        
        print(f"✅ Audio files accessible: {audio_files_exist}/{len(chatml_samples)} samples")
        
        # Summary
        print(f"\n📊 Processing Summary:")
        print(f"   • Total samples processed: {len(chatml_samples)}")
        print(f"   • Valid ChatML structure: {valid_samples}")
        print(f"   • Audio files accessible: {audio_files_exist}")
        print(f"   • Output saved to: {output_dir}")
        
        if valid_samples == len(chatml_samples) and audio_files_exist > 0:
            print("\n🎉 Data processing test PASSED!")
            print("You can now proceed with full dataset processing.")
            return True
        else:
            print("\n⚠️  Data processing test had issues.")
            print("Please check the warnings above before proceeding.")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_usage():
    """Show usage instructions"""
    print("\n📖 Usage Instructions:")
    print("1. Update the dataset_path variable in this script to point to your data")
    print("2. Ensure your dataset has the structure:")
    print("   data/raw_dataset/")
    print("   ├── metadata.json")
    print("   ├── target_audio_*.wav")
    print("   ├── ref_audio_*.wav")
    print("   └── target_text_*.txt")
    print("3. Run: python scripts/test_data_processing.py")


def main():
    """Main function with command-line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Test zero-shot voice cloning data processing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        required=True,
        help="Path to your dataset directory containing metadata.json"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/workspace/data/test_processed",
        help="Output directory for processed test data"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=10,
        help="Maximum number of samples to process for testing"
    )
    parser.add_argument(
        "--target_sample_rate", 
        type=int, 
        default=24000,
        help="Target sample rate for audio processing"
    )
    parser.add_argument(
        "--max_duration", 
        type=float, 
        default=30.0,
        help="Maximum audio duration in seconds"
    )
    parser.add_argument(
        "--min_duration", 
        type=float, 
        default=0.5,
        help="Minimum audio duration in seconds"
    )
    
    args = parser.parse_args()
    
    # Run the test
    success = test_data_processing(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        target_sample_rate=args.target_sample_rate,
        max_duration=args.max_duration,
        min_duration=args.min_duration
    )
    
    if not success:
        show_usage()
        return 1
    else:
        print("\n🚀 Next steps:")
        print("1. Run full processing:")
        print(f"   python scripts/data_processing/zero_shot_processor.py \\")
        print(f"     --dataset_path {args.dataset_path} \\")
        print(f"     --output_dir /workspace/data/processed_chatml")
        print("2. Start training: ./scripts/launch_training.sh")
        return 0


if __name__ == "__main__":
    exit(main())
