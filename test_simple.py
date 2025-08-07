#!/usr/bin/env python3
"""
Simple test script for zero-shot voice cloning data processing
Works from any directory with robust import handling
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add current directory and parent directories to path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Try to import the processor with multiple fallback strategies
def import_processor():
    """Import the zero-shot processor with fallback strategies"""
    
    # Strategy 1: Direct relative import
    try:
        from scripts.data_processing.zero_shot_processor import ZeroShotDataProcessor
        return ZeroShotDataProcessor
    except ImportError:
        pass
    
    # Strategy 2: Add scripts directory to path
    try:
        scripts_dir = current_dir / "scripts"
        if scripts_dir.exists():
            sys.path.insert(0, str(scripts_dir))
        from data_processing.zero_shot_processor import ZeroShotDataProcessor
        return ZeroShotDataProcessor
    except ImportError:
        pass
    
    # Strategy 3: Direct file execution
    try:
        import importlib.util
        processor_file = current_dir / "scripts" / "data_processing" / "zero_shot_processor.py"
        if processor_file.exists():
            spec = importlib.util.spec_from_file_location("zero_shot_processor", processor_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.ZeroShotDataProcessor
    except Exception:
        pass
    
    raise ImportError("Could not import ZeroShotDataProcessor. Please check your installation.")


def test_dataset_processing(dataset_path, max_samples=10):
    """Test dataset processing with minimal dependencies"""
    
    print("🧪 Simple Zero-Shot Voice Cloning Test")
    print("=" * 40)
    print(f"📁 Dataset: {dataset_path}")
    print(f"🔢 Samples: {max_samples}")
    print()
    
    # Check dataset exists
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return False
    
    # Check metadata file
    metadata_file = dataset_path / "metadata.json"
    if not metadata_file.exists():
        print(f"❌ metadata.json not found in {dataset_path}")
        return False
    
    print("✅ Dataset path and metadata found")
    
    # Load and validate metadata
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        if 'samples' not in metadata:
            print("❌ No 'samples' field in metadata.json")
            return False
        
        samples = metadata['samples']
        print(f"✅ Found {len(samples)} samples in metadata")
        
        if len(samples) == 0:
            print("❌ No samples found in dataset")
            return False
        
        # Test first few samples
        test_samples = samples[:min(max_samples, len(samples))]
        valid_samples = 0
        
        print(f"\n🔍 Validating {len(test_samples)} samples...")
        
        for i, sample in enumerate(test_samples):
            sample_id = sample.get('id', f'sample_{i}')
            
            # Check required fields
            required_fields = ['audio_file', 'transcript_file', 'ref_audio_file', 'ref_transcript']
            missing_fields = [field for field in required_fields if field not in sample]
            
            if missing_fields:
                print(f"⚠️  Sample {sample_id}: Missing fields {missing_fields}")
                continue
            
            # Check files exist
            audio_file = dataset_path / sample['audio_file']
            ref_audio_file = dataset_path / sample['ref_audio_file']
            transcript_file = dataset_path / sample['transcript_file']
            
            files_exist = all([
                audio_file.exists(),
                ref_audio_file.exists(), 
                transcript_file.exists()
            ])
            
            if files_exist:
                valid_samples += 1
                if i < 3:  # Show details for first 3 samples
                    print(f"✅ Sample {sample_id}: All files found")
            else:
                missing_files = []
                if not audio_file.exists():
                    missing_files.append(sample['audio_file'])
                if not ref_audio_file.exists():
                    missing_files.append(sample['ref_audio_file'])
                if not transcript_file.exists():
                    missing_files.append(sample['transcript_file'])
                print(f"❌ Sample {sample_id}: Missing files {missing_files}")
        
        print(f"\n📊 Validation Results:")
        print(f"   • Total samples checked: {len(test_samples)}")
        print(f"   • Valid samples: {valid_samples}")
        print(f"   • Success rate: {valid_samples/len(test_samples)*100:.1f}%")
        
        if valid_samples > 0:
            print("\n🎉 Dataset validation PASSED!")
            print("Your dataset is ready for processing.")
            
            # Show sample structure
            sample = test_samples[0]
            print(f"\n📋 Sample structure:")
            print(f"   • ID: {sample.get('id', 'N/A')}")
            print(f"   • Target audio: {sample.get('audio_file', 'N/A')}")
            print(f"   • Reference audio: {sample.get('ref_audio_file', 'N/A')}")
            print(f"   • Target text file: {sample.get('transcript_file', 'N/A')}")
            print(f"   • Reference text: {sample.get('ref_transcript', 'N/A')[:50]}...")
            print(f"   • Duration: {sample.get('duration', 'N/A')} seconds")
            
            return True
        else:
            print("\n⚠️  Dataset validation had issues.")
            print("Please check the file paths in your metadata.json")
            return False
            
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Simple dataset validation test")
    parser.add_argument("dataset_path", help="Path to dataset directory")
    parser.add_argument("--max_samples", type=int, default=10, help="Max samples to check")
    
    args = parser.parse_args()
    
    success = test_dataset_processing(args.dataset_path, args.max_samples)
    
    if success:
        print(f"\n🚀 Next steps:")
        print(f"1. Run full processing:")
        print(f"   python3 scripts/data_processing/zero_shot_processor.py \\")
        print(f"     --dataset_path {args.dataset_path} \\")
        print(f"     --output_dir ./processed_data")
        print(f"2. Or use the test script:")
        print(f"   python3 scripts/test_data_processing.py \\")
        print(f"     --dataset_path {args.dataset_path} \\")
        print(f"     --max_samples 10")
        return 0
    else:
        print(f"\n❌ Please fix the dataset issues before proceeding.")
        return 1


if __name__ == "__main__":
    exit(main())
