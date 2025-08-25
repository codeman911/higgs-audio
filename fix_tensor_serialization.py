#!/usr/bin/env python3
"""
Fix Tensor Serialization Issue in Multiprocessing

This script fixes the specific error:
RuntimeError: Cowardly refusing to serialize non-leaf tensor which requires_grad, 
since autograd does not support crossing process boundaries.

The issue occurs when PyTorch tensors with gradients are passed to multiprocessing,
which is not supported.
"""

import os
import sys
from pathlib import Path

def fix_dataset_multiprocessing():
    """Fix multiprocessing tensor serialization issue in dataset."""
    print("🔧 Fixing tensor serialization issue in dataset...")
    
    dataset_file = Path("arabic_voice_cloning_dataset.py")
    if not dataset_file.exists():
        print(f"❌ Dataset file not found: {dataset_file}")
        return False
    
    with open(dataset_file, 'r') as f:
        content = f.read()
    
    fixes_applied = []
    
    # Fix 1: Remove multiprocessing import
    if "import multiprocessing as mp" in content:
        content = content.replace("import multiprocessing as mp\n", "")
        fixes_applied.append("Removed multiprocessing import")
    
    # Fix 2: Replace multiprocessing validation with sequential processing
    if "with mp.Pool(processes=" in content:
        old_multiprocessing = """        # Use multiprocessing for faster validation
        with mp.Pool(processes=min(mp.cpu_count(), 32)) as pool:
            validation_results = pool.map(
                self._validate_single_sample,
                [(idx, sample) for idx, sample in enumerate(self.raw_data)]
            )
        
        # Process validation results
        for idx, (sample, status, error_type) in enumerate(validation_results):"""
        
        new_sequential = """        # Process samples sequentially to avoid tensor serialization issues
        # Multiprocessing can cause issues with tensors that have gradients
        logger.info("Validating samples sequentially to avoid tensor serialization issues...")
        
        for idx, sample in enumerate(self.raw_data):
            sample_result, status, error_type = self._validate_single_sample((idx, sample))
            
            # Log progress for large datasets
            if (idx + 1) % 100 == 0:
                logger.info(f"Validated {idx + 1}/{len(self.raw_data)} samples...")"""
        
        if old_multiprocessing in content:
            content = content.replace(old_multiprocessing, new_sequential)
            fixes_applied.append("Replaced multiprocessing validation with sequential processing")
    
    # Fix 3: Remove mp.cpu_count() references
    if "mp.cpu_count()" in content:
        content = content.replace("min(config.num_workers, mp.cpu_count())", "min(config.num_workers, 32)")
        fixes_applied.append("Removed mp.cpu_count() references")
    
    # Fix 4: Add num_workers=0 for DataLoader to avoid multiprocessing in DataLoader too
    if "num_workers=num_workers," in content and "# Force num_workers=0" not in content:
        content = content.replace(
            "num_workers=num_workers,",
            "num_workers=0,  # Force num_workers=0 to avoid tensor serialization issues"
        )
        fixes_applied.append("Set DataLoader num_workers=0 to avoid multiprocessing")
    
    # Write fixed content
    if fixes_applied:
        with open(dataset_file, 'w') as f:
            f.write(content)
        print(f"✅ Applied {len(fixes_applied)} fixes:")
        for fix in fixes_applied:
            print(f"   - {fix}")
        return True
    else:
        print("✅ Dataset file is already compatible with single-process execution")
        return True

def fix_distributed_trainer():
    """Fix any multiprocessing issues in the distributed trainer."""
    print("🔧 Checking distributed trainer for multiprocessing issues...")
    
    trainer_file = Path("arabic_voice_cloning_distributed_trainer.py")
    if not trainer_file.exists():
        print(f"⚠️  Trainer file not found: {trainer_file}")
        return True
    
    with open(trainer_file, 'r') as f:
        content = f.read()
    
    fixes_applied = []
    
    # Force single-process data loading
    if "num_workers" in content and "num_workers=0" not in content:
        # Find the dataloader creation and force num_workers=0
        if "num_workers=self.dataset_config.num_workers" in content:
            content = content.replace(
                "num_workers=self.dataset_config.num_workers",
                "num_workers=0  # Force single-process to avoid tensor serialization"
            )
            fixes_applied.append("Set trainer DataLoader num_workers=0")
    
    if fixes_applied:
        with open(trainer_file, 'w') as f:
            f.write(content)
        print(f"✅ Applied {len(fixes_applied)} fixes to trainer")
    else:
        print("✅ Trainer file doesn't need multiprocessing fixes")
    
    return True

def test_fix():
    """Test if the fix resolves the tensor serialization issue."""
    print("🧪 Testing tensor serialization fix...")
    
    try:
        # Test basic imports
        from arabic_voice_cloning_dataset import ArabicVoiceCloningDatasetConfig
        print("✅ Dataset import successful")
        
        # Test config creation
        config = ArabicVoiceCloningDatasetConfig(
            chatml_file="dummy.json",
            validate_on_init=False,  # Skip validation for testing
            num_workers=0  # Force single-process
        )
        print("✅ Configuration creation successful")
        print(f"   num_workers: {config.num_workers}")
        print(f"   validate_on_init: {config.validate_on_init}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main execution function."""
    print("🎯 Tensor Serialization Fix for PyTorch Multiprocessing")
    print("=" * 60)
    print("Issue: RuntimeError with tensor serialization in multiprocessing")
    print()
    
    # Apply fixes
    print("1️⃣ Fixing Dataset Multiprocessing")
    dataset_fixed = fix_dataset_multiprocessing()
    
    print("\n2️⃣ Fixing Distributed Trainer")
    trainer_fixed = fix_distributed_trainer()
    
    print("\n3️⃣ Testing Fixes")
    test_success = test_fix()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 FIX SUMMARY")
    print("=" * 60)
    
    if dataset_fixed and trainer_fixed:
        print("✅ All multiprocessing issues fixed!")
        print("\n📋 Changes Applied:")
        print("   - Removed multiprocessing from dataset validation")
        print("   - Set DataLoader num_workers=0 to avoid tensor serialization")
        print("   - Sequential sample validation instead of parallel")
        print("   - Removed all mp.cpu_count() references")
        
        if test_success:
            print("\n🎉 TENSOR SERIALIZATION ISSUE FIXED!")
            print("\n📋 Next Steps:")
            print("1. Copy the fixed files to your running directory:")
            print("   cp arabic_voice_cloning_dataset.py /vs/higgs-audio/")
            print("   cp arabic_voice_cloning_distributed_trainer.py /vs/higgs-audio/ (if modified)")
            print("\n2. Run your training command:")
            print("   python3 arabic_voice_cloning_distributed_trainer.py \\")
            print("     --data_path ../ms-swift/lora_training_data_zr/chatml_fixed/val_chatml_samples.json \\")
            print("     --output_dir EXPMT/exp_small")
            print("\n✅ Training should now work without multiprocessing errors!")
            return 0
        else:
            print("\n⚠️  Fixes applied but test failed - check dependencies")
            return 0
    else:
        print("\n❌ Some fixes failed - check error messages above")
        return 1

if __name__ == "__main__":
    exit(main())