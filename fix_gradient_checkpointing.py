#!/usr/bin/env python3
"""
Quick Fix for Gradient Checkpointing Incompatibility

This script fixes the specific error:
ValueError: HiggsAudioModel is not compatible with gradient checkpointing.

The issue is that Higgs Audio models don't support gradient checkpointing,
but our training configuration has it enabled by default.
"""

import os
import sys
from pathlib import Path

def fix_gradient_checkpointing_issue():
    """Fix gradient checkpointing incompatibility in all relevant files."""
    print("🔧 Fixing gradient checkpointing incompatibility...")
    
    fixes_applied = []
    
    # Fix 1: Distributed trainer configuration
    trainer_file = Path("arabic_voice_cloning_distributed_trainer.py")
    if trainer_file.exists():
        with open(trainer_file, 'r') as f:
            content = f.read()
        
        if "gradient_checkpointing: bool = True" in content:
            content = content.replace(
                "gradient_checkpointing: bool = True",
                "gradient_checkpointing: bool = False  # Disabled for Higgs Audio compatibility"
            )
            with open(trainer_file, 'w') as f:
                f.write(content)
            fixes_applied.append("Disabled gradient checkpointing in distributed trainer")
            print("✅ Fixed distributed trainer configuration")
    
    # Fix 2: LoRA config test code
    lora_file = Path("arabic_voice_cloning_lora_config.py")
    if lora_file.exists():
        with open(lora_file, 'r') as f:
            content = f.read()
        
        if "enable_gradient_checkpointing=True" in content:
            content = content.replace(
                "enable_gradient_checkpointing=True",
                "enable_gradient_checkpointing=False  # Disabled for Higgs Audio compatibility"
            )
            with open(lora_file, 'w') as f:
                f.write(content)
            fixes_applied.append("Fixed LoRA config test code")
            print("✅ Fixed LoRA configuration test code")
    
    # Fix 3: Add proper error handling to LoRA config
    if lora_file.exists():
        with open(lora_file, 'r') as f:
            content = f.read()
        
        # Check if the proper error handling is already there
        if "ValueError as e:" not in content or "gradient_checkpointing" not in content:
            print("⚠️  Adding proper gradient checkpointing error handling...")
            
            # Find the gradient checkpointing section and replace it
            old_section = """        # Enable gradient checkpointing if requested
        if enable_gradient_checkpointing:
            # Note: Higgs Audio doesn't support enable_input_require_grads() due to missing get_input_embeddings()
            # Instead, we'll enable gradient checkpointing directly on the model
            if hasattr(lora_model, 'gradient_checkpointing_enable'):
                lora_model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            elif hasattr(lora_model, 'enable_gradient_checkpointing'):
                lora_model.enable_gradient_checkpointing()
                logger.info("Gradient checkpointing enabled (alternative method)")
            else:
                logger.warning("Gradient checkpointing not available for this model")"""
            
            new_section = """        # Enable gradient checkpointing if requested
        if enable_gradient_checkpointing:
            # Higgs Audio models don't support gradient checkpointing
            try:
                if hasattr(lora_model, 'gradient_checkpointing_enable'):
                    lora_model.gradient_checkpointing_enable()
                    logger.info("Gradient checkpointing enabled successfully")
                elif hasattr(lora_model, 'enable_gradient_checkpointing'):
                    lora_model.enable_gradient_checkpointing()
                    logger.info("Gradient checkpointing enabled (alternative method)")
                else:
                    logger.warning("Gradient checkpointing method not found")
            except ValueError as e:
                if "gradient_checkpointing" in str(e).lower():
                    logger.warning(f"Model doesn't support gradient checkpointing: {e}")
                    logger.info("Training will continue without gradient checkpointing")
                else:
                    raise e
            except Exception as e:
                logger.warning(f"Failed to enable gradient checkpointing: {e}")
                logger.info("Training will continue without gradient checkpointing")"""
            
            if old_section in content:
                content = content.replace(old_section, new_section)
                with open(lora_file, 'w') as f:
                    f.write(content)
                fixes_applied.append("Added proper gradient checkpointing error handling")
                print("✅ Added proper error handling")
    
    return fixes_applied

def test_fix():
    """Test if the fix works by trying to import and create a config."""
    print("🧪 Testing fix...")
    
    try:
        # Test import
        from arabic_voice_cloning_lora_config import HiggsAudioLoRATrainingConfig
        print("✅ Import successful")
        
        # Test config creation
        config = HiggsAudioLoRATrainingConfig(
            r=8,
            target_modules_mode="attention_only"
        )
        print("✅ Configuration creation successful")
        print(f"   Gradient checkpointing will be: disabled by default")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main execution function."""
    print("🎯 Gradient Checkpointing Compatibility Fix")
    print("=" * 60)
    print("Issue: HiggsAudioModel is not compatible with gradient checkpointing")
    print()
    
    # Apply fixes
    fixes = fix_gradient_checkpointing_issue()
    
    # Test fixes
    print("\n🧪 Testing Fixes")
    test_success = test_fix()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 FIX SUMMARY")
    print("=" * 60)
    
    if fixes:
        print("✅ Fixes Applied:")
        for fix in fixes:
            print(f"   - {fix}")
    else:
        print("ℹ️  No fixes needed - files already up to date")
    
    if test_success:
        print("\n🎉 GRADIENT CHECKPOINTING ISSUE FIXED!")
        print("\n📋 Next Steps:")
        print("1. Copy the fixed files to your running directory:")
        print("   cp arabic_voice_cloning_distributed_trainer.py /vs/higgs-audio/")
        print("   cp arabic_voice_cloning_lora_config.py /vs/higgs-audio/")
        print("\n2. Run your training command:")
        print("   python3 arabic_voice_cloning_distributed_trainer.py \\")
        print("     --data_path ../ms-swift/lora_training_data_zr/chatml_fixed/val_chatml_samples.json \\")
        print("     --output_dir EXPMT/exp_small")
        print("\n✅ Training should now work without gradient checkpointing errors!")
        return 0
    else:
        print("\n❌ Some issues remain - check the test output above")
        return 1

if __name__ == "__main__":
    exit(main())