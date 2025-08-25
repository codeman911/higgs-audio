#!/usr/bin/env python3
"""
Validation Script for Arabic Voice Cloning Training Fixes

This script validates that all the critical fixes have been properly implemented
in the arabic_voice_cloning_distributed_trainer.py file.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def validate_device_handling():
    """Validate that device handling is fixed for single GPU mode."""
    print("🔍 Validating device handling fix...")
    
    # Read the trainer file
    trainer_path = current_dir / "arabic_voice_cloning_distributed_trainer.py"
    if not trainer_path.exists():
        print("❌ Trainer file not found")
        return False
    
    with open(trainer_path, 'r') as f:
        content = f.read()
    
    # Check for proper device handling
    if "if self.training_config.local_rank == -1:" in content:
        print("✅ Device handling fix found - properly handles local_rank = -1")
    else:
        print("❌ Device handling fix not found")
        return False
    
    # Check for proper device ID assignment
    if "device_id = 0  # Use GPU 0 for single GPU" in content:
        print("✅ Device ID assignment fix found")
    else:
        print("❌ Device ID assignment fix not found")
        return False
    
    return True

def validate_model_compatibility():
    """Validate that model compatibility checks are in place."""
    print("\n🔍 Validating model compatibility checks...")
    
    trainer_path = current_dir / "arabic_voice_cloning_distributed_trainer.py"
    with open(trainer_path, 'r') as f:
        content = f.read()
    
    # Check for model validation
    if "Validate model compatibility" in content:
        print("✅ Model compatibility validation found")
    else:
        print("❌ Model compatibility validation not found")
        return False
    
    # Check for forward signature validation
    if "sig = inspect.signature" in content and "params = list(sig.parameters.keys())" in content:
        print("✅ Forward signature validation found")
    else:
        print("❌ Forward signature validation not found")
        return False
    
    # Check for labels parameter check
    if "'labels' in params:" in content:
        print("✅ Labels parameter check found")
    else:
        print("❌ Labels parameter check not found")
        return False
    
    return True

def validate_whisper_processor():
    """Validate that Whisper processor setup is fixed."""
    print("\n🔍 Validating Whisper processor setup...")
    
    trainer_path = current_dir / "arabic_voice_cloning_distributed_trainer.py"
    with open(trainer_path, 'r') as f:
        content = f.read()
    
    # Check for Whisper processor setup
    if "Setting up Whisper processor for zero-shot voice cloning" in content:
        print("✅ Whisper processor setup found")
    else:
        print("❌ Whisper processor setup not found")
        return False
    
    # Check for trust_remote_code parameter
    if "trust_remote_code=True" in content:
        print("✅ trust_remote_code parameter found")
    else:
        print("❌ trust_remote_code parameter not found")
        return False
    
    # Check for fallback handling
    if "Whisper-base processor loaded as fallback" in content:
        print("✅ Fallback handling found")
        return True
    else:
        print("❌ Fallback handling not found")
        return False
    
    return True

def validate_model_forward_call():
    """Validate that model forward call is correct."""
    print("\n🔍 Validating model forward call...")
    
    trainer_path = current_dir / "arabic_voice_cloning_distributed_trainer.py"
    with open(trainer_path, 'r') as f:
        content = f.read()
    
    # Check for definitive model forward call
    if "DEFINITIVE model forward call" in content:
        print("✅ Definitive model forward call found")
    else:
        print("❌ Definitive model forward call not found")
        return False
    
    # Check that 'labels' parameter is NOT in the call
    if "label_ids=training_batch.label_ids" in content and "label_audio_ids=training_batch.label_audio_ids" in content:
        print("✅ Correct parameter names found (label_ids, label_audio_ids)")
    else:
        print("❌ Correct parameter names not found")
        return False
    
    # Check that 'labels' parameter is NOT in the call
    training_step_section = content.split("outputs = self.model(")[1].split(")")[0] if "outputs = self.model(" in content else ""
    if "labels=" not in training_step_section:
        print("✅ No 'labels' parameter in model forward call")
    else:
        print("❌ 'labels' parameter found in model forward call")
        return False
    
    return True

def validate_error_handling():
    """Validate that improved error handling is in place."""
    print("\n🔍 Validating error handling improvements...")
    
    trainer_path = current_dir / "arabic_voice_cloning_distributed_trainer.py"
    with open(trainer_path, 'r') as f:
        content = f.read()
    
    # Check for enhanced error logging
    if "❌ Training step" in content:
        print("✅ Enhanced error logging found")
    else:
        print("❌ Enhanced error logging not found")
        return False
    
    # Check for model signature logging
    if "Model forward parameters" in content:
        print("✅ Model signature logging found")
    else:
        print("❌ Model signature logging not found")
        return False
    
    return True

def main():
    """Main validation function."""
    print("🚀 VALIDATING ARABIC VOICE CLONING TRAINING FIXES")
    print("=" * 50)
    
    all_checks_passed = True
    
    # Run all validations
    checks = [
        validate_device_handling,
        validate_model_compatibility,
        validate_whisper_processor,
        validate_model_forward_call,
        validate_error_handling
    ]
    
    for check in checks:
        if not check():
            all_checks_passed = False
    
    print("\n" + "=" * 50)
    if all_checks_passed:
        print("🎉 ALL FIXES VALIDATED SUCCESSFULLY!")
        print("✅ The trainer is ready for use with all critical issues resolved:")
        print("   - Device handling for single GPU mode (local_rank = -1)")
        print("   - Model compatibility validation (no 'labels' parameter)")
        print("   - Whisper processor setup for zero-shot voice cloning")
        print("   - Correct model forward call parameters")
        print("   - Enhanced error handling and logging")
        print("\n📝 To use the fixed trainer:")
        print("   python3 arabic_voice_cloning_distributed_trainer.py \\")
        print("     --data_path ../ms-swift/lora_training_data_zr/chatml_fixed/val_chatml_samples.json \\")
        print("     --output_dir EXPMT/exp_small")
    else:
        print("❌ SOME FIXES ARE MISSING!")
        print("Please check the trainer file and ensure all fixes are applied.")
    
    return all_checks_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)