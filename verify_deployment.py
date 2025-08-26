#!/usr/bin/env python3
"""
Pre-Training Verification Script

Run this before training to ensure all fixes are properly deployed.

Usage:
    python3 verify_deployment.py
"""

import os
import sys
import json
import importlib.util
from pathlib import Path


def check_file_exists(file_path: str, description: str) -> bool:
    """Check if a file exists and report status."""
    if os.path.exists(file_path):
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} - NOT FOUND")
        return False


def check_validation_fix() -> bool:
    """Check if the validation fix is properly deployed."""
    print("\n🔍 Checking Validation Fix Deployment")
    print("-" * 40)
    
    # Check if utils.py has the fixed validation
    utils_path = "utils.py"
    if os.path.exists(utils_path):
        with open(utils_path, 'r') as f:
            content = f.read()
        
        # Check for key indicators of the fix
        has_arb_inference_logic = "EXACT arb_inference.py logic" in content
        has_content_list_handling = "isinstance(content, list)" in content
        has_text_parts_logic = "text_parts = []" in content
        
        if has_arb_inference_logic and has_content_list_handling and has_text_parts_logic:
            print("✅ utils.py: Contains fixed validation logic")
            return True
        else:
            print("❌ utils.py: Missing validation fix")
            print(f"   - arb_inference logic: {'✅' if has_arb_inference_logic else '❌'}")
            print(f"   - content list handling: {'✅' if has_content_list_handling else '❌'}")
            print(f"   - text_parts logic: {'✅' if has_text_parts_logic else '❌'}")
            return False
    else:
        print("❌ utils.py: File not found")
        return False


def check_trainer_dataset_fix() -> bool:
    """Check if trainer/dataset.py has the validation fix."""
    print("\n🔍 Checking Trainer Dataset Fix")
    print("-" * 32)
    
    dataset_path = "trainer/dataset.py"
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r') as f:
            content = f.read()
        
        # Check for key indicators
        has_exact_logic = "EXACT arb_inference.py" in content
        has_extract_components = "_extract_sample_components" in content
        
        if has_exact_logic and has_extract_components:
            print("✅ trainer/dataset.py: Contains fixed validation logic")
            return True
        else:
            print("❌ trainer/dataset.py: Missing validation fix")
            return False
    else:
        print("❌ trainer/dataset.py: File not found")
        return False


def test_validation_import() -> bool:
    """Test if validation functions can be imported and work."""
    print("\n🧪 Testing Validation Function Import")
    print("-" * 37)
    
    try:
        # Test utils import
        if os.path.exists("utils.py"):
            spec = importlib.util.spec_from_file_location("utils", "utils.py")
            utils_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(utils_module)
            
            # Test the validation function
            if hasattr(utils_module, '_validate_sample_structure'):
                print("✅ utils._validate_sample_structure: Function available")
                
                # Test with a sample
                test_sample = {
                    "messages": [
                        {"role": "user", "content": "test"},
                        {"role": "assistant", "content": {"type": "audio", "audio_url": "test.wav"}},
                        {"role": "user", "content": "target"}
                    ]
                }
                
                try:
                    result = utils_module._validate_sample_structure(test_sample, 0)
                    print("✅ Validation function: Can process test sample")
                    return True
                except Exception as e:
                    print(f"❌ Validation function: Error processing test sample: {e}")
                    return False
            else:
                print("❌ utils._validate_sample_structure: Function not found")
                return False
        else:
            print("❌ utils.py: File not found for import test")
            return False
            
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


def check_training_scripts() -> bool:
    """Check if training scripts are available."""
    print("\n📋 Checking Training Scripts")
    print("-" * 28)
    
    scripts = [
        ("launch_training_fixed.py", "Fixed training launcher (RECOMMENDED)"),
        ("launch_training_direct.py", "Direct training launcher"),
        ("test_validation_fix.py", "Validation test script"),
        ("run_training_with_monitoring.sh", "Monitoring script")
    ]
    
    available_count = 0
    for script, description in scripts:
        if check_file_exists(script, description):
            available_count += 1
    
    return available_count > 0


def check_environment() -> bool:
    """Check training environment."""
    print("\n🌐 Checking Training Environment")
    print("-" * 31)
    
    checks_passed = 0
    
    # Check if in higgs-audio directory
    if os.path.exists("arb_inference.py") and os.path.exists("trainer"):
        print("✅ Working directory: In higgs-audio root")
        checks_passed += 1
    else:
        print("❌ Working directory: Not in higgs-audio root")
        print("   Please run: cd /vs/higgs-audio")
    
    # Check trainer directory
    if os.path.exists("trainer/train.py"):
        print("✅ Trainer module: Available")
        checks_passed += 1
    else:
        print("❌ Trainer module: trainer/train.py not found")
    
    # Check boson_multimodal
    if os.path.exists("boson_multimodal"):
        print("✅ boson_multimodal: Available")
        checks_passed += 1
    else:
        print("❌ boson_multimodal: Directory not found")
    
    return checks_passed >= 2


def main():
    """Main verification function."""
    print("🔧 Pre-Training Verification Script")
    print("=" * 40)
    print("Checking if all validation fixes are properly deployed...")
    
    # Run all checks
    checks = [
        ("Environment", check_environment()),
        ("Validation Fix in utils.py", check_validation_fix()),
        ("Trainer Dataset Fix", check_trainer_dataset_fix()),
        ("Validation Function Import", test_validation_import()),
        ("Training Scripts", check_training_scripts())
    ]
    
    # Summary
    print("\n📊 Verification Summary")
    print("=" * 23)
    
    passed_checks = 0
    for check_name, result in checks:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{check_name:.<30} {status}")
        if result:
            passed_checks += 1
    
    print(f"\n📈 Overall Status: {passed_checks}/{len(checks)} checks passed")
    
    if passed_checks == len(checks):
        print("\n🎉 All checks passed! Ready to run training.")
        print("\n🚀 Next steps:")
        print("   1. Test validation: python3 test_validation_fix.py --train_data YOUR_DATA_FILE")
        print("   2. Run training: python3 launch_training_fixed.py --train_data YOUR_DATA_FILE")
        print("   3. Or use monitoring: bash run_training_with_monitoring.sh")
        return True
    elif passed_checks >= 3:
        print("\n⚠️ Most checks passed. You can try running training but may encounter issues.")
        print("\n💡 Recommended actions:")
        for check_name, result in checks:
            if not result:
                print(f"   - Fix: {check_name}")
        return False
    else:
        print("\n❌ Critical checks failed. Please fix issues before training.")
        print("\n🆘 Required actions:")
        for check_name, result in checks:
            if not result:
                print(f"   - MUST FIX: {check_name}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)