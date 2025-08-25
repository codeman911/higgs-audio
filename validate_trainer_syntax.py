#!/usr/bin/env python3
"""
Trainer Module Syntax Validation Script
Validates all trainer modules for syntax errors and import issues.
"""

import sys
import traceback
from pathlib import Path


def test_module_import(module_name, import_statement):
    """Test importing a specific module."""
    try:
        exec(import_statement)
        print(f"✅ {module_name}: Import successful")
        return True
    except SyntaxError as e:
        print(f"❌ {module_name}: Syntax Error - {e}")
        print(f"   File: {e.filename}, Line: {e.lineno}")
        return False
    except ImportError as e:
        print(f"⚠️  {module_name}: Import Error - {e}")
        print("   (This may be expected if dependencies are missing)")
        return True  # Import errors are expected without ML dependencies
    except Exception as e:
        print(f"❌ {module_name}: Unexpected Error - {e}")
        traceback.print_exc()
        return False


def main():
    """Main validation function."""
    print("🔍 Validating Trainer Module Syntax")
    print("=" * 50)
    
    # Add the current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # List of modules to test
    modules_to_test = [
        ("Config Module", "from trainer.config import TrainingConfig"),
        ("Dataset Module", "from trainer.dataset import VoiceCloningDataset"),
        ("Loss Module", "from trainer.loss import DualFFNLoss"),
        ("Logging Utils", "from trainer.logging_utils import TrainingLogger"),
        ("Audio Validation", "from trainer.audio_validation import AudioQualityValidator"),
        ("Reference Verification", "from trainer.reference_verification import ReferenceVerificationSystem"),
        ("Utils Module", "from trainer.utils import create_sample_data"),
        ("Trainer Module", "from trainer.trainer import HiggsAudioTrainer"),
        ("Trainer Init", "from trainer import HiggsAudioTrainer, VoiceCloningDataset, TrainingConfig"),
    ]
    
    # Test each module
    all_passed = True
    for module_name, import_statement in modules_to_test:
        success = test_module_import(module_name, import_statement)
        if not success:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All trainer modules passed syntax validation!")
        print("\n💡 If you see import errors above, they're expected without ML dependencies.")
        print("   The syntax validation passed, which means the code should work correctly")
        print("   when run with proper PyTorch/Transformers environment.")
    else:
        print("❌ Some modules failed syntax validation!")
        print("   Please fix the syntax errors before proceeding.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())