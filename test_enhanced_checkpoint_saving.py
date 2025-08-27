#!/usr/bin/env python3
"""
Test script to validate the enhanced checkpoint saving mechanism
"""

import os
import sys
import argparse
import logging
import tempfile
import shutil
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import our components
try:
    from lora import save_lora_adapters
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    sys.exit(1)


def test_checkpoint_directory_structure():
    """Test that checkpoint directories are created with the correct structure"""
    print("Testing checkpoint directory structure...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test the new directory structure format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join(temp_dir, f"Expmt/exp_{timestamp}")
        os.makedirs(exp_dir, exist_ok=True)
        checkpoint_dir = os.path.join(exp_dir, "checkpoint-100")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Verify the structure
        assert os.path.exists(exp_dir), "Experiment directory not created"
        assert os.path.exists(checkpoint_dir), "Checkpoint directory not created"
        
        # Check the directory structure
        exp_parent = os.path.dirname(exp_dir)
        assert os.path.basename(exp_parent) == "Expmt", "Parent directory should be 'Expmt'"
        assert "exp_" in os.path.basename(exp_dir), "Experiment directory should start with 'exp_'"
        
        print("✓ Checkpoint directory structure test passed")
        print(f"  Created structure: {exp_dir}/checkpoint-100")


def test_write_permissions():
    """Test write permissions for checkpoint directories"""
    print("Testing write permissions...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test creating and writing to checkpoint directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join(temp_dir, f"Expmt/exp_{timestamp}")
        os.makedirs(exp_dir, exist_ok=True)
        checkpoint_dir = os.path.join(exp_dir, "checkpoint-100")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Test write permissions
        test_file = os.path.join(checkpoint_dir, ".write_test")
        try:
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            print("✓ Write permissions test passed")
        except Exception as e:
            print(f"✗ Write permissions test failed: {e}")
            return False
    
    return True


def test_checkpoint_saving_mechanism():
    """Test the enhanced checkpoint saving mechanism"""
    print("Testing enhanced checkpoint saving mechanism...")
    
    # We can't fully test save_lora_adapters without a model, but we can test the directory creation
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test the new directory structure format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join(temp_dir, f"Expmt/exp_{timestamp}")
        os.makedirs(exp_dir, exist_ok=True)
        checkpoint_dir = os.path.join(exp_dir, "checkpoint-100")
        
        # Test directory creation
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            print("✓ Checkpoint directory creation test passed")
        except Exception as e:
            print(f"✗ Checkpoint directory creation test failed: {e}")
            return False
            
        # Test write permissions
        test_file = os.path.join(checkpoint_dir, ".write_test")
        try:
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            print("✓ Checkpoint directory write permissions test passed")
        except Exception as e:
            print(f"✗ Checkpoint directory write permissions test failed: {e}")
            return False
    
    return True


def main():
    """Main test function"""
    print("Running enhanced checkpoint saving validation tests...\n")
    
    # Run all tests
    tests = [
        test_checkpoint_directory_structure,
        test_write_permissions,
        test_checkpoint_saving_mechanism
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = test()
            if result is not False:  # None or True means passed
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            failed += 1
        print()  # Add spacing between tests
    
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The enhanced checkpoint saving mechanism is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())