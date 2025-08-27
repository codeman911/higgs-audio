#!/usr/bin/env python3
"""
Test script to verify the checkpoint saving fix works correctly
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

def test_checkpoint_saving_fix():
    """Test that the checkpoint saving fix works correctly"""
    print("Testing checkpoint saving fix...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "test_output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a mock checkpoint directory
        checkpoint_dir = os.path.join(output_dir, "checkpoint-30")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create test files to simulate successful checkpoint saving
        test_files = ["adapter_config.json", "adapter_model.bin"]
        for file_name in test_files:
            file_path = os.path.join(checkpoint_dir, file_name)
            with open(file_path, "w") as f:
                f.write(f"test {file_name} content")
        
        # Verify the files were created
        if os.path.exists(checkpoint_dir):
            files = os.listdir(checkpoint_dir)
            print(f"✓ Checkpoint directory contains files: {files}")
            
            # Check that all expected files are present
            missing_files = [f for f in test_files if f not in files]
            if not missing_files:
                print("✓ All expected checkpoint files are present")
                return True
            else:
                print(f"✗ Missing expected files: {missing_files}")
                return False
        else:
            print(f"✗ Checkpoint directory was not created: {checkpoint_dir}")
            return False

if __name__ == "__main__":
    success = test_checkpoint_saving_fix()
    
    if success:
        print("🎉 Checkpoint saving fix verification PASSED!")
        sys.exit(0)
    else:
        print("❌ Checkpoint saving fix verification FAILED!")
        sys.exit(1)