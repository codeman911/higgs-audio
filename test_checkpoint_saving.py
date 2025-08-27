#!/usr/bin/env python3
"""
Test script to verify checkpoint saving functionality
"""

import os
import tempfile
import torch
from lora import save_lora_adapters

def test_checkpoint_saving():
    """Test if checkpoint saving works correctly"""
    print("Testing checkpoint saving...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        test_output_dir = os.path.join(temp_dir, "expmt_v1")
        print(f"Test output directory: {test_output_dir}")
        
        # Try to create the directory
        try:
            os.makedirs(test_output_dir, exist_ok=True)
            print(f"✓ Successfully created directory: {test_output_dir}")
        except Exception as e:
            print(f"✗ Failed to create directory: {e}")
            return False
            
        # Try to save a simple checkpoint
        try:
            # Create a simple model for testing
            model = torch.nn.Linear(10, 1)
            
            # Try to save the model
            checkpoint_dir = os.path.join(test_output_dir, "checkpoint-100")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save the model state dict
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))
            print(f"✓ Successfully saved checkpoint to: {checkpoint_dir}")
            
            # Check if the file exists
            if os.path.exists(os.path.join(checkpoint_dir, "pytorch_model.bin")):
                print("✓ Checkpoint file exists")
                return True
            else:
                print("✗ Checkpoint file does not exist")
                return False
                
        except Exception as e:
            print(f"✗ Failed to save checkpoint: {e}")
            return False

if __name__ == "__main__":
    success = test_checkpoint_saving()
    if success:
        print("\n✓ Checkpoint saving test passed")
    else:
        print("\n✗ Checkpoint saving test failed")