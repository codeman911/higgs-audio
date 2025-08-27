#!/usr/bin/env python3
"""
Test script to verify checkpoint saving functionality with proper error handling
"""

import os
import tempfile
import torch
import traceback
from lora import save_lora_adapters

class MockModel(torch.nn.Module):
    """A simple mock model for testing"""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
        
    def forward(self, x):
        return self.linear(x)

def test_checkpoint_saving_with_error_handling():
    """Test if checkpoint saving works correctly with error handling"""
    print("Testing checkpoint saving with error handling...")
    
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
            
        # Try to save a checkpoint
        try:
            # Create a simple model for testing
            model = MockModel()
            
            # Try to save the model
            checkpoint_dir = os.path.join(test_output_dir, "checkpoint-100")
            print(f"Attempting to create checkpoint directory: {checkpoint_dir}")
            
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"✓ Successfully created checkpoint directory: {checkpoint_dir}")
            
            # Try to save LoRA adapters
            print("Attempting to save LoRA adapters...")
            save_lora_adapters(model, checkpoint_dir)
            print(f"✓ Successfully saved LoRA adapters to: {checkpoint_dir}")
            
            # Check if files exist
            files = os.listdir(checkpoint_dir)
            print(f"Files in checkpoint directory: {files}")
            
            if len(files) > 0:
                print("✓ Checkpoint files exist")
                return True
            else:
                print("✗ No checkpoint files found")
                return False
                
        except Exception as e:
            print(f"✗ Failed to save checkpoint: {e}")
            traceback.print_exc()
            return False

def test_actual_save_function():
    """Test the actual save function with a real model"""
    print("\nTesting actual save function...")
    
    try:
        # Import the actual model and save function
        from transformers import AutoModel
        import torch
        
        # Create a simple model
        model = torch.nn.Linear(10, 1)
        
        # Test saving in a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = os.path.join(temp_dir, "checkpoint-test")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            print(f"Saving to: {checkpoint_dir}")
            save_lora_adapters(model, checkpoint_dir)
            
            files = os.listdir(checkpoint_dir)
            print(f"Files created: {files}")
            
            if "adapter_model.safetensors" in files or "pytorch_model.bin" in files:
                print("✓ Save function works correctly")
                return True
            else:
                print("✗ Save function did not create expected files")
                return False
                
    except Exception as e:
        print(f"✗ Error in actual save function test: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    success1 = test_checkpoint_saving_with_error_handling()
    success2 = test_actual_save_function()
    
    if success1 and success2:
        print("\n✓ All checkpoint saving tests passed")
    else:
        print("\n✗ Some checkpoint saving tests failed")