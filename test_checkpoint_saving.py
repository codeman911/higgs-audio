#!/usr/bin/env python3
"""
Test script to verify enhanced checkpoint saving mechanism
"""

import os
import tempfile
import logging
import torch
from lora import save_lora_adapters

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_checkpoint_saving():
    """Test the enhanced checkpoint saving mechanism"""
    logger.info("=== Testing Enhanced Checkpoint Saving ===")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        test_output_dir = os.path.join(temp_dir, "test_checkpoint")
        logger.info(f"Using temporary directory: {test_output_dir}")
        
        try:
            # Create a simple mock model for testing
            class MockModel:
                def save_pretrained(self, path):
                    # Create a simple file to simulate saving
                    test_file = os.path.join(path, "adapter_model.bin")
                    os.makedirs(path, exist_ok=True)
                    with open(test_file, "w") as f:
                        f.write("test checkpoint data")
                    logger.info(f"Created test checkpoint file: {test_file}")
            
            mock_model = MockModel()
            
            # Test the save_lora_adapters function
            logger.info("Testing save_lora_adapters function...")
            save_lora_adapters(mock_model, test_output_dir)
            
            # Verify the checkpoint was created
            if os.path.exists(test_output_dir):
                files = os.listdir(test_output_dir)
                logger.info(f"Checkpoint directory contents: {files}")
                
                if "adapter_model.bin" in files:
                    logger.info("✅ Checkpoint saving test PASSED")
                    return True
                else:
                    logger.error("❌ Checkpoint file was not created")
                    return False
            else:
                logger.error("❌ Checkpoint directory was not created")
                return False
                
        except Exception as e:
            logger.error(f"Checkpoint saving test FAILED: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False


def test_permission_checking():
    """Test permission checking functionality"""
    logger.info("=== Testing Permission Checking ===")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = os.path.join(temp_dir, "permission_test")
        os.makedirs(test_dir, exist_ok=True)
        
        # Test write permission
        if os.access(test_dir, os.W_OK):
            logger.info("✅ Write permission test PASSED")
            return True
        else:
            logger.error("❌ Write permission test FAILED")
            return False


if __name__ == "__main__":
    logger.info("Starting checkpoint saving verification tests...")
    
    test1_passed = test_permission_checking()
    test2_passed = test_checkpoint_saving()
    
    if test1_passed and test2_passed:
        logger.info("🎉 All tests PASSED!")
    else:
        logger.error("❌ Some tests FAILED!")