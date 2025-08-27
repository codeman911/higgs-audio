#!/usr/bin/env python3
"""
Manual checkpoint saving test to diagnose issues in the actual training environment
"""

import os
import sys
import logging
import torch
from lora import save_lora_adapters

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleMockModel:
    """Simple mock model to test saving in the actual environment"""
    def save_pretrained(self, path):
        """Simulate saving model weights"""
        logger.info(f"Attempting to save to: {path}")
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Create adapter config
        adapter_config = os.path.join(path, "adapter_config.json")
        try:
            with open(adapter_config, "w") as f:
                f.write('{"test": "manual_config"}')
            logger.info(f"Created adapter config: {adapter_config}")
        except Exception as e:
            logger.error(f"Failed to create adapter config: {e}")
            raise
            
        # Create adapter model
        adapter_model = os.path.join(path, "adapter_model.bin")
        try:
            with open(adapter_model, "w") as f:
                f.write("manual test model weights")
            logger.info(f"Created adapter model: {adapter_model}")
        except Exception as e:
            logger.error(f"Failed to create adapter model: {e}")
            raise
            
        logger.info(f"Successfully saved mock checkpoint files in: {path}")


def test_manual_checkpoint_saving():
    """Test manual checkpoint saving in the actual environment"""
    logger.info("=== Manual Checkpoint Saving Test ===")
    
    # Use the same output directory structure as the training
    output_dir = "expmt_v2"
    checkpoint_dir = f"{output_dir}/checkpoint-manual-test"
    
    logger.info(f"Testing checkpoint saving to: {checkpoint_dir}")
    
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
        
        # Check write permissions
        if not os.access(output_dir, os.W_OK):
            logger.error(f"No write permission for output directory: {output_dir}")
            return False
            
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"Created checkpoint directory: {checkpoint_dir}")
        
        # Verify checkpoint directory is writable
        test_file = os.path.join(checkpoint_dir, ".write_test")
        try:
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            logger.info("Write permission test passed")
        except Exception as perm_error:
            logger.error(f"No write permission for checkpoint directory: {checkpoint_dir}, Error: {perm_error}")
            return False
        
        # Test save_lora_adapters function
        mock_model = SimpleMockModel()
        logger.info("Attempting to save LoRA adapters...")
        save_lora_adapters(mock_model, checkpoint_dir)
        
        # Verify files were created
        if os.path.exists(checkpoint_dir):
            files = os.listdir(checkpoint_dir)
            logger.info(f"Checkpoint directory contains files: {files}")
            
            expected_files = ["adapter_config.json", "adapter_model.bin"]
            missing_files = [f for f in expected_files if f not in files]
            
            if not missing_files:
                logger.info("✅ Manual checkpoint saving test PASSED")
                return True
            else:
                logger.error(f"❌ Missing expected files: {missing_files}")
                return False
        else:
            logger.error(f"❌ Checkpoint directory was not created: {checkpoint_dir}")
            return False
            
    except Exception as e:
        logger.error(f"Manual checkpoint saving test FAILED: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = test_manual_checkpoint_saving()
    
    if success:
        logger.info("🎉 Manual checkpoint saving test PASSED!")
        sys.exit(0)
    else:
        logger.error("❌ Manual checkpoint saving test FAILED!")
        sys.exit(1)