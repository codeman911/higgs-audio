#!/usr/bin/env python3
"""
Validation script to check the current checkpoint directory
"""

import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_checkpoint_directory(checkpoint_path):
    """Validate the checkpoint directory"""
    logger.info(f"Validating checkpoint directory: {checkpoint_path}")
    
    # Check if directory exists
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint directory does not exist: {checkpoint_path}")
        return False
        
    # Check if it's a directory
    if not os.path.isdir(checkpoint_path):
        logger.error(f"Path is not a directory: {checkpoint_path}")
        return False
        
    # Check permissions
    if not os.access(checkpoint_path, os.R_OK):
        logger.error(f"No read permission for directory: {checkpoint_path}")
        return False
        
    if not os.access(checkpoint_path, os.W_OK):
        logger.error(f"No write permission for directory: {checkpoint_path}")
        return False
        
    # List contents
    try:
        files = os.listdir(checkpoint_path)
        logger.info(f"Directory contents: {files}")
        
        # Check for expected files (newer PEFT versions use safetensors)
        expected_files = ["adapter_config.json", "adapter_model.safetensors"]
        alternate_files = ["adapter_config.json", "adapter_model.bin"]  # Older format
        
        found_files = [f for f in files if f in expected_files]
        found_alternate = [f for f in files if f in alternate_files]
        
        if len(found_files) >= 1:  # At least one of the main files
            logger.info(f"Found expected files: {found_files}")
        elif len(found_alternate) >= 1:  # At least one of the alternate files
            logger.info(f"Found alternate expected files: {found_alternate}")
        else:
            logger.warning("No expected checkpoint files found")
            
        # Check if we have the config file
        if "adapter_config.json" in files:
            logger.info("✅ Found adapter_config.json")
        else:
            logger.warning("❌ Missing adapter_config.json")
            
        # Check if we have at least one model file
        model_files = ["adapter_model.safetensors", "adapter_model.bin"]
        if any(f in files for f in model_files):
            logger.info("✅ Found at least one model file")
        else:
            logger.warning("❌ Missing model files")
            
        return True
        
    except Exception as e:
        logger.error(f"Error reading directory contents: {e}")
        return False


def main():
    """Main validation function"""
    # Check the existing checkpoint directory
    checkpoint_path = "/Users/vikram.solanki/Projects/exp/level1/higgs-audio/expmt_v1/checkpoint-100"
    
    if validate_checkpoint_directory(checkpoint_path):
        logger.info("✅ Checkpoint directory validation PASSED")
    else:
        logger.error("❌ Checkpoint directory validation FAILED")
        
    # Also check the parent directory
    parent_dir = "/Users/vikram.solanki/Projects/exp/level1/higgs-audio/expmt_v1"
    logger.info(f"Checking parent directory: {parent_dir}")
    
    if os.path.exists(parent_dir):
        try:
            contents = os.listdir(parent_dir)
            logger.info(f"Parent directory contents: {contents}")
        except Exception as e:
            logger.error(f"Error reading parent directory: {e}")


if __name__ == "__main__":
    main()