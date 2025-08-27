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
        
        # Check for expected files
        expected_files = ["adapter_config.json", "adapter_model.bin"]
        found_files = [f for f in files if f in expected_files]
        missing_files = [f for f in expected_files if f not in files]
        
        if found_files:
            logger.info(f"Found expected files: {found_files}")
        else:
            logger.warning("No expected checkpoint files found")
            
        if missing_files:
            logger.warning(f"Missing expected files: {missing_files}")
            
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