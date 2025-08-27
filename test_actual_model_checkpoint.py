#!/usr/bin/env python3
"""
Test script to check checkpoint saving with the actual Higgs Audio model
"""

import os
import sys
import logging
import torch
from lora import apply_lora, create_lora_config, save_lora_adapters

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_actual_model_checkpoint_saving():
    """Test checkpoint saving with the actual Higgs Audio model"""
    logger.info("=== Testing Checkpoint Saving with Actual Higgs Audio Model ===")
    
    try:
        # Import the actual model
        from boson_multimodal.model.higgs_audio import HiggsAudioModel
        from transformers import AutoTokenizer
        
        # Create a temporary directory for testing
        test_dir = "actual_model_checkpoint_test"
        checkpoint_dir = os.path.join(test_dir, "checkpoint_actual")
        
        # Clean up any existing test directory
        if os.path.exists(test_dir):
            import shutil
            shutil.rmtree(test_dir)
        
        os.makedirs(test_dir, exist_ok=True)
        logger.info(f"Created test directory: {test_dir}")
        
        # Load the base model
        logger.info("Loading HiggsAudioModel...")
        model = HiggsAudioModel.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")
        logger.info(f"Model loaded successfully. Model type: {type(model)}")
        
        # Check if model has the required methods
        logger.info("Checking model methods...")
        if hasattr(model, 'save_pretrained'):
            logger.info("✅ Model has save_pretrained method")
        else:
            logger.error("❌ Model does not have save_pretrained method")
            return False
            
        # Apply LoRA
        logger.info("Applying LoRA configuration...")
        lora_config = create_lora_config()
        lora_model = apply_lora(model, lora_config)
        logger.info(f"LoRA applied successfully. Model type: {type(lora_model)}")
        
        # Check if LoRA model has the required methods
        logger.info("Checking LoRA model methods...")
        if hasattr(lora_model, 'save_pretrained'):
            logger.info("✅ LoRA model has save_pretrained method")
        else:
            logger.error("❌ LoRA model does not have save_pretrained method")
            return False
            
        # Test saving the LoRA model
        logger.info("Attempting to save LoRA adapters...")
        try:
            save_lora_adapters(lora_model, checkpoint_dir)
            logger.info("✅ save_lora_adapters completed successfully")
        except Exception as e:
            logger.error(f"❌ save_lora_adapters failed: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
            
        # Verify files were created
        if os.path.exists(checkpoint_dir):
            files = os.listdir(checkpoint_dir)
            logger.info(f"Checkpoint directory contents: {files}")
            
            expected_files = ["adapter_config.json", "adapter_model.bin"]
            found_files = [f for f in files if f in expected_files]
            missing_files = [f for f in expected_files if f not in files]
            
            if not missing_files:
                logger.info("✅ All expected checkpoint files were created")
                logger.info("✅ Actual model checkpoint saving test PASSED")
                return True
            else:
                logger.error(f"❌ Missing expected files: {missing_files}")
                return False
        else:
            logger.error(f"❌ Checkpoint directory was not created: {checkpoint_dir}")
            return False
            
    except Exception as e:
        logger.error(f"Actual model checkpoint saving test FAILED: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    finally:
        # Cleanup
        try:
            if os.path.exists("actual_model_checkpoint_test"):
                import shutil
                shutil.rmtree("actual_model_checkpoint_test")
                logger.info("Cleaned up test directory")
        except Exception as e:
            logger.warning(f"Failed to cleanup test directory: {e}")


def main():
    """Main test function"""
    logger.info("Starting actual model checkpoint saving test...")
    
    success = test_actual_model_checkpoint_saving()
    
    if success:
        logger.info("🎉 Actual model checkpoint saving test PASSED!")
        return 0
    else:
        logger.error("❌ Actual model checkpoint saving test FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())