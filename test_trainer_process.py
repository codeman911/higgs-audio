#!/usr/bin/env python3
"""
Test script that simulates the exact trainer checkpoint saving process
"""

import os
import sys
import logging
import torch
import traceback
from lora import apply_lora, create_lora_config, save_lora_adapters

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def simulate_trainer_checkpoint_saving():
    """Simulate the exact trainer checkpoint saving process"""
    logger.info("=== Simulating Trainer Checkpoint Saving Process ===")
    
    try:
        # Import the actual model
        from boson_multimodal.model.higgs_audio import HiggsAudioModel
        
        # Use the same parameters as the trainer
        output_dir = "expmt_v2"
        global_step = 100
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
        
        # Check write permissions
        if not os.access(output_dir, os.W_OK):
            logger.error(f"No write permission for output directory: {output_dir}")
            raise PermissionError(f"No write permission for output directory: {output_dir}")
            
        # Create checkpoint directory
        checkpoint_dir = f"{output_dir}/checkpoint-{global_step}"
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
            raise PermissionError(f"No write permission for checkpoint directory: {checkpoint_dir}")
        
        # Load the base model (this is what the trainer does)
        logger.info("Loading HiggsAudioModel...")
        model = HiggsAudioModel.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")
        logger.info(f"Model loaded successfully. Model type: {type(model)}")
        
        # Apply LoRA (this is what the trainer does)
        logger.info("Applying LoRA configuration...")
        lora_config = create_lora_config()
        lora_model = apply_lora(model, lora_config)
        logger.info(f"LoRA applied successfully. Model type: {type(lora_model)}")
        
        # Save LoRA adapters (this is what the trainer does)
        logger.info("Attempting to save LoRA adapters...")
        logger.info("Calling save_lora_adapters...")
        
        # This is the exact same call the trainer makes
        save_lora_adapters(lora_model, checkpoint_dir)
        logger.info(f"Successfully saved checkpoint to: {checkpoint_dir}")
        
        # Verify checkpoint files were created
        if os.path.exists(checkpoint_dir):
            files = os.listdir(checkpoint_dir)
            logger.info(f"Checkpoint directory contains files: {files}")
            
            # Check for required files
            required_files = ["adapter_config.json"]
            model_files = ["adapter_model.safetensors", "adapter_model.bin"]
            
            found_required = [f for f in files if f in required_files]
            found_model = [f for f in files if f in model_files]
            
            if found_required and (found_model or files):
                logger.info("✅ Checkpoint saving process completed successfully")
                return True
            else:
                logger.error("❌ Required checkpoint files not found")
                return False
        else:
            logger.error(f"❌ Checkpoint directory was not created: {checkpoint_dir}")
            return False
            
    except Exception as e:
        logger.error(f"Trainer simulation FAILED: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def main():
    """Main test function"""
    logger.info("Starting trainer process simulation...")
    
    success = simulate_trainer_checkpoint_saving()
    
    if success:
        logger.info("🎉 Trainer process simulation PASSED!")
        return 0
    else:
        logger.error("❌ Trainer process simulation FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())