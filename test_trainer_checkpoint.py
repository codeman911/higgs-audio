#!/usr/bin/env python3
"""
Test script to verify checkpoint saving in a trainer-like environment
"""

import os
import tempfile
import logging
import torch
import argparse
from lora import save_lora_adapters

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockModel:
    """Mock model class to simulate HiggsAudioModel"""
    def __init__(self):
        pass
        
    def save_pretrained(self, path):
        """Simulate saving model weights"""
        # Create mock adapter files
        os.makedirs(path, exist_ok=True)
        
        # Create adapter config
        adapter_config = os.path.join(path, "adapter_config.json")
        with open(adapter_config, "w") as f:
            f.write('{"test": "config"}')
            
        # Create adapter model
        adapter_model = os.path.join(path, "adapter_model.bin")
        with open(adapter_model, "w") as f:
            f.write("mock model weights")
            
        logger.info(f"Created mock checkpoint files in: {path}")


class MockTrainer:
    """Mock trainer class to simulate checkpoint saving"""
    
    def __init__(self, output_dir, local_rank=0, world_size=1):
        self.args = argparse.Namespace(
            output_dir=output_dir,
            save_steps=10
        )
        self.local_rank = local_rank
        self.world_size = world_size
        self.global_step = 0
        
    def _test_checkpoint_saving(self):
        """Test the enhanced checkpoint saving mechanism"""
        try:
            # Only save from main process in distributed training
            if self.local_rank == 0:
                # Ensure output directory exists
                os.makedirs(self.args.output_dir, exist_ok=True)
                logger.info(f"Created output directory: {self.args.output_dir}")
                
                # Check write permissions
                if not os.access(self.args.output_dir, os.W_OK):
                    logger.error(f"No write permission for output directory: {self.args.output_dir}")
                    raise PermissionError(f"No write permission for output directory: {self.args.output_dir}")
                    
                # Create checkpoint directory
                checkpoint_dir = f"{self.args.output_dir}/checkpoint-{self.global_step}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                logger.info(f"Created checkpoint directory: {checkpoint_dir}")
                
                # Verify checkpoint directory is writable
                test_file = os.path.join(checkpoint_dir, ".write_test")
                try:
                    with open(test_file, "w") as f:
                        f.write("test")
                    os.remove(test_file)
                except Exception as perm_error:
                    logger.error(f"No write permission for checkpoint directory: {checkpoint_dir}, Error: {perm_error}")
                    raise PermissionError(f"No write permission for checkpoint directory: {checkpoint_dir}")
                
                # Save LoRA adapters
                model_to_save = MockModel()  # In real implementation, this would be self.model.module if self.world_size > 1 else self.model
                logger.info("Attempting to save LoRA adapters...")
                save_lora_adapters(model_to_save, checkpoint_dir)
                logger.info(f"Successfully saved checkpoint to: {checkpoint_dir}")
                
                # Verify checkpoint files were created
                if os.path.exists(checkpoint_dir):
                    files = os.listdir(checkpoint_dir)
                    logger.info(f"Checkpoint directory contains files: {files}")
                    if not files or (len(files) == 1 and '.write_test' in files):
                        logger.warning(f"Checkpoint directory may be incomplete: {checkpoint_dir}")
                        return False
                    return True
                else:
                    logger.error(f"Checkpoint directory was not created: {checkpoint_dir}")
                    return False
                    
            # In distributed training, synchronize all processes
            if self.world_size > 1:
                # In real implementation: torch.distributed.barrier()
                logger.info("Checkpoint saved and synchronized across all processes")
                
            return True
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint at step {self.global_step}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False


def test_checkpoint_saving_single_process():
    """Test checkpoint saving in single process mode"""
    logger.info("=== Testing Checkpoint Saving (Single Process) ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "test_output")
        trainer = MockTrainer(output_dir, local_rank=0, world_size=1)
        trainer.global_step = 100
        
        success = trainer._test_checkpoint_saving()
        
        if success:
            logger.info("✅ Single process checkpoint saving test PASSED")
            return True
        else:
            logger.error("❌ Single process checkpoint saving test FAILED")
            return False


def test_checkpoint_saving_distributed():
    """Test checkpoint saving in distributed mode (main process only)"""
    logger.info("=== Testing Checkpoint Saving (Distributed - Main Process) ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "test_output_dist")
        trainer = MockTrainer(output_dir, local_rank=0, world_size=8)  # Simulate 8 GPU setup
        trainer.global_step = 200
        
        success = trainer._test_checkpoint_saving()
        
        if success:
            logger.info("✅ Distributed checkpoint saving test PASSED")
            return True
        else:
            logger.error("❌ Distributed checkpoint saving test FAILED")
            return False


def test_permission_errors():
    """Test error handling for permission issues"""
    logger.info("=== Testing Permission Error Handling ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "permission_test")
        os.makedirs(output_dir, exist_ok=True)
        
        # Make directory read-only to simulate permission error
        os.chmod(output_dir, 0o444)
        
        trainer = MockTrainer(output_dir, local_rank=0, world_size=1)
        trainer.global_step = 300
        
        try:
            success = trainer._test_checkpoint_saving()
            if not success:
                logger.info("✅ Permission error handling test PASSED")
                return True
            else:
                logger.error("❌ Permission error handling test FAILED - should have failed")
                return False
        except PermissionError:
            logger.info("✅ Permission error handling test PASSED")
            return True
        except Exception as e:
            logger.error(f"❌ Permission error handling test FAILED - unexpected error: {e}")
            return False
        finally:
            # Restore permissions for cleanup
            os.chmod(output_dir, 0o755)


if __name__ == "__main__":
    logger.info("Starting comprehensive checkpoint saving tests...")
    
    test1_passed = test_checkpoint_saving_single_process()
    test2_passed = test_checkpoint_saving_distributed()
    test3_passed = test_permission_errors()
    
    if test1_passed and test2_passed and test3_passed:
        logger.info("🎉 All comprehensive tests PASSED!")
    else:
        logger.error("❌ Some comprehensive tests FAILED!")