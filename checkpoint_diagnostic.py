#!/usr/bin/env python3
"""
Comprehensive diagnostic script for checkpoint saving issues
"""

import os
import sys
import logging
import tempfile
from lora import save_lora_adapters

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def diagnose_environment():
    """Diagnose the environment for checkpoint saving issues"""
    logger.info("=== Environment Diagnosis ===")
    
    # Check current working directory
    cwd = os.getcwd()
    logger.info(f"Current working directory: {cwd}")
    
    # Check Python version
    logger.info(f"Python version: {sys.version}")
    
    # Check available disk space
    import shutil
    total, used, free = shutil.disk_usage(cwd)
    logger.info(f"Disk space - Total: {total//1024//1024} MB, Used: {used//1024//1024} MB, Free: {free//1024//1024} MB")
    
    # Check user permissions
    logger.info(f"Current user: {os.getlogin()}")
    
    # Check if we can write to current directory
    test_file = os.path.join(cwd, ".permission_test")
    try:
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        logger.info("✅ Write permission to current directory: PASSED")
    except Exception as e:
        logger.error(f"❌ Write permission to current directory: FAILED - {e}")
        return False
    
    return True


def test_checkpoint_saving_with_temp_dir():
    """Test checkpoint saving with a temporary directory"""
    logger.info("=== Testing Checkpoint Saving with Temporary Directory ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = os.path.join(temp_dir, "test_checkpoint")
        logger.info(f"Using temporary directory: {checkpoint_dir}")
        
        try:
            # Create a simple mock model
            class MockModel:
                def save_pretrained(self, path):
                    logger.info(f"Mock model.save_pretrained called with path: {path}")
                    # Create directory
                    os.makedirs(path, exist_ok=True)
                    # Create files
                    with open(os.path.join(path, "adapter_config.json"), "w") as f:
                        f.write('{"test": "temp_config"}')
                    with open(os.path.join(path, "adapter_model.bin"), "w") as f:
                        f.write("temp model data")
                    logger.info("Mock model.save_pretrained completed successfully")
            
            mock_model = MockModel()
            
            # Test save_lora_adapters
            logger.info("Calling save_lora_adapters...")
            save_lora_adapters(mock_model, checkpoint_dir)
            
            # Verify files
            if os.path.exists(checkpoint_dir):
                files = os.listdir(checkpoint_dir)
                logger.info(f"Files created: {files}")
                
                expected_files = ["adapter_config.json", "adapter_model.bin"]
                missing_files = [f for f in expected_files if f not in files]
                
                if not missing_files:
                    logger.info("✅ Temporary directory checkpoint saving: PASSED")
                    return True
                else:
                    logger.error(f"❌ Missing files: {missing_files}")
                    return False
            else:
                logger.error("❌ Checkpoint directory not created")
                return False
                
        except Exception as e:
            logger.error(f"Temporary directory checkpoint saving FAILED: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False


def test_checkpoint_saving_in_project_dir():
    """Test checkpoint saving in the project directory"""
    logger.info("=== Testing Checkpoint Saving in Project Directory ===")
    
    # Use a test directory in the project
    test_dir = "checkpoint_diagnostic_test"
    checkpoint_dir = os.path.join(test_dir, "checkpoint_test")
    
    try:
        # Create test directory
        os.makedirs(test_dir, exist_ok=True)
        logger.info(f"Created test directory: {test_dir}")
        
        # Check permissions
        if not os.access(test_dir, os.W_OK):
            logger.error(f"No write permission for test directory: {test_dir}")
            return False
        
        # Create a simple mock model
        class MockModel:
            def save_pretrained(self, path):
                logger.info(f"Mock model.save_pretrained called with path: {path}")
                # Create directory
                os.makedirs(path, exist_ok=True)
                # Create files
                with open(os.path.join(path, "adapter_config.json"), "w") as f:
                    f.write('{"test": "project_config"}')
                with open(os.path.join(path, "adapter_model.bin"), "w") as f:
                    f.write("project model data")
                logger.info("Mock model.save_pretrained completed successfully")
        
        mock_model = MockModel()
        
        # Test save_lora_adapters
        logger.info("Calling save_lora_adapters...")
        save_lora_adapters(mock_model, checkpoint_dir)
        
        # Verify files
        if os.path.exists(checkpoint_dir):
            files = os.listdir(checkpoint_dir)
            logger.info(f"Files created: {files}")
            
            expected_files = ["adapter_config.json", "adapter_model.bin"]
            missing_files = [f for f in expected_files if f not in files]
            
            if not missing_files:
                logger.info("✅ Project directory checkpoint saving: PASSED")
                return True
            else:
                logger.error(f"❌ Missing files: {missing_files}")
                return False
        else:
            logger.error("❌ Checkpoint directory not created")
            return False
            
    except Exception as e:
        logger.error(f"Project directory checkpoint saving FAILED: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    finally:
        # Cleanup
        try:
            if os.path.exists(test_dir):
                import shutil
                shutil.rmtree(test_dir)
                logger.info(f"Cleaned up test directory: {test_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup test directory: {e}")


def analyze_existing_checkpoint_issues():
    """Analyze existing checkpoint issues"""
    logger.info("=== Analyzing Existing Checkpoint Issues ===")
    
    # Check expmt_v1/checkpoint-100
    checkpoint_path = "expmt_v1/checkpoint-100"
    if os.path.exists(checkpoint_path):
        logger.info(f"Checking existing checkpoint: {checkpoint_path}")
        
        # List contents
        try:
            files = os.listdir(checkpoint_path)
            logger.info(f"Existing checkpoint contents: {files}")
            
            # Check file sizes
            for file in files:
                file_path = os.path.join(checkpoint_path, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    logger.info(f"  {file}: {size} bytes")
                    
        except Exception as e:
            logger.error(f"Failed to read checkpoint directory: {e}")
    else:
        logger.info(f"Checkpoint directory does not exist: {checkpoint_path}")
        return True  # This is not a failure, just that the directory doesn't exist
    
    # Check expmt_v2 (if it exists)
    expmt_v2_path = "expmt_v2"
    if os.path.exists(expmt_v2_path):
        logger.info(f"Checking expmt_v2 directory: {expmt_v2_path}")
        
        try:
            contents = os.listdir(expmt_v2_path)
            logger.info(f"expmt_v2 contents: {contents}")
        except Exception as e:
            logger.error(f"Failed to read expmt_v2 directory: {e}")
            return False
    else:
        logger.info(f"expmt_v2 directory does not exist: {expmt_v2_path}")
    
    return True


def main():
    """Main diagnostic function"""
    logger.info("Starting comprehensive checkpoint saving diagnostics...")
    
    # Run all diagnostics
    diagnostics = [
        diagnose_environment,
        test_checkpoint_saving_with_temp_dir,
        test_checkpoint_saving_in_project_dir,
        analyze_existing_checkpoint_issues
    ]
    
    results = []
    for diagnostic in diagnostics:
        try:
            result = diagnostic()
            results.append(result)
        except Exception as e:
            logger.error(f"Diagnostic {diagnostic.__name__} failed: {e}")
            results.append(False)
    
    # Summary
    passed = sum(1 for r in results if r)
    total = len(results)
    
    logger.info(f"=== Diagnostic Summary ===")
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All diagnostics PASSED!")
        logger.info("Checkpoint saving mechanism appears to be working correctly.")
        logger.info("If checkpoint saving is still failing during training, the issue may be:")
        logger.info("1. With the actual model's save_pretrained method")
        logger.info("2. With specific conditions during training (memory, GPU, etc.)")
        logger.info("3. With the specific model being trained")
        return 0
    else:
        logger.error("❌ Some diagnostics FAILED!")
        logger.error("There may be environment or permission issues affecting checkpoint saving.")
        return 1


if __name__ == "__main__":
    sys.exit(main())