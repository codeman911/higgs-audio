#!/usr/bin/env python3
"""
Trainer Package Setup Helper for Distributed Training

This script ensures that the trainer package is properly configured for 
distributed training environments where module resolution can be problematic.

Usage:
    # Run before distributed training to set up the environment
    python3 trainer_setup.py
    
    # Or import as a module to ensure setup
    import trainer_setup
"""

import os
import sys
from pathlib import Path
import importlib.util

def setup_trainer_package():
    """
    Set up trainer package for distributed training compatibility.
    
    This function ensures that:
    1. The higgs-audio root is in sys.path
    2. The trainer package is properly registered
    3. All trainer modules are pre-loaded into sys.modules
    4. Package structure is validated for distributed training
    
    Returns:
        bool: True if setup was successful, False otherwise
    """
    # Get paths
    if __name__ == "__main__":
        # Running as script
        script_dir = Path(__file__).parent
        higgs_audio_root = script_dir
    else:
        # Being imported
        script_dir = Path(__file__).parent
        higgs_audio_root = script_dir
    
    trainer_dir = higgs_audio_root / "trainer"
    
    print(f"🔧 Setting up trainer package for distributed training...")
    print(f"   Higgs-audio root: {higgs_audio_root}")
    print(f"   Trainer directory: {trainer_dir}")
    
    # 🚀 DISTRIBUTED TRAINING DETECTION
    is_distributed = (
        'RANK' in os.environ or 
        'LOCAL_RANK' in os.environ or 
        'WORLD_SIZE' in os.environ or
        'MASTER_ADDR' in os.environ
    )
    
    if is_distributed:
        print(f"   🌐 Distributed training environment detected")
    
    # Ensure paths are in sys.path
    paths_to_add = [str(higgs_audio_root)]
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
            print(f"   Added to Python path: {path}")
    
    # Check if trainer directory exists
    if not trainer_dir.exists():
        print(f"❌ Trainer directory not found: {trainer_dir}")
        return False
    
    # 🎯 CRITICAL: Ensure trainer package is properly initialized
    trainer_init_path = trainer_dir / "__init__.py"
    
    # Step 1: Force trainer to be recognized as a package
    if 'trainer' not in sys.modules:
        if trainer_init_path.exists():
            try:
                spec = importlib.util.spec_from_file_location('trainer', trainer_init_path)
                trainer_module = importlib.util.module_from_spec(spec)
                trainer_module.__path__ = [str(trainer_dir)]
                trainer_module.__package__ = 'trainer'
                sys.modules['trainer'] = trainer_module
                spec.loader.exec_module(trainer_module)
                print(f"   ✅ Trainer package initialized in sys.modules")
            except Exception as e:
                print(f"   ⚠️  Failed to initialize trainer package: {e}")
                # Create minimal package as fallback
                import types
                trainer_module = types.ModuleType('trainer')
                trainer_module.__path__ = [str(trainer_dir)]
                trainer_module.__package__ = 'trainer'
                sys.modules['trainer'] = trainer_module
                print(f"   🔧 Created minimal trainer package as fallback")
        else:
            # Create minimal trainer package if __init__.py is missing
            import types
            trainer_module = types.ModuleType('trainer')
            trainer_module.__path__ = [str(trainer_dir)]
            trainer_module.__package__ = 'trainer'
            sys.modules['trainer'] = trainer_module
            print(f"   🔧 Created minimal trainer package (no __init__.py found)")
    
    # Pre-load trainer modules into sys.modules to avoid import issues
    trainer_modules = {
        'trainer.__init__': trainer_dir / '__init__.py',
        'trainer.config': trainer_dir / 'config.py',
        'trainer.trainer': trainer_dir / 'trainer.py',
        'trainer.loss': trainer_dir / 'loss.py',
        'trainer.dataset': trainer_dir / 'dataset.py',
        'trainer.logging_utils': trainer_dir / 'logging_utils.py',
        'trainer.audio_validation': trainer_dir / 'audio_validation.py',
        'trainer.distributed_trainer': trainer_dir / 'distributed_trainer.py',
    }
    
    successfully_loaded = []
    failed_to_load = []
    
    for module_name, module_path in trainer_modules.items():
        if module_path.exists():
            try:
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                # Set package info for proper import resolution
                if '.' in module_name:
                    module.__package__ = module_name.rsplit('.', 1)[0]
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                successfully_loaded.append(module_name)
                print(f"   ✅ Pre-loaded: {module_name}")
            except Exception as e:
                failed_to_load.append((module_name, str(e)))
                print(f"   ⚠️  Failed to pre-load {module_name}: {e}")
        else:
            print(f"   ⚠️  Module file not found: {module_path}")
    
    # Summary
    print(f"📊 Trainer package setup summary:")
    print(f"   Successfully loaded: {len(successfully_loaded)} modules")
    print(f"   Failed to load: {len(failed_to_load)} modules")
    
    if is_distributed:
        print(f"   🌐 Distributed training compatibility: {'✅ Ready' if successfully_loaded else '❌ Issues detected'}")
    
    if successfully_loaded:
        print(f"✅ Trainer package setup completed successfully")
        return True
    else:
        print(f"❌ Trainer package setup failed - no modules could be loaded")
        return False

def verify_trainer_package():
    """Verify that the trainer package is working correctly."""
    print(f"🔍 Verifying trainer package...")
    
    try:
        # Test basic imports
        from trainer.config import TrainingConfig
        print(f"   ✅ TrainingConfig import successful")
        
        # Test if we can create a basic config
        config = TrainingConfig()
        print(f"   ✅ TrainingConfig instantiation successful")
        
        return True
    except Exception as e:
        print(f"   ❌ Trainer package verification failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Trainer Package Setup for Distributed Training")
    print("=" * 60)
    
    # Setup the package
    setup_success = setup_trainer_package()
    
    if setup_success:
        # Verify it works
        verify_success = verify_trainer_package()
        
        if verify_success:
            print("🎉 Trainer package is ready for distributed training!")
            sys.exit(0)
        else:
            print("❌ Trainer package setup completed but verification failed")
            sys.exit(1)
    else:
        print("❌ Trainer package setup failed")
        sys.exit(1)