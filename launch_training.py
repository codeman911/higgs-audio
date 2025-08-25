#!/usr/bin/env python3
"""
Higgs-Audio LoRA Training Launcher
Cross-platform launcher that ensures proper directory setup and environment validation.
"""

import os
import sys
import subprocess
from pathlib import Path


def print_colored(text, color_code="0"):
    """Print colored text to terminal."""
    colors = {
        "red": "31",
        "green": "32", 
        "yellow": "33",
        "blue": "34",
        "reset": "0"
    }
    code = colors.get(color_code, color_code)
    print(f"\033[{code}m{text}\033[0m")


def validate_environment():
    """Validate the training environment."""
    print_colored("🔍 Validating environment...", "blue")
    
    # Get script directory (should be higgs-audio root)
    script_dir = Path(__file__).parent.absolute()
    trainer_dir = script_dir / "trainer"
    
    print_colored(f"📍 Script location: {script_dir}", "blue")
    print_colored(f"📍 Trainer location: {trainer_dir}", "blue")
    
    # Check directory structure
    if not (script_dir / "boson_multimodal").exists():
        print_colored("❌ Error: boson_multimodal directory not found", "red")
        print_colored("💡 This script should be run from the higgs-audio root directory", "yellow")
        print_colored("💡 Make sure you're in the correct repository location", "yellow")
        return False
    
    if not (trainer_dir / "train.py").exists():
        print_colored("❌ Error: trainer/train.py not found", "red")
        print_colored("💡 Make sure the trainer directory is properly set up", "yellow")
        return False
    
    print_colored("✅ Directory structure validated", "green")
    
    # Check Python dependencies
    print_colored("🔍 Checking Python dependencies...", "blue")
    
    try:
        import torch
        print_colored(f"✅ PyTorch: {torch.__version__}", "green")
    except ImportError:
        print_colored("❌ PyTorch not available", "red")
        return False
    
    try:
        import transformers
        print_colored(f"✅ Transformers: {transformers.__version__}", "green")
    except ImportError:
        print_colored("❌ Transformers not available", "red")
        return False
    
    try:
        import peft
        print_colored(f"✅ PEFT: {peft.__version__}", "green")
    except ImportError:
        print_colored("❌ PEFT not available", "red")
        return False
    
    return True


def show_usage():
    """Show usage examples."""
    print_colored("⚠️  No arguments provided", "yellow")
    print_colored("Usage examples:", "blue")
    print()
    print("  # Basic training:")
    print("  python3 launch_training.py --train_data path/to/training_data.json")
    print()
    print("  # Advanced training:")
    print("  python3 launch_training.py --train_data path/to/training_data.json --batch_size 2 --learning_rate 1e-4")
    print()
    print("  # Quick test:")
    print("  python3 launch_training.py --train_data path/to/training_data.json --quick_test")
    print()
    print("  # Validate data only:")
    print("  python3 launch_training.py --validate_data_only --train_data path/to/training_data.json")
    print()


def main():
    """Main launcher function."""
    print_colored("🎵 Higgs-Audio LoRA Training Launcher", "blue")
    print("=" * 50)
    
    # Show usage if no arguments
    if len(sys.argv) == 1:
        show_usage()
        return 0
    
    # Validate environment
    if not validate_environment():
        print_colored("❌ Environment validation failed", "red")
        return 1
    
    # Change to script directory (higgs-audio root)
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    print_colored(f"📂 Working directory: {os.getcwd()}", "blue")
    
    # Build command
    cmd = [sys.executable, "trainer/train.py"] + sys.argv[1:]
    
    print_colored("🚀 Starting training pipeline...", "blue")
    print_colored(f"Command: {' '.join(cmd)}", "blue")
    print()
    
    # Execute training script
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print_colored("\n⏸️ Training interrupted by user", "yellow")
        return 1
    except Exception as e:
        print_colored(f"\n❌ Error executing training: {e}", "red")
        return 1


if __name__ == "__main__":
    sys.exit(main())