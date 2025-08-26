"""
Higgs-Audio LoRA Training Pipeline

A zero-shot voice cloning training pipeline that reuses existing boson_multimodal
components and follows the exact patterns from generation.py and arb_inference.py.

Key Components:
- HiggsAudioTrainer: Main training class with DualFFN loss computation
- VoiceCloningDataset: Simple dataset wrapper using prepare_chatml_sample
- TrainingConfig: Configuration management for training hyperparameters
"""

# Basic version info
__version__ = "1.0.0"

# Minimal imports to avoid distributed training conflicts
try:
    # Import configuration first (most stable)
    from trainer.config import TrainingConfig
    __all__ = ["TrainingConfig"]
except ImportError:
    # Fallback for different execution contexts
    try:
        from .config import TrainingConfig
        __all__ = ["TrainingConfig"]
    except ImportError:
        # If nothing works, provide empty namespace
        __all__ = []
        TrainingConfig = None

# Only attempt to import other components if running in proper environment
# This prevents import conflicts during distributed training
try:
    # Check if we're in a context where full imports are safe
    import sys
    if 'torch' in sys.modules and 'transformers' in sys.modules:
        # Safe to import trainer components
        try:
            from trainer.trainer import HiggsAudioTrainer
            __all__.append("HiggsAudioTrainer")
        except ImportError:
            HiggsAudioTrainer = None
            
        try:
            from trainer.dataset import VoiceCloningDataset
            __all__.append("VoiceCloningDataset")
        except ImportError:
            VoiceCloningDataset = None
            
        try:
            from trainer.audio_validation import AudioQualityValidator, audio_validator
            __all__.extend(["AudioQualityValidator", "audio_validator"])
        except ImportError:
            AudioQualityValidator = None
            audio_validator = None
except Exception:
    # In case of any issues, just provide minimal interface
    pass