"""
Higgs-Audio LoRA Training Pipeline

A zero-shot voice cloning training pipeline that reuses existing boson_multimodal
components and follows the exact patterns from generation.py and arb_inference.py.

Key Components:
- HiggsAudioTrainer: Main training class with DualFFN loss computation
- VoiceCloningDataset: Simple dataset wrapper using prepare_chatml_sample
- TrainingConfig: Configuration management for training hyperparameters
"""

# Import configuration first
from trainer.config import TrainingConfig

# Conditional imports to handle dependencies
try:
    from trainer.trainer import HiggsAudioTrainer
except (ImportError, SyntaxError) as e:
    import warnings
    warnings.warn(f"Could not import HiggsAudioTrainer: {e}")
    HiggsAudioTrainer = None

try:
    from trainer.dataset import VoiceCloningDataset
except (ImportError, SyntaxError) as e:
    VoiceCloningDataset = None

# Always available imports
try:
    from trainer.audio_validation import audio_validator, AudioQualityValidator
except ImportError:
    audio_validator = None
    AudioQualityValidator = None

__version__ = "1.0.0"
__all__ = ["TrainingConfig"]

# Add available components to __all__
if HiggsAudioTrainer is not None:
    __all__.append("HiggsAudioTrainer")
if VoiceCloningDataset is not None:
    __all__.append("VoiceCloningDataset")
if AudioQualityValidator is not None:
    __all__.extend(["AudioQualityValidator", "audio_validator"])