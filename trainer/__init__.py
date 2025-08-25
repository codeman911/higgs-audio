"""
Higgs-Audio LoRA Training Pipeline

A zero-shot voice cloning training pipeline that reuses existing boson_multimodal
components and follows the exact patterns from generation.py and arb_inference.py.

Key Components:
- HiggsAudioTrainer: Main training class with DualFFN loss computation
- VoiceCloningDataset: Simple dataset wrapper using prepare_chatml_sample
- TrainingConfig: Configuration management for training hyperparameters
"""

from .trainer import HiggsAudioTrainer
from .dataset import VoiceCloningDataset
from .config import TrainingConfig

__version__ = "1.0.0"
__all__ = ["HiggsAudioTrainer", "VoiceCloningDataset", "TrainingConfig"]