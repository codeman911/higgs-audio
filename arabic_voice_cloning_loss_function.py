#!/usr/bin/env python3
"""
Comprehensive Loss Function for Arabic Voice Cloning DualFFN Training

This module implements a sophisticated loss function for training Higgs Audio v2
with DualFFN architecture on Arabic voice cloning tasks. It handles both text
and audio generation losses with proper weighting and regularization.

Key Features:
- Dual FFN loss computation (text and audio pathways)
- Multi-codebook audio loss with delay pattern support
- Voice similarity contrastive loss for better cloning
- Adaptive loss weighting and curriculum learning
- Comprehensive logging and metrics tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from loguru import logger
import math

from arabic_voice_cloning_training_collator import HiggsAudioTrainingBatch
from boson_multimodal.model.higgs_audio.configuration_higgs_audio import HiggsAudioConfig


@dataclass
class LossConfig:
    """Configuration for loss computation in DualFFN training."""
    
    # Basic loss weights
    text_loss_weight: float = 1.0         # Weight for text generation loss
    audio_loss_weight: float = 1.0        # Weight for audio generation loss
    
    # Advanced loss components
    contrastive_loss_weight: float = 0.1  # Weight for voice similarity loss
    consistency_loss_weight: float = 0.05 # Weight for audio-text consistency
    
    # Multi-codebook audio loss settings
    codebook_loss_weighting: str = "uniform"  # "uniform", "weighted", "adaptive"
    delay_pattern_loss_weight: float = 1.0    # Weight for delay pattern alignment
    
    # Regularization
    l2_regularization: float = 0.0        # L2 regularization strength
    label_smoothing: float = 0.0          # Label smoothing for text tokens
    audio_label_smoothing: float = 0.0    # Label smoothing for audio tokens
    
    # Curriculum learning
    enable_curriculum_learning: bool = False  # Enable curriculum learning
    curriculum_start_weight: float = 0.1     # Starting weight for difficult losses
    curriculum_end_weight: float = 1.0       # Final weight for difficult losses
    curriculum_steps: int = 10000             # Steps to reach final weight
    
    # Loss masking and filtering
    ignore_index: int = -100              # Index to ignore in loss computation
    min_audio_length: int = 5             # Minimum audio length for loss computation
    max_gradient_norm: float = 1.0        # Maximum gradient norm for clipping
    
    # Performance optimization
    use_mixed_precision: bool = True      # Use mixed precision for loss computation
    accumulate_grad_batches: int = 1      # Gradient accumulation steps


class HiggsAudioDualFFNLoss(nn.Module):
    """
    Comprehensive loss function for Higgs Audio DualFFN training.
    
    This loss function handles the complex requirements of training a dual FFN
    architecture where text and audio tokens are processed through separate
    pathways while sharing attention mechanisms.
    """
    
    def __init__(
        self, 
        config: HiggsAudioConfig,
        loss_config: LossConfig,
        vocab_size: int,
        audio_codebook_size: int,
        num_codebooks: int = 12
    ):
        """
        Initialize the DualFFN loss function.
        
        Args:
            config: Higgs Audio model configuration
            loss_config: Loss computation configuration
            vocab_size: Text vocabulary size
            audio_codebook_size: Audio codebook size
            num_codebooks: Number of audio codebooks
        """
        super().__init__()
        self.config = config
        self.loss_config = loss_config
        self.vocab_size = vocab_size
        self.audio_codebook_size = audio_codebook_size
        self.num_codebooks = num_codebooks
        
        # Initialize loss functions
        self.text_loss_fn = nn.CrossEntropyLoss(
            ignore_index=loss_config.ignore_index,
            label_smoothing=loss_config.label_smoothing
        )
        
        self.audio_loss_fn = nn.CrossEntropyLoss(
            ignore_index=loss_config.ignore_index,
            label_smoothing=loss_config.audio_label_smoothing
        )
        
        # Codebook loss weighting
        self.codebook_weights = self._initialize_codebook_weights()
        
        # Curriculum learning state
        self.training_step = 0
        self.curriculum_weights = self._get_curriculum_weights(0)
        
        logger.info(f"DualFFN Loss initialized:")
        logger.info(f"  - Text loss weight: {loss_config.text_loss_weight}")
        logger.info(f"  - Audio loss weight: {loss_config.audio_loss_weight}")
        logger.info(f"  - Contrastive loss weight: {loss_config.contrastive_loss_weight}")
        logger.info(f"  - Num codebooks: {num_codebooks}")
        logger.info(f"  - Curriculum learning: {loss_config.enable_curriculum_learning}")
    
    def _initialize_codebook_weights(self) -> torch.Tensor:
        """Initialize weights for different codebooks."""
        if self.loss_config.codebook_loss_weighting == "uniform":
            weights = torch.ones(self.num_codebooks)
        elif self.loss_config.codebook_loss_weighting == "weighted":
            # Give higher weight to lower-order codebooks (more important)
            weights = torch.tensor([1.0 / (i + 1) for i in range(self.num_codebooks)])
            weights = weights / weights.sum() * self.num_codebooks  # Normalize
        else:  # adaptive - will be updated during training
            weights = torch.ones(self.num_codebooks)
        
        return nn.Parameter(weights, requires_grad=False)
    
    def _get_curriculum_weights(self, step: int) -> Dict[str, float]:
        """Get curriculum learning weights for the current step."""
        if not self.loss_config.enable_curriculum_learning:
            return {"contrastive": 1.0, "consistency": 1.0}
        
        progress = min(step / self.loss_config.curriculum_steps, 1.0)
        
        # Smooth curriculum progression using cosine annealing
        curriculum_progress = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        
        weights = {}
        for loss_type in ["contrastive", "consistency"]:
            start_weight = self.loss_config.curriculum_start_weight
            end_weight = self.loss_config.curriculum_end_weight
            weights[loss_type] = start_weight + (end_weight - start_weight) * curriculum_progress
        
        return weights
    
    def forward(
        self,
        text_logits: torch.Tensor,
        audio_logits: Optional[torch.Tensor],
        batch: HiggsAudioTrainingBatch,
        audio_features: Optional[torch.Tensor] = None,
        step: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive loss for DualFFN training.
        
        Args:
            text_logits: Text generation logits [batch_size, seq_len, vocab_size]
            audio_logits: Audio generation logits [num_codebooks, audio_seq_len, codebook_size]
            batch: Training batch with labels and metadata
            audio_features: Audio features for contrastive loss
            step: Current training step for curriculum learning
            
        Returns:
            Dictionary containing all loss components and metrics
        """
        if step is not None:
            self.training_step = step
            self.curriculum_weights = self._get_curriculum_weights(step)
        
        losses = {}
        metrics = {}
        total_loss = 0.0
        
        # 1. Text Generation Loss
        if text_logits is not None and batch.labels is not None:
            text_loss = self._compute_text_loss(text_logits, batch.labels)
            losses['text_loss'] = text_loss
            total_loss += self.loss_config.text_loss_weight * text_loss
            
            # Text generation metrics
            with torch.no_grad():
                text_accuracy = self._compute_text_accuracy(text_logits, batch.labels)
                metrics['text_accuracy'] = text_accuracy
        
        # 2. Audio Generation Loss
        if audio_logits is not None and batch.audio_labels is not None:
            audio_loss, audio_metrics = self._compute_audio_loss(audio_logits, batch.audio_labels)
            losses['audio_loss'] = audio_loss
            total_loss += self.loss_config.audio_loss_weight * audio_loss
            metrics.update(audio_metrics)
        
        # 3. Voice Similarity Contrastive Loss
        if (audio_features is not None and 
            self.loss_config.contrastive_loss_weight > 0 and
            batch.audio_out_ids is not None):
            
            contrastive_loss = self._compute_contrastive_loss(audio_features, batch)
            losses['contrastive_loss'] = contrastive_loss
            weight = self.loss_config.contrastive_loss_weight * self.curriculum_weights["contrastive"]
            total_loss += weight * contrastive_loss
        
        # 4. Audio-Text Consistency Loss
        if (text_logits is not None and audio_logits is not None and
            self.loss_config.consistency_loss_weight > 0):
            
            consistency_loss = self._compute_consistency_loss(text_logits, audio_logits, batch)
            losses['consistency_loss'] = consistency_loss
            weight = self.loss_config.consistency_loss_weight * self.curriculum_weights["consistency"]
            total_loss += weight * consistency_loss
        
        # 5. Regularization
        if self.loss_config.l2_regularization > 0:
            l2_loss = self._compute_l2_regularization()
            losses['l2_loss'] = l2_loss
            total_loss += self.loss_config.l2_regularization * l2_loss
        
        # Store total loss
        losses['total_loss'] = total_loss
        
        # Add curriculum weights to metrics
        if self.loss_config.enable_curriculum_learning:
            metrics.update({f'curriculum_{k}': v for k, v in self.curriculum_weights.items()})
        
        # Log loss components periodically
        if step is not None and step % 100 == 0:
            self._log_loss_components(losses, metrics, step)
        
        return {'losses': losses, 'metrics': metrics}
    
    def _compute_text_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute text generation loss."""
        # Reshape for loss computation
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten for CrossEntropyLoss
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        
        return self.text_loss_fn(flat_logits, flat_labels)
    
    def _compute_text_accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute text generation accuracy."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        predictions = torch.argmax(shift_logits, dim=-1)
        
        # Only compute accuracy for non-ignored tokens
        mask = shift_labels != self.loss_config.ignore_index
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        
        correct = (predictions == shift_labels) & mask
        accuracy = correct.sum().float() / mask.sum().float()
        
        return accuracy
    
    def _compute_audio_loss(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-codebook audio generation loss.
        
        Args:
            logits: Audio logits [num_codebooks, seq_len, codebook_size]
            labels: Audio labels [num_codebooks, seq_len]
            
        Returns:
            Tuple of (total_audio_loss, metrics_dict)
        """
        if logits.size(0) != self.num_codebooks:
            logger.warning(f"Expected {self.num_codebooks} codebooks, got {logits.size(0)}")
        
        total_loss = 0.0
        codebook_losses = []
        codebook_accuracies = []
        
        num_codebooks = min(logits.size(0), labels.size(0))
        
        for i in range(num_codebooks):
            # Compute loss for each codebook
            codebook_logits = logits[i]  # [seq_len, codebook_size]
            codebook_labels = labels[i]  # [seq_len]
            
            # Skip if no valid labels for this codebook
            valid_mask = codebook_labels != self.loss_config.ignore_index
            if valid_mask.sum() == 0:
                codebook_losses.append(torch.tensor(0.0, device=logits.device))
                codebook_accuracies.append(torch.tensor(0.0, device=logits.device))
                continue
            
            # Compute codebook-specific loss
            codebook_loss = self.audio_loss_fn(codebook_logits, codebook_labels)
            
            # Apply codebook weighting
            weighted_loss = self.codebook_weights[i] * codebook_loss
            codebook_losses.append(weighted_loss)
            total_loss += weighted_loss
            
            # Compute codebook accuracy
            with torch.no_grad():
                predictions = torch.argmax(codebook_logits, dim=-1)
                correct = (predictions == codebook_labels) & valid_mask
                accuracy = correct.sum().float() / valid_mask.sum().float()
                codebook_accuracies.append(accuracy)
        
        # Average over codebooks
        if num_codebooks > 0:
            total_loss = total_loss / num_codebooks
        
        # Metrics
        metrics = {
            'audio_loss_total': total_loss,
            'audio_accuracy_mean': torch.stack(codebook_accuracies).mean() if codebook_accuracies else torch.tensor(0.0),
        }
        
        # Individual codebook metrics (first few for monitoring)
        for i in range(min(4, len(codebook_losses))):
            metrics[f'audio_loss_cb_{i}'] = codebook_losses[i]
            metrics[f'audio_acc_cb_{i}'] = codebook_accuracies[i]
        
        return total_loss, metrics
    
    def _compute_contrastive_loss(
        self, 
        audio_features: torch.Tensor, 
        batch: HiggsAudioTrainingBatch
    ) -> torch.Tensor:
        """
        Compute contrastive loss for voice similarity.
        
        This encourages the model to generate audio that is similar to the
        reference audio in terms of voice characteristics.
        """
        if audio_features.size(0) < 2:
            return torch.tensor(0.0, device=audio_features.device)
        
        # Extract reference and target audio features
        # Assuming audio features are ordered as [ref_audio, target_audio, ...]
        batch_size = audio_features.size(0) // 2
        
        ref_features = audio_features[:batch_size]  # Reference audio features
        target_features = audio_features[batch_size:batch_size*2]  # Target audio features
        
        # Normalize features
        ref_features = F.normalize(ref_features, dim=-1)
        target_features = F.normalize(target_features, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(ref_features, target_features.t())
        
        # Create positive pairs (same speaker) and negative pairs
        labels = torch.arange(batch_size, device=audio_features.device)
        
        # InfoNCE-style contrastive loss
        temperature = 0.1
        logits = similarity_matrix / temperature
        
        contrastive_loss = F.cross_entropy(logits, labels)
        
        return contrastive_loss
    
    def _compute_consistency_loss(
        self,
        text_logits: torch.Tensor,
        audio_logits: torch.Tensor,
        batch: HiggsAudioTrainingBatch
    ) -> torch.Tensor:
        """
        Compute consistency loss between text and audio representations.
        
        This encourages alignment between text and audio generation pathways.
        """
        # Extract text and audio token positions
        if batch.input_ids is None:
            return torch.tensor(0.0, device=text_logits.device)
        
        # Find positions where text describes audio content
        # This is a simplified implementation - can be made more sophisticated
        text_probs = F.softmax(text_logits, dim=-1)
        audio_probs = F.softmax(audio_logits.mean(dim=0), dim=-1)  # Average over codebooks
        
        # KL divergence between text and audio distributions (simplified)
        # In practice, this would require more sophisticated alignment
        consistency_loss = F.kl_div(
            torch.log(text_probs[:, :min(text_probs.size(1), audio_probs.size(0))].mean(dim=0) + 1e-8),
            audio_probs[:min(text_probs.size(1), audio_probs.size(0))].mean(dim=0),
            reduction='batchmean'
        )
        
        return consistency_loss
    
    def _compute_l2_regularization(self) -> torch.Tensor:
        """Compute L2 regularization loss."""
        l2_loss = 0.0
        param_count = 0
        
        for param in self.parameters():
            if param.requires_grad:
                l2_loss += torch.sum(param ** 2)
                param_count += param.numel()
        
        if param_count > 0:
            l2_loss = l2_loss / param_count
        
        return l2_loss
    
    def update_codebook_weights(self, codebook_losses: List[torch.Tensor]):
        """Update adaptive codebook weights based on performance."""
        if self.loss_config.codebook_loss_weighting != "adaptive":
            return
        
        # Update weights based on relative performance
        with torch.no_grad():
            loss_array = torch.stack(codebook_losses)
            # Inverse weighting - give more weight to harder codebooks
            inv_weights = 1.0 / (loss_array + 1e-8)
            normalized_weights = inv_weights / inv_weights.sum() * self.num_codebooks
            
            # Exponential moving average update
            alpha = 0.1
            self.codebook_weights.data = (1 - alpha) * self.codebook_weights.data + alpha * normalized_weights
    
    def _log_loss_components(
        self, 
        losses: Dict[str, torch.Tensor], 
        metrics: Dict[str, torch.Tensor], 
        step: int
    ):
        """Log loss components and metrics."""
        logger.info(f"Step {step} - Loss Components:")
        
        for name, loss in losses.items():
            if isinstance(loss, torch.Tensor):
                logger.info(f"  {name}: {loss.item():.6f}")
        
        # Log key metrics
        if 'text_accuracy' in metrics:
            logger.info(f"  text_accuracy: {metrics['text_accuracy'].item():.4f}")
        if 'audio_accuracy_mean' in metrics:
            logger.info(f"  audio_accuracy: {metrics['audio_accuracy_mean'].item():.4f}")
        
        # Log curriculum weights if enabled
        if self.loss_config.enable_curriculum_learning:
            logger.info(f"  curriculum_weights: {self.curriculum_weights}")


class LossMetricsTracker:
    """Track and analyze loss metrics during training."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history = {}
        self.step_count = 0
    
    def update(self, losses: Dict[str, torch.Tensor], metrics: Dict[str, torch.Tensor]):
        """Update metrics tracking."""
        self.step_count += 1
        
        # Convert tensors to scalars and store
        for name, value in {**losses, **metrics}.items():
            if isinstance(value, torch.Tensor):
                scalar_value = value.item()
            else:
                scalar_value = float(value)
            
            if name not in self.history:
                self.history[name] = []
            
            self.history[name].append(scalar_value)
            
            # Keep only recent history
            if len(self.history[name]) > self.window_size:
                self.history[name] = self.history[name][-self.window_size:]
    
    def get_averages(self, last_n: Optional[int] = None) -> Dict[str, float]:
        """Get average values over the last n steps."""
        if last_n is None:
            last_n = self.window_size
        
        averages = {}
        for name, values in self.history.items():
            if values:
                recent_values = values[-last_n:]
                averages[name] = sum(recent_values) / len(recent_values)
        
        return averages
    
    def get_trends(self) -> Dict[str, str]:
        """Get trend analysis for metrics."""
        trends = {}
        
        for name, values in self.history.items():
            if len(values) >= 10:
                recent_avg = sum(values[-5:]) / 5
                older_avg = sum(values[-10:-5]) / 5
                
                if recent_avg > older_avg * 1.05:
                    trends[name] = "increasing"
                elif recent_avg < older_avg * 0.95:
                    trends[name] = "decreasing"
                else:
                    trends[name] = "stable"
        
        return trends


# Factory function
def create_loss_function(
    config: HiggsAudioConfig,
    vocab_size: int,
    loss_config: Optional[LossConfig] = None
) -> HiggsAudioDualFFNLoss:
    """
    Factory function to create the loss function with sensible defaults.
    
    Args:
        config: Higgs Audio model configuration
        vocab_size: Text vocabulary size
        loss_config: Custom loss configuration
        
    Returns:
        Configured loss function
    """
    if loss_config is None:
        loss_config = LossConfig()
    
    return HiggsAudioDualFFNLoss(
        config=config,
        loss_config=loss_config,
        vocab_size=vocab_size,
        audio_codebook_size=config.audio_codebook_size,
        num_codebooks=config.audio_num_codebooks
    )


# Example usage
if __name__ == "__main__":
    from boson_multimodal.model.higgs_audio.configuration_higgs_audio import HiggsAudioConfig
    
    # Test configuration
    config = HiggsAudioConfig()
    loss_config = LossConfig(
        text_loss_weight=1.0,
        audio_loss_weight=1.0,
        contrastive_loss_weight=0.1,
        enable_curriculum_learning=True
    )
    
    # Create loss function
    loss_fn = create_loss_function(config, vocab_size=128256, loss_config=loss_config)
    
    logger.info("✅ Loss function created successfully")
    logger.info(f"Codebook weights: {loss_fn.codebook_weights}")