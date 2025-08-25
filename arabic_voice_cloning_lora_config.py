#!/usr/bin/env python3
"""
LoRA Configuration for Higgs Audio DualFFN Architecture

This module implements LoRA (Low-Rank Adaptation) configuration specifically
optimized for Higgs Audio v2's DualFFN architecture. It targets both text and
audio pathways for efficient fine-tuning while maintaining model performance.

Key Features:
- Comprehensive module targeting for DualFFN layers
- Audio-focused and text-focused LoRA modes
- Proper handling of dual FFN pathways (text mlp and audio_mlp)
- Shared attention layer adaptation
- Audio head and projector targeting
- Memory-efficient rank configurations
"""

import torch
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from loguru import logger

# CRITICAL: Force correct model import path for LoRA config
current_dir = Path(__file__).parent.resolve()

# Remove any existing boson_multimodal paths from sys.path to avoid conflicts
sys_path_cleaned = []
for path in sys.path:
    path_obj = Path(path).resolve()
    # Remove paths that contain train-higgs-audio to avoid wrong model imports
    if "train-higgs-audio" not in str(path_obj):
        sys_path_cleaned.append(path)
sys.path = sys_path_cleaned

# Insert our project root at the beginning to ensure correct imports
sys.path.insert(0, str(current_dir))

# CRITICAL: Import from CORRECT boson_multimodal path (not train-higgs-audio)
# Force import the correct version by directly importing from the file
correct_model_path = current_dir / "boson_multimodal" / "model" / "higgs_audio" / "modeling_higgs_audio.py"
if correct_model_path.exists():
    # Import the correct model directly
    from boson_multimodal.model.higgs_audio.modeling_higgs_audio import (
        HiggsAudioModel, 
        HiggsAudioDualFFNDecoderLayer
    )
    from boson_multimodal.model.higgs_audio.configuration_higgs_audio import HiggsAudioConfig
else:
    # Fallback to standard import
    from boson_multimodal.model.higgs_audio.modeling_higgs_audio import (
        HiggsAudioModel, 
        HiggsAudioDualFFNDecoderLayer
    )
    from boson_multimodal.model.higgs_audio.configuration_higgs_audio import HiggsAudioConfig

from peft import LoraConfig, TaskType, get_peft_model, PeftModel


@dataclass
class HiggsAudioLoRATrainingConfig:
    """Configuration for LoRA training on Higgs Audio DualFFN architecture."""
    
    # LoRA hyperparameters
    r: int = 16                           # LoRA rank
    lora_alpha: int = 32                  # LoRA scaling parameter
    lora_dropout: float = 0.1             # LoRA dropout
    
    # Target module configuration
    target_modules_mode: str = "comprehensive"  # "comprehensive", "audio_focused", "attention_only", "custom"
    custom_target_modules: Optional[List[str]] = None
    
    # Advanced LoRA settings
    bias: str = "none"                    # "none", "all", "lora_only"
    use_rslora: bool = False              # Use rank-stabilized LoRA
    use_dora: bool = False                # Use DoRA (Weight-Decomposed Low-Rank Adaptation)
    
    # Modules to save completely (not as LoRA) - updated for actual Higgs Audio structure
    modules_to_save: List[str] = field(default_factory=lambda: ["audio_decoder_proj.text_lm_head", "audio_decoder_proj.audio_lm_head", "audio_codebook_embeddings"])
    
    # Training-specific settings
    init_lora_weights: Union[bool, str] = True  # True, False, "gaussian", "pissa"
    enable_lora_for_audio_head: bool = True     # Whether to apply LoRA to audio head
    enable_lora_for_audio_projector: bool = True  # Whether to apply LoRA to audio projector
    
    # Performance settings
    inference_mode: bool = False          # Set to True for inference-only
    
    # Layer-specific configurations
    audio_layer_r_multiplier: float = 1.5  # Higher rank for audio layers
    attention_layer_r_multiplier: float = 1.0  # Standard rank for attention
    
    def __post_init__(self):
        """Validate configuration parameters."""
        valid_modes = ["comprehensive", "audio_focused", "attention_only", "custom"]
        if self.target_modules_mode not in valid_modes:
            raise ValueError(f"target_modules_mode must be one of {valid_modes}")
        
        if self.target_modules_mode == "custom" and not self.custom_target_modules:
            raise ValueError("custom_target_modules must be provided when using 'custom' mode")
        
        if self.r <= 0:
            raise ValueError("LoRA rank (r) must be positive")
        
        if self.lora_alpha <= 0:
            raise ValueError("LoRA alpha must be positive")


class HiggsAudioLoRAConfigurator:
    """
    LoRA configuration manager for Higgs Audio DualFFN architecture.
    
    This class provides intelligent LoRA configuration based on the model's
    dual FFN architecture, targeting both text and audio processing pathways.
    """
    
    def __init__(self, model: HiggsAudioModel, config: HiggsAudioConfig):
        """
        Initialize the LoRA configurator.
        
        Args:
            model: Higgs Audio model instance
            config: Higgs Audio model configuration
        """
        self.model = model
        self.config = config
        self.dual_ffn_layers = config.audio_dual_ffn_layers or []
        
        # Analyze model architecture
        self.model_analysis = self._analyze_model_architecture()
        logger.info(f"LoRA Configurator initialized for Higgs Audio model")
        logger.info(f"DualFFN layers: {self.dual_ffn_layers}")
        logger.info(f"Model analysis: {self.model_analysis}")
    
    def _analyze_model_architecture(self) -> Dict[str, Any]:
        """Analyze the model architecture to understand available modules."""
        analysis = {
            "total_layers": 0,
            "dual_ffn_layers": [],
            "regular_layers": [],
            "has_audio_head": False,
            "has_audio_projector": False,
            "has_audio_tower": False,
            "available_modules": []
        }
        
        # Analyze transformer layers
        if hasattr(self.model, 'layers'):
            analysis["total_layers"] = len(self.model.layers)
            
            for i, layer in enumerate(self.model.layers):
                if isinstance(layer, HiggsAudioDualFFNDecoderLayer):
                    analysis["dual_ffn_layers"].append(i)
                else:
                    analysis["regular_layers"].append(i)
        
        # Check for audio-specific components
        analysis["has_audio_head"] = hasattr(self.model, 'audio_head') and self.model.audio_head is not None
        analysis["has_audio_projector"] = hasattr(self.model, 'audio_encoder_proj') and self.model.audio_encoder_proj is not None
        analysis["has_audio_tower"] = hasattr(self.model, 'audio_tower') and self.model.audio_tower is not None
        
        # Collect all available module names
        analysis["available_modules"] = self._get_all_module_names()
        
        return analysis
    
    def _get_all_module_names(self) -> List[str]:
        """Get all module names in the model for LoRA targeting."""
        module_names = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                module_names.append(name)
        return module_names
    
    def get_target_modules(self, mode: str, custom_modules: Optional[List[str]] = None) -> List[str]:
        """
        Get target modules for LoRA based on the specified mode.
        
        Args:
            mode: Target mode ("comprehensive", "audio_focused", "attention_only", "custom")
            custom_modules: Custom module list for "custom" mode
            
        Returns:
            List of target module names
        """
        if mode == "custom":
            if not custom_modules:
                raise ValueError("custom_modules must be provided for custom mode")
            return custom_modules
        
        target_modules = []
        
        if mode == "comprehensive":
            # Target all major components for maximum adaptation
            # Based on actual model structure: layers.X.module.submodule
            target_modules.extend([
                # Shared attention modules (confirmed available)
                "self_attn.q_proj", "self_attn.k_proj", 
                "self_attn.v_proj", "self_attn.o_proj",
                
                # Text FFN modules (confirmed available)
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                
                # Audio FFN modules (confirmed available - key for voice cloning)
                "audio_mlp.gate_proj", "audio_mlp.up_proj", "audio_mlp.down_proj",
            ])
            
            # Add audio decoder components that actually exist
            if "audio_decoder_proj.text_lm_head" in self.model_analysis["available_modules"]:
                target_modules.append("audio_decoder_proj.text_lm_head")
            if "audio_decoder_proj.audio_lm_head" in self.model_analysis["available_modules"]:
                target_modules.append("audio_decoder_proj.audio_lm_head")
        
        elif mode == "audio_focused":
            # Focus primarily on audio-specific modules
            target_modules.extend([
                # Audio-specific DualFFN modules (highest priority)
                "audio_mlp.gate_proj", "audio_mlp.up_proj", "audio_mlp.down_proj",
                
                # Shared attention for audio-text interaction
                "self_attn.q_proj", "self_attn.v_proj",
            ])
            
            # Add audio decoder if available
            if "audio_decoder_proj.audio_lm_head" in self.model_analysis["available_modules"]:
                target_modules.append("audio_decoder_proj.audio_lm_head")
        
        elif mode == "attention_only":
            # Target only attention mechanisms for efficient adaptation
            target_modules.extend([
                # Shared attention modules
                "self_attn.q_proj", "self_attn.k_proj", 
                "self_attn.v_proj", "self_attn.o_proj",
                
                # Audio attention modules (if present)
                "audio_attn.q_proj", "audio_attn.k_proj", 
                "audio_attn.v_proj", "audio_attn.o_proj",
            ])
        
        # Filter out modules that don't exist in the model
        existing_modules = []
        available_modules = self.model_analysis["available_modules"]
        
        for target in target_modules:
            # Check if any available module ends with this target pattern
            matching_modules = [mod for mod in available_modules if mod.endswith(target)]
            existing_modules.extend(matching_modules)
        
        # Remove duplicates while preserving order
        unique_modules = list(dict.fromkeys(existing_modules))
        
        logger.info(f"Selected {len(unique_modules)} target modules for mode '{mode}':")
        for module in unique_modules[:10]:  # Show first 10
            logger.info(f"  - {module}")
        if len(unique_modules) > 10:
            logger.info(f"  - ... and {len(unique_modules) - 10} more")
        
        return unique_modules
    
    def create_lora_config(self, training_config: HiggsAudioLoRATrainingConfig) -> LoraConfig:
        """
        Create LoRA configuration optimized for Higgs Audio DualFFN architecture.
        
        Args:
            training_config: LoRA training configuration
            
        Returns:
            Configured LoraConfig instance
        """
        # Get target modules based on mode
        target_modules = self.get_target_modules(
            training_config.target_modules_mode,
            training_config.custom_target_modules
        )
        
        if not target_modules:
            raise ValueError("No valid target modules found for the specified configuration")
        
        # Create LoRA configuration
        lora_config = LoraConfig(
            r=training_config.r,
            lora_alpha=training_config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=training_config.lora_dropout,
            bias=training_config.bias,
            task_type=TaskType.CAUSAL_LM,
            inference_mode=training_config.inference_mode,
            modules_to_save=training_config.modules_to_save,
            init_lora_weights=training_config.init_lora_weights,
            use_rslora=training_config.use_rslora,
            use_dora=training_config.use_dora,
        )
        
        logger.info(f"Created LoRA configuration:")
        logger.info(f"  - Rank: {training_config.r}")
        logger.info(f"  - Alpha: {training_config.lora_alpha}")
        logger.info(f"  - Dropout: {training_config.lora_dropout}")
        logger.info(f"  - Target modules: {len(target_modules)}")
        logger.info(f"  - Modules to save: {training_config.modules_to_save}")
        
        return lora_config
    
    def apply_lora_to_model(
        self, 
        training_config: HiggsAudioLoRATrainingConfig,
        enable_gradient_checkpointing: bool = False
    ) -> PeftModel:
        """
        Apply LoRA to the Higgs Audio model.
        
        Args:
            training_config: LoRA training configuration
            enable_gradient_checkpointing: Whether to enable gradient checkpointing
            
        Returns:
            LoRA-adapted model
        """
        # Create LoRA configuration
        lora_config = self.create_lora_config(training_config)
        
        # Apply LoRA to the model
        logger.info("Applying LoRA to Higgs Audio model...")
        lora_model = get_peft_model(self.model, lora_config)
        
        # IMPORTANT: Do not call lora_model.enable_input_require_grads() here
        # Higgs Audio model doesn't implement get_input_embeddings() method, causing NotImplementedError
        # The training will work fine without this call
        
        # Enable gradient computation for specific audio modules if needed
        self._configure_audio_module_gradients(lora_model, training_config)
        
        # Enable gradient checkpointing if requested
        if enable_gradient_checkpointing:
            # Check if the model supports gradient checkpointing
            if self._supports_gradient_checkpointing(lora_model):
                try:
                    if hasattr(lora_model, 'gradient_checkpointing_enable'):
                        lora_model.gradient_checkpointing_enable()
                        logger.info("Gradient checkpointing enabled successfully")
                    elif hasattr(lora_model, 'enable_gradient_checkpointing'):
                        lora_model.enable_gradient_checkpointing()
                        logger.info("Gradient checkpointing enabled (alternative method)")
                    else:
                        logger.warning("Gradient checkpointing method not found")
                except ValueError as e:
                    if "gradient_checkpointing" in str(e).lower():
                        logger.warning(f"Higgs Audio model doesn't support gradient checkpointing: {e}")
                        logger.info("Training will continue without gradient checkpointing")
                    else:
                        raise e
                except Exception as e:
                    logger.error(f"Failed to enable gradient checkpointing: {e}")
                    logger.info("Training will continue without gradient checkpointing")
            else:
                logger.warning("Model architecture doesn't support gradient checkpointing")
                logger.info("Training will continue without gradient checkpointing")
        
        # Log parameter statistics
        self._log_parameter_statistics(lora_model)
        
        return lora_model
    
    def _supports_gradient_checkpointing(self, model) -> bool:
        """Check if the model supports gradient checkpointing."""
        # Check if model has gradient_checkpointing attribute
        if hasattr(model, 'gradient_checkpointing'):
            return True
        
        # Check if underlying model (for PEFT models) supports it
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'gradient_checkpointing'):
            return True
            
        # Check if any of the model's modules support gradient checkpointing
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing'):
                return True
                
        # Higgs Audio models typically don't support gradient checkpointing
        return False
    
    def _configure_audio_module_gradients(
        self, 
        lora_model: PeftModel, 
        training_config: HiggsAudioLoRATrainingConfig
    ):
        """Configure gradient computation for specific audio modules."""
        
        # Enable gradients for audio head if configured
        if training_config.enable_lora_for_audio_head and self.model_analysis["has_audio_head"]:
            for name, module in lora_model.named_modules():
                if "audio_head" in name:
                    for param in module.parameters():
                        param.requires_grad = True
            logger.info("Enabled gradients for audio_head modules")
        
        # Enable gradients for audio projector if configured
        if training_config.enable_lora_for_audio_projector and self.model_analysis["has_audio_projector"]:
            for name, module in lora_model.named_modules():
                if "audio_encoder_proj" in name or "audio_tower" in name:
                    for param in module.parameters():
                        param.requires_grad = True
            logger.info("Enabled gradients for audio projector modules")
    
    def _log_parameter_statistics(self, lora_model: PeftModel):
        """Log parameter statistics for the LoRA model."""
        total_params = sum(p.numel() for p in lora_model.parameters())
        trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        
        logger.info(f"LoRA Model Parameter Statistics:")
        logger.info(f"  - Total parameters: {total_params:,}")
        logger.info(f"  - Trainable parameters: {trainable_params:,}")
        logger.info(f"  - Trainable percentage: {100 * trainable_params / total_params:.2f}%")
        
        # Log LoRA-specific statistics
        lora_params = sum(p.numel() for n, p in lora_model.named_parameters() if "lora_" in n)
        logger.info(f"  - LoRA parameters: {lora_params:,}")
        logger.info(f"  - LoRA percentage: {100 * lora_params / total_params:.2f}%")
    
    def get_recommended_config(self, training_scenario: str = "voice_cloning") -> HiggsAudioLoRATrainingConfig:
        """
        Get recommended LoRA configuration for specific training scenarios.
        
        Args:
            training_scenario: Training scenario ("voice_cloning", "general_audio", "efficiency")
            
        Returns:
            Recommended LoRA training configuration
        """
        if training_scenario == "voice_cloning":
            # Optimized for voice cloning with focus on audio pathways
            return HiggsAudioLoRATrainingConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules_mode="comprehensive",
                audio_layer_r_multiplier=1.5,
                enable_lora_for_audio_head=True,
                enable_lora_for_audio_projector=True,
                modules_to_save=["audio_decoder_proj.text_lm_head", "audio_decoder_proj.audio_lm_head", "audio_codebook_embeddings"]
            )
        
        elif training_scenario == "general_audio":
            # Balanced configuration for general audio tasks
            return HiggsAudioLoRATrainingConfig(
                r=12,
                lora_alpha=24,
                lora_dropout=0.1,
                target_modules_mode="audio_focused",
                enable_lora_for_audio_head=True,
                enable_lora_for_audio_projector=False,
                modules_to_save=["audio_decoder_proj.audio_lm_head", "audio_codebook_embeddings"]
            )
        
        elif training_scenario == "efficiency":
            # Minimal configuration for fast training
            return HiggsAudioLoRATrainingConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules_mode="attention_only",
                enable_lora_for_audio_head=False,
                enable_lora_for_audio_projector=False,
                modules_to_save=[]
            )
        
        else:
            raise ValueError(f"Unknown training scenario: {training_scenario}")


# Utility functions
def create_higgs_audio_lora_model(
    model_path: str = "bosonai/higgs-audio-v2-generation-3B-base",
    training_scenario: str = "voice_cloning",
    custom_config: Optional[HiggsAudioLoRATrainingConfig] = None,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
    enable_gradient_checkpointing: bool = False  # Disabled by default for Higgs Audio
) -> Tuple[PeftModel, HiggsAudioConfig, HiggsAudioLoRATrainingConfig]:
    """
    Factory function to create a LoRA-adapted Higgs Audio model.
    
    Args:
        model_path: Path to the Higgs Audio model
        training_scenario: Training scenario for recommended config
        custom_config: Custom LoRA configuration (overrides training_scenario)
        device_map: Device mapping for model loading
        torch_dtype: Torch data type for model
        enable_gradient_checkpointing: Whether to enable gradient checkpointing
        
    Returns:
        Tuple of (LoRA model, model config, LoRA config)
    """
    # Load the base model
    logger.info(f"Loading Higgs Audio model from {model_path}")
    model = HiggsAudioModel.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    config = HiggsAudioConfig.from_pretrained(model_path)
    
    # Create LoRA configurator
    configurator = HiggsAudioLoRAConfigurator(model, config)
    
    # Use custom config or get recommended config
    if custom_config is not None:
        lora_config = custom_config
        logger.info(f"Using custom LoRA configuration")
    else:
        lora_config = configurator.get_recommended_config(training_scenario)
        logger.info(f"Using recommended LoRA configuration for '{training_scenario}'")
    
    # Apply LoRA
    lora_model = configurator.apply_lora_to_model(
        lora_config,
        enable_gradient_checkpointing=enable_gradient_checkpointing
    )
    
    return lora_model, config, lora_config


# Example usage
if __name__ == "__main__":
    # Test LoRA configuration
    try:
        # Create LoRA-adapted model for voice cloning
        lora_model, config, lora_config = create_higgs_audio_lora_model(
            training_scenario="voice_cloning",
            enable_gradient_checkpointing=False  # Disabled for Higgs Audio compatibility
        )
        
        logger.info("✅ LoRA configuration test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ LoRA configuration test failed: {e}")
        raise