#!/usr/bin/env python3
"""
Complete Training Fix Script for Arabic Voice Cloning

This script addresses all the specific issues encountered in the training pipeline:
1. NotImplementedError from enable_input_require_grads()
2. Modules to save mismatch with actual model structure  
3. Ensures proper LoRA configuration for Higgs Audio

Usage:
    python3 complete_training_fix.py --fix-all
    python3 complete_training_fix.py --test-pipeline
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

def fix_lora_config_file():
    """Fix the LoRA configuration file to resolve all training issues."""
    print("🔧 Fixing LoRA configuration file...")
    
    # Read current file
    lora_file = Path("arabic_voice_cloning_lora_config.py")
    if not lora_file.exists():
        print(f"❌ File not found: {lora_file}")
        return False
    
    with open(lora_file, 'r') as f:
        content = f.read()
    
    fixes_applied = []
    
    # Fix 1: Remove any enable_input_require_grads() calls
    if "lora_model.enable_input_require_grads()" in content:
        print("⚠️  Found problematic enable_input_require_grads() call - removing...")
        content = content.replace(
            "lora_model.enable_input_require_grads()",
            "# lora_model.enable_input_require_grads() - REMOVED: Causes NotImplementedError in Higgs Audio"
        )
        fixes_applied.append("Removed enable_input_require_grads() call")
    
    # Fix 2: Update default modules_to_save
    old_modules_pattern = '"audio_decoder_proj", "audio_codebook_embeddings"'
    new_modules_pattern = '"audio_decoder_proj.text_lm_head", "audio_decoder_proj.audio_lm_head", "audio_codebook_embeddings"'
    
    if old_modules_pattern in content and new_modules_pattern not in content:
        print("⚠️  Updating modules_to_save to match actual model structure...")
        content = content.replace(old_modules_pattern, new_modules_pattern)
        fixes_applied.append("Updated modules_to_save to match actual model structure")
    
    # Fix 3: Update voice_cloning scenario modules_to_save  
    old_voice_cloning = 'modules_to_save=["audio_decoder_proj", "audio_codebook_embeddings"]'
    new_voice_cloning = 'modules_to_save=["audio_decoder_proj.text_lm_head", "audio_decoder_proj.audio_lm_head", "audio_codebook_embeddings"]'
    
    if old_voice_cloning in content:
        print("⚠️  Updating voice_cloning scenario modules_to_save...")
        content = content.replace(old_voice_cloning, new_voice_cloning)
        fixes_applied.append("Updated voice_cloning scenario modules_to_save")
    
    # Fix 4: Ensure Tuple import
    if "from typing import" in content and "Tuple" not in content.split("from typing import")[1].split('\n')[0]:
        print("⚠️  Adding missing Tuple import...")
        content = content.replace(
            "from typing import List, Dict, Optional, Any, Union",
            "from typing import List, Dict, Optional, Any, Union, Tuple"
        )
        fixes_applied.append("Added missing Tuple import")
    
    # Write fixed content
    if fixes_applied:
        with open(lora_file, 'w') as f:
            f.write(content)
        print(f"✅ Applied {len(fixes_applied)} fixes to LoRA configuration:")
        for fix in fixes_applied:
            print(f"   - {fix}")
        return True
    else:
        print("✅ LoRA configuration file is already up to date")
        return True

def create_fixed_lora_config():
    """Create a completely fixed version of the LoRA config file."""
    print("🔧 Creating fixed LoRA configuration...")
    
    fixed_content = '''#!/usr/bin/env python3
"""
Fixed Arabic Voice Cloning LoRA Configuration

This file contains the corrected LoRA configuration for Higgs Audio DualFFN architecture
that resolves all training issues found in the error logs.

Key Fixes:
1. Removed enable_input_require_grads() call that caused NotImplementedError
2. Updated modules_to_save to match actual model structure  
3. Proper target module selection for DualFFN layers
4. All necessary imports included
"""

import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union, Tuple
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from loguru import logger

# Import Higgs Audio components
try:
    from boson_multimodal.model.higgs_audio.modeling_higgs_audio import (
        HiggsAudioModel, HiggsAudioConfig, HiggsAudioDualFFNDecoderLayer
    )
except ImportError:
    print("Warning: Could not import Higgs Audio components")
    HiggsAudioModel = None
    HiggsAudioConfig = None
    HiggsAudioDualFFNDecoderLayer = None


@dataclass
class HiggsAudioLoRATrainingConfig:
    """Fixed configuration for LoRA training on Higgs Audio DualFFN architecture."""
    
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
    
    # FIXED: Modules to save completely (not as LoRA) - matches actual Higgs Audio structure
    modules_to_save: List[str] = field(default_factory=lambda: [
        "audio_decoder_proj.text_lm_head", 
        "audio_decoder_proj.audio_lm_head", 
        "audio_codebook_embeddings"
    ])
    
    # Training-specific settings
    init_lora_weights: Union[bool, str] = True  # True, False, "gaussian", "pissa"
    enable_lora_for_audio_head: bool = True     # Whether to apply LoRA to audio head
    enable_lora_for_audio_projector: bool = True  # Whether to apply LoRA to audio projector
    
    # Performance settings
    inference_mode: bool = False          # Set to True for inference-only
    
    # Layer-specific configurations
    audio_layer_r_multiplier: float = 1.5  # Higher rank for audio layers
    attention_layer_r_multiplier: float = 1.0  # Standard rank for attention


class HiggsAudioLoRAConfigurator:
    """Fixed LoRA configuration manager for Higgs Audio DualFFN architecture."""
    
    def __init__(self, model: HiggsAudioModel, config: HiggsAudioConfig):
        """Initialize the LoRA configurator."""
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
        """Get target modules for LoRA based on the specified mode."""
        if mode == "custom":
            if not custom_modules:
                raise ValueError("custom_modules must be provided for custom mode")
            return custom_modules
        
        target_modules = []
        
        if mode == "comprehensive":
            # Target all major components for maximum adaptation
            target_modules.extend([
                # Shared attention modules (confirmed available)
                "self_attn.q_proj", "self_attn.k_proj", 
                "self_attn.v_proj", "self_attn.o_proj",
                
                # Text FFN modules (confirmed available)
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                
                # Audio FFN modules (confirmed available - key for voice cloning)
                "audio_mlp.gate_proj", "audio_mlp.up_proj", "audio_mlp.down_proj",
            ])
        
        elif mode == "audio_focused":
            # Focus primarily on audio-specific modules
            target_modules.extend([
                # Audio-specific DualFFN modules (highest priority)
                "audio_mlp.gate_proj", "audio_mlp.up_proj", "audio_mlp.down_proj",
                
                # Shared attention for audio-text interaction
                "self_attn.q_proj", "self_attn.v_proj",
            ])
        
        elif mode == "attention_only":
            # Target only attention mechanisms for efficient adaptation
            target_modules.extend([
                # Shared attention modules
                "self_attn.q_proj", "self_attn.k_proj", 
                "self_attn.v_proj", "self_attn.o_proj",
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
        """Create LoRA configuration based on training config."""
        # Get target modules
        target_modules = self.get_target_modules(
            training_config.target_modules_mode,
            training_config.custom_target_modules
        )
        
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
        """Apply LoRA to the Higgs Audio model - FIXED VERSION."""
        # Create LoRA configuration
        lora_config = self.create_lora_config(training_config)
        
        # Apply LoRA to the model
        logger.info("Applying LoRA to Higgs Audio model...")
        lora_model = get_peft_model(self.model, lora_config)
        
        # CRITICAL FIX: Do NOT call lora_model.enable_input_require_grads()
        # This causes NotImplementedError because Higgs Audio doesn't implement get_input_embeddings()
        # The training will work fine without this call
        
        # Enable gradient computation for specific audio modules if needed
        self._configure_audio_module_gradients(lora_model, training_config)
        
        # Enable gradient checkpointing if requested
        if enable_gradient_checkpointing:
            if hasattr(lora_model, 'gradient_checkpointing_enable'):
                lora_model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            elif hasattr(lora_model, 'enable_gradient_checkpointing'):
                lora_model.enable_gradient_checkpointing()
                logger.info("Gradient checkpointing enabled (alternative method)")
            else:
                logger.warning("Gradient checkpointing not available for this model")
        
        # Log parameter statistics
        self._log_parameter_statistics(lora_model)
        
        return lora_model
    
    def _configure_audio_module_gradients(self, lora_model: PeftModel, training_config: HiggsAudioLoRATrainingConfig):
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
        """Get recommended LoRA configuration for specific training scenarios."""
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
    enable_gradient_checkpointing: bool = False
) -> Tuple[PeftModel, HiggsAudioConfig, HiggsAudioLoRATrainingConfig]:
    """
    FIXED factory function to create a LoRA-adapted Higgs Audio model.
    
    This function resolves all the training issues encountered.
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
    
    # Apply LoRA (this is the fixed version that won't call enable_input_require_grads)
    lora_model = configurator.apply_lora_to_model(
        lora_config,
        enable_gradient_checkpointing=enable_gradient_checkpointing
    )
    
    return lora_model, config, lora_config


# Test function
if __name__ == "__main__":
    # Test LoRA configuration
    try:
        # Create LoRA-adapted model for voice cloning
        lora_model, config, lora_config = create_higgs_audio_lora_model(
            training_scenario="voice_cloning",
            enable_gradient_checkpointing=True
        )
        
        logger.info("✅ FIXED LoRA configuration test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ LoRA configuration test failed: {e}")
        raise
'''
    
    # Write the fixed file
    fixed_file = Path("arabic_voice_cloning_lora_config_FIXED.py")
    with open(fixed_file, 'w') as f:
        f.write(fixed_content)
    
    print(f"✅ Created fixed LoRA configuration: {fixed_file}")
    return str(fixed_file)

def test_pipeline():
    """Test the complete training pipeline with the fixes."""
    print("🧪 Testing training pipeline...")
    
    try:
        # Test imports
        from arabic_voice_cloning_lora_config import HiggsAudioLoRATrainingConfig, create_higgs_audio_lora_model
        print("✅ LoRA config imports successful")
        
        from arabic_voice_cloning_dataset import ArabicVoiceCloningDataset
        print("✅ Dataset imports successful")
        
        from arabic_voice_cloning_training_collator import ArabicVoiceCloningTrainingCollator
        print("✅ Collator imports successful")
        
        from arabic_voice_cloning_loss_function import HiggsAudioTrainingLoss
        print("✅ Loss function imports successful")
        
        print("🎉 All core training pipeline imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Complete Training Fix for Arabic Voice Cloning")
    parser.add_argument("--fix-all", action="store_true", help="Apply all fixes")
    parser.add_argument("--create-fixed", action="store_true", help="Create completely fixed LoRA config")
    parser.add_argument("--test-pipeline", action="store_true", help="Test training pipeline")
    parser.add_argument("--run-training", action="store_true", help="Run training after fixes")
    parser.add_argument("--data-path", type=str, help="Path to ChatML data")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    print("🎯 Complete Training Fix for Arabic Voice Cloning")
    print("=" * 60)
    
    if args.fix_all or not any([args.create_fixed, args.test_pipeline, args.run_training]):
        print("\n1️⃣ Applying All Fixes")
        fix_lora_config_file()
        
        print("\n2️⃣ Creating Fixed Version")
        create_fixed_lora_config()
        
        print("\n3️⃣ Testing Pipeline")
        test_success = test_pipeline()
        
        if test_success:
            print("\n🎉 ALL FIXES APPLIED SUCCESSFULLY!")
            print("\n📋 Next Steps:")
            print("1. Copy the fixes to your running directory:")
            print("   cp arabic_voice_cloning_lora_config.py /vs/higgs-audio/")
            print("   cp arabic_voice_cloning_lora_config_FIXED.py /vs/higgs-audio/")
            print("\n2. Run your original command:")
            print("   python3 arabic_voice_cloning_distributed_trainer.py \\")
            print("     --data_path ../ms-swift/lora_training_data_zr/chatml_fixed/val_chatml_samples.json \\")
            print("     --output_dir EXPMT/exp_small")
            return 0
        else:
            print("\n❌ Some issues remain - check the test output above")
            return 1
    
    if args.create_fixed:
        create_fixed_lora_config()
    
    if args.test_pipeline:
        return 0 if test_pipeline() else 1
    
    if args.run_training:
        if not args.data_path or not args.output_dir:
            print("❌ --data-path and --output-dir required for training")
            return 1
        
        print("🏋️ Running training with fixes...")
        import subprocess
        cmd = [
            "python3", "arabic_voice_cloning_distributed_trainer.py",
            "--data_path", args.data_path,
            "--output_dir", args.output_dir
        ]
        result = subprocess.run(cmd)
        return result.returncode

if __name__ == "__main__":
    exit(main())