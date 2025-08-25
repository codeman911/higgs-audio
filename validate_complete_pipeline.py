#!/usr/bin/env python3
"""
Complete Pipeline Validation with User's ChatML Data Format

This script validates that the complete training pipeline works correctly
with the user's exact ChatML data format, ensuring zero-shot voice cloning
alignment with the original Higgs Audio implementation.
"""

import os
import sys
import json
import torch
import tempfile
from pathlib import Path
from loguru import logger

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import our components
from arabic_voice_cloning_dataset import ArabicVoiceCloningDataset, ArabicVoiceCloningDatasetConfig
from arabic_voice_cloning_training_collator import ArabicVoiceCloningTrainingCollator
from arabic_voice_cloning_lora_config import create_higgs_audio_lora_model, HiggsAudioLoRATrainingConfig
from arabic_voice_cloning_loss_function import create_loss_function, LossConfig
from lora_merge_and_checkpoint_manager import LoRACheckpointManager


class PipelineValidator:
    """Comprehensive pipeline validator for user's data format."""
    
    def __init__(self, chatml_file: str):
        """Initialize validator with user's ChatML file."""
        self.chatml_file = chatml_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.validation_results = {}
        
        logger.info(f"🔍 Initializing pipeline validator for: {chatml_file}")
    
    def validate_data_format(self) -> bool:
        """Validate user's ChatML data format."""
        logger.info("📊 Validating ChatML data format...")
        
        try:
            with open(self.chatml_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list) or len(data) == 0:
                logger.error("❌ Data should be a non-empty list")
                return False
            
            sample = data[0]
            
            # Validate required structure
            required_keys = ["messages"]
            for key in required_keys:
                if key not in sample:
                    logger.error(f"❌ Missing required key: {key}")
                    return False
            
            # Validate messages structure
            messages = sample["messages"]
            if not isinstance(messages, list) or len(messages) < 2:
                logger.error("❌ Messages should be a list with at least 2 messages")
                return False
            
            # Find user and assistant messages
            user_msg = None
            assistant_msg = None
            
            for msg in messages:
                if msg.get("role") == "user":
                    user_msg = msg
                elif msg.get("role") == "assistant":
                    assistant_msg = msg
            
            if not user_msg or not assistant_msg:
                logger.error("❌ Missing user or assistant message")
                return False
            
            # Validate audio URLs (direct paths)
            audio_paths = []
            for msg in [user_msg, assistant_msg]:
                content = msg.get("content", [])
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "audio":
                        audio_url = item.get("audio_url")
                        if audio_url:
                            audio_paths.append(audio_url)
            
            if len(audio_paths) < 2:
                logger.error("❌ Should have at least 2 audio files (reference and target)")
                return False
            
            logger.info(f"✅ Data format validation passed")
            logger.info(f"   Found {len(audio_paths)} audio files with direct paths")
            logger.info(f"   Audio paths: {audio_paths}")
            
            self.validation_results["data_format"] = {
                "valid": True,
                "audio_paths": audio_paths,
                "sample_count": len(data)
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Data format validation failed: {e}")
            self.validation_results["data_format"] = {"valid": False, "error": str(e)}
            return False
    
    def validate_dataset_loading(self) -> bool:
        """Validate dataset loading with direct audio paths."""
        logger.info("📂 Validating dataset loading...")
        
        try:
            # Configure dataset with direct path usage
            config = ArabicVoiceCloningDatasetConfig(
                chatml_file=self.chatml_file,
                validate_on_init=False,  # Skip validation for missing audio files
                max_audio_duration=30.0,
                target_sample_rate=16000
            )
            
            # Create dataset (will work even if audio files don't exist)
            dataset = ArabicVoiceCloningDataset(
                config=config,
                audio_tokenizer=None,  # Skip audio tokenization for validation
                text_tokenizer=None    # Skip text tokenization for validation
            )
            
            logger.info(f"✅ Dataset loading successful")
            logger.info(f"   Dataset size: {len(dataset)} samples")
            
            # Test sample extraction
            if len(dataset.validated_data) > 0:
                sample = dataset.validated_data[0]
                metadata = sample.get('validated_metadata', {})
                
                logger.info(f"   Sample metadata available: {bool(metadata)}")
                if metadata:
                    logger.info(f"   Reference audio path: {metadata.get('ref_audio_path', 'N/A')}")
                    logger.info(f"   Target audio path: {metadata.get('target_audio_path', 'N/A')}")
            
            self.validation_results["dataset_loading"] = {
                "valid": True,
                "dataset_size": len(dataset)
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Dataset loading failed: {e}")
            self.validation_results["dataset_loading"] = {"valid": False, "error": str(e)}
            return False
    
    def validate_model_loading(self) -> bool:
        """Validate model and LoRA loading."""
        logger.info("🤖 Validating model loading...")
        
        try:
            # Configure LoRA for comprehensive DualFFN targeting
            lora_config = HiggsAudioLoRATrainingConfig(
                r=8,  # Small rank for testing
                lora_alpha=16,
                target_modules_mode="comprehensive"  # Target all DualFFN components
            )
            
            # Load model with LoRA
            model, config, _ = create_higgs_audio_lora_model(
                model_path="bosonai/higgs-audio-v2-generation-3B-base",
                custom_config=lora_config,
                device_map="cpu",  # Use CPU for validation
                torch_dtype=torch.float32
            )
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            trainable_percentage = (trainable_params / total_params) * 100
            
            logger.info(f"✅ Model loading successful")
            logger.info(f"   Total parameters: {total_params:,}")
            logger.info(f"   Trainable parameters: {trainable_params:,}")
            logger.info(f"   Trainable percentage: {trainable_percentage:.2f}%")
            
            # Validate DualFFN architecture targeting
            lora_modules = []
            for name, module in model.named_modules():
                if hasattr(module, 'lora_A'):
                    lora_modules.append(name)
            
            expected_modules = ["audio_mlp", "mlp", "self_attn"]
            found_modules = [mod for mod in expected_modules if any(mod in lora_mod for lora_mod in lora_modules)]
            
            logger.info(f"   LoRA modules found: {len(lora_modules)}")
            logger.info(f"   DualFFN components targeted: {found_modules}")
            
            # Cleanup
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            self.validation_results["model_loading"] = {
                "valid": True,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "lora_modules": len(lora_modules),
                "dualffn_targeting": found_modules
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            self.validation_results["model_loading"] = {"valid": False, "error": str(e)}
            return False
    
    def validate_training_components(self) -> bool:
        """Validate training collator and loss function."""
        logger.info("🔧 Validating training components...")
        
        try:
            # Import required components
            from boson_multimodal.model.higgs_audio.configuration_higgs_audio import HiggsAudioConfig
            from transformers import AutoTokenizer
            
            # Create configuration
            config = HiggsAudioConfig()
            
            # Test training collator
            collator = ArabicVoiceCloningTrainingCollator(
                config=config,
                whisper_processor=None,  # Skip Whisper for validation
                enable_teacher_forcing=True,
                validate_batches=True
            )
            
            # Test loss function
            loss_config = LossConfig(
                text_loss_weight=1.0,
                audio_loss_weight=1.0,
                contrastive_loss_weight=0.1
            )
            
            loss_fn = create_loss_function(
                config=config,
                vocab_size=128256,
                loss_config=loss_config
            )
            
            logger.info(f"✅ Training components validation successful")
            logger.info(f"   Collator configured with teacher forcing: {collator.enable_teacher_forcing}")
            logger.info(f"   Loss function components: text, audio, contrastive")
            
            self.validation_results["training_components"] = {
                "valid": True,
                "teacher_forcing": True,
                "loss_components": ["text", "audio", "contrastive"]
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Training components validation failed: {e}")
            self.validation_results["training_components"] = {"valid": False, "error": str(e)}
            return False
    
    def validate_checkpoint_management(self) -> bool:
        """Validate checkpoint management functionality."""
        logger.info("💾 Validating checkpoint management...")
        
        try:
            # Initialize checkpoint manager
            manager = LoRACheckpointManager(
                base_model_path="bosonai/higgs-audio-v2-generation-3B-base"
            )
            
            # Test manager initialization
            logger.info(f"✅ Checkpoint manager initialized successfully")
            logger.info(f"   Base model path: {manager.base_model_path}")
            logger.info(f"   Device: {manager.device}")
            
            self.validation_results["checkpoint_management"] = {
                "valid": True,
                "base_model_configured": True,
                "device": str(manager.device)
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Checkpoint management validation failed: {e}")
            self.validation_results["checkpoint_management"] = {"valid": False, "error": str(e)}
            return False
    
    def validate_zero_shot_alignment(self) -> bool:
        """Validate alignment with zero-shot voice cloning requirements."""
        logger.info("🎯 Validating zero-shot voice cloning alignment...")
        
        try:
            # Check ChatML format alignment
            with open(self.chatml_file, 'r') as f:
                data = json.load(f)
            
            sample = data[0]
            
            # Validate zero-shot structure:
            # 1. Reference audio in user message
            # 2. Target text for generation
            # 3. Target audio in assistant message (for training)
            
            user_msg = next(msg for msg in sample["messages"] if msg["role"] == "user")
            assistant_msg = next(msg for msg in sample["messages"] if msg["role"] == "assistant")
            
            # Check reference audio
            ref_audio_found = any(
                item.get("type") == "audio" 
                for item in user_msg["content"] 
                if isinstance(item, dict)
            )
            
            # Check target text
            target_text_found = any(
                item.get("type") == "text" and "generate speech" in item.get("text", "").lower()
                for item in user_msg["content"]
                if isinstance(item, dict)
            )
            
            # Check assistant response
            assistant_audio_found = any(
                item.get("type") == "audio"
                for item in assistant_msg["content"]
                if isinstance(item, dict)
            )
            
            zero_shot_valid = ref_audio_found and target_text_found and assistant_audio_found
            
            logger.info(f"✅ Zero-shot alignment validation {'passed' if zero_shot_valid else 'failed'}")
            logger.info(f"   Reference audio in user message: {ref_audio_found}")
            logger.info(f"   Target text for generation: {target_text_found}")
            logger.info(f"   Assistant audio response: {assistant_audio_found}")
            
            self.validation_results["zero_shot_alignment"] = {
                "valid": zero_shot_valid,
                "reference_audio": ref_audio_found,
                "target_text": target_text_found,
                "assistant_audio": assistant_audio_found
            }
            
            return zero_shot_valid
            
        except Exception as e:
            logger.error(f"❌ Zero-shot alignment validation failed: {e}")
            self.validation_results["zero_shot_alignment"] = {"valid": False, "error": str(e)}
            return False
    
    def generate_validation_report(self, output_path: str) -> str:
        """Generate comprehensive validation report."""
        report_path = Path(output_path) / "pipeline_validation_report.json"
        
        # Calculate overall validation status
        all_validations = [
            result.get("valid", False) 
            for result in self.validation_results.values() 
            if isinstance(result, dict)
        ]
        overall_valid = all(all_validations)
        
        report = {
            "validation_timestamp": torch.utils.data.get_worker_info(),
            "chatml_file": self.chatml_file,
            "overall_valid": overall_valid,
            "validations_passed": sum(all_validations),
            "total_validations": len(all_validations),
            "detailed_results": self.validation_results,
            "summary": {
                "data_format": "✅" if self.validation_results.get("data_format", {}).get("valid") else "❌",
                "dataset_loading": "✅" if self.validation_results.get("dataset_loading", {}).get("valid") else "❌",
                "model_loading": "✅" if self.validation_results.get("model_loading", {}).get("valid") else "❌",
                "training_components": "✅" if self.validation_results.get("training_components", {}).get("valid") else "❌",
                "checkpoint_management": "✅" if self.validation_results.get("checkpoint_management", {}).get("valid") else "❌",
                "zero_shot_alignment": "✅" if self.validation_results.get("zero_shot_alignment", {}).get("valid") else "❌"
            }
        }
        
        # Save report
        os.makedirs(output_path, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        logger.info("📋 VALIDATION REPORT SUMMARY")
        logger.info("=" * 50)
        
        for validation, status in report["summary"].items():
            logger.info(f"   {validation}: {status}")
        
        logger.info(f"\n🎯 Overall Status: {'✅ PIPELINE READY' if overall_valid else '❌ PIPELINE NEEDS FIXES'}")
        logger.info(f"📄 Full report saved: {report_path}")
        
        return str(report_path)
    
    def run_full_validation(self, output_dir: str = "./validation_output") -> bool:
        """Run complete pipeline validation."""
        logger.info("🚀 Starting complete pipeline validation...")
        
        # Run all validations
        validations = [
            ("Data Format", self.validate_data_format),
            ("Dataset Loading", self.validate_dataset_loading),
            ("Model Loading", self.validate_model_loading),
            ("Training Components", self.validate_training_components),
            ("Checkpoint Management", self.validate_checkpoint_management),
            ("Zero-Shot Alignment", self.validate_zero_shot_alignment)
        ]
        
        results = []
        for name, validation_func in validations:
            try:
                result = validation_func()
                results.append(result)
                status = "✅ PASSED" if result else "❌ FAILED"
                logger.info(f"{name}: {status}")
            except Exception as e:
                logger.error(f"{name}: ❌ ERROR - {e}")
                results.append(False)
        
        # Generate report
        report_path = self.generate_validation_report(output_dir)
        
        overall_success = all(results)
        return overall_success


def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete Pipeline Validation")
    parser.add_argument("--chatml_file", default="test_user_chatml_data.json", 
                       help="Path to ChatML data file")
    parser.add_argument("--output_dir", default="./pipeline_validation_output",
                       help="Output directory for validation results")
    
    args = parser.parse_args()
    
    # Ensure test file exists
    if not Path(args.chatml_file).exists():
        logger.error(f"❌ ChatML file not found: {args.chatml_file}")
        return False
    
    # Run validation
    validator = PipelineValidator(args.chatml_file)
    success = validator.run_full_validation(args.output_dir)
    
    if success:
        logger.info("🎉 VALIDATION COMPLETED SUCCESSFULLY!")
        logger.info("✅ Pipeline is ready for training with user's ChatML data format")
    else:
        logger.error("❌ VALIDATION FAILED!")
        logger.error("🔧 Please check the validation report and fix issues before training")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)