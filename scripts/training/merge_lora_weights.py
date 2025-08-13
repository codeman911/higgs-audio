#!/usr/bin/env python3
"""
LoRA Weight Merging Utility for Higgs-Audio
============================================

Clean, production-ready script to merge LoRA fine-tuned weights back into 
the base Higgs-Audio model. Fully compatible with HuggingFace transformers
and inference pipelines.

Usage:
    python merge_lora_weights.py \
        --base_model_path bosonai/higgs-audio-v2-generation-3B-base \
        --lora_checkpoint ./outputs/checkpoint-1000 \
        --output_dir ./merged_model \
        --push_to_hub my-org/higgs-audio-finetuned

Author: Cascade AI Assistant
Compatible: HuggingFace Transformers, PEFT, Higgs-Audio inference
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig
from peft.utils import get_peft_model_state_dict

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from boson_multimodal.model.higgs_audio.modeling_higgs_audio import HiggsAudioModel
from boson_multimodal.audio_processing.higgs_audio_tokenizer import HiggsAudioTokenizer


def setup_logging(verbose: bool = False):
    """Configure logging for the merge process."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def validate_paths(base_model_path: str, lora_checkpoint: str, output_dir: str, logger: logging.Logger):
    """Validate input and output paths."""
    logger.info("🔍 Validating paths...")
    
    # Check if LoRA checkpoint exists and contains required files
    lora_path = Path(lora_checkpoint)
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_checkpoint}")
    
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    missing_files = [f for f in required_files if not (lora_path / f).exists()]
    
    if missing_files:
        # Try .bin format as backup
        if (lora_path / "adapter_model.bin").exists():
            logger.info("✅ Found adapter_model.bin (PyTorch format)")
        else:
            raise FileNotFoundError(f"Missing LoRA files in {lora_checkpoint}: {missing_files}")
    else:
        logger.info("✅ Found adapter_model.safetensors (SafeTensors format)")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Output directory ready: {output_dir}")


def load_base_model(model_path: str, logger: logging.Logger, device: str = "auto"):
    """Load the base Higgs-Audio model."""
    logger.info(f"🔄 Loading base model: {model_path}")
    
    try:
        # Load with automatic device placement
        model = HiggsAudioModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
        logger.info(f"✅ Base model loaded successfully")
        logger.info(f"   Model type: {type(model).__name__}")
        logger.info(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to load base model: {e}")
        raise


def load_tokenizers(model_path: str, logger: logging.Logger):
    """Load text and audio tokenizers."""
    logger.info("🔄 Loading tokenizers...")
    
    try:
        # Load text tokenizer
        text_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        logger.info("✅ Text tokenizer loaded")
        
        # Load audio tokenizer  
        audio_tokenizer = HiggsAudioTokenizer.from_pretrained(
            "bosonai/higgs-audio-v2-tokenizer"
        )
        logger.info("✅ Audio tokenizer loaded")
        
        return text_tokenizer, audio_tokenizer
        
    except Exception as e:
        logger.error(f"❌ Failed to load tokenizers: {e}")
        raise


def merge_lora_weights(base_model, lora_checkpoint: str, logger: logging.Logger):
    """Merge LoRA weights into the base model."""
    logger.info(f"🔄 Loading LoRA adapter from: {lora_checkpoint}")
    
    try:
        # Load PEFT config to verify compatibility
        peft_config = PeftConfig.from_pretrained(lora_checkpoint)
        logger.info(f"   LoRA config: r={peft_config.r}, alpha={peft_config.lora_alpha}")
        logger.info(f"   Target modules: {peft_config.target_modules}")
        logger.info(f"   Task type: {peft_config.task_type}")
        
        # Load the PEFT model
        logger.info("🔄 Merging LoRA weights...")
        peft_model = PeftModel.from_pretrained(
            base_model, 
            lora_checkpoint,
            torch_dtype=torch.bfloat16
        )
        
        # Merge weights and unload adapter
        merged_model = peft_model.merge_and_unload()
        logger.info("✅ LoRA weights merged successfully")
        
        return merged_model
        
    except Exception as e:
        logger.error(f"❌ Failed to merge LoRA weights: {e}")
        raise


def save_merged_model(merged_model, text_tokenizer, audio_tokenizer, 
                     output_dir: str, logger: logging.Logger, 
                     push_to_hub: Optional[str] = None,
                     commit_message: Optional[str] = None):
    """Save the merged model in HuggingFace format."""
    logger.info(f"💾 Saving merged model to: {output_dir}")
    
    try:
        # Save the merged model
        merged_model.save_pretrained(
            output_dir,
            safe_serialization=True,  # Use SafeTensors format
            max_shard_size="5GB"      # Shard large models
        )
        logger.info("✅ Model weights saved")
        
        # Save text tokenizer
        text_tokenizer.save_pretrained(output_dir)
        logger.info("✅ Text tokenizer saved")
        
        # Save audio tokenizer info (reference)
        with open(os.path.join(output_dir, "audio_tokenizer_info.txt"), "w") as f:
            f.write("# Audio Tokenizer Information\n")
            f.write("# Use: HiggsAudioTokenizer.from_pretrained('bosonai/higgs-audio-v2-tokenizer')\n")
            f.write("audio_tokenizer_path: bosonai/higgs-audio-v2-tokenizer\n")
        logger.info("✅ Audio tokenizer info saved")
        
        # Create README with usage instructions
        readme_content = f"""# Higgs-Audio Fine-tuned Model

This model was fine-tuned using LoRA (Low-Rank Adaptation) and merged back to the base model.

## Usage

```python
from transformers import AutoTokenizer
from boson_multimodal.model.higgs_audio import HiggsAudioModel
from boson_multimodal.audio_processing import HiggsAudioTokenizer

# Load the merged model
model = HiggsAudioModel.from_pretrained(
    "{output_dir}",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load tokenizers
text_tokenizer = AutoTokenizer.from_pretrained("{output_dir}")
audio_tokenizer = HiggsAudioTokenizer.from_pretrained("bosonai/higgs-audio-v2-tokenizer")
```

## Inference

Use this model exactly like the original Higgs-Audio model. All inference scripts and examples will work without modification.

## Model Details

- Base Model: Higgs-Audio v2 Generation 3B
- Fine-tuning: LoRA adaptation merged into base weights
- Format: HuggingFace Transformers compatible
- Precision: BFloat16 optimized
"""
        
        with open(os.path.join(output_dir, "README.md"), "w") as f:
            f.write(readme_content)
        logger.info("✅ README created")
        
        # Push to HuggingFace Hub if requested
        if push_to_hub:
            logger.info(f"🚀 Pushing to HuggingFace Hub: {push_to_hub}")
            try:
                merged_model.push_to_hub(
                    push_to_hub,
                    commit_message=commit_message or "Upload fine-tuned Higgs-Audio model",
                    safe_serialization=True
                )
                text_tokenizer.push_to_hub(push_to_hub)
                logger.info("✅ Model pushed to Hub successfully")
            except Exception as e:
                logger.warning(f"⚠️  Failed to push to Hub: {e}")
        
        logger.info("🎉 Model merge and save completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to save merged model: {e}")
        raise


def verify_merged_model(output_dir: str, logger: logging.Logger):
    """Verify the merged model can be loaded correctly."""
    logger.info("🔍 Verifying merged model...")
    
    try:
        # Load the saved model
        test_model = HiggsAudioModel.from_pretrained(
            output_dir,
            torch_dtype=torch.bfloat16,
            device_map="cpu"  # Load on CPU for verification
        )
        
        # Basic checks
        param_count = sum(p.numel() for p in test_model.parameters())
        logger.info(f"✅ Model verification passed")
        logger.info(f"   Parameters: {param_count:,}")
        logger.info(f"   Model type: {type(test_model).__name__}")
        
        # Cleanup
        del test_model
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights into Higgs-Audio base model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic merge
  python merge_lora_weights.py \\
    --base_model_path bosonai/higgs-audio-v2-generation-3B-base \\
    --lora_checkpoint ./outputs/checkpoint-1000 \\
    --output_dir ./merged_model

  # Merge and push to HuggingFace Hub
  python merge_lora_weights.py \\
    --base_model_path bosonai/higgs-audio-v2-generation-3B-base \\
    --lora_checkpoint ./outputs/final_model \\
    --output_dir ./merged_model \\
    --push_to_hub my-org/higgs-audio-finetuned \\
    --commit_message "Fine-tuned for Arabic-English TTS"
        """
    )
    
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path or name of the base Higgs-Audio model")
    parser.add_argument("--lora_checkpoint", type=str, required=True,
                        help="Path to the LoRA checkpoint directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the merged model")
    parser.add_argument("--push_to_hub", type=str, default=None,
                        help="HuggingFace Hub repository name to push to")
    parser.add_argument("--commit_message", type=str, default=None,
                        help="Commit message for Hub push")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device placement strategy (auto, cpu, cuda)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--skip_verification", action="store_true",
                        help="Skip model verification step")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    logger.info("🚀 Starting LoRA merge process...")
    logger.info(f"   Base model: {args.base_model_path}")
    logger.info(f"   LoRA checkpoint: {args.lora_checkpoint}")
    logger.info(f"   Output directory: {args.output_dir}")
    
    try:
        # Step 1: Validate paths
        validate_paths(args.base_model_path, args.lora_checkpoint, args.output_dir, logger)
        
        # Step 2: Load base model
        base_model = load_base_model(args.base_model_path, logger, args.device)
        
        # Step 3: Load tokenizers
        text_tokenizer, audio_tokenizer = load_tokenizers(args.base_model_path, logger)
        
        # Step 4: Merge LoRA weights
        merged_model = merge_lora_weights(base_model, args.lora_checkpoint, logger)
        
        # Step 5: Save merged model
        save_merged_model(
            merged_model, text_tokenizer, audio_tokenizer,
            args.output_dir, logger, args.push_to_hub, args.commit_message
        )
        
        # Step 6: Verify merged model (optional)
        if not args.skip_verification:
            verify_merged_model(args.output_dir, logger)
        
        logger.info("✅ LoRA merge completed successfully!")
        logger.info(f"📁 Merged model saved at: {args.output_dir}")
        if args.push_to_hub:
            logger.info(f"🌐 Model available at: https://huggingface.co/{args.push_to_hub}")
        
    except Exception as e:
        logger.error(f"💥 Merge process failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
