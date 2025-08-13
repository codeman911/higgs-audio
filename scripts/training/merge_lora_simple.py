#!/usr/bin/env python3
"""
Simple LoRA merge script for HiggsAudio using native classes.
Merges LoRA adapters into base model and saves for inference.
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from boson_multimodal.model.higgs_audio.modeling_higgs_audio import HiggsAudioModel
from peft import PeftModel


def merge_lora_weights(base_model_path: str, lora_checkpoint: str, output_dir: str):
    """
    Merge LoRA weights into base model and save.
    
    Args:
        base_model_path: Path to base HiggsAudio model
        lora_checkpoint: Path to LoRA checkpoint directory
        output_dir: Output directory for merged model
    """
    print(f"🚀 Merging LoRA weights...")
    print(f"   Base model: {base_model_path}")
    print(f"   LoRA checkpoint: {lora_checkpoint}")
    print(f"   Output: {output_dir}")
    
    # Load base model with native HiggsAudio class
    print("🔄 Loading base model...")
    base_model = HiggsAudioModel.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu"  # Keep on CPU for merge
    )
    print(f"✅ Base model loaded: {type(base_model).__name__}")
    
    # Load LoRA model
    print("🔄 Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_checkpoint)
    print("✅ LoRA adapter loaded")
    
    # Merge and unload LoRA weights
    print("🔄 Merging weights...")
    merged_model = model.merge_and_unload()
    print("✅ Weights merged")
    
    # Save merged model
    print(f"💾 Saving to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir)
    print("✅ Model saved")
    
    # Copy tokenizer files from base model (CRITICAL: needed for inference)
    print("🔄 Copying tokenizer files from base model...")
    try:
        from transformers import AutoTokenizer
        
        # Load tokenizer from base model
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        
        # Save tokenizer to merged model directory
        tokenizer.save_pretrained(output_dir)
        print("✅ Tokenizer files copied")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not copy tokenizer files: {e}")
        print("   You may need to manually copy tokenizer files for inference")
    
    # Verify model can be loaded
    print("🔍 Verifying merged model...")
    try:
        test_model = HiggsAudioModel.from_pretrained(output_dir, torch_dtype=torch.bfloat16, device_map="cpu")
        print(f"✅ Verification successful: {type(test_model).__name__}")
        print(f"   Parameters: {sum(p.numel() for p in test_model.parameters()):,}")
        del test_model  # Free memory
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        raise
    
    print("🎉 LoRA merge completed successfully!")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into HiggsAudio base model")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--lora_checkpoint", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.lora_checkpoint):
        raise FileNotFoundError(f"LoRA checkpoint not found: {args.lora_checkpoint}")
    
    adapter_config = os.path.join(args.lora_checkpoint, "adapter_config.json")
    if not os.path.exists(adapter_config):
        raise FileNotFoundError(f"LoRA adapter config not found: {adapter_config}")
    
    # Merge weights
    output_path = merge_lora_weights(args.base_model_path, args.lora_checkpoint, args.output_dir)
    
    print(f"""
📋 USAGE INSTRUCTIONS:
To use the merged model for zero-shot voice cloning:

```python
from boson_multimodal.model.higgs_audio.modeling_higgs_audio import HiggsAudioModel

model = HiggsAudioModel.from_pretrained(
    "{output_path}",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

The merged model is ready for inference and zero-shot voice cloning!
""")


if __name__ == "__main__":
    main()
