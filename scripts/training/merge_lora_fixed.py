#!/usr/bin/env python3
"""
FIXED LoRA merge script that handles module path mismatches.
Addresses the PEFT warning about missing adapter keys.
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from boson_multimodal.model.higgs_audio.modeling_higgs_audio import HiggsAudioModel
from peft import PeftModel


def fix_adapter_keys(checkpoint_path: str):
    """
    Fix module path mismatches in adapter checkpoint.
    Handles double .model.model -> single .model and missing .default
    """
    print("🔧 Fixing adapter key names...")
    
    adapter_file = os.path.join(checkpoint_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        adapter_file = os.path.join(checkpoint_path, "adapter_model.bin")
    
    if not os.path.exists(adapter_file):
        raise FileNotFoundError(f"No adapter_model file found in {checkpoint_path}")
    
    # Load original adapter weights
    if adapter_file.endswith(".safetensors"):
        from safetensors import safe_open
        original_state = {}
        with safe_open(adapter_file, framework="pt") as f:
            for key in f.keys():
                original_state[key] = f.get_tensor(key)
    else:
        original_state = torch.load(adapter_file, map_location="cpu")
    
    print(f"📊 Original adapter has {len(original_state)} keys")
    
    # Remap keys to fix path mismatches
    remapped_state = {}
    for old_key, tensor in original_state.items():
        # Fix double model.model -> single model
        new_key = old_key.replace("base_model.model.model.", "base_model.model.")
        
        # Add .default if missing (PEFT expects this)
        if ".lora_A.weight" in new_key:
            new_key = new_key.replace(".lora_A.weight", ".lora_A.default.weight")
        elif ".lora_B.weight" in new_key:
            new_key = new_key.replace(".lora_B.weight", ".lora_B.default.weight")
        
        remapped_state[new_key] = tensor
        if old_key != new_key:
            print(f"  📝 {old_key[:60]}... -> {new_key[:60]}...")
    
    # Save the fixed adapter
    backup_file = adapter_file + ".backup"
    os.rename(adapter_file, backup_file)
    print(f"💾 Backed up original to: {backup_file}")
    
    if adapter_file.endswith(".safetensors"):
        from safetensors.torch import save_file
        save_file(remapped_state, adapter_file)
    else:
        torch.save(remapped_state, adapter_file)
    
    print(f"✅ Fixed adapter saved with {len(remapped_state)} keys")
    return len(original_state), len(remapped_state)


def verify_adapter_match(peft_model, checkpoint_path: str):
    """
    Verify how many adapter keys actually matched during loading.
    """
    print("🔍 Verifying adapter key matching...")
    
    # What does PEFT expect?
    expected_keys = [name for name, param in peft_model.named_parameters() 
                     if "lora_A" in name or "lora_B" in name]
    
    # What's in the checkpoint?
    adapter_file = os.path.join(checkpoint_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        adapter_file = os.path.join(checkpoint_path, "adapter_model.bin")
    
    if adapter_file.endswith(".safetensors"):
        from safetensors import safe_open
        with safe_open(adapter_file, framework="pt") as f:
            checkpoint_keys = list(f.keys())
    else:
        checkpoint_state = torch.load(adapter_file, map_location="cpu")
        checkpoint_keys = list(checkpoint_state.keys())
    
    # Calculate match rate
    expected_set = set(expected_keys)
    checkpoint_set = set(checkpoint_keys)
    
    matched = len(expected_set & checkpoint_set)
    total_expected = len(expected_set)
    match_rate = (matched / max(1, total_expected)) * 100
    
    print(f"📊 Adapter Loading Results:")
    print(f"   Expected keys: {total_expected}")
    print(f"   Checkpoint keys: {len(checkpoint_keys)}")
    print(f"   Matched: {matched}")
    print(f"   Match rate: {match_rate:.1f}%")
    
    if match_rate < 90:
        print(f"⚠️  WARNING: Low match rate indicates merge problems!")
        print(f"   Missing: {len(expected_set - checkpoint_set)}")
        print(f"   Extra: {len(checkpoint_set - expected_set)}")
        return False
    else:
        print(f"✅ Good match rate - merge should be successful")
        return True


def merge_lora_fixed(base_model_path: str, lora_checkpoint: str, output_dir: str):
    """
    Merge LoRA weights with proper key matching verification.
    """
    print(f"🚀 Fixed LoRA merge starting...")
    print(f"   Base model: {base_model_path}")
    print(f"   LoRA checkpoint: {lora_checkpoint}")
    print(f"   Output: {output_dir}")
    
    # Step 1: Fix adapter keys if needed
    try:
        orig_count, fixed_count = fix_adapter_keys(lora_checkpoint)
    except Exception as e:
        print(f"⚠️  Key fixing failed: {e}")
        print("   Proceeding with original keys...")
    
    # Step 2: Load base model
    print("🔄 Loading base model...")
    base_model = HiggsAudioModel.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )
    print(f"✅ Base model loaded: {type(base_model).__name__}")
    
    # Step 3: Load LoRA adapter with fixed keys
    print("🔄 Loading LoRA adapter...")
    try:
        peft_model = PeftModel.from_pretrained(base_model, lora_checkpoint)
        print("✅ LoRA adapter loaded")
        
        # Step 4: Verify the match rate
        match_success = verify_adapter_match(peft_model, lora_checkpoint)
        if not match_success:
            print("❌ Poor adapter matching - merge quality will be compromised!")
        
    except Exception as e:
        print(f"❌ LoRA loading failed: {e}")
        raise
    
    # Step 5: Merge and save
    print("🔄 Merging weights...")
    merged_model = peft_model.merge_and_unload()
    print("✅ Weights merged")
    
    print(f"💾 Saving to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir)
    
    # Copy tokenizer files
    print("🔄 Copying tokenizer files...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)
        print("✅ Tokenizer files copied")
    except Exception as e:
        print(f"⚠️  Tokenizer copy failed: {e}")
    
    # Verify final model
    print("🔍 Verifying merged model...")
    test_model = HiggsAudioModel.from_pretrained(output_dir, torch_dtype=torch.bfloat16, device_map="cpu")
    print(f"✅ Verification successful: {type(test_model).__name__}")
    print(f"   Parameters: {sum(p.numel() for p in test_model.parameters()):,}")
    
    print("🎉 Fixed LoRA merge completed successfully!")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Fixed LoRA merge with key matching verification")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--lora_checkpoint", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    
    args = parser.parse_args()
    
    merge_lora_fixed(args.base_model_path, args.lora_checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
