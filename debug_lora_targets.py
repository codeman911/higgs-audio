#!/usr/bin/env python3
"""
Emergency diagnostic script to verify LoRA target modules are correctly identified
"""

import sys
import os
sys.path.insert(0, '/Users/vikram.solanki/Projects/tts/higgs-audio')

import torch
from transformers import AutoConfig
from boson_multimodal.model.higgs_audio import HiggsAudioModel
from peft import LoraConfig, get_peft_model, TaskType

def main():
    print("🔍 EMERGENCY LORA TARGET DIAGNOSTIC")
    print("=" * 50)
    
    # Load model config
    try:
        model_config = AutoConfig.from_pretrained("bosonai/higgs-audio-v2")
        print(f"✅ Model config loaded")
    except Exception as e:
        print(f"❌ Failed to load model config: {e}")
        return
    
    # Load model
    try:
        model = HiggsAudioModel.from_pretrained(
            "bosonai/higgs-audio-v2",
            torch_dtype=torch.bfloat16,
            device_map="cpu"
        )
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Define our current LoRA targets
    target_modules = [
        # Audio output projection layers - CRITICAL for audio generation
        "audio_decoder_proj.audio_lm_head",
        
        # Audio embedding layers - CRITICAL for audio token processing
        "audio_out_embed_projector",
        "audio_encoder_proj",
        
        # Extended audio MLP layers for better coverage (layers 8-15)
        "layers.8.audio_mlp.gate_proj",
        "layers.8.audio_mlp.up_proj", 
        "layers.8.audio_mlp.down_proj",
        "layers.9.audio_mlp.gate_proj",
        "layers.9.audio_mlp.up_proj",
        "layers.9.audio_mlp.down_proj",
        "layers.10.audio_mlp.gate_proj",
        "layers.10.audio_mlp.up_proj",
        "layers.10.audio_mlp.down_proj",
        "layers.11.audio_mlp.gate_proj",
        "layers.11.audio_mlp.up_proj",
        "layers.11.audio_mlp.down_proj",
        "layers.12.audio_mlp.gate_proj",
        "layers.12.audio_mlp.up_proj",
        "layers.12.audio_mlp.down_proj",
        "layers.13.audio_mlp.gate_proj",
        "layers.13.audio_mlp.up_proj",
        "layers.13.audio_mlp.down_proj",
        "layers.14.audio_mlp.gate_proj",
        "layers.14.audio_mlp.up_proj",
        "layers.14.audio_mlp.down_proj",
        "layers.15.audio_mlp.gate_proj",
        "layers.15.audio_mlp.up_proj",
        "layers.15.audio_mlp.down_proj",
        
        # Audio attention layers for better conditioning
        "layers.10.audio_attn.q_proj",
        "layers.10.audio_attn.k_proj", 
        "layers.10.audio_attn.v_proj",
        "layers.10.audio_attn.o_proj",
        "layers.12.audio_attn.q_proj",
        "layers.12.audio_attn.k_proj",
        "layers.12.audio_attn.v_proj", 
        "layers.12.audio_attn.o_proj",
        "layers.14.audio_attn.q_proj",
        "layers.14.audio_attn.k_proj",
        "layers.14.audio_attn.v_proj",
        "layers.14.audio_attn.o_proj",
    ]
    
    print(f"\n🎯 CHECKING TARGET MODULE EXISTENCE:")
    print("-" * 50)
    
    # Get all model module names
    all_modules = set()
    for name, module in model.named_modules():
        all_modules.add(name)
    
    # Check each target
    found_targets = []
    missing_targets = []
    
    for target in target_modules:
        if target in all_modules:
            found_targets.append(target)
            print(f"✅ FOUND: {target}")
        else:
            missing_targets.append(target)
            print(f"❌ MISSING: {target}")
    
    print(f"\n📊 SUMMARY:")
    print(f"Found targets: {len(found_targets)}/{len(target_modules)}")
    print(f"Missing targets: {len(missing_targets)}")
    
    if missing_targets:
        print(f"\n🚨 CRITICAL MISSING TARGETS:")
        for target in missing_targets:
            print(f"  - {target}")
    
    # Focus on critical audio generation module
    print(f"\n🎯 CRITICAL AUDIO GENERATION PATH:")
    audio_lm_head_found = "audio_decoder_proj.audio_lm_head" in all_modules
    print(f"audio_decoder_proj.audio_lm_head: {'✅ FOUND' if audio_lm_head_found else '❌ MISSING'}")
    
    if not audio_lm_head_found:
        print(f"\n🔍 SEARCHING FOR AUDIO_LM_HEAD ALTERNATIVES:")
        for name in all_modules:
            if "audio" in name.lower() and "lm_head" in name.lower():
                print(f"  - {name}")
    
    # Show actual audio-related modules
    print(f"\n🔍 ALL AUDIO-RELATED MODULES IN MODEL:")
    audio_modules = []
    for name in sorted(all_modules):
        if "audio" in name.lower():
            audio_modules.append(name)
            if any(x in name for x in ["lm_head", "proj", "embed", "mlp", "attn"]):
                print(f"  ⭐ {name}")
            else:
                print(f"     {name}")
    
    print(f"\n📈 TOTAL AUDIO MODULES: {len(audio_modules)}")
    
    # Test LoRA config creation
    print(f"\n🧪 TESTING LORA CONFIG CREATION:")
    try:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=found_targets if found_targets else ["audio_decoder_proj.audio_lm_head"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()
        print(f"✅ LoRA config creation successful")
    except Exception as e:
        print(f"❌ LoRA config creation failed: {e}")

if __name__ == "__main__":
    main()
