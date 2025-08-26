#!/usr/bin/env python3
"""
Test script to validate PEFT compatibility fix
"""

import torch
import logging
from boson_multimodal.model.higgs_audio import HiggsAudioModel, HiggsAudioConfig
from lora import apply_lora, create_lora_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_peft_bypass():
    """Test the PEFT bypass functionality"""
    
    print("🧪 Testing PEFT Compatibility Fix")
    print("=" * 50)
    
    # Load base model
    print("1. Loading base model...")
    config = HiggsAudioConfig.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")
    base_model = HiggsAudioModel.from_pretrained(
        "bosonai/higgs-audio-v2-generation-3B-base",
        config=config,
        torch_dtype=torch.bfloat16
    )
    
    print("2. Applying LoRA...")
    lora_config = create_lora_config(r=16, lora_alpha=32, lora_dropout=0.05)
    peft_model = apply_lora(base_model, lora_config)
    
    print(f"PEFT model type: {type(peft_model).__name__}")
    
    # Test the unwrapping logic
    print("\n3. Testing model unwrapping...")
    
    def get_base_higgs_model(model):
        """Extract the actual HiggsAudioModel from all wrapper layers."""
        path = []
        
        # Iteratively unwrap until we find HiggsAudioModel
        max_depth = 20
        for depth in range(max_depth):
            model_type = type(model).__name__
            path.append(model_type)
            
            # Found the target model
            if model_type == 'HiggsAudioModel':
                print(f"Found HiggsAudioModel at depth {depth}: {' -> '.join(path)}")
                return model
            
            # Try different unwrapping attributes
            if hasattr(model, 'module'):  # DDP wrapper
                model = model.module
                continue
            elif hasattr(model, 'base_model'):  # PEFT wrapper
                model = model.base_model
                continue
            elif hasattr(model, 'model'):  # Generic wrapper
                model = model.model
                continue
            else:
                # No more wrappers found
                break
        
        print(f"Could not find HiggsAudioModel, using: {type(model).__name__}")
        print(f"Unwrapping path: {' -> '.join(path)}")
        return model
    
    unwrapped_model = get_base_higgs_model(peft_model)
    
    # Test forward call with clean inputs
    print("\n4. Testing forward call...")
    
    batch_size = 1
    seq_len = 10
    
    # Create clean inputs (no labels)
    clean_inputs = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len))
    }
    
    try:
        print("Testing direct forward call on unwrapped model...")
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            outputs = unwrapped_model.forward(**clean_inputs)
        print("✅ SUCCESS: Direct forward call worked!")
        print(f"Output type: {type(outputs)}")
        
        # Check outputs
        if hasattr(outputs, 'logits'):
            print(f"Text logits shape: {outputs.logits.shape}")
        if hasattr(outputs, 'audio_logits'):
            print(f"Audio logits shape: {outputs.audio_logits.shape}")
            
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False
    
    # Test with labels (should fail on PEFT model but work on unwrapped)
    print("\n5. Testing label injection protection...")
    
    inputs_with_labels = clean_inputs.copy()
    inputs_with_labels['labels'] = torch.randint(0, 1000, (batch_size, seq_len))
    
    try:
        print("Testing PEFT model with labels (should fail)...")
        outputs = peft_model(**inputs_with_labels)
        print("⚠️  WARNING: PEFT model accepted labels - this might be the problem!")
    except Exception as e:
        print(f"✅ EXPECTED: PEFT model rejected labels: {type(e).__name__}")
    
    try:
        print("Testing unwrapped model with labels...")
        outputs = unwrapped_model.forward(**inputs_with_labels)
        print("✅ SUCCESS: Unwrapped model handled labels correctly!")
    except Exception as e:
        print(f"ℹ️  Note: Unwrapped model also rejected labels: {type(e).__name__}")
    
    print("\n🎉 PEFT Bypass Test Completed!")
    print("The unwrapping logic should work in training.")
    return True

if __name__ == "__main__":
    test_peft_bypass()