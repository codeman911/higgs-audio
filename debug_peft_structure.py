#!/usr/bin/env python3
"""
Debug script to analyze PEFT model structure and find the root cause
"""

import torch
from boson_multimodal.model.higgs_audio import HiggsAudioModel, HiggsAudioConfig
from lora import apply_lora, create_lora_config

def analyze_model_structure(model, name="model"):
    """Recursively analyze model structure"""
    print(f"\n{'='*50}")
    print(f"Analyzing: {name}")
    print(f"Type: {type(model).__name__}")
    print(f"Has __dict__: {hasattr(model, '__dict__')}")
    
    # Check common wrapper attributes
    attrs_to_check = ['module', 'base_model', 'model', 'peft_model']
    for attr in attrs_to_check:
        if hasattr(model, attr):
            inner_model = getattr(model, attr)
            print(f"  Has {attr}: {type(inner_model).__name__}")
            
    # Check if it has forward method signature
    if hasattr(model, 'forward'):
        import inspect
        try:
            sig = inspect.signature(model.forward)
            params = list(sig.parameters.keys())
            has_labels = 'labels' in params
            has_label_ids = 'label_ids' in params
            has_label_audio_ids = 'label_audio_ids' in params
            print(f"  Forward params: {len(params)} total")
            print(f"  Has 'labels': {has_labels}")
            print(f"  Has 'label_ids': {has_label_ids}")
            print(f"  Has 'label_audio_ids': {has_label_audio_ids}")
        except Exception as e:
            print(f"  Error getting signature: {e}")

def unwrap_model_step_by_step(model):
    """Step by step unwrapping to see what happens"""
    print(f"\n{'='*60}")
    print("STEP BY STEP MODEL UNWRAPPING")
    print(f"{'='*60}")
    
    current_model = model
    step = 0
    
    while step < 10:  # Prevent infinite loop
        step += 1
        print(f"\nStep {step}:")
        print(f"  Current type: {type(current_model).__name__}")
        
        # Check for wrappers
        if hasattr(current_model, 'module'):
            print("  Found 'module' attribute (likely DDP)")
            current_model = current_model.module
            continue
            
        if hasattr(current_model, 'base_model'):
            print("  Found 'base_model' attribute (likely PEFT)")
            current_model = current_model.base_model
            continue
            
        if hasattr(current_model, 'model'):
            print("  Found 'model' attribute")
            current_model = current_model.model
            continue
            
        # No more wrappers found
        print(f"  No more wrappers found!")
        break
    
    print(f"\nFinal unwrapped model type: {type(current_model).__name__}")
    return current_model

def test_forward_call(model, name="model"):
    """Test forward call with different input combinations"""
    print(f"\n{'='*50}")
    print(f"Testing forward call for: {name}")
    
    # Create dummy inputs
    batch_size = 1
    seq_len = 10
    
    dummy_inputs = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
        'attention_mask': torch.ones(batch_size, seq_len),
    }
    
    # Test with clean inputs (no labels)
    try:
        print("Testing with clean inputs (no labels)...")
        outputs = model(**dummy_inputs)
        print(f"✅ SUCCESS: Forward pass worked")
        print(f"  Output type: {type(outputs)}")
        if hasattr(outputs, 'logits'):
            print(f"  Has logits: {outputs.logits.shape if outputs.logits is not None else None}")
        if hasattr(outputs, 'audio_logits'):
            print(f"  Has audio_logits: {outputs.audio_logits.shape if outputs.audio_logits is not None else None}")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        
    # Test with labels parameter
    dummy_inputs_with_labels = dummy_inputs.copy()
    dummy_inputs_with_labels['labels'] = torch.randint(0, 1000, (batch_size, seq_len))
    
    try:
        print("Testing with 'labels' parameter...")
        outputs = model(**dummy_inputs_with_labels)
        print(f"✅ SUCCESS: Forward pass with labels worked")
    except Exception as e:
        print(f"❌ FAILED with labels: {e}")

def main():
    print("PEFT Structure Debugging Script")
    print("="*60)
    
    # Load base model
    print("\n1. Loading base model...")
    config = HiggsAudioConfig.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")
    base_model = HiggsAudioModel.from_pretrained(
        "bosonai/higgs-audio-v2-generation-3B-base",
        config=config,
        torch_dtype=torch.bfloat16
    )
    
    analyze_model_structure(base_model, "Base HiggsAudioModel")
    test_forward_call(base_model, "Base HiggsAudioModel")
    
    # Apply LoRA
    print("\n2. Applying LoRA...")
    lora_config = create_lora_config(r=16, lora_alpha=32, lora_dropout=0.05)
    peft_model = apply_lora(base_model, lora_config)
    
    analyze_model_structure(peft_model, "PEFT Model")
    
    # Step by step unwrapping
    final_model = unwrap_model_step_by_step(peft_model)
    analyze_model_structure(final_model, "Final Unwrapped Model")
    test_forward_call(final_model, "Final Unwrapped Model")
    
    # Test the PEFT model directly
    test_forward_call(peft_model, "PEFT Model")

if __name__ == "__main__":
    main()