#!/usr/bin/env python3
"""
Test script to check which version of HiggsAudioModel is being loaded
"""

import sys
from pathlib import Path

# Force correct model import path
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.insert(0, str(project_root))

# Import from CORRECT boson_multimodal path
from boson_multimodal.model.higgs_audio.modeling_higgs_audio import HiggsAudioModel
import inspect

def check_model_signature():
    """Check the model signature to see if it has 'labels' or the correct parameters."""
    print("🔍 Checking HiggsAudioModel signature...")
    
    # Get the forward method signature
    sig = inspect.signature(HiggsAudioModel.forward)
    params = list(sig.parameters.keys())
    
    print(f"📊 HiggsAudioModel.forward() parameters ({len(params)} total):")
    for i, param in enumerate(params, 1):
        print(f"  {i:2d}. {param}")
    
    # Check for problematic 'labels' parameter
    if 'labels' in params:
        print("\n❌ CRITICAL ERROR: HiggsAudioModel has 'labels' parameter!")
        print("❌ You're using the WRONG model version!")
        return False
    else:
        print("\n✅ CORRECT: HiggsAudioModel does NOT have 'labels' parameter")
        
    # Check for required parameters
    required_params = ['label_ids', 'label_audio_ids', 'audio_out_ids', 'audio_features']
    missing_params = [p for p in required_params if p not in params]
    
    if missing_params:
        print(f"❌ Missing required parameters: {missing_params}")
        return False
    else:
        print("✅ All required parameters present in model forward signature")
        return True

if __name__ == "__main__":
    try:
        success = check_model_signature()
        if success:
            print("\n🎉 Model signature check PASSED")
        else:
            print("\n💥 Model signature check FAILED")
            sys.exit(1)
    except Exception as e:
        print(f"💥 Error checking model signature: {e}")
        sys.exit(1)