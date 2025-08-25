#!/usr/bin/env python3
"""
Test script to check which version of HiggsAudioModel is being imported
"""

import sys
from pathlib import Path

# Force correct model import path
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.insert(0, str(project_root))

print("Python path:")
for i, path in enumerate(sys.path):
    print(f"  {i}: {path}")

print("\nTrying to import HiggsAudioModel...")

try:
    # Import from CORRECT boson_multimodal path
    from boson_multimodal.model.higgs_audio.modeling_higgs_audio import HiggsAudioModel
    print(f"✅ Successfully imported HiggsAudioModel from: {HiggsAudioModel.__module__}")
    print(f"✅ File location: {HiggsAudioModel.__module__}")
    
    # Try to get the file path
    import inspect
    file_path = inspect.getfile(HiggsAudioModel)
    print(f"✅ File path: {file_path}")
    
except Exception as e:
    print(f"❌ Error importing HiggsAudioModel: {e}")
    
print("\nChecking for conflicting imports...")

# Check if there's a local boson_multimodal
if (current_dir / "boson_multimodal").exists():
    print("⚠️  Local boson_multimodal directory found - this might cause import conflicts")
    
if (current_dir / "train-higgs-audio" / "boson_multimodal").exists():
    print("⚠️  train-higgs-audio/boson_multimodal directory found - this might cause import conflicts")