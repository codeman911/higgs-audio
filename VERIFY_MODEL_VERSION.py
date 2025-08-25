#!/usr/bin/env python3
"""
Comprehensive test script to verify the correct HiggsAudioModel version is being imported
"""

import sys
from pathlib import Path
import inspect

# Force correct model import path
current_dir = Path(__file__).parent.resolve()
project_root = current_dir

# Remove any existing boson_multimodal paths from sys.path to avoid conflicts
sys_path_cleaned = []
for path in sys.path:
    path_obj = Path(path).resolve()
    # Remove paths that contain train-higgs-audio to avoid wrong model imports
    if "train-higgs-audio" not in str(path_obj):
        sys_path_cleaned.append(path)
sys.path = sys_path_cleaned

# Insert our project root at the beginning to ensure correct imports
sys.path.insert(0, str(project_root))

print("🔍 Python path after cleaning:")
for i, path in enumerate(sys.path[:10]):  # Show first 10 paths
    print(f"  {i}: {path}")
if len(sys.path) > 10:
    print(f"  ... and {len(sys.path) - 10} more")

print("\n🔍 Attempting to import HiggsAudioModel...")

try:
    # Import from CORRECT boson_multimodal path
    from boson_multimodal.model.higgs_audio.modeling_higgs_audio import HiggsAudioModel
    print(f"✅ Successfully imported HiggsAudioModel")
    print(f"✅ Module: {HiggsAudioModel.__module__}")
    
    # Get the file path
    file_path = inspect.getfile(HiggsAudioModel)
    print(f"✅ File path: {file_path}")
    
    # Check if it's the correct version (should NOT have 'labels' parameter)
    sig = inspect.signature(HiggsAudioModel.forward)
    params = list(sig.parameters.keys())
    
    print(f"\n📊 HiggsAudioModel.forward() parameters ({len(params)} total):")
    for i, param in enumerate(params[:15], 1):  # Show first 15
        print(f"  {i:2d}. {param}")
    if len(params) > 15:
        print(f"  ... and {len(params) - 15} more")
    
    # Critical checks
    has_labels = 'labels' in params
    has_label_ids = 'label_ids' in params
    has_label_audio_ids = 'label_audio_ids' in params
    
    print(f"\n🔍 Validation Results:")
    if has_labels:
        print("❌ CRITICAL ERROR: Model has 'labels' parameter!")
        print("❌ This is the WRONG version - it will cause 'unexpected keyword argument labels' error")
        print("❌ Expected: NO 'labels' parameter")
        sys.exit(1)
    else:
        print("✅ CORRECT: Model does NOT have 'labels' parameter")
    
    if has_label_ids and has_label_audio_ids:
        print("✅ CORRECT: Model has required parameters (label_ids, label_audio_ids)")
    else:
        missing = []
        if not has_label_ids:
            missing.append('label_ids')
        if not has_label_audio_ids:
            missing.append('label_audio_ids')
        print(f"❌ MISSING: Required parameters missing: {missing}")
        sys.exit(1)
    
    print("\n🎉 MODEL VERSION VERIFICATION PASSED")
    print("✅ You are using the correct version of HiggsAudioModel")
    print("✅ Training should proceed without 'labels' parameter errors")
    
except ImportError as e:
    print(f"❌ ImportError: {e}")
    print("❌ Could not import HiggsAudioModel")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)