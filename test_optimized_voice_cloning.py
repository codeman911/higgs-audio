#!/usr/bin/env python3
"""
Simple validation script to check key optimizations without loading heavy dependencies.

This script validates:
1. Code syntax is correct
2. Key optimization functions are properly defined
3. Parameter changes are in place
"""

import sys
import os


def test_code_syntax():
    """Test that the code compiles without syntax errors."""
    print("Testing code syntax...")
    
    try:
        with open('/Users/vikram.solanki/Projects/exp/level1/higgs-audio/arabic_voice_cloning_inference.py', 'r') as f:
            code = f.read()
        
        # Try to compile the code
        compile(code, 'arabic_voice_cloning_inference.py', 'exec')
        print("✓ Code syntax is valid")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ Compilation error: {e}")
        return False


def test_optimization_features():
    """Test that key optimization features are present in the code."""
    print("\nTesting optimization features...")
    
    with open('/Users/vikram.solanki/Projects/exp/level1/higgs-audio/arabic_voice_cloning_inference.py', 'r') as f:
        code = f.read()
    
    # Check for adaptive token calculation
    if 'def calculate_adaptive_max_tokens' in code:
        print("✓ Adaptive token calculation method present")
    else:
        print("❌ Missing adaptive token calculation method")
        return False
    
    # Check for reference audio saving
    if 'def save_reference_and_generated_audio' in code:
        print("✓ Reference audio saving method present")
    else:
        print("❌ Missing reference audio saving method")
        return False
    
    # Check for proper ChatML structure with AudioContent
    if 'AudioContent(audio_url=ref_audio_path)' in code:
        print("✓ Proper ChatML structure with AudioContent")
    else:
        print("❌ Missing AudioContent usage in ChatML structure")
        return False
    
    # Check for reduced max_new_tokens default
    if 'max_new_tokens: int = 512' in code:
        print("✓ Reduced max_new_tokens default (512 instead of 2048)")
    else:
        print("❌ max_new_tokens default not reduced")
        return False
    
    # Check for adaptive_max_tokens parameter
    if 'adaptive_max_tokens: bool = True' in code:
        print("✓ Adaptive max tokens parameter added")
    else:
        print("❌ Missing adaptive_max_tokens parameter")
        return False
    
    # Check for enhanced stopping criteria
    if '"<|audio_eos|>"' in code and 'stop_strings=' in code:
        print("✓ Enhanced stopping criteria with audio_eos")
    else:
        print("❌ Missing enhanced stopping criteria")
        return False
    
    # Check for Whisper conditioning improvements
    if 'encode_whisper_embed' in code and 'audio_waveforms_concat=ref_waveform' in code:
        print("✓ Enhanced Whisper embedding integration")
    else:
        print("❌ Missing Whisper embedding enhancements")
        return False
    
    return True


def test_file_structure():
    """Test that the file structure is maintained correctly."""
    print("\nTesting file structure...")
    
    with open('/Users/vikram.solanki/Projects/exp/level1/higgs-audio/arabic_voice_cloning_inference.py', 'r') as f:
        code = f.read()
    
    # Check for proper class definition
    if 'class ArabicVoiceCloningInference:' in code:
        print("✓ Main class definition present")
    else:
        print("❌ Missing main class definition")
        return False
    
    # Check for proper imports
    essential_imports = [
        'import torch',
        'import torchaudio',
        'import soundfile as sf',
        'import shutil'
    ]
    
    for imp in essential_imports:
        if imp in code:
            print(f"✓ {imp} present")
        else:
            print(f"❌ Missing {imp}")
            return False
    
    return True


def create_test_sample():
    """Create a test ChatML sample to validate the structure."""
    print("\nCreating test ChatML sample...")
    
    import json
    
    test_sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": "Hello, this is a test reference"
                    },
                    {
                        "type": "audio",
                        "audio_url": "/fake/path/reference.wav"
                    },
                    {
                        "type": "text",
                        "text": "Please generate speech for given text: مرحبا، هذا اختبار للذكاء الاصطناعي"
                    }
                ]
            }
        ],
        "speaker": "test_speaker"
    }
    
    # Save test sample
    test_file = "/tmp/test_chatml_sample.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump([test_sample], f, indent=2, ensure_ascii=False)
    
    print(f"✓ Test ChatML sample created: {test_file}")
    return test_file


def main():
    """Run all validation tests."""
    print("Arabic Voice Cloning Optimization Validation")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Code syntax
    if test_code_syntax():
        tests_passed += 1
    
    # Test 2: Optimization features
    if test_optimization_features():
        tests_passed += 1
    
    # Test 3: File structure
    if test_file_structure():
        tests_passed += 1
    
    # Create test sample
    test_file = create_test_sample()
    
    print("\n" + "=" * 50)
    if tests_passed == total_tests:
        print("✅ All validation tests passed!")
        print("\nKey Optimizations Implemented:")
        print("  • Adaptive audio generation length control")
        print("  • Proper ChatML structure with AudioContent")
        print("  • Reference audio file saving")
        print("  • Enhanced Whisper embedding integration")
        print("  • Better audio termination conditions")
        print("  • Reduced max_new_tokens from 2048 to 512")
        
        print(f"\nYou can now test the optimized pipeline with:")
        print(f"python3 arabic_voice_cloning_inference.py --chatml_file {test_file} --output_dir ./test_output")
    else:
        print(f"❌ {tests_passed}/{total_tests} tests passed")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())