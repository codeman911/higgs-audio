#!/usr/bin/env python3
"""
Quick script to validate there's no data leakage between reference and target audio
"""

import torch
import json
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer

def check_data_leakage(chatml_file: str, audio_tokenizer_path: str, max_samples: int = 10):
    """Check for data leakage between reference and target audio"""
    
    print("🔍 Loading audio tokenizer...")
    audio_tokenizer = load_higgs_audio_tokenizer(audio_tokenizer_path)
    
    print(f"📄 Loading ChatML samples from: {chatml_file}")
    with open(chatml_file, 'r') as f:
        samples = json.load(f)
    
    if isinstance(samples, dict):
        samples = samples.get('samples', samples.get('data', []))
    
    samples = samples[:max_samples]
    print(f"✅ Checking {len(samples)} samples for data leakage")
    
    identical_count = 0
    very_similar_count = 0
    
    for i, sample in enumerate(samples):
        messages = sample.get('messages', [])
        
        ref_audio_path = None
        target_audio_path = None
        
        # Extract audio paths
        for msg in messages:
            if msg.get('role') == 'user':
                content = msg.get('content', [])
                if isinstance(content, list):
                    for item in content:
                        if item.get('type') == 'audio':
                            ref_audio_path = item.get('audio_url')
            elif msg.get('role') == 'assistant':
                content = msg.get('content', [])
                if isinstance(content, list):
                    for item in content:
                        if item.get('type') == 'audio':
                            target_audio_path = item.get('audio_url')
        
        if not ref_audio_path or not target_audio_path:
            print(f"⚠️  Sample {i}: Missing audio paths")
            continue
            
        if not os.path.exists(ref_audio_path) or not os.path.exists(target_audio_path):
            print(f"⚠️  Sample {i}: Audio files don't exist")
            continue
        
        # Check if paths are identical
        if ref_audio_path == target_audio_path:
            print(f"🚨 Sample {i}: IDENTICAL PATHS - {ref_audio_path}")
            identical_count += 1
            continue
        
        # Tokenize both audio files
        try:
            ref_tokens = audio_tokenizer.encode(ref_audio_path)
            target_tokens = audio_tokenizer.encode(target_audio_path)
            
            # Check token similarity
            if ref_tokens.shape == target_tokens.shape:
                # Exact match check
                exact_match = torch.equal(ref_tokens, target_tokens)
                if exact_match:
                    print(f"🚨 Sample {i}: IDENTICAL TOKENS - ref and target are the same!")
                    identical_count += 1
                    continue
                
                # Similarity check (>95% token overlap)
                total_tokens = ref_tokens.numel()
                matching_tokens = (ref_tokens == target_tokens).sum().item()
                similarity = matching_tokens / total_tokens
                
                if similarity > 0.95:
                    print(f"🚨 Sample {i}: VERY SIMILAR ({similarity:.2%}) - potential leakage")
                    very_similar_count += 1
                elif similarity > 0.7:
                    print(f"⚠️  Sample {i}: Similar ({similarity:.2%}) - worth investigating")
                else:
                    print(f"✅ Sample {i}: Different ({similarity:.2%}) - healthy")
            else:
                print(f"✅ Sample {i}: Different shapes {ref_tokens.shape} vs {target_tokens.shape}")
                
        except Exception as e:
            print(f"❌ Sample {i}: Error tokenizing - {str(e)}")
    
    print(f"\n🎯 DATA LEAKAGE SUMMARY:")
    print(f"   Identical: {identical_count}/{len(samples)} ({identical_count/len(samples)*100:.1f}%)")
    print(f"   Very similar (>95%): {very_similar_count}/{len(samples)} ({very_similar_count/len(samples)*100:.1f}%)")
    
    if identical_count > 0 or very_similar_count > 0:
        print(f"🚨 CRITICAL: Found {identical_count + very_similar_count} samples with data leakage!")
        print(f"   This explains why the model loss drops so fast - it's memorizing identical patterns!")
        return True
    else:
        print(f"✅ No obvious data leakage detected - issue is likely elsewhere")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chatml_file", default="test_processed/test_chatml_samples.json")
    parser.add_argument("--audio_tokenizer_path", default="../train-higgs-audio/audio_tokenizer/")
    parser.add_argument("--max_samples", type=int, default=20)
    
    args = parser.parse_args()
    
    leakage_found = check_data_leakage(args.chatml_file, args.audio_tokenizer_path, args.max_samples)
    
    if leakage_found:
        print(f"\n🔧 RECOMMENDED ACTIONS:")
        print(f"   1. Audit data processing pipeline")
        print(f"   2. Ensure reference ≠ target audio")
        print(f"   3. Add validation to prevent identical audio pairs")
    else:
        print(f"\n🔍 NEXT STEPS:")
        print(f"   1. Check for delay pattern issues")
        print(f"   2. Investigate collator masking logic")
        print(f"   3. Run micro-overfit test on single sample")
