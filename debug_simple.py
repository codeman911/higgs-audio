#!/usr/bin/env python3
"""
Simple debug script to verify label creation without Whisper dependencies
"""

import json
import torch
import os
from transformers import AutoTokenizer
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from dataset import HiggsAudioDataset

def debug_simple():
    """Debug the dataset processing pipeline with a simple example"""
    # Load tokenizers
    tokenizer = AutoTokenizer.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")
    audio_tokenizer = load_higgs_audio_tokenizer("bosonai/higgs-audio-v2-tokenizer", device='cpu')
    
    # Create a mock manifest with a simple sample that has audio
    mock_manifest = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Can you say this in the voice of the reference audio?"
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Hello world, this is a test."
                        }
                    ]
                }
            ]
        }
    ]
    
    # Save mock manifest
    with open("/tmp/mock_manifest.json", "w") as f:
        json.dump(mock_manifest, f)
    
    try:
        # Create dataset
        dataset = HiggsAudioDataset(
            manifest_path="/tmp/mock_manifest.json",
            tokenizer=tokenizer,
            audio_tokenizer=audio_tokenizer
        )
        
        print("=== DEBUG DATASET PROCESSING ===")
        print(f"Dataset length: {len(dataset)}")
        
        # Get first sample
        sample = dataset[0]
        print(f"Sample input_ids shape: {sample.input_ids.shape}")
        print(f"Sample label_ids shape: {sample.label_ids.shape}")
        
        # Check label distribution
        label_tensor = sample.label_ids
        total_tokens = label_tensor.shape[0]
        ignore_tokens = (label_tensor == -100).sum().item()
        valid_tokens = total_tokens - ignore_tokens
        
        print(f"Total tokens: {total_tokens}")
        print(f"Tokens with -100 (ignored): {ignore_tokens}")
        print(f"Valid tokens (not -100): {valid_tokens}")
        print(f"Percentage of valid tokens: {100 * valid_tokens / total_tokens:.2f}%")
        
        # Decode input and labels for better understanding
        input_text = tokenizer.decode(sample.input_ids, skip_special_tokens=False)
        print(f"\nFull input text: {input_text}")
        
        # Show which tokens are actually learnable
        print("\nLearnable tokens (not -100):")
        for i in range(total_tokens):
            if sample.label_ids[i] != -100:
                input_token = tokenizer.decode([sample.input_ids[i]])
                label_token = tokenizer.decode([sample.label_ids[i]])
                print(f"  Position {i}: input='{input_token}' -> label='{label_token}'")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if os.path.exists("/tmp/mock_manifest.json"):
            os.remove("/tmp/mock_manifest.json")

if __name__ == "__main__":
    debug_simple()