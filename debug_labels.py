#!/usr/bin/env python3
"""
Debug script to verify label creation in the dataset pipeline
"""

import json
import torch
from transformers import AutoTokenizer
from boson_multimodal.dataset.chatml_dataset import prepare_chatml_sample

def create_test_sample():
    """Create a test ChatML sample for debugging"""
    return {
        "messages": [
            {
                "role": "user",
                "content": "Hello, can you say 'hello world' in the voice of the reference audio?"
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello world"
                    }
                ]
            }
        ]
    }

def debug_label_creation():
    """Debug the label creation process"""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")
    
    # Create test sample
    sample = create_test_sample()
    
    # Process with prepare_chatml_sample
    input_tokens, label_tokens, audio_contents, speaker_id = prepare_chatml_sample(sample, tokenizer)
    
    print("=== DEBUG LABEL CREATION ===")
    print(f"Input tokens length: {len(input_tokens)}")
    print(f"Label tokens length: {len(label_tokens)}")
    
    # Decode some tokens for readability
    input_text = tokenizer.decode(input_tokens[:20])
    print(f"First 20 input tokens decoded: {input_text}")
    
    # Check label distribution
    label_tensor = torch.tensor(label_tokens)
    total_tokens = len(label_tokens)
    ignore_tokens = (label_tensor == -100).sum().item()
    valid_tokens = total_tokens - ignore_tokens
    
    print(f"Total tokens: {total_tokens}")
    print(f"Tokens with -100 (ignored): {ignore_tokens}")
    print(f"Valid tokens (not -100): {valid_tokens}")
    print(f"Percentage of valid tokens: {100 * valid_tokens / total_tokens:.2f}%")
    
    # Show first 10 tokens
    print("\nFirst 10 token pairs (input, label):")
    for i in range(min(10, len(input_tokens))):
        input_token = tokenizer.decode([input_tokens[i]])
        if label_tokens[i] == -100:
            label_token = "-100(IGNORED)"
        else:
            label_token = tokenizer.decode([label_tokens[i]])
        print(f"  {i}: ({input_token}, {label_token})")
    
    # Show last 10 tokens
    print("\nLast 10 token pairs (input, label):")
    start_idx = max(0, len(input_tokens) - 10)
    for i in range(start_idx, len(input_tokens)):
        input_token = tokenizer.decode([input_tokens[i]])
        if label_tokens[i] == -100:
            label_token = "-100(IGNORED)"
        else:
            label_token = tokenizer.decode([label_tokens[i]])
        print(f"  {i}: ({input_token}, {label_token})")

if __name__ == "__main__":
    debug_label_creation()