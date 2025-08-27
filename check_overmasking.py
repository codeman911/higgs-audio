#!/usr/bin/env python3

import json
import torch
from transformers import AutoTokenizer

# Import exact components from boson_multimodal
from boson_multimodal.dataset.chatml_dataset import prepare_chatml_sample

def check_overmasking():
    # Create a simple test sample
    sample = {
        "messages": [
            {
                "role": "user",
                "content": "Hello, how are you?"
            },
            {
                "role": "assistant",
                "content": "I'm doing well, thank you for asking!"
            }
        ]
    }
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")
    
    # Process the sample
    input_tokens, label_tokens, audio_contents, speaker_id = prepare_chatml_sample(sample, tokenizer)
    
    print("Input tokens:", input_tokens)
    print("Label tokens:", label_tokens)
    print("Input text:", tokenizer.decode(input_tokens))
    
    # Count masked vs unmasked tokens
    masked_count = label_tokens.count(-100)
    unmasked_count = len(label_tokens) - masked_count
    total_count = len(label_tokens)
    
    print(f"\nOriginal label statistics:")
    print(f"  Masked tokens: {masked_count}")
    print(f"  Unmasked tokens: {unmasked_count}")
    print(f"  Total tokens: {total_count}")
    print(f"  Percentage unmasked: {unmasked_count/total_count*100:.1f}%")
    
    # Show which tokens are unmasked
    print(f"\nUnmasked token positions and values:")
    for i, token_id in enumerate(label_tokens):
        if token_id != -100:
            token_text = tokenizer.decode([token_id])
            print(f"  Position {i}: {token_id} -> '{token_text}'")

if __name__ == "__main__":
    check_overmasking()