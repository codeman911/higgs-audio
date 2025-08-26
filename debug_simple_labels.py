#!/usr/bin/env python3

import json
import torch
from transformers import AutoTokenizer

# Import exact components from boson_multimodal
from boson_multimodal.dataset.chatml_dataset import prepare_chatml_sample
from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent

def debug_simple_labels():
    # Create a simple test sample without audio
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
    
    # Decode label tokens (ignore -100)
    label_text_parts = []
    for token_id in label_tokens:
        if token_id != -100:
            label_text_parts.append(tokenizer.decode([token_id]))
        else:
            label_text_parts.append("[MASKED]")
    
    print("Label text:", " ".join(label_text_parts))
    
    # Count masked vs unmasked tokens
    masked_count = label_tokens.count(-100)
    unmasked_count = len(label_tokens) - masked_count
    print(f"Masked tokens: {masked_count}, Unmasked tokens: {unmasked_count}")
    
    # Show which specific tokens are unmasked
    print("\nUnmasked tokens:")
    for i, token_id in enumerate(label_tokens):
        if token_id != -100:
            print(f"  Position {i}: {token_id} -> '{tokenizer.decode([token_id])}'")

if __name__ == "__main__":
    debug_simple_labels()