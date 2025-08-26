#!/usr/bin/env python3
"""
Debug script to verify dataset processing with actual data
"""

import json
import torch
import os
from transformers import AutoTokenizer, AutoProcessor
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from dataset import HiggsAudioDataset, create_collator
from boson_multimodal.model.higgs_audio import HiggsAudioConfig

def debug_dataset_processing():
    """Debug the dataset processing pipeline"""
    # Load tokenizers
    tokenizer = AutoTokenizer.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")
    audio_tokenizer = load_higgs_audio_tokenizer("bosonai/higgs-audio-v2-tokenizer", device='cpu')
    whisper_processor = AutoProcessor.from_pretrained("openai/whisper-large-v3", trust_remote_code=True)
    
    # Create a mock manifest with a simple sample
    mock_manifest = [
        {
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
        
        # Show first 10 tokens
        print("\nFirst 10 token pairs (input, label):")
        for i in range(min(10, total_tokens)):
            input_token = tokenizer.decode([sample.input_ids[i]])
            if sample.label_ids[i] == -100:
                label_token = "-100(IGNORED)"
            else:
                label_token = tokenizer.decode([sample.label_ids[i]])
            print(f"  {i}: ({repr(input_token)}, {repr(label_token)})")
        
        # Show last 10 tokens
        print("\nLast 10 token pairs (input, label):")
        start_idx = max(0, total_tokens - 10)
        for i in range(start_idx, total_tokens):
            input_token = tokenizer.decode([sample.input_ids[i]])
            if sample.label_ids[i] == -100:
                label_token = "-100(IGNORED)"
            else:
                label_token = tokenizer.decode([sample.label_ids[i]])
            print(f"  {i}: ({repr(input_token)}, {repr(label_token)})")
        
        # Test collator
        print("\n=== TESTING COLLATOR ===")
        config = HiggsAudioConfig.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")
        config.encode_whisper_embed = True
        config.audio_in_token_idx = 128015
        config.audio_out_token_idx = 128016
        config.audio_stream_bos_id = 128013
        config.audio_stream_eos_id = 128014
        config.pad_token_id = 128001
        
        collator = create_collator(config, whisper_processor)
        batch = collator([sample])
        
        print(f"Batch label_ids shape: {batch.label_ids.shape}")
        print(f"Batch input_ids shape: {batch.input_ids.shape}")
        
        # Check batch label distribution
        batch_label_tensor = batch.label_ids[0]  # First sample in batch
        total_batch_tokens = batch_label_tensor.shape[0]
        ignore_batch_tokens = (batch_label_tensor == -100).sum().item()
        valid_batch_tokens = total_batch_tokens - ignore_batch_tokens
        
        print(f"Batch total tokens: {total_batch_tokens}")
        print(f"Batch tokens with -100 (ignored): {ignore_batch_tokens}")
        print(f"Batch valid tokens (not -100): {valid_batch_tokens}")
        print(f"Batch percentage of valid tokens: {100 * valid_batch_tokens / total_batch_tokens:.2f}%")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if os.path.exists("/tmp/mock_manifest.json"):
            os.remove("/tmp/mock_manifest.json")

if __name__ == "__main__":
    debug_dataset_processing()