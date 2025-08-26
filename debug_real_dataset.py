#!/usr/bin/env python3

import json
import torch
import librosa
from transformers import AutoTokenizer, AutoProcessor

# Import exact components from boson_multimodal
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer

# Import our components
from dataset import HiggsAudioDataset, create_collator
from boson_multimodal.model.higgs_audio import HiggsAudioConfig

def debug_real_dataset():
    # Create a mock manifest with a simple conversation
    mock_manifest = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, how are you today?"
                },
                {
                    "role": "assistant",
                    "content": "I'm doing well, thank you for asking!"
                }
            ]
        }
    ]
    
    # Save mock manifest
    with open("/tmp/mock_manifest.json", "w") as f:
        json.dump(mock_manifest, f, indent=2)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")
    
    # Create mock config
    config = HiggsAudioConfig.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")
    
    # Load audio tokenizer
    audio_tokenizer = load_higgs_audio_tokenizer("bosonai/higgs-audio-v2-tokenizer", device='cpu')
    
    # Create dataset
    dataset = HiggsAudioDataset(
        manifest_path="/tmp/mock_manifest.json",
        tokenizer=tokenizer,
        audio_tokenizer=audio_tokenizer
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Get first sample
    sample = dataset[0]
    print(f"\nSample type: {type(sample)}")
    print(f"Input IDs: {sample.input_ids}")
    print(f"Label IDs: {sample.label_ids}")
    print(f"Input text: {tokenizer.decode(sample.input_ids)}")
    
    # Decode label tokens (ignore -100)
    label_tokens = sample.label_ids.tolist()
    label_text_parts = []
    for token_id in label_tokens:
        if token_id != -100:
            label_text_parts.append(tokenizer.decode([token_id]))
        else:
            label_text_parts.append("[MASKED]")
    
    print(f"Label text: {' '.join(label_text_parts)}")
    
    # Count masked vs unmasked tokens
    masked_count = label_tokens.count(-100)
    unmasked_count = len(label_tokens) - masked_count
    print(f"Masked tokens: {masked_count}, Unmasked tokens: {unmasked_count}")
    
    # Create collator
    whisper_processor = AutoProcessor.from_pretrained("openai/whisper-large-v3", trust_remote_code=True)
    collator = create_collator(config, whisper_processor)
    
    print(f"\nCollator mask_audio_out_token_label: {collator.mask_audio_out_token_label}")
    
    # Process with collator
    batch_input = collator([sample])
    
    print(f"\nAfter collator:")
    print(f"Input IDs shape: {batch_input.input_ids.shape}")
    print(f"Label IDs shape: {batch_input.label_ids.shape}")
    print(f"Input IDs: {batch_input.input_ids[0].tolist()}")
    print(f"Label IDs: {batch_input.label_ids[0].tolist()}")
    
    # Count masked vs unmasked tokens after collator
    label_list = batch_input.label_ids[0].tolist()
    masked_count = label_list.count(-100)
    unmasked_count = len(label_list) - masked_count
    print(f"Masked tokens: {masked_count}, Unmasked tokens: {unmasked_count}")

if __name__ == "__main__":
    debug_real_dataset()