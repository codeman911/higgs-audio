#!/usr/bin/env python3

import json
import torch
import librosa
import os
from transformers import AutoTokenizer, AutoProcessor

# Import exact components from boson_multimodal
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent
from boson_multimodal.model.higgs_audio import HiggsAudioConfig, HiggsAudioModel

def debug_model_forward():
    # Create a mock manifest with audio tokens
    mock_manifest = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please generate speech for the following text:"
                        },
                        {
                            "type": "audio",
                            "audio_url": "/Users/vikram.solanki/Projects/exp/level1/higgs-audio/train-higgs-audio/higgs_training_data_mini/huo_speaker_000000.wav"
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "I'm doing well, thank you for asking!"
                        },
                        {
                            "type": "audio",
                            "audio_url": "/Users/vikram.solanki/Projects/exp/level1/higgs-audio/train-higgs-audio/higgs_training_data_mini/huo_speaker_000001.wav"
                        }
                    ]
                }
            ]
        }
    ]
    
    # Save mock manifest
    with open("/tmp/mock_audio_manifest.json", "w") as f:
        json.dump(mock_manifest, f, indent=2)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")
    
    # Create mock config
    config = HiggsAudioConfig.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")
    
    # Load audio tokenizer
    audio_tokenizer = load_higgs_audio_tokenizer("bosonai/higgs-audio-v2-tokenizer", device='cpu')
    
    # Create dataset
    from dataset import HiggsAudioDataset, create_collator
    dataset = HiggsAudioDataset(
        manifest_path="/tmp/mock_audio_manifest.json",
        tokenizer=tokenizer,
        audio_tokenizer=audio_tokenizer
    )
    
    # Get first sample
    sample = dataset[0]
    
    # Create collator
    whisper_processor = AutoProcessor.from_pretrained("openai/whisper-large-v3", trust_remote_code=True)
    collator = create_collator(config, whisper_processor)
    
    # Process with collator
    batch_input = collator([sample])
    
    print(f"Input IDs: {batch_input.input_ids}")
    print(f"Label IDs: {batch_input.label_ids}")
    
    # Load model
    model = HiggsAudioModel.from_pretrained(
        "bosonai/higgs-audio-v2-generation-3B-base",
        config=config,
        torch_dtype=torch.bfloat16
    )
    
    # Convert batch input to dict
    batch_dict = {}
    for key, value in batch_input.__dict__.items():
        if isinstance(value, torch.Tensor):
            batch_dict[key] = value
        else:
            batch_dict[key] = value
    
    print(f"\nBatch keys: {list(batch_dict.keys())}")
    
    # Run forward pass
    with torch.no_grad():
        outputs = model(**batch_dict)
    
    print(f"\nModel outputs keys: {list(outputs.__dict__.keys())}")
    
    if hasattr(outputs, 'logits'):
        print(f"Logits shape: {outputs.logits.shape}")
    
    if hasattr(outputs, 'expanded_labels'):
        print(f"Expanded labels shape: {outputs.expanded_labels.shape}")
        print(f"Expanded labels: {outputs.expanded_labels}")
        
        # Compare with original labels
        print(f"Original labels: {batch_input.label_ids}")
        
        # Count masked vs unmasked in both
        expanded_masked = (outputs.expanded_labels == -100).sum().item()
        expanded_unmasked = (outputs.expanded_labels != -100).sum().item()
        original_masked = (batch_input.label_ids == -100).sum().item()
        original_unmasked = (batch_input.label_ids != -100).sum().item()
        
        print(f"Expanded labels - Masked: {expanded_masked}, Unmasked: {expanded_unmasked}")
        print(f"Original labels - Masked: {original_masked}, Unmasked: {original_unmasked}")

if __name__ == "__main__":
    debug_model_forward()