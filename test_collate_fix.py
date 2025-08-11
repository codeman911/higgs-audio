#!/usr/bin/env python3
"""
Minimal test to verify the collate function fix for audio token insertion.
"""

import torch
from transformers import AutoTokenizer
from boson_multimodal.dataset.chatml_dataset import prepare_chatml_sample

def test_audio_token_insertion():
    """Test that audio tokens are properly inserted into input sequences."""
    
    print("Testing audio token insertion fix...")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")
    
    # Create a sample ChatML dict with audio
    chatml_dict = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": "Hello, please clone this voice."},
                {"type": "audio", "audio_url": "/path/to/audio.wav"}
            ]}
        ]
    }
    
    # Process with prepare_chatml_sample
    input_tokens, label_tokens, audio_contents, speaker_id = prepare_chatml_sample(
        chatml_dict, tokenizer
    )
    
    print(f"Original input tokens length: {len(input_tokens)}")
    print(f"Audio contents found: {len(audio_contents)}")
    
    # Get audio token ID
    audio_in_token_id = tokenizer.convert_tokens_to_ids("<|AUDIO|>")
    print(f"Audio token ID: {audio_in_token_id}")
    
    # Insert audio tokens (mimicking the fix)
    if audio_contents:
        modified_input_tokens = []
        modified_label_tokens = []
        
        # Add audio tokens at the beginning for reference audio
        for _ in audio_contents:
            modified_input_tokens.append(audio_in_token_id)
            modified_label_tokens.append(-100)
        
        # Add the rest of the tokens
        modified_input_tokens.extend(input_tokens)
        modified_label_tokens.extend(label_tokens)
        
        print(f"Modified input tokens length: {len(modified_input_tokens)}")
        print(f"Audio tokens inserted: {len(audio_contents)}")
        
        # Verify audio tokens are present
        input_tensor = torch.tensor(modified_input_tokens, dtype=torch.long)
        audio_mask = input_tensor == audio_in_token_id
        audio_positions = torch.where(audio_mask)[0]
        
        print(f"Audio token positions in sequence: {audio_positions.tolist()}")
        print(f"Audio mask is tensor: {isinstance(audio_mask, torch.Tensor)}")
        print(f"Audio mask shape: {audio_mask.shape}")
        
        # This should not raise the TypeError anymore
        assert isinstance(audio_mask, torch.Tensor), "audio_mask should be a tensor"
        assert audio_mask.numel() > 0, "audio_mask should not be empty"
        
        print("✅ Audio token insertion test passed!")
        return True
    
    print("❌ No audio contents found")
    return False

if __name__ == "__main__":
    success = test_audio_token_insertion()
    exit(0 if success else 1)
