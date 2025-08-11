#!/usr/bin/env python3
"""
Diagnostic script to reproduce and understand the collator TypeError.
"""

import torch
from transformers import AutoTokenizer, WhisperProcessor
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator

def test_collator_issue():
    """Reproduce the exact collator issue to understand the problem."""
    
    print("Reproducing collator issue...")
    print("="*60)
    
    # Initialize tokenizer and collator
    tokenizer = AutoTokenizer.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")
    audio_tokenizer = None  # Not needed for this test
    
    # Initialize collator with correct parameters
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    
    collator = HiggsAudioSampleCollator(
        whisper_processor=whisper_processor,
        audio_in_token_id=tokenizer.convert_tokens_to_ids("<|AUDIO|>"),  
        audio_out_token_id=tokenizer.convert_tokens_to_ids("<|AUDIO_OUT|>"),  
        pad_token_id=tokenizer.pad_token_id,
        audio_stream_bos_id=tokenizer.convert_tokens_to_ids("<|audio_stream_bos|>"),  
        audio_stream_eos_id=tokenizer.convert_tokens_to_ids("<|audio_stream_eos|>"),  
        round_to=8,
        pad_left=False,
        encode_whisper_embed=True,
        return_audio_in_tokens=True,
        audio_num_codebooks=8,
        use_delay_pattern=False,
        disable_audio_codes_transform=False
    )
    
    audio_in_token_id = tokenizer.convert_tokens_to_ids("<|AUDIO|>")
    print(f"Audio token ID: {audio_in_token_id}")
    
    # Create a sample WITH audio content (like our training data)
    chatml_dict = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": "Hello, please clone this voice."},
                {"type": "audio", "audio_url": "/path/to/audio.wav"}
            ]}
        ]
    }
    
    # Use prepare_chatml_sample
    input_tokens, label_tokens, audio_contents, speaker_id = prepare_chatml_sample(
        chatml_dict, tokenizer
    )
    
    print(f"\nAfter prepare_chatml_sample:")
    print(f"  Input tokens length: {len(input_tokens)}")
    print(f"  Input tokens type: {type(input_tokens)}")
    print(f"  Audio contents: {len(audio_contents)}")
    
    # Check for audio tokens
    if isinstance(input_tokens, list):
        num_audio_tokens = input_tokens.count(audio_in_token_id)
    else:
        num_audio_tokens = (input_tokens == audio_in_token_id).sum().item()
    print(f"  Audio tokens in input: {num_audio_tokens}")
    
    # Create ChatMLDatasetSample as the training code does
    sample = ChatMLDatasetSample(
        input_ids=torch.tensor(input_tokens, dtype=torch.long),
        label_ids=torch.tensor(label_tokens, dtype=torch.long),
        audio_ids_concat=torch.empty((8, 0), dtype=torch.long),
        audio_ids_start=torch.tensor([], dtype=torch.long),
        audio_waveforms_concat=torch.empty(1, 0, dtype=torch.float32),  # No actual waveform
        audio_waveforms_start=torch.tensor([], dtype=torch.long),
        audio_sample_rate=torch.tensor([], dtype=torch.float32),
        audio_speaker_indices=torch.tensor([], dtype=torch.long)
    )
    
    print(f"\nChatMLDatasetSample created:")
    print(f"  input_ids type: {type(sample.input_ids)}")
    print(f"  input_ids shape: {sample.input_ids.shape}")
    print(f"  input_ids dtype: {sample.input_ids.dtype}")
    
    # Test the comparison that fails in the collator
    print(f"\nTesting the problematic comparison:")
    audio_in_mask = sample.input_ids == audio_in_token_id
    print(f"  audio_in_mask type: {type(audio_in_mask)}")
    print(f"  audio_in_mask is tensor: {isinstance(audio_in_mask, torch.Tensor)}")
    if isinstance(audio_in_mask, torch.Tensor):
        print(f"  audio_in_mask shape: {audio_in_mask.shape}")
        print(f"  audio_in_mask dtype: {audio_in_mask.dtype}")
        print(f"  Number of True values: {audio_in_mask.sum().item()}")
    else:
        print(f"  audio_in_mask value: {audio_in_mask}")
    
    # Now test with the actual collator
    print(f"\nTesting with actual collator:")
    batch = [sample]
    
    try:
        # This should trigger the error if there's an issue
        collated = collator(batch)
        print("✅ Collation succeeded!")
        print(f"  Collated batch keys: {collated.__dict__.keys()}")
    except TypeError as e:
        print(f"❌ Collation failed with TypeError: {e}")
        
        # Debug the exact issue
        print("\nDebugging the issue:")
        print(f"  sample.input_ids: {sample.input_ids}")
        print(f"  audio_in_token_id: {audio_in_token_id}")
        
        # Try the comparison directly
        try:
            mask = sample.input_ids == audio_in_token_id
            print(f"  Direct comparison result type: {type(mask)}")
            print(f"  Direct comparison result: {mask}")
        except Exception as e2:
            print(f"  Direct comparison failed: {e2}")
    
    # Test with 0-dimensional tensor edge case
    print(f"\n[Edge Case Test] 0-dimensional tensor:")
    scalar_tensor = torch.tensor(audio_in_token_id)
    print(f"  Scalar tensor: {scalar_tensor}")
    print(f"  Scalar tensor shape: {scalar_tensor.shape}")
    print(f"  Scalar tensor ndim: {scalar_tensor.ndim}")
    
    comparison = sample.input_ids == scalar_tensor
    print(f"  Comparison with scalar tensor type: {type(comparison)}")
    print(f"  Is tensor: {isinstance(comparison, torch.Tensor)}")
    
    print("\n" + "="*60)
    print("Diagnosis complete.")

if __name__ == "__main__":
    test_collator_issue()
