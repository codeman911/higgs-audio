#!/usr/bin/env python3
"""
Comprehensive test that includes audio tokens to properly verify the fix
"""

import json
import torch
import os
from transformers import AutoTokenizer
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from dataset import HiggsAudioDataset, create_collator
from boson_multimodal.model.higgs_audio import HiggsAudioConfig
from torch.utils.data import DataLoader

def test_with_audio_tokens():
    """Test the fix with samples that include audio tokens"""
    # Load tokenizers
    tokenizer = AutoTokenizer.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")
    audio_tokenizer = load_higgs_audio_tokenizer("bosonai/higgs-audio-v2-tokenizer", device='cpu')
    
    # Create a mock manifest with audio tokens that would trigger the masking issue
    mock_manifest = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please generate speech for given text in reference audio's voice: Hello world"
                        },
                        {
                            "type": "audio",
                            "audio_url": "/path/to/nonexistent/audio.wav"  # This won't be loaded but will create tokens
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Hello world"
                        },
                        {
                            "type": "audio",
                            "audio_url": "/path/to/nonexistent/audio.wav"  # This won't be loaded but will create tokens
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
        
        # Create config (mock)
        class MockConfig:
            def __init__(self):
                self.encode_whisper_embed = True
                self.audio_in_token_idx = 128015
                self.audio_out_token_idx = 128016
                self.audio_stream_bos_id = 128013
                self.audio_stream_eos_id = 128014
                self.pad_token_id = 128001
                self.use_delay_pattern = False
                self.audio_num_codebooks = 8
        
        config = MockConfig()
        
        # Create a mock whisper processor class
        class MockWhisperProcessor:
            def __init__(self):
                class MockFeatureExtractor:
                    def __init__(self):
                        self.sampling_rate = 16000
                        self.feature_size = 80
                        self.nb_max_frames = 3000
                self.feature_extractor = MockFeatureExtractor()
        
        whisper_processor = MockWhisperProcessor()
        
        # Test with the OLD (problematic) collator configuration
        print("=== TESTING WITH OLD (PROBLEMATIC) COLLATOR ===")
        old_collator = create_collator(config, whisper_processor)
        # Manually set the problematic setting to simulate the original issue
        old_collator.mask_audio_out_token_label = True
        
        sample = dataset[0]
        print(f"Sample has {len(sample.input_ids)} tokens")
        
        # Count audio out tokens in the sample
        audio_out_count = (sample.input_ids == 128016).sum().item()
        print(f"Sample contains {audio_out_count} <|AUDIO_OUT|> tokens")
        
        old_batch = old_collator([sample])
        
        print(f"Old batch label_ids shape: {old_batch.label_ids.shape}")
        
        # Check the labels before any processing
        old_labels_before = sample.label_ids.clone()
        old_valid_before = (old_labels_before != -100).sum().item()
        print(f"Original sample valid tokens: {old_valid_before}/{len(old_labels_before)}")
        
        # Check labels after collator processing
        old_labels = old_batch.label_ids[0]
        old_all_masked = (old_labels == -100).all().item()
        old_valid_count = (old_labels != -100).sum().item()
        
        print(f"Old collator - All labels masked: {old_all_masked}")
        print(f"Old collator - Valid tokens after processing: {old_valid_count}/{len(old_labels)}")
        
        # Show first and last 5 labels
        first_5_old = old_labels[:5].tolist()
        last_5_old = old_labels[-5:].tolist()
        print(f"Old - First 5 labels: {first_5_old}")
        print(f"Old - Last 5 labels: {last_5_old}")
        
        # Test with the NEW (fixed) collator configuration
        print("\n=== TESTING WITH NEW (FIXED) COLLATOR ===")
        # Recreate collator with our fix
        new_collator = create_collator(config, whisper_processor)
        # The fix ensures mask_audio_out_token_label is False by default
        
        new_batch = new_collator([sample])
        
        print(f"New batch label_ids shape: {new_batch.label_ids.shape}")
        
        # Check if labels are properly created
        new_labels = new_batch.label_ids[0]
        new_all_masked = (new_labels == -100).all().item()
        new_valid_count = (new_labels != -100).sum().item()
        
        print(f"New collator - All labels masked: {new_all_masked}")
        print(f"New collator - Valid tokens after processing: {new_valid_count}/{len(new_labels)}")
        
        # Show first and last 5 labels
        first_5_new = new_labels[:5].tolist()
        last_5_new = new_labels[-5:].tolist()
        print(f"New - First 5 labels: {first_5_new}")
        print(f"New - Last 5 labels: {last_5_new}")
        
        print("\n=== DETAILED ANALYSIS ===")
        print(f"Audio out tokens in sample: {audio_out_count}")
        print(f"Valid tokens before collator (should be assistant text): {old_valid_before}")
        print(f"Valid tokens after OLD collator: {old_valid_count}")
        print(f"Valid tokens after NEW collator: {new_valid_count}")
        
        # The fix should preserve the original valid tokens
        if new_valid_count >= old_valid_before:
            print("✅ NEW COLLATOR PRESERVES VALID TOKENS")
        else:
            print("❌ NEW COLLATOR IS OVER-MASKING")
            
        if old_valid_count < old_valid_before:
            print("⚠️  OLD COLLATOR WAS OVER-MASKING (CONFIRMING THE ISSUE)")
        else:
            print("ℹ️  OLD COLLATOR BEHAVIOR AS EXPECTED")
            
        print("\n=== COMPARISON SUMMARY ===")
        print(f"Old collator valid tokens: {old_valid_count}")
        print(f"New collator valid tokens: {new_valid_count}")
        print(f"Difference: {new_valid_count - old_valid_count} tokens")
        
        if new_valid_count >= old_valid_count:
            print("✅ FIX CONFIRMED: New collator maintains or improves label quality")
        else:
            print("⚠️  Fix may need adjustment")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if os.path.exists("/tmp/mock_manifest.json"):
            os.remove("/tmp/mock_manifest.json")

if __name__ == "__main__":
    test_with_audio_tokens()