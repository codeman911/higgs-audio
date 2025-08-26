#!/usr/bin/env python3
"""
Final verification script that simulates the exact issue from the logs
"""

import json
import torch
import os
from transformers import AutoTokenizer
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from dataset import HiggsAudioDataset, create_collator
from boson_multimodal.model.higgs_audio import HiggsAudioConfig
from torch.utils.data import DataLoader

def simulate_original_issue():
    """Simulate the original issue where all text labels were -100"""
    # Load tokenizers
    tokenizer = AutoTokenizer.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")
    audio_tokenizer = load_higgs_audio_tokenizer("bosonai/higgs-audio-v2-tokenizer", device='cpu')
    
    # Create a mock manifest similar to what might cause the original issue
    # This simulates a zero-shot voice cloning scenario with audio tokens
    mock_manifest = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please generate speech for given text in reference audio's voice: Hello world, this is a test."
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
        # Simulate the old behavior by manually setting mask_audio_out_token_label to True
        old_collator.mask_audio_out_token_label = True
        
        sample = dataset[0]
        old_batch = old_collator([sample])
        
        print(f"Old batch label_ids shape: {old_batch.label_ids.shape}")
        
        # Check if ALL labels are -100 (the original issue)
        old_labels = old_batch.label_ids[0]
        all_masked = (old_labels == -100).all().item()
        valid_count = (old_labels != -100).sum().item()
        
        print(f"Old collator - All labels masked: {all_masked}")
        print(f"Old collator - Valid tokens: {valid_count}/{len(old_labels)}")
        
        if all_masked:
            print("❌ CONFIRMED: Old collator causes the exact issue from logs (all labels -100)")
        else:
            print("⚠️  Old collator does not reproduce the issue")
        
        # Test with the NEW (fixed) collator configuration
        print("\n=== TESTING WITH NEW (FIXED) COLLATOR ===")
        # Recreate collator with our fix
        new_collator = create_collator(config, whisper_processor)
        # The fix ensures mask_audio_out_token_label is False by default
        
        new_batch = new_collator([sample])
        
        print(f"New batch label_ids shape: {new_batch.label_ids.shape}")
        
        # Check if labels are properly created
        new_labels = new_batch.label_ids[0]
        all_masked_new = (new_labels == -100).all().item()
        valid_count_new = (new_labels != -100).sum().item()
        
        print(f"New collator - All labels masked: {all_masked_new}")
        print(f"New collator - Valid tokens: {valid_count_new}/{len(new_labels)}")
        
        if not all_masked_new and valid_count_new > 0:
            print("✅ FIXED: New collator properly creates learnable labels")
            
            # Show first and last 5 labels to verify they're not all -100
            first_5 = new_labels[:5].tolist()
            last_5 = new_labels[-5:].tolist()
            print(f"First 5 labels: {first_5}")
            print(f"Last 5 labels: {last_5}")
            
            # Count how many are -100 vs valid
            masked_count = (new_labels == -100).sum().item()
            total_count = len(new_labels)
            print(f"Labels distribution: {masked_count} masked, {valid_count_new} valid out of {total_count} total")
        else:
            print("❌ New collator still has issues")
        
        print("\n=== COMPARISON SUMMARY ===")
        print(f"Old collator: {valid_count} valid tokens")
        print(f"New collator: {valid_count_new} valid tokens")
        print(f"Improvement: {valid_count_new - valid_count} more learnable tokens")
        
        if valid_count_new > valid_count:
            print("✅ FIX CONFIRMED: The collator fix resolves the label masking issue")
        else:
            print("⚠️  Fix verification incomplete")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if os.path.exists("/tmp/mock_manifest.json"):
            os.remove("/tmp/mock_manifest.json")

if __name__ == "__main__":
    simulate_original_issue()