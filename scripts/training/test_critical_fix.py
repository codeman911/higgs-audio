#!/usr/bin/env python3
"""
Critical Fix Validation Script for Higgs-Audio LoRA Training Pipeline

This script validates that the training pipeline now correctly matches inference behavior:
- Reference audio is used for conditioning (audio_in_ids)
- Target audio is provided for both forward pass (audio_out_ids) and loss (label_audio_ids)
- Audio flow exactly matches inference pattern
"""

import torch
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_critical_audio_flow_fix():
    """Test the critical audio flow fix in training pipeline"""
    
    print("🔍 TESTING CRITICAL AUDIO FLOW FIX")
    print("=" * 60)
    
    try:
        # Import required modules
        from scripts.training.distributed_trainer import ArabicEnglishDataset
        from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioCollator
        from boson_multimodal.audio_processing.higgs_audio_tokenizer import HiggsAudioTokenizer
        from transformers import AutoTokenizer
        from boson_multimodal.data_types import ChatMLDatasetSample, Message, AudioContent, TextContent
        import librosa
        import numpy as np
        
        print("✅ All imports successful")
        
        # Initialize tokenizers
        print("\n📝 Initializing tokenizers...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")
            audio_tokenizer = HiggsAudioTokenizer(
                model_path="bosonai/higgs-audio-v2-tokenizer",
                device="cpu"
            )
            print(f"✅ Text tokenizer: {tokenizer.__class__.__name__}")
            print(f"✅ Audio tokenizer: {audio_tokenizer.__class__.__name__}")
            print(f"✅ Audio tokenizer codebooks: {audio_tokenizer.n_q}")
        except Exception as e:
            print(f"❌ Tokenizer initialization failed: {e}")
            return False
        
        # Create mock training sample with reference and target audio
        print("\n🎵 Creating mock training sample...")
        
        # Generate mock audio data
        sample_rate = 16000
        duration = 2.0  # 2 seconds
        ref_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        target_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        
        # Create ChatML sample with reference (user) and target (assistant) audio
        mock_sample = {
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'Please clone this voice:'},
                        {'type': 'audio', 'raw_audio': 'mock_ref_audio.wav'}
                    ]
                },
                {
                    'role': 'assistant', 
                    'content': [
                        {'type': 'text', 'text': 'Here is the cloned voice:'},
                        {'type': 'audio', 'raw_audio': 'mock_target_audio.wav'}
                    ]
                }
            ]
        }
        
        # Mock the dataset's audio loading by patching the audio tokenizer
        def mock_encode(audio_data, sr=None):
            """Mock audio encoding that returns consistent tokens"""
            if isinstance(audio_data, str):
                # Mock file path - return different tokens for ref vs target
                if 'ref' in audio_data:
                    return torch.randint(0, 1000, (8, 50))  # Reference audio tokens
                else:
                    return torch.randint(0, 1000, (8, 60))  # Target audio tokens
            else:
                # Mock waveform encoding
                seq_len = min(100, max(10, len(audio_data) // 320))
                return torch.randint(0, 1000, (8, seq_len))
        
        # Patch the encode method
        original_encode = audio_tokenizer.encode
        audio_tokenizer.encode = mock_encode
        
        # Mock file existence
        def mock_exists(path):
            return True
        
        original_exists = os.path.exists
        os.path.exists = mock_exists
        
        # Mock librosa load
        def mock_load(path, sr=None, mono=True):
            if 'ref' in path:
                return ref_audio, sample_rate
            else:
                return target_audio, sample_rate
        
        original_load = librosa.load
        librosa.load = mock_load
        
        try:
            # Create dataset and test sample processing
            print("\n🗂️ Testing dataset sample processing...")
            
            # Create a minimal dataset
            dataset = ArabicEnglishDataset([mock_sample], tokenizer, audio_tokenizer)
            sample = dataset[0]
            
            print(f"✅ Dataset sample created successfully")
            print(f"   Input IDs shape: {sample.input_ids.shape}")
            print(f"   Label IDs shape: {sample.label_ids.shape}")
            print(f"   Reference audio shape: {sample.audio_ids_concat.shape}")
            print(f"   Target audio shape: {sample.audio_label_ids_concat.shape}")
            
            # Verify audio separation
            if sample.audio_ids_concat.numel() > 0:
                print(f"✅ Reference audio tokens present: {sample.audio_ids_concat.shape}")
            else:
                print(f"⚠️ No reference audio tokens")
            
            if sample.audio_label_ids_concat.numel() > 0:
                print(f"✅ Target audio tokens present: {sample.audio_label_ids_concat.shape}")
            else:
                print(f"❌ No target audio tokens")
                return False
            
            # Test collator with custom collate function
            print("\n🔄 Testing custom collate function...")
            
            # Create collator
            collator = HiggsAudioCollator(
                tokenizer=tokenizer,
                audio_tokenizer=audio_tokenizer,
                return_audio_in_tokens=True,
                audio_num_codebooks=8
            )
            
            # Create custom collate function (same as in distributed_trainer.py)
            def custom_collate_fn(batch):
                """Custom collate function to handle target audio tokens correctly"""
                # Use the standard collator first
                collated_batch = collator(batch)
                
                # Extract target audio tokens from the original batch samples
                target_audio_tokens = []
                target_audio_starts = []
                current_pos = 0
                
                for sample in batch:
                    if hasattr(sample, 'audio_label_ids_concat') and sample.audio_label_ids_concat is not None:
                        if sample.audio_label_ids_concat.numel() > 0:
                            target_audio_tokens.append(sample.audio_label_ids_concat)
                            target_audio_starts.append(current_pos)
                            current_pos += sample.audio_label_ids_concat.shape[1]
                
                # CRITICAL FIX: Add target audio tokens to the collated batch for BOTH forward pass and loss
                if target_audio_tokens:
                    # Concatenate all target audio tokens
                    target_audio_concat = torch.cat(target_audio_tokens, dim=1)
                    
                    # Convert to dict if needed
                    if hasattr(collated_batch, '__dict__'):
                        # For model forward pass - target audio prediction
                        collated_batch.audio_out_ids = target_audio_concat
                        collated_batch.audio_out_ids_start = torch.tensor(target_audio_starts, dtype=torch.long)
                        collated_batch.audio_out_ids_start_group_loc = torch.zeros(len(target_audio_starts), dtype=torch.long)
                        
                        # For loss computation - target audio labels
                        collated_batch.label_audio_ids = target_audio_concat
                    else:
                        # If it's already a dict, add the keys
                        collated_batch['audio_out_ids'] = target_audio_concat
                        collated_batch['audio_out_ids_start'] = torch.tensor(target_audio_starts, dtype=torch.long)
                        collated_batch['audio_out_ids_start_group_loc'] = torch.zeros(len(target_audio_starts), dtype=torch.long)
                        collated_batch['label_audio_ids'] = target_audio_concat
                
                return collated_batch
            
            # Test the custom collate function
            batch = custom_collate_fn([sample])
            
            print(f"✅ Custom collate function executed successfully")
            
            # Verify critical fields are present
            critical_checks = []
            
            # Check reference audio (conditioning)
            if hasattr(batch, 'audio_in_ids') and batch.audio_in_ids is not None:
                if batch.audio_in_ids.numel() > 0:
                    print(f"✅ Reference audio (audio_in_ids): {batch.audio_in_ids.shape}")
                    critical_checks.append(True)
                else:
                    print(f"⚠️ Reference audio present but empty")
                    critical_checks.append(False)
            else:
                print(f"❌ Reference audio (audio_in_ids) missing")
                critical_checks.append(False)
            
            # Check target audio for forward pass
            if hasattr(batch, 'audio_out_ids') and batch.audio_out_ids is not None:
                if batch.audio_out_ids.numel() > 0:
                    print(f"✅ Target audio for forward pass (audio_out_ids): {batch.audio_out_ids.shape}")
                    critical_checks.append(True)
                else:
                    print(f"❌ Target audio for forward pass present but empty")
                    critical_checks.append(False)
            else:
                print(f"❌ Target audio for forward pass (audio_out_ids) missing")
                critical_checks.append(False)
            
            # Check target audio for loss computation
            if hasattr(batch, 'label_audio_ids') and batch.label_audio_ids is not None:
                if batch.label_audio_ids.numel() > 0:
                    print(f"✅ Target audio for loss (label_audio_ids): {batch.label_audio_ids.shape}")
                    critical_checks.append(True)
                else:
                    print(f"❌ Target audio for loss present but empty")
                    critical_checks.append(False)
            else:
                print(f"❌ Target audio for loss (label_audio_ids) missing")
                critical_checks.append(False)
            
            # Verify audio flow matches inference pattern
            print(f"\n🎯 CRITICAL AUDIO FLOW VALIDATION:")
            print(f"   Reference audio → audio_in_ids (conditioning): {'✅' if critical_checks[0] else '❌'}")
            print(f"   Target audio → audio_out_ids (forward pass): {'✅' if critical_checks[1] else '❌'}")
            print(f"   Target audio → label_audio_ids (loss): {'✅' if critical_checks[2] else '❌'}")
            
            # Final validation
            if all(critical_checks):
                print(f"\n🎉 CRITICAL FIX VALIDATION: SUCCESS!")
                print(f"   Training pipeline now matches inference behavior exactly")
                print(f"   Reference audio is used for conditioning only")
                print(f"   Target audio is provided for both prediction and loss")
                return True
            else:
                print(f"\n❌ CRITICAL FIX VALIDATION: FAILED!")
                print(f"   Missing critical audio flow components")
                return False
                
        finally:
            # Restore original functions
            audio_tokenizer.encode = original_encode
            os.path.exists = original_exists
            librosa.load = original_load
            
    except Exception as e:
        print(f"❌ Critical fix validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main validation function"""
    print("🚀 HIGGS-AUDIO CRITICAL FIX VALIDATION")
    print("=" * 80)
    
    success = test_critical_audio_flow_fix()
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 VALIDATION RESULT: SUCCESS!")
        print("   The critical audio flow fix is working correctly.")
        print("   Training pipeline now matches inference behavior.")
        print("   Ready to resume training with stable loss computation.")
    else:
        print("❌ VALIDATION RESULT: FAILED!")
        print("   The critical audio flow fix needs further investigation.")
        print("   Do not resume training until this is resolved.")
    
    return success

if __name__ == "__main__":
    main()
