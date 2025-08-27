#!/usr/bin/env python3
"""
Test script to verify the training pipeline works correctly
"""

import torch
import argparse
import os
import sys
from dataset import HiggsAudioDataset, create_collator
from transformers import AutoTokenizer, AutoProcessor, AutoConfig
from boson_multimodal.model.higgs_audio import HiggsAudioConfig, HiggsAudioModel
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from torch.utils.data import DataLoader, Subset


def test_dataset_loading():
    """Test if we can load and process a sample from the dataset"""
    print("🔍 Testing dataset loading...")
    
    try:
        # Load tokenizers
        tokenizer = AutoTokenizer.from_pretrained('bosonai/higgs-audio-v2-generation-3B-base')
        audio_tokenizer = load_higgs_audio_tokenizer('bosonai/higgs-audio-v2-tokenizer', device='cpu')
        
        # Create a small test dataset (assuming there's a test manifest)
        test_manifest = "../ms-swift/lora_training_data_zr/chatml_fixed/train_chatml_samples.json"
        if not os.path.exists(test_manifest):
            print(f"❌ Test manifest not found: {test_manifest}")
            return False
            
        dataset = HiggsAudioDataset(
            manifest_path=test_manifest,
            tokenizer=tokenizer,
            audio_tokenizer=audio_tokenizer
        )
        
        print(f"✅ Dataset loaded successfully with {len(dataset)} samples")
        
        # Test loading a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✅ Sample loaded successfully")
            print(f"  - Input IDs shape: {sample.input_ids.shape}")
            print(f"  - Label IDs shape: {sample.label_ids.shape}")
            print(f"  - Audio IDs concat shape: {sample.audio_ids_concat.shape}")
            if hasattr(sample, 'audio_label_ids_concat') and sample.audio_label_ids_concat is not None:
                print(f"  - Audio label IDs concat shape: {sample.audio_label_ids_concat.shape}")
            else:
                print(f"  - Audio label IDs concat: None")
                
        return True
        
    except Exception as e:
        print(f"❌ Error in dataset loading: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_collator():
    """Test if the collator works correctly"""
    print("\n🔍 Testing collator...")
    
    try:
        # Load components
        tokenizer = AutoTokenizer.from_pretrained('bosonai/higgs-audio-v2-generation-3B-base')
        audio_tokenizer = load_higgs_audio_tokenizer('bosonai/higgs-audio-v2-tokenizer', device='cpu')
        config = HiggsAudioConfig.from_pretrained('bosonai/higgs-audio-v2-generation-3B-base')
        whisper_processor = AutoProcessor.from_pretrained('openai/whisper-large-v3')
        
        # Create collator
        collator = create_collator(config, whisper_processor)
        print(f"✅ Collator created successfully: {type(collator)}")
        
        # Load a small dataset
        test_manifest = "../ms-swift/lora_training_data_zr/chatml_fixed/train_chatml_samples.json"
        if not os.path.exists(test_manifest):
            print(f"❌ Test manifest not found: {test_manifest}")
            return False
            
        dataset = HiggsAudioDataset(
            manifest_path=test_manifest,
            tokenizer=tokenizer,
            audio_tokenizer=audio_tokenizer
        )
        
        # Create a small batch
        if len(dataset) > 0:
            small_dataset = Subset(dataset, [0])  # Just one sample
            dataloader = DataLoader(small_dataset, batch_size=1, collate_fn=collator)
            
            # Test collation
            batch = next(iter(dataloader))
            print(f"✅ Batch created successfully")
            print(f"  - Batch type: {type(batch)}")
            print(f"  - Input IDs shape: {batch.input_ids.shape}")
            print(f"  - Label IDs shape: {batch.label_ids.shape}")
            if hasattr(batch, 'label_audio_ids') and batch.label_audio_ids is not None:
                print(f"  - Label audio IDs shape: {batch.label_audio_ids.shape}")
            else:
                print(f"  - Label audio IDs: None")
                
        return True
        
    except Exception as e:
        print(f"❌ Error in collator test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_forward():
    """Test if we can run a forward pass through the model"""
    print("\n🔍 Testing model forward pass...")
    
    try:
        # Load model
        config = HiggsAudioConfig.from_pretrained('bosonai/higgs-audio-v2-generation-3B-base')
        config.encode_whisper_embed = True  # Enable for training
        model = HiggsAudioModel.from_pretrained(
            'bosonai/higgs-audio-v2-generation-3B-base',
            config=config,
            torch_dtype=torch.bfloat16
        )
        
        print(f"✅ Model loaded successfully: {type(model)}")
        
        # Load components
        tokenizer = AutoTokenizer.from_pretrained('bosonai/higgs-audio-v2-generation-3B-base')
        audio_tokenizer = load_higgs_audio_tokenizer('bosonai/higgs-audio-v2-tokenizer', device='cpu')
        whisper_processor = AutoProcessor.from_pretrained('openai/whisper-large-v3')
        
        # Create collator
        collator = create_collator(config, whisper_processor)
        
        # Load a small dataset
        test_manifest = "../ms-swift/lora_training_data_zr/chatml_fixed/train_chatml_samples.json"
        if not os.path.exists(test_manifest):
            print(f"❌ Test manifest not found: {test_manifest}")
            return False
            
        dataset = HiggsAudioDataset(
            manifest_path=test_manifest,
            tokenizer=tokenizer,
            audio_tokenizer=audio_tokenizer
        )
        
        # Create a small batch
        if len(dataset) > 0:
            small_dataset = Subset(dataset, [0])  # Just one sample
            dataloader = DataLoader(small_dataset, batch_size=1, collate_fn=collator)
            
            # Test forward pass
            batch = next(iter(dataloader))
            
            # Move to device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # Prepare inputs
            model_inputs = {}
            for attr_name in dir(batch):
                if not attr_name.startswith('_') and not callable(getattr(batch, attr_name)):
                    value = getattr(batch, attr_name)
                    if isinstance(value, torch.Tensor):
                        model_inputs[attr_name] = value.to(device)
                    else:
                        model_inputs[attr_name] = value
            
            # Run forward pass
            with torch.no_grad():
                with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16):
                    outputs = model(**model_inputs)
                    
            print(f"✅ Forward pass successful")
            print(f"  - Output type: {type(outputs)}")
            if hasattr(outputs, 'loss'):
                print(f"  - Loss: {outputs.loss}")
            if hasattr(outputs, 'logits'):
                print(f"  - Logits shape: {outputs.logits.shape}")
            if hasattr(outputs, 'audio_logits'):
                print(f"  - Audio logits shape: {outputs.audio_logits.shape}")
                
        return True
        
    except Exception as e:
        print(f"❌ Error in model forward test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("🧪 Running Higgs Audio Training Pipeline Tests\n")
    
    tests = [
        ("Dataset Loading", test_dataset_loading),
        ("Collator", test_collator),
        ("Model Forward Pass", test_model_forward),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*50)
    print("📊 TEST RESULTS")
    print("="*50)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        if not result:
            all_passed = False
    
    print("="*50)
    if all_passed:
        print("🎉 All tests passed! The training pipeline should work correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)