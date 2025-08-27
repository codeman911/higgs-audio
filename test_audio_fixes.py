#!/usr/bin/env python3
"""
Test script to validate audio training fixes for near-zero audio loss issue
"""

import torch
import logging
from dataset import HiggsAudioDataset, create_collator
from transformers import AutoTokenizer, AutoProcessor
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_audio_label_masking():
    """Test that audio labels are not over-masked"""
    logger.info("=== Testing Audio Label Masking ===")
    
    # Create a minimal test dataset
    test_samples = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Please say hello"
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Hello there!"
                        },
                        {
                            "type": "audio",
                            "audio_url": "/Users/vikram.solanki/Projects/exp/level1/higgs-audio/sample_audio.wav"  # Non-existent file for testing
                        }
                    ]
                }
            ]
        }
    ]
    
    # Save test data to a temporary file
    import json
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_samples, f)
        test_manifest_path = f.name
    
    try:
        # Initialize tokenizers
        tokenizer = AutoTokenizer.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")
        audio_tokenizer = load_higgs_audio_tokenizer("bosonai/higgs-audio-v2-tokenizer", device='cpu')
        whisper_processor = AutoProcessor.from_pretrained("openai/whisper-large-v3", trust_remote_code=True)
        
        # Create dataset
        dataset = HiggsAudioDataset(
            manifest_path=test_manifest_path,
            tokenizer=tokenizer,
            audio_tokenizer=audio_tokenizer
        )
        
        logger.info(f"Dataset created with {len(dataset)} samples")
        
        # Create collator with our fixes
        from boson_multimodal.model.higgs_audio import HiggsAudioConfig
        config = HiggsAudioConfig.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")
        
        collator = create_collator(config, whisper_processor)
        
        # Test with a single sample
        if len(dataset) > 0:
            sample = dataset[0]
            logger.info(f"Sample input_ids shape: {sample.input_ids.shape}")
            logger.info(f"Sample label_ids shape: {sample.label_ids.shape}")
            logger.info(f"Sample audio_label_ids_concat: {sample.audio_label_ids_concat}")
            
            # Collate a batch
            batch = collator([sample])
            logger.info(f"Batch label_audio_ids shape: {batch.label_audio_ids.shape if hasattr(batch, 'label_audio_ids') and batch.label_audio_ids is not None else 'None'}")
            
            if hasattr(batch, 'label_audio_ids') and batch.label_audio_ids is not None:
                label_audio_ids = batch.label_audio_ids
                total_tokens = label_audio_ids.numel()
                masked_tokens = (label_audio_ids == -100).sum().item()
                valid_tokens = total_tokens - masked_tokens
                mask_ratio = masked_tokens / max(total_tokens, 1)
                
                logger.info(f"Audio label masking analysis:")
                logger.info(f"  Total tokens: {total_tokens}")
                logger.info(f"  Masked tokens: {masked_tokens}")
                logger.info(f"  Valid tokens: {valid_tokens}")
                logger.info(f"  Mask ratio: {mask_ratio:.2%}")
                
                if mask_ratio > 0.9:
                    logger.warning("⚠️  HIGH MASKING DETECTED - This could cause near-zero audio loss!")
                else:
                    logger.info("✅ Audio label masking appears reasonable")
                    
                # Check first few tokens
                if total_tokens > 0:
                    sample_tokens = label_audio_ids.flatten()[:10].tolist()
                    logger.info(f"Sample tokens (first 10): {sample_tokens}")
                    
        else:
            logger.warning("No samples in dataset")
            
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
    finally:
        # Clean up temporary file
        if os.path.exists(test_manifest_path):
            os.unlink(test_manifest_path)


def test_collator_configuration():
    """Test that collator is configured correctly"""
    logger.info("=== Testing Collator Configuration ===")
    
    from boson_multimodal.model.higgs_audio import HiggsAudioConfig
    from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
    
    config = HiggsAudioConfig.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")
    whisper_processor = AutoProcessor.from_pretrained("openai/whisper-large-v3", trust_remote_code=True)
    
    # Test our custom collator
    collator = create_collator(config, whisper_processor)
    
    # Check if it's using the base collator correctly
    if hasattr(collator, 'base_collator'):
        base_collator = collator.base_collator
        if hasattr(base_collator, 'base_collator'):
            actual_collator = base_collator.base_collator
            if hasattr(actual_collator, 'mask_audio_out_token_label'):
                mask_setting = actual_collator.mask_audio_out_token_label
                logger.info(f"mask_audio_out_token_label setting: {mask_setting}")
                if mask_setting:
                    logger.warning("⚠️  mask_audio_out_token_label is True - This may cause over-masking!")
                else:
                    logger.info("✅ mask_audio_out_token_label is False - Correct setting")
            else:
                logger.warning("⚠️  Could not find mask_audio_out_token_label attribute")
        else:
            logger.warning("⚠️  Could not access base collator")
    else:
        logger.warning("⚠️  Could not access base collator")


if __name__ == "__main__":
    logger.info("Starting audio training fixes validation...")
    
    test_collator_configuration()
    test_audio_label_masking()
    
    logger.info("Validation complete!")