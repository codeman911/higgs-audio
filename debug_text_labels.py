#!/usr/bin/env python3
"""
CRITICAL DEBUG: Text Label Construction Analysis
Investigate why text labels contain only ChatML structure tokens instead of actual text content.
"""

import json
import torch
from transformers import LlamaTokenizer
from boson_multimodal.dataset.chatml_dataset import prepare_chatml_sample
from loguru import logger

# Load tokenizer
tokenizer = LlamaTokenizer.from_pretrained("/Users/vikram.solanki/Projects/tts/higgs-audio/bosonai/higgs-audio-v2-text-tokenizer")
tokenizer.pad_token = tokenizer.eos_token

def debug_prepare_chatml_sample():
    """Critical debugging of prepare_chatml_sample to see where text labels go wrong."""
    
    # Load a real training sample
    test_file = "/Users/vikram.solanki/Projects/tts/higgs-audio/VC_samples/vc_stress_test.json"
    with open(test_file, 'r', encoding='utf-8') as f:
        sample_data = json.load(f)[0]  # First sample
    
    logger.info("🔍 DEBUGGING SAMPLE DATA:")
    logger.info(f"Sample keys: {list(sample_data.keys())}")
    logger.info(f"Messages: {json.dumps(sample_data.get('messages', []), indent=2, ensure_ascii=False)}")
    logger.info(f"Start index: {sample_data.get('start_index')}")
    
    # Call prepare_chatml_sample with debugging
    logger.info("\n🚨 CALLING prepare_chatml_sample...")
    input_tokens, label_tokens, audio_contents, speaker_id = prepare_chatml_sample(sample_data, tokenizer)
    
    if input_tokens is None:
        logger.error("❌ prepare_chatml_sample returned None!")
        return
    
    # Analyze the results
    logger.info(f"\n📊 RESULTS:")
    logger.info(f"Input tokens length: {len(input_tokens)}")
    logger.info(f"Label tokens length: {len(label_tokens)}")
    logger.info(f"Audio contents: {len(audio_contents)}")
    
    # Find non-masked label tokens
    valid_labels = [token for token in label_tokens if token != -100]
    logger.info(f"\n🎯 VALID LABELS (non -100): {len(valid_labels)}")
    logger.info(f"Valid label tokens: {valid_labels[:20]}...")  # First 20
    
    # Decode the valid labels
    if valid_labels:
        try:
            decoded_labels = tokenizer.decode(valid_labels, skip_special_tokens=False)
            logger.info(f"📝 DECODED LABELS: '{decoded_labels}'")
        except Exception as e:
            logger.error(f"❌ Failed to decode labels: {e}")
    
    # Analyze token types
    special_tokens = [128009, 128012, 128013]  # eot_id, start_header_id, end_header_id
    special_count = sum(1 for token in valid_labels if token in special_tokens)
    text_count = len(valid_labels) - special_count
    
    logger.info(f"\n🔍 TOKEN ANALYSIS:")
    logger.info(f"Special tokens (structure): {special_count}")
    logger.info(f"Text tokens (content): {text_count}")
    logger.info(f"Ratio: {text_count}/{special_count} = {text_count/max(special_count,1):.2f}")
    
    # CRITICAL: Check what happens in each message
    logger.info(f"\n🚨 DETAILED MESSAGE ANALYSIS:")
    messages = sample_data.get('messages', [])
    for turn_id, message in enumerate(messages):
        role = message.get('role')
        content = message.get('content', '')
        start_index = sample_data.get('start_index')
        
        logger.info(f"Turn {turn_id}: role='{role}', start_index={start_index}")
        
        # Check condition that determines if text gets added to labels
        condition_met = role == "assistant" and (start_index is None or turn_id >= start_index)
        logger.info(f"  Condition for labels: {condition_met}")
        
        if isinstance(content, str):
            if content.strip():
                text_tokens = tokenizer.encode(content, add_special_tokens=False)
                logger.info(f"  Text content: '{content[:100]}...'")
                logger.info(f"  Text tokens: {len(text_tokens)} tokens")
                logger.info(f"  Would add to labels: {condition_met}")
            else:
                logger.info(f"  Text content: EMPTY!")
        else:
            logger.info(f"  Content type: {type(content)}")

if __name__ == "__main__":
    logger.info("🚨 CRITICAL DEBUG: Text Label Construction Analysis")
    debug_prepare_chatml_sample()
