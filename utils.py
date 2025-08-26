#!/usr/bin/env python3
"""
Simple validation utilities that work without ML dependencies.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional


def validate_dataset_format(data_path: str) -> bool:
    """
    Validate dataset format without requiring ML dependencies.
    
    Args:
        data_path: Path to dataset JSON file
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        print(f"🔍 Validating dataset format: {data_path}")
        
        if not os.path.exists(data_path):
            print(f"❌ Dataset file not found: {data_path}")
            return False
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            data = [data]
        
        print(f"📊 Found {len(data)} samples to validate")
        
        valid_samples = 0
        for idx, sample in enumerate(data):
            if _validate_sample_structure(sample, idx):
                valid_samples += 1
        
        print(f"✅ Validation complete: {valid_samples}/{len(data)} samples valid")
        
        if valid_samples == 0:
            print("❌ No valid samples found!")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


def _validate_sample_structure(sample: Dict[str, Any], idx: int) -> bool:
    """Validate individual sample structure using arb_inference.py process_chatml_sample logic."""
    try:
        # Check required fields
        if "messages" not in sample:
            print(f"  Sample {idx}: Missing 'messages' field")
            return False
        
        messages = sample["messages"]
        if not isinstance(messages, list) or len(messages) == 0:
            print(f"  Sample {idx}: Invalid 'messages' field")
            return False
        
        # Use EXACT same logic as arb_inference.py process_chatml_sample
        ref_audio_path = None
        ref_text = None
        target_text = None
        
        for message in messages:
            if not isinstance(message, dict):
                print(f"  Sample {idx}: Invalid message format")
                return False
            
            if "role" not in message or "content" not in message:
                print(f"  Sample {idx}: Missing 'role' or 'content'")
                return False
            
            if message["role"] == "user":
                content = message["content"]
                if isinstance(content, list):
                    # Look for text and audio content (EXACT arb_inference.py logic)
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                text_parts.append(item["text"])
                            elif item.get("type") == "audio":
                                if ref_audio_path is None:  # First audio is reference
                                    ref_audio_path = item.get("audio_url")
                    
                    if len(text_parts) >= 2:
                        ref_text = text_parts[0]  # First text is reference
                        # Look for target text
                        for text_part in text_parts[1:]:
                            if "Please generate speech" in text_part:
                                # Extract target text after the instruction
                                target_text = text_part.split(":")[-1].strip()
                                break
                        if target_text is None and len(text_parts) > 1:
                            target_text = text_parts[-1]  # Last text as fallback
                elif isinstance(content, str):
                    # Simple string content
                    if ref_text is None:
                        ref_text = content
                    else:
                        target_text = content
                        
            elif message["role"] == "assistant":
                content = message["content"]
                if isinstance(content, dict) and content.get("type") == "audio":
                    if ref_audio_path is None:
                        ref_audio_path = content.get("audio_url")
        
        # Validate that we found all required components (EXACT arb_inference.py logic)
        if not all([ref_audio_path, ref_text, target_text]):
            print(f"  Sample {idx}: Missing required components: ref_audio={ref_audio_path is not None}, ref_text={ref_text is not None}, target_text={target_text is not None}")
            return False
        
        print(f"  Sample {idx}: ✅ Valid structure (ref_audio={ref_audio_path}, ref_text='{ref_text[:30]}...', target_text='{target_text[:30]}...')")
        return True
        
    except Exception as e:
        print(f"  Sample {idx}: Validation error: {e}")
        return False


def create_sample_data(output_path: str, num_samples: int = 5):
    """Create sample training data for testing."""
    samples = []
    
    for i in range(num_samples):
        sample = {
            "messages": [
                {
                    "role": "system",
                    "content": "Generate speech in the provided voice."
                },
                {
                    "role": "user",
                    "content": f"This is reference text number {i+1} for voice cloning."
                },
                {
                    "role": "assistant",
                    "content": {
                        "type": "audio",
                        "audio_url": f"data/reference_audio/speaker_{i % 3}_ref.wav"
                    }
                },
                {
                    "role": "user",
                    "content": f"Now generate speech for this target text: Hello, this is generated speech sample {i+1}."
                }
            ],
            "speaker": f"speaker_{i % 3}",
            "start_index": 3
        }
        samples.append(sample)
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"Created {num_samples} sample data entries at {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        validate_dataset_format(sys.argv[1])
    else:
        print("Usage: python utils.py <dataset_path>")