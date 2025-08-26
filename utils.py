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
    """Validate individual sample structure."""
    try:
        # Check required fields
        if "messages" not in sample:
            print(f"  Sample {idx}: Missing 'messages' field")
            return False
        
        messages = sample["messages"]
        if not isinstance(messages, list) or len(messages) == 0:
            print(f"  Sample {idx}: Invalid 'messages' field")
            return False
        
        # Check message structure
        has_system = False
        has_user = False
        has_assistant_audio = False
        
        for msg_idx, message in enumerate(messages):
            if not isinstance(message, dict):
                print(f"  Sample {idx}, Message {msg_idx}: Invalid message format")
                return False
            
            if "role" not in message or "content" not in message:
                print(f"  Sample {idx}, Message {msg_idx}: Missing 'role' or 'content'")
                return False
            
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                has_system = True
            elif role == "user":
                has_user = True
            elif role == "assistant":
                if isinstance(content, dict) and content.get("type") == "audio":
                    has_assistant_audio = True
        
        # Check that we have the expected structure
        if not (has_system and has_user and has_assistant_audio):
            print(f"  Sample {idx}: Missing required message types")
            return False
        
        print(f"  Sample {idx}: ✅ Valid structure")
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