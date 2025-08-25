"""
Utility functions for Higgs-Audio training pipeline.

These functions don't require ML dependencies and can be used for
data preparation and validation tasks.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any


def create_sample_data(output_path: str, num_samples: int = 10):
    """
    Create sample training data in the correct ChatML format.
    
    This is a utility function to generate example training data
    that follows the exact format expected by the training pipeline.
    """
    samples = []
    
    for i in range(num_samples):
        # Voice cloning sample format (matching arb_inference.py)
        sample = {
            "messages": [
                {
                    "role": "system",
                    "content": "Generate speech in the provided voice."
                },
                {
                    "role": "user", 
                    "content": f"This is reference text number {i + 1} for voice cloning."
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
                    "content": f"Now generate speech for this target text: Hello, this is generated speech sample {i + 1}."
                }
            ],
            "speaker": f"speaker_{i % 3}",
            "start_index": 3  # Start generating from the last user message
        }
        samples.append(sample)
    
    # Save sample data
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if there is one
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"📝 Created {num_samples} sample training examples at {output_path}")
    print("📋 Sample format:")
    print(json.dumps(samples[0], indent=2, ensure_ascii=False))


def validate_dataset_format(data_path: str) -> bool:
    """
    Validate that a dataset file follows the correct ChatML format.
    
    Args:
        data_path: Path to the dataset JSON file
        
    Returns:
        True if format is valid, False otherwise
    """
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        if not isinstance(samples, list):
            print("❌ Dataset must be a list of samples")
            return False
        
        for i, sample in enumerate(samples[:5]):  # Check first 5 samples
            if not isinstance(sample, dict):
                print(f"❌ Sample {i}: Must be a dictionary")
                return False
            
            if 'messages' not in sample:
                print(f"❌ Sample {i}: Missing 'messages' field")
                return False
            
            messages = sample['messages']
            if not isinstance(messages, list):
                print(f"❌ Sample {i}: 'messages' must be a list")
                return False
            
            # Check for required roles and structure
            has_system = any(msg.get('role') == 'system' for msg in messages)
            has_audio = any(
                isinstance(msg.get('content'), dict) and 
                msg['content'].get('type') == 'audio' 
                for msg in messages
            )
            
            if not has_system:
                print(f"⚠️ Sample {i}: Missing system message (recommended)")
            
            if not has_audio:
                print(f"❌ Sample {i}: Missing audio content")
                return False
        
        print(f"✅ Dataset format validation passed for {data_path}")
        print(f"📊 Found {len(samples)} samples")
        return True
        
    except Exception as e:
        print(f"❌ Dataset validation failed: {e}")
        return False


def create_data_directory_structure(base_path: str):
    """Create the standard directory structure for training data."""
    base_path = Path(base_path)
    
    directories = [
        base_path / "data",
        base_path / "data" / "reference_audio",
        base_path / "checkpoints",
        base_path / "logs",
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    print("✅ Directory structure created successfully")


if __name__ == "__main__":
    # Example usage: Create sample data
    sample_data_path = "data/sample_training_data.json"
    create_sample_data(sample_data_path, num_samples=5)
    
    # Validate the created data
    validate_dataset_format(sample_data_path)