#!/usr/bin/env python3
"""
Test script to verify the training pipeline works correctly
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

def create_test_data():
    """Create sample training data for testing"""
    test_data = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, how are you today?"
                },
                {
                    "role": "assistant",
                    "content": "I'm doing well, thank you for asking!"
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Can you tell me a joke?"
                },
                {
                    "role": "assistant",
                    "content": "Why don't scientists trust atoms? Because they make up everything!"
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather like?"
                },
                {
                    "role": "assistant",
                    "content": "I'm an AI assistant, so I don't have access to real-time weather information."
                }
            ]
        }
    ]
    
    # Create a temporary file with the test data
    temp_dir = tempfile.mkdtemp()
    train_file = os.path.join(temp_dir, "test_train_data.json")
    
    with open(train_file, 'w') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    return train_file, temp_dir

def test_training_pipeline():
    """Test the training pipeline with sample data"""
    print("Creating test data...")
    train_file, temp_dir = create_test_data()
    
    try:
        print(f"Test data created at: {train_file}")
        print("You can now run training with:")
        print(f"python trainer.py --train_manifest {train_file} --output_dir ./test_output --epochs 1 --batch_size 2")
        
        # Verify the file exists and has content
        if os.path.exists(train_file):
            with open(train_file, 'r') as f:
                data = json.load(f)
                print(f"✓ Training data file created successfully with {len(data)} samples")
        else:
            print("✗ Failed to create training data file")
            
    except Exception as e:
        print(f"Error creating test data: {e}")
    finally:
        # Cleanup
        # shutil.rmtree(temp_dir)  # Uncomment if you want to automatically clean up
        pass

if __name__ == "__main__":
    test_training_pipeline()