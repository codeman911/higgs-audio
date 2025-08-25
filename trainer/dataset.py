"""
Dataset implementation for Higgs-Audio LoRA training.

Reuses existing boson_multimodal components without modification:
- prepare_chatml_sample for data processing
- ChatMLDatasetSample for data structure
- Existing audio tokenizer for audio encoding
"""

import json
import torch
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from torch.utils.data import Dataset

# Import existing boson_multimodal components
from boson_multimodal.dataset.chatml_dataset import (
    prepare_chatml_sample,
    ChatMLDatasetSample,
)
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent, TextContent


class VoiceCloningDataset(Dataset):
    """
    Simple dataset wrapper that reuses existing boson_multimodal functions.
    
    Processes ChatML format data for zero-shot voice cloning training.
    The data format follows the same pattern as arb_inference.py:
    
    {
      "messages": [
        {"role": "system", "content": "Generate speech in the provided voice."},
        {"role": "user", "content": "Reference text spoken in the audio"},
        {"role": "assistant", "content": {"type": "audio", "audio_url": "path/to/ref.wav"}},
        {"role": "user", "content": "Target text to generate speech for"}
      ],
      "speaker": "speaker_id",
      "start_index": 3
    }
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        audio_tokenizer,
        audio_base_path: str = "",
        validate_audio_paths: bool = True,
    ):
        """
        Initialize the voice cloning dataset.
        
        Args:
            data_path: Path to the JSON file containing training samples
            tokenizer: Text tokenizer from HiggsAudioModel
            audio_tokenizer: Audio tokenizer for encoding reference audio
            audio_base_path: Base path for resolving relative audio paths
            validate_audio_paths: Whether to validate audio file existence
        """
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.audio_base_path = Path(audio_base_path) if audio_base_path else Path()
        
        # Load training data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
        
        print(f"📂 Loaded {len(self.samples)} training samples from {data_path}")
        
        # Validate and filter samples
        if validate_audio_paths:
            valid_samples = []
            for i, sample in enumerate(self.samples):
                if self._validate_sample(sample, i):
                    valid_samples.append(sample)
            
            print(f"✅ {len(valid_samples)}/{len(self.samples)} samples passed validation")
            self.samples = valid_samples
        
        if len(self.samples) == 0:
            raise ValueError("No valid training samples found!")
    
    def _validate_sample(self, sample: Dict[str, Any], idx: int) -> bool:
        """Validate a single training sample."""
        try:
            # Check required fields
            if 'messages' not in sample:
                print(f"⚠️ Sample {idx}: Missing 'messages' field")
                return False
            
            messages = sample['messages']
            if not isinstance(messages, list) or len(messages) < 3:
                print(f"⚠️ Sample {idx}: Invalid messages structure")
                return False
            
            # Find audio content and validate paths
            for msg in messages:
                if isinstance(msg.get('content'), dict) and msg['content'].get('type') == 'audio':
                    audio_url = msg['content'].get('audio_url', '')
                    if audio_url:
                        audio_path = self._resolve_audio_path(audio_url)
                        if not audio_path.exists():
                            print(f"⚠️ Sample {idx}: Audio file not found: {audio_path}")
                            return False
            
            return True
        
        except Exception as e:
            print(f"⚠️ Sample {idx}: Validation error: {e}")
            return False
    
    def _resolve_audio_path(self, audio_url: str) -> Path:
        """Resolve audio path (handle both absolute and relative paths)."""
        audio_path = Path(audio_url)
        if not audio_path.is_absolute():
            audio_path = self.audio_base_path / audio_path
        return audio_path
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a training sample processed through prepare_chatml_sample.
        
        Returns:
            Dictionary containing:
            - input_tokens: Input token sequence
            - label_tokens: Label token sequence  
            - audio_contents: List of AudioContent objects
            - speaker_id: Speaker identifier
            - audio_ids: List of encoded audio token sequences
        """
        try:
            sample = self.samples[idx]
            
            # Use existing prepare_chatml_sample function
            input_tokens, label_tokens, audio_contents, speaker_id = prepare_chatml_sample(
                sample, self.tokenizer
            )
            
            # Handle None returns from prepare_chatml_sample
            if input_tokens is None:
                print(f"⚠️ Failed to process sample {idx}, using empty tokens")
                input_tokens = []
                label_tokens = []
                audio_contents = []
                speaker_id = None
            
            # Process audio using existing audio_tokenizer
            audio_ids_list = []
            processed_audio_contents = []
            
            for audio_content in audio_contents:
                if audio_content and hasattr(audio_content, 'audio_url') and audio_content.audio_url:
                    try:
                        audio_path = self._resolve_audio_path(audio_content.audio_url)
                        
                        # Encode audio using existing audio tokenizer
                        audio_codes = self.audio_tokenizer.encode(str(audio_path))
                        audio_ids_list.append(audio_codes)
                        processed_audio_contents.append(audio_content)
                        
                    except Exception as e:
                        print(f"⚠️ Failed to encode audio {audio_content.audio_url}: {e}")
                        # Continue without this audio
                        continue
            
            return {
                'input_tokens': torch.tensor(input_tokens, dtype=torch.long) if input_tokens else torch.tensor([], dtype=torch.long),
                'label_tokens': torch.tensor(label_tokens, dtype=torch.long) if label_tokens else torch.tensor([], dtype=torch.long),
                'audio_contents': processed_audio_contents,
                'speaker_id': speaker_id,
                'audio_ids': audio_ids_list,
                'sample_idx': idx,
            }
        
        except Exception as e:
            print(f"⚠️ Error processing sample {idx}: {e}")
            # Return empty sample to prevent training failure
            return {
                'input_tokens': torch.tensor([], dtype=torch.long),
                'label_tokens': torch.tensor([], dtype=torch.long),
                'audio_contents': [],
                'speaker_id': None,
                'audio_ids': [],
                'sample_idx': idx,
            }


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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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


if __name__ == "__main__":
    # Example usage: Create sample data
    sample_data_path = "data/sample_training_data.json"
    create_sample_data(sample_data_path, num_samples=5)
    
    # Validate the created data
    validate_dataset_format(sample_data_path)