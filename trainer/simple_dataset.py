#!/usr/bin/env python3
"""
Inference-Aligned Dataset Implementation

This dataset uses the EXACT same data processing logic as arb_inference.py
to ensure perfect compatibility between training and inference.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from loguru import logger

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available - creating mock torch for basic functionality")
    # Create minimal torch mock for structure compatibility
    class MockTorch:
        @staticmethod
        def tensor(data, dtype=None):
            return data
        
        class dtype:
            long = "long"
            float32 = "float32"
    
    torch = MockTorch()
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers not available - training will be limited")
    TRANSFORMERS_AVAILABLE = False

# Critical imports from boson_multimodal - EXACT same as inference
from boson_multimodal.dataset.chatml_dataset import (
    ChatMLDatasetSample,
    prepare_chatml_sample,
)
from boson_multimodal.data_types import Message, TextContent, AudioContent


class InferenceAlignedDataset(torch.utils.data.Dataset):
    """
    Dataset that processes data EXACTLY like the inference pipeline.
    
    Uses the same boson_multimodal prepare_chatml_sample function
    and handles the exact ChatML format from your sample data.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        audio_tokenizer=None,
        audio_base_path: str = "",
        create_missing_audio: bool = True
    ):
        """
        Initialize with exact same approach as inference.
        
        Args:
            data_path: Path to ChatML JSON file
            tokenizer: Text tokenizer (same as inference)
            audio_tokenizer: Audio tokenizer (same as inference) 
            audio_base_path: Base path for resolving audio files
            create_missing_audio: Whether to create dummy audio for missing files
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.audio_base_path = Path(audio_base_path) if audio_base_path else Path.cwd()
        self.create_missing_audio = create_missing_audio
        
        # Load samples using exact same format as your data
        self.samples = self._load_samples()
        
        # Create audio files if missing (since the sample data references non-existent files)
        if self.create_missing_audio:
            self._ensure_audio_files_exist()
        
        logger.info(f"📂 Loaded {len(self.samples)} samples from {data_path}")
    
    def _load_samples(self) -> List[Dict]:
        """Load samples in the exact format from your sample data."""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both list and single sample formats
            if not isinstance(data, list):
                data = [data]
            
            logger.info(f"✅ Loaded {len(data)} samples with exact format match")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load samples: {e}")
            raise
    
    def _ensure_audio_files_exist(self):
        """Create dummy audio files for missing references in sample data."""
        audio_files_created = set()
        
        for sample in self.samples:
            # Handle the complex mixed content format
            audio_urls = self._extract_audio_urls_from_sample(sample)
            
            for audio_url in audio_urls:
                audio_path = self._resolve_audio_path(audio_url)
                
                if not audio_path.exists() and str(audio_path) not in audio_files_created:
                    self._create_dummy_audio(audio_path)
                    audio_files_created.add(str(audio_path))
        
        if audio_files_created:
            logger.info(f"🎵 Created {len(audio_files_created)} dummy audio files for missing references")
    
    def _extract_audio_urls_from_sample(self, sample: Dict[str, Any]) -> List[str]:
        """Extract all audio URLs from a sample, handling complex mixed content."""
        audio_urls = []
        
        for message in sample["messages"]:
            content = message["content"]
            
            if isinstance(content, list):
                # Handle mixed content arrays
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "audio":
                        audio_url = item.get("audio_url")
                        if audio_url:
                            audio_urls.append(audio_url)
            
            elif isinstance(content, dict) and content.get("type") == "audio":
                # Handle direct audio content
                audio_url = content.get("audio_url")
                if audio_url:
                    audio_urls.append(audio_url)
        
        return audio_urls
    
    def _resolve_audio_path(self, audio_url: str) -> Path:
        """Resolve audio path exactly like inference pipeline."""
        audio_path = Path(audio_url)
        if not audio_path.is_absolute():
            audio_path = self.audio_base_path / audio_path
        return audio_path
    
    def _create_dummy_audio(self, audio_path: Path):
        """Create a realistic dummy audio file."""
        try:
            # Create directory if needed
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create 2-second dummy audio with slight variation to simulate real speech
            import numpy as np
            import soundfile as sf
            
            sample_rate = 24000  # Match expected sample rate
            duration = 2.0
            samples = int(sample_rate * duration)
            
            # Generate realistic audio-like waveform (not pure noise)
            t = np.linspace(0, duration, samples)
            # Mix of frequencies to simulate speech-like characteristics
            waveform = (
                0.1 * np.sin(2 * np.pi * 200 * t) +  # Low frequency component
                0.05 * np.sin(2 * np.pi * 800 * t) +  # Mid frequency
                0.02 * np.random.normal(0, 1, samples)  # Slight noise
            )
            
            # Apply envelope to make it more speech-like
            envelope = np.exp(-t * 0.5)  # Decay envelope
            waveform = waveform * envelope
            
            # Normalize
            waveform = waveform.astype(np.float32)
            waveform = waveform / np.max(np.abs(waveform)) * 0.5  # Prevent clipping
            
            # Save as WAV
            sf.write(str(audio_path), waveform, sample_rate)
            logger.debug(f"Created dummy audio: {audio_path}")
            
        except Exception as e:
            logger.error(f"Failed to create dummy audio {audio_path}: {e}")
            raise
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> ChatMLDatasetSample:
        """Process sample using EXACT same logic as inference pipeline."""
        sample = self.samples[idx]
        
        try:
            # Convert to the exact format expected by boson_multimodal prepare_chatml_sample
            # Handle the complex mixed content structure from your actual data
            converted_sample = self._convert_to_boson_format(sample)
            
            # Debug logging
            logger.debug(f"Sample {idx}: Original messages: {len(sample['messages'])}")
            logger.debug(f"Sample {idx}: Converted messages: {len(converted_sample['messages'])}")
            
            # Check for audio content in converted sample
            audio_count = 0
            for msg in converted_sample['messages']:
                if isinstance(msg['content'], dict) and msg['content'].get('type') == 'audio':
                    audio_count += 1
            logger.debug(f"Sample {idx}: Found {audio_count} audio messages after conversion")
            
            # Use boson_multimodal prepare_chatml_sample - EXACT same as inference
            input_tokens, label_tokens, audio_contents, speaker_id = prepare_chatml_sample(
                converted_sample, self.tokenizer
            )
            
            if input_tokens is None:
                logger.error(f"Failed to process sample {idx} with prepare_chatml_sample")
                return self._create_fallback_sample()
            
            logger.debug(f"Sample {idx}: prepare_chatml_sample returned {len(audio_contents)} audio contents")
            
            # Process audio contents using same logic as inference
            audio_data = self._process_audio_contents(audio_contents)
            
            # Create ChatMLDatasetSample in the exact format expected by collator
            return ChatMLDatasetSample(
                input_ids=torch.tensor(input_tokens, dtype=torch.long),
                label_ids=torch.tensor(label_tokens, dtype=torch.long) if label_tokens else None,
                audio_ids_concat=audio_data['audio_ids_concat'],
                audio_ids_start=audio_data['audio_ids_start'],
                audio_waveforms_concat=audio_data['audio_waveforms_concat'],
                audio_waveforms_start=audio_data['audio_waveforms_start'],
                audio_sample_rate=audio_data['audio_sample_rate'],
                audio_speaker_indices=audio_data['audio_speaker_indices'],
            )
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._create_fallback_sample()
    
    def _convert_to_boson_format(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Convert complex ChatML format to boson_multimodal expected format."""
        # For your data format, we need to restructure it to match the expected pattern
        # Your data has: user message with mixed content (text + audio) + assistant with mixed content
        # We need to convert this to: system + user + assistant (audio) + user + assistant (audio)
        
        messages = sample["messages"]
        converted_messages = []
        
        # First, find the system message
        system_message = None
        user_with_mixed_content = None
        assistant_with_mixed_content = None
        
        for message in messages:
            if message["role"] == "system":
                system_message = message
            elif message["role"] == "user" and isinstance(message["content"], list):
                user_with_mixed_content = message
            elif message["role"] == "assistant" and isinstance(message["content"], list):
                assistant_with_mixed_content = message
        
        # Extract components from the mixed content user message
        ref_text = None
        ref_audio_url = None
        target_text = None
        
        if user_with_mixed_content:
            for item in user_with_mixed_content["content"]:
                if item["type"] == "text":
                    text_content = item["text"]
                    if "Please generate speech for given text" in text_content:
                        # This contains the target text
                        parts = text_content.split("Please generate speech for given text in reference audio's voice:")
                        if len(parts) > 1:
                            target_text = parts[1].strip()
                        if len(parts) > 0 and parts[0].strip():
                            ref_text = parts[0].strip()
                    else:
                        # This is the reference text
                        ref_text = text_content
                elif item["type"] == "audio":
                    ref_audio_url = item["audio_url"]
        
        # Extract target audio from assistant message
        target_audio_url = None
        assistant_text = None
        
        if assistant_with_mixed_content:
            for item in assistant_with_mixed_content["content"]:
                if item["type"] == "text":
                    assistant_text = item["text"]
                elif item["type"] == "audio":
                    target_audio_url = item["audio_url"]
        
        # Create the converted messages in the expected format
        converted_messages = []
        
        # 1. System message
        if system_message:
            converted_messages.append({
                "role": "system",
                "content": system_message["content"]
            })
        
        # 2. User message with reference text
        if ref_text:
            converted_messages.append({
                "role": "user",
                "content": ref_text
            })
        
        # 3. Assistant message with reference audio
        if ref_audio_url:
            converted_messages.append({
                "role": "assistant",
                "content": {
                    "type": "audio",
                    "audio_url": ref_audio_url
                }
            })
        
        # 4. User message with target text
        if target_text:
            converted_messages.append({
                "role": "user",
                "content": target_text
            })
        
        # 5. Assistant message with target audio (for training)
        if target_audio_url:
            converted_messages.append({
                "role": "assistant",
                "content": {
                    "type": "audio",
                    "audio_url": target_audio_url
                }
            })
        elif assistant_text:
            # If no target audio, use the text content
            converted_messages.append({
                "role": "assistant",
                "content": assistant_text
            })
        
        return {
            "messages": converted_messages,
            "speaker": sample.get("speaker"),
            "start_index": sample.get("start_index", 1)  # Start from first assistant response
        }
    
    def _process_audio_contents(self, audio_contents: List) -> Dict[str, torch.Tensor]:
        """Process audio contents exactly like inference pipeline."""
        audio_tokens_list = []
        audio_waveforms = []
        audio_sample_rates = []
        
        for audio_content in audio_contents:
            if audio_content and hasattr(audio_content, 'audio_url'):
                audio_path = self._resolve_audio_path(audio_content.audio_url)
                
                # Process audio tokens (DAC encoding)
                if self.audio_tokenizer:
                    try:
                        audio_tokens = self.audio_tokenizer.encode(str(audio_path))
                        audio_tokens_list.append(audio_tokens)
                    except Exception as e:
                        logger.warning(f"Failed to encode audio {audio_path}: {e}")
                        # Create dummy tokens matching expected shape
                        audio_tokens_list.append(torch.zeros((12, 50), dtype=torch.long))
                
                # Process waveform for Whisper conditioning
                try:
                    import torchaudio
                    import torchaudio.transforms as T
                    
                    waveform, sr = torchaudio.load(str(audio_path))
                    
                    # Convert to mono if stereo
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    
                    # Resample to 16kHz for Whisper (EXACT same as inference)
                    target_sr = 16000
                    if sr != target_sr:
                        resampler = T.Resample(sr, target_sr)
                        waveform = resampler(waveform)
                        sr = target_sr
                    
                    audio_waveforms.append(waveform.squeeze(0))
                    audio_sample_rates.append(sr)
                    
                except Exception as e:
                    logger.warning(f"Failed to load audio {audio_path}: {e}")
                    # Create dummy waveform
                    audio_waveforms.append(torch.zeros(16000))  # 1 second at 16kHz
                    audio_sample_rates.append(16000)
        
        # Combine audio data
        if audio_tokens_list:
            audio_ids_concat = torch.cat(audio_tokens_list, dim=1)
            audio_ids_start = torch.tensor([0, audio_ids_concat.size(1)], dtype=torch.long)
        else:
            audio_ids_concat = torch.zeros((12, 0), dtype=torch.long)
            audio_ids_start = torch.tensor([], dtype=torch.long)
        
        if audio_waveforms:
            audio_waveforms_concat = torch.cat(audio_waveforms, dim=0)
            audio_waveforms_start = torch.tensor([0, len(audio_waveforms_concat)], dtype=torch.long)
            audio_sample_rate_tensor = torch.tensor(audio_sample_rates, dtype=torch.float32)
            audio_speaker_indices = torch.tensor([0], dtype=torch.long)
        else:
            audio_waveforms_concat = torch.zeros(0, dtype=torch.float32)
            audio_waveforms_start = torch.tensor([], dtype=torch.long)
            audio_sample_rate_tensor = torch.tensor([], dtype=torch.float32)
            audio_speaker_indices = torch.tensor([], dtype=torch.long)
        
        return {
            'audio_ids_concat': audio_ids_concat,
            'audio_ids_start': audio_ids_start,
            'audio_waveforms_concat': audio_waveforms_concat,
            'audio_waveforms_start': audio_waveforms_start,
            'audio_sample_rate': audio_sample_rate_tensor,
            'audio_speaker_indices': audio_speaker_indices,
        }
    
    def _create_fallback_sample(self) -> ChatMLDatasetSample:
        """Create fallback sample to prevent training crash."""
        return ChatMLDatasetSample(
            input_ids=torch.tensor([1, 2, 3], dtype=torch.long),
            label_ids=torch.tensor([-100, -100, -100], dtype=torch.long),
            audio_ids_concat=torch.zeros((12, 0), dtype=torch.long),
            audio_ids_start=torch.tensor([], dtype=torch.long),
            audio_waveforms_concat=torch.zeros(0, dtype=torch.float32),
            audio_waveforms_start=torch.tensor([], dtype=torch.long),
            audio_sample_rate=torch.tensor([], dtype=torch.float32),
            audio_speaker_indices=torch.tensor([], dtype=torch.long),
        )


# Keep the original SimpleChatMLDataset name for backward compatibility
SimpleChatMLDataset = InferenceAlignedDataset


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
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created {num_samples} sample data entries at {output_path}")