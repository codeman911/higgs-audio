#!/usr/bin/env python3
"""
Arabic ChatML Dataset for LoRA Fine-tuning

This module implements a dataset class for loading and processing Arabic ChatML data
for LoRA fine-tuning of Higgs Audio v2 model for zero-shot voice cloning.
"""

import json
import os
import torch
import librosa
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import Dataset
from loguru import logger

# Higgs Audio imports
from boson_multimodal.dataset.chatml_dataset import (
    ChatMLDatasetSample, 
    prepare_chatml_sample,
    ChatMLSample
)
from boson_multimodal.data_types import Message, AudioContent, TextContent
from boson_multimodal.audio_processing.higgs_audio_tokenizer import HiggsAudioTokenizer


def normalize_arabic_text(text: str) -> str:
    """
    Normalize Arabic text for better processing.
    
    Args:
        text: Raw Arabic text
        
    Returns:
        Normalized Arabic text
    """
    import re
    
    # Remove diacritics (tashkeel)
    diacritics = re.compile(r'[\u064B-\u065F\u0670\u0640]')
    text = diacritics.sub('', text)
    
    # Normalize Arabic punctuation to English equivalents
    arabic_punctuation = {
        '،': ',',    # Arabic comma
        '؟': '?',    # Arabic question mark
        '؛': ';',    # Arabic semicolon
        '٪': '%',    # Arabic percent
        '٫': ',',    # Arabic decimal separator
        '٬': ',',    # Arabic thousands separator
    }
    
    for ar_punct, en_punct in arabic_punctuation.items():
        text = text.replace(ar_punct, en_punct)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Handle parentheses and brackets commonly used in Arabic
    text = text.replace('(', ' ').replace(')', ' ')
    text = text.replace('[', ' ').replace(']', ' ')
    
    # Ensure proper sentence endings
    if not any(text.endswith(c) for c in ['.', '!', '?', ',', ';']):
        text += '.'
    
    return text


def preprocess_audio_file(audio_path: str, target_sr: int = 16000) -> Optional[torch.Tensor]:
    """
    Preprocess audio file for Higgs Audio tokenizer.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate
        
    Returns:
        Preprocessed audio tensor or None if failed
    """
    try:
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return None
        
        # Load audio
        waveform, sr = librosa.load(audio_path, sr=None, mono=True)
        
        # Resample if needed
        if sr != target_sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        
        # Normalize audio
        if len(waveform) > 0:
            # Normalize to [-1, 1] range
            max_val = np.abs(waveform).max()
            if max_val > 0:
                waveform = waveform / max_val
        
        return torch.tensor(waveform, dtype=torch.float32)
        
    except Exception as e:
        logger.error(f"Error preprocessing audio {audio_path}: {e}")
        return None


@dataclass
class ArabicChatMLSample:
    """Data structure for Arabic ChatML training sample."""
    
    sample_id: str
    ref_audio_path: str
    ref_text: str  # Arabic reference text
    target_text: str  # Arabic target text  
    target_audio_path: str
    speaker_id: str
    duration: Optional[float] = None
    misc: Optional[Dict[str, Any]] = None


class ArabicChatMLDataset(Dataset):
    """
    Dataset for Arabic ChatML data for LoRA fine-tuning.
    
    Loads and processes ChatML format data for training Higgs Audio
    on Arabic zero-shot voice cloning task.
    """
    
    def __init__(
        self,
        chatml_file: str,
        audio_tokenizer: Union[str, HiggsAudioTokenizer],
        text_tokenizer,
        audio_base_path: Optional[str] = None,
        max_audio_length: float = 30.0,  # Maximum audio length in seconds
        normalize_text: bool = True,
        validate_files: bool = True,
    ):
        """
        Initialize Arabic ChatML dataset.
        
        Args:
            chatml_file: Path to ChatML JSON file
            audio_tokenizer: Audio tokenizer instance or path
            text_tokenizer: Text tokenizer for Arabic text
            audio_base_path: Base path for resolving relative audio paths
            max_audio_length: Maximum audio length to include
            normalize_text: Whether to normalize Arabic text
            validate_files: Whether to validate audio files exist
        """
        self.chatml_file = chatml_file
        self.text_tokenizer = text_tokenizer
        self.audio_base_path = audio_base_path or os.path.dirname(chatml_file)
        self.max_audio_length = max_audio_length
        self.normalize_text = normalize_text
        self.validate_files = validate_files
        
        # Load audio tokenizer
        if isinstance(audio_tokenizer, str):
            from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
            self.audio_tokenizer = load_higgs_audio_tokenizer(audio_tokenizer)
        else:
            self.audio_tokenizer = audio_tokenizer
        
        # Load and parse ChatML data
        self.samples = self._load_chatml_data()
        
        logger.info(f"Loaded {len(self.samples)} Arabic ChatML samples")
    
    def _load_chatml_data(self) -> List[ArabicChatMLSample]:
        """Load and parse ChatML data file."""
        logger.info(f"Loading ChatML data from {self.chatml_file}")
        
        with open(self.chatml_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            data = [data]
        
        samples = []
        for i, item in enumerate(data):
            try:
                sample = self._parse_chatml_item(item, i)
                if sample and self._validate_sample(sample):
                    samples.append(sample)
                else:
                    logger.warning(f"Skipping invalid sample {i}")
            except Exception as e:
                logger.error(f"Error parsing sample {i}: {e}")
                continue
        
        return samples
    
    def _parse_chatml_item(self, item: Dict[str, Any], index: int) -> Optional[ArabicChatMLSample]:
        """Parse a single ChatML item into ArabicChatMLSample."""
        try:
            messages = item.get("messages", [])
            speaker_id = item.get("speaker", f"speaker_{index}")
            misc = item.get("misc", {})
            
            # Extract reference and target information
            ref_audio_path = None
            ref_text = None
            target_text = None
            target_audio_path = None
            duration = misc.get("duration")
            
            for message in messages:
                role = message["role"]
                content = message["content"]
                
                if role == "user" and isinstance(content, list):
                    # Look for reference text and audio, target text
                    text_parts = []
                    for item_content in content:
                        if item_content["type"] == "text":
                            text_parts.append(item_content["text"])
                        elif item_content["type"] == "audio" and ref_audio_path is None:
                            ref_audio_path = item_content["audio_url"]
                    
                    if len(text_parts) >= 2:
                        ref_text = text_parts[0]
                        # Extract target text from instruction
                        for text_part in text_parts[1:]:
                            if "Please generate speech" in text_part:
                                target_text = text_part.split(":")[-1].strip()
                                break
                        if target_text is None:
                            target_text = text_parts[-1]
                
                elif role == "assistant" and isinstance(content, list):
                    # Look for target audio
                    for item_content in content:
                        if item_content["type"] == "audio":
                            target_audio_path = item_content["audio_url"]
                            break
            
            if not all([ref_audio_path, ref_text, target_text, target_audio_path]):
                logger.warning(f"Missing required components in sample {index}")
                return None
            
            # Resolve audio paths
            ref_audio_path = self._resolve_audio_path(ref_audio_path)
            target_audio_path = self._resolve_audio_path(target_audio_path)
            
            return ArabicChatMLSample(
                sample_id=f"sample_{index}",
                ref_audio_path=ref_audio_path,
                ref_text=ref_text,
                target_text=target_text,
                target_audio_path=target_audio_path,
                speaker_id=speaker_id,
                duration=duration,
                misc=misc
            )
            
        except Exception as e:
            logger.error(f"Error parsing ChatML item {index}: {e}")
            return None
    
    def _resolve_audio_path(self, audio_path: str) -> str:
        """Resolve relative audio paths."""
        if os.path.isabs(audio_path):
            return audio_path
        return os.path.join(self.audio_base_path, audio_path)
    
    def _validate_sample(self, sample: ArabicChatMLSample) -> bool:
        """Validate a sample has required files and properties."""
        if self.validate_files:
            if not os.path.exists(sample.ref_audio_path):
                logger.warning(f"Reference audio not found: {sample.ref_audio_path}")
                return False
            if not os.path.exists(sample.target_audio_path):
                logger.warning(f"Target audio not found: {sample.target_audio_path}")
                return False
        
        # Check audio duration if specified
        if sample.duration and self.max_audio_length:
            if sample.duration > self.max_audio_length:
                logger.warning(f"Audio too long ({sample.duration}s): {sample.sample_id}")
                return False
        
        return True
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> ChatMLDatasetSample:
        """
        Get a sample and convert it to ChatMLDatasetSample format.
        
        Args:
            idx: Sample index
            
        Returns:
            ChatMLDatasetSample for training
        """
        sample = self.samples[idx]
        
        try:
            # Normalize Arabic text if requested
            ref_text = sample.ref_text
            target_text = sample.target_text
            
            if self.normalize_text:
                ref_text = normalize_arabic_text(ref_text)
                target_text = normalize_arabic_text(target_text)
            
            # Create ChatML sample structure for training
            chatml_sample = self._create_training_chatml_sample(
                sample, ref_text, target_text
            )
            
            # Prepare tokens using existing ChatML processing
            input_tokens, label_tokens, audio_contents, speaker_id = prepare_chatml_sample(
                chatml_sample, self.text_tokenizer
            )
            
            if input_tokens is None:
                logger.error(f"Failed to prepare tokens for sample {idx}")
                # Return a minimal valid sample
                return self._create_empty_sample()
            
            # Process audio files
            ref_audio_codes, ref_waveform = self._process_audio(sample.ref_audio_path)
            target_audio_codes, target_waveform = self._process_audio(sample.target_audio_path)
            
            if ref_audio_codes is None or target_audio_codes is None:
                logger.error(f"Failed to process audio for sample {idx}")
                return self._create_empty_sample()
            
            # Concatenate audio data
            audio_ids_concat = torch.cat([ref_audio_codes, target_audio_codes], dim=1)
            audio_ids_start = torch.tensor([0, ref_audio_codes.shape[1]], dtype=torch.long)
            
            audio_waveforms_concat = torch.cat([ref_waveform, target_waveform], dim=0)
            audio_waveforms_start = torch.tensor([0, len(ref_waveform)], dtype=torch.long)
            audio_sample_rate = torch.tensor([16000, 16000])  # Both audios at 16kHz
            
            return ChatMLDatasetSample(
                input_ids=torch.LongTensor(input_tokens),
                label_ids=torch.LongTensor(label_tokens),
                audio_ids_concat=audio_ids_concat,
                audio_ids_start=audio_ids_start,
                audio_waveforms_concat=audio_waveforms_concat,
                audio_waveforms_start=audio_waveforms_start,
                audio_sample_rate=audio_sample_rate,
                audio_speaker_indices=torch.tensor([0, 0], dtype=torch.long),  # Same speaker
                audio_label_ids_concat=target_audio_codes,  # Use target as label
            )
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            return self._create_empty_sample()
    
    def _create_training_chatml_sample(
        self, 
        sample: ArabicChatMLSample, 
        ref_text: str, 
        target_text: str
    ) -> ChatMLSample:
        """Create ChatML sample structure for training."""
        
        # System message
        system_message = Message(
            role="system",
            content="You are a helpful assistant capable of generating speech in the voice of the provided reference audio."
        )
        
        # User message with reference text
        user_ref_message = Message(
            role="user",
            content=ref_text
        )
        
        # Assistant message with reference audio
        assistant_ref_message = Message(
            role="assistant", 
            content=AudioContent(audio_url=sample.ref_audio_path)
        )
        
        # User message with target text
        user_target_message = Message(
            role="user",
            content=target_text
        )
        
        # Assistant message with target audio (for training labels)
        assistant_target_message = Message(
            role="assistant",
            content=AudioContent(audio_url=sample.target_audio_path)
        )
        
        return ChatMLSample(
            messages=[
                system_message,
                user_ref_message, 
                assistant_ref_message,
                user_target_message,
                assistant_target_message
            ],
            start_index=2,  # Start training from the target generation
            speaker=sample.speaker_id,
            misc=sample.misc
        )
    
    def _process_audio(self, audio_path: str) -> tuple:
        """
        Process audio file into tokens and waveform.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_codes, waveform)
        """
        try:
            # Preprocess audio
            waveform = preprocess_audio_file(audio_path, target_sr=16000)
            if waveform is None:
                return None, None
            
            # Encode to audio tokens
            audio_codes = self.audio_tokenizer.encode(audio_path)
            
            return audio_codes, waveform
            
        except Exception as e:
            logger.error(f"Error processing audio {audio_path}: {e}")
            return None, None
    
    def _create_empty_sample(self) -> ChatMLDatasetSample:
        """Create an empty sample for error cases."""
        return ChatMLDatasetSample(
            input_ids=torch.LongTensor([0]),
            label_ids=torch.LongTensor([-100]),
            audio_ids_concat=torch.zeros((12, 0), dtype=torch.long),  # 12 codebooks
            audio_ids_start=torch.zeros(0, dtype=torch.long),
            audio_waveforms_concat=torch.zeros(0),
            audio_waveforms_start=torch.zeros(0, dtype=torch.long),
            audio_sample_rate=torch.zeros(0),
            audio_speaker_indices=torch.zeros(0, dtype=torch.long),
        )
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get information about a specific sample."""
        sample = self.samples[idx]
        return {
            "sample_id": sample.sample_id,
            "speaker_id": sample.speaker_id,
            "ref_text": sample.ref_text,
            "target_text": sample.target_text,
            "ref_audio": sample.ref_audio_path,
            "target_audio": sample.target_audio_path,
            "duration": sample.duration,
        }
    
    def get_speaker_stats(self) -> Dict[str, int]:
        """Get statistics about speakers in the dataset."""
        speaker_counts = {}
        for sample in self.samples:
            speaker_id = sample.speaker_id
            speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
        return speaker_counts
    
    def filter_by_duration(self, min_duration: float = 1.0, max_duration: float = 30.0):
        """Filter samples by audio duration."""
        filtered_samples = []
        for sample in self.samples:
            if sample.duration:
                if min_duration <= sample.duration <= max_duration:
                    filtered_samples.append(sample)
            else:
                # Keep samples without duration info
                filtered_samples.append(sample)
        
        original_count = len(self.samples)
        self.samples = filtered_samples
        logger.info(f"Filtered dataset: {len(self.samples)}/{original_count} samples remaining")
    
    def split_dataset(self, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42):
        """
        Split dataset into train/val/test splits.
        
        Args:
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set  
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        import random
        
        # Set seed for reproducibility
        random.seed(seed)
        
        # Shuffle samples
        shuffled_samples = self.samples.copy()
        random.shuffle(shuffled_samples)
        
        # Calculate split indices
        total = len(shuffled_samples)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        # Create splits
        train_samples = shuffled_samples[:train_end]
        val_samples = shuffled_samples[train_end:val_end]
        test_samples = shuffled_samples[val_end:]
        
        # Create new dataset instances
        train_dataset = self._create_subset(train_samples)
        val_dataset = self._create_subset(val_samples)
        test_dataset = self._create_subset(test_samples)
        
        logger.info(f"Dataset split: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def _create_subset(self, samples: List[ArabicChatMLSample]) -> 'ArabicChatMLDataset':
        """Create a subset dataset with given samples."""
        subset = ArabicChatMLDataset.__new__(ArabicChatMLDataset)
        subset.chatml_file = self.chatml_file
        subset.text_tokenizer = self.text_tokenizer
        subset.audio_tokenizer = self.audio_tokenizer
        subset.audio_base_path = self.audio_base_path
        subset.max_audio_length = self.max_audio_length
        subset.normalize_text = self.normalize_text
        subset.validate_files = self.validate_files
        subset.samples = samples
        return subset


def create_arabic_chatml_datasets(
    chatml_file: str,
    audio_tokenizer: Union[str, HiggsAudioTokenizer],
    text_tokenizer,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    **kwargs
) -> tuple:
    """
    Convenience function to create train/val/test datasets.
    
    Args:
        chatml_file: Path to ChatML JSON file
        audio_tokenizer: Audio tokenizer instance or path
        text_tokenizer: Text tokenizer
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        **kwargs: Additional arguments for ArabicChatMLDataset
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Create full dataset
    full_dataset = ArabicChatMLDataset(
        chatml_file=chatml_file,
        audio_tokenizer=audio_tokenizer,
        text_tokenizer=text_tokenizer,
        **kwargs
    )
    
    # Split into train/val/test
    return full_dataset.split_dataset(train_ratio, val_ratio)


# Example usage and testing
if __name__ == "__main__":
    import sys
    from transformers import AutoTokenizer
    
    if len(sys.argv) < 2:
        print("Usage: python arabic_chatml_dataset.py <chatml_file>")
        sys.exit(1)
    
    chatml_file = sys.argv[1]
    
    # Load tokenizers (you would use actual model paths)
    print("Loading tokenizers...")
    text_tokenizer = AutoTokenizer.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")
    audio_tokenizer_path = "bosonai/higgs-audio-v2-tokenizer"
    
    # Create dataset
    print(f"Creating dataset from {chatml_file}")
    dataset = ArabicChatMLDataset(
        chatml_file=chatml_file,
        audio_tokenizer=audio_tokenizer_path,
        text_tokenizer=text_tokenizer,
        validate_files=False  # Skip file validation for testing
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Speaker stats: {dataset.get_speaker_stats()}")
    
    # Test first sample
    if len(dataset) > 0:
        print("\nTesting first sample...")
        sample = dataset[0]
        print(f"Input IDs shape: {sample.input_ids.shape}")
        print(f"Audio IDs shape: {sample.audio_ids_concat.shape}")
        print(f"Sample info: {dataset.get_sample_info(0)}")