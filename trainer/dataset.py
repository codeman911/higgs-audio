"""
Dataset implementation for Higgs-Audio LoRA training.

Follows EXACT patterns from arb_inference.py and generation.py:
- Proper ChatML message structure matching inference
- Dual audio processing (Whisper + DAC) following training pipeline
- Reference audio conditioning exactly like arb_inference.py 
- Teacher forcing training with proper label generation
- Robust error handling and validation

Reuses existing boson_multimodal components without modification:
- prepare_chatml_sample for data processing
- ChatMLDatasetSample for data structure  
- Existing audio tokenizer for audio encoding
"""

import json
import os
import sys
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from loguru import logger

# Add parent directory to path for boson_multimodal imports
current_dir = Path(__file__).parent  
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Conditional imports for ML dependencies
try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy Dataset class for utility functions
    class Dataset:
        pass

# Import existing boson_multimodal components (conditional)
try:
    from boson_multimodal.dataset.chatml_dataset import (
        prepare_chatml_sample,
        ChatMLDatasetSample,
    )
    from boson_multimodal.data_types import ChatMLSample, Message, AudioContent, TextContent
    BOSON_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Failed to import boson_multimodal components: {e}")
    logger.error(f"   Make sure you're running from the higgs-audio directory")
    BOSON_AVAILABLE = False
    
    # Create dummy classes to prevent import errors
    def prepare_chatml_sample(*args, **kwargs):
        return None, None, None, None
    
    class ChatMLDatasetSample:
        pass
    
    class ChatMLSample:
        pass
    
    class Message:
        pass
    
    class AudioContent:
        pass
    
    class TextContent:
        pass


class VoiceCloningDataset(Dataset):
    """
    Voice cloning dataset that EXACTLY matches arb_inference.py patterns.
    
    Processes ChatML format data for zero-shot voice cloning training using
    the same dual audio conditioning pipeline as the inference implementation.
    
    The data format follows arb_inference.py exactly:
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
        force_whisper_embed: bool = True,  # Match arb_inference.py
    ):
        """
        Initialize the voice cloning dataset.
        
        Args:
            data_path: Path to the JSON file containing training samples
            tokenizer: Text tokenizer from HiggsAudioModel
            audio_tokenizer: Audio tokenizer for encoding reference audio
            audio_base_path: Base path for resolving relative audio paths
            validate_audio_paths: Whether to validate audio file existence
            force_whisper_embed: Force Whisper embedding like arb_inference.py
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for dataset functionality. Install with: pip install torch")
        
        if not BOSON_AVAILABLE:
            raise ImportError("boson_multimodal is required for dataset functionality. Ensure it's in the Python path.")
        
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.audio_base_path = Path(audio_base_path) if audio_base_path else Path()
        self.force_whisper_embed = force_whisper_embed
        
        # Load training data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
        
        logger.info(f"📂 Loaded {len(self.samples)} training samples from {data_path}")
        
        # Validate and filter samples
        if validate_audio_paths:
            valid_samples = []
            for i, sample in enumerate(self.samples):
                if self._validate_sample(sample, i):
                    valid_samples.append(sample)
            
            logger.info(f"✅ {len(valid_samples)}/{len(self.samples)} samples passed validation")
            self.samples = valid_samples
        
        if len(self.samples) == 0:
            raise ValueError("No valid training samples found!")
    
    def _validate_sample(self, sample: Dict[str, Any], idx: int) -> bool:
        """Validate a single training sample following arb_inference.py patterns."""
        try:
            # Check required fields
            if 'messages' not in sample:
                logger.warning(f"⚠️ Sample {idx}: Missing 'messages' field")
                return False
            
            messages = sample['messages']
            if not isinstance(messages, list) or len(messages) < 3:
                logger.warning(f"⚠️ Sample {idx}: Invalid messages structure (need >= 3 messages)")
                return False
            
            # Extract components using arb_inference.py pattern
            ref_audio_path, ref_text, target_text, speaker_id = self._extract_sample_components(sample)
            
            if not all([ref_audio_path, ref_text, target_text]):
                logger.warning(f"⚠️ Sample {idx}: Missing required components")
                return False
            
            # Validate audio file exists
            audio_path = self._resolve_audio_path(ref_audio_path)
            if not audio_path.exists():
                logger.warning(f"⚠️ Sample {idx}: Audio file not found: {audio_path}")
                return False
            
            return True
        
        except Exception as e:
            logger.warning(f"⚠️ Sample {idx}: Validation error: {e}")
            return False
    
    def _extract_sample_components(self, sample: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        Extract components exactly like arb_inference.py process_chatml_sample method.
        
        Returns:
            Tuple of (ref_audio_path, ref_text, target_text, speaker_id)
        """
        try:
            messages = sample["messages"]
            
            ref_audio_path = None
            ref_text = None
            target_text = None
            speaker_id = sample.get("speaker", "unknown")
            
            # Parse messages following arb_inference.py pattern
            for message in messages:
                if message["role"] == "user":
                    content = message["content"]
                    if isinstance(content, list):
                        # Look for text and audio content
                        text_parts = []
                        for item in content:
                            if item["type"] == "text":
                                text_parts.append(item["text"])
                            elif item["type"] == "audio":
                                if ref_audio_path is None:  # First audio is reference
                                    ref_audio_path = item["audio_url"]
                        
                        if len(text_parts) >= 2:
                            ref_text = text_parts[0]  # First text is reference
                            # Look for target text in the format "Please generate speech for given text..."
                            for text_part in text_parts[1:]:
                                if "Please generate speech" in text_part:
                                    # Extract target text after the instruction
                                    target_text = text_part.split(":")[-1].strip()
                                    break
                            if target_text is None and len(text_parts) > 1:
                                target_text = text_parts[-1]  # Last text as fallback
                    elif isinstance(content, str):
                        # Simple string content - determine if it's reference or target
                        if ref_text is None:
                            ref_text = content
                        else:
                            target_text = content
                
                elif message["role"] == "assistant":
                    content = message["content"]
                    if isinstance(content, dict) and content.get("type") == "audio":
                        if ref_audio_path is None:
                            ref_audio_path = content["audio_url"]
            
            return ref_audio_path, ref_text, target_text, speaker_id
            
        except Exception as e:
            logger.error(f"Error extracting sample components: {e}")
            return None, None, None, None
    
    def _resolve_audio_path(self, audio_url: str) -> Path:
        """Resolve audio path (handle both absolute and relative paths)."""
        audio_path = Path(audio_url)
        if not audio_path.is_absolute():
            audio_path = self.audio_base_path / audio_path
        return audio_path
    
    def _create_inference_aligned_messages(
        self, 
        ref_text: str, 
        ref_audio_path: str, 
        target_text: str
    ) -> List[Message]:
        """
        Create messages following EXACT arb_inference.py patterns.
        
        This matches the _prepare_generation_context method in arb_inference.py.
        """
        messages = [
            # System message (concise like arb_inference.py)
            Message(
                role="system",
                content="Generate speech in the provided voice."
            ),
            # User message with reference text and audio token
            Message(
                role="user", 
                content=f"{ref_text} <|audio_bos|><|AUDIO|><|audio_eos|>"
            ),
            # Assistant message with reference audio
            Message(
                role="assistant",
                content=AudioContent(audio_url=ref_audio_path)
            ),
            # User message with target text
            Message(
                role="user",
                content=target_text
            )
        ]
        
        return messages
    
    def _process_reference_audio_dual_pathway(
        self, 
        ref_audio_path: str
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
        """
        Process reference audio through dual pathway exactly like arb_inference.py.
        
        Returns:
            Tuple of (ref_waveform, audio_tokens, sample_rate)
        """
        try:
            # Load audio file
            waveform, sr = torchaudio.load(ref_audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample to 16kHz for Whisper (matching arb_inference.py)
            target_sr = 16000
            if sr != target_sr:
                resampler = T.Resample(sr, target_sr)
                waveform_16k = resampler(waveform)
            else:
                waveform_16k = waveform
            
            # Encode audio tokens using DAC tokenizer
            audio_tokens = self.audio_tokenizer.encode(ref_audio_path)
            
            return waveform_16k.squeeze(0), audio_tokens, target_sr
        
        except Exception as e:
            logger.warning(f"⚠️ Failed to process reference audio {ref_audio_path}: {e}")
            return None, None, 16000
    
    def _create_training_sample(
        self,
        input_tokens: List[int],
        label_tokens: List[int],
        audio_tokens: Optional[torch.Tensor],
        ref_waveform: Optional[torch.Tensor],
        ref_sample_rate: int,
        sample_idx: int
    ) -> ChatMLDatasetSample:
        """
        Create ChatMLDatasetSample for training with proper teacher forcing setup.
        
        This method creates the sample structure needed for teacher forcing training
        while maintaining compatibility with the collator used in inference.
        """
        # Process DAC tokens
        if audio_tokens is not None:
            audio_ids_concat = audio_tokens.cpu()
            audio_ids_start = torch.tensor([0], dtype=torch.long)
        else:
            # Empty tensors if no audio (fallback)
            audio_ids_concat = torch.zeros((12, 0), dtype=torch.long)  # 12 codebooks
            audio_ids_start = torch.tensor([], dtype=torch.long)
        
        # Create sample with conditional Whisper processing
        if self.force_whisper_embed and ref_waveform is not None:
            # Full pipeline mode: include waveforms for Whisper conditioning
            logger.debug(f"✅ Creating training sample {sample_idx} with Whisper conditioning: waveform shape={ref_waveform.shape}")
            
            sample = ChatMLDatasetSample(
                input_ids=torch.LongTensor(input_tokens),
                label_ids=torch.LongTensor(label_tokens) if label_tokens else None,
                audio_ids_concat=audio_ids_concat,
                audio_ids_start=audio_ids_start,
                audio_waveforms_concat=ref_waveform,
                audio_waveforms_start=torch.tensor([0], dtype=torch.long),
                audio_sample_rate=torch.tensor([ref_sample_rate], dtype=torch.float32),
                audio_speaker_indices=torch.tensor([0], dtype=torch.long),
                # Add target audio tokens as labels for teacher forcing
                audio_label_ids_concat=audio_ids_concat if audio_tokens is not None else None,
            )
        else:
            # DAC-only mode: follow serve_engine.py pattern
            logger.debug(f"✅ Creating training sample {sample_idx} in DAC-only mode: DAC tokens shape={audio_ids_concat.shape}")
            
            sample = ChatMLDatasetSample(
                input_ids=torch.LongTensor(input_tokens),
                label_ids=torch.LongTensor(label_tokens) if label_tokens else None,
                audio_ids_concat=audio_ids_concat,
                audio_ids_start=audio_ids_start,
                audio_waveforms_concat=torch.tensor([]),  # Empty tensor
                audio_waveforms_start=torch.tensor([], dtype=torch.long),
                audio_sample_rate=torch.tensor([], dtype=torch.float32),
                audio_speaker_indices=torch.tensor([], dtype=torch.long),
                # Add target audio tokens as labels for teacher forcing
                audio_label_ids_concat=audio_ids_concat if audio_tokens is not None else None,
            )
        
        return sample
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> ChatMLDatasetSample:
        """
        Get a training sample processed exactly like arb_inference.py.
        
        This method follows the same pipeline as the inference implementation
        to ensure perfect alignment between training and inference.
        """
        try:
            sample = self.samples[idx]
            
            # Extract components using arb_inference.py pattern
            ref_audio_path, ref_text, target_text, speaker_id = self._extract_sample_components(sample)
            
            if not all([ref_audio_path, ref_text, target_text]):
                logger.warning(f"⚠️ Failed to extract components from sample {idx}, using empty sample")
                return self._create_empty_sample(idx)
            
            # Resolve audio path
            resolved_audio_path = str(self._resolve_audio_path(ref_audio_path))
            
            # Create messages following arb_inference.py patterns
            messages = self._create_inference_aligned_messages(ref_text, resolved_audio_path, target_text)
            
            # Prepare tokens using existing prepare_chatml_sample function
            input_tokens, label_tokens, audio_contents, _ = prepare_chatml_sample(
                ChatMLSample(messages=messages), self.tokenizer
            )
            
            if input_tokens is None:
                logger.warning(f"⚠️ Failed to process tokens for sample {idx}, using empty sample")
                return self._create_empty_sample(idx)
            
            # Process reference audio through dual pathway
            ref_waveform, audio_tokens, ref_sample_rate = self._process_reference_audio_dual_pathway(resolved_audio_path)
            
            # Create training sample with teacher forcing setup
            training_sample = self._create_training_sample(
                input_tokens=input_tokens,
                label_tokens=label_tokens or [],
                audio_tokens=audio_tokens,
                ref_waveform=ref_waveform,
                ref_sample_rate=ref_sample_rate,
                sample_idx=idx
            )
            
            return training_sample
        
        except Exception as e:
            logger.error(f"⚠️ Error processing sample {idx}: {e}")
            return self._create_empty_sample(idx)
    
    def _create_empty_sample(self, idx: int) -> ChatMLDatasetSample:
        """Create an empty sample as fallback to prevent training failure."""
        logger.debug(f"Creating empty fallback sample for index {idx}")
        
        return ChatMLDatasetSample(
            input_ids=torch.tensor([], dtype=torch.long),
            label_ids=torch.tensor([], dtype=torch.long),
            audio_ids_concat=torch.zeros((12, 0), dtype=torch.long),
            audio_ids_start=torch.tensor([], dtype=torch.long),
            audio_waveforms_concat=torch.tensor([]),
            audio_waveforms_start=torch.tensor([], dtype=torch.long),
            audio_sample_rate=torch.tensor([], dtype=torch.float32),
            audio_speaker_indices=torch.tensor([], dtype=torch.long),
        )


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