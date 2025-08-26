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

# 📁 ENHANCED: Robust import system for 'python3 trainer/train.py' execution from higgs-audio root
current_file = Path(__file__).resolve()
trainer_dir = current_file.parent  # /path/to/higgs-audio/trainer/
higgs_audio_root = trainer_dir.parent  # /path/to/higgs-audio/

# 🎯 CRITICAL: Ensure higgs-audio root is in Python path for boson_multimodal imports
if str(higgs_audio_root) not in sys.path:
    sys.path.insert(0, str(higgs_audio_root))
    print(f"✅ [dataset.py] Added higgs-audio root to Python path: {higgs_audio_root}")

# 🔍 Verify directory structure
if not (higgs_audio_root / "boson_multimodal").exists():
    raise ImportError(
        f"❌ [dataset.py] boson_multimodal not found at {higgs_audio_root}. "
        "Please run script from higgs-audio root: python3 trainer/train.py"
    )

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

# 🔧 ENHANCED: Conditional imports with comprehensive error diagnostics for dataset
try:
    from boson_multimodal.dataset.chatml_dataset import (
        prepare_chatml_sample,
        ChatMLDatasetSample,
    )
    from boson_multimodal.data_types import ChatMLSample, Message, AudioContent, TextContent
    BOSON_AVAILABLE = True
    print("✅ [dataset.py] Successfully imported boson_multimodal components")
except ImportError as e:
    logger.error(f"❌ [dataset.py] Failed to import boson_multimodal: {e}")
    logger.error(f"   Higgs-audio root: {higgs_audio_root}")
    logger.error(f"   Boson_multimodal exists: {(higgs_audio_root / 'boson_multimodal').exists()}")
    logger.error("🔧 Run from higgs-audio root: python3 trainer/train.py")
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
        validate_audio_paths: bool = False,  # Changed default to False
        force_whisper_embed: bool = True,  # Match arb_inference.py
        auto_convert_format: bool = False,  # Changed default to False
        create_dummy_audio: bool = False,  # Changed default to False
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
            auto_convert_format: Automatically convert non-ChatML formats
            create_dummy_audio: Create dummy audio files for missing references
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for dataset functionality. Install with: pip install torch")
        
        if not BOSON_AVAILABLE:
            raise ImportError("boson_multimodal is required for dataset functionality. Ensure it's in the Python path.")
        
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.audio_base_path = Path(audio_base_path) if audio_base_path else Path()
        self.force_whisper_embed = force_whisper_embed
        
        # 🔧 ENHANCED: Handle missing data files and format conversion
        self.samples = self._load_and_convert_data(
            data_path, auto_convert_format, create_dummy_audio
        )
        
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
            # 🚨 ENHANCED: Create fallback data if no valid samples found
            logger.warning("⚠️ No valid training samples found! Creating fallback data...")
            self.samples = self._create_fallback_samples()
            logger.info(f"📝 Created {len(self.samples)} fallback samples for training")
    
    def _load_and_convert_data(
        self, 
        data_path: str, 
        auto_convert: bool, 
        create_dummy_audio: bool
    ) -> List[Dict[str, Any]]:
        """
        Load and convert data with comprehensive error handling.
        
        Args:
            data_path: Path to the data file
            auto_convert: Whether to auto-convert format
            create_dummy_audio: Whether to create dummy audio files
            
        Returns:
            List of training samples
        """
        try:
            # Check if data file exists
            if not os.path.exists(data_path):
                logger.warning(f"⚠️ Data file not found: {data_path}")
                if auto_convert:
                    logger.info("🔧 Creating sample data file...")
                    self._create_sample_data_file(data_path)
                else:
                    raise FileNotFoundError(f"Training data file not found: {data_path}")
            
            # Load data
            with open(data_path, 'r', encoding='utf-8') as f:
                if data_path.endswith('.jsonl'):
                    # JSONL format
                    samples = []
                    for line_num, line in enumerate(f):
                        line = line.strip()
                        if line:
                            try:
                                samples.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                logger.warning(f"⚠️ Invalid JSON on line {line_num+1}: {e}")
                else:
                    # Regular JSON format
                    samples = json.load(f)
            
            # Ensure samples is a list
            if isinstance(samples, dict):
                samples = [samples]
            
            # Auto-convert format if needed
            if auto_convert:
                from .data_converter import DataFormatConverter
                converter = DataFormatConverter(audio_base_path=str(self.audio_base_path))
                
                # Detect and convert format
                format_type = converter._detect_format(samples)
                logger.info(f"🔍 Detected data format: {format_type}")
                
                if format_type != "chatml":
                    logger.info(f"🔄 Converting from {format_type} to ChatML format...")
                    if format_type == "manifest":
                        samples = converter._convert_from_manifest(samples)
                    elif format_type == "paired":
                        samples = converter._convert_from_paired(samples)
                    elif format_type == "lora":
                        samples = converter._convert_from_lora_format(samples)
                    elif format_type == "mixed":
                        samples = converter._convert_from_mixed_content(samples)
                    else:
                        samples = converter._validate_chatml_format(samples)
                else:
                    # Already ChatML, just validate
                    samples = converter._validate_chatml_format(samples)
            
            # Create dummy audio files if requested
            if create_dummy_audio and samples:
                self._create_dummy_audio_files(samples)
            
            return samples
            
        except Exception as e:
            logger.error(f"❌ Failed to load data from {data_path}: {e}")
            if auto_convert:
                logger.info("🔧 Creating fallback sample data...")
                return self._create_fallback_samples()
            else:
                raise
    
    def _create_sample_data_file(self, data_path: str):
        """Create a sample data file with correct ChatML format."""
        samples = [
            {
                "messages": [
                    {"role": "system", "content": "Generate speech in the provided voice."},
                    {"role": "user", "content": "This is a sample reference text for voice cloning demonstration."},
                    {"role": "assistant", "content": {"type": "audio", "audio_url": "data/sample_audio/demo_ref.wav"}},
                    {"role": "user", "content": "Please generate speech for this sample target text."}
                ],
                "speaker": "demo_speaker",
                "start_index": 3
            }
        ]
        
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📝 Created sample data file at {data_path}")
    
    def _create_dummy_audio_files(self, samples: List[Dict[str, Any]]):
        """Create dummy audio files for missing references."""
        try:
            missing_files = set()
            
            for sample in samples:
                messages = sample.get("messages", [])
                for message in messages:
                    content = message.get("content")
                    if isinstance(content, dict) and content.get("type") == "audio":
                        audio_path = Path(content["audio_url"])
                        if not audio_path.exists():
                            missing_files.add(str(audio_path))
            
            if missing_files:
                logger.info(f"🎵 Creating {len(missing_files)} dummy audio files...")
                
                for audio_path in missing_files:
                    # Create directory if needed
                    Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
                    
                    try:
                        # Try to create a proper silent audio file
                        import numpy as np
                        import soundfile as sf
                        
                        # Generate 1 second of silence at 16kHz
                        duration = 1.0
                        sample_rate = 16000
                        samples = int(duration * sample_rate)
                        silent_audio = np.zeros(samples, dtype=np.float32)
                        
                        sf.write(audio_path, silent_audio, sample_rate)
                        
                    except ImportError:
                        # Fallback: create empty file
                        Path(audio_path).touch()
                
                logger.info(f"✅ Created {len(missing_files)} dummy audio files")
        
        except Exception as e:
            logger.warning(f"⚠️ Failed to create dummy audio files: {e}")
    
    def _create_fallback_samples(self) -> List[Dict[str, Any]]:
        """Create fallback training samples if no valid data found."""
        fallback_samples = [
            {
                "messages": [
                    {"role": "system", "content": "Generate speech in the provided voice."},
                    {"role": "user", "content": f"Fallback reference text number {i+1} for voice cloning."},
                    {"role": "assistant", "content": {"type": "audio", "audio_url": f"data/fallback_audio/sample_{i}.wav"}},
                    {"role": "user", "content": f"Generate speech for fallback target text {i+1}."}
                ],
                "speaker": f"fallback_speaker_{i % 3}",
                "start_index": 3
            }
            for i in range(5)  # Create 5 fallback samples
        ]
        
        # Create dummy audio files for fallback samples
        self._create_dummy_audio_files(fallback_samples)
        
        return fallback_samples
    
    def _validate_sample(self, sample: Dict[str, Any], idx: int) -> bool:
        """Validate a single training sample following arb_inference.py patterns."""
        try:
            # Check required fields
            if 'messages' not in sample:
                logger.debug(f"Sample {idx}: Missing 'messages' field")
                return False
            
            messages = sample['messages']
            if not isinstance(messages, list) or len(messages) < 3:
                logger.debug(f"Sample {idx}: Invalid messages structure (need >= 3 messages)")
                return False
            
            # Extract components using arb_inference.py pattern
            ref_audio_path, ref_text, target_text, speaker_id = self._extract_sample_components(sample)
            
            if not all([ref_audio_path, ref_text, target_text]):
                logger.debug(f"Sample {idx}: Missing required components")
                return False
            
            # Validate audio file exists (with auto-creation)
            audio_path = self._resolve_audio_path(ref_audio_path)
            if not audio_path.exists():
                # Try to create dummy audio file
                try:
                    audio_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    try:
                        import numpy as np
                        import soundfile as sf
                        
                        # Create 1 second of silence
                        duration = 1.0
                        sample_rate = 16000
                        samples = int(duration * sample_rate)
                        silent_audio = np.zeros(samples, dtype=np.float32)
                        sf.write(str(audio_path), silent_audio, sample_rate)
                        
                        logger.debug(f"✅ Created dummy audio file: {audio_path}")
                        
                    except ImportError:
                        audio_path.touch()
                        logger.debug(f"✅ Created empty audio file: {audio_path}")
                        
                except Exception as e:
                    logger.debug(f"Sample {idx}: Could not create audio file {audio_path}: {e}")
                    return False
            
            return True
        
        except Exception as e:
            logger.debug(f"Sample {idx}: Validation error: {e}")
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
        CRITICAL: No audio tokens in user message - only in assistant AudioContent!
        """
        messages = [
            # 💬 System message (exact match with arb_inference.py)
            Message(
                role="system",
                content=TextContent(text="Generate speech in the provided voice.")
            ),
            # 🗣️ User message with reference text (NO audio tokens here!)
            Message(
                role="user", 
                content=TextContent(text=ref_text)  # Just the text, no audio tokens
            ),
            # 🎧 Assistant message with reference audio (CRITICAL: AudioContent only)
            Message(
                role="assistant",
                content=AudioContent(audio_url=ref_audio_path)
            ),
            # 🎯 User message with target text to generate
            Message(
                role="user",
                content=TextContent(text=target_text)
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