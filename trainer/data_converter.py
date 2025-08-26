#!/usr/bin/env python3
"""
Data Format Converter for Higgs-Audio Training Pipeline

This module converts various data formats into the correct ChatML format
expected by the training pipeline, ensuring compatibility with arb_inference.py
and the existing boson_multimodal components.

Expected ChatML format:
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

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from loguru import logger

# Add higgs-audio root to path for imports
current_file = Path(__file__).resolve()
trainer_dir = current_file.parent
higgs_audio_root = trainer_dir.parent

if str(higgs_audio_root) not in sys.path:
    sys.path.insert(0, str(higgs_audio_root))


class DataFormatConverter:
    """
    Converts various data formats into the correct ChatML format for training.
    
    Handles multiple input formats and normalizes them to match the exact
    structure expected by the training pipeline and inference implementations.
    """
    
    def __init__(self, audio_base_path: str = ""):
        """
        Initialize the data format converter.
        
        Args:
            audio_base_path: Base path for resolving relative audio file paths
        """
        self.audio_base_path = Path(audio_base_path) if audio_base_path else Path()
        
    def convert_file(
        self, 
        input_path: str, 
        output_path: str,
        format_type: str = "auto"
    ) -> bool:
        """
        Convert a data file from various formats to ChatML format.
        
        Args:
            input_path: Path to input data file
            output_path: Path to save converted ChatML data
            format_type: Input format type ('auto', 'manifest', 'paired', 'lora')
            
        Returns:
            True if conversion successful, False otherwise
        """
        try:
            logger.info(f"🔄 Converting data file: {input_path} → {output_path}")
            
            # Load input data
            with open(input_path, 'r', encoding='utf-8') as f:
                if input_path.endswith('.jsonl'):
                    # JSONL format - each line is a JSON object
                    raw_data = []
                    for line in f:
                        line = line.strip()
                        if line:
                            raw_data.append(json.loads(line))
                else:
                    # Regular JSON format
                    raw_data = json.load(f)
            
            # Auto-detect format if needed
            if format_type == "auto":
                format_type = self._detect_format(raw_data)
                logger.info(f"🔍 Auto-detected format: {format_type}")
            
            # Convert based on format type
            if format_type == "chatml":
                # Already in ChatML format, just validate
                converted_data = self._validate_chatml_format(raw_data)
            elif format_type == "manifest":
                # Manifest format with audio_filepath, text pairs
                converted_data = self._convert_from_manifest(raw_data)
            elif format_type == "paired":
                # Paired format with reference and target data
                converted_data = self._convert_from_paired(raw_data)
            elif format_type == "lora":
                # LoRA training format
                converted_data = self._convert_from_lora_format(raw_data)
            elif format_type == "mixed":
                # Mixed content format (text + audio in same message)
                converted_data = self._convert_from_mixed_content(raw_data)
            else:
                raise ValueError(f"Unknown format type: {format_type}")
            
            # Save converted data
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(converted_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Conversion complete: {len(converted_data)} samples saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Conversion failed: {e}")
            return False
    
    def _detect_format(self, data: Union[List, Dict]) -> str:
        """Auto-detect the input data format."""
        if isinstance(data, dict):
            data = [data]
        
        if not data:
            return "unknown"
        
        sample = data[0]
        
        # Check for ChatML format
        if "messages" in sample and isinstance(sample["messages"], list):
            return "chatml"
        
        # Check for manifest format
        if "audio_filepath" in sample and "text" in sample:
            return "manifest"
        
        # Check for LoRA format
        if "conversations" in sample or "instruction" in sample:
            return "lora"
        
        # Check for paired format
        if "reference_audio" in sample and "target_text" in sample:
            return "paired"
        
        # Check for mixed content format
        if "messages" in sample:
            for msg in sample["messages"]:
                if "content" in msg and isinstance(msg["content"], list):
                    return "mixed"
        
        return "unknown"
    
    def _validate_chatml_format(self, data: List[Dict]) -> List[Dict]:
        """Validate and fix existing ChatML format data."""
        validated_samples = []
        
        for i, sample in enumerate(data):
            try:
                # Ensure required fields exist
                if "messages" not in sample:
                    logger.warning(f"⚠️ Sample {i}: Missing 'messages' field, skipping")
                    continue
                
                messages = sample["messages"]
                if not isinstance(messages, list):
                    logger.warning(f"⚠️ Sample {i}: 'messages' is not a list, skipping")
                    continue
                
                # Validate message structure
                validated_messages = []
                for msg_idx, message in enumerate(messages):
                    if not isinstance(message, dict):
                        logger.warning(f"⚠️ Sample {i}, Message {msg_idx}: Invalid format")
                        continue
                    
                    if "role" not in message or "content" not in message:
                        logger.warning(f"⚠️ Sample {i}, Message {msg_idx}: Missing role or content")
                        continue
                    
                    validated_messages.append(message)
                
                if len(validated_messages) < 3:  # Need at least system, user, assistant
                    logger.warning(f"⚠️ Sample {i}: Insufficient messages ({len(validated_messages)})")
                    continue
                
                # Create validated sample
                validated_sample = {
                    "messages": validated_messages,
                    "speaker": sample.get("speaker", f"speaker_{i % 10}"),
                    "start_index": sample.get("start_index", len(validated_messages) - 1)
                }
                
                validated_samples.append(validated_sample)
                
            except Exception as e:
                logger.warning(f"⚠️ Sample {i}: Validation error: {e}")
                continue
        
        logger.info(f"✅ ChatML validation: {len(validated_samples)}/{len(data)} samples valid")
        return validated_samples
    
    def _convert_from_manifest(self, data: List[Dict]) -> List[Dict]:
        """Convert from manifest format (audio_filepath, text pairs)."""
        converted_samples = []
        
        for i, item in enumerate(data):
            try:
                audio_path = item.get("audio_filepath") or item.get("audio_url") or item.get("audio")
                text = item.get("text") or item.get("transcript") or item.get("content")
                
                if not audio_path or not text:
                    logger.warning(f"⚠️ Manifest sample {i}: Missing audio_path or text")
                    continue
                
                # Resolve audio path
                resolved_audio_path = str(self._resolve_audio_path(audio_path))
                
                # Split text for reference and target (simple approach)
                words = text.split()
                mid_point = len(words) // 2
                
                ref_text = " ".join(words[:mid_point]) if mid_point > 0 else text[:len(text)//2]
                target_text = " ".join(words[mid_point:]) if mid_point > 0 else text[len(text)//2:]
                
                # Ensure minimum length
                if len(ref_text.strip()) < 5:
                    ref_text = text[:len(text)//3]
                    target_text = text[len(text)//3:]
                
                sample = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "Generate speech in the provided voice."
                        },
                        {
                            "role": "user",
                            "content": ref_text.strip()
                        },
                        {
                            "role": "assistant",
                            "content": {
                                "type": "audio",
                                "audio_url": resolved_audio_path
                            }
                        },
                        {
                            "role": "user",
                            "content": target_text.strip()
                        }
                    ],
                    "speaker": item.get("speaker_id", f"speaker_{i % 10}"),
                    "start_index": 3
                }
                
                converted_samples.append(sample)
                
            except Exception as e:
                logger.warning(f"⚠️ Manifest sample {i}: Conversion error: {e}")
                continue
        
        logger.info(f"✅ Manifest conversion: {len(converted_samples)}/{len(data)} samples converted")
        return converted_samples
    
    def _convert_from_paired(self, data: List[Dict]) -> List[Dict]:
        """Convert from paired format (reference_audio, reference_text, target_text)."""
        converted_samples = []
        
        for i, item in enumerate(data):
            try:
                ref_audio = item.get("reference_audio") or item.get("ref_audio") or item.get("audio_path")
                ref_text = item.get("reference_text") or item.get("ref_text")
                target_text = item.get("target_text") or item.get("text") or item.get("content")
                
                if not all([ref_audio, ref_text, target_text]):
                    logger.warning(f"⚠️ Paired sample {i}: Missing required fields")
                    continue
                
                # Resolve audio path
                resolved_audio_path = str(self._resolve_audio_path(ref_audio))
                
                sample = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "Generate speech in the provided voice."
                        },
                        {
                            "role": "user",
                            "content": ref_text.strip()
                        },
                        {
                            "role": "assistant",
                            "content": {
                                "type": "audio",
                                "audio_url": resolved_audio_path
                            }
                        },
                        {
                            "role": "user",
                            "content": target_text.strip()
                        }
                    ],
                    "speaker": item.get("speaker_id", f"speaker_{i % 10}"),
                    "start_index": 3
                }
                
                converted_samples.append(sample)
                
            except Exception as e:
                logger.warning(f"⚠️ Paired sample {i}: Conversion error: {e}")
                continue
        
        logger.info(f"✅ Paired conversion: {len(converted_samples)}/{len(data)} samples converted")
        return converted_samples
    
    def _convert_from_lora_format(self, data: List[Dict]) -> List[Dict]:
        """Convert from LoRA training format."""
        converted_samples = []
        
        for i, item in enumerate(data):
            try:
                # Handle different LoRA formats
                conversations = item.get("conversations", [])
                instruction = item.get("instruction", "")
                audio_path = item.get("audio", "") or item.get("audio_path", "")
                
                if conversations:
                    # Multi-turn conversation format
                    ref_text = ""
                    target_text = ""
                    
                    for conv in conversations:
                        role = conv.get("from", "")
                        content = conv.get("value", "")
                        
                        if role == "human" and not ref_text:
                            ref_text = content
                        elif role == "gpt" and not target_text:
                            target_text = content
                
                elif instruction:
                    # Instruction format
                    ref_text = instruction
                    target_text = item.get("output", "") or item.get("response", "")
                
                else:
                    logger.warning(f"⚠️ LoRA sample {i}: Unknown format")
                    continue
                
                if not audio_path:
                    # Create dummy audio path if missing
                    audio_path = f"data/audio/sample_{i}.wav"
                
                # Resolve audio path
                resolved_audio_path = str(self._resolve_audio_path(audio_path))
                
                sample = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "Generate speech in the provided voice."
                        },
                        {
                            "role": "user",
                            "content": ref_text.strip()
                        },
                        {
                            "role": "assistant",
                            "content": {
                                "type": "audio",
                                "audio_url": resolved_audio_path
                            }
                        },
                        {
                            "role": "user",
                            "content": target_text.strip()
                        }
                    ],
                    "speaker": item.get("speaker_id", f"speaker_{i % 10}"),
                    "start_index": 3
                }
                
                converted_samples.append(sample)
                
            except Exception as e:
                logger.warning(f"⚠️ LoRA sample {i}: Conversion error: {e}")
                continue
        
        logger.info(f"✅ LoRA conversion: {len(converted_samples)}/{len(data)} samples converted")
        return converted_samples
    
    def _convert_from_mixed_content(self, data: List[Dict]) -> List[Dict]:
        """Convert from mixed content format (text + audio in same message)."""
        converted_samples = []
        
        for i, item in enumerate(data):
            try:
                messages = item.get("messages", [])
                if not messages:
                    continue
                
                ref_text = ""
                target_text = ""
                audio_path = ""
                
                # Extract information from mixed content
                for message in messages:
                    content = message.get("content", [])
                    if isinstance(content, list):
                        text_parts = []
                        for part in content:
                            if part.get("type") == "text":
                                text_parts.append(part["text"])
                            elif part.get("type") == "audio":
                                if not audio_path:
                                    audio_path = part["audio_url"]
                        
                        if text_parts and not ref_text:
                            ref_text = " ".join(text_parts)
                    elif isinstance(content, str):
                        if not target_text and message.get("role") == "user":
                            target_text = content
                
                if not all([ref_text, audio_path]):
                    logger.warning(f"⚠️ Mixed content sample {i}: Missing required data")
                    continue
                
                if not target_text:
                    # Generate target text from reference
                    target_text = f"Please generate speech for: {ref_text[:50]}..."
                
                # Resolve audio path
                resolved_audio_path = str(self._resolve_audio_path(audio_path))
                
                sample = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "Generate speech in the provided voice."
                        },
                        {
                            "role": "user",
                            "content": ref_text.strip()
                        },
                        {
                            "role": "assistant",
                            "content": {
                                "type": "audio",
                                "audio_url": resolved_audio_path
                            }
                        },
                        {
                            "role": "user",
                            "content": target_text.strip()
                        }
                    ],
                    "speaker": item.get("speaker", f"speaker_{i % 10}"),
                    "start_index": 3
                }
                
                converted_samples.append(sample)
                
            except Exception as e:
                logger.warning(f"⚠️ Mixed content sample {i}: Conversion error: {e}")
                continue
        
        logger.info(f"✅ Mixed content conversion: {len(converted_samples)}/{len(data)} samples converted")
        return converted_samples
    
    def _resolve_audio_path(self, audio_url: str) -> Path:
        """Resolve audio path (handle both absolute and relative paths)."""
        audio_path = Path(audio_url)
        if not audio_path.is_absolute():
            audio_path = self.audio_base_path / audio_path
        return audio_path
    
    def create_dummy_audio_files(self, data_path: str, output_dir: str) -> bool:
        """
        Create dummy audio files for missing references to prevent training failures.
        
        Args:
            data_path: Path to ChatML data file
            output_dir: Directory to create dummy audio files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            os.makedirs(output_dir, exist_ok=True)
            missing_files = set()
            
            for sample in data:
                messages = sample.get("messages", [])
                for message in messages:
                    content = message.get("content")
                    if isinstance(content, dict) and content.get("type") == "audio":
                        audio_path = Path(content["audio_url"])
                        if not audio_path.exists():
                            missing_files.add(str(audio_path))
            
            if missing_files:
                logger.info(f"🎵 Creating {len(missing_files)} dummy audio files...")
                
                # Create a simple silent audio file (1 second, 16kHz)
                try:
                    import numpy as np
                    import soundfile as sf
                    
                    # Generate 1 second of silence at 16kHz
                    duration = 1.0
                    sample_rate = 16000
                    samples = int(duration * sample_rate)
                    silent_audio = np.zeros(samples, dtype=np.float32)
                    
                    for audio_path in missing_files:
                        # Create directory if needed
                        Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
                        
                        # Save silent audio file
                        sf.write(audio_path, silent_audio, sample_rate)
                    
                    logger.info(f"✅ Created {len(missing_files)} dummy audio files")
                    return True
                    
                except ImportError:
                    logger.warning("⚠️ soundfile not available, creating empty placeholder files")
                    for audio_path in missing_files:
                        Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
                        Path(audio_path).touch()
                    return True
            
            else:
                logger.info("✅ All audio files already exist")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to create dummy audio files: {e}")
            return False


def main():
    """Command-line interface for data conversion."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert data to ChatML format for Higgs-Audio training")
    parser.add_argument("input_path", help="Path to input data file")
    parser.add_argument("output_path", help="Path to save converted ChatML data")
    parser.add_argument("--format", choices=["auto", "manifest", "paired", "lora", "mixed", "chatml"], 
                       default="auto", help="Input data format")
    parser.add_argument("--audio_base_path", default="", help="Base path for audio files")
    parser.add_argument("--create_dummy_audio", action="store_true", 
                       help="Create dummy audio files for missing references")
    
    args = parser.parse_args()
    
    converter = DataFormatConverter(audio_base_path=args.audio_base_path)
    
    # Convert data
    success = converter.convert_file(
        args.input_path, 
        args.output_path, 
        args.format
    )
    
    if success and args.create_dummy_audio:
        converter.create_dummy_audio_files(args.output_path, "data/dummy_audio")
    
    return success


if __name__ == "__main__":
    main()