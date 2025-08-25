#!/usr/bin/env python3
"""
Zero-Shot Voice Cloning Inference Script for Arabic Language
Based on Higgs Audio v2 architecture and generation.py patterns

This script processes ChatML format data to perform zero-shot voice cloning
for Arabic language text using reference audio conditioning.

KEY FIXES IMPLEMENTED:
1. **Whisper Processor Integration**: Added proper Whisper processor loading 
   for reference audio conditioning (encode_whisper_embed=True)
2. **Reference Audio Waveform Processing**: Load and process reference audio
   waveforms for Whisper feature extraction at 16kHz
3. **Proper ChatML Structure**: Use <|AUDIO|> tokens for reference audio
   instead of only DAC codes in generation context
4. **Dual Audio Pathway**: 
   - Whisper embeddings for reference audio conditioning (via <|AUDIO|> tokens)
   - DAC codes for generation context (via audio_ids)
5. **Audio-Text Conditioning**: Proper integration of reference audio features
   with text tokens for cross-modal attention

The script now properly follows the training pipeline's audio conditioning
mechanism where reference audio is processed through both:
- Whisper encoder (for semantic conditioning)
- DAC encoder (for acoustic tokens)
"""

import click
import json
import os
import shutil
import torch
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from loguru import logger

import shutil  # Added for file copying

# Higgs Audio imports
from boson_multimodal.model.higgs_audio import HiggsAudioConfig, HiggsAudioModel
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.dataset.chatml_dataset import (
    ChatMLDatasetSample,
    prepare_chatml_sample,
)
from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent
from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern
from transformers import AutoConfig, AutoTokenizer, AutoProcessor


class ArabicVoiceCloningInference:
    """
    Zero-shot voice cloning inference engine for Arabic language.
    
    Processes ChatML format data to generate speech in the reference voice
    while speaking Arabic target text.
    """
    
    def __init__(
        self,
        model_path: str = "bosonai/higgs-audio-v2-generation-3B-base",
        audio_tokenizer_path: str = "bosonai/higgs-audio-v2-tokenizer",
        device: str = "auto",
        device_id: Optional[int] = None,
        max_new_tokens: int = 512,  # Reduced from 2048 to prevent extended generation
        use_static_kv_cache: bool = True,
        adaptive_max_tokens: bool = True,  # Enable adaptive token calculation
        base_tokens_per_second: int = 25,  # 25Hz token rate for audio
    ):
        """
        Initialize the Arabic voice cloning inference engine.
        
        Args:
            model_path: Path to the Higgs Audio model
            audio_tokenizer_path: Path to the audio tokenizer
            device: Device to use ('auto', 'cuda', 'mps', 'cpu')
            device_id: Specific device ID for CUDA
            max_new_tokens: Maximum tokens to generate
            use_static_kv_cache: Whether to use static KV cache for speed
        """
        self.max_new_tokens = max_new_tokens
        self.use_static_kv_cache = use_static_kv_cache
        self.adaptive_max_tokens = adaptive_max_tokens
        self.base_tokens_per_second = base_tokens_per_second
        
        # Setup device
        self._setup_device(device, device_id)
        
        # Load audio tokenizer
        logger.info(f"Loading audio tokenizer from {audio_tokenizer_path}")
        audio_tokenizer_device = "cpu" if self._device == "mps" else self._device
        self.audio_tokenizer = load_higgs_audio_tokenizer(
            audio_tokenizer_path, 
            device=audio_tokenizer_device
        )
        
        # Load model
        logger.info(f"Loading Higgs Audio model from {model_path}")
        self.model = HiggsAudioModel.from_pretrained(
            model_path,
            device_map=self._device,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        
        # Load tokenizer and config
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.config = AutoConfig.from_pretrained(model_path)
        
        # Load Whisper processor for reference audio conditioning
        if self.config.encode_whisper_embed:
            logger.info("Loading Whisper processor for reference audio conditioning")
            try:
                # Try the recommended model first
                whisper_processor = AutoProcessor.from_pretrained(
                    "openai/whisper-large-v3",  # Changed from v3-turbo to standard v3
                    trust_remote_code=True
                )
                logger.info("✅ Successfully loaded Whisper processor")
            except Exception as e:
                logger.warning(f"Failed to load whisper-large-v3: {e}")
                try:
                    # Fallback to base model
                    whisper_processor = AutoProcessor.from_pretrained(
                        "openai/whisper-base",
                        trust_remote_code=True
                    )
                    logger.info("✅ Successfully loaded Whisper base processor as fallback")
                except Exception as e2:
                    logger.error(f"Failed to load any Whisper processor: {e2}")
                    whisper_processor = None
        else:
            logger.info("Whisper embedding disabled in config")
            whisper_processor = None
        
        # Setup collator
        logger.info(f"Setting up collator with Whisper processor: {whisper_processor is not None}")
        self.collator = HiggsAudioSampleCollator(
            whisper_processor=whisper_processor,
            audio_in_token_id=self.config.audio_in_token_idx,
            audio_out_token_id=self.config.audio_out_token_idx,
            audio_stream_bos_id=self.config.audio_stream_bos_id,
            audio_stream_eos_id=self.config.audio_stream_eos_id,
            encode_whisper_embed=self.config.encode_whisper_embed,
            pad_token_id=self.config.pad_token_id,
            return_audio_in_tokens=self.config.encode_audio_in_tokens,
            use_delay_pattern=self.config.use_delay_pattern,
            round_to=1,
            audio_num_codebooks=self.config.audio_num_codebooks,
        )
        
        # Verify collator setup
        logger.info(f"Collator configuration:")
        logger.info(f"  - encode_whisper_embed: {self.collator.encode_whisper_embed}")
        logger.info(f"  - whisper_processor available: {self.collator.whisper_processor is not None}")
        logger.info(f"  - return_audio_in_tokens: {self.collator.return_audio_in_tokens}")
        
        logger.info(f"Arabic Voice Cloning Inference Engine initialized on {self._device}")
        logger.info(f"Audio generation settings: max_tokens={max_new_tokens}, adaptive={adaptive_max_tokens}")
    
    def _setup_device(self, device: str, device_id: Optional[int]):
        """Setup the compute device."""
        if device_id is not None:
            self._device = f"cuda:{device_id}"
        else:
            if device == "auto":
                if torch.cuda.is_available():
                    self._device = "cuda:0"
                elif torch.backends.mps.is_available():
                    self._device = "mps"
                else:
                    self._device = "cpu"
            elif device == "cuda":
                self._device = "cuda:0"
            elif device == "mps":
                self._device = "mps"
            else:
                self._device = "cpu"
        
        # Disable static KV cache on MPS since it relies on CUDA graphs
        if self._device == "mps" and self.use_static_kv_cache:
            self.use_static_kv_cache = False
            
        logger.info(f"Using device: {self._device}")
    
    def calculate_adaptive_max_tokens(self, target_text: str) -> int:
        """
        Calculate appropriate max tokens based on target text length.
        
        Args:
            target_text: Text to generate audio for
            
        Returns:
            Calculated max tokens for generation
        """
        if not self.adaptive_max_tokens:
            return self.max_new_tokens
            
        # Estimate speaking rate: ~150 words per minute for Arabic
        word_count = len(target_text.split())
        
        # For Arabic, account for script complexity
        char_count = len(target_text)
        
        # Estimate duration based on both words and characters
        word_duration = (word_count / 150) * 60  # seconds
        char_duration = (char_count / 8) * 60 / 150  # approximate character rate
        
        # Use the longer estimate for better coverage
        estimated_duration = max(word_duration, char_duration)
        
        # Add buffer for natural speech variations (1.5x)
        buffer_factor = 1.5
        max_duration = estimated_duration * buffer_factor
        
        # Convert to tokens (25 Hz rate)
        calculated_tokens = int(max_duration * self.base_tokens_per_second)
        
        # Apply reasonable bounds: minimum 64, maximum 512 for Arabic
        bounded_tokens = max(min(calculated_tokens, self.max_new_tokens), 64)
        
        logger.info(f"Text length: {word_count} words, {char_count} chars")
        logger.info(f"Estimated duration: {estimated_duration:.1f}s -> {bounded_tokens} tokens")
        
        return bounded_tokens
    
    def process_chatml_sample(self, sample: Dict[str, Any]) -> tuple:
        """
        Extract reference audio, reference text, and target text from ChatML sample.
        
        Args:
            sample: ChatML format sample dictionary
            
        Returns:
            Tuple of (ref_audio_path, ref_text, target_text, speaker_id)
        """
        try:
            # Parse the ChatML sample structure
            messages = sample["messages"]
            
            # Find user message with reference content
            ref_audio_path = None
            ref_text = None
            target_text = None
            speaker_id = sample.get("speaker", "unknown")
            
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
            
            if not all([ref_audio_path, ref_text, target_text]):
                logger.warning(f"Missing required components: ref_audio={ref_audio_path}, ref_text={ref_text}, target_text={target_text}")
                return None, None, None, None
                
            return ref_audio_path, ref_text, target_text, speaker_id
            
        except Exception as e:
            logger.error(f"Error processing ChatML sample: {e}")
            return None, None, None, None
    
    def create_generation_messages(
        self, 
        ref_text: str, 
        ref_audio_path: str, 
        target_text: str
    ) -> tuple:
        """
        Create the message structure for generation, following Higgs Audio v2 voice cloning patterns.
        
        Args:
            ref_text: Reference text that was spoken in reference audio
            ref_audio_path: Path to reference audio file
            target_text: Text to generate in the reference voice
            
        Returns:
            Tuple of (messages, audio_ids, ref_waveform, sample_rate)
        """
        # Create system message for Arabic voice cloning - keep it concise
        system_message = Message(
            role="system",
            content="Generate speech in the provided voice."
        )
        
        # Create user message with reference text and audio token placeholder
        # This follows the proper ChatML format for voice cloning conditioning
        user_ref_message = Message(
            role="user",
            content=f"{ref_text} <|audio_bos|><|AUDIO|><|audio_eos|>"
        )
        
        # Create assistant message with reference audio using AudioContent
        # This is the key difference - using AudioContent instead of text confirmation
        assistant_ref_message = Message(
            role="assistant",
            content=AudioContent(audio_url=ref_audio_path)
        )
        
        # Create user message with target text for generation
        user_target_message = Message(
            role="user", 
            content=target_text
        )
        
        messages = [system_message, user_ref_message, assistant_ref_message, user_target_message]
        
        # Load and prepare reference audio waveform for Whisper processing
        logger.info(f"Loading reference audio waveform: {ref_audio_path}")
        if not os.path.exists(ref_audio_path):
            logger.error(f"Reference audio file not found: {ref_audio_path}")
            return messages, []
            
        try:
            # Load audio waveform for Whisper embeddings
            waveform, sr = torchaudio.load(ref_audio_path)
            logger.info(f"Loaded audio: shape={waveform.shape}, sr={sr}")
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                logger.info(f"Converted to mono: shape={waveform.shape}")
            
            # Resample to 16kHz for Whisper if needed
            target_sr = 16000
            if sr != target_sr:
                resampler = T.Resample(sr, target_sr)
                waveform = resampler(waveform)
                sr = target_sr
                logger.info(f"Resampled to {target_sr}Hz: shape={waveform.shape}")
            
            # Encode reference audio for DAC tokens (for context)
            audio_tokens = self.audio_tokenizer.encode(ref_audio_path)
            audio_ids = [audio_tokens]
            logger.info(f"Audio tokens shape: {audio_tokens.shape}")
            
            # Store waveform for Whisper processing in dataset sample
            ref_waveform = waveform.squeeze(0)  # Remove channel dimension
            logger.info(f"Final waveform for Whisper: shape={ref_waveform.shape}, type={type(ref_waveform)}")
            
            # Validate the waveform
            if ref_waveform.numel() == 0:
                logger.error("Waveform is empty after processing!")
                ref_waveform = None
            elif torch.isnan(ref_waveform).any() or torch.isinf(ref_waveform).any():
                logger.error("Waveform contains NaN or Inf values!")
                ref_waveform = None
            else:
                logger.info(f"Waveform validation passed: min={ref_waveform.min():.4f}, max={ref_waveform.max():.4f}")
            
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            audio_ids = []
            ref_waveform = None
            sr = None
        
        return messages, audio_ids, ref_waveform, sr
    
    @torch.inference_mode()
    def generate_arabic_speech(
        self,
        messages: List[Message],
        audio_ids: List[torch.Tensor],
        ref_waveform: Optional[torch.Tensor] = None,
        ref_sample_rate: Optional[int] = None,
        target_text: str = "",  # Added for adaptive token calculation
        temperature: float = 0.3,
        top_k: int = 50,
        top_p: float = 0.95,
        seed: Optional[int] = None
    ) -> tuple:
        """
        Generate Arabic speech using the provided messages and reference audio.
        
        Args:
            messages: List of Message objects for generation context
            audio_ids: List of audio token tensors
            ref_waveform: Reference audio waveform for Whisper conditioning
            ref_sample_rate: Sample rate of reference waveform
            target_text: Target text for adaptive token calculation
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (waveform, sample_rate, text_output)
        """
        try:
            # Create ChatML sample
            chatml_sample = ChatMLSample(messages=messages)
            
            # Prepare input tokens
            input_tokens, _, _, _ = prepare_chatml_sample(chatml_sample, self.tokenizer)
            if input_tokens is None:
                logger.error("Failed to prepare ChatML sample")
                return None, None, None
                
            # Add assistant header for generation
            postfix = self.tokenizer.encode(
                "<|start_header_id|>assistant<|end_header_id|>\n\n", 
                add_special_tokens=False
            )
            input_tokens.extend(postfix)
            
            logger.info("=== Generation Input ===")
            logger.info(self.tokenizer.decode(input_tokens))
            
            # Create dataset sample with proper audio waveform for Whisper processing
            logger.info(f"Checking Whisper conditioning requirements:")
            logger.info(f"  - ref_waveform is not None: {ref_waveform is not None}")
            logger.info(f"  - config.encode_whisper_embed: {self.config.encode_whisper_embed}")
            logger.info(f"  - collator.whisper_processor is not None: {self.collator.whisper_processor is not None}")
            
            if ref_waveform is not None and self.config.encode_whisper_embed and self.collator.whisper_processor is not None:
                # Dual pathway: Whisper embeddings + DAC tokens for proper voice cloning
                curr_sample = ChatMLDatasetSample(
                    input_ids=torch.LongTensor(input_tokens),
                    label_ids=None,
                    # DAC audio tokens for generation context
                    audio_ids_concat=torch.concat([ele.cpu() for ele in audio_ids], dim=1) if audio_ids else torch.empty(0, 0, dtype=torch.long),
                    audio_ids_start=torch.cumsum(
                        torch.tensor([0] + [ele.shape[1] for ele in audio_ids], dtype=torch.long), 
                        dim=0
                    ) if audio_ids else torch.empty(0, dtype=torch.long),
                    # Reference waveform for Whisper feature extraction (key for voice cloning)
                    audio_waveforms_concat=ref_waveform,
                    audio_waveforms_start=torch.tensor([0, len(ref_waveform)], dtype=torch.long),
                    audio_sample_rate=torch.tensor([ref_sample_rate], dtype=torch.float32),
                    audio_speaker_indices=torch.empty(0, dtype=torch.long),
                )
                logger.info(f"✅ Using Whisper conditioning with waveform: {ref_waveform.shape}, DAC tokens: {audio_ids[0].shape if audio_ids else 'None'}")
            else:
                # Fallback to audio tokens only (original behavior) - less optimal for voice cloning
                reasons = []
                if ref_waveform is None:
                    reasons.append("ref_waveform is None")
                if not self.config.encode_whisper_embed:
                    reasons.append("encode_whisper_embed is False")
                if self.collator.whisper_processor is None:
                    reasons.append("whisper_processor is None")
                
                logger.warning(f"❌ Creating sample without Whisper embeddings - Reasons: {', '.join(reasons)} - Voice similarity may be reduced")
                curr_sample = ChatMLDatasetSample(
                    input_ids=torch.LongTensor(input_tokens),
                    label_ids=None,
                    audio_ids_concat=torch.concat([ele.cpu() for ele in audio_ids], dim=1) if audio_ids else torch.empty(0, 0, dtype=torch.long),
                    audio_ids_start=torch.cumsum(
                        torch.tensor([0] + [ele.shape[1] for ele in audio_ids], dtype=torch.long), 
                        dim=0
                    ) if audio_ids else torch.empty(0, dtype=torch.long),
                    audio_waveforms_concat=torch.empty(0, dtype=torch.float32),
                    audio_waveforms_start=torch.empty(0, dtype=torch.long),
                    audio_sample_rate=torch.empty(0, dtype=torch.float32),
                    audio_speaker_indices=torch.empty(0, dtype=torch.long),
                )
            
            # Collate data
            batch_data = self.collator([curr_sample])
            batch = asdict(batch_data)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.contiguous().to(self._device)
            
            # Calculate adaptive max tokens based on target text
            adaptive_max_tokens = self.calculate_adaptive_max_tokens(target_text)
            
            # Generate
            logger.info(f"Starting generation with {adaptive_max_tokens} max tokens...")
            outputs = self.model.generate(
                **batch,
                max_new_tokens=adaptive_max_tokens,
                use_cache=True,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stop_strings=["<|end_of_text|>", "<|eot_id|>", "<|audio_eos|>"],  # Added audio_eos
                tokenizer=self.tokenizer,
                seed=seed,
            )
            
            # Process audio outputs
            audio_out_ids_list = []
            for ele in outputs[1]:
                audio_out_ids = ele
                if self.config.use_delay_pattern:
                    audio_out_ids = revert_delay_pattern(audio_out_ids)
                audio_out_ids_list.append(
                    audio_out_ids.clip(0, self.audio_tokenizer.codebook_size - 1)[:, 1:-1]
                )
            
            if audio_out_ids_list:
                concat_audio_out_ids = torch.concat(audio_out_ids_list, dim=1)
                
                # Handle MPS compatibility
                if concat_audio_out_ids.device.type == "mps":
                    concat_audio_out_ids_cpu = concat_audio_out_ids.detach().cpu()
                else:
                    concat_audio_out_ids_cpu = concat_audio_out_ids
                
                # Decode audio
                logger.info("Decoding generated audio...")
                waveform = self.audio_tokenizer.decode(concat_audio_out_ids_cpu.unsqueeze(0))[0, 0]
                sample_rate = 24000  # Higgs Audio output sample rate
                
                # Get text output
                text_result = self.tokenizer.decode(outputs[0][0])
                
                logger.info("=== Generation Output Text ===")
                logger.info(text_result)
                
                return waveform, sample_rate, text_result
            else:
                logger.error("No audio output generated")
                return None, None, None
                
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def save_reference_and_generated_audio(
        self,
        ref_audio_path: str,
        generated_waveform: np.ndarray,
        sample_rate: int,
        output_dir: str,
        sample_id: int,
        speaker_id: str
    ) -> dict:
        """
        Save both reference and generated audio with consistent naming.
        
        Args:
            ref_audio_path: Path to reference audio file
            generated_waveform: Generated audio waveform
            sample_rate: Sample rate for generated audio
            output_dir: Output directory
            sample_id: Sample identifier
            speaker_id: Speaker identifier
            
        Returns:
            Dictionary with file paths and metadata
        """
        # Create filenames
        base_filename = f"arabic_generated_{sample_id:03d}_{speaker_id}"
        generated_file = os.path.join(output_dir, f"{base_filename}.wav")
        reference_file = os.path.join(output_dir, f"{base_filename}_ref.wav")
        
        # Save generated audio
        sf.write(generated_file, generated_waveform, sample_rate)
        logger.info(f"Saved generated audio to {generated_file}")
        
        # Copy reference audio
        if os.path.exists(ref_audio_path):
            try:
                shutil.copy2(ref_audio_path, reference_file)
                logger.info(f"Saved reference audio to {reference_file}")
            except Exception as e:
                logger.warning(f"Failed to copy reference audio: {e}")
                reference_file = None
        else:
            logger.warning(f"Reference audio file not found: {ref_audio_path}")
            reference_file = None
        
        return {
            "generated_audio": generated_file,
            "reference_audio": reference_file,
            "sample_rate": sample_rate
        }
    
    def process_chatml_file(
        self,
        chatml_file: str,
        output_dir: str,
        temperature: float = 0.3,
        top_k: int = 50,
        top_p: float = 0.95,
        seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a ChatML file and generate Arabic speech for all samples.
        
        Args:
            chatml_file: Path to ChatML JSON file
            output_dir: Directory to save generated audio files
            temperature: Sampling temperature
            top_k: Top-k sampling parameter  
            top_p: Top-p sampling parameter
            seed: Random seed
            
        Returns:
            List of processing results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load ChatML data
        logger.info(f"Loading ChatML data from {chatml_file}")
        with open(chatml_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        if not isinstance(samples, list):
            samples = [samples]
        
        results = []
        
        for i, sample in enumerate(samples):
            logger.info(f"Processing sample {i+1}/{len(samples)}")
            
            # Extract components from ChatML sample
            ref_audio_path, ref_text, target_text, speaker_id = self.process_chatml_sample(sample)
            
            if not all([ref_audio_path, ref_text, target_text]):
                logger.warning(f"Skipping sample {i} due to missing components")
                results.append({
                    "sample_id": i,
                    "status": "failed", 
                    "error": "Missing required components",
                    "speaker_id": speaker_id
                })
                continue
            
            # Create generation messages
            messages, audio_ids, ref_waveform, ref_sr = self.create_generation_messages(
                ref_text, ref_audio_path, target_text
            )
            
            if not audio_ids:
                logger.warning(f"Skipping sample {i} due to audio encoding failure")
                results.append({
                    "sample_id": i,
                    "status": "failed",
                    "error": "Audio encoding failed", 
                    "speaker_id": speaker_id
                })
                continue
            
            # Generate speech
            waveform, sample_rate, text_output = self.generate_arabic_speech(
                messages, audio_ids, ref_waveform, ref_sr, target_text, temperature, top_k, top_p, seed
            )
            
            if waveform is not None:
                # Save both generated and reference audio
                file_info = self.save_reference_and_generated_audio(
                    ref_audio_path, waveform, sample_rate, output_dir, i, speaker_id
                )
                
                results.append({
                    "sample_id": i,
                    "status": "success",
                    "output_file": file_info["generated_audio"],
                    "reference_file": file_info["reference_audio"],
                    "speaker_id": speaker_id,
                    "ref_audio": ref_audio_path,
                    "ref_text": ref_text,
                    "target_text": target_text,
                    "generated_text": text_output,
                    "duration_estimate": f"{len(target_text.split())} words"
                })
            else:
                logger.error(f"Generation failed for sample {i}")
                results.append({
                    "sample_id": i,
                    "status": "failed",
                    "error": "Generation failed",
                    "speaker_id": speaker_id
                })
        
        # Save results summary
        results_file = os.path.join(output_dir, "generation_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved results summary to {results_file}")
        return results


@click.command()
@click.option(
    "--chatml_file",
    type=str,
    required=True,
    help="Path to ChatML JSON file containing Arabic voice cloning data"
)
@click.option(
    "--output_dir", 
    type=str,
    default="./arabic_generated_audio",
    help="Directory to save generated audio files"
)
@click.option(
    "--model_path",
    type=str, 
    default="bosonai/higgs-audio-v2-generation-3B-base",
    help="Path to Higgs Audio model"
)
@click.option(
    "--audio_tokenizer_path",
    type=str,
    default="bosonai/higgs-audio-v2-tokenizer", 
    help="Path to audio tokenizer"
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cuda", "mps", "cpu"]),
    default="auto",
    help="Device to use for inference"
)
@click.option(
    "--device_id",
    type=int,
    default=None,
    help="Specific device ID for CUDA"
)
@click.option(
    "--temperature",
    type=float,
    default=0.3,
    help="Sampling temperature"
)
@click.option(
    "--top_k",
    type=int,
    default=50,
    help="Top-k sampling parameter"
)
@click.option(
    "--top_p", 
    type=float,
    default=0.95,
    help="Top-p sampling parameter"
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility"
)
@click.option(
    "--max_new_tokens",
    type=int,
    default=512,
    help="Maximum number of tokens to generate (reduced from 2048 for better control)"
)
@click.option(
    "--adaptive_max_tokens",
    type=bool,
    default=True,
    help="Enable adaptive token calculation based on text length"
)
def main(
    chatml_file,
    output_dir,
    model_path,
    audio_tokenizer_path,
    device,
    device_id,
    temperature,
    top_k,
    top_p,
    seed,
    max_new_tokens,
    adaptive_max_tokens
):
    """
    Arabic Zero-Shot Voice Cloning Inference Script
    
    Process ChatML format data to generate Arabic speech with reference voice characteristics.
    """
    logger.info("Starting Arabic Voice Cloning Inference")
    
    # Initialize inference engine
    inference_engine = ArabicVoiceCloningInference(
        model_path=model_path,
        audio_tokenizer_path=audio_tokenizer_path,
        device=device,
        device_id=device_id,
        max_new_tokens=max_new_tokens,
        use_static_kv_cache=True,
        adaptive_max_tokens=adaptive_max_tokens
    )
    
    # Process ChatML file
    results = inference_engine.process_chatml_file(
        chatml_file=chatml_file,
        output_dir=output_dir,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed
    )
    
    # Print summary
    successful = sum(1 for r in results if r["status"] == "success")
    total = len(results)
    
    logger.info(f"Processing complete: {successful}/{total} samples successful")
    logger.info(f"Generated audio files saved in: {output_dir}")


if __name__ == "__main__":
    main()