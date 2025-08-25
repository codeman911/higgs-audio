#!/usr/bin/env python3
"""
Zero-Shot Voice Cloning Inference Script for Arabic Language
Based on Higgs Audio v2 architecture and generation.py patterns

This script processes ChatML format data to perform zero-shot voice cloning
for Arabic language text using reference audio conditioning.

KEY FIXES IMPLEMENTED:
1. **Forced Whisper Embedding**: Override model config to always enable Whisper 
   conditioning (encode_whisper_embed=True) for optimal voice cloning
2. **Whisper Processor Integration**: Added robust Whisper processor loading 
   with fallback models (whisper-large-v3 → whisper-base)
3. **Reference Audio Waveform Processing**: Load and process reference audio
   waveforms for Whisper feature extraction at 16kHz
4. **Proper ChatML Structure**: Use <|AUDIO|> tokens for reference audio
   conditioning in the input sequence
5. **Dual Audio Pathway**: 
   - Whisper embeddings for reference audio conditioning (via <|AUDIO|> tokens)
   - DAC codes for generation context (via audio_ids)
6. **Training Pipeline Alignment**: Follow the same Whisper forcing pattern
   used in training (trainer.py/trainer_ddp.py) for consistent behavior

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
        
        # CRITICAL: Initialize audio codebook size like serve_engine.py
        self.audio_codebook_size = self.config.audio_codebook_size
        
        # Force enable Whisper embeddings for better voice cloning (override model config)
        # This follows the training pipeline pattern where Whisper is always enabled
        original_whisper_setting = self.config.encode_whisper_embed
        self.config.encode_whisper_embed = True
        logger.info(f"Whisper embedding: original={original_whisper_setting}, forced=True for voice cloning")
        
        # Load Whisper processor for reference audio conditioning
        logger.info("Loading Whisper processor for reference audio conditioning")
        whisper_processor = None
        whisper_models = [
            "openai/whisper-large-v3",
            "openai/whisper-base", 
            "openai/whisper-tiny"
        ]
        
        for model_name in whisper_models:
            try:
                whisper_processor = AutoProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                logger.info(f"✅ Successfully loaded Whisper processor: {model_name}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
        
        if whisper_processor is None:
            logger.warning("⚠️ No Whisper processor available - will use DAC-only mode")
        
        # Force enable Whisper embeddings if processor is available
        encode_whisper_embed = whisper_processor is not None
        if encode_whisper_embed != original_whisper_setting:
            logger.info(f"Overriding encode_whisper_embed: {original_whisper_setting} → {encode_whisper_embed}")
        
        # CRITICAL FIX: Setup collator with intelligent configuration
        logger.info(f"Setting up collator with Whisper processor: {whisper_processor is not None}")
        self.collator = HiggsAudioSampleCollator(
            whisper_processor=whisper_processor,
            audio_in_token_id=self.config.audio_in_token_idx,
            audio_out_token_id=self.config.audio_out_token_idx,
            audio_stream_bos_id=self.config.audio_stream_bos_id,
            audio_stream_eos_id=self.config.audio_stream_eos_id,
            encode_whisper_embed=encode_whisper_embed,  # Use the determined value
            pad_token_id=self.config.pad_token_id,
            return_audio_in_tokens=False,  # CRITICAL: serve_engine.py uses False
            use_delay_pattern=self.config.use_delay_pattern,
            round_to=1,  # CRITICAL: serve_engine.py uses fixed round_to=1
            audio_num_codebooks=self.config.audio_num_codebooks,
        )
        
        # Verify collator setup
        logger.info(f"Collator configuration:")
        logger.info(f"  - encode_whisper_embed: {self.collator.encode_whisper_embed} ({'adaptive' if encode_whisper_embed else 'disabled'})")
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
            logger.info(f"Using fixed max tokens: {self.max_new_tokens}")
            return self.max_new_tokens
            
        # CRITICAL FIX: More conservative token calculation for Arabic
        # Reduced from 150 WPM to 130 WPM for Arabic complexity
        word_count = len(target_text.split())
        char_count = len(target_text)
        
        # Arabic-specific rates (more conservative)
        arabic_wpm = 130  # Slower than English due to morphological complexity
        char_rate_factor = 8  # Characters per word approximation
        
        # Estimate duration based on both words and characters
        word_duration = (word_count / arabic_wpm) * 60  # seconds
        char_duration = (char_count / char_rate_factor) * 60 / arabic_wpm  # seconds
        
        # Use the longer estimate for better coverage
        estimated_duration = max(word_duration, char_duration)
        
        # CRITICAL: Reduced buffer factor to prevent excessive generation
        buffer_factor = 1.2  # Reduced from 1.5 to 1.2
        max_duration = estimated_duration * buffer_factor
        
        # Convert to tokens (25 Hz rate)
        calculated_tokens = int(max_duration * self.base_tokens_per_second)
        
        # CRITICAL: More restrictive bounds to prevent silence generation
        # Reduced maximum from 512 to 384 tokens
        min_tokens = 48   # ~2 seconds minimum
        max_tokens = 384  # ~15 seconds maximum (reduced from 512)
        bounded_tokens = max(min(calculated_tokens, max_tokens), min_tokens)
        
        logger.info(f"\n=== Adaptive Token Calculation ===")
        logger.info(f"Target text: '{target_text[:100]}{'...' if len(target_text) > 100 else ''}'")
        logger.info(f"Text stats: {word_count} words, {char_count} chars")
        logger.info(f"Duration estimates: word={word_duration:.1f}s, char={char_duration:.1f}s")
        logger.info(f"Selected duration: {estimated_duration:.1f}s (buffered: {max_duration:.1f}s)")
        logger.info(f"Token calculation: {calculated_tokens} -> bounded to {bounded_tokens}")
        logger.info(f"Expected audio duration: ~{bounded_tokens / self.base_tokens_per_second:.1f}s")
        
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
        
        # CRITICAL FIX: Pass text directly without any processing/filtering
        # Remove any text normalization as requested
        processed_ref_text = ref_text  # Direct pass-through, no filtering
        
        # Create user message with reference text and audio token placeholder
        # This follows the proper ChatML format for voice cloning conditioning
        user_ref_message = Message(
            role="user",
            content=f"{processed_ref_text} <|audio_bos|><|AUDIO|><|audio_eos|>"
        )
        
        # Create assistant message with reference audio using AudioContent
        # This is the key difference - using AudioContent instead of text confirmation
        assistant_ref_message = Message(
            role="assistant",
            content=AudioContent(audio_url=ref_audio_path)
        )
        
        # CRITICAL FIX: Pass target text directly without any processing/filtering
        # Remove any text normalization as requested  
        processed_target_text = target_text  # Direct pass-through, no filtering
        
        # Create user message with target text for generation
        user_target_message = Message(
            role="user", 
            content=processed_target_text
        )
        
        messages = [system_message, user_ref_message, assistant_ref_message, user_target_message]
        
        # Load and prepare reference audio waveform for Whisper processing
        logger.info(f"Loading reference audio waveform: {ref_audio_path}")
        if not os.path.exists(ref_audio_path):
            logger.error(f"Reference audio file not found: {ref_audio_path}")
            return messages, [], None, None
            
        try:
            # Load audio waveform
            waveform, sr = torchaudio.load(ref_audio_path)
            logger.info(f"Loaded audio: shape={waveform.shape}, sr={sr}")
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                logger.info(f"Converted to mono: shape={waveform.shape}")
            
            # Encode reference audio for DAC tokens (always needed)
            audio_tokens = self.audio_tokenizer.encode(ref_audio_path)
            audio_ids = [audio_tokens]
            logger.info(f"Audio tokens shape: {audio_tokens.shape}")
            
            # Prepare waveform for Whisper processing (conditional)
            ref_waveform = None
            if self.collator.whisper_processor is not None and self.collator.encode_whisper_embed:
                # Resample to 16kHz for Whisper if needed
                target_sr = 16000
                if sr != target_sr:
                    resampler = T.Resample(sr, target_sr)
                    waveform_16k = resampler(waveform)
                    logger.info(f"Resampled to {target_sr}Hz: shape={waveform_16k.shape}")
                else:
                    waveform_16k = waveform
                
                # Store waveform for Whisper processing
                ref_waveform = waveform_16k.squeeze(0)  # Remove channel dimension
                logger.info(f"Final waveform for Whisper: shape={ref_waveform.shape}, type={type(ref_waveform)}")
                
                # Validate the waveform
                if ref_waveform.numel() == 0:
                    logger.warning("Waveform is empty after processing! Disabling Whisper conditioning.")
                    ref_waveform = None
                elif torch.isnan(ref_waveform).any() or torch.isinf(ref_waveform).any():
                    logger.warning("Waveform contains NaN or Inf values! Disabling Whisper conditioning.")
                    ref_waveform = None
                else:
                    logger.info(f"Waveform validation passed: min={ref_waveform.min():.4f}, max={ref_waveform.max():.4f}")
                    sr = target_sr  # Update sample rate for return
            else:
                logger.info("Whisper processor not available or disabled - using DAC-only mode")
            
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            audio_ids = []
            ref_waveform = None
            sr = None
        
        return messages, audio_ids, ref_waveform, sr
    
    def _create_robust_sample(
        self,
        input_tokens: List[int],
        audio_ids: List[torch.Tensor],
        ref_waveform: Optional[torch.Tensor] = None,
        ref_sample_rate: Optional[int] = None
    ) -> ChatMLDatasetSample:
        """
        Create ChatMLDatasetSample with proper conditional waveform handling.
        
        This method handles two scenarios:
        1. Full pipeline: Whisper available + valid waveform -> include waveforms for conditioning
        2. DAC-only pipeline: Whisper unavailable or no waveform -> serve_engine.py pattern
        
        Args:
            input_tokens: Input text tokens
            audio_ids: List of audio token tensors
            ref_waveform: Reference audio waveform (optional)
            ref_sample_rate: Sample rate of reference waveform (optional)
            
        Returns:
            Properly configured ChatMLDatasetSample
        """
        # Process DAC tokens (always needed)
        if audio_ids:
            audio_ids_start = torch.tensor(
                np.cumsum(np.array([0] + [audio_ids[i].shape[1] for i in range(len(audio_ids))])),
                dtype=torch.long, device=self._device,
            )[:-1]
            audio_ids_concat = torch.cat([ele.cpu() for ele in audio_ids], dim=1)
        else:
            audio_ids_start = torch.tensor([], dtype=torch.long)
            audio_ids_concat = torch.zeros((self.config.audio_num_codebooks, 0), dtype=torch.long)
        
        # Check if we should use Whisper conditioning
        whisper_available = (
            self.collator.whisper_processor is not None and 
            self.collator.encode_whisper_embed
        )
        
        if whisper_available and ref_waveform is not None:
            # Full pipeline mode: include waveforms for Whisper conditioning
            logger.info(f"✅ Using full pipeline mode (Whisper + DAC): waveform shape={ref_waveform.shape}")
            
            return ChatMLDatasetSample(
                input_ids=torch.LongTensor(input_tokens),
                label_ids=None,
                audio_ids_concat=audio_ids_concat,
                audio_ids_start=audio_ids_start,
                audio_waveforms_concat=ref_waveform,
                audio_waveforms_start=torch.tensor([0], dtype=torch.long),
                audio_sample_rate=torch.tensor([ref_sample_rate or 16000], dtype=torch.float32),
                audio_speaker_indices=torch.tensor([0], dtype=torch.long),
            )
        else:
            # DAC-only mode: follow serve_engine.py pattern
            reason = "Whisper unavailable" if not whisper_available else "No waveform provided"
            logger.info(f"✅ Using DAC-only mode ({reason}): DAC tokens shape={audio_ids_concat.shape}")
            
            return ChatMLDatasetSample(
                input_ids=torch.LongTensor(input_tokens),
                label_ids=None,
                audio_ids_concat=audio_ids_concat,
                audio_ids_start=audio_ids_start,
                audio_waveforms_concat=torch.tensor([]),  # Empty tensor, not None
                audio_waveforms_start=torch.tensor([], dtype=torch.long),
                audio_sample_rate=torch.tensor([], dtype=torch.float32),
                audio_speaker_indices=torch.tensor([], dtype=torch.long),
            )
    
    def _validate_sample_for_collator(self, sample: ChatMLDatasetSample) -> ChatMLDatasetSample:
        """
        Defensive validation to ensure sample is compatible with collator configuration.
        
        Args:
            sample: The ChatMLDatasetSample to validate
            
        Returns:
            Validated or corrected ChatMLDatasetSample
        """
        try:
            # Check for Whisper configuration mismatch
            if self.collator.encode_whisper_embed:
                audio_in_mask = sample.input_ids == self.collator.audio_in_token_id
                has_audio_tokens = audio_in_mask.any()
                
                if has_audio_tokens:
                    # Audio tokens present, validate waveform data
                    if (sample.audio_waveforms_concat is None or 
                        sample.audio_waveforms_start is None or
                        len(sample.audio_waveforms_concat) == 0):
                        
                        logger.warning("Sample has audio tokens but missing waveforms for Whisper. Converting to DAC-only mode.")
                        
                        # Convert to DAC-only compatible sample
                        return ChatMLDatasetSample(
                            input_ids=sample.input_ids,
                            label_ids=sample.label_ids,
                            audio_ids_concat=sample.audio_ids_concat if sample.audio_ids_concat is not None else torch.zeros((self.config.audio_num_codebooks, 0), dtype=torch.long),
                            audio_ids_start=sample.audio_ids_start if sample.audio_ids_start is not None else torch.tensor([], dtype=torch.long),
                            audio_waveforms_concat=torch.tensor([]),
                            audio_waveforms_start=torch.tensor([], dtype=torch.long),
                            audio_sample_rate=torch.tensor([], dtype=torch.float32),
                            audio_speaker_indices=torch.tensor([], dtype=torch.long),
                        )
            
            return sample
            
        except Exception as e:
            logger.error(f"Error validating sample: {e}")
            return sample
    
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
            
            # CRITICAL FIX: Use robust sample creation with conditional waveform handling
            curr_sample = self._create_robust_sample(
                input_tokens=input_tokens,
                audio_ids=audio_ids,
                ref_waveform=ref_waveform,
                ref_sample_rate=ref_sample_rate
            )
            
            # CRITICAL: Validate sample for collator compatibility
            curr_sample = self._validate_sample_for_collator(curr_sample)
            
            # Collate data
            batch_data = self.collator([curr_sample])
            batch = asdict(batch_data)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.contiguous().to(self._device)
            
            # Calculate adaptive max tokens based on target text
            adaptive_max_tokens = self.calculate_adaptive_max_tokens(target_text)
            
            # CRITICAL FIX: Use serve_engine.py generation parameters
            logger.info(f"Starting generation with {adaptive_max_tokens} max tokens...")
            
            # serve_engine.py default stop strings
            stop_strings = ["<|end_of_text|>", "<|eot_id|>"]
            
            # CRITICAL: Match serve_engine.py generation exactly
            outputs = self.model.generate(
                **batch,
                max_new_tokens=adaptive_max_tokens,
                use_cache=True,
                stop_strings=stop_strings,
                tokenizer=self.tokenizer,
                do_sample=False if temperature == 0.0 else True,  # serve_engine.py pattern
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                ras_win_len=7,  # serve_engine.py default
                ras_win_max_num_repeat=2,  # serve_engine.py default
                seed=seed,
            )
            
            # CRITICAL FIX: Process audio outputs using serve_engine.py EXACT pattern
            if len(outputs[1]) > 0:
                wv_list = []
                for i, output_audio in enumerate(outputs[1]):
                    logger.info(f"Raw audio output {i}: shape={output_audio.shape}, first_tokens={output_audio[:, :5] if output_audio.shape[1] > 5 else output_audio}")
                    
                    # CRITICAL: serve_engine.py exact one-line pattern
                    vq_code = revert_delay_pattern(output_audio).clip(0, self.audio_codebook_size - 1)[:, 1:-1]
                    logger.info(f"After serve_engine.py processing {i}: shape={vq_code.shape}")
                    
                    # Decode audio
                    wv_numpy = self.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]
                    wv_list.append(wv_numpy)
                    
                    # Log processing results
                    if isinstance(wv_numpy, torch.Tensor):
                        wv_numpy_debug = wv_numpy.detach().cpu().numpy()
                    else:
                        wv_numpy_debug = wv_numpy
                    
                    energy = (wv_numpy_debug ** 2).mean()
                    logger.info(f"Audio {i}: energy={energy:.2e}, duration={len(wv_numpy_debug)/24000:.2f}s")
                
                # Concatenate all audio
                wv_numpy = np.concatenate(wv_list)
                
                # Final validation
                final_energy = (wv_numpy ** 2).mean()
                logger.info(f"Final concatenated audio: energy={final_energy:.2e}, duration={len(wv_numpy)/24000:.2f}s")
                
                if final_energy < 1e-8:
                    logger.error(f"🚨 CRITICAL: Still generating silence - check model/tokenizer compatibility")
                    return None, None, None
                
                sample_rate = 24000  # Higgs Audio output sample rate
                
                # Get text output
                text_result = self.tokenizer.decode(outputs[0][0])
                
                logger.info("=== Generation Output Text ===")
                logger.info(text_result)
                
                return wv_numpy, sample_rate, text_result
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
        CRITICAL: Added comprehensive audio validation and debugging info.
        
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
        
        # CRITICAL: Validate generated audio before saving
        logger.info(f"\n=== Audio Validation for Sample {sample_id} ===")
        logger.info(f"Generated audio shape: {generated_waveform.shape}")
        logger.info(f"Generated audio dtype: {generated_waveform.dtype}")
        logger.info(f"Sample rate: {sample_rate}")
        
        # Check for common issues
        audio_duration = len(generated_waveform) / sample_rate
        audio_min = generated_waveform.min()
        audio_max = generated_waveform.max()
        audio_mean = generated_waveform.mean()
        audio_std = generated_waveform.std()
        audio_energy = (generated_waveform ** 2).mean()
        
        logger.info(f"Audio duration: {audio_duration:.2f} seconds")
        logger.info(f"Audio range: [{audio_min:.6f}, {audio_max:.6f}]")
        logger.info(f"Audio statistics: mean={audio_mean:.6f}, std={audio_std:.6f}")
        logger.info(f"Audio energy: {audio_energy:.2e}")
        
        # Detect potential issues
        issues = []
        if audio_energy < 1e-6:
            issues.append(f"Very low energy ({audio_energy:.2e}) - likely silence")
        if abs(audio_mean) > 0.1:
            issues.append(f"High DC offset ({audio_mean:.6f})")
        if audio_std < 1e-4:
            issues.append(f"Very low variance ({audio_std:.6f}) - likely constant signal")
        if np.isnan(generated_waveform).any():
            issues.append("Contains NaN values")
        if np.isinf(generated_waveform).any():
            issues.append("Contains infinite values")
        
        if issues:
            logger.warning(f"⚠️ Audio issues detected: {', '.join(issues)}")
        else:
            logger.info(f"✅ Audio validation passed - no issues detected")
        
        # Save generated audio
        try:
            sf.write(generated_file, generated_waveform, sample_rate)
            logger.info(f"✅ Saved generated audio to {generated_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save generated audio: {e}")
            generated_file = None
        
        # Copy and validate reference audio (CRITICAL: Added reference audio saving)
        if os.path.exists(ref_audio_path):
            try:
                # Load and validate reference audio
                ref_waveform, ref_sr = sf.read(ref_audio_path)
                ref_duration = len(ref_waveform) / ref_sr
                ref_energy = (ref_waveform ** 2).mean()
                
                logger.info(f"Reference audio: duration={ref_duration:.2f}s, energy={ref_energy:.2e}, sr={ref_sr}")
                
                # Copy reference audio to output directory for easy comparison
                shutil.copy2(ref_audio_path, reference_file)
                logger.info(f"✅ Saved reference audio to {reference_file}")
                
                # Compare reference vs generated
                logger.info(f"\n=== Audio Comparison ===")
                logger.info(f"Duration ratio (gen/ref): {audio_duration/ref_duration:.2f}")
                logger.info(f"Energy ratio (gen/ref): {audio_energy/ref_energy:.2e}")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to process reference audio: {e}")
                reference_file = None
        else:
            logger.warning(f"⚠️ Reference audio file not found: {ref_audio_path}")
            reference_file = None
        
        return {
            "generated_audio": generated_file,
            "reference_audio": reference_file,
            "sample_rate": sample_rate,
            "audio_duration": audio_duration,
            "audio_energy": audio_energy,
            "validation_issues": issues
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
                # Save both generated and reference audio (CRITICAL: Added reference audio saving)
                file_info = self.save_reference_and_generated_audio(
                    ref_audio_path, waveform, sample_rate, output_dir, i, speaker_id
                )
                
                # Add comprehensive logging for debugging silence issues
                logger.info(f"✅ Sample {i} completed successfully:")
                logger.info(f"   - Generated audio: {file_info['generated_audio']}")
                logger.info(f"   - Reference audio: {file_info['reference_audio']}")
                logger.info(f"   - Audio duration: {len(waveform) / sample_rate:.2f}s")
                logger.info(f"   - Audio stats: min={waveform.min():.4f}, max={waveform.max():.4f}, mean={waveform.mean():.4f}")
                
                # Check for silence (potential issue indicator)
                audio_energy = (waveform ** 2).mean()
                if audio_energy < 1e-6:
                    logger.warning(f"⚠️ Sample {i}: Very low audio energy detected ({audio_energy:.2e}) - possible silence issue")
                else:
                    logger.info(f"✅ Sample {i}: Good audio energy level ({audio_energy:.2e})")
                
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
    
    def validate_pipeline_configuration(self) -> dict:
        """
        Comprehensive validation of the Arabic TTS pipeline configuration.
        CRITICAL: Helps identify why generated audio contains silence.
        
        Returns:
            Dictionary with validation results and recommendations
        """
        logger.info("\n" + "="*80)
        logger.info("🔍 ARABIC TTS PIPELINE VALIDATION")
        logger.info("="*80)
        
        validation_results = {
            "whisper_integration": False,
            "audio_tokenizer": False,
            "model_config": False,
            "special_tokens": False,
            "issues": [],
            "recommendations": []
        }
        
        # 1. Whisper Integration Check
        logger.info("\n1️⃣ Whisper Integration Status:")
        if self.collator.whisper_processor is not None:
            logger.info("   ✅ Whisper processor loaded successfully")
            validation_results["whisper_integration"] = True
        else:
            logger.error("   ❌ Whisper processor NOT loaded - CRITICAL for voice cloning")
            validation_results["issues"].append("Missing Whisper processor")
            validation_results["recommendations"].append("Install transformers and ensure Whisper model availability")
        
        logger.info(f"   - encode_whisper_embed: {self.collator.encode_whisper_embed}")
        logger.info(f"   - Original config setting: {getattr(self.config, '_original_whisper_embed', 'unknown')}")
        
        # 2. Audio Tokenizer Check
        logger.info("\n2️⃣ Audio Tokenizer Status:")
        try:
            tokenizer_device = str(self.audio_tokenizer.device) if hasattr(self.audio_tokenizer, 'device') else "unknown"
            logger.info(f"   ✅ Audio tokenizer loaded on device: {tokenizer_device}")
            logger.info(f"   - Codebook size: {self.audio_tokenizer.codebook_size}")
            validation_results["audio_tokenizer"] = True
        except Exception as e:
            logger.error(f"   ❌ Audio tokenizer issue: {e}")
            validation_results["issues"].append(f"Audio tokenizer error: {e}")
        
        # 3. Model Configuration Check
        logger.info("\n3️⃣ Model Configuration:")
        config_checks = [
            ("encode_whisper_embed", self.config.encode_whisper_embed),
            ("audio_num_codebooks", self.config.audio_num_codebooks),
            ("audio_codebook_size", self.config.audio_codebook_size),
            ("use_delay_pattern", self.config.use_delay_pattern),
            ("audio_stream_bos_id", self.config.audio_stream_bos_id),
            ("audio_stream_eos_id", self.config.audio_stream_eos_id),
        ]
        
        all_config_ok = True
        for param_name, param_value in config_checks:
            logger.info(f"   - {param_name}: {param_value}")
            if param_value is None:
                logger.warning(f"     ⚠️ {param_name} is None")
                all_config_ok = False
        
        validation_results["model_config"] = all_config_ok
        
        # 4. Special Tokens Check
        logger.info("\n4️⃣ Special Tokens Configuration:")
        special_token_checks = [
            ("audio_in_token_idx", self.config.audio_in_token_idx),
            ("audio_out_token_idx", self.config.audio_out_token_idx),
            ("audio_stream_bos_id", self.config.audio_stream_bos_id),
            ("audio_stream_eos_id", self.config.audio_stream_eos_id),
        ]
        
        tokens_ok = True
        for token_name, token_id in special_token_checks:
            if token_id is not None:
                logger.info(f"   ✅ {token_name}: {token_id}")
            else:
                logger.error(f"   ❌ {token_name}: None")
                tokens_ok = False
                validation_results["issues"].append(f"Missing {token_name}")
        
        validation_results["special_tokens"] = tokens_ok
        
        # 5. Device Compatibility Check
        logger.info("\n5️⃣ Device Configuration:")
        logger.info(f"   - Model device: {self._device}")
        logger.info(f"   - Static KV cache: {self.use_static_kv_cache}")
        
        if self._device == "mps" and self.use_static_kv_cache:
            logger.warning("   ⚠️ MPS with static KV cache may cause issues")
            validation_results["recommendations"].append("Disable static KV cache on MPS")
        
        # 6. Overall Assessment
        logger.info("\n6️⃣ Overall Assessment:")
        critical_components = [
            validation_results["whisper_integration"],
            validation_results["audio_tokenizer"],
            validation_results["special_tokens"]
        ]
        
        if all(critical_components):
            logger.info("   🎉 All critical components validated successfully")
            logger.info("   📝 If still experiencing silence, check audio token processing")
        else:
            logger.error("   💥 Critical validation failures detected")
            logger.error("   🔧 Address these issues before proceeding")
        
        # 7. Debugging Recommendations
        if validation_results["issues"] or not all(critical_components):
            logger.info("\n🔧 DEBUGGING RECOMMENDATIONS:")
            for i, rec in enumerate(validation_results["recommendations"], 1):
                logger.info(f"   {i}. {rec}")
            
            if not validation_results["whisper_integration"]:
                logger.info("   - Voice similarity will be severely degraded without Whisper")
            
            logger.info("   - Check that audio tokens are not being corrupted in generation")
            logger.info("   - Verify special token usage matches training patterns")
            logger.info("   - Ensure reference audio files are accessible and valid")
        
        logger.info("\n" + "="*80)
        return validation_results


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
    
    # CRITICAL: Validate pipeline configuration for debugging silence issues
    logger.info("\n🔍 Running comprehensive pipeline validation...")
    validation_results = inference_engine.validate_pipeline_configuration()
    
    # Check for critical issues that would cause silence generation
    critical_issues = validation_results.get("issues", [])
    if critical_issues:
        logger.error(f"\n🚨 CRITICAL ISSUES DETECTED: {critical_issues}")
        logger.error("🗺️ These issues may cause silence generation or poor voice cloning quality")
        logger.warning("⚠️ Proceeding with known issues - results may be suboptimal")
    else:
        logger.info("✅ Pipeline validation passed - proceeding with generation")
    
    # Process ChatML file
    results = inference_engine.process_chatml_file(
        chatml_file=chatml_file,
        output_dir=output_dir,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed
    )
    
    # Print comprehensive summary with debugging insights
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    total = len(results)
    
    logger.info("\n" + "="*80)
    logger.info("📊 ARABIC TTS PROCESSING SUMMARY")
    logger.info("="*80)
    logger.info(f"📈 Overall Results: {successful}/{total} samples successful ({failed} failed)")
    logger.info(f"📋 Generated audio files saved in: {output_dir}")
    
    # Analyze results for silence issues
    if successful > 0:
        logger.info(f"\n🔍 Quality Analysis:")
        energy_stats = []
        duration_stats = []
        
        for result in results:
            if result["status"] == "success" and "audio_energy" in result:
                energy_stats.append(result["audio_energy"])
            if result["status"] == "success" and "audio_duration" in result:
                duration_stats.append(result["audio_duration"])
        
        if energy_stats:
            avg_energy = sum(energy_stats) / len(energy_stats)
            low_energy_count = sum(1 for e in energy_stats if e < 1e-6)
            logger.info(f"   - Average audio energy: {avg_energy:.2e}")
            logger.info(f"   - Low energy samples: {low_energy_count}/{len(energy_stats)}")
            
            if low_energy_count > 0:
                logger.warning(f"   ⚠️ {low_energy_count} samples have very low energy (possible silence)")
            else:
                logger.info(f"   ✅ All samples have good energy levels")
        
        if duration_stats:
            avg_duration = sum(duration_stats) / len(duration_stats)
            logger.info(f"   - Average audio duration: {avg_duration:.2f}s")
    
    # Debugging recommendations if issues found
    if failed > 0 or (successful > 0 and any(r.get("validation_issues", []) for r in results)):
        logger.info(f"\n🔧 DEBUGGING RECOMMENDATIONS:")
        logger.info(f"   1. Check validation report above for pipeline issues")
        logger.info(f"   2. Verify reference audio files are accessible and valid")
        logger.info(f"   3. Ensure Whisper processor is properly loaded")
        logger.info(f"   4. Review audio token processing for boundary corruption")
        logger.info(f"   5. Check special token usage matches training patterns")
        
        if critical_issues:
            logger.info(f"   6. Address critical issues: {critical_issues}")
    
    logger.info("\n✨ Processing completed - check generated audio files for quality assessment")
    logger.info("="*80)


if __name__ == "__main__":
    main()