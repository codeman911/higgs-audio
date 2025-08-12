#!/usr/bin/env python3
"""
Debug script to compare inference vs training data handling
to identify why training loss is collapsing so rapidly.
"""

import torch
import logging
from pathlib import Path
import sys
import argparse
import json
import os
import random
from typing import List, Dict, Tuple

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from boson_multimodal.model.higgs_audio import HiggsAudioModel
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent, TextContent
from transformers import AutoTokenizer
import torchaudio

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_chatml_samples(json_file: str, max_samples: int = 1) -> List[Dict]:
    """Load ChatML samples from JSON file with robust structure handling."""
    logger.info(f"Loading ChatML samples from: {json_file}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, list):
        # JSON file contains array of samples directly
        samples = data
        logger.info(f"Loaded JSON as direct array with {len(samples)} samples")
    elif isinstance(data, dict):
        # JSON file contains dictionary with 'samples' key
        samples = data.get('samples', [])
        if not samples:
            # Try other common keys
            samples = data.get('data', [])
            if not samples:
                samples = data.get('items', [])
        logger.info(f"Loaded JSON as dictionary with {len(samples)} samples")
    else:
        raise ValueError(f"Unexpected JSON structure in {json_file}. Expected list or dict, got {type(data)}")
    
    if not samples:
        raise ValueError(f"No samples found in {json_file}")
    
    # Select first few samples
    selected_samples = samples[:max_samples]
    logger.info(f"Selected {len(selected_samples)} samples for debugging")
    
    return selected_samples

def convert_chatml_to_messages(chatml_sample: Dict, audio_tokenizer) -> Tuple[List[Message], List[torch.Tensor], str, str]:
    """
    Convert ChatML sample to proper Message format matching training data structure.
    Returns: messages, audio_ids, reference_audio_path, target_audio_path
    """
    messages = []
    audio_ids = []
    ref_audio_path = None
    target_audio_path = None
    
    chatml_messages = chatml_sample.get('messages', [])
    if not chatml_messages:
        raise ValueError("No messages found in ChatML sample")
    
    for msg in chatml_messages:
        role = msg.get('role')
        content = msg.get('content')
        
        if role == 'user':
            # User message with text and reference audio
            if isinstance(content, list):
                message_content = []
                
                for item in content:
                    if item.get('type') == 'text':
                        # Add text content
                        text = item.get('text', '')
                        message_content.append(TextContent(text=text))
                        logger.info(f"Added text: {text[:50]}...")
                        
                    elif item.get('type') == 'audio':
                        # Add reference audio
                        audio_url = item.get('audio_url', '')
                        if audio_url and os.path.exists(audio_url):
                            try:
                                logger.info(f"Loading reference audio: {audio_url}")
                                ref_audio_path = audio_url
                                
                                # Tokenize reference audio for audio_ids
                                ref_audio_tokens = audio_tokenizer.encode(audio_url)
                                logger.info(f"✅ Reference audio tokenized: {ref_audio_tokens.shape}")
                                
                                # Add to message content using audio_url (not tokens)
                                message_content.append(AudioContent(audio_url=audio_url))
                                # Add tokens to audio_ids for generation
                                audio_ids.append(ref_audio_tokens)
                                
                            except Exception as e:
                                logger.error(f"❌ Error processing reference audio {audio_url}: {str(e)}")
                                continue
                        else:
                            logger.warning(f"Reference audio not found or empty: {audio_url}")
                
                if message_content:
                    messages.append(Message(role="user", content=message_content))
        
        elif role == 'assistant':
            # Extract target audio path from assistant message
            if isinstance(content, list):
                for item in content:
                    if item.get('type') == 'audio':
                        target_audio_path = item.get('audio_url', '')
                        if target_audio_path:
                            logger.info(f"Found target audio: {target_audio_path}")
    
    return messages, audio_ids, ref_audio_path, target_audio_path

def load_sample_audio(audio_path: str, sample_rate: int = 22050):
    """Load and resample audio file"""
    try:
        waveform, orig_sr = torchaudio.load(audio_path)
        if orig_sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
            waveform = resampler(waveform)
        return waveform.squeeze(0)  # Remove channel dim if mono
    except Exception as e:
        logger.error(f"Error loading {audio_path}: {e}")
        return None

def debug_inference_pipeline(args):
    """Run inference pipeline with detailed logging"""
    
    logger.info("🔍 === INFERENCE PIPELINE DEBUG ===")
    
    # Load tokenizers and model
    logger.info("📥 Loading model and tokenizers...")
    text_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    audio_tokenizer = load_higgs_audio_tokenizer(args.audio_tokenizer_path)
    model = HiggsAudioModel.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    
    # Investigate audio tokenizer attributes
    logger.info(f"🔍 AUDIO TOKENIZER INVESTIGATION:")
    logger.info(f"  Type: {type(audio_tokenizer)}")
    tokenizer_attrs = [attr for attr in dir(audio_tokenizer) if 'pad' in attr.lower()]
    logger.info(f"  Pad-related attributes: {tokenizer_attrs}")
    
    # Check all possible pad token attributes
    pad_candidates = []
    for attr in ['pad_id', 'pad_token_id', 'padding_idx', 'pad_index']:
        if hasattr(audio_tokenizer, attr):
            pad_val = getattr(audio_tokenizer, attr)
            pad_candidates.append(f"{attr}={pad_val}")
    logger.info(f"  Pad candidates: {pad_candidates}")
    
    # Check tokenizer vocab size and special tokens
    logger.info(f"  Vocab size: {getattr(audio_tokenizer, 'vocab_size', 'unknown')}")
    logger.info(f"  N_q (codebooks): {getattr(audio_tokenizer, 'n_q', 'unknown')}")
    
    # Load real ChatML samples like in training
    logger.info("📄 Loading real ChatML samples...")
    try:
        chatml_samples = load_chatml_samples(args.chatml_file, max_samples=1)
        chatml_sample = chatml_samples[0]
        logger.info(f"✅ Loaded ChatML sample with {len(chatml_sample.get('messages', []))} messages")
        
        # Convert ChatML to messages like inference pipeline
        messages, audio_ids, ref_audio_path, target_audio_path = convert_chatml_to_messages(chatml_sample, audio_tokenizer)
        logger.info(f"✅ Converted to {len(messages)} messages, {len(audio_ids)} audio references")
        
        if not ref_audio_path or not target_audio_path:
            logger.error("❌ Could not extract reference or target audio paths from ChatML")
            return
            
        logger.info(f"📁 Reference audio: {ref_audio_path}")
        logger.info(f"📁 Target audio: {target_audio_path}")
        
    except Exception as e:
        logger.error(f"❌ Error loading ChatML samples: {e}")
        logger.info("Falling back to hardcoded audio files...")
        ref_audio_path = "/Users/vikram.solanki/Projects/tts/higgs-audio/arabic_english_processed_20250109_222259/samples/sample_0_ref.wav"
        target_audio_path = "/Users/vikram.solanki/Projects/tts/higgs-audio/arabic_english_processed_20250109_222259/samples/sample_0_target.wav"
    
    # Load and analyze reference and target audio
    logger.info("🎵 Loading and analyzing audio files...")
    ref_audio = load_sample_audio(ref_audio_path)
    target_audio = load_sample_audio(target_audio_path)
    
    if ref_audio is None or target_audio is None:
        logger.error("❌ Could not load sample audio files")
        return
    
    logger.info(f"  Reference audio shape: {ref_audio.shape}")
    logger.info(f"  Target audio shape: {target_audio.shape}")
    
    # Tokenize audio directly with audio tokenizer (using file paths)
    logger.info("🎯 Tokenizing audio with audio tokenizer...")
    with torch.no_grad():
        ref_tokens = audio_tokenizer.encode(ref_audio_path)  # Pass file path
        target_tokens = audio_tokenizer.encode(target_audio_path)  # Pass file path
    
    logger.info(f"  Reference tokens shape: {ref_tokens.shape}")
    logger.info(f"  Target tokens shape: {target_tokens.shape}")
    logger.info(f"  Reference tokens sample: {ref_tokens[:, :10]}")
    logger.info(f"  Target tokens sample: {target_tokens[:, :10]}")
    
    # Analyze token distribution
    ref_unique, ref_counts = torch.unique(ref_tokens, return_counts=True)
    target_unique, target_counts = torch.unique(target_tokens, return_counts=True)
    
    logger.info(f"🔍 TOKEN DISTRIBUTION ANALYSIS:")
    logger.info(f"  Reference: {len(ref_unique)} unique tokens, range {ref_unique.min()}-{ref_unique.max()}")
    logger.info(f"  Target: {len(target_unique)} unique tokens, range {target_unique.min()}-{target_unique.max()}")
    
    # Check for token 1025
    ref_1025_count = (ref_tokens == 1025).sum().item()
    target_1025_count = (target_tokens == 1025).sum().item()
    ref_total = ref_tokens.numel()
    target_total = target_tokens.numel()
    
    logger.info(f"🚨 TOKEN 1025 ANALYSIS:")
    logger.info(f"  Reference: {ref_1025_count}/{ref_total} ({ref_1025_count/ref_total*100:.1f}%)")
    logger.info(f"  Target: {target_1025_count}/{target_total} ({target_1025_count/target_total*100:.1f}%)")
    
    # Check sequence endings
    logger.info(f"🔍 SEQUENCE ENDINGS:")
    logger.info(f"  Reference last 10 tokens: {ref_tokens[:, -10:]}")
    logger.info(f"  Target last 10 tokens: {target_tokens[:, -10:]}")
    
    # Create ChatML sample like in training
    logger.info("📝 Creating ChatML sample like training pipeline...")
    
    # Create sample using the exact same structure as training
    sample = ChatMLDatasetSample(
        messages=[
            {"role": "user", "content": "Please clone this voice and say: Hello world"},
            {"role": "assistant", "content": "Hello world"}
        ],
        reference_audio=ref_audio,
        target_audio=target_audio,
        audio_tokenizer=audio_tokenizer,
        text_tokenizer=text_tokenizer,
    )
    
    logger.info(f"📋 CHATML SAMPLE ANALYSIS:")
    logger.info(f"  Input IDs shape: {sample.input_ids.shape}")
    logger.info(f"  Audio in IDs shape: {sample.audio_in_ids.shape}")
    logger.info(f"  Audio out IDs shape: {sample.audio_out_ids.shape if hasattr(sample, 'audio_out_ids') else 'None'}")
    logger.info(f"  Label audio IDs shape: {sample.label_audio_ids.shape if hasattr(sample, 'label_audio_ids') else 'None'}")
    
    # Use collator like in training
    logger.info("🔗 Using collator like training...")
    collator = HiggsAudioSampleCollator(
        audio_tokenizer=audio_tokenizer,
        text_tokenizer=text_tokenizer,
        round_to=8,
    )
    
    batch = collator([sample])
    
    logger.info(f"📦 BATCH ANALYSIS (POST-COLLATOR):")
    logger.info(f"  Input IDs: {batch.input_ids.shape}")
    logger.info(f"  Audio in IDs: {batch.audio_in_ids.shape}")
    logger.info(f"  Audio out IDs: {batch.audio_out_ids.shape if hasattr(batch, 'audio_out_ids') else 'None'}")
    logger.info(f"  Label audio IDs: {batch.label_audio_ids.shape if hasattr(batch, 'label_audio_ids') else 'None'}")
    
    # Analyze labels for padding
    if hasattr(batch, 'label_audio_ids'):
        labels = batch.label_audio_ids
        logger.info(f"🚨 LABEL ANALYSIS (CRITICAL FOR TRAINING COLLAPSE):")
        logger.info(f"  Labels shape: {labels.shape}")
        logger.info(f"  Labels sample: {labels[:, :10]}")
        
        # Check for -100 (ignore tokens)
        ignore_count = (labels == -100).sum().item()
        total_labels = labels.numel()
        logger.info(f"  Ignore tokens (-100): {ignore_count}/{total_labels} ({ignore_count/total_labels*100:.1f}%)")
        
        # Check for token 1025 in labels
        label_1025_count = (labels == 1025).sum().item()
        non_ignore = (labels != -100).sum().item()
        logger.info(f"  Token 1025 in labels: {label_1025_count}/{non_ignore} ({label_1025_count/max(non_ignore,1)*100:.1f}%) of non-ignore")
        
        # Check sequence endings in labels
        logger.info(f"🔍 LABEL SEQUENCE ENDINGS:")
        logger.info(f"  Labels last 10 tokens: {labels[:, -10:]}")
        
        # Compare direct tokenization vs collated labels
        logger.info(f"🔍 DIRECT TOKENIZATION vs COLLATED LABELS:")
        logger.info(f"  Direct target tokens: {target_tokens.shape}")
        logger.info(f"  Collated label shape: {labels.shape}")
        
        # Check if collated labels match direct tokenization
        if target_tokens.shape == labels.shape:
            match_count = (target_tokens == labels).sum().item()
            mismatch_count = ((target_tokens != labels) & (labels != -100)).sum().item()
            logger.info(f"  Exact matches: {match_count}/{labels.numel()}")
            logger.info(f"  Mismatches (non-ignore): {mismatch_count}")
            if mismatch_count > 0:
                logger.warning(f"⚠️  LABELS DON'T MATCH DIRECT TOKENIZATION - POTENTIAL ISSUE!")
        else:
            logger.warning(f"⚠️  SHAPE MISMATCH: direct={target_tokens.shape} vs collated={labels.shape}")
    
    logger.info("✅ === INFERENCE DEBUG COMPLETE ===")

def main():
    parser = argparse.ArgumentParser(description="Debug inference vs training pipeline")
    parser.add_argument("--model_path", type=str, default="bosonai/higgs-audio-v2", help="Path to model")
    parser.add_argument("--audio_tokenizer_path", type=str, default="bosonai/higgs-audio-v2-tokenizer", help="Path to audio tokenizer")
    parser.add_argument("--chatml_file", type=str, default="arabic_english_processed_20250109_222259/processed_samples.json", 
                       help="Path to ChatML samples JSON file")
    
    args = parser.parse_args()
    
    debug_inference_pipeline(args)

if __name__ == "__main__":
    main()
