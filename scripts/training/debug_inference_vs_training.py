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

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from boson_multimodal.model.higgs_audio import HiggsAudioModel
from boson_multimodal.audio_processing.higgs_audio_tokenizer import HiggsAudioTokenizer
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample
from boson_multimodal.data_collator.higgs_audio_sample_collator import HiggsAudioSampleCollator
from transformers import AutoTokenizer
import torchaudio

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    audio_tokenizer = HiggsAudioTokenizer.from_pretrained(args.audio_tokenizer_path)
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
    
    # Load sample audio files
    logger.info("🎵 Loading sample audio files...")
    ref_audio = load_sample_audio("/Users/vikram.solanki/Projects/tts/higgs-audio/arabic_english_processed_20250109_222259/samples/sample_0_ref.wav")
    target_audio = load_sample_audio("/Users/vikram.solanki/Projects/tts/higgs-audio/arabic_english_processed_20250109_222259/samples/sample_0_target.wav")
    
    if ref_audio is None or target_audio is None:
        logger.error("❌ Could not load sample audio files")
        return
    
    logger.info(f"  Reference audio shape: {ref_audio.shape}")
    logger.info(f"  Target audio shape: {target_audio.shape}")
    
    # Tokenize audio
    logger.info("🎯 Tokenizing audio...")
    with torch.no_grad():
        ref_tokens = audio_tokenizer.encode(ref_audio.unsqueeze(0))  # Add batch dim
        target_tokens = audio_tokenizer.encode(target_audio.unsqueeze(0))
    
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
    logger.info("📝 Creating ChatML sample...")
    
    # Simulate the training data structure
    messages = [
        {"role": "user", "content": "Please clone this voice and say: Hello world"},
        {"role": "assistant", "content": "Hello world"}
    ]
    
    # Create sample
    sample = ChatMLDatasetSample(
        messages=messages,
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
    logger.info("🔗 Using collator...")
    collator = HiggsAudioSampleCollator(
        audio_tokenizer=audio_tokenizer,
        text_tokenizer=text_tokenizer,
        round_to=8,
    )
    
    batch = collator([sample])
    
    logger.info(f"📦 BATCH ANALYSIS:")
    logger.info(f"  Input IDs: {batch.input_ids.shape}")
    logger.info(f"  Audio in IDs: {batch.audio_in_ids.shape}")
    logger.info(f"  Audio out IDs: {batch.audio_out_ids.shape if hasattr(batch, 'audio_out_ids') else 'None'}")
    logger.info(f"  Label audio IDs: {batch.label_audio_ids.shape if hasattr(batch, 'label_audio_ids') else 'None'}")
    
    # Analyze labels for padding
    if hasattr(batch, 'label_audio_ids'):
        labels = batch.label_audio_ids
        logger.info(f"🚨 LABEL ANALYSIS:")
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
    
    logger.info("✅ === INFERENCE DEBUG COMPLETE ===")

def main():
    parser = argparse.ArgumentParser(description="Debug inference vs training pipeline")
    parser.add_argument("--model_path", type=str, default="bosonai/higgs-audio-v2", help="Path to model")
    parser.add_argument("--audio_tokenizer_path", type=str, default="bosonai/higgs-audio-v2-tokenizer", help="Path to audio tokenizer")
    
    args = parser.parse_args()
    
    debug_inference_pipeline(args)

if __name__ == "__main__":
    main()
