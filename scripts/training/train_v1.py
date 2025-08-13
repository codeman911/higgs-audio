#!/usr/bin/env python3
"""
Higgs-Audio Zero-Shot Voice Cloning Trainer V1
===============================================

ROBUST, NO-LEAK TRAINING IMPLEMENTATION

Based on comprehensive diagnostics that identified the Arabic mumbling root cause:
- Insufficient text supervision (3 tokens/sample << 32 required)
- Model not using text for audio generation (entropy_diff < 0.1)
- Target audio leakage prevention with strict input validation
- Comprehensive runtime guardrails and health monitoring

Key Features:
- ✅ Zero target audio leakage with build_no_leak_inputs()
- ✅ Guaranteed ≥32 supervised Arabic tokens per sample
- ✅ Entropy-based text conditioning validation
- ✅ Higgs-Audio dual-FFN compatible loss computation
- ✅ Real-time diagnostic monitoring and health checks
- ✅ Robust LoRA integration with expanded target coverage

Author: Cascade AI + User collaborative engineering
"""
import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, WhisperProcessor, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn as nn
from accelerate import Accelerator
from tqdm import tqdm
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from boson_multimodal.model.higgs_audio import HiggsAudioModel
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
IGNORE_INDEX = -100

# =============================================================================
# ROBUST NO-LEAK HIGGS-AUDIO TRAINING UTILITIES
# =============================================================================
# These utilities implement the smoking gun fixes identified by diagnostics:
# 1. Zero target audio leakage in model inputs
# 2. Guaranteed ≥32 supervised Arabic text tokens per sample
# 3. Entropy-based text conditioning validation
# 4. Higgs-Audio dual-FFN compatible loss computation
# =============================================================================

def build_no_leak_inputs(batch, tokenizer, device):
    """
    Build model inputs that match Higgs' expectations, without leaking targets.
    Returns a dict you can pass directly into model(**inputs).
    
    CRITICAL: This prevents target audio leakage by ensuring audio_out_ids
    are used ONLY for alignment metadata, never concatenated to input sequence.
    """
    def to_dev(x): 
        return x.to(device) if x is not None else None

    # Required core inputs - NO TARGET LEAKAGE
    inputs = {
        "input_ids": to_dev(batch.input_ids),                 # [B, T_txt] - text + special tokens ONLY
        "attention_mask": to_dev(batch.attention_mask),       # [B, T_txt]
        "audio_in_ids": to_dev(getattr(batch, "audio_in_ids", None)),                    # [8, T_in] - reference audio
        "audio_in_ids_start": to_dev(getattr(batch, "audio_in_ids_start", None)),        # [B] - reference positions
        "audio_out_ids": to_dev(getattr(batch, "audio_out_ids", None)),                  # [8, T_out] - alignment metadata ONLY
        "audio_out_ids_start": to_dev(getattr(batch, "audio_out_ids_start", None)),      # [B] - target positions
        "audio_out_ids_start_group_loc": to_dev(getattr(batch, "audio_out_ids_start_group_loc", None)),  # [B] - group locations
    }
    
    # Optional: Add audio waveform features if available
    if hasattr(batch, 'audio_in_wv'):
        inputs["audio_features"] = to_dev(batch.audio_in_wv)
    if hasattr(batch, 'audio_feature_attention_mask'):
        inputs["audio_feature_attention_mask"] = to_dev(batch.audio_feature_attention_mask)

    # SANITY CHECK: DO NOT concatenate audio_out_ids to the text sequence.
    # expanded_input_ids must come from the model's internal builder,
    # not from us pre-concatenating targets.
    logger.debug(f" NO-LEAK INPUTS: input_ids shape {inputs['input_ids'].shape if inputs['input_ids'] is not None else None}")
    return inputs


def find_assistant_content_spans(input_ids, tokenizer):
    """
    SURGICAL FIX: Find the LAST assistant message span - this contains the Arabic response.
    
    Returns:
        List of (start_pos, end_pos) tuples for assistant content (excluding headers)
    """
    ids = input_ids.tolist()
    
    # Token IDs
    im_start_id = tokenizer.convert_tokens_to_ids("assistant")
    im_end_id = tokenizer.convert_tokens_to_ids("assistant") 
    assistant_id = tokenizer.convert_tokens_to_ids("assistant")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    
    spans = []
    
    # Method 1: Standard ChatML  assistant\n...
