#!/usr/bin/env python3
"""
Higgs Audio v2 Training Script - FIXED - Official Architecture + Your Exact ChatML Format
Matches your exact working implementation while using HuggingFace Trainer for stability
"""

import os
import json
import logging
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoConfig,
    TrainingArguments, 
    Trainer,
    WhisperProcessor,
    get_linear_schedule_with_warmup
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    PeftModel
)
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import librosa
import re

# Import Higgs Audio modules
from boson_multimodal.model.higgs_audio import HiggsAudioConfig, HiggsAudioModel
from boson_multimodal.model.higgs_audio.modeling_higgs_audio import HiggsAudioModel, HiggsAudioConfig, HiggsAudioDualFFNDecoderLayer
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator, HiggsAudioBatchInput
from boson_multimodal.audio_processing.higgs_audio_tokenizer import HiggsAudioTokenizer, load_higgs_audio_tokenizer
from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinalDataCollator:
    """
    DEFINITIVE FIX: A simple collator that prepares a clean dictionary for the standard Trainer.
    It fixes the 'labels' issue at the source.
    """
    def __init__(self, **kwargs):
        # Use the official collator internally
        self.base_collator = HiggsAudioSampleCollator(**kwargs)

    def __call__(self, batch: List[ChatMLDatasetSample]):
        # 1. Get the batch object from the official collator
        batch_input = self.base_collator(batch)
        
        # 2. Convert the batch object to a clean dictionary
        model_inputs = {}
        for attr_name in dir(batch_input):
            if not attr_name.startswith('_') and not callable(getattr(batch_input, attr_name)):
                model_inputs[attr_name] = getattr(batch_input, attr_name)

        # 3. Forcefully remove the original 'labels' key if it exists
        if 'labels' in model_inputs:
            model_inputs.pop('labels')

        # 4. CRITICAL FIX: Ensure 'label_ids' is present and there is NO 'labels' key.
        # DO NOT rename 'label_ids' to 'labels'. The model only accepts 'label_ids'.

        return model_inputs


class InferenceStyleDatasetForTrainer(Dataset):
    """
    FIXED: Uses your exact working approach - matches distributed_trainer.py exactly
    Uses the same data loading pattern as your working implementation
    """
    
    def __init__(
        self, 
        data_dir: str,
        tokenizer: AutoTokenizer,
        audio_tokenizer,
        sample_rate: int = 24000,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.sample_rate = sample_rate
        
        # Load samples using your exact approach
        self.samples = self._load_samples()
        
        if not self.samples:
            raise RuntimeError(f"No valid samples found in {data_dir}")
            
        logger.info(f"✅ DATASET: Loaded {len(self.samples)} samples from {data_dir}")
    
    def _load_samples(self) -> List[Dict]:
        """Load samples using your exact approach"""
        samples = []
        json_files = list(self.data_dir.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        samples.extend(data)
                    else:
                        samples.append(data)
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> ChatMLDatasetSample:
        """
        FIXED: Uses your exact working approach from distributed_trainer.py
        Matches the exact function signature and data processing
        """
        sample = self.samples[idx]
        
        try:
            # FIXED: Use exact same approach as your working distributed_trainer.py
            # This matches line 64: input_tokens, label_tokens, audio_contents, speaker_id = prepare_chatml_sample(sample, tokenizer)
            result = prepare_chatml_sample(sample, self.tokenizer)
            
            if result is None or len(result) != 4:
                logger.error(f"Invalid result from prepare_chatml_sample: {result}")
                return self.__getitem__((idx + 1) % len(self))
            
            input_tokens, label_tokens, audio_contents, speaker_id = result
            
            # Process audio exactly like your working distributed_trainer.py
            audio_ids_list = []
            audio_waveforms_list = []
            
            for audio_content in audio_contents:
                if audio_content and hasattr(audio_content, 'audio_url'):
                    audio_path = audio_content.audio_url
                    if audio_path and os.path.exists(audio_path):
                        try:
                            # Tokenize audio (returns tensor on device of tokenizer)
                            audio_codes = self.audio_tokenizer.encode(audio_path)
                            # Ensure CPU and proper dtype
                            audio_codes = audio_codes.detach().cpu().long()
                            
                            # Load waveform (always on CPU)
                            waveform, sr = librosa.load(audio_path, sr=24000, mono=True)
                            waveform = torch.tensor(waveform, dtype=torch.float32)  # CPU by default
                            
                            audio_ids_list.append(audio_codes)
                            audio_waveforms_list.append(waveform)
                            
                        except Exception as e:
                            logger.warning(f"Failed to process audio {audio_path}: {e}")
            
            # Create concatenations (ensure all CPU)
            if audio_ids_list:
                # Use .detach().clone() to avoid tensor creation warning
                audio_ids_concat = torch.cat([codes.detach().clone() for codes in audio_ids_list], dim=1)
                # FIXED: Use exact working approach for proper cumulative indexing
                audio_ids_start = torch.tensor([0] + [codes.shape[1] for codes in audio_ids_list[:-1]], dtype=torch.long).cumsum(dim=0)
                
                audio_waveforms_concat = torch.cat(audio_waveforms_list, dim=0) if audio_waveforms_list else torch.zeros(1)
                # FIXED: Use exact working approach for proper cumulative indexing
                audio_waveforms_start = torch.tensor([0] + [wv.shape[0] for wv in audio_waveforms_list[:-1]], dtype=torch.long).cumsum(dim=0)
            else:
                # Dummy values (matches your approach)
                audio_ids_concat = torch.zeros((8, 10), dtype=torch.long)  # 8 codebooks, 10 dummy tokens
                audio_ids_start = torch.tensor([0, 10], dtype=torch.long)
                
                audio_waveforms_concat = torch.zeros(24000)  # 1 second of silence
                audio_waveforms_start = torch.tensor([0], dtype=torch.long)
            
            # Create dataset sample (exact same structure as your working code)
            dataset_sample = ChatMLDatasetSample(
                input_ids=torch.tensor(input_tokens, dtype=torch.long),
                label_ids=torch.tensor(label_tokens, dtype=torch.long),
                audio_ids_concat=audio_ids_concat,
                audio_ids_start=audio_ids_start,
                audio_label_ids_concat=None,  # FIXED: Correct field name (was label_audio_ids)
                audio_waveforms_concat=audio_waveforms_concat,
                audio_waveforms_start=audio_waveforms_start,
                audio_sample_rate=torch.tensor([24000], dtype=torch.float32),
                audio_speaker_indices=torch.tensor([0], dtype=torch.long),
            )
            
            return dataset_sample
            
        except Exception as e:
            logger.error(f"Error processing sample at index {idx}: {e}", exc_info=True)
            # Return next sample to avoid training interruption
            return self.__getitem__((idx + 1) % len(self))


def setup_lora_config(model: nn.Module, lora_config: Dict) -> nn.Module:
    """
    Setup LoRA configuration - official approach extended with audio attention
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Build target modules - start with official basic modules
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    # Add audio attention modules if they exist (your advancement)
    if hasattr(model.config, 'use_audio_out_self_attention') and model.config.use_audio_out_self_attention:
        audio_attn_modules = [
            "audio_attn.q_proj", "audio_attn.k_proj", 
            "audio_attn.v_proj", "audio_attn.o_proj"
        ]
        target_modules.extend(audio_attn_modules)
        logger.info(f"🎯 LORA TARGETING: Including audio attention modules: {audio_attn_modules}")
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_config.get("rank", 16),
        lora_alpha=lora_config.get("alpha", 32),
        lora_dropout=lora_config.get("dropout", 0.1),
        target_modules=target_modules,
        auto_mapping=True
    )
    
    model = model.to(device)
    
    # CRITICAL FIX: Apply LoRA to the underlying model, not the wrapper
    # This avoids PEFT's automatic label handling
    model = get_peft_model(model, peft_config)
    logger.info("🔧 APPLIED PEFT TO: model (direct)")
    
    model = model.to(device)
    return model


def create_model_and_tokenizer(args):
    """
    Create model and tokenizer - following OFFICIAL WORKING approach from train_v1.py
    """
    # Load TEXT tokenizer using AutoTokenizer (CORRECT METHOD)
    logger.info(f"Loading TEXT tokenizer from {args.tokenizer_path or args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or args.model_path,
        trust_remote_code=True
    )
    
    # Load model
    model = HiggsAudioModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    # Enable audio attention for cross-modal conditioning
    if not model.config.use_audio_out_self_attention:
        logger.info("🔧 ENABLING: use_audio_out_self_attention for cross-modal conditioning")
        model.config.use_audio_out_self_attention = True
        
        # Rebuild layers with audio attention
        new_layers = nn.ModuleList()
        for i, layer in enumerate(model.layers):
            new_layer = HiggsAudioDualFFNDecoderLayer(
                model.config,
                layer_idx=i,
                fast_forward=False,
                use_audio_attention=True  # Enable audio attention
            )
            
            # Copy existing weights
            new_layer.load_state_dict(layer.state_dict(), strict=False)
            new_layers.append(new_layer)
        
        model.layers = new_layers
        logger.info(f"✅ CROSS-MODAL CONDITIONING: Rebuilt {len(new_layers)} layers with audio attention")
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Higgs Audio v2 Training - FIXED - Official + Your Exact ChatML")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model")
    parser.add_argument("--tokenizer_path", type=str, help="Path to tokenizer (defaults to model_path)")
    parser.add_argument("--audio_tokenizer_path", type=str, required=True, help="Path to audio tokenizer")
    
    # Data arguments
    parser.add_argument("--train_data_dir", type=str, required=True, help="Training data directory")
    parser.add_argument("--eval_data_dir", type=str, help="Evaluation data directory")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device during training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Batch size per device during evaluation")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Number of warmup steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Run evaluation every X steps")
    parser.add_argument("--fp16", action="store_true", help="Use 16-bit floating point precision")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing")
    
    # LoRA arguments - FIXED: Support both your argument names
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (matches your argument)")
    parser.add_argument("--lora_rank", type=int, help="LoRA rank (alternative)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    
    # Logging
    parser.add_argument("--report_to", type=str, default="none", help="Report to wandb, tensorboard, etc.")
    parser.add_argument("--logging_dir", type=str, help="Logging directory")
    
    args = parser.parse_args()
    
    # FIXED: Handle both lora_r and lora_rank arguments
    if args.lora_rank is None:
        args.lora_rank = args.lora_r
    
    # Setup paths
    if not args.tokenizer_path:
        args.tokenizer_path = args.model_path
    
    # Load model and tokenizer - OFFICIAL APPROACH (no wrapper)
    logger.info(f"Loading model from {args.model_path}")
    model, tokenizer = create_model_and_tokenizer(args)
    
    logger.info(f"Loading audio tokenizer from {args.audio_tokenizer_path}")
    audio_tokenizer = load_higgs_audio_tokenizer(args.audio_tokenizer_path)
    
    # Setup LoRA (official approach extended for audio attention)
    if args.use_lora:
        lora_config = {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
        }
        model = setup_lora_config(model, lora_config)
        logger.info("✅ LORA: Configuration applied with cross-modal targeting")
    
    # FIXED: Load datasets using your exact working approach
    train_dataset = InferenceStyleDatasetForTrainer(
        args.train_data_dir,
        tokenizer,
        audio_tokenizer,
    )
    
    eval_dataset = None
    if args.eval_data_dir:
        eval_dataset = InferenceStyleDatasetForTrainer(
            args.eval_data_dir,
            tokenizer,
            audio_tokenizer,
        )
    
    # Setup training arguments (official approach)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps" if eval_dataset else "no",
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=args.report_to if args.report_to != "none" else None,
        logging_dir=args.logging_dir,
        gradient_accumulation_steps=2,  # Conservative for stability
        max_grad_norm=0.1,  # Gradient clipping for stability
    )
    
    # Setup data collator (official approach adapted)
    try:
        whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        
        data_collator = FinalDataCollator(
            whisper_processor=whisper_processor,
            audio_in_token_id=model.config.audio_in_token_idx,
            audio_out_token_id=model.config.audio_out_token_idx,
            audio_stream_bos_id=model.config.audio_stream_bos_id,
            audio_stream_eos_id=model.config.audio_stream_eos_id,
            encode_whisper_embed=True,
            pad_token_id=tokenizer.pad_token_id,
            return_audio_in_tokens=True,
            use_delay_pattern=False,
            round_to=8,
            audio_num_codebooks=8,
        )
        logger.info("✅ COLLATOR: Official collator setup successful")
    except Exception as e:
        logger.error(f"❌ COLLATOR: Failed to setup official collator: {e}")
        raise
    
    # Initialize trainer (official approach)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    logger.info("🚀 TRAINING: Starting with official architecture + your exact ChatML format + cross-modal conditioning")
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    logger.info(f"✅ MODEL SAVED: Final model saved to {args.output_dir}")
    
    # Save LoRA adapters separately
    if args.use_lora:
        lora_output_dir = os.path.join(args.output_dir, "lora_adapters")
        model.save_pretrained(lora_output_dir)
        logger.info(f"✅ LORA SAVED: LoRA adapters saved to {lora_output_dir}")


if __name__ == "__main__":
    main()
