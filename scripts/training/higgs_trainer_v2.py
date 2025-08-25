#!/usr/bin/env python3
"""
Higgs Audio v2 Training Script - Official Architecture + ChatML + Cross-Modal Conditioning
Combines the stability of official trainer with advanced zero-shot voice cloning capabilities
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
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator, HiggsAudioBatchInput
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExtendedHiggsAudioBatchInput:
    """Extended HiggsAudioBatchInput with __len__ method for Trainer compatibility"""
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __len__(self):
        """Return the batch size based on input_ids"""
        if hasattr(self, 'input_ids') and self.input_ids is not None:
            return self.input_ids.shape[0]
        else:
            return 0
    
    def __getitem__(self, key):
        """Allow dictionary-style access for compatibility"""
        return getattr(self, key)
    
    def __contains__(self, key):
        """Check if attribute exists"""
        return hasattr(self, key)
    
    def keys(self):
        """Return all attribute names for compatibility"""
        return [attr for attr in dir(self) if not attr.startswith('_') and not callable(getattr(self, attr))]


class ExtendedHiggsAudioSampleCollator:
    """Extended collator that returns our custom batch input class"""
    
    def __init__(self, **kwargs):
        self.base_collator = HiggsAudioSampleCollator(**kwargs)
    
    def __call__(self, batch: List[ChatMLDatasetSample]):
        # Use official collator
        batch_input = self.base_collator(batch)
        
        # Convert to extended batch input for Trainer compatibility
        extended_batch = ExtendedHiggsAudioBatchInput()
        
        # Copy all attributes from base batch
        for attr_name in dir(batch_input):
            if not attr_name.startswith('_') and not callable(getattr(batch_input, attr_name)):
                setattr(extended_batch, attr_name, getattr(batch_input, attr_name))
        
        return extended_batch


class ChatMLDatasetForTrainer(Dataset):
    """
    ChatML Dataset adapted for HuggingFace Trainer with zero-shot voice cloning
    Preserves your ChatML format and cross-modal conditioning
    """
    
    def __init__(
        self, 
        data_dir: str,
        tokenizer: AutoTokenizer,
        audio_tokenizer,
        sample_rate: int = 24000,
        device: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.sample_rate = sample_rate
        self.device = device
        
        # Load ChatML samples
        self.samples = self._load_chatml_samples()
        
        if not self.samples:
            raise RuntimeError(f"No valid ChatML samples found in {data_dir}")
            
        logger.info(f"Loaded {len(self.samples)} ChatML samples from {data_dir}")
    
    def _load_chatml_samples(self) -> List[Dict]:
        """Load ChatML samples from directory"""
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
    
    def _load_audio_waveform(self, audio_path: str):
        """Load audio waveform"""
        try:
            if not os.path.exists(audio_path):
                # Try relative to data_dir
                audio_path = str(self.data_dir / audio_path)
            
            waveform, sr = torchaudio.load(audio_path)
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            return waveform.squeeze(0), self.sample_rate
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            # Return silent audio as fallback
            return torch.zeros(self.sample_rate), self.sample_rate
    
    def _encode_audio_tokens(self, audio_path: str):
        """Encode audio to tokens"""
        try:
            waveform, _ = self._load_audio_waveform(audio_path)
            tokens = self.audio_tokenizer.encode(waveform.unsqueeze(0).numpy(), sample_rate=self.sample_rate)
            return torch.tensor(tokens, dtype=torch.long)
        except Exception as e:
            logger.error(f"Error encoding audio {audio_path}: {e}")
            # Return dummy tokens
            return torch.zeros((8, 100), dtype=torch.long)  # 8 codebooks, 100 tokens
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> ChatMLDatasetSample:
        sample = self.samples[idx]
        
        try:
            # Parse ChatML structure
            messages = []
            for msg in sample.get("messages", []):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if isinstance(content, str):
                    messages.append(Message(role=role, content=content))
                elif isinstance(content, dict) and content.get("type") == "audio":
                    audio_url = content.get("audio_url", "")
                    messages.append(Message(role=role, content=AudioContent(audio_url=audio_url)))
            
            chatml_sample = ChatMLSample(messages=messages)
            
            # Process ChatML sample
            result = prepare_chatml_sample(chatml_sample, self.tokenizer)
            if result is None or len(result) != 4:
                logger.error(f"Invalid result from prepare_chatml_sample: {result}")
                return self.__getitem__((idx + 1) % len(self))
            
            input_tokens, label_tokens, audio_contents, speaker_id = result
            
            # Process context audio (reference audio for voice conditioning)
            context_audio_tokens = []
            for audio_content in (audio_contents or []):
                if audio_content.audio_url:
                    tokens = self._encode_audio_tokens(audio_content.audio_url)
                    if tokens is not None: 
                        context_audio_tokens.append(tokens)
            
            # Handle your ChatML format - extract target audio from assistant messages
            label_audio_tokens = []
            target_audio_path = None
            
            # Look for target audio in assistant messages (your zero-shot format)
            for msg in sample.get("messages", []):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, dict) and content.get("type") == "audio":
                        target_audio_path = content.get("audio_url", "")
                        if target_audio_path:
                            tokens = self._encode_audio_tokens(target_audio_path)
                            if tokens is not None:
                                label_audio_tokens.append(tokens)
            
            # Concatenate audio tokens
            if context_audio_tokens:
                audio_ids_concat = torch.cat(context_audio_tokens, dim=1)
                audio_ids_start = torch.tensor([0] + [t.shape[1] for t in context_audio_tokens[:-1]], dtype=torch.long).cumsum(0)
            else:
                audio_ids_concat = torch.zeros((8, 0), dtype=torch.long)  # 8 codebooks
                audio_ids_start = torch.tensor([0], dtype=torch.long)
            
            label_audio_ids = torch.cat(label_audio_tokens, dim=1) if label_audio_tokens else None
            
            # Load reference audio waveform for Whisper (first audio in context)
            ref_audio_path = None
            if audio_contents and len(audio_contents) > 0 and audio_contents[0].audio_url:
                ref_audio_path = audio_contents[0].audio_url
            
            if ref_audio_path:
                waveform, sr = self._load_audio_waveform(ref_audio_path)
            else:
                waveform = torch.zeros(self.sample_rate)
                sr = self.sample_rate
            
            # Create dataset sample
            dataset_sample = ChatMLDatasetSample(
                input_ids=torch.tensor(input_tokens, dtype=torch.long),
                label_ids=torch.tensor(label_tokens, dtype=torch.long),
                audio_ids_concat=audio_ids_concat,
                audio_ids_start=audio_ids_start,
                label_audio_ids=label_audio_ids,
                audio_waveforms_concat=waveform,
                audio_waveforms_start=torch.tensor([0], dtype=torch.long),
                audio_sample_rate=torch.tensor([sr], dtype=torch.float32),
                audio_speaker_indices=torch.tensor([0], dtype=torch.long),
            )
            
            # Move to device if specified
            return dataset_sample.to(self.device) if self.device else dataset_sample
            
        except Exception as e:
            logger.error(f"Error processing sample at index {idx}: {e}", exc_info=True)
            # Return next sample to avoid training interruption
            return self.__getitem__((idx + 1) % len(self))


class HiggsAudioModelWrapper(nn.Module):
    """
    Model wrapper that handles cross-modal conditioning and audio attention
    Based on official pattern but with your advanced capabilities
    """
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        super().__init__()
        
        # Load model and config
        self.config = HiggsAudioConfig.from_pretrained(model_path)
        self.model = HiggsAudioModel.from_pretrained(model_path, config=self.config)
        
        # CRITICAL: Enable cross-modal conditioning (your advancement)
        if not self.config.use_audio_out_self_attention:
            logger.info("🔧 ENABLING CROSS-MODAL CONDITIONING: Setting use_audio_out_self_attention=True")
            self.config.use_audio_out_self_attention = True
            self.model.use_audio_out_self_attention = True
            
            # Rebuild layers with audio attention modules
            self._rebuild_layers_with_audio_attention()
        
        self.device = device
        self.to(device)
    
    def _rebuild_layers_with_audio_attention(self):
        """Rebuild decoder layers with audio attention modules enabled"""
        logger.info("🏗️  REBUILDING LAYERS: Adding audio attention modules for cross-modal conditioning")
        
        from boson_multimodal.model.higgs_audio.modeling_higgs_audio import HiggsAudioDualFFNDecoderLayer
        
        # Rebuild layers with audio attention
        new_layers = nn.ModuleList()
        for i, layer in enumerate(self.model.layers):
            new_layer = HiggsAudioDualFFNDecoderLayer(
                self.config,
                layer_idx=i,
                fast_forward=False,
                use_audio_attention=True  # Enable audio attention
            )
            
            # Copy existing weights
            new_layer.load_state_dict(layer.state_dict(), strict=False)
            new_layers.append(new_layer)
        
        self.model.layers = new_layers
        logger.info(f"✅ CROSS-MODAL CONDITIONING: Rebuilt {len(new_layers)} layers with audio attention")
    
    def forward(self, **kwargs):
        return self.model(**kwargs)


class HiggsAudioTrainer(Trainer):
    """
    Custom trainer combining official stability with your cross-modal conditioning
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = self.model.config
        
        # Initialize newly created audio attention modules (your fix)
        self._initialize_audio_attention_modules()
    
    def _initialize_audio_attention_modules(self):
        """Initialize newly created audio attention modules for stability"""
        logger.info("🎯 WEIGHT INITIALIZATION: Initializing audio attention modules")
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'audio_attn' in name and 'weight' in name:
                    if 'lora_A' in name:
                        # LoRA A matrices should be near zero
                        torch.nn.init.normal_(param, mean=0.0, std=0.001)
                    elif 'lora_B' in name:
                        # LoRA B matrices should be zero initially
                        torch.nn.init.zeros_(param)
                    elif any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                        # Attention projections should be small
                        torch.nn.init.xavier_uniform_(param, gain=0.02)
                elif 'audio_post_audio_attn_layer_norm' in name and 'weight' in name:
                    torch.nn.init.ones_(param)
        
        logger.info("✅ WEIGHT INITIALIZATION: Audio attention modules initialized")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss computation using model's built-in loss (official approach)
        But with audio token masking (your improvement)
        """
        # Convert ExtendedHiggsAudioBatchInput to model inputs
        if isinstance(inputs, ExtendedHiggsAudioBatchInput):
            model_inputs = {}
            for attr_name in ['input_ids', 'attention_mask', 'label_ids', 
                            'audio_features', 'audio_feature_attention_mask',
                            'audio_out_ids', 'audio_out_ids_start', 
                            'audio_in_ids', 'audio_in_ids_start',
                            'label_audio_ids']:
                attr_value = getattr(inputs, attr_name, None)
                if attr_value is not None:
                    model_inputs[attr_name] = attr_value
        else:
            model_inputs = dict(inputs)
            if 'labels' in model_inputs:
                model_inputs['label_ids'] = model_inputs.pop('labels')
        
        # Apply audio token masking in text labels (your improvement)
        if 'label_ids' in model_inputs and model_inputs['label_ids'] is not None:
            model_inputs['label_ids'] = self._mask_audio_tokens_in_text_labels(model_inputs['label_ids'])
        
        # Ensure all inputs are on the same device
        for key, value in model_inputs.items():
            if isinstance(value, torch.Tensor):
                model_inputs[key] = value.to(model.device)
        
        # Forward pass - use model's built-in loss (official approach)
        outputs = model(**model_inputs)
        
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        
        # Add NaN detection (your safety improvement)
        if torch.isnan(loss):
            logger.warning("⚠️  NaN loss detected! Skipping this batch.")
            loss = torch.tensor(0.0, requires_grad=True, device=loss.device)
        
        return (loss, outputs) if return_outputs else loss
    
    def _mask_audio_tokens_in_text_labels(self, label_ids):
        """Mask audio tokens in text labels to prevent contamination"""
        # Audio token IDs that should be masked in text labels
        audio_token_ids = [128000, 128001, 128002, 128003, 128004, 128005, 128006, 128007, 
                          128008, 128009, 128010, 128011, 128012, 128013, 128014, 128015,
                          128016, 128017, 128018, 128019, 128020, 128021, 128022, 128023]
        
        masked_labels = label_ids.clone()
        for token_id in audio_token_ids:
            masked_labels[masked_labels == token_id] = -100
        
        return masked_labels


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
    
    # Apply LoRA - use official pattern
    if hasattr(model, 'model') and hasattr(model.model, 'text_model'):
        model.model.text_model = get_peft_model(model.model.text_model, peft_config)
    elif hasattr(model, 'model'):
        model.model = get_peft_model(model.model, peft_config)
    else:
        model = get_peft_model(model, peft_config)
    
    model = model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser(description="Higgs Audio v2 Training - Official + ChatML + Cross-Modal")
    
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
    
    # LoRA arguments - FIXED: Match user's argument names
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (matches user's argument)")
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
    
    # Load tokenizer and audio tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    logger.info(f"Loading audio tokenizer from {args.audio_tokenizer_path}")
    audio_tokenizer = load_higgs_audio_tokenizer(args.audio_tokenizer_path)
    
    # Load model with wrapper (official pattern + your improvements)
    logger.info(f"Loading model from {args.model_path}")
    model = HiggsAudioModelWrapper(args.model_path)
    
    # Setup LoRA (official approach extended for audio attention)
    if args.use_lora:
        lora_config = {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
        }
        model = setup_lora_config(model, lora_config)
        logger.info("✅ LORA: Configuration applied with cross-modal targeting")
    
    # Load datasets (your ChatML format preserved)
    train_dataset = ChatMLDatasetForTrainer(
        args.train_data_dir,
        tokenizer,
        audio_tokenizer,
    )
    
    eval_dataset = None
    if args.eval_data_dir:
        eval_dataset = ChatMLDatasetForTrainer(
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
        
        data_collator = ExtendedHiggsAudioSampleCollator(
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
    
    # Initialize trainer (official approach + your improvements)
    trainer = HiggsAudioTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    logger.info("🚀 TRAINING: Starting with official architecture + ChatML + cross-modal conditioning")
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    logger.info(f"✅ MODEL SAVED: Final model saved to {args.output_dir}")
    
    # Save LoRA adapters separately
    if args.use_lora:
        lora_output_dir = os.path.join(args.output_dir, "lora_adapters")
        if hasattr(model, 'model') and hasattr(model.model, 'text_model'):
            model.model.text_model.save_pretrained(lora_output_dir)
        elif hasattr(model, 'model'):
            model.model.save_pretrained(lora_output_dir)
        else:
            model.save_pretrained(lora_output_dir)
        logger.info(f"✅ LORA SAVED: LoRA adapters saved to {lora_output_dir}")


if __name__ == "__main__":
    main()
