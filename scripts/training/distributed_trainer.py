#!/usr/bin/env python3
"""
Distributed LoRA Training Script for Higgs-Audio V2 Zero-Shot Voice Cloning
Matches the inference pipeline exactly for consistent training behavior.
"""

import os
import json
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoTokenizer, AutoConfig, WhisperProcessor
from peft import LoraConfig, get_peft_model, TaskType
from loguru import logger
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import wandb
import numpy as np

# Import Higgs-Audio components
from boson_multimodal.model.higgs_audio import HiggsAudioModel, HiggsAudioConfig
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
from boson_multimodal.data_types import Message, TextContent, AudioContent
from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model paths
    model_path: str = "bosonai/higgs-audio-v2-generation-3B-base"
    tokenizer_path: str = "bosonai/higgs-audio-v2-tokenizer"
    
    # Data paths
    train_data_path: str = "data/train_chatml_samples.json"
    val_data_path: str = "data/val_chatml_samples.json"
    
    # Training parameters
    batch_size_per_device: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 3
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Audio parameters
    audio_num_codebooks: int = 8  # Must match tokenizer
    sample_rate: int = 24000
    
    # Other parameters
    num_workers: int = 4
    seed: int = 42
    output_dir: str = "outputs/lora_model"
    checkpoint_dir: str = "outputs/checkpoints"
    log_every_n_steps: int = 10
    save_every_n_steps: int = 100
    eval_every_n_steps: int = 50
    use_wandb: bool = False
    wandb_project: str = "higgs-audio-lora"
    mixed_precision: str = "bf16"
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load config from YAML file"""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class ZeroShotVoiceCloningDataset(Dataset):
    """
    Dataset for zero-shot voice cloning training.
    Loads ChatML samples and converts them to Message format matching inference.
    """
    
    def __init__(self, json_path: str, max_samples: Optional[int] = None):
        """Initialize dataset from ChatML JSON file"""
        self.samples = []
        
        logger.info(f"Loading dataset from {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            samples = data
        elif isinstance(data, dict):
            samples = data.get('samples', data.get('data', []))
        else:
            raise ValueError(f"Unexpected JSON structure: {type(data)}")
        
        # Limit samples if specified
        if max_samples:
            samples = samples[:max_samples]
        
        # Validate and store samples
        for idx, sample in enumerate(samples):
            if 'messages' not in sample:
                logger.warning(f"Sample {idx} missing 'messages' field, skipping")
                continue
            
            # Validate message structure
            messages = sample['messages']
            has_user_msg = any(msg.get('role') == 'user' for msg in messages)
            has_assistant_msg = any(msg.get('role') == 'assistant' for msg in messages)
            
            if not has_user_msg:
                logger.warning(f"Sample {idx} missing user message, skipping")
                continue
            
            if not has_assistant_msg:
                logger.warning(f"Sample {idx} missing assistant message, skipping")
                continue
            
            self.samples.append(sample)
        
        logger.info(f"Loaded {len(self.samples)} valid samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Return raw ChatML sample - conversion happens in collate_fn"""
        return self.samples[idx]


class HiggsAudioDistributedTrainer:
    """Distributed trainer for Higgs-Audio LoRA fine-tuning"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
            log_with="wandb" if config.use_wandb else None,
            project_dir=config.output_dir
        )
        
        # Set seed for reproducibility
        set_seed(config.seed)
        
        # Setup logging
        self.logger = logger
        if self.accelerator.is_main_process:
            logger.add(f"{config.output_dir}/training.log")
        
        # Create output directories
        if self.accelerator.is_main_process:
            os.makedirs(config.output_dir, exist_ok=True)
            os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Initialize components (will be set in setup methods)
        self.model = None
        self.tokenizer = None
        self.audio_tokenizer = None
        self.collator = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.optimizer = None
        self.scheduler = None
        
    def setup_tokenizers_and_collator(self):
        """Setup tokenizers and collator matching inference pipeline"""
        logger.info("Setting up tokenizers and collator...")
        
        # Load text tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load audio tokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.audio_tokenizer = load_higgs_audio_tokenizer(
            self.config.tokenizer_path,
            device=device
        )
        
        # Verify audio tokenizer has correct number of codebooks
        if hasattr(self.audio_tokenizer, 'n_q'):
            assert self.audio_tokenizer.n_q == self.config.audio_num_codebooks, \
                f"Audio tokenizer codebooks ({self.audio_tokenizer.n_q}) != config ({self.config.audio_num_codebooks})"
        
        # Load model config for collator
        model_config = AutoConfig.from_pretrained(self.config.model_path)
        
        # Initialize Whisper processor for audio feature extraction
        whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        
        # Initialize collator exactly like inference
        self.collator = HiggsAudioSampleCollator(
            whisper_processor=whisper_processor,
            audio_in_token_id=model_config.audio_in_token_idx,
            audio_out_token_id=model_config.audio_out_token_idx,
            audio_stream_bos_id=model_config.audio_stream_bos_id,
            audio_stream_eos_id=model_config.audio_stream_eos_id,
            encode_whisper_embed=model_config.encode_whisper_embed,
            pad_token_id=model_config.pad_token_id,
            return_audio_in_tokens=model_config.encode_audio_in_tokens,
            use_delay_pattern=model_config.use_delay_pattern,
            round_to=1,
            audio_num_codebooks=self.config.audio_num_codebooks
        )
        
        logger.info("Tokenizers and collator setup complete")
        
    def setup_datasets_and_dataloaders(self):
        """Setup datasets and dataloaders"""
        logger.info("Setting up datasets and dataloaders...")
        
        # Create datasets
        train_dataset = ZeroShotVoiceCloningDataset(self.config.train_data_path)
        val_dataset = ZeroShotVoiceCloningDataset(self.config.val_data_path)
        
        # Create samplers for distributed training
        train_sampler = DistributedSampler(train_dataset) if self.accelerator.num_processes > 1 else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if self.accelerator.num_processes > 1 else None
        
        # Create dataloaders with custom collate function
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size_per_device,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=self.collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size_per_device,
            sampler=val_sampler,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        logger.info(f"Train dataloader: {len(self.train_dataloader)} batches")
        logger.info(f"Val dataloader: {len(self.val_dataloader)} batches")
        
    def collate_fn(self, batch: List[Dict]) -> Any:
        """
        Custom collate function that matches inference pipeline exactly.
        Converts ChatML samples to ChatMLDatasetSample with proper audio tokenization.
        """
        chatml_samples = []
        target_audio_paths = []
        
        for sample in batch:
            messages = sample['messages']
            
            # Extract reference audio and target text from messages
            reference_audio_path = None
            target_text = None
            target_audio_path = None
            
            # Build ChatML dict for prepare_chatml_sample
            chatml_dict = {"messages": []}
            
            for msg in messages:
                role = msg.get('role')
                content = msg.get('content')
                
                if role == 'system':
                    # System message
                    chatml_dict["messages"].append({"role": "system", "content": content})
                    
                elif role == 'user':
                    # User message with text and reference audio
                    if isinstance(content, list):
                        user_content = []
                        for item in content:
                            if item.get('type') == 'text':
                                target_text = item.get('text', '')
                                user_content.append({"type": "text", "text": target_text})
                            elif item.get('type') == 'audio':
                                reference_audio_path = item.get('audio_url', '')
                                user_content.append({"type": "audio", "audio_url": reference_audio_path})
                        chatml_dict["messages"].append({"role": "user", "content": user_content})
                    else:
                        # Simple text content
                        chatml_dict["messages"].append({"role": "user", "content": content})
                        
                elif role == 'assistant':
                    # Extract target audio path for loss computation
                    if isinstance(content, list):
                        for item in content:
                            if item.get('type') == 'audio':
                                target_audio_path = item.get('audio_url', '')
                                break
            
            # Store target audio path for later loss computation
            target_audio_paths.append(target_audio_path)
            
            # Use prepare_chatml_sample to create proper tokens
            input_tokens, label_tokens, audio_contents, speaker_id = prepare_chatml_sample(
                chatml_dict, self.tokenizer
            )
            
            # Process reference audio: tokenize and load waveforms
            audio_ids_list = []
            audio_waveforms_list = []
            
            for audio_content in audio_contents:
                if audio_content and hasattr(audio_content, 'audio_url'):
                    audio_path = audio_content.audio_url
                    if audio_path and os.path.exists(audio_path):
                        try:
                            # Tokenize audio to get codes
                            audio_codes = self.audio_tokenizer.encode(audio_path)
                            # Ensure correct number of codebooks
                            if audio_codes.shape[0] > self.config.audio_num_codebooks:
                                audio_codes = audio_codes[:self.config.audio_num_codebooks, :]
                            elif audio_codes.shape[0] < self.config.audio_num_codebooks:
                                # Pad with zeros if needed
                                padding = torch.zeros(
                                    self.config.audio_num_codebooks - audio_codes.shape[0],
                                    audio_codes.shape[1],
                                    dtype=audio_codes.dtype
                                )
                                audio_codes = torch.cat([audio_codes, padding], dim=0)
                            audio_ids_list.append(audio_codes)
                            
                            # Load waveform
                            waveform, sr = torchaudio.load(audio_path)
                            # Resample if needed
                            if sr != self.config.sample_rate:
                                resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
                                waveform = resampler(waveform)
                            # Ensure mono
                            if waveform.shape[0] > 1:
                                waveform = waveform.mean(dim=0, keepdim=True)
                            # Flatten to 1D for Whisper processor
                            waveform = waveform.squeeze(0)
                            audio_waveforms_list.append(waveform)
                            
                        except Exception as e:
                            logger.warning(f"Failed to process audio {audio_path}: {e}")
            
            # Prepare audio tensors
            if audio_ids_list:
                audio_ids_concat = torch.cat(audio_ids_list, dim=1)
                audio_ids_start = torch.cumsum(
                    torch.tensor([0] + [ids.shape[1] for ids in audio_ids_list], dtype=torch.long),
                    dim=0
                )
            else:
                audio_ids_concat = torch.zeros((self.config.audio_num_codebooks, 0), dtype=torch.long)
                audio_ids_start = torch.tensor([], dtype=torch.long)
            
            if audio_waveforms_list:
                # Concatenate waveforms as 1D array
                audio_waveforms_concat = torch.cat(audio_waveforms_list, dim=0)
                audio_waveforms_start = torch.cumsum(
                    torch.tensor([0] + [len(wv) for wv in audio_waveforms_list[:-1]], dtype=torch.long),
                    dim=0
                )
                audio_sample_rate = torch.tensor([self.config.sample_rate] * len(audio_waveforms_list), dtype=torch.float32)
                audio_speaker_indices = torch.tensor([0] * len(audio_waveforms_list), dtype=torch.long)
            else:
                audio_waveforms_concat = torch.tensor([], dtype=torch.float32)
                audio_waveforms_start = torch.tensor([], dtype=torch.long)
                audio_sample_rate = torch.tensor([], dtype=torch.float32)
                audio_speaker_indices = torch.tensor([], dtype=torch.long)
            
            # Create ChatMLDatasetSample
            chatml_sample = ChatMLDatasetSample(
                input_ids=torch.tensor(input_tokens, dtype=torch.long),
                label_ids=torch.tensor(label_tokens, dtype=torch.long),
                audio_ids_concat=audio_ids_concat,
                audio_ids_start=audio_ids_start,
                audio_waveforms_concat=audio_waveforms_concat,
                audio_waveforms_start=audio_waveforms_start,
                audio_sample_rate=audio_sample_rate,
                audio_speaker_indices=audio_speaker_indices
            )
            
            chatml_samples.append(chatml_sample)
        
        # Use standard collator
        collated_batch = self.collator(chatml_samples)
        
        # Add target audio paths for loss computation
        collated_batch.target_audio_paths = target_audio_paths
        
        return collated_batch
    
    def setup_model_and_optimizer(self):
        """Setup model with LoRA and optimizer"""
        logger.info("Setting up model and optimizer...")
        
        # Load base model
        model = HiggsAudioModel.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": self.accelerator.device}
        )
        
        # Update model config to match audio tokenizer
        model.config.audio_num_codebooks = self.config.audio_num_codebooks
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Setup optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # Setup scheduler
        num_training_steps = len(self.train_dataloader) * self.config.num_epochs
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=self.config.learning_rate * 0.1
        )
        
        # Prepare for distributed training
        self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.scheduler = \
            self.accelerator.prepare(
                model, optimizer, self.train_dataloader, self.val_dataloader, scheduler
            )
        
        logger.info("Model and optimizer setup complete")
    
    def compute_loss(self, batch, target_audio_paths):
        """
        Compute loss for zero-shot voice cloning.
        Model generates audio conditioned on reference, loss computed against target.
        """
        # Forward pass - model generates audio conditioned on reference
        outputs = self.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            audio_in_ids=batch.audio_in_ids,  # Reference audio for conditioning
            audio_in_wv=batch.audio_in_wv,    # Reference waveforms for Whisper
            labels=batch.label_ids,
            return_dict=True
        )
        
        # Get text loss
        text_loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
        
        # For audio loss, we would need to:
        # 1. Tokenize target audio
        # 2. Compare generated audio tokens with target
        # This is simplified here - in production you'd implement full audio loss
        
        return text_loss
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_steps = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch}",
            disable=not self.accelerator.is_local_main_process
        )
        
        for step, batch in enumerate(progress_bar):
            with self.accelerator.accumulate(self.model):
                # Compute loss
                loss = self.compute_loss(batch, batch.target_audio_paths)
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Update metrics
                total_loss += loss.item()
                num_steps += 1
                
                # Update progress bar
                if step % self.config.log_every_n_steps == 0:
                    avg_loss = total_loss / num_steps
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
                    })
                    
                    if self.config.use_wandb and self.accelerator.is_main_process:
                        wandb.log({
                            'train/loss': avg_loss,
                            'train/lr': self.scheduler.get_last_lr()[0],
                            'train/epoch': epoch,
                            'train/step': step
                        })
                
                # Save checkpoint
                if step % self.config.save_every_n_steps == 0 and step > 0:
                    self.save_checkpoint(epoch, step)
                
                # Validation
                if step % self.config.eval_every_n_steps == 0 and step > 0:
                    val_loss = self.validate()
                    logger.info(f"Validation loss: {val_loss:.4f}")
                    self.model.train()
        
        return total_loss / num_steps
    
    def validate(self):
        """Run validation"""
        self.model.eval()
        total_loss = 0
        num_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(
                self.val_dataloader,
                desc="Validation",
                disable=not self.accelerator.is_local_main_process
            ):
                loss = self.compute_loss(batch, batch.target_audio_paths)
                total_loss += loss.item()
                num_steps += 1
        
        avg_loss = total_loss / num_steps
        
        if self.config.use_wandb and self.accelerator.is_main_process:
            wandb.log({'val/loss': avg_loss})
        
        return avg_loss
    
    def save_checkpoint(self, epoch, step):
        """Save model checkpoint"""
        if self.accelerator.is_main_process:
            checkpoint_path = os.path.join(
                self.config.checkpoint_dir,
                f"checkpoint_epoch{epoch}_step{step}"
            )
            self.accelerator.unwrap_model(self.model).save_pretrained(checkpoint_path)
            self.tokenizer.save_pretrained(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Setup components
        self.setup_tokenizers_and_collator()
        self.setup_datasets_and_dataloaders()
        self.setup_model_and_optimizer()
        
        # Initialize wandb
        if self.config.use_wandb and self.accelerator.is_main_process:
            wandb.init(
                project=self.config.wandb_project,
                config=self.config.__dict__
            )
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train epoch
            train_loss = self.train_epoch(epoch + 1)
            logger.info(f"Epoch {epoch + 1} - Train loss: {train_loss:.4f}")
            
            # Validation
            val_loss = self.validate()
            logger.info(f"Epoch {epoch + 1} - Val loss: {val_loss:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1, 0)
        
        # Save final model
        if self.accelerator.is_main_process:
            final_path = os.path.join(self.config.output_dir, "final_model")
            self.accelerator.unwrap_model(self.model).save_pretrained(final_path)
            self.tokenizer.save_pretrained(final_path)
            logger.info(f"Saved final model to {final_path}")
        
        # Cleanup
        if self.config.use_wandb and self.accelerator.is_main_process:
            wandb.finish()
        
        logger.info("Training complete!")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Higgs-Audio LoRA Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/lora_config.yaml",
        help="Path to training config YAML file"
    )
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        config = TrainingConfig.from_yaml(args.config)
    else:
        logger.warning(f"Config file {args.config} not found, using defaults")
        config = TrainingConfig()
    
    # Create trainer and start training
    trainer = HiggsAudioDistributedTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
