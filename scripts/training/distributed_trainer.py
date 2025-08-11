#!/usr/bin/env python3
"""
Distributed Training Pipeline for Higgs-Audio V2 LoRA Fine-tuning
Optimized for 8x H200 GPUs with DeepSpeed and Accelerate integration.
"""

import os
import sys
import json
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import deepspeed
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from transformers import get_linear_schedule_with_warmup
import wandb
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from dataclasses import dataclass, field
import yaml

# Robust import handling for both CLI and module usage
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
    from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
    from boson_multimodal.data_types import Message, TextContent, AudioContent
    from scripts.training.lora_integration import HiggsAudioLoRAConfig, create_lora_model
except ImportError:
    # Fallback for different project structures
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    sys.path.insert(0, project_root)
    from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
    from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
    from boson_multimodal.data_types import Message, TextContent, AudioContent
    from scripts.training.lora_integration import HiggsAudioLoRAConfig, create_lora_model


@dataclass
class TrainingConfig:
    """Training configuration for distributed LoRA fine-tuning"""
    
    # Model and data paths
    model_path: str = "bosonai/higgs-audio-v2-generation-3B-base"
    audio_tokenizer_path: str = "bosonai/higgs-audio-v2-tokenizer"
    dataset_path: str = "/workspace/data/processed_chatml"
    output_dir: str = "/workspace/outputs/higgs-lora-arabic-english"
    
    # Training hyperparameters
    num_epochs: int = 3
    batch_size_per_device: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5  # CRITICAL FIX: Reduced from 2e-4 to prevent gradient explosion
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Audio-specific settings
    max_audio_length: int = 1500  # Max audio tokens
    max_text_length: int = 512    # Max text tokens
    
    # Distributed training
    deepspeed_config: Optional[str] = None
    use_deepspeed: bool = True
    use_accelerate: bool = True
    
    # Logging and checkpointing
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # Wandb logging
    use_wandb: bool = True
    wandb_project: str = "higgs-audio-lora-arabic-english"
    wandb_run_name: Optional[str] = None
    
    # Data processing
    train_split_ratio: float = 0.9
    val_split_ratio: float = 0.1
    num_workers: int = 8
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = True
    
    # Optimization
    optimizer_type: str = "adamw"
    scheduler_type: str = "linear"
    
    # Language-specific settings
    arabic_weight: float = 1.0
    english_weight: float = 1.0
    mixed_weight: float = 1.5  # Higher weight for code-switching samples


def _resolve_path_multi(raw_path: str, dataset_dir: str, audio_base_dir: str) -> str | None:
    """Resolve an audio path by trying multiple bases in order:
    1) If raw_path is absolute and exists -> use it
    2) CWD + raw_path
    3) dataset_dir + raw_path
    4) audio_base_dir + raw_path
    Returns the first existing resolved path, else None.
    """
    if not raw_path:
        return None
    try_order = []
    # 1) Absolute
    if os.path.isabs(raw_path):
        try_order.append(raw_path)
    # 2) CWD
    try_order.append(os.path.normpath(os.path.join(os.getcwd(), raw_path)))
    # 3) Dataset dir
    try_order.append(os.path.normpath(os.path.join(dataset_dir, raw_path)))
    # 4) Audio base dir
    try_order.append(os.path.normpath(os.path.join(audio_base_dir, raw_path)))

    for p in try_order:
        if p and os.path.exists(p):
            return p
    return None


class InferenceStyleDataset:
    """Dataset that processes data exactly like inference pipeline - using Message objects"""
    
    def __init__(self, data_file, audio_tokenizer, text_tokenizer, max_length=2048):
        self.data_file = data_file
        self.audio_tokenizer = audio_tokenizer
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length
        
        # Load ChatML samples
        with open(data_file, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
        
        print(f"Loaded {len(self.samples)} samples from {data_file}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert ChatML to Message objects exactly like inference
        messages = []
        reference_waveforms = []
        target_text = ""
        
        for msg in sample['messages']:
            role = msg.get('role')
            content = msg.get('content')
            
            if role == 'system':
                # System message
                messages.append(Message(role="system", content=content))
                
            elif role == 'user':
                # User message with text and reference audio
                if isinstance(content, list):
                    message_content = []
                    
                    for item in content:
                        if item.get('type') == 'text':
                            # Extract target text
                            text = item.get('text', '')
                            target_text = text
                            message_content.append(TextContent(text=text))
                            
                        elif item.get('type') == 'audio':
                            # Reference audio - load on-demand like inference
                            audio_url = item.get('audio_url', '')
                            if audio_url and os.path.exists(audio_url):
                                try:
                                    # Load reference audio waveform
                                    ref_waveform, ref_sr = torchaudio.load(audio_url)
                                    if ref_sr != 24000:
                                        ref_waveform = torchaudio.functional.resample(ref_waveform, ref_sr, 24000)
                                    
                                    # Add to message content (like inference)
                                    message_content.append(AudioContent(audio_url=audio_url))
                                    reference_waveforms.append(ref_waveform)
                                    
                                except Exception as e:
                                    print(f"Error loading reference audio {audio_url}: {e}")
                                    continue
                    
                    if message_content:
                        messages.append(Message(role="user", content=message_content))
                        
            elif role == 'assistant':
                # Extract target audio path for loss computation (but don't pass to model input)
                target_audio_path = None
                if isinstance(content, list):
                    for item in content:
                        if item.get('type') == 'audio':
                            target_audio_path = item.get('audio_url', '')
                            break
        
        return {
            'messages': messages,
            'reference_waveforms': reference_waveforms,
            'target_text': target_text,
            'target_audio_path': target_audio_path,
            'sample_id': f"sample_{idx}"
        }


class HiggsAudioDistributedTrainer:
    """Distributed trainer for Higgs-Audio LoRA fine-tuning"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.accelerator = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.collator = None
        self.text_tokenizer = None
        self.audio_tokenizer_ref = None
        
        # Setup logging
        self._setup_logging()
        
        # Initialize distributed training
        self._setup_distributed()
        
    def _setup_logging(self):
        """Setup logging configuration"""
        # Ensure output directory exists before creating file handler
        try:
            os.makedirs(self.config.output_dir, exist_ok=True)
        except Exception as e:
            # We'll still set up console logging below
            pass
        
        handlers = [logging.StreamHandler()]
        # Try to attach a file handler; if it fails, continue with stream only
        try:
            log_path = os.path.join(self.config.output_dir, "training.log")
            handlers.insert(0, logging.FileHandler(log_path))
        except Exception as e:
            # Fallback silently; console logs will still work
            pass
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers,
        )
        self.logger = logging.getLogger(__name__)
        
    def _setup_distributed(self):
        """Setup distributed training with Accelerate"""
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        deepspeed_plugin = None
        if self.config.deepspeed_config:
            try:
                from accelerate import DeepSpeedPlugin
                if os.path.exists(self.config.deepspeed_config):
                    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=self.config.deepspeed_config)
                else:
                    self.logger.warning(f"DeepSpeed config not found: {self.config.deepspeed_config}. Continuing without DeepSpeed.")
            except Exception as e:
                self.logger.warning(f"Failed to initialize DeepSpeed plugin: {e}. Continuing without DeepSpeed.")
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision="bf16" if self.config.bf16 else ("fp16" if self.config.fp16 else "no"),
            log_with="wandb" if self.config.use_wandb else None,
            project_dir=self.config.output_dir,
            kwargs_handlers=[ddp_kwargs],
            deepspeed_plugin=deepspeed_plugin,
        )
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Setup wandb
        if self.config.use_wandb and self.accelerator.is_main_process:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=self.config.__dict__
            )
    
    def load_datasets(self):
        """Load and prepare datasets"""
        self.logger.info("Loading datasets...")
        
        # Load training data from unified pipeline output
        # Files are directly in the dataset_path directory
        train_path = os.path.join(self.config.dataset_path, "train_chatml_samples.json")
        val_path = os.path.join(self.config.dataset_path, "val_chatml_samples.json")
        
        # Alternative paths in case they're in a chatml subdirectory
        if not os.path.exists(train_path):
            train_path = os.path.join(self.config.dataset_path, "chatml", "train_chatml_samples.json")
        if not os.path.exists(val_path):
            val_path = os.path.join(self.config.dataset_path, "chatml", "val_chatml_samples.json")
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found at: {train_path}")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation data not found at: {val_path}")
        
        self.logger.info(f"Loading training data from: {train_path}")
        self.logger.info(f"Loading validation data from: {val_path}")
        
        # Load ChatML samples
        with open(train_path, 'r', encoding='utf-8') as f:
            train_samples = json.load(f)
        
        with open(val_path, 'r', encoding='utf-8') as f:
            val_samples = json.load(f)
        
        self.logger.info(f"Loaded {len(train_samples)} training samples")
        self.logger.info(f"Loaded {len(val_samples)} validation samples")
        
        # Load processing statistics for monitoring
        stats_path = os.path.join(self.config.dataset_path, "processing_stats.json")
        if not os.path.exists(stats_path):
            stats_path = os.path.join(self.config.dataset_path, "chatml", "processing_stats.json")
        
        if os.path.exists(stats_path):
            with open(stats_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            
            manifest_meta = stats.get('manifest_metadata', {})
            self.logger.info(f"Dataset statistics:")
            self.logger.info(f"  • Total duration: {manifest_meta.get('total_duration_hours', 0):.2f} hours")
            self.logger.info(f"  • Total samples: {manifest_meta.get('total_samples', 0):,}")
            self.logger.info(f"  • Directories processed: {manifest_meta.get('directories_processed', 0)}")
        
        return train_samples, val_samples
    
    def create_dataloaders(self, train_samples, val_samples):
        """Create data loaders"""
        self.logger.info("Creating data loaders...")
        
        # Load tokenizers
        from transformers import AutoTokenizer
        from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)
        audio_tokenizer = load_higgs_audio_tokenizer(self.config.audio_tokenizer_path, device="cpu")
        
        # Create datasets using inference-style processing
        train_dataset = InferenceStyleDataset(
            data_file=train_samples,
            audio_tokenizer=audio_tokenizer,
            text_tokenizer=tokenizer,
            max_length=self.config.max_text_length
        )
        
        val_dataset = InferenceStyleDataset(
            data_file=val_samples,
            audio_tokenizer=audio_tokenizer,
            text_tokenizer=tokenizer,
            max_length=self.config.max_text_length
        )
        
        # Create standard collator (no custom collate function)
        from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
        from transformers import WhisperProcessor
        
        whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        
        self.collator = HiggsAudioSampleCollator(
            whisper_processor=whisper_processor,
            audio_in_token_id=tokenizer.convert_tokens_to_ids("<AUDIO>"),
            audio_out_token_id=tokenizer.convert_tokens_to_ids("<AUDIO_OUT>"),
            pad_token_id=tokenizer.pad_token_id,
            audio_stream_bos_id=tokenizer.convert_tokens_to_ids("<audio_stream_bos>"),
            audio_stream_eos_id=tokenizer.convert_tokens_to_ids("<audio_stream_eos>"),
            round_to=8,
            pad_left=False,
            encode_whisper_embed=True,
            return_audio_in_tokens=True,
            audio_num_codebooks=8,  # Fixed to 8 codebooks
            use_delay_pattern=False,
            disable_audio_codes_transform=False
        )
        
        # Store tokenizers for use in collate function
        self.text_tokenizer = tokenizer
        self.audio_tokenizer_ref = audio_tokenizer
        
        # Create data loaders
        train_sampler = DistributedSampler(train_dataset) if self.accelerator.num_processes > 1 else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if self.accelerator.num_processes > 1 else None
        
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size_per_device,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=self._inference_style_collate_fn,  # Custom collate to convert to ChatMLDatasetSample
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size_per_device,
            sampler=val_sampler,
            shuffle=False,
            collate_fn=self._inference_style_collate_fn,  # Custom collate to convert to ChatMLDatasetSample
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        self.logger.info(f"Created train dataloader with {len(self.train_dataloader)} batches")
        self.logger.info(f"Created val dataloader with {len(self.val_dataloader)} batches")
        
        return tokenizer, audio_tokenizer

    def _inference_style_collate_fn(self, batch):
        """Convert inference-style batch to ChatMLDatasetSample format for standard collator"""
        from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
        from boson_multimodal.data_types import TextContent, AudioContent
        
        chatml_samples = []
        target_audio_paths = []
        
        for item in batch:
            messages = item['messages']
            target_audio_path = item['target_audio_path']
            target_audio_paths.append(target_audio_path)
            
            # Convert Messages to ChatML dict format for prepare_chatml_sample
            chatml_dict = {"messages": []}
            
            for msg in messages:
                if msg.role == "system":
                    chatml_dict["messages"].append({"role": "system", "content": msg.content})
                elif msg.role == "user":
                    user_content = []
                    for content_item in msg.content:
                        if isinstance(content_item, TextContent):
                            user_content.append({"type": "text", "text": content_item.text})
                        elif isinstance(content_item, AudioContent):
                            user_content.append({"type": "audio", "audio_url": content_item.audio_url})
                    chatml_dict["messages"].append({"role": "user", "content": user_content})
            
            # Use prepare_chatml_sample to create proper input/label tokens
            input_tokens, label_tokens, audio_contents, speaker_id = prepare_chatml_sample(
                chatml_dict, self.text_tokenizer
            )
            
            # Create ChatMLDatasetSample with only reference audio (like inference)
            # Target audio will be generated by model, not provided as input
            chatml_sample = ChatMLDatasetSample(
                input_ids=torch.tensor(input_tokens, dtype=torch.long),
                label_ids=torch.tensor(label_tokens, dtype=torch.long),
                audio_ids_concat=torch.empty((8, 0), dtype=torch.long),  # Will be filled by collator
                audio_ids_start=torch.tensor([], dtype=torch.long),
                audio_waveforms_concat=torch.cat(item['reference_waveforms'], dim=1) if item['reference_waveforms'] else torch.empty(0, dtype=torch.float32),
                audio_waveforms_start=torch.tensor([0] if item['reference_waveforms'] else [], dtype=torch.long),
                audio_sample_rate=torch.tensor([24000] if item['reference_waveforms'] else [], dtype=torch.float32),
                audio_speaker_ids=torch.tensor([0] if item['reference_waveforms'] else [], dtype=torch.long)
            )
            
            chatml_samples.append(chatml_sample)
        
        # Use standard collator (exactly like inference)
        collated_batch = self.collator(chatml_samples)
        
        # Add target audio paths for loss computation
        collated_batch.target_audio_paths = target_audio_paths
        
        return collated_batch

    def setup_model_and_optimizer(self, tokenizer, audio_tokenizer):
        """Setup model and optimizer"""
        self.logger.info("Setting up model and optimizer...")
        
        # Create LoRA configuration
        lora_config = HiggsAudioLoRAConfig(
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            freeze_base_model=True,
            freeze_audio_tower=True,
            freeze_audio_encoder_proj=False,
            enable_audio_lora=True,
            enable_multilingual_lora=True
        )
        
        # Create LoRA model
        trainer = create_lora_model(
            model_path=self.config.model_path,
            lora_config=lora_config,
            device="cpu",  # Build on CPU; Accelerate will place on the correct GPU
            model_config=self.corrected_model_config
        )
        
        model = trainer.prepare_for_training()
        
        # Setup optimizer
        if self.config.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
        
        return model, optimizer, trainer
    
    def setup_scheduler(self, optimizer, num_training_steps):
        """Setup learning rate scheduler"""
        num_warmup_steps = int(self.config.warmup_ratio * num_training_steps)
        
        if self.config.scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler_type}")
        
        return scheduler
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load datasets
        train_samples, val_samples = self.load_datasets()
        
        # Create data loaders
        train_dataloader, val_dataloader, tokenizer, audio_tokenizer = self.create_dataloaders(
            train_samples, val_samples
        )
        
        # Setup model and optimizer
        model, optimizer, lora_trainer = self.setup_model_and_optimizer(tokenizer, audio_tokenizer)
        
        # Use Accelerate to place model/optimizer/dataloaders on the correct device(s)
        model, optimizer, train_dataloader, val_dataloader = self.accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader
        )
        
        # Calculate training steps
        steps_per_epoch = max(1, (len(train_dataloader) + self.config.gradient_accumulation_steps - 1) // self.config.gradient_accumulation_steps)
        num_training_steps = steps_per_epoch * self.config.num_epochs
        
        # Setup scheduler
        scheduler = self.setup_scheduler(optimizer, num_training_steps)
        scheduler = self.accelerator.prepare(scheduler)
        
        # Training loop
        global_step = 0
        best_val_loss = float('inf')
        train_loss = 0.0
        
        for epoch in range(self.config.num_epochs):
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):
                model.train()
                
                # Convert batch to dict
                if hasattr(batch, "__dict__"):
                    batch_dict = {k: v for k, v in vars(batch).items() if v is not None}
                else:
                    from dataclasses import asdict
                    batch_dict = {k: v for k, v in asdict(batch).items() if v is not None}
                
                # Extract labels before removing them from batch
                labels = batch_dict.get('label_ids')
                audio_labels = batch_dict.get('label_audio_ids')
                
                # CRITICAL FIX: Create clean model inputs without any labels
                model_inputs = {
                    'input_ids': batch_dict.get('input_ids'),
                    'attention_mask': batch_dict.get('attention_mask'),
                    'audio_features': batch_dict.get('audio_features'),
                    'audio_feature_attention_mask': batch_dict.get('audio_feature_attention_mask'),
                    'audio_in_ids': batch_dict.get('audio_in_ids'),
                    'audio_in_ids_start': batch_dict.get('audio_in_ids_start'),
                    'audio_out_ids': batch_dict.get('audio_out_ids'),
                    'audio_out_ids_start': batch_dict.get('audio_out_ids_start'),
                    'audio_out_ids_start_group_loc': batch_dict.get('audio_out_ids_start_group_loc'),
                    'reward': batch_dict.get('reward'),
                }
                # Remove None values
                model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
                
                # Get the underlying HiggsAudioModel and call it directly
                if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
                    # PEFT wrapped: model.base_model.model is the actual HiggsAudioModel
                    actual_model = model.base_model.model
                elif hasattr(model, 'module'):
                    # Accelerate wrapped: model.module is the actual model
                    actual_model = model.module
                else:
                    actual_model = model
                
                # CRITICAL FIX: Ensure all model inputs are on the same device as the model
                model_device = next(actual_model.parameters()).device
                model_inputs = {
                    k: v.to(model_device) if torch.is_tensor(v) else v 
                    for k, v in model_inputs.items()
                }
                
                # Also ensure labels are on the correct device
                if torch.is_tensor(labels):
                    labels = labels.to(model_device)
                if torch.is_tensor(audio_labels):
                    audio_labels = audio_labels.to(model_device)

                # Forward pass - call model.forward() directly with explicit arguments
                outputs = actual_model(**model_inputs)
                
                # Prepare batch for loss computation
                loss_batch = {
                    'labels': labels,
                    'audio_labels': audio_labels
                }
                
                # Compute loss using LoRA trainer
                loss_dict = lora_trainer.compute_loss(loss_batch, outputs)
                
                # Extract individual losses
                text_loss = loss_dict['text_loss']
                audio_loss = loss_dict['audio_loss'] 
                combined_loss = loss_dict['combined_loss']
                
                # CRITICAL: Backward pass and optimizer steps
                self.accelerator.backward(combined_loss)
                
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                train_loss += combined_loss.item()
                global_step += 1
                
                # Logging with detailed loss breakdown
                if global_step % self.config.logging_steps == 0:
                    avg_loss = train_loss / self.config.logging_steps
                    text_loss_val = text_loss.item() if isinstance(text_loss, torch.Tensor) else text_loss
                    audio_loss_val = audio_loss.item() if isinstance(audio_loss, torch.Tensor) else audio_loss
                    combined_loss_val = combined_loss.item() if isinstance(combined_loss, torch.Tensor) else combined_loss
                    
                    self.logger.info(f"Step {global_step}: Text Loss = {text_loss_val:.4f}, Audio Loss = {audio_loss_val:.4f}, Combined Loss = {combined_loss_val:.4f}")
                    
                    if self.config.use_wandb and self.accelerator.is_main_process:
                        wandb.log({
                            "train/text_loss": text_loss_val,
                            "train/audio_loss": audio_loss_val,
                            "train/combined_loss": combined_loss_val,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train/step": global_step
                        })
                    
                    train_loss = 0.0
            
            # Validation
            if global_step % self.config.eval_steps == 0:
                val_loss = self.validate(model, val_dataloader, lora_trainer)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(model, global_step, "best")
                
                if self.config.use_wandb and self.accelerator.is_main_process:
                    wandb.log({"val_loss": val_loss, "global_step": global_step})
            
            # Save checkpoint
            if global_step % self.config.save_steps == 0:
                self.save_checkpoint(model, global_step)
        
        # Final save
        self.save_checkpoint(model, global_step, "final")
        self.logger.info("Training completed!")
    
    def validate(self, model, val_dataloader, lora_trainer):
        """Validation loop"""
        model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                # Convert batch to dict
                if hasattr(batch, "__dict__"):
                    batch_dict = {k: v for k, v in vars(batch).items() if v is not None}
                else:
                    from dataclasses import asdict
                    batch_dict = {k: v for k, v in asdict(batch).items() if v is not None}
                
                # Extract labels before removing them from batch
                labels = batch_dict.get('label_ids')
                audio_labels = batch_dict.get('label_audio_ids')
                
                # CRITICAL FIX: Create clean model inputs without any labels
                model_inputs = {
                    'input_ids': batch_dict.get('input_ids'),
                    'attention_mask': batch_dict.get('attention_mask'),
                    'audio_features': batch_dict.get('audio_features'),
                    'audio_feature_attention_mask': batch_dict.get('audio_feature_attention_mask'),
                    'audio_in_ids': batch_dict.get('audio_in_ids'),
                    'audio_in_ids_start': batch_dict.get('audio_in_ids_start'),
                    'audio_out_ids': batch_dict.get('audio_out_ids'),
                    'audio_out_ids_start': batch_dict.get('audio_out_ids_start'),
                    'audio_out_ids_start_group_loc': batch_dict.get('audio_out_ids_start_group_loc'),
                    'reward': batch_dict.get('reward'),
                }
                # Remove None values
                model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
                
                # Get the underlying HiggsAudioModel and call it directly
                if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
                    # PEFT wrapped: model.base_model.model is the actual HiggsAudioModel
                    actual_model = model.base_model.model
                elif hasattr(model, 'module'):
                    # Accelerate wrapped: model.module is the actual model
                    actual_model = model.module
                else:
                    actual_model = model
                
                # CRITICAL FIX: Ensure all model inputs are on the same device as the model
                model_device = next(actual_model.parameters()).device
                model_inputs = {
                    k: v.to(model_device) if torch.is_tensor(v) else v 
                    for k, v in model_inputs.items()
                }
                
                # Also ensure labels are on the correct device
                if torch.is_tensor(labels):
                    labels = labels.to(model_device)
                if torch.is_tensor(audio_labels):
                    audio_labels = audio_labels.to(model_device)

                # Forward pass - call model.forward() directly with explicit arguments
                outputs = actual_model(**model_inputs)
                
                # Prepare batch for loss computation
                loss_batch = {
                    'labels': labels,
                    'audio_labels': audio_labels
                }
                
                # Compute loss using LoRA trainer
                loss_dict = lora_trainer.compute_loss(loss_batch, outputs)
                
                # Extract individual losses
                text_loss = loss_dict['text_loss']
                audio_loss = loss_dict['audio_loss'] 
                combined_loss = loss_dict['combined_loss']
                
                val_loss += combined_loss.item()
                num_batches += 1
        
        avg_val_loss = val_loss / num_batches
        self.logger.info(f"Validation loss: {avg_val_loss:.4f}")
        
        model.train()
        return avg_val_loss
    
    def save_checkpoint(self, model, global_step, suffix=""):
        """Save model checkpoint"""
        if not self.accelerator.is_main_process:
            return
        
        save_dir = Path(self.config.output_dir) / f"checkpoint-{global_step}"
        if suffix:
            save_dir = Path(self.config.output_dir) / f"checkpoint-{suffix}"
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA adapters
        unwrapped_model = self.accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(save_dir)
        
        # Save training config
        with open(save_dir / "training_config.json", 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        self.logger.info(f"Checkpoint saved to {save_dir}")


def main():
    """Main training function"""
    import argparse
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        return v.lower() in ("yes", "true", "t", "1")
    
    parser = argparse.ArgumentParser(description="Distributed LoRA training for Higgs-Audio")
    parser.add_argument("--config", type=str, help="Path to training config YAML file")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to processed dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    # Training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Alias for batch_size_per_device")
    parser.add_argument("--batch_size_per_device", type=int, default=None, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=None, help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=None, help="Max gradient norm")
    # Precision
    parser.add_argument("--bf16", type=str2bool, nargs='?', const=True, default=None, help="Use bfloat16")
    parser.add_argument("--fp16", type=str2bool, nargs='?', const=True, default=None, help="Use float16")
    # Logging / eval / save
    parser.add_argument("--logging_steps", type=int, default=None, help="Logging frequency in steps")
    parser.add_argument("--eval_steps", type=int, default=None, help="Evaluation frequency in steps")
    parser.add_argument("--save_steps", type=int, default=None, help="Checkpoint save frequency in steps")
    # Dataloader
    parser.add_argument("--num_workers", type=int, default=None, help="Number of DataLoader workers")
    # Wandb
    parser.add_argument("--use_wandb", type=str2bool, default=True)
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")
    # DeepSpeed
    parser.add_argument("--deepspeed_config", type=str, default=None, help="Path to DeepSpeed JSON config")
    # Debug controls for audio flow logging
    parser.add_argument("--debug_audio_flow", action="store_true", default=False,
                        help="Enable verbose logs for reference/target audio resolution and batching.")
    parser.add_argument("--debug_audio_warn_limit", type=int, default=50,
                        help="Max number of dataset-level audio warnings to print when debug_audio_flow is on.")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = TrainingConfig(**config_dict)
    else:
        config = TrainingConfig()
    
    # Override with command line arguments
    if args.dataset_path:
        config.dataset_path = args.dataset_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    # Batch size alias support
    if args.batch_size_per_device is not None:
        config.batch_size_per_device = args.batch_size_per_device
    elif args.batch_size is not None:
        config.batch_size_per_device = args.batch_size
    if args.gradient_accumulation_steps is not None:
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.warmup_ratio is not None:
        config.warmup_ratio = args.warmup_ratio
    if args.max_grad_norm is not None:
        config.max_grad_norm = args.max_grad_norm
    if args.bf16 is not None:
        config.bf16 = args.bf16
    if args.fp16 is not None:
        config.fp16 = args.fp16
    if args.logging_steps is not None:
        config.logging_steps = args.logging_steps
    if args.eval_steps is not None:
        config.eval_steps = args.eval_steps
    if args.save_steps is not None:
        config.save_steps = args.save_steps
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.use_wandb is not None:
        config.use_wandb = args.use_wandb
    if args.wandb_project is not None:
        config.wandb_project = args.wandb_project
    if args.wandb_run_name is not None:
        config.wandb_run_name = args.wandb_run_name
    if args.deepspeed_config is not None:
        config.deepspeed_config = args.deepspeed_config
    config.debug_audio_flow = args.debug_audio_flow
    config.debug_audio_warn_limit = args.debug_audio_warn_limit
    
    # Start training
    trainer = HiggsAudioDistributedTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
