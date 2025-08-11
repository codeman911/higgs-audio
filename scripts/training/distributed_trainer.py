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
import torchaudio  # Add missing import for audio processing

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
        
        return train_path, val_path
    
    def create_dataloaders(self, train_path, val_path):
        """Create data loaders"""
        self.logger.info("Creating data loaders...")
        
        # Load tokenizers
        from transformers import AutoTokenizer
        from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)
        audio_tokenizer = load_higgs_audio_tokenizer(self.config.audio_tokenizer_path, device="cpu")
        
        # Create datasets using inference-style processing
        train_dataset = InferenceStyleDataset(
            data_file=train_path,
            audio_tokenizer=audio_tokenizer,
            text_tokenizer=tokenizer,
            max_length=self.config.max_text_length
        )
        
        val_dataset = InferenceStyleDataset(
            data_file=val_path,
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
        """
        Collate function that matches inference pipeline exactly for zero-shot voice cloning.
        Uses Message objects and loads reference audio on-demand.
        """
        from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
        from boson_multimodal.data_types import TextContent, AudioContent
        import torchaudio
        
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
                    # System message
                    chatml_dict["messages"].append({"role": "system", "content": msg.content})
                elif msg.role == "user":
                    # User message with text and reference audio
                    if isinstance(msg.content, list):
                        user_content = []
                        
                        for content_item in msg.content:
                            if isinstance(content_item, TextContent):
                                # Extract target text
                                text = content_item.text
                                user_content.append({"type": "text", "text": text})
                            
                            elif isinstance(content_item, AudioContent):
                                # Reference audio - load on-demand like inference
                                audio_url = content_item.audio_url
                                user_content.append({"type": "audio", "audio_url": audio_url})
                        
                        chatml_dict["messages"].append({"role": "user", "content": user_content})
                    
            # Use prepare_chatml_sample to create proper input/label tokens
            input_tokens, label_tokens, audio_contents, speaker_id = prepare_chatml_sample(
                chatml_dict, self.text_tokenizer
            )
            
            # Load reference audio waveforms from AudioContent objects
            reference_waveforms = []
            audio_positions = []  # Track where audio tokens should be inserted
            
            for i, audio_content in enumerate(audio_contents):
                if audio_content and hasattr(audio_content, 'audio_url'):
                    audio_path = audio_content.audio_url
                    if audio_path and os.path.exists(audio_path):
                        try:
                            waveform, sample_rate = torchaudio.load(audio_path)
                            # Ensure mono audio
                            if waveform.shape[0] > 1:
                                waveform = waveform.mean(dim=0, keepdim=True)
                            reference_waveforms.append(waveform)
                            
                            # Find position of this audio in the token sequence
                            # Audio contents are typically placed at the beginning for reference
                            audio_positions.append(i)
                        except Exception as e:
                            self.logger.warning(f"Failed to load reference audio {audio_path}: {e}")
            
            # Insert audio tokens into input_ids at appropriate positions
            # The audio_in_token_id should be inserted where reference audio is expected
            audio_in_token_id = self.text_tokenizer.convert_tokens_to_ids("<|AUDIO|>")
            
            # For zero-shot voice cloning, reference audio typically comes at the beginning
            # Insert audio tokens at the start of the sequence for each reference audio
            if reference_waveforms:
                # Insert audio tokens at the beginning of the user message
                # Find where user content starts (after system message if present)
                modified_input_tokens = []
                modified_label_tokens = []
                
                # Add audio tokens for reference audio at the beginning
                for _ in reference_waveforms:
                    modified_input_tokens.append(audio_in_token_id)
                    modified_label_tokens.append(-100)  # Don't compute loss on audio tokens
                
                # Add the rest of the tokens
                modified_input_tokens.extend(input_tokens)
                modified_label_tokens.extend(label_tokens)
                
                input_tokens = modified_input_tokens
                label_tokens = modified_label_tokens
            
            # Create ChatMLDatasetSample with only reference audio (like inference)
            # Target audio will be generated by model, not provided as input
            chatml_sample = ChatMLDatasetSample(
                input_ids=torch.tensor(input_tokens, dtype=torch.long),
                label_ids=torch.tensor(label_tokens, dtype=torch.long),
                audio_ids_concat=torch.empty((8, 0), dtype=torch.long),  # Will be filled by collator
                audio_ids_start=torch.tensor([], dtype=torch.long),
                audio_waveforms_concat=torch.cat(reference_waveforms, dim=1) if reference_waveforms else torch.empty(1, 0, dtype=torch.float32),  # Ensure 2D tensor
                audio_waveforms_start=torch.tensor([0] if reference_waveforms else [], dtype=torch.long),
                audio_sample_rate=torch.tensor([24000] if reference_waveforms else [], dtype=torch.float32),
                audio_speaker_indices=torch.tensor([speaker_id if speaker_id is not None else 0] if reference_waveforms else [], dtype=torch.long)  # Use speaker_id from prepare_chatml_sample or default to 0
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
        
        # Load model configuration (like inference pipeline)
        from boson_multimodal.model.higgs_audio.configuration_higgs_audio import HiggsAudioConfig
        model_config = HiggsAudioConfig.from_pretrained(self.config.model_path)
        
        # Ensure 8-codebook configuration alignment (from memories)
        expected_codebooks = 8
        if model_config.audio_num_codebooks != expected_codebooks:
            self.logger.info(f"Updating model config: {model_config.audio_num_codebooks} -> {expected_codebooks} codebooks")
            model_config.audio_num_codebooks = expected_codebooks
        
        # Validate alignment with audio tokenizer
        if audio_tokenizer.num_codebooks != expected_codebooks:
            self.logger.error(f"Audio tokenizer codebook mismatch: {audio_tokenizer.num_codebooks} != {expected_codebooks}")
            raise ValueError(f"Audio tokenizer must have {expected_codebooks} codebooks")
        
        self.logger.info(f" Model and tokenizer aligned: {expected_codebooks} codebooks")
        
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
        
        # Create LoRA model (like inference but with LoRA)
        trainer = create_lora_model(
            model_path=self.config.model_path,
            lora_config=lora_config,
            device="cpu",  # Build on CPU; Accelerate will place on the correct GPU
            model_config=model_config  # Use the loaded and validated config
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
        
        # Setup model and optimizer
        tokenizer, audio_tokenizer = self.create_dataloaders(
            os.path.join(self.config.dataset_path, "train_chatml_samples.json"),
            os.path.join(self.config.dataset_path, "val_chatml_samples.json")
        )
        
        # Store tokenizers for training loop
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        
        model, optimizer, lora_trainer = self.setup_model_and_optimizer(tokenizer, audio_tokenizer)
        
        # Calculate total training steps
        num_training_steps = len(self.train_dataloader) * self.config.num_epochs
        scheduler = self.setup_scheduler(optimizer, num_training_steps)
        
        # Prepare for distributed training
        model, optimizer, self.train_dataloader, self.val_dataloader, scheduler = self.accelerator.prepare(
            model, optimizer, self.train_dataloader, self.val_dataloader, scheduler
        )
        
        # Initialize wandb
        if self.config.use_wandb and self.accelerator.is_main_process:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=vars(self.config)
            )
        
        # Training loop
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            model.train()
            total_loss = 0
            
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
                disable=not self.accelerator.is_main_process
            )
            
            for step, batch in enumerate(progress_bar):
                with self.accelerator.accumulate(model):
                    # Model input is exactly like inference: only reference audio + target text
                    model_inputs = {
                        'input_ids': batch.input_ids,
                        'attention_mask': batch.attention_mask,
                        'audio_in_ids': batch.audio_in_ids,  # Reference audio conditioning
                        'audio_in_wv': batch.audio_in_wv,    # Reference waveforms for Whisper
                    }
                    
                    # Forward pass - model generates audio tokens (like inference)
                    outputs = model(**model_inputs)
                    
                    # Load target audio on-demand for loss computation
                    target_audio_tokens = []
                    for target_path in batch.target_audio_paths:
                        if target_path and os.path.exists(target_path):
                            try:
                                # Tokenize target audio for loss computation
                                target_tokens = self.audio_tokenizer.encode(target_path)  # (8, seq_len)
                                target_audio_tokens.append(target_tokens)
                            except Exception as e:
                                self.logger.warning(f"Failed to tokenize target audio {target_path}: {e}")
                                # Use empty tensor as fallback
                                target_audio_tokens.append(torch.empty((8, 0), dtype=torch.long))
                        else:
                            # Use empty tensor for missing target audio
                            target_audio_tokens.append(torch.empty((8, 0), dtype=torch.long))
                    
                    # Concatenate target audio tokens for loss computation
                    if target_audio_tokens:
                        # Pad to same length for batching
                        max_len = max(t.shape[1] for t in target_audio_tokens)
                        if max_len > 0:
                            padded_targets = []
                            for tokens in target_audio_tokens:
                                if tokens.shape[1] < max_len:
                                    padding = torch.zeros((8, max_len - tokens.shape[1]), dtype=torch.long)
                                    tokens = torch.cat([tokens, padding], dim=1)
                                padded_targets.append(tokens)
                            
                            target_audio_batch = torch.stack(padded_targets, dim=0)  # (batch_size, 8, max_len)
                            target_audio_batch = target_audio_batch.to(outputs.logits.device)
                        else:
                            target_audio_batch = None
                    else:
                        target_audio_batch = None
                    
                    # Compute loss using LoRA trainer
                    loss_dict = lora_trainer.compute_loss(
                        text_logits=outputs.logits,
                        text_labels=batch.label_ids,
                        audio_logits=outputs.audio_logits if hasattr(outputs, 'audio_logits') else None,
                        audio_labels=target_audio_batch
                    )
                    
                    loss = loss_dict['combined_loss']
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    
                    # Gradient clipping
                    if self.config.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Logging
                total_loss += loss.item()
                global_step += 1
                
                if global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / (step + 1)
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                    })
                    
                    if self.config.use_wandb and self.accelerator.is_main_process:
                        wandb.log({
                            'train/loss': avg_loss,
                            'train/text_loss': loss_dict.get('text_loss', 0),
                            'train/audio_loss': loss_dict.get('audio_loss', 0),
                            'train/learning_rate': scheduler.get_last_lr()[0],
                            'train/global_step': global_step
                        })
                
                # Validation
                if global_step % self.config.eval_steps == 0:
                    val_loss = self.validate(model, self.val_dataloader, lora_trainer)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_checkpoint(model, global_step, suffix="best")
                    
                    model.train()
                
                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    self.save_checkpoint(model, global_step)
        
        # Final checkpoint
        self.save_checkpoint(model, global_step, suffix="final")
        
        if self.config.use_wandb and self.accelerator.is_main_process:
            wandb.finish()
        
        self.logger.info("Training completed!")
    
    def validate(self, model, val_dataloader, lora_trainer):
        """Validation loop"""
        model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                # Model input is exactly like inference: only reference audio + target text
                model_inputs = {
                    'input_ids': batch.input_ids,
                    'attention_mask': batch.attention_mask,
                    'audio_in_ids': batch.audio_in_ids,  # Reference audio conditioning
                    'audio_in_wv': batch.audio_in_wv,    # Reference waveforms for Whisper
                }
                
                # Forward pass - model generates audio tokens (like inference)
                outputs = model(**model_inputs)
                
                # Load target audio on-demand for loss computation
                target_audio_tokens = []
                for target_path in batch.target_audio_paths:
                    if target_path and os.path.exists(target_path):
                        try:
                            # Tokenize target audio for loss computation
                            target_tokens = self.audio_tokenizer.encode(target_path)  # (8, seq_len)
                            target_audio_tokens.append(target_tokens)
                        except Exception as e:
                            self.logger.warning(f"Failed to tokenize target audio {target_path}: {e}")
                            # Use empty tensor as fallback
                            target_audio_tokens.append(torch.empty((8, 0), dtype=torch.long))
                    else:
                        # Use empty tensor for missing target audio
                        target_audio_tokens.append(torch.empty((8, 0), dtype=torch.long))
                
                # Concatenate target audio tokens for loss computation
                if target_audio_tokens:
                    # Pad to same length for batching
                    max_len = max(t.shape[1] for t in target_audio_tokens)
                    if max_len > 0:
                        padded_targets = []
                        for tokens in target_audio_tokens:
                            if tokens.shape[1] < max_len:
                                padding = torch.zeros((8, max_len - tokens.shape[1]), dtype=torch.long)
                                tokens = torch.cat([tokens, padding], dim=1)
                            padded_targets.append(tokens)
                        
                        target_audio_batch = torch.stack(padded_targets, dim=0)  # (batch_size, 8, max_len)
                        target_audio_batch = target_audio_batch.to(outputs.logits.device)
                    else:
                        target_audio_batch = None
                else:
                    target_audio_batch = None
                
                # Compute loss using LoRA trainer
                loss_dict = lora_trainer.compute_loss(
                    text_logits=outputs.logits,
                    text_labels=batch.label_ids,
                    audio_logits=outputs.audio_logits if hasattr(outputs, 'audio_logits') else None,
                    audio_labels=target_audio_batch
                )
                
                val_loss += loss_dict['combined_loss'].item()
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
