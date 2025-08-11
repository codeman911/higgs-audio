#!/usr/bin/env python3
"""
Distributed LoRA Training Script for Higgs-Audio V2
Simple, robust, and working implementation.
"""

import os
import sys
import json
import argparse
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, WhisperProcessor
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn as nn
from accelerate import Accelerator
from tqdm import tqdm
from pathlib import Path
import logging

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


class SimpleDataset(Dataset):
    """Simple dataset that loads ChatML JSON files"""
    
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, list):
            self.samples = data
        elif isinstance(data, dict):
            self.samples = data.get('samples', data.get('data', []))
        else:
            self.samples = []
        
        logger.info(f"Loaded {len(self.samples)} samples from {json_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, tokenizer, audio_tokenizer, collator, sample_rate=24000):
    """Simple collate function that processes samples for training"""
    
    chatml_samples = []
    
    for sample in batch:
        # Get messages from sample
        messages = sample.get('messages', [])
        
        # Build ChatML dict
        chatml_dict = {"messages": []}
        
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content')
            
            if role and content:
                # Handle different content formats
                if isinstance(content, list):
                    # Multi-modal content
                    processed_content = []
                    for item in content:
                        if item.get('type') == 'text':
                            processed_content.append({"type": "text", "text": item.get('text', '')})
                        elif item.get('type') == 'audio':
                            audio_url = item.get('audio_url', '')
                            if audio_url:
                                processed_content.append({"type": "audio", "audio_url": audio_url})
                    chatml_dict["messages"].append({"role": role, "content": processed_content})
                else:
                    # Simple text content
                    chatml_dict["messages"].append({"role": role, "content": content})
        
        # Tokenize with prepare_chatml_sample
        try:
            input_tokens, label_tokens, audio_contents, speaker_id = prepare_chatml_sample(
                chatml_dict, tokenizer
            )
        except Exception as e:
            logger.warning(f"Failed to prepare sample: {e}")
            # Create empty sample
            input_tokens = [tokenizer.pad_token_id]
            label_tokens = [-100]
            audio_contents = []
            speaker_id = 0
        
        # Process audio if present
        audio_ids_list = []
        audio_waveforms_list = []
        
        for audio_content in audio_contents:
            if audio_content and hasattr(audio_content, 'audio_url'):
                audio_path = audio_content.audio_url
                if audio_path and os.path.exists(audio_path):
                    try:
                        # Tokenize audio
                        audio_codes = audio_tokenizer.encode(audio_path)
                        # Ensure tensor is on CPU
                        if audio_codes.is_cuda:
                            audio_codes = audio_codes.cpu()
                        # Ensure 8 codebooks
                        if audio_codes.shape[0] != 8:
                            if audio_codes.shape[0] > 8:
                                audio_codes = audio_codes[:8, :]
                            else:
                                padding = torch.zeros(8 - audio_codes.shape[0], audio_codes.shape[1])
                                audio_codes = torch.cat([audio_codes, padding], dim=0)
                        audio_ids_list.append(audio_codes)
                        
                        # Load waveform
                        waveform, sr = torchaudio.load(audio_path)
                        if sr != sample_rate:
                            waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
                        if waveform.shape[0] > 1:
                            waveform = waveform.mean(dim=0, keepdim=True)
                        waveform = waveform.squeeze(0)  # Flatten to 1D
                        audio_waveforms_list.append(waveform)
                    except Exception as e:
                        logger.warning(f"Failed to process audio {audio_path}: {e}")
        
        # Create tensors
        if audio_ids_list:
            audio_ids_concat = torch.cat(audio_ids_list, dim=1)
            audio_ids_start = torch.cumsum(
                torch.tensor([0] + [ids.shape[1] for ids in audio_ids_list]), dim=0
            )
        else:
            audio_ids_concat = torch.zeros((8, 0), dtype=torch.long)
            audio_ids_start = torch.tensor([0], dtype=torch.long)
        
        if audio_waveforms_list:
            audio_waveforms_concat = torch.cat(audio_waveforms_list, dim=0)
            lengths = [len(wv) for wv in audio_waveforms_list]
            audio_waveforms_start = torch.tensor([0] + lengths[:-1]).cumsum(dim=0)
            audio_sample_rate = torch.tensor([sample_rate] * len(audio_waveforms_list))
            audio_speaker_indices = torch.zeros(len(audio_waveforms_list), dtype=torch.long)
        else:
            audio_waveforms_concat = torch.tensor([])
            audio_waveforms_start = torch.tensor([0], dtype=torch.long)
            audio_sample_rate = torch.tensor([sample_rate])
            audio_speaker_indices = torch.tensor([0], dtype=torch.long)
        
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
    return collator(chatml_samples)


def main():
    parser = argparse.ArgumentParser(description="Higgs-Audio LoRA Training")
    
    # Data arguments
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to dataset directory containing train/val JSON files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for model and checkpoints")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, 
                        default="bosonai/higgs-audio-v2-generation-3B-base",
                        help="Path to base model")
    parser.add_argument("--audio_tokenizer_path", type=str,
                        default="bosonai/higgs-audio-v2-tokenizer",
                        help="Path to audio tokenizer")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")
    
    # Other arguments
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of dataloader workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--log_steps", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every N steps")
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision training")
    
    args = parser.parse_args()
    
    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision
    )
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Loading data from {args.dataset_path}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Load tokenizers
    logger.info("Loading tokenizers...")
    # Text tokenizer from model path
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Audio tokenizer - load on CPU, accelerator will handle device placement
    audio_tokenizer = load_higgs_audio_tokenizer(args.audio_tokenizer_path, device="cpu")
    
    # Load model config
    model_config = AutoConfig.from_pretrained(args.model_path)
    
    # Initialize collator
    logger.info("Initializing collator...")
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    
    collator = HiggsAudioSampleCollator(
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
        audio_num_codebooks=8
    )
    
    # Load datasets
    train_path = os.path.join(args.dataset_path, "train_chatml_samples.json")
    val_path = os.path.join(args.dataset_path, "val_chatml_samples.json")
    
    if not os.path.exists(train_path):
        logger.error(f"Training file not found: {train_path}")
        sys.exit(1)
    
    logger.info(f"Loading training data from {train_path}")
    train_dataset = SimpleDataset(train_path)
    
    val_dataset = None
    if os.path.exists(val_path):
        logger.info(f"Loading validation data from {val_path}")
        val_dataset = SimpleDataset(val_path)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, audio_tokenizer, collator),
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, tokenizer, audio_tokenizer, collator),
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    # Load model
    logger.info("Loading model...")
    model = HiggsAudioModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": accelerator.device}
    )
    
    # Apply LoRA
    logger.info("Applying LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            # Audio output projection layers
            "audio_decoder_proj.audio_lm_head",
            "audio_decoder_proj.text_lm_head",
            # Audio MLP layers in transformer blocks (targeting later layers for fine-tuning)
            "layers.10.audio_mlp.gate_proj",
            "layers.10.audio_mlp.up_proj",
            "layers.10.audio_mlp.down_proj",
            "layers.11.audio_mlp.gate_proj",
            "layers.11.audio_mlp.up_proj",
            "layers.11.audio_mlp.down_proj",
            "layers.12.audio_mlp.gate_proj",
            "layers.12.audio_mlp.up_proj",
            "layers.12.audio_mlp.down_proj",
            "layers.13.audio_mlp.gate_proj",
            "layers.13.audio_mlp.up_proj",
            "layers.13.audio_mlp.down_proj",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Create a wrapper to handle the labels -> label_ids mapping
    class HiggsAudioModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, **kwargs):
            # PEFT passes 'labels' but HiggsAudioModel expects 'label_ids'
            if 'labels' in kwargs:
                kwargs['label_ids'] = kwargs.pop('labels')
            return self.model(**kwargs)
    
    # Wrap the model to handle argument mapping
    wrapped_model = HiggsAudioModelWrapper(model)
    model = get_peft_model(wrapped_model, lora_config)
    model.print_trainable_parameters()
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ... (rest of the code remains the same)
    num_training_steps = len(train_dataloader) * args.num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_training_steps
    )
    
    # Prepare for training
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    
    if val_dataloader:
        val_dataloader = accelerator.prepare(val_dataloader)
    
    # Training loop
    logger.info("Starting training...")
    global_step = 0
    
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                # Forward pass - map collator output to model input correctly
                # The collator returns audio_in_wv but model expects audio_features
                # Use 'labels' here because PEFT wrapper will convert it to 'label_ids'
                outputs = model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    audio_features=batch.audio_in_wv if hasattr(batch, 'audio_in_wv') else None,  
                    audio_feature_attention_mask=batch.audio_feature_attention_mask if hasattr(batch, 'audio_feature_attention_mask') else None,
                    audio_in_ids=batch.audio_in_ids if hasattr(batch, 'audio_in_ids') else None,
                    audio_in_ids_start=batch.audio_in_ids_start if hasattr(batch, 'audio_in_ids_start') else None,
                    audio_out_ids=batch.audio_out_ids if hasattr(batch, 'audio_out_ids') else None,
                    audio_out_ids_start=batch.audio_out_ids_start if hasattr(batch, 'audio_out_ids_start') else None,
                    audio_out_ids_start_group_loc=batch.audio_out_ids_start_group_loc if hasattr(batch, 'audio_out_ids_start_group_loc') else None,
                    labels=batch.label_ids,  # Use 'labels' because PEFT expects it
                    label_audio_ids=batch.label_audio_ids if hasattr(batch, 'label_audio_ids') else None,
                    return_dict=True
                )
                
                loss = outputs.loss
                
                # Backward pass
                accelerator.backward(loss)
                
                # Gradient clipping
                if args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                global_step += 1
                
                # Logging
                if global_step % args.log_steps == 0:
                    avg_loss = total_loss / (step + 1)
                    progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
                    logger.info(f"Step {global_step}: loss={avg_loss:.4f}")
                
                # Save checkpoint
                if global_step % args.save_steps == 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        # Validation
        if val_dataloader:
            model.eval()
            val_loss = 0
            val_steps = 0
            
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validation"):
                    outputs = model(
                        input_ids=batch.input_ids,
                        attention_mask=batch.attention_mask,
                        audio_features=batch.audio_in_wv if hasattr(batch, 'audio_in_wv') else None,  
                        audio_feature_attention_mask=batch.audio_feature_attention_mask if hasattr(batch, 'audio_feature_attention_mask') else None,
                        audio_in_ids=batch.audio_in_ids if hasattr(batch, 'audio_in_ids') else None,
                        audio_in_ids_start=batch.audio_in_ids_start if hasattr(batch, 'audio_in_ids_start') else None,
                        audio_out_ids=batch.audio_out_ids if hasattr(batch, 'audio_out_ids') else None,
                        audio_out_ids_start=batch.audio_out_ids_start if hasattr(batch, 'audio_out_ids_start') else None,
                        audio_out_ids_start_group_loc=batch.audio_out_ids_start_group_loc if hasattr(batch, 'audio_out_ids_start_group_loc') else None,
                        labels=batch.label_ids,  # Use 'labels' because PEFT expects it
                        label_audio_ids=batch.label_audio_ids if hasattr(batch, 'label_audio_ids') else None,
                        return_dict=True
                    )
                    val_loss += outputs.loss.item()
                    val_steps += 1
            
            avg_val_loss = val_loss / val_steps
            logger.info(f"Epoch {epoch+1} - Validation loss: {avg_val_loss:.4f}")
    
    # Save final model
    final_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    logger.info(f"Training complete! Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
