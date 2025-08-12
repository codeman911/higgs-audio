#!/usr/bin/env python3
import os
import sys
import json
import argparse
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, WhisperProcessor, get_cosine_schedule_with_warmup
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


def collate_fn(batch, tokenizer, audio_tokenizer, collator, sample_rate=24000, use_cached_codes=False):
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
                        # Tokenize audio (with optional caching for speed)
                        audio_codes = None
                        if use_cached_codes:
                            cached_codes = f"{audio_path}.codes.pt"
                            if os.path.exists(cached_codes):
                                try:
                                    audio_codes = torch.load(cached_codes, map_location="cpu")
                                except Exception:
                                    audio_codes = None
                        if audio_codes is None:
                            audio_codes = audio_tokenizer.encode(audio_path)
                        # Ensure tensor is on CPU
                        if audio_codes.is_cuda:
                            audio_codes = audio_codes.cpu()
                        # Ensure 8 codebooks
                        if audio_codes.shape[0] != 8:
                            if audio_codes.shape[0] > 8:
                                audio_codes = audio_codes[:8, :]
                            else:
                                padding = torch.zeros(
                                    (8 - audio_codes.shape[0], audio_codes.shape[1]),
                                    dtype=torch.long, device=audio_codes.device
                                )
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
    parser.add_argument("--learning_rate", type=float, default=5e-4,
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
    parser.add_argument("--prefetch_factor", type=int, default=8,
                        help="DataLoader prefetch factor per worker (if num_workers>0)")
    parser.add_argument("--persistent_workers", action="store_true", default=True,
                        help="Keep workers alive across epochs for speed")
    parser.add_argument("--audio_label_smoothing", type=float, default=0.05,
                        help="Label smoothing for audio CE over codebooks")
    parser.add_argument("--compile_model", action="store_true", default=False,
                        help="Enable torch.compile (PyTorch >= 2.4) for extra speed")
    parser.add_argument("--use_cached_codes", action="store_true", default=False,
                        help="Use <audio_path>.codes.pt if present (faster training)")
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
    
    # Fast + stable matmul on Hopper; keep BF16 for mixed precision
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
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
        round_to=8,  # CRITICAL: Documentation recommends round_to=8 for optimal batching
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
        collate_fn=lambda b: collate_fn(b, tokenizer, audio_tokenizer, collator, use_cached_codes=args.use_cached_codes),
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
        persistent_workers=(args.persistent_workers if args.num_workers > 0 else False),
    )
    
    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, tokenizer, audio_tokenizer, collator, use_cached_codes=args.use_cached_codes),
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
            persistent_workers=(args.persistent_workers if args.num_workers > 0 else False),
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
            # CRITICAL: Audio output head - generates final audio tokens
            "audio_decoder_proj.audio_lm_head",
            
            # STRATEGY 1: Audio MLP layers for ALL layers (0-27) - audio generation pathway
            # Based on model analysis: these are the ACTUAL audio generation modules
        ] + [f"layers.{i}.audio_mlp.gate_proj" for i in range(28)] + \
        [f"layers.{i}.audio_mlp.up_proj" for i in range(28)] + \
        [f"layers.{i}.audio_mlp.down_proj" for i in range(28)] + [
            
            # STRATEGY 2: Standard attention for reference conditioning (q_proj, k_proj, v_proj, o_proj only for efficiency)
            # These help with understanding reference audio context
        ] + [f"layers.{i}.self_attn.q_proj" for i in range(28)] + \
        [f"layers.{i}.self_attn.k_proj" for i in range(28)] + \
        [f"layers.{i}.self_attn.v_proj" for i in range(28)] + \
        [f"layers.{i}.self_attn.o_proj" for i in range(28)],
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
        
        def __getattr__(self, name):
            """Delegate all other attributes to the underlying model."""
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.model, name)
    
    # Wrap the model to handle argument mapping
    wrapped_model = HiggsAudioModelWrapper(model)
    model = get_peft_model(wrapped_model, lora_config)
    model.print_trainable_parameters()
    
    # Optional torch.compile for free speedups
    if args.compile_model:
        try:
            model = torch.compile(model, mode="max-autotune")
            logger.info("Torch.compile enabled (max-autotune)")
        except Exception as e:
            logger.warning(f"torch.compile could not be enabled: {e}")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Calculate training steps for proper warmup scheduling
    num_training_steps = len(train_dataloader) * args.num_epochs
    
    # CRITICAL FIX: Use warmup + cosine scheduler (Point B Fix #2)
    # Large models with PEFT benefit from warmup to avoid early instability
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
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
        
        # rolling means for clearer telemetry
        running_audio = running_text = running_total = 0.0
        running_n = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                # Move batch tensors to the correct device and dtype
                # Accelerate sometimes doesn't handle custom batch objects properly
                device = accelerator.device
                
                # Get model dtype for audio features (model uses mixed precision)
                model_dtype = next(model.parameters()).dtype
                
                # Helper function to move tensor to device and optionally convert dtype
                def to_device(tensor, convert_dtype=False):
                    if tensor is not None and hasattr(tensor, 'to'):
                        if convert_dtype and tensor.dtype in [torch.float32, torch.float64]:
                            # Convert float tensors to match model dtype (for audio features)
                            return tensor.to(device=device, dtype=model_dtype)
                        else:
                            return tensor.to(device)
                    return tensor
                
                # Forward pass - map collator output to model input correctly
                # The collator returns audio_in_wv but model expects audio_features
                # CRITICAL FIX: Clean separation of model inputs (NO LABELS to model)
                # This is the proper approach for zero-shot voice cloning training
                model_inputs = {
                    'input_ids': to_device(batch.input_ids),
                    'attention_mask': to_device(batch.attention_mask),
                    'audio_features': to_device(batch.audio_in_wv, convert_dtype=True) if hasattr(batch, 'audio_in_wv') else None,
                    'audio_feature_attention_mask': to_device(batch.audio_feature_attention_mask) if hasattr(batch, 'audio_feature_attention_mask') else None,
                    'audio_in_ids': to_device(batch.audio_in_ids) if hasattr(batch, 'audio_in_ids') else None,
                    'audio_in_ids_start': to_device(batch.audio_in_ids_start) if hasattr(batch, 'audio_in_ids_start') else None,
                    'audio_out_ids': to_device(batch.audio_out_ids) if hasattr(batch, 'audio_out_ids') else None,
                    'audio_out_ids_start': to_device(batch.audio_out_ids_start) if hasattr(batch, 'audio_out_ids_start') else None,
                    'audio_out_ids_start_group_loc': to_device(batch.audio_out_ids_start_group_loc) if hasattr(batch, 'audio_out_ids_start_group_loc') else None,
                }
                # Remove None values for clean forward pass
                model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
                
                # Get the underlying model (handle PEFT wrapping)
                if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
                    actual_model = model.base_model.model  # PEFT wrapped
                elif hasattr(model, 'module'):
                    actual_model = model.module  # Accelerate wrapped
                else:
                    actual_model = model
                
                # 🔍 CRITICAL DEBUGGING: Verify reference audio conditioning
                if step == 0 or step % 10 == 0:  # Log every 10 steps for ongoing monitoring
                    logger.info(f"\n🔍 === DEBUGGING STEP {step} ===")
                    logger.info(f"📥 MODEL INPUTS:")
                    for k, v in model_inputs.items():
                        if torch.is_tensor(v):
                            logger.info(f"  {k}: {v.shape} dtype={v.dtype}")
                        else:
                            logger.info(f"  {k}: {v}")
                    
                    # Critical: Verify audio conditioning inputs
                    if 'audio_in_ids' in model_inputs:
                        audio_in_ids = model_inputs['audio_in_ids']
                        logger.info(f"🎤 REFERENCE AUDIO CONDITIONING:")
                        logger.info(f"  audio_in_ids shape: {audio_in_ids.shape}")
                        logger.info(f"  audio_in_ids non-zero: {(audio_in_ids != 0).sum().item()}/{audio_in_ids.numel()}")
                        logger.info(f"  audio_in_ids sample: {audio_in_ids[0, :10] if audio_in_ids.numel() > 10 else audio_in_ids}")
                    
                    if 'audio_features' in model_inputs:
                        audio_features = model_inputs['audio_features']
                        logger.info(f"  audio_features shape: {audio_features.shape}")
                        logger.info(f"  audio_features mean: {audio_features.mean().item():.4f}")
                        logger.info(f"  audio_features std: {audio_features.std().item():.4f}")
                
                # Forward pass - call model directly WITHOUT labels
                outputs = actual_model(**model_inputs)
                
                # CRITICAL: Extract labels separately for loss computation
                text_labels = to_device(batch.label_ids) if hasattr(batch, 'label_ids') else None
                audio_labels = to_device(batch.label_audio_ids) if hasattr(batch, 'label_audio_ids') else None
                
                # 🚨 CRITICAL PAD TOKEN FIX: Map pad tokens to -100 if applicable
                if audio_labels is not None:
                    if step == 0 or step % 50 == 0:
                        # Deep investigation of audio_tokenizer attributes
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
                        
                        # Manual investigation of token 1025
                        token_1025_count = (audio_labels == 1025).sum().item()
                        total_labels = (audio_labels != -100).sum().item()
                        token_1025_ratio = token_1025_count / max(total_labels, 1) * 100
                        logger.info(f"🚨 TOKEN 1025 ANALYSIS: {token_1025_count}/{total_labels} ({token_1025_ratio:.1f}%) of non-ignore labels")
                        
                        # Check if 1025 appears at sequence ends (typical for pad tokens)
                        batch_size, seq_len = audio_labels.shape
                        end_positions = []
                        for b in range(min(3, batch_size)):  # Check first 3 samples
                            last_non_ignore = -1
                            for t in range(seq_len-1, -1, -1):
                                if audio_labels[b, t] != -100:
                                    last_non_ignore = t
                                    break
                            if last_non_ignore >= 0 and last_non_ignore < seq_len - 1:
                                # Check tokens after last non-ignore
                                trailing_tokens = audio_labels[b, last_non_ignore+1:last_non_ignore+6].tolist()
                                end_positions.append(f"sample_{b}_end: {trailing_tokens}")
                        logger.info(f"🔍 SEQUENCE END ANALYSIS: {end_positions}")
                    
                    # Apply pad token mapping if we find the right attribute
                    pad_id = None
                    for attr in ['pad_id', 'pad_token_id', 'padding_idx', 'pad_index']:
                        if hasattr(audio_tokenizer, attr):
                            pad_id = getattr(audio_tokenizer, attr)
                            break
                    
                    if pad_id is not None:
                        pad_count_before = (audio_labels == pad_id).sum().item()
                        if pad_count_before > 0:
                            logger.info(f"🔧 MAPPING PAD TOKENS: {pad_count_before} tokens ({pad_id}) → -100")
                            audio_labels[audio_labels == pad_id] = -100
                    else:
                        # CRITICAL: The collator ALREADY handles BOS/EOS tokens correctly!
                        # - BOS (1024) is added at start and masked to -100 in labels
                        # - EOS (1025) is added at end and preserved for learning
                        # DO NOT duplicate this logic here - it causes training/inference mismatch
                        
                        # Only check for truly invalid tokens (> 1025)
                        invalid_mask = audio_labels > 1025
                        invalid_mask = invalid_mask & (audio_labels != -100)  # Exclude already masked
                        invalid_count = invalid_mask.sum().item()
                        if invalid_count > 0:
                            invalid_tokens = audio_labels[invalid_mask].unique().tolist()
                            logger.warning(f"🚨 FOUND INVALID TOKENS: {invalid_count} tokens with IDs {invalid_tokens} - masking to -100")
                            audio_labels[invalid_mask] = -100
                        
                        # Log token distribution for debugging (but don't modify!)
                        token_1024_count = (audio_labels == 1024).sum().item()
                        token_1025_count = (audio_labels == 1025).sum().item()
                        masked_count = (audio_labels == -100).sum().item()
                        
                        if step == 0 or step % 10 == 0:
                            logger.info(f"📊 Token Distribution After Collator:")
                            logger.info(f"   • BOS (1024): {token_1024_count} (should be 0 - already masked by collator)")
                            logger.info(f"   • EOS (1025): {token_1025_count} (preserved for stopping logic)")
                            logger.info(f"   • Masked (-100): {masked_count} (includes BOS + any padding)")
                            logger.info(f"   • Valid audio tokens (0-1023): {((audio_labels >= 0) & (audio_labels <= 1023)).sum().item()}")
                
                # 🔍 DEBUGGING: Verify what model outputs
                if step == 0 or step % 10 == 0:
                    logger.info(f"📤 MODEL OUTPUTS:")
                    if hasattr(outputs, 'keys'):
                        logger.info(f"  Output keys: {list(outputs.keys())}")
                    else:
                        logger.info(f"  Output type: {type(outputs)}")
                
                # PROPER LOSS COMPUTATION FOR ZERO-SHOT VOICE CLONING
                total_loss = None  # use None sentinel; keep this a Tensor
                loss_components = {}
                
                # 1. Audio Loss (PRIMARY for voice cloning)
                if hasattr(outputs, 'audio_logits') and outputs.audio_logits is not None and audio_labels is not None:
                    audio_logits = outputs.audio_logits
                    
                    # 🔍 DEBUGGING: Verify audio loss computation
                    if step == 0 or step % 10 == 0:
                        logger.info(f"🔊 AUDIO LOSS COMPUTATION:")
                        logger.info(f"  audio_logits shape: {audio_logits.shape}")
                        logger.info(f"  audio_labels shape: {audio_labels.shape}")
                        logger.info(f"  audio_labels non-ignore: {(audio_labels != -100).sum().item()}/{audio_labels.numel()}")
                        logger.info(f"  audio_labels sample: {audio_labels[0, :10] if audio_labels.numel() > 10 else audio_labels[0]}")
                        
                        # CRITICAL: Verify loss tensor shapes before flattening
                        logits_flat = audio_logits.view(-1, audio_logits.size(-1))
                        labels_flat = audio_labels.view(-1)
                        logger.info(f"🚨 CRITICAL SHAPE CHECK:")
                        logger.info(f"  logits_flat: {logits_flat.shape} (should be [N, 1026])")
                        logger.info(f"  labels_flat: {labels_flat.shape} (should be [N])")
                        logger.info(f"  labels_flat min/max: {labels_flat[labels_flat != -100].min().item() if (labels_flat != -100).any() else 'N/A'} / {labels_flat[labels_flat != -100].max().item() if (labels_flat != -100).any() else 'N/A'}")
                        logger.info(f"  Expected range: 0-1025 (vocab size 1026)")
                    
                    # 🚨 CRITICAL FIX: Align tensor dimensions before loss computation
                    # Model outputs: [T, 8, V] (time-major)
                    # Labels:        [8, T]    (codebook-major)
                    # Without alignment, flattening happens in different orders → random CE!
                    
                    if audio_logits.dim() == 3 and audio_logits.shape[1] == 8:
                        # Permute to [8, T, V] to match label order (codebook-major)
                        audio_logits = audio_logits.permute(1, 0, 2).contiguous()
                        if step == 0 or step % 10 == 0:
                            logger.info(f"🔧 TENSOR ALIGNMENT: Permuted audio_logits from [T,8,V] to [8,T,V]: {audio_logits.shape}")
                    
                    # Compute audio token prediction loss (the CORE of voice cloning)
                    audio_loss_fct = torch.nn.CrossEntropyLoss(
                        ignore_index=-100,
                        label_smoothing=args.audio_label_smoothing
                    )
                    
                    # CRITICAL VERIFICATION: Check tensor alignment after fix
                    logits_for_loss = audio_logits.view(-1, audio_logits.size(-1))  # [(8*T), vocab]
                    labels_for_loss = audio_labels.contiguous().view(-1)           # [(8*T)]
                    
                    # Verify no invalid labels
                    valid_mask = labels_for_loss != -100
                    if valid_mask.any():
                        valid_labels = labels_for_loss[valid_mask]
                        if valid_labels.min() < 0 or valid_labels.max() >= 1026:
                            logger.error(f"🚨 INVALID AUDIO LABELS: min={valid_labels.min()}, max={valid_labels.max()} (expected 0-1025)")
                    
                    audio_loss = audio_loss_fct(logits_for_loss, labels_for_loss)
                    total_loss = audio_loss if total_loss is None else total_loss + audio_loss
                    loss_components['audio_loss'] = audio_loss.item()
                    
                    # 🔍 CRITICAL: Monitor audio loss trends + SANITY CHECKS FOR MODEL COLLAPSE
                    if step % 10 == 0:
                        logger.info(f"🔊 AUDIO LOSS (Step {step}): {audio_loss.item():.4f}")
                        
                        # 🚨 SANITY CHECK 1: Per-codebook CE breakdown
                        with torch.no_grad():
                            L = audio_logits  # Already permuted to [8, T, V]
                            y = audio_labels.contiguous()  # [8, T]
                            ce_per_q = []
                            for q in range(8):
                                mask_q = (y[q] != -100)
                                if mask_q.any():
                                    ce_q = torch.nn.functional.cross_entropy(L[q][mask_q], y[q][mask_q])
                                    ce_per_q.append(ce_q.item())
                            logger.info(f"📊 Per-codebook CE: {[f'{x:.3f}' for x in ce_per_q]}")
                            
                            # 🚨 SANITY CHECK 2: Prediction collapse detection
                            pred = L.argmax(-1)  # [8, T] 
                            valid_mask = (y != -100)
                            if valid_mask.any():
                                pred_tokens = pred[valid_mask]
                                label_tokens = y[valid_mask]
                                
                                # Count unique predictions vs unique labels
                                pred_unique = len(torch.unique(pred_tokens))
                                label_unique = len(torch.unique(label_tokens))
                                logger.info(f"🔍 Token diversity: pred={pred_unique}, labels={label_unique}")
                                
                                # Check for mode collapse (model predicting same few tokens)
                                pred_hist = torch.bincount(pred_tokens, minlength=1026)
                                top_5_pred = torch.topk(pred_hist, 5)
                                pred_concentration = top_5_pred.values.sum().item() / pred_tokens.numel()
                                logger.info(f"🚨 Top-5 prediction concentration: {pred_concentration:.3f}")
                                
                                if pred_concentration > 0.8:
                                    logger.warning(f"⚠️  HIGH PREDICTION CONCENTRATION: {pred_concentration:.3f} - POSSIBLE COLLAPSE!")
                                
                                # Log most frequent predictions vs labels
                                logger.info(f"🔍 Top pred tokens: {top_5_pred.indices[:5].tolist()}")
                                label_hist = torch.bincount(label_tokens, minlength=1026)
                                top_5_label = torch.topk(label_hist, 5)
                                logger.info(f"🔍 Top label tokens: {top_5_label.indices[:5].tolist()}")
                        
                        # 🚨 SANITY CHECK 3: Mask boundary verification
                        audio_out_mask = model_inputs.get('audio_out_mask')
                        if audio_out_mask is not None:
                            mask_sum = audio_out_mask.sum().item()
                            non_ignore = (audio_labels != -100).sum().item()
                            logger.info(f"🔍 Mask alignment: audio_out_mask_sum={mask_sum}, non_ignore_labels={non_ignore}")
                            
                            if abs(mask_sum - non_ignore) > 100:  # Allow some tolerance
                                logger.warning(f"⚠️  MASK MISMATCH: mask_sum({mask_sum}) != non_ignore({non_ignore})")
                        
                        # 🚨 SANITY CHECK 4: First token masking check
                        # First label in each codebook should be -100 (no training on t=0)
                        first_labels = audio_labels[:, 0]  # [8] - first token per codebook
                        first_ignore_count = (first_labels == -100).sum().item()
                        logger.info(f"🔍 First token masking: {first_ignore_count}/8 codebooks have -100 at t=0")
                        if first_ignore_count < 7:  # Allow some tolerance
                            logger.warning(f"⚠️  INSUFFICIENT FIRST TOKEN MASKING: only {first_ignore_count}/8 masked")
                        
                        # 🚨 SANITY CHECK 5: Check for suspicious loss ratios
                        if audio_loss.item() < 0.3:
                            logger.warning(f"🚨 EXTREMELY LOW AUDIO LOSS: {audio_loss.item():.4f} - INVESTIGATE!")
                        
                        if step > 50 and loss_components.get('text_loss', 10) < 0.1:
                            logger.warning(f"🚨 EXTREMELY LOW TEXT LOSS: {loss_components.get('text_loss', 0):.4f} - POSSIBLE COLLAPSE!")
                        
                        # 🚨 SANITY CHECK 6: Reference conditioning ablation test (every 200 steps)
                        if step > 0 and step % 200 == 0:
                            logger.info(f"🧪 Consider running reference ablation test at step {step} to verify conditioning dependency")
                
                # 2. Text Loss (SECONDARY - for text understanding)
                if hasattr(outputs, 'logits') and outputs.logits is not None and text_labels is not None:
                    text_logits = outputs.logits
                    
                    # Only use text loss if we have reasonable dimensions
                    min_seq_len = min(text_logits.size(1), text_labels.size(1))
                    if min_seq_len > 1:  # Need at least 2 tokens for shifting
                        # Trim to matching sequence length 
                        text_logits = text_logits[:, :min_seq_len, :]
                        text_labels = text_labels[:, :min_seq_len]
                        
                        # Shift for next-token prediction
                        shift_logits = text_logits[..., :-1, :].contiguous()
                        shift_labels = text_labels[..., 1:].contiguous()
                        
                        # 🔍 DEBUGGING: Verify text loss computation
                        if step == 0 or step % 10 == 0:
                            logger.info(f"📝 TEXT LOSS COMPUTATION:")
                            logger.info(f"  text_logits original: {text_logits.shape}")
                            logger.info(f"  text_labels original: {text_labels.shape}")
                            logger.info(f"  after shift - logits: {shift_logits.shape}, labels: {shift_labels.shape}")
                            logger.info(f"  text_labels non-ignore: {(shift_labels != -100).sum().item()}/{shift_labels.numel()}")
                        
                        # Compute text loss (weighted lower for voice cloning)
                        text_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                        text_loss = text_loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )
                        
                        # Weight text loss lower (audio is primary for voice cloning)
                        weighted_text_loss = 0.1 * text_loss  
                        total_loss = weighted_text_loss if total_loss is None else total_loss + weighted_text_loss
                        loss_components['text_loss'] = text_loss.item()
                        loss_components['weighted_text_loss'] = weighted_text_loss.item()
                        
                        # 🔍 CRITICAL: Monitor text loss trends  
                        if step % 10 == 0:
                            logger.info(f"📝 TEXT LOSS (Step {step}): {text_loss.item():.4f} (weighted: {weighted_text_loss.item():.4f})")
                
                # Final loss for backward pass
                if total_loss is None:
                    logger.warning("No valid loss this batch; skipping")
                    continue
                    
                loss = total_loss
                loss_components['total_loss'] = float(loss.detach().item())
                
                # Rolling means
                running_audio += loss_components.get('audio_loss', 0.0)
                running_text  += loss_components.get('weighted_text_loss', 0.0)
                running_total += loss_components.get('total_loss', 0.0)
                running_n     += 1
                if (step % args.log_steps) == 0 and running_n > 0:
                    logger.info(f"[rolling/{args.log_steps}] audio_ce={running_audio/running_n:.4f} "
                                f"text_w={running_text/running_n:.4f} total={running_total/running_n:.4f}")
                    running_audio = running_text = running_total = 0.0
                    running_n = 0
                
                # 🔍 CRITICAL: Always log loss breakdown every 10 steps
                if step % 10 == 0:
                    logger.info(f"🎯 TOTAL LOSS (Step {step}): {total_loss.item():.4f}")
                    logger.info(f"📊 Loss breakdown: {loss_components}")
                    
                    # CRITICAL DIAGNOSTIC: Check if learning rate is active
                    for i, pg in enumerate(optimizer.param_groups):
                        logger.info(f"📈 LR[{i}]={pg['lr']:.6e} (step {step})")
                    
                    # CRITICAL: Check if random baseline comparison
                    audio_loss_val = loss_components.get('audio_loss', 0)
                    random_baseline = 6.9334  # ln(1026) for 1026-class codebook
                    if audio_loss_val > random_baseline - 0.05:
                        logger.warning(f"⚠️  AUDIO LOSS ({audio_loss_val:.4f}) NEAR/ABOVE RANDOM BASELINE ({random_baseline:.4f}) - NOT LEARNING YET!")
                    else:
                        logger.info(f"✅ AUDIO LOSS ({audio_loss_val:.4f}) BELOW RANDOM BASELINE ({random_baseline:.4f}) - LEARNING ACTIVE!")
                    
                    # 🚨 CRITICAL: Check for suspicious loss patterns
                    if loss_components.get('audio_loss', 0) < 2.0:
                        logger.warning(f"⚠️  SUSPICIOUS: Audio loss very low ({loss_components.get('audio_loss', 0):.4f}) - possible model collapse or wrong labels!")
                    
                    if step > 0:
                        logger.info(f"✅ Zero-shot voice cloning training - Reference audio conditioning ACTIVE")
                    logger.info(f"🔄 === END DEBUG STEP {step} ===\n")
                
                # Backward pass
                accelerator.backward(loss)
                
                # CRITICAL DIAGNOSTIC: Check LoRA gradient norms every 50 steps
                if step % 50 == 0:
                    total_lora_grad_norm = 0.0
                    lora_param_count = 0
                    for n, p in model.named_parameters():
                        if p.requires_grad and hasattr(p, 'grad') and p.grad is not None:
                            grad_norm = p.grad.data.float().norm().item()
                            # Focus on LoRA targets
                            if any(k in n for k in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'audio_lm_head', 'audio_mlp']):
                                logger.info(f"🔍 Grad ||{n}|| = {grad_norm:.4e}")
                                total_lora_grad_norm += grad_norm
                                lora_param_count += 1
                    
                    if lora_param_count > 0:
                        avg_lora_grad_norm = total_lora_grad_norm / lora_param_count
                        logger.info(f"📊 Average LoRA grad norm: {avg_lora_grad_norm:.4e} ({lora_param_count} params)")
                        if avg_lora_grad_norm < 1e-6:
                            logger.warning(f"⚠️  VERY LOW LORA GRAD NORMS - POSSIBLE GRADIENT FLOW ISSUE!")
                    else:
                        logger.warning(f"⚠️  NO LORA GRADIENTS FOUND - LoRA NOT ACTIVE!")
                
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
                    # Move batch tensors to the correct device and dtype
                    device = accelerator.device
                    
                    # Get model dtype for audio features (model uses mixed precision)
                    model_dtype = next(model.parameters()).dtype
                    
                    # Helper function to move tensor to device and optionally convert dtype
                    def to_device(tensor, convert_dtype=False):
                        if tensor is not None and hasattr(tensor, 'to'):
                            if convert_dtype and tensor.dtype in [torch.float32, torch.float64]:
                                # Convert float tensors to match model dtype (for audio features)
                                return tensor.to(device=device, dtype=model_dtype)
                            else:
                                return tensor.to(device)
                        return tensor
                    
                    # CRITICAL FIX: Same proper approach for validation
                    model_inputs = {
                        'input_ids': to_device(batch.input_ids),
                        'attention_mask': to_device(batch.attention_mask),
                        'audio_features': to_device(batch.audio_in_wv, convert_dtype=True) if hasattr(batch, 'audio_in_wv') else None,
                        'audio_feature_attention_mask': to_device(batch.audio_feature_attention_mask) if hasattr(batch, 'audio_feature_attention_mask') else None,
                        'audio_in_ids': to_device(batch.audio_in_ids) if hasattr(batch, 'audio_in_ids') else None,
                        'audio_in_ids_start': to_device(batch.audio_in_ids_start) if hasattr(batch, 'audio_in_ids_start') else None,
                        'audio_out_ids': to_device(batch.audio_out_ids) if hasattr(batch, 'audio_out_ids') else None,
                        'audio_out_ids_start': to_device(batch.audio_out_ids_start) if hasattr(batch, 'audio_out_ids_start') else None,
                        'audio_out_ids_start_group_loc': to_device(batch.audio_out_ids_start_group_loc) if hasattr(batch, 'audio_out_ids_start_group_loc') else None,
                    }
                    model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
                    
                    # Get underlying model
                    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
                        actual_model = model.base_model.model
                    elif hasattr(model, 'module'):
                        actual_model = model.module
                    else:
                        actual_model = model
                    
                    # Forward pass without labels
                    outputs = actual_model(**model_inputs)
                    
                    # Extract labels for validation loss
                    text_labels = to_device(batch.label_ids) if hasattr(batch, 'label_ids') else None
                    audio_labels = to_device(batch.label_audio_ids) if hasattr(batch, 'label_audio_ids') else None
                    
                    # PROPER VALIDATION LOSS for zero-shot voice cloning
                    batch_loss = 0.0
                    
                    # Primary: Audio Loss - WITH TENSOR ALIGNMENT FIX
                    if hasattr(outputs, 'audio_logits') and outputs.audio_logits is not None and audio_labels is not None:
                        audio_logits = outputs.audio_logits
                        
                        # 🚨 CRITICAL FIX: Same tensor alignment as training loop
                        if audio_logits.dim() == 3 and audio_logits.shape[1] == 8:
                            # Permute to [8, T, V] to match label order (codebook-major)
                            audio_logits = audio_logits.permute(1, 0, 2).contiguous()
                        
                        audio_loss_fct = torch.nn.CrossEntropyLoss(
                            ignore_index=-100,
                            label_smoothing=args.audio_label_smoothing
                        )
                        audio_loss = audio_loss_fct(
                            audio_logits.view(-1, audio_logits.size(-1)),   # [(8*T), vocab]
                            audio_labels.contiguous().view(-1)               # [(8*T)]
                        )
                        batch_loss += audio_loss.item()
                    
                    # Secondary: Text Loss (weighted)
                    if hasattr(outputs, 'logits') and outputs.logits is not None and text_labels is not None:
                        text_logits = outputs.logits
                        min_seq_len = min(text_logits.size(1), text_labels.size(1))
                        if min_seq_len > 1:
                            text_logits = text_logits[:, :min_seq_len, :]
                            text_labels = text_labels[:, :min_seq_len]
                            
                            shift_logits = text_logits[..., :-1, :].contiguous()
                            shift_labels = text_labels[..., 1:].contiguous()
                            
                            text_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                            text_loss = text_loss_fct(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1)
                            )
                            
                            # Weight text loss lower for voice cloning
                            batch_loss += 0.1 * text_loss.item()
                    
                    # Add to validation totals
                    if batch_loss > 0:
                        val_loss += batch_loss
                        val_steps += 1
                    else:
                        continue  # Skip if no valid loss
            
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
