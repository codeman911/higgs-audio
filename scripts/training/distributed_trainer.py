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
        
        # Create ChatMLDatasetSample - SIMPLE and WORKING format
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
    parser.add_argument('--lora_r', type=int, default=32, help='LoRA rank (increased for Arabic learning)')
    parser.add_argument('--lora_alpha', type=int, default=64, help='LoRA alpha (increased for Arabic learning)')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--text_loss_weight', type=float, default=1.0, help='Text loss weight (critical for language learning)')
    
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
    parser.add_argument("--log_steps", type=int, default=100,
                        help="Log every N steps")
    parser.add_argument("--val_steps", type=int, default=1000,
                        help="Run validation every N steps")
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
        round_to=8,  # Documentation recommends round_to=8 for optimal batching
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
        ] + [f"layers.{i}.audio_mlp.gate_proj" for i in range(28)] + \
        [f"layers.{i}.audio_mlp.up_proj" for i in range(28)] + \
        [f"layers.{i}.audio_mlp.down_proj" for i in range(28)] + [
            
            # STRATEGY 2: Standard attention for reference conditioning
        ] + [f"layers.{i}.self_attn.q_proj" for i in range(28)] + \
        [f"layers.{i}.self_attn.k_proj" for i in range(28)] + \
        [f"layers.{i}.self_attn.v_proj" for i in range(28)] + \
        [f"layers.{i}.self_attn.o_proj" for i in range(28)] + [
            
            # CRITICAL FIX: TEXT BACKBONE ADAPTATION for Arabic phonetics
            # Target top LLaMA layers (20-27) for language learning
        ] + [f"layers.{i}.mlp.gate_proj" for i in range(20, 28)] + \
        [f"layers.{i}.mlp.up_proj" for i in range(20, 28)] + \
        [f"layers.{i}.mlp.down_proj" for i in range(20, 28)],
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
                # Forward pass - map collator output to model input correctly
                # CRITICAL: Target audio tokens are needed for STRUCTURE but must NOT leak into embeddings
                model_inputs = {
                    'input_ids': to_device(batch.input_ids),
                    'attention_mask': to_device(batch.attention_mask),
                    'audio_features': to_device(batch.audio_in_wv, convert_dtype=True) if hasattr(batch, 'audio_in_wv') else None,
                    'audio_feature_attention_mask': to_device(batch.audio_feature_attention_mask) if hasattr(batch, 'audio_feature_attention_mask') else None,
                    'audio_in_ids': to_device(batch.audio_in_ids) if hasattr(batch, 'audio_in_ids') else None,
                    'audio_in_ids_start': to_device(batch.audio_in_ids_start) if hasattr(batch, 'audio_in_ids_start') else None,
                    # ✅ RESTORED: These are needed for audio structure - leakage is in embedding, not here!
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
                

                
                # 🚨 CRITICAL DEBUG: Log model inputs to diagnose T=0 audio logits issue
                if global_step % 10 == 0:
                    logger.info(f"🔍 MODEL INPUT DEBUG (Step {global_step}):")
                    for key, value in model_inputs.items():
                        if value is not None and hasattr(value, 'shape'):
                            logger.info(f"  {key}: {value.shape}")
                        elif value is not None:
                            logger.info(f"  {key}: {type(value)} (non-tensor)")
                        else:
                            logger.info(f"  {key}: None")
                    
                    # CRITICAL: Check if audio structure fields are present and non-empty
                    if 'audio_out_ids' in model_inputs and model_inputs['audio_out_ids'] is not None:
                        audio_out_shape = model_inputs['audio_out_ids'].shape
                        logger.info(f"🎯 AUDIO_OUT_IDS SHAPE: {audio_out_shape} (should NOT be [*, 0])")
                        if len(audio_out_shape) > 1 and audio_out_shape[1] == 0:
                            logger.error(f"🚨 CRITICAL: audio_out_ids has 0 tokens! This causes T=0 audio logits!")
                    else:
                        logger.error(f"🚨 CRITICAL: audio_out_ids is None! Model cannot generate audio logits!")
                
                # 🔍 ARCHITECTURAL VERIFICATION: Quick ablation to confirm NO target leakage (as per USER guidance)
                if global_step % 50 == 0:  # Run less frequently - architecture is confirmed correct
                    logger.info(f"🔍 ARCHITECTURAL VERIFICATION (Step {global_step}):")
                    
                    with torch.no_grad():
                        # Ablation 1: Remove audio_out_* fields and verify model still produces logits
                        inputs_no_targets = {k: v for k, v in model_inputs.items() if not k.startswith('audio_out')}
                        try:
                            outputs_no_tf = actual_model(**inputs_no_targets)
                            if hasattr(outputs_no_tf, 'audio_logits') and outputs_no_tf.audio_logits is not None:
                                logger.info(f"  ✅ WITHOUT audio_out_*: Model still produces audio_logits {outputs_no_tf.audio_logits.shape}")
                            else:
                                logger.info(f"  ⚠️  WITHOUT audio_out_*: No audio_logits (expected for some architectures)")
                        except Exception as e:
                            logger.info(f"  📝 WITHOUT audio_out_*: Model forward failed (expected): {str(e)[:100]}")
                    
                    # Note: expanded_input_ids check moved to after forward pass
                    logger.info(f"  📝 expanded_input_ids check will be performed after forward pass")
                    
                    # First token masking verification
                    if audio_labels is not None and audio_labels.dim() >= 2 and audio_labels.shape[0] >= 8:
                        first_tokens_masked = (audio_labels[:8, 0] == -100).sum().item() if audio_labels.shape[1] > 0 else 0
                        logger.info(f"  🔒 First token masking: {first_tokens_masked}/8 codebooks (-100)")
                        if first_tokens_masked == 8:
                            logger.info(f"  ✅ INVARIANT: All BOS tokens properly masked")
                
                # 🚨 CRITICAL: Focus on REAL issues - Arabic text supervision debugging
                if global_step % 10 == 0 and text_labels is not None:
                    logger.info(f"🔍 ARABIC TEXT SUPERVISION DEBUG (Step {global_step}):")
                    
                    # Check text supervision quantity
                    total_text_labels = text_labels.numel()
                    text_non_ignore = (text_labels != -100).sum().item()
                    text_supervision_ratio = text_non_ignore / total_text_labels if total_text_labels > 0 else 0.0
                    
                    logger.info(f"  📊 Text labels: {total_text_labels} total, {text_non_ignore} non-ignore ({text_supervision_ratio:.1%})")
                    
                    # CRITICAL: Check if text supervision is too low
                    if text_non_ignore < 50:
                        logger.error(f"🚨 INSUFFICIENT TEXT SUPERVISION: Only {text_non_ignore} tokens - need ~100+ for Arabic learning!")
                    elif text_non_ignore >= 100:
                        logger.info(f"✅ ADEQUATE TEXT SUPERVISION: {text_non_ignore} tokens for Arabic learning")
                    else:
                        logger.warning(f"⚠️  MARGINAL TEXT SUPERVISION: {text_non_ignore} tokens - may need more for robust Arabic learning")
                    
                    # Sample text tokens to verify Arabic tokenization
                    if text_non_ignore > 0:
                        non_ignore_mask = text_labels != -100
                        sample_tokens = text_labels[non_ignore_mask][:10].tolist()
                        logger.info(f"  🔤 Sample text tokens: {sample_tokens} (check Arabic tokenization quality)")

                # Forward pass - call model directly WITHOUT labels
                outputs = actual_model(**model_inputs)
                
                # CRITICAL: Extract labels separately for loss computation
                text_labels = to_device(batch.label_ids) if hasattr(batch, 'label_ids') else None
                audio_labels = to_device(batch.label_audio_ids) if hasattr(batch, 'label_audio_ids') else None
                
                # 🚨 CRITICAL PAD TOKEN FIX: Map pad tokens to -100 if applicable
                if audio_labels is not None:

                    
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
                        # Safety net: Mask BOS tokens to -100 and invalid tokens
                        bos_count_before = (audio_labels == 1024).sum().item()
                        if bos_count_before > 0:
                            logger.info(f"🔧 MAPPING BOS TOKENS: {bos_count_before} tokens (1024) → -100")
                            audio_labels[audio_labels == 1024] = -100
                        
                        # Mask truly invalid tokens (> 1025)
                        invalid_mask = (audio_labels > 1025) & (audio_labels != -100)
                        invalid_count = invalid_mask.sum().item()
                        if invalid_count > 0:
                            logger.info(f"🔧 MAPPING INVALID TOKENS: {invalid_count} tokens → -100")
                            audio_labels[invalid_mask] = -100
                

                
                # PROPER LOSS COMPUTATION FOR ZERO-SHOT VOICE CLONING
                total_loss = None  # use None sentinel; keep this a Tensor

                loss_components = {}
                
                # 1. Audio Loss (PRIMARY for voice cloning)
                if hasattr(outputs, 'audio_logits') and outputs.audio_logits is not None and audio_labels is not None:
                    audio_logits = outputs.audio_logits
                    
                    # 🔍 DEBUGGING: Verify audio loss computation

                    
                    # 🚨 CRITICAL FIX: Align tensor dimensions before loss computation
                    # Model outputs: [T, 8, V] (time-major)
                    # Labels:        [8, T]    (codebook-major)
                    # Without alignment, flattening happens in different orders → random CE!
                    
                    if audio_logits.dim() == 3 and audio_logits.shape[1] == 8:
                        # Permute to [8, T, V] to match label order (codebook-major)
                        audio_logits = audio_logits.permute(1, 0, 2).contiguous()
                        if global_step % 100 == 0:
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
                    if global_step % args.log_steps == 0:
                        logger.info(f"🔊 AUDIO LOSS (Step {global_step}): {audio_loss.item():.4f}")
                        
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
                            
                            # 🚨 SANITY CHECK 2: Audio token comparison (first/last 10)
                            pred = L.argmax(-1)  # [8, T] 
                            valid_mask = (y != -100)
                            
                            # Show first and last 10 predicted vs actual tokens for codebook 0
                            if valid_mask[0].any():
                                valid_positions = torch.where(valid_mask[0])[0]
                                if len(valid_positions) >= 10:
                                    first_10_idx = valid_positions[:10]
                                    last_10_idx = valid_positions[-10:]
                                    
                                    first_pred = pred[0][first_10_idx].cpu().tolist()
                                    first_true = y[0][first_10_idx].cpu().tolist()
                                    last_pred = pred[0][last_10_idx].cpu().tolist()
                                    last_true = y[0][last_10_idx].cpu().tolist()
                                    
                                    logger.info(f"🎯 First 10 tokens: pred={first_pred} | true={first_true}")
                                    logger.info(f"🎯 Last 10 tokens:  pred={last_pred} | true={last_true}")
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
                        
                        # 🚨 CRITICAL: Monitor for TARGET AUDIO LEAKAGE FIX SUCCESS
                        audio_ce = audio_loss.item()
                        if audio_ce < 1.0:
                            logger.error(f"🚨 CRITICAL: AUDIO CE TOO LOW ({audio_ce:.4f}) - POSSIBLE TARGET AUDIO LEAKAGE STILL PRESENT!")
                            logger.error(f"🚨 Expected: CE should be 2-6 for healthy learning, not <1 (copying behavior)")
                        elif 1.0 <= audio_ce < 2.0:
                            logger.warning(f"⚠️  AUDIO CE BORDERLINE ({audio_ce:.4f}) - Monitor for leakage")
                        elif 2.0 <= audio_ce <= 6.0:
                            logger.info(f"✅ HEALTHY AUDIO CE ({audio_ce:.4f}) - Target audio leakage fix WORKING!")
                        elif audio_ce > 6.0:
                            logger.warning(f"🚨 HIGH AUDIO CE ({audio_ce:.4f}) - Model struggling to learn audio")
                        

                
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
                        
                        # 🔍 CRITICAL: Debug text loss computation (like audio debugging)
                        if global_step % args.log_steps == 0:
                            with torch.no_grad():
                                # Text token analysis (similar to audio)
                                text_pred = shift_logits.argmax(-1)  # [B, T-1]
                                text_true = shift_labels  # [B, T-1]
                                text_valid_mask = (text_true != -100)
                                
                                if text_valid_mask.any():
                                    # Flatten for analysis
                                    valid_pred = text_pred[text_valid_mask]
                                    valid_true = text_true[text_valid_mask]
                                    
                                    # Show first 10 predicted vs actual text tokens
                                    if len(valid_pred) >= 10:
                                        first_10_pred = valid_pred[:10].cpu().tolist()
                                        first_10_true = valid_true[:10].cpu().tolist()
                                        logger.info(f"📝 TEXT First 10 tokens: pred={first_10_pred} | true={first_10_true}")
                                        
                                        # Show last 10 as well
                                        last_10_pred = valid_pred[-10:].cpu().tolist()
                                        last_10_true = valid_true[-10:].cpu().tolist()
                                        logger.info(f"📝 TEXT Last 10 tokens:  pred={last_10_pred} | true={last_10_true}")
                                    
                                    # Token diversity check
                                    pred_unique = len(torch.unique(valid_pred))
                                    true_unique = len(torch.unique(valid_true))
                                    logger.info(f"📝 TEXT Token diversity: pred={pred_unique}, labels={true_unique}")
                                    
                                    # Accuracy calculation
                                    correct = (valid_pred == valid_true).sum().item()
                                    total = len(valid_pred)
                                    accuracy = correct / total if total > 0 else 0.0
                                    logger.info(f"📝 TEXT Accuracy: {correct}/{total} = {accuracy:.4f} ({accuracy*100:.1f}%)")
                                    
                                    # Show actual text tokens if possible (decode a few)
                                    try:
                                        # Try to decode first few tokens to see actual text
                                        sample_pred_text = tokenizer.decode(first_10_pred[:5], skip_special_tokens=True)
                                        sample_true_text = tokenizer.decode(first_10_true[:5], skip_special_tokens=True)
                                        logger.info(f"📝 TEXT Sample: pred='{sample_pred_text}' | true='{sample_true_text}'")
                                    except Exception as e:
                                        logger.info(f"📝 TEXT decode error: {e}")
                                else:
                                    logger.warning(f"⚠️  NO VALID TEXT LABELS FOUND - This explains zero text loss!")
                        
                        # Compute text loss (weighted lower for voice cloning)
                        text_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                        text_loss = text_loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )
                        
                        # CRITICAL FIX: Increase text loss weight for Arabic language learning
                        # Text loss is ESSENTIAL for pronunciation and phonetics
                        weighted_text_loss = args.text_loss_weight * text_loss  
                        total_loss = weighted_text_loss if total_loss is None else total_loss + weighted_text_loss
                        loss_components['text_loss'] = text_loss.item()
                        loss_components['weighted_text_loss'] = weighted_text_loss.item()
                        
                        # 🔍 CRITICAL: Monitor text loss trends for Arabic learning
                        if global_step % args.log_steps == 0:
                            logger.info(f"📝 TEXT LOSS (Step {global_step}): {text_loss.item():.4f} (weighted: {weighted_text_loss.item():.4f}, weight: {args.text_loss_weight})")
                            
                            # CRITICAL: Check for text learning problems
                            if text_loss.item() < 0.001:
                                logger.error(f"🚨 CRITICAL: TEXT LOSS TOO LOW ({text_loss.item():.6f}) - Model NOT learning text! Check labels!")
                            elif text_loss.item() > 4.0:
                                logger.warning(f"🚨 HIGH TEXT LOSS: {text_loss.item():.4f} - Arabic pronunciation may be poor!")
                            elif text_loss.item() < 1.5 and text_loss.item() > 0.1:
                                logger.info(f"✅ GOOD TEXT LOSS: {text_loss.item():.4f} - Language learning active!")
                    else:
                        if global_step % args.log_steps == 0:
                            logger.warning(f"⚠️  NO TEXT LOSS: min_seq_len={min_seq_len} - Check text sequence lengths!")
                else:
                    if global_step % args.log_steps == 0:
                        logger.warning(f"⚠️  NO TEXT LOGITS/LABELS: logits={hasattr(outputs, 'logits')}, labels={text_labels is not None}")
                
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
                
                # Validation accuracy every 1000 steps
                if global_step % args.val_steps == 0 and global_step > 0 and val_dataloader:
                    model.eval()
                    val_audio_correct = 0
                    val_audio_total = 0
                    val_loss_accum = 0
                    val_steps_count = 0
                    
                    with torch.no_grad():
                        # Sample a few batches for validation accuracy
                        for i, batch in enumerate(val_dataloader):
                            if i >= 5:  # Only validate on 5 batches for speed
                                break
                                
                            device = accelerator.device
                            model_dtype = next(model.parameters()).dtype
                            
                            def to_device(tensor, convert_dtype=False):
                                if tensor is not None and hasattr(tensor, 'to'):
                                    if convert_dtype and tensor.dtype in [torch.float32, torch.float64]:
                                        return tensor.to(device=device, dtype=model_dtype)
                                    else:
                                        return tensor.to(device)
                                return tensor
                            
                            # Validation forward pass - restore audio structure fields 
                            model_inputs = {
                                'input_ids': to_device(batch.input_ids),
                                'attention_mask': to_device(batch.attention_mask),
                                'audio_features': to_device(batch.audio_in_wv, convert_dtype=True) if hasattr(batch, 'audio_in_wv') else None,
                                'audio_feature_attention_mask': to_device(batch.audio_feature_attention_mask) if hasattr(batch, 'audio_feature_attention_mask') else None,
                                'audio_in_ids': to_device(batch.audio_in_ids) if hasattr(batch, 'audio_in_ids') else None,
                                'audio_in_ids_start': to_device(batch.audio_in_ids_start) if hasattr(batch, 'audio_in_ids_start') else None,
                                # ✅ RESTORED: These are needed for audio structure in validation too
                                'audio_out_ids': to_device(batch.audio_out_ids) if hasattr(batch, 'audio_out_ids') else None,
                                'audio_out_ids_start': to_device(batch.audio_out_ids_start) if hasattr(batch, 'audio_out_ids_start') else None,  
                                'audio_out_ids_start_group_loc': to_device(batch.audio_out_ids_start_group_loc) if hasattr(batch, 'audio_out_ids_start_group_loc') else None,
                            }
                            model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
                            
                            if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
                                actual_model = model.base_model.model
                            elif hasattr(model, 'module'):
                                actual_model = model.module
                            else:
                                actual_model = model
                            
                            outputs = actual_model(**model_inputs)
                            audio_labels = to_device(batch.label_audio_ids) if hasattr(batch, 'label_audio_ids') else None
                            
                            if hasattr(outputs, 'audio_logits') and outputs.audio_logits is not None and audio_labels is not None:
                                audio_logits = outputs.audio_logits
                                
                                # Apply same tensor alignment as training
                                if audio_logits.dim() == 3 and audio_logits.shape[1] == 8:
                                    audio_logits = audio_logits.permute(1, 0, 2).contiguous()
                                
                                # Calculate accuracy
                                audio_preds = torch.argmax(audio_logits, dim=-1)
                                valid_mask = (audio_labels != -100)
                                
                                if valid_mask.sum() > 0:
                                    correct = (audio_preds[valid_mask] == audio_labels[valid_mask]).sum().item()
                                    total = valid_mask.sum().item()
                                    val_audio_correct += correct
                                    val_audio_total += total
                                    
                                    # Calculate loss for this batch
                                    audio_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                                    audio_loss = audio_loss_fct(
                                        audio_logits.view(-1, audio_logits.size(-1)),
                                        audio_labels.contiguous().view(-1)
                                    )
                                    val_loss_accum += audio_loss.item()
                                    val_steps_count += 1
                                    
                                    # Log first/last tokens for first batch only
                                    if i == 0:
                                        flat_preds = audio_preds.view(-1)[valid_mask.view(-1)]
                                        flat_labels = audio_labels.view(-1)[valid_mask.view(-1)]
                                        
                                        if len(flat_preds) >= 10:
                                            first_10_pred = flat_preds[:10].tolist()
                                            first_10_true = flat_labels[:10].tolist()
                                            last_10_pred = flat_preds[-10:].tolist()
                                            last_10_true = flat_labels[-10:].tolist()
                                            
                                            logger.info(f"🎯 VAL First 10: pred={first_10_pred} | true={first_10_true}")
                                            logger.info(f"🎯 VAL Last 10:  pred={last_10_pred} | true={last_10_true}")
                    
                    # Log validation results
                    if val_audio_total > 0:
                        val_accuracy = val_audio_correct / val_audio_total
                        avg_val_loss = val_loss_accum / val_steps_count if val_steps_count > 0 else 0
                        logger.info(f"📊 VALIDATION (Step {global_step}): Loss={avg_val_loss:.4f}, Audio Accuracy={val_accuracy:.4f} ({val_audio_correct}/{val_audio_total})")
                    
                    model.train()  # Return to training mode
        
                # Training logs every 100 steps  
                if global_step % args.log_steps == 0 and running_n > 0:
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
                    if loss_components.get('audio_loss', 0) < 1.5:
                        logger.warning(f"⚠️  SUSPICIOUS: Audio loss very low ({loss_components.get('audio_loss', 0):.4f}) - possible model collapse or wrong labels!")
                    
                    if step > 0:
                        logger.info(f"✅ Zero-shot voice cloning training - Reference audio conditioning ACTIVE")
                    logger.info(f"🔄 === END DEBUG STEP {step} ===\n")
                
                # Backward pass
                accelerator.backward(loss)
                
                # Simplified LoRA health check
                if global_step % 100 == 0:
                    total_lora_grad_norm = 0.0
                    lora_param_count = 0
                    for n, p in model.named_parameters():
                        if p.requires_grad and hasattr(p, 'grad') and p.grad is not None:
                            if 'lora' in n.lower():
                                total_lora_grad_norm += p.grad.data.float().norm().item()
                                lora_param_count += 1
                    
                    if lora_param_count > 0:
                        avg_lora_grad_norm = total_lora_grad_norm / lora_param_count
                        logger.info(f"📊 LoRA grad health: {avg_lora_grad_norm:.2e} avg ({lora_param_count} params)")
                    else:
                        logger.warning(f"⚠️  NO LORA GRADIENTS FOUND")
                
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
                    
                    # 🚨 CRITICAL FIX: Remove target audio leakage in validation (final instance)
                    model_inputs = {
                        'input_ids': to_device(batch.input_ids),
                        'attention_mask': to_device(batch.attention_mask),
                        'audio_features': to_device(batch.audio_in_wv, convert_dtype=True) if hasattr(batch, 'audio_in_wv') else None,
                        'audio_feature_attention_mask': to_device(batch.audio_feature_attention_mask) if hasattr(batch, 'audio_feature_attention_mask') else None,
                        'audio_in_ids': to_device(batch.audio_in_ids) if hasattr(batch, 'audio_in_ids') else None,
                        'audio_in_ids_start': to_device(batch.audio_in_ids_start) if hasattr(batch, 'audio_in_ids_start') else None,
                        # 🚨 CRITICAL FIX: REMOVED final target audio leakage:
                        # 'audio_out_ids': REMOVED - target audio codes leak into model input!
                        # 'audio_out_ids_start': REMOVED - enables target audio access!  
                        # 'audio_out_ids_start_group_loc': REMOVED - target audio metadata!
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
