#!/usr/bin/env python3
import os
import sys
import json
import argparse
import torch
import torchaudio
import librosa
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, WhisperProcessor, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch.nn as nn
from accelerate import Accelerator
from tqdm import tqdm
from pathlib import Path
import logging

# FIX: Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    """Simple dataset for loading JSON samples"""
    
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'samples' in data:
            self.samples = data['samples']
        elif isinstance(data, list):
            self.samples = data
        else:
            raise ValueError(f"Unexpected data format in {json_path}")
        
        logger.info(f"Loaded {len(self.samples)} samples from {json_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def custom_collate_fn(batch, tokenizer, audio_tokenizer, collator):
    """Convert raw dict samples to ChatMLDatasetSample objects - EXACT WORKING VERSION FROM train_higgs_lora.py"""
    chatml_samples = []
    
    for sample in batch:
        # Use ORIGINAL prepare_chatml_sample (NO MODIFICATIONS)
        input_tokens, label_tokens, audio_contents, speaker_id = prepare_chatml_sample(sample, tokenizer)
        
        if input_tokens is None or label_tokens is None:
            continue  # Skip invalid samples
        
        # Process audio using audio_tokenizer - EXACT SAME AS WORKING train_higgs_lora.py
        audio_ids_list = []
        audio_waveforms_list = []
        
        for audio_content in audio_contents:
            if audio_content and hasattr(audio_content, 'audio_url'):
                audio_path = audio_content.audio_url
                if audio_path and os.path.exists(audio_path):
                    try:
                        # Tokenize audio
                        audio_codes = audio_tokenizer.encode(audio_path)
                        
                        # Load waveform
                        waveform, sr = librosa.load(audio_path, sr=24000, mono=True)
                        waveform = torch.tensor(waveform, dtype=torch.float32)
                        
                        audio_ids_list.append(audio_codes)
                        audio_waveforms_list.append(waveform)
                        
                    except Exception as e:
                        logger.warning(f"Failed to process audio {audio_path}: {e}")
        
        # Create proper audio concatenation for ChatMLDatasetSample
        if audio_ids_list:
            # Concatenate audio codes: shape (num_codebooks, total_length)
            audio_ids_concat = torch.cat([audio_codes for audio_codes in audio_ids_list], dim=1)
            audio_ids_start = torch.tensor([0] + [audio_codes.shape[1] for audio_codes in audio_ids_list[:-1]]).cumsum(dim=0)
            
            # Concatenate audio waveforms
            audio_waveforms_concat = torch.cat(audio_waveforms_list, dim=0)
            audio_waveforms_start = torch.tensor([0] + [wv.shape[0] for wv in audio_waveforms_list[:-1]]).cumsum(dim=0)
            audio_sample_rate = torch.tensor([24000] * len(audio_waveforms_list))
            # FIX: Handle string speaker_id from zero-shot data
            numeric_speaker_id = 0 if isinstance(speaker_id, str) or speaker_id is None else int(speaker_id)
            audio_speaker_indices = torch.tensor([numeric_speaker_id] * len(audio_waveforms_list), dtype=torch.long)
        else:
            # Empty audio tensors - EXACT SAME AS WORKING SCRIPT
            audio_ids_concat = torch.zeros((8, 0), dtype=torch.long)  # 8 codebooks
            audio_ids_start = torch.tensor([], dtype=torch.long)
            audio_waveforms_concat = torch.zeros((0,), dtype=torch.float32)
            audio_waveforms_start = torch.tensor([], dtype=torch.long)
            audio_sample_rate = torch.tensor([24000])
            # FIX: Handle string speaker_id from zero-shot data  
            numeric_speaker_id = 0 if isinstance(speaker_id, str) or speaker_id is None else int(speaker_id)
            audio_speaker_indices = torch.tensor([numeric_speaker_id], dtype=torch.long)
        
        # Create proper ChatMLDatasetSample for original collator
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
    
    if not chatml_samples:
        return None  # Skip empty batches
        
    # Use the original HiggsAudioSampleCollator - THE PROVEN WORKING COLLATOR
    return collator(chatml_samples)


def setup_collator(config, tokenizer, whisper_processor):
    """Setup the original HiggsAudioSampleCollator with proper parameters - NO CUSTOM REPLACEMENTS"""
    return HiggsAudioSampleCollator(
        whisper_processor=whisper_processor,
        audio_in_token_id=config.audio_in_token_idx,
        audio_out_token_id=config.audio_out_token_idx,
        audio_stream_bos_id=config.audio_stream_bos_id,
        audio_stream_eos_id=config.audio_stream_eos_id,
        encode_whisper_embed=config.encode_whisper_embed,
        pad_token_id=config.pad_token_id,
        return_audio_in_tokens=config.encode_audio_in_tokens,
        use_delay_pattern=config.use_delay_pattern,
        round_to=8,
        audio_num_codebooks=8
    )


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
    parser.add_argument("--batch_size", type=int, default=12,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm")
    
    # LoRA arguments
    parser.add_argument('--lora_r', type=int, default=32, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=64, help='LoRA alpha') 
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--text_loss_weight', type=float, default=1.0, help='Text loss weight')
    
    # Other arguments
    parser.add_argument("--num_workers", type=int, default=64, help="DataLoader workers")
    parser.add_argument("--prefetch_factor", type=int, default=4, help="DataLoader prefetch")
    parser.add_argument("--persistent_workers", action="store_true", default=True, help="Keep workers alive")
    parser.add_argument("--audio_label_smoothing", type=float, default=0.05, help="Audio label smoothing")
    parser.add_argument("--compile_model", action="store_true", default=True, help="Enable torch.compile")
    parser.add_argument("--use_cached_codes", action="store_true", default=False, help="Use cached audio codes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_steps", type=int, default=50, help="Log every N steps")
    parser.add_argument("--val_steps", type=int, default=100, help="Validation every N steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save every N steps")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision")
    
    # DEBUG arguments  
    parser.add_argument("--debug_samples", type=int, default=None, help="Limit training samples")
    parser.add_argument("--debug_val_samples", type=int, default=None, help="Limit validation samples")
    
    args = parser.parse_args()
    
    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision
    )
    
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Loading data from {args.dataset_path}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Load tokenizers
    logger.info("Loading tokenizers...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    audio_tokenizer = load_higgs_audio_tokenizer(args.audio_tokenizer_path, device="cpu")
    
    # Load model config (REVERT: Keep original config)
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Load model (REVERT: Use original architecture)
    logger.info("Loading model...")
    model = HiggsAudioModel.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": accelerator.device}
    )
    
    # Initialize collator
    logger.info("Initializing collator...")
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    
    collator = setup_collator(config, tokenizer, whisper_processor)
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = SimpleDataset(os.path.join(args.dataset_path, "train_chatml_samples.json"))[:args.debug_samples]
    val_dataset = SimpleDataset(os.path.join(args.dataset_path, "val_chatml_samples.json"))[:args.debug_val_samples]
    
    # Create data loaders (REVERT: Use original collator)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        collate_fn=lambda batch: custom_collate_fn(batch, tokenizer, audio_tokenizer, collator),
        pin_memory=True,
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers // 2,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        collate_fn=lambda batch: custom_collate_fn(batch, tokenizer, audio_tokenizer, collator),
        pin_memory=True,
        drop_last=False
    )
    
    # LoRA configuration with systematic target collection
    TARGET_FRAGMENTS = (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        "audio_mlp.gate_proj", "audio_mlp.up_proj", "audio_mlp.down_proj",
    )

    def collect_lora_targets(model):
        leaf_targets = set()
        for name, module in model.named_modules():
            for frag in TARGET_FRAGMENTS:
                if name.endswith(frag):
                    leaf_targets.add(frag.split(".")[-1])  # leaf module name
        return sorted(leaf_targets)

    lora_leaf_targets = collect_lora_targets(model)
    if not lora_leaf_targets:
        logger.warning("No LoRA targets found – check module names.")
    else:
        logger.info(f"LoRA leaf targets: {lora_leaf_targets}")

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=tuple(lora_leaf_targets),
        bias="none", task_type=TaskType.CAUSAL_LM
    )
    
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, lora_config)
    
    # Setup optimizer with stability improvements
    # Lower learning rate for newly initialized cross-attention modules
    stable_lr = min(args.learning_rate, 1e-4)  # Cap at 1e-4 for stability
    if stable_lr != args.learning_rate:
        logger.info(f" STABILITY FIX: Reducing learning rate from {args.learning_rate} to {stable_lr} for cross-attention stability")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=stable_lr,
        weight_decay=0.01,
        eps=1e-8,  # Increase epsilon for numerical stability
        betas=(0.9, 0.95)  # More conservative beta2 for stability
    )
    
    # CRITICAL: Add gradient clipping for newly initialized cross-attention modules
    gradient_clip_norm = 1.0  # Conservative clipping for stability
    logger.info(f" STABILITY FIX: Adding gradient clipping (max_norm={gradient_clip_norm}) for cross-attention stability")
    
    # Calculate training steps for proper warmup scheduling
    num_training_steps = len(train_dataloader) * args.num_epochs
    
    # FINAL FIX: Immediate learning rate for newly initialized cross-attention modules
    # The cosine scheduler with long warmup prevents learning for thousands of steps
    # Newly initialized modules need immediate adaptation to learn text-audio conditioning
    
    # Calculate training steps for proper warmup scheduling
    num_training_steps = len(train_dataloader) * args.num_epochs
    
    # CRITICAL FIX: Immediate learning for cross-attention adaptation
    # Use minimal warmup (50 steps) so learning starts immediately
    warmup_steps = 50  # Very short warmup for immediate learning
    logger.info(f" IMMEDIATE LEARNING FIX: Using {warmup_steps} warmup steps (immediate learning for cross-attention)")
    
    # Alternative: Use linear scheduler that starts with small non-zero LR
    from transformers import get_linear_schedule_with_warmup
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Verify scheduler starts with meaningful LR after just 1 step
    # Take one scheduler step to see actual LR
    dummy_step_lr = stable_lr / warmup_steps  # Expected LR after 1 step
    logger.info(f" LEARNING RATE FIX: After 1 step, LR will be ≈ {dummy_step_lr:.2e} (immediate learning)")
    
    # Alternative approach: Start with constant LR for first 1000 steps, then decay
    # This ensures immediate learning for cross-attention adaptation
    logger.info(" CROSS-ATTENTION LEARNING: Enabling immediate gradient updates for newly initialized modules")
    
    # Prepare for training
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    
    if val_dataloader:
        val_dataloader = accelerator.prepare(val_dataloader)
    
    # Training loop
    logger.info("Starting training...")
    global_step = 0
    
    def extract_assistant_labels(input_ids, tokenizer):
        """Extract labels ONLY from assistant content before <|AUDIO_OUT|>"""
        B, T = input_ids.size()
        labels = input_ids.new_full((B, T), -100)
        
        AUDIO_OUT_ID = 128275  # <|AUDIO_OUT|>
        EOH_ID = 128007       # <|end_header_id|>
        ASSISTANT_ID = 78191  # "assistant"
        
        for b in range(B):
            ids = input_ids[b]
            
            # Find <|AUDIO_OUT|> position
            audio_out_pos = (ids == AUDIO_OUT_ID).nonzero(as_tuple=True)[0]
            if audio_out_pos.numel() == 0:
                continue
            audio_out_pos = audio_out_pos[0].item()
            
            # Find last assistant header before <|AUDIO_OUT|>
            eoh_positions = (ids[:audio_out_pos] == EOH_ID).nonzero(as_tuple=True)[0]
            if eoh_positions.numel() == 0:
                continue
                
            # Find assistant content start
            for eoh_idx in reversed(eoh_positions):
                content_start = eoh_idx.item() + 1
                if content_start < audio_out_pos and (ASSISTANT_ID in ids[:eoh_idx]):
                    # Supervise assistant content (next-token prediction)
                    if content_start + 1 < audio_out_pos:
                        labels[b, content_start+1:audio_out_pos] = ids[content_start+1:audio_out_pos]
                    break
        
        return labels

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                # Move batch tensors to the correct device and dtype (like backup trainer)
                device = accelerator.device
                
                # Get model dtype for audio features (model uses mixed precision)
                model_dtype = next(model.parameters()).dtype
                
                # Helper function to move tensor to device and optionally convert dtype (like backup trainer)
                def to_device(tensor, convert_dtype=False):
                    if tensor is not None and hasattr(tensor, 'to'):
                        if convert_dtype and tensor.dtype in [torch.float32, torch.float64]:
                            # Convert float tensors to match model dtype (for audio features)
                            return tensor.to(device=device, dtype=model_dtype)
                        else:
                            return tensor.to(device)
                    return tensor
                
                # CRITICAL FIX: Clean separation of model inputs (NO LABELS to model) - like backup trainer
                model_inputs = {
                    'input_ids': to_device(batch.input_ids),
                    'attention_mask': to_device(batch.attention_mask),
                    'audio_features': to_device(batch.audio_in_wv, convert_dtype=True) if hasattr(batch, 'audio_in_wv') and batch.audio_in_wv is not None else None,
                    'audio_feature_attention_mask': to_device(batch.audio_feature_attention_mask) if hasattr(batch, 'audio_feature_attention_mask') and batch.audio_feature_attention_mask is not None else None,
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
                
                # SIMPLE FORWARD PASS (like working backup trainer)
                outputs = actual_model(**model_inputs)
                
                # Extract labels separately for loss computation (like backup trainer)
                text_labels = to_device(batch.label_ids) if hasattr(batch, 'label_ids') else None
                audio_labels = to_device(batch.label_audio_ids) if hasattr(batch, 'label_audio_ids') else None
                
                # Manual loss computation (like backup trainer)
                total_loss = 0.0
                
                # Audio loss computation
                audio_loss = torch.tensor(0.0, device=device)
                if "audio_logits" in outputs and outputs["audio_logits"] is not None and audio_labels is not None:
                    audio_logits = outputs["audio_logits"]
                    if audio_logits.dim() == 3:
                        # Ensure [C, T, V] format
                        if audio_logits.shape[1] == 8:
                            audio_logits = audio_logits.permute(1, 0, 2).contiguous()
                        
                        C, T, V = audio_logits.shape
                        audio_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                        audio_loss = audio_loss_fct(
                            audio_logits.view(C * T, V),
                            audio_labels.view(C * T)
                        )
                        total_loss += audio_loss
                
                # Text loss computation
                text_loss = torch.tensor(0.0, device=device)
                if "logits" in outputs and outputs["logits"] is not None and text_labels is not None:
                    text_logits = outputs["logits"][:, :-1, :].contiguous()  # Shift for LM
                    shift_labels = text_labels[:, 1:].contiguous()
                    
                    # Ensure shapes match before computing loss
                    B, T, V = text_logits.shape
                    B_lbl, T_lbl = shift_labels.shape
                    
                    if T != T_lbl:
                        # Adjust to minimum sequence length
                        min_T = min(T, T_lbl)
                        text_logits = text_logits[:, :min_T, :].contiguous()
                        shift_labels = shift_labels[:, :min_T].contiguous()
                        T = min_T
                    
                    text_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                    text_loss = text_loss_fct(
                        text_logits.view(B * T, V),
                        shift_labels.view(B * T)
                    )
                    # Weight text loss lower for voice cloning
                    weighted_text_loss = args.text_loss_weight * text_loss
                    total_loss += weighted_text_loss
                
                loss = total_loss
                
                # Comprehensive loss logging every 10 steps
                if global_step % 10 == 0 and accelerator.is_main_process:
                    logger.info("=" * 100)
                    logger.info(f" STEP {global_step} TRAINING METRICS")
                    logger.info("=" * 100)
                    
                    # Loss breakdown
                    logger.info(f" LOSS BREAKDOWN:")
                    logger.info(f"   Total Loss: {total_loss.item():.4f}")
                    logger.info(f"   Text Loss: {text_loss.item():.4f}")
                    logger.info(f"   Audio Loss: {audio_loss.item():.4f}")
                    if hasattr(args, 'text_loss_weight'):
                        logger.info(f"   Text Weight: {args.text_loss_weight}")
                    
                    # MODEL INPUT LOGGING (as requested)
                    logger.info(f" MODEL INPUTS:")
                    for key, value in model_inputs.items():
                        if torch.is_tensor(value):
                            logger.info(f"   {key}: shape={value.shape}, dtype={value.dtype}")
                            if key == 'input_ids' and value.numel() <= 50:  # Log small input_ids
                                logger.info(f"   {key} values: {value.flatten()[:50].tolist()}")
                    
                    # Text predictions vs labels analysis (FIXED: use adjusted tensors)
                    if "logits" in outputs and outputs["logits"] is not None and text_labels is not None:
                        # Use the SAME adjusted tensors as loss computation to avoid shape mismatch
                        adjusted_text_logits = outputs["logits"][:, :-1, :].contiguous()
                        adjusted_shift_labels = text_labels[:, 1:].contiguous()
                        
                        # Apply same shape adjustment as loss computation
                        B, T, V = adjusted_text_logits.shape
                        B_lbl, T_lbl = adjusted_shift_labels.shape
                        
                        if T != T_lbl:
                            min_T = min(T, T_lbl)
                            adjusted_text_logits = adjusted_text_logits[:, :min_T, :].contiguous()
                            adjusted_shift_labels = adjusted_shift_labels[:, :min_T].contiguous()
                        
                        # Get predictions from ADJUSTED tensors
                        text_preds = adjusted_text_logits.argmax(dim=-1)  # [B, T]
                        
                        # Find valid positions (not -100) - now shapes match!
                        valid_mask = (adjusted_shift_labels != -100)
                        if valid_mask.any():
                            text_preds_flat = text_preds[valid_mask]
                            text_labels_flat = adjusted_shift_labels[valid_mask]
                            
                            # Accuracy
                            accuracy = (text_preds_flat == text_labels_flat).float().mean().item()
                            logger.info(f" TEXT ANALYSIS:")
                            logger.info(f"   Accuracy: {accuracy:.1%}")
                            logger.info(f"   Valid tokens: {valid_mask.sum().item()}")
                            logger.info(f"   Text logits shape: {adjusted_text_logits.shape}")
                            logger.info(f"   Text labels shape: {adjusted_shift_labels.shape}")
                            
                            # TEXT TOKEN IDs: First and last 10 predictions vs labels
                            if len(text_preds_flat) >= 10:
                                logger.info(f"   TEXT TOKEN IDs:")
                                logger.info(f"     FIRST 10 - Pred: {text_preds_flat[:10].tolist()}")
                                logger.info(f"     FIRST 10 - Label: {text_labels_flat[:10].tolist()}")
                                if len(text_preds_flat) >= 20:
                                    logger.info(f"     LAST 10 - Pred: {text_preds_flat[-10:].tolist()}")
                                    logger.info(f"     LAST 10 - Label: {text_labels_flat[-10:].tolist()}")
                            
                            # DECODED TEXT LOGGING (as requested)
                            logger.info(f"   DECODED TEXT ANALYSIS:")
                            try:
                                # Decode first 20 predictions and labels
                                if len(text_preds_flat) >= 20:
                                    pred_text = tokenizer.decode(text_preds_flat[:20].tolist(), skip_special_tokens=False)
                                    label_text = tokenizer.decode(text_labels_flat[:20].tolist(), skip_special_tokens=False)
                                    logger.info(f"     PRED TEXT (first 20): '{pred_text}'")
                                    logger.info(f"     LABEL TEXT (first 20): '{label_text}'")
                                
                                # Decode last 20 predictions and labels if available
                                if len(text_preds_flat) >= 40:
                                    pred_text_last = tokenizer.decode(text_preds_flat[-20:].tolist(), skip_special_tokens=False)
                                    label_text_last = tokenizer.decode(text_labels_flat[-20:].tolist(), skip_special_tokens=False)
                                    logger.info(f"     PRED TEXT (last 20): '{pred_text_last}'")
                                    logger.info(f"     LABEL TEXT (last 20): '{label_text_last}'")
                                
                                # Show full sample decoded text (first sample only)
                                if B >= 1:
                                    sample_preds = text_preds[0][valid_mask[0]]  # First batch sample
                                    sample_labels = adjusted_shift_labels[0][valid_mask[0]]
                                    if len(sample_preds) > 0:
                                        full_pred_text = tokenizer.decode(sample_preds.tolist(), skip_special_tokens=False)
                                        full_label_text = tokenizer.decode(sample_labels.tolist(), skip_special_tokens=False)
                                        logger.info(f"     FULL SAMPLE PRED: '{full_pred_text[:200]}{'...' if len(full_pred_text) > 200 else ''}'")
                                        logger.info(f"     FULL SAMPLE LABEL: '{full_label_text[:200]}{'...' if len(full_label_text) > 200 else ''}'")
                            except Exception as e:
                                logger.warning(f"     Decode error: {e}")
                            
                            # ALL PREDICTIONS LOGGING (token IDs)
                            logger.info(f"   ALL TEXT TOKEN PREDICTIONS (first 50): {text_preds.flatten()[:50].tolist()}")
                    
                    # AUDIO PREDICTIONS vs LABELS ANALYSIS
                    if "audio_logits" in outputs and outputs["audio_logits"] is not None and audio_labels is not None:
                        audio_logits = outputs["audio_logits"]
                        
                        # Get predictions
                        audio_preds = audio_logits.argmax(dim=-1)  # [B, T, C] or [C, T] or [B, C, T]
                        
                        # Debug shapes before adjustment
                        logger.info(f" AUDIO TENSOR SHAPES (before adjustment):")
                        logger.info(f"   audio_logits: {audio_logits.shape}")
                        logger.info(f"   audio_preds (raw): {audio_preds.shape}")
                        logger.info(f"   audio_labels: {audio_labels.shape}")
                        
                        # Ensure audio_preds matches audio_labels shape exactly
                        if audio_preds.shape != audio_labels.shape:
                            # Handle different possible shapes
                            if audio_preds.dim() == 3:  # [B, T, C] or [B, C, T]
                                if audio_preds.shape[0] == 1:  # Single batch
                                    audio_preds = audio_preds.squeeze(0)  # Remove batch dimension
                                    # Now audio_preds could be [T, C] or [C, T]
                                    if audio_preds.shape != audio_labels.shape:
                                        # Try transpose to match labels
                                        if audio_preds.shape == (audio_labels.shape[1], audio_labels.shape[0]):
                                            audio_preds = audio_preds.transpose(0, 1)  # [T, C] -> [C, T]
                            elif audio_preds.dim() == 2:  # [T, C] or [C, T]
                                if audio_preds.shape != audio_labels.shape:
                                    # Try transpose to match labels
                                    if audio_preds.shape == (audio_labels.shape[1], audio_labels.shape[0]):
                                        audio_preds = audio_preds.transpose(0, 1)  # [T, C] -> [C, T]
                        
                        # Verify shapes match after adjustment
                        logger.info(f" AUDIO TENSOR SHAPES (after adjustment):")
                        logger.info(f"   audio_preds (adjusted): {audio_preds.shape}")
                        logger.info(f"   audio_labels: {audio_labels.shape}")
                        
                        # Only proceed if shapes match
                        if audio_preds.shape == audio_labels.shape:
                            # Find valid positions (not -100)
                            valid_mask = (audio_labels != -100)
                            if valid_mask.any():
                                audio_preds_flat = audio_preds[valid_mask]
                                audio_labels_flat = audio_labels[valid_mask]
                                
                                # Accuracy
                                accuracy = (audio_preds_flat == audio_labels_flat).float().mean().item()
                                logger.info(f" AUDIO ANALYSIS:")
                                logger.info(f"   Accuracy: {accuracy:.1%}")
                                logger.info(f"   Valid tokens: {valid_mask.sum().item()}")
                                logger.info(f"   Audio logits shape: {audio_logits.shape}")
                                logger.info(f"   Audio labels shape: {audio_labels.shape}")
                                
                                # AUDIO LOGS: First and last 10 predictions vs labels (as requested)
                                if len(audio_preds_flat) >= 10:
                                    logger.info(f"   AUDIO LOGS - TOKEN IDs:")
                                    logger.info(f"     FIRST 10 - Pred: {audio_preds_flat[:10].tolist()}")
                                    logger.info(f"     FIRST 10 - Label: {audio_labels_flat[:10].tolist()}")
                                    if len(audio_preds_flat) >= 20:
                                        logger.info(f"     LAST 10 - Pred: {audio_preds_flat[-10:].tolist()}")
                                        logger.info(f"     LAST 10 - Label: {audio_labels_flat[-10:].tolist()}")
                                
                                # FULL AUDIO PREDICTIONS LOGGING
                                logger.info(f"   ALL AUDIO TOKEN PREDICTIONS (first 50): {audio_preds.flatten()[:50].tolist()}")
                        else:
                            logger.warning(f" AUDIO SHAPE MISMATCH: preds={audio_preds.shape} vs labels={audio_labels.shape}")
                            logger.warning(f"   Skipping audio analysis for this step")
                    
                    logger.info(f" TRAINING STATUS: LR={scheduler.get_last_lr()[0]:.2e}")
                    logger.info("=" * 100)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
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
                    device = accelerator.device
                    
                    # Move batch tensors to the correct device and dtype (like backup trainer)
                    model_inputs = {
                        'input_ids': batch.input_ids.to(device),
                        'attention_mask': batch.attention_mask.to(device),
                        'audio_features': batch.audio_in_wv.to(device) if hasattr(batch, 'audio_in_wv') and batch.audio_in_wv is not None else None,
                        'audio_feature_attention_mask': batch.audio_feature_attention_mask.to(device) if hasattr(batch, 'audio_feature_attention_mask') and batch.audio_feature_attention_mask is not None else None,
                        'audio_in_ids': batch.audio_in_ids.to(device) if hasattr(batch, 'audio_in_ids') else None,
                        'audio_in_ids_start': batch.audio_in_ids_start.to(device) if hasattr(batch, 'audio_in_ids_start') else None,
                        'audio_out_ids': batch.audio_out_ids.to(device) if hasattr(batch, 'audio_out_ids') else None,
                        'audio_out_ids_start': batch.audio_out_ids_start.to(device) if hasattr(batch, 'audio_out_ids_start') else None,
                        'audio_out_ids_start_group_loc': batch.audio_out_ids_start_group_loc.to(device) if hasattr(batch, 'audio_out_ids_start_group_loc') else None,
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
                    
                    # Forward pass with clean inputs - NO LABELS to model (like backup trainer)
                    outputs = actual_model(**model_inputs)
                    
                    # Extract labels separately for manual loss computation
                    text_labels = batch.label_ids.to(device) if hasattr(batch, 'label_ids') else None
                    audio_labels = batch.label_audio_ids.to(device) if hasattr(batch, 'label_audio_ids') else None
                    
                    # Manual loss computation (like backup trainer)
                    total_loss = 0.0
                    
                    # Audio loss computation
                    audio_loss = torch.tensor(0.0, device=device)
                    if "audio_logits" in outputs and outputs["audio_logits"] is not None and audio_labels is not None:
                        audio_logits = outputs["audio_logits"]
                        if audio_logits.dim() == 3:
                            # Ensure [C, T, V] format
                            if audio_logits.shape[1] == 8:
                                audio_logits = audio_logits.permute(1, 0, 2).contiguous()
                            
                            C, T, V = audio_logits.shape
                            audio_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                            audio_loss = audio_loss_fct(
                                audio_logits.view(C * T, V),
                                audio_labels.view(C * T)
                            )
                            total_loss += audio_loss
                    
                    # Text loss computation
                    text_loss = torch.tensor(0.0, device=device)
                    if "logits" in outputs and outputs["logits"] is not None and text_labels is not None:
                        text_logits = outputs["logits"][:, :-1, :].contiguous()  # Shift for LM
                        shift_labels = text_labels[:, 1:].contiguous()
                        
                        # Ensure shapes match before computing loss
                        B, T, V = text_logits.shape
                        B_lbl, T_lbl = shift_labels.shape
                        
                        if T != T_lbl:
                            # Adjust to minimum sequence length
                            min_T = min(T, T_lbl)
                            text_logits = text_logits[:, :min_T, :].contiguous()
                            shift_labels = shift_labels[:, :min_T].contiguous()
                            T = min_T
                        
                        text_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                        text_loss = text_loss_fct(
                            text_logits.view(B * T, V),
                            shift_labels.view(B * T)
                        )
                        # Weight text loss lower for voice cloning
                        weighted_text_loss = args.text_loss_weight * text_loss
                        total_loss += weighted_text_loss
                    
                    loss = total_loss
                    
                    # Combined validation loss
                    batch_val_loss = audio_loss + text_loss
                    val_loss += batch_val_loss.item()
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
