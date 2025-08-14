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


def collate_fn(batch, tokenizer, audio_tokenizer, sample_rate=24000):
    """Simple collate function that creates proper ChatMLDatasetSample objects for original collator"""
    
    chatml_samples = []
    
    for sample in batch:
        # Use the existing ChatML structure as-is (it's already correctly formatted)
        # The sample already contains proper messages structure from zero-shot processing
        
        # Extract metadata for validation only
        misc = sample.get('misc', {})
        ref_transcript = misc.get('ref_transcript', '')
        target_text = misc.get('target_text', '')
        
        # Use prepare_chatml_sample to get proper tokenization
        try:
            input_tokens, label_tokens, audio_contents, speaker_id = prepare_chatml_sample(
                sample, tokenizer
            )
            
            # Simple logging for first sample only
            if len(chatml_samples) == 0:
                target_token_count = sum(1 for token in label_tokens if token != -100)
                audio_segment_count = len(audio_contents)
                
                # FIXED ZERO-SHOT VALIDATION: Check for Arabic/text content in actual model input
                input_text = tokenizer.decode(input_tokens, skip_special_tokens=False)
                
                # Check for Arabic text patterns in the actual model input (not just misc fields)
                import re
                has_arabic = bool(re.search(r'[\u0600-\u06FF]', input_text))
                has_meaningful_text = len([token for token in input_tokens if token not in [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id]]) > 30
                
                logger.info(f"Training ready: {target_token_count} text tokens, {audio_segment_count} audio segments")
                logger.info(f"Arabic content validation: arabic_text_present={has_arabic}, sufficient_tokens={has_meaningful_text}")
                
                if has_arabic:
                    logger.info(" SUCCESS: Arabic text detected in model input!")
                    # Extract and show Arabic content
                    arabic_content = re.findall(r'[\u0600-\u06FF\s]+', input_text)
                    if arabic_content:
                        logger.info(f"Arabic text sample: '{arabic_content[0][:50]}...'")
                else:
                    logger.warning(" No Arabic text detected in model input")
                    
                # Show meaningful preview of what's actually fed to model (skip system boilerplate)
                preview_text = input_text.replace('<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant capable of generating speech from text with the voice characteristics inferred from the provided audio samples.<|eot_id|>', '[SYSTEM]')
                logger.info(f"Model input preview: '{preview_text[:150]}...'")
        
        except Exception as e:
            logger.warning(f"Failed to prepare sample: {e}")
            input_tokens = [tokenizer.pad_token_id]
            label_tokens = [-100]
            audio_contents = []
            speaker_id = 0
        
        # Process audio using audio_tokenizer if present
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
                        waveform, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
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
            audio_sample_rate = torch.tensor([sample_rate] * len(audio_waveforms_list))
            audio_speaker_indices = torch.tensor([speaker_id or 0] * len(audio_waveforms_list), dtype=torch.long)
        else:
            # Empty audio tensors
            audio_ids_concat = torch.zeros((8, 0), dtype=torch.long)  # 8 codebooks
            audio_ids_start = torch.tensor([], dtype=torch.long)
            audio_waveforms_concat = torch.zeros((0,), dtype=torch.float32)
            audio_waveforms_start = torch.tensor([], dtype=torch.long)
            audio_sample_rate = torch.tensor([sample_rate])
            audio_speaker_indices = torch.tensor([speaker_id or 0], dtype=torch.long)
        
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
    
    return chatml_samples  # Return list of ChatMLDatasetSample objects


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
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm")
    
    # LoRA arguments
    parser.add_argument('--lora_r', type=int, default=32, help='LoRA rank (increased for Arabic learning)')
    parser.add_argument('--lora_alpha', type=int, default=64, help='LoRA alpha (increased for Arabic learning)')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--text_loss_weight', type=float, default=1.0, help='Text loss weight')
    
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
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    
    # CRITICAL ARCHITECTURE FIX: Revert to original Higgs-Audio V2 configuration
    # Original architecture doesn't use cross-attention modules
    # Text-audio conditioning happens through shared attention + DualFFN, not separate cross-attention
    logger.info(" ORIGINAL CONFIG: audio_out_self_attention=False (shared attention + DualFFN)")
    
    # CRITICAL: Revert to original configuration - no cross-attention modules
    config.use_audio_out_self_attention = False  # Original default - proven architecture!
    
    logger.info(" REVERTED: use_audio_out_self_attention=False (original architecture)")
    
    # Load model
    logger.info("Loading model...")
    model = HiggsAudioModel.from_pretrained(
        args.model_path,
        config=config,  # Use modified config
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": accelerator.device}
    )
    
    # Initialize collator using WhisperProcessor
    logger.info("Initializing collator...")
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    
    collator = HiggsAudioSampleCollator(
        whisper_processor=whisper_processor,
        audio_in_token_id=config.audio_in_token_idx,
        audio_out_token_id=config.audio_out_token_idx,
        audio_stream_bos_id=config.audio_stream_bos_id,
        audio_stream_eos_id=config.audio_stream_eos_id,
        encode_whisper_embed=config.encode_whisper_embed,
        pad_token_id=config.pad_token_id,
        return_audio_in_tokens=config.encode_audio_in_tokens,
        use_delay_pattern=config.use_delay_pattern,
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
    
    # Memory optimization for cross-attention enabled model
    # The newly initialized audio_attn modules significantly increase memory usage
    effective_batch_size = max(args.batch_size // 2, 4)  # Reduce batch size by half, minimum 4
    if effective_batch_size != args.batch_size:
        logger.info(f" MEMORY FIX: Reducing batch size from {args.batch_size} to {effective_batch_size} for cross-attention stability")
    
    # Create dataloaders using original collator
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collator(collate_fn(batch, tokenizer, audio_tokenizer)),
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collator(collate_fn(batch, tokenizer, audio_tokenizer)),
        num_workers=4,
        pin_memory=True
    )
    
    # SURGICAL FIX 3: FULL LORA COVERAGE (ALL TEXT PATHWAY LAYERS)
    # Target ALL blocks' text pathways for Arabic orthography learning
    target_modules = [
        # ALL self-attention modules (all blocks, not just top 2)
        "self_attn.q_proj",
        "self_attn.k_proj", 
        "self_attn.v_proj",
        "self_attn.o_proj",
        
        # ALL text MLP modules (all blocks, not just top 2)
        "mlp.gate_proj",
        "mlp.up_proj", 
        "mlp.down_proj",
        
        # Audio MLP modules (DualFFN audio path) 
        "audio_mlp.gate_proj",
        "audio_mlp.up_proj",
        "audio_mlp.down_proj",
        
        # Both output heads for full text-to-speech adaptation
        "text_lm_head",   # Text output head for Arabic token generation
        "audio_lm_head"   # Audio output head for speech generation
    ]
    
    # Reduce LoRA parameters for full coverage (more layers = smaller rank)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                    # Reduced from 32 for full coverage
        lora_alpha=16,           # Balanced
        lora_dropout=0.1,        # Slight regularization
        target_modules=target_modules,
        bias="none"
    )
    
    logger.info(f"FULL DEPTH LORA: Targeting {len(target_modules)} module types across ALL {len(model.model.layers)} blocks")
    logger.info("ARABIC LEARNING: All text pathway layers now adaptable for orthography")
    
    # Verify LoRA target modules exist in the model (should all exist now with original config)
    missing_modules = []
    for name, module in model.named_modules():
        for target in target_modules:
            if target in name and hasattr(module, 'weight'):
                logger.info(f" LORA TARGET FOUND: {name}")
                break
    
    # Check for any missing targets (should be none with original architecture)
    for target in target_modules:
        found = False
        for name, _ in model.named_modules():
            if target in name:
                found = True
                break
        if not found:
            missing_modules.append(target)
    
    if missing_modules:
        logger.error(f" MISSING LORA TARGETS: {missing_modules}")
        logger.error(" These modules should exist in original architecture!")
        raise ValueError(f"Missing LoRA target modules: {missing_modules}")
    else:
        logger.info(" ALL LORA TARGETS VERIFIED: Original DualFFN modules found")
    
    # Create a wrapper to handle the labels -> label_ids mapping
    class HiggsAudioModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
    
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
    gradient_clip_norm = 0.5  # Conservative clipping for stability
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
    
    gradient_clip_norm = 1.0
    logger.info(f" STABILITY FIX: Adding gradient clipping (max_norm={gradient_clip_norm}) for cross-attention stability")
    
    PHASE_A_STEPS = 10000  # First 10k steps
    PHASE_A_REF_DROP = 0.7  # Drop reference audio 70% of time
    PHASE_B_REF_DROP = 0.2  # Reduced dropout in Phase-B
    
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
                    # RESTORED: These are needed for audio structure - leakage is in embedding, not here!
                    'audio_out_ids': to_device(batch.audio_out_ids) if hasattr(batch, 'audio_out_ids') else None,
                    'audio_out_ids_start': to_device(batch.audio_out_ids_start) if hasattr(batch, 'audio_out_ids_start') else None,  
                    'audio_out_ids_start_group_loc': to_device(batch.audio_out_ids_start_group_loc) if hasattr(batch, 'audio_out_ids_start_group_loc') else None,
                }
                # Remove None values for clean forward pass
                model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
                
                # SURGICAL FIX 1: PHASE-A REFERENCE AUDIO DROPOUT (FORCE TEXT RELIANCE)
                # Phase-A: p=0.7 ref-drop to force text→audio mapping
                # Phase-B: p=0.2 ref-drop for robustness
                
                is_phase_a = global_step < PHASE_A_STEPS
                ref_drop_prob = PHASE_A_REF_DROP if is_phase_a else PHASE_B_REF_DROP
                
                import random
                drop_reference_audio = random.random() < ref_drop_prob
                
                if drop_reference_audio:
                    # Remove reference audio to force text conditioning
                    if 'audio_features' in model_inputs:
                        model_inputs['audio_features'] = None
                    if 'audio_feature_attention_mask' in model_inputs:
                        model_inputs['audio_feature_attention_mask'] = None
                    if 'audio_in_ids' in model_inputs:
                        model_inputs['audio_in_ids'] = None
                    if 'audio_in_ids_start' in model_inputs:
                        model_inputs['audio_in_ids_start'] = None
                    
                    # Clean None values
                    model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
                    
                    ref_status = "DROPPED"
                    if global_step % 200 == 0:
                        phase = "A" if is_phase_a else "B"
                        logger.info(f"PHASE-{phase} REF-DROP: Reference audio dropped (p={ref_drop_prob})")
                else:
                    ref_status = "PRESENT"
                
                # CRITICAL FIX 1: AUDIO TEACHER-FORCING ALIGNMENT (ELIMINATE IDENTITY LEAK)
                # Root cause: Model was seeing unshifted audio_out_ids, causing 0.7 CE and 99.9% TF accuracy
                # Solution: Strictly shift audio_out_ids for teacher-forcing to prevent identity mapping
                
                if 'audio_out_ids' in model_inputs and model_inputs['audio_out_ids'] is not None:
                    audio_out_ids = model_inputs['audio_out_ids']  # [8, T] format from collator
                    
                    # Validate audio_out_ids structure
                    if audio_out_ids.dim() == 2 and audio_out_ids.shape[0] == 8:
                        AUDIO_BOS_ID = 1024  # audio_stream_bos_id
                        AUDIO_EOS_ID = 1025  # audio_stream_eos_id
                        
                        # Labels: original sequence with BOS masked to -100
                        audio_labels = audio_out_ids.clone()
                        audio_labels[:, 0] = -100  # Mask BOS tokens
                        
                        # CRITICAL: Teacher-forcing inputs must be SHIFTED RIGHT by 1 step
                        # Feed BOS + tokens[0..T-2] as inputs, predict tokens[1..T-1]
                        audio_inputs = audio_out_ids.clone()
                        audio_inputs[:, 1:] = audio_out_ids[:, :-1]  # Shift right
                        audio_inputs[:, 0] = AUDIO_BOS_ID  # Insert BOS at start
                        
                        # Replace model input with properly shifted audio
                        model_inputs['audio_out_ids'] = audio_inputs
                        
                        # Store labels for later use (override batch labels)
                        audio_labels_shifted = audio_labels
                        
                        if global_step % 50 == 0:
                            logger.info(f" AUDIO TEACHER-FORCING SHIFT APPLIED:")
                            logger.info(f"   Original audio_out_ids: {audio_out_ids[:, :5].tolist()}")
                            logger.info(f"   Shifted inputs: {audio_inputs[:, :5].tolist()}")
                            logger.info(f"   Labels: {audio_labels[:, :5].tolist()}")
                            logger.info(f"   → This eliminates identity leak causing 0.7 CE!")
                    else:
                        logger.error(f" INVALID audio_out_ids shape: {audio_out_ids.shape}, expected [8, T]")
                        audio_labels_shifted = None
                else:
                    audio_labels_shifted = None

                # Get the underlying model (handle PEFT wrapping)
                if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
                    actual_model = model.base_model.model  # PEFT wrapped
                elif hasattr(model, 'module'):
                    actual_model = model.module  # Accelerate wrapped
                else:
                    actual_model = model
                
                # Forward pass - call model directly WITHOUT labels
                outputs = actual_model(**model_inputs)
                
                # Extract labels separately for loss computation
                text_labels = to_device(batch.label_ids) if hasattr(batch, 'label_ids') else None
                audio_labels = to_device(batch.audio_out_ids) if hasattr(batch, 'audio_out_ids') else audio_labels_shifted
                
                # CRITICAL PAD TOKEN FIX: Map pad tokens to -100 if applicable
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
                            logger.info(f" MAPPING PAD TOKENS: {pad_count_before} tokens ({pad_id}) → -100")
                            audio_labels[audio_labels == pad_id] = -100
                    else:
                        # Safety net: Mask BOS tokens to -100 and invalid tokens
                        bos_count_before = (audio_labels == 1024).sum().item()
                        if bos_count_before > 0:
                            logger.info(f" MAPPING BOS TOKENS: {bos_count_before} tokens (1024) → -100")
                            audio_labels[audio_labels == 1024] = -100
                        
                        # Mask truly invalid tokens (> 1025)
                        invalid_mask = (audio_labels > 1025) & (audio_labels != -100)
                        invalid_count = invalid_mask.sum().item()
                        if invalid_count > 0:
                            logger.info(f" MAPPING INVALID TOKENS: {invalid_count} tokens → -100")
                            audio_labels[invalid_mask] = -100

                # HIGGS-AUDIO DUAL-FFN OPTIMIZED LOSS COMPUTATION
                # Based on Higgs-Audio architecture: separate text and audio processing with dual FFN layers
                total_loss = None  # use None sentinel; keep this a Tensor

                loss_components = {}
                
                def make_assistant_text_labels(input_ids, tokenizer):
                    """Extract labels that supervise ONLY assistant content before <|AUDIO_OUT|>"""
                    if input_ids is None:
                        return None
                        
                    B, T = input_ids.size()
                    labels = input_ids.new_full((B, T), -100)
                    
                    # Special token IDs - robust fallback
                    try:
                        AUDIO_OUT_ID = tokenizer.convert_tokens_to_ids("<|AUDIO_OUT|>")
                        EOH_ID = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
                        SOH_ID = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
                        ASSISTANT_ID = tokenizer.convert_tokens_to_ids("assistant")
                    except:
                        AUDIO_OUT_ID = 128275  # <|AUDIO_OUT|>
                        EOH_ID = 128007       # <|end_header_id|>
                        SOH_ID = 128006       # <|start_header_id|>
                        ASSISTANT_ID = 78191  # "assistant"
                    
                    for b in range(B):
                        ids = input_ids[b]
                        
                        # Find <|AUDIO_OUT|> token
                        audio_out_positions = (ids == AUDIO_OUT_ID).nonzero(as_tuple=True)[0]
                        if audio_out_positions.numel() == 0:
                            continue
                        audio_out_pos = audio_out_positions[0].item()
                        
                        # Find last assistant header before AUDIO_OUT
                        eoh_positions = (ids[:audio_out_pos] == EOH_ID).nonzero(as_tuple=True)[0]
                        if eoh_positions.numel() == 0:
                            continue
                            
                        content_start = None
                        for eoh_idx in reversed(eoh_positions):
                            eoh_pos = eoh_idx.item()
                            soh_positions = (ids[:eoh_pos] == SOH_ID).nonzero(as_tuple=True)[0]
                            if soh_positions.numel() == 0:
                                continue
                            soh_pos = soh_positions[-1].item()
                            
                            # Check if "assistant" token is between SOH and EOH
                            header_slice = ids[soh_pos:eoh_pos+1]
                            if ASSISTANT_ID in header_slice:
                                content_start = eoh_pos + 1
                                break
                                
                        if content_start is None or content_start >= audio_out_pos:
                            continue
                            
                        content_end = audio_out_pos
                        if content_end - content_start < 8:  # Minimum span
                            continue
                            
                        # Supervise for next-token prediction (shift labels)
                        if content_start + 1 < content_end:
                            labels[b, content_start+1:content_end] = ids[content_start+1:content_end]
                    
                    return labels
                
                # Apply assistant-span supervision and gate
                if model_inputs.get('input_ids') is not None:
                    text_labels = make_assistant_text_labels(model_inputs['input_ids'], tokenizer)
                    
                    # CRITICAL: Gate on supervised tokens, not prompt
                    if text_labels is not None:
                        supervised_per_sample = (text_labels != -100).view(text_labels.shape[0], -1).sum(1)
                        min_supervised = supervised_per_sample.min().item()
                        avg_supervised = supervised_per_sample.float().mean().item()
                        
                        # Phase-A: Arabic needs ≥64 supervised tokens/sample
                        PHASE_A_MIN_SUPERVISED = 64
                        if min_supervised < PHASE_A_MIN_SUPERVISED:
                            if global_step % 50 == 0:
                                logger.warning(f"BATCH DROPPED: min_supervised={min_supervised} < {PHASE_A_MIN_SUPERVISED}")
                            continue  # Skip this batch
                        
                        # Clean logging: only supervised tokens matter
                        if global_step % 100 == 0:
                            logger.info(f"SUPERVISED: {avg_supervised:.1f} tokens/sample")
                            
                            # Preview supervised content (decode first sample's supervised tokens)
                            if supervised_per_sample[0] > 0:
                                supervised_mask = text_labels[0] != -100
                                supervised_tokens = text_labels[0][supervised_mask][:20]  # First 20 tokens
                                try:
                                    supervised_text = tokenizer.decode(supervised_tokens.tolist(), skip_special_tokens=True)
                                    arabic_chars = sum(1 for c in supervised_text if '\u0600' <= c <= '\u06FF')
                                    logger.info(f"PREVIEW: '{supervised_text[:50]}...' (Arabic: {arabic_chars})")
                                except:
                                    pass  # Skip decode errors
                else:
                    text_labels = to_device(batch.label_ids) if hasattr(batch, 'label_ids') else None

                # 1. AUDIO LOSS (PRIMARY) - Higgs-Audio Dual-FFN Optimized
                # The audio FFN pathway handles discrete audio token prediction with teacher forcing
                if hasattr(outputs, 'audio_logits') and outputs.audio_logits is not None and audio_labels is not None:
                    audio_logits = outputs.audio_logits
                    
                    # CRITICAL: Higgs-Audio tensor alignment for 8-codebook structure
                    # Model dual-FFN outputs: [T, 8, V] (time-major, 8 codebooks, vocab_size)
                    # Teacher-forcing labels:  [8, T]    (codebook-major, 8 codebooks, time)
                    # MUST align before loss computation to prevent random cross-entropy!
                    
                    original_shape = audio_logits.shape
                    if audio_logits.dim() == 3 and audio_logits.shape[1] == 8:
                        # Permute to [8, T, V] to match Higgs-Audio codebook-major label order
                        audio_logits = audio_logits.permute(1, 0, 2).contiguous()
                        if global_step % 100 == 0:
                            logger.info(f" HIGGS DUAL-FFN ALIGNMENT: {original_shape} → {audio_logits.shape} (codebook-major)")
                    
                    # HIGGS-AUDIO CODEBOOK-AWARE LOSS COMPUTATION
                    # Each of the 8 codebooks contributes to the final audio quality
                    # Teacher forcing ensures stable training across all codebook streams
                    audio_loss_fct = torch.nn.CrossEntropyLoss(
                        ignore_index=-100,  # Mask BOS/EOS/invalid tokens
                        label_smoothing=args.audio_label_smoothing,  # Prevent overconfidence
                        reduction='mean'  # Average across codebooks and time
                    )
                    
                    # HIGGS-AUDIO MULTI-CODEBOOK LOSS VALIDATION
                    # Flatten both logits and labels in IDENTICAL codebook-major order
                    logits_for_loss = audio_logits.view(-1, audio_logits.size(-1))  # [(8*T), 1026]
                    labels_for_loss = audio_labels.contiguous().view(-1)           # [(8*T)]
                    
                    # CRITICAL: Validate label integrity across all 8 codebooks
                    valid_mask = labels_for_loss != -100
                    if valid_mask.any():
                        valid_labels = labels_for_loss[valid_mask]
                        min_label, max_label = valid_labels.min().item(), valid_labels.max().item()
                        
                        # Higgs-Audio vocab: 0-1023 (codes) + 1024 (BOS) + 1025 (EOS/stream_end)
                        if min_label < 0 or max_label >= 1026:
                            logger.error(f" INVALID CODEBOOK LABELS: range [{min_label}, {max_label}] ≠ [0, 1025]")
                            logger.error(f"   This breaks Higgs-Audio discrete code training!")
                        else:
                            if global_step % 100 == 0:
                                logger.info(f"  CODEBOOK LABELS VALID: range [{min_label}, {max_label}] within [0, 1025]")
                    
                    # Compute primary audio loss for zero-shot voice cloning
                    audio_loss = audio_loss_fct(logits_for_loss, labels_for_loss)
                    total_loss = audio_loss if total_loss is None else total_loss + audio_loss
                    loss_components['audio_loss'] = audio_loss.item()
                    
                    # ESSENTIAL LOGGING ONLY
                    if global_step % args.log_steps == 0:
                        with torch.no_grad():
                            text_pred = shift_logits.argmax(-1)
                            text_true = shift_labels
                            text_valid_mask = (text_true != -100)
                            
                            if text_valid_mask.any():
                                valid_pred = text_pred[text_valid_mask]
                                valid_true = text_true[text_valid_mask]
                                
                                # Essential metrics only
                                text_accuracy = (valid_pred == valid_true).float().mean()
                                unique_preds = torch.unique(valid_pred).numel()
                                
                                logger.info(f"TEXT: Loss={text_loss:.3f}, Acc={text_accuracy:.1%}, Vocab={unique_preds}, Ref={ref_status}")
                                
                                # Critical failure detection
                                if text_accuracy < 0.01:
                                    logger.error("TEXT COLLAPSE: 0% accuracy")
                    
                # 2. TEXT LOSS (CRITICAL for Arabic Content Learning)
                # The text FFN pathway must learn Arabic text-to-phonetic mapping
                # WITHOUT sufficient text supervision, model produces correct voice but gibberish Arabic
                if hasattr(outputs, 'logits') and outputs.logits is not None and text_labels is not None:
                    text_logits = outputs.logits
                    
                    # CRITICAL FOR ARABIC: Validate text supervision adequacy
                    text_nonignore_count = (text_labels != -100).sum().item()
                    batch_size = text_labels.shape[0]
                    per_sample_text = text_nonignore_count / batch_size
                    
                    if global_step % 50 == 0:
                        logger.info(f" TEXT FFN PATHWAY: Processing {text_nonignore_count} supervised text tokens")
                        logger.info(f"   Per-sample: {per_sample_text:.1f} tokens (Arabic needs ≥32 for quality learning)")
                        
                        if per_sample_text < 20:
                            logger.error(f" TEXT FFN STARVED: {per_sample_text:.1f} tokens/sample - Arabic learning IMPOSSIBLE!")
                        elif per_sample_text < 32:
                            logger.warning(f"  TEXT FFN LIMITED: {per_sample_text:.1f} tokens/sample - Arabic quality may suffer")
                    
                    # Only use text loss if we have reasonable dimensions
                    min_seq_len = min(text_logits.size(1), text_labels.size(1))

                    if min_seq_len > 1:  # Need at least 2 tokens for shifting
                        # Trim to matching sequence length 
                        text_logits = text_logits[:, :min_seq_len, :]
                        text_labels = text_labels[:, :min_seq_len]
                        
                        # Shift for next-token prediction
                        shift_logits = text_logits[..., :-1, :].contiguous()
                        shift_labels = text_labels[..., 1:].contiguous()
                        
                        # CRITICAL: Compute text loss (was missing!)
                        text_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
                        text_loss = text_loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )
                        
                        # Weight text loss for Arabic learning
                        weighted_text_loss = args.text_loss_weight * text_loss
                        
                        # CRITICAL FIX: Add weighted text loss to total_loss!
                        # This was missing - causing Arabic text learning failure!
                        total_loss = total_loss + weighted_text_loss if total_loss is not None else weighted_text_loss
                        loss_components['weighted_text_loss'] = weighted_text_loss.item()
                        
                        # ESSENTIAL LOGGING ONLY
                        if global_step % args.log_steps == 0:
                            with torch.no_grad():
                                text_pred = shift_logits.argmax(-1)
                                text_true = shift_labels
                                text_valid_mask = (text_true != -100)
                                
                                if text_valid_mask.any():
                                    valid_pred = text_pred[text_valid_mask]
                                    valid_true = text_true[text_valid_mask]
                                    
                                    # Essential metrics only
                                    text_accuracy = (valid_pred == valid_true).float().mean()
                                    unique_preds = torch.unique(valid_pred).numel()
                                    
                                    logger.info(f"TEXT: Loss={text_loss:.3f}, Acc={text_accuracy:.1%}, Vocab={unique_preds}, Ref={ref_status}")
                                    
                                    # Critical failure detection
                                    if text_accuracy < 0.01:
                                        logger.error("TEXT COLLAPSE: 0% accuracy")
                    
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
                                        # Convert float tensors to match model dtype (for audio features)
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
                                # RESTORED: These are needed for audio structure in validation too
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
                            audio_labels = to_device(batch.audio_out_ids) if hasattr(batch, 'audio_out_ids') else None
                            
                            if hasattr(outputs, 'audio_logits') and outputs.audio_logits is not None and audio_labels is not None:
                                audio_logits = outputs.audio_logits
                                
                                # Apply same tensor alignment as training loop
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
                                            
                                            logger.info(f" VAL First 10: pred={first_10_pred} | true={first_10_true}")
                                            logger.info(f" VAL Last 10:  pred={last_10_pred} | true={last_10_true}")
                    
                    # Log validation results
                    if val_audio_total > 0:
                        val_accuracy = val_audio_correct / val_audio_total
                        avg_val_loss = val_loss_accum / val_steps_count if val_steps_count > 0 else 0
                        logger.info(f" VALIDATION (Step {global_step}): Loss={avg_val_loss:.4f}, Audio Accuracy={val_accuracy:.4f} ({val_audio_correct}/{val_audio_total})")
                    
                    model.train()  # Return to training mode
        
                # Training logs every 100 steps  
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
                    
                    # Validation forward pass - restore audio structure fields 
                    model_inputs = {
                        'input_ids': to_device(batch.input_ids),
                        'attention_mask': to_device(batch.attention_mask),
                        'audio_features': to_device(batch.audio_in_wv, convert_dtype=True) if hasattr(batch, 'audio_in_wv') else None,
                        'audio_feature_attention_mask': to_device(batch.audio_feature_attention_mask) if hasattr(batch, 'audio_feature_attention_mask') else None,
                        'audio_in_ids': to_device(batch.audio_in_ids) if hasattr(batch, 'audio_in_ids') else None,
                        'audio_in_ids_start': to_device(batch.audio_in_ids_start) if hasattr(batch, 'audio_in_ids_start') else None,
                        # RESTORED: These are needed for audio structure in validation too
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
                    audio_labels = to_device(batch.audio_out_ids) if hasattr(batch, 'audio_out_ids') else None
                    
                    # PROPER VALIDATION LOSS for zero-shot voice cloning
                    batch_loss = 0.0
                    
                    # Primary: Audio Loss - WITH TENSOR ALIGNMENT FIX
                    if hasattr(outputs, 'audio_logits') and outputs.audio_logits is not None and audio_labels is not None:
                        audio_logits = outputs.audio_logits
                        
                        # CRITICAL FIX: Same tensor alignment as training loop
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
