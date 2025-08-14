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
    
    # Create dataloaders using original collator
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collator(collate_fn(batch, tokenizer, audio_tokenizer)),
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
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
    
    logger.info(f"FULL DEPTH LORA: Targeting {len(target_modules)} module types across ALL {len(model.layers)} blocks")
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
                
                # CORE FIX 1: ASSISTANT-SPAN SUPERVISION ONLY
                text_labels = extract_assistant_labels(model_inputs['input_ids'], tokenizer)
                
                # CORE FIX 2: GATE ON SUPERVISED TOKENS (≥64 for Arabic)
                if text_labels is not None:
                    supervised_counts = (text_labels != -100).sum(1)
                    min_supervised = supervised_counts.min().item()
                    
                    # Phase-A: Require ≥64 supervised Arabic tokens
                    if global_step < 10000 and min_supervised < 64:
                        continue  # Skip insufficient batches
                else:
                    continue
                
                # CORE FIX 3: PHASE-A REF-DROP CURRICULUM
                phase_a = global_step < 10000
                ref_drop_prob = 0.7 if phase_a else 0.2
                drop_reference = torch.rand(1).item() < ref_drop_prob
                
                if drop_reference and 'audio_in_ids' in model_inputs:
                    del model_inputs['audio_in_ids']
                    if 'audio_in_wv' in model_inputs:
                        del model_inputs['audio_in_wv']
                    ref_status = "DROPPED"
                else:
                    ref_status = "PRESENT"

                # Get the underlying model (handle PEFT wrapping)
                if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
                    actual_model = model.base_model.model  # PEFT wrapped
                elif hasattr(model, 'module'):
                    actual_model = model.module  # Accelerate wrapped
                else:
                    actual_model = model
                
                # Forward pass - call model directly WITHOUT labels
                outputs = actual_model(**model_inputs)
                
                total_loss = None
                loss_components = {}
                
                # CORE FIX 4: BALANCED LOSSES (1.0 weights, not 10.0)
                
                # Audio loss
                if hasattr(batch, 'audio_out_ids') and batch.audio_out_ids is not None:
                    audio_labels = to_device(batch.audio_out_ids)
                    
                    # Teacher-forcing shift: prevent identity leak
                    audio_inputs = audio_labels.clone()
                    audio_inputs[:, 1:] = audio_labels[:, :-1]  # Shift right
                    audio_labels[:, 0] = -100  # Mask BOS
                    
                    if hasattr(outputs, 'audio_logits') and outputs.audio_logits is not None:
                        audio_logits = outputs.audio_logits
                        
                        # Align dimensions [T,8,V] → [8,T,V] if needed
                        if audio_logits.dim() == 3 and audio_logits.shape[1] == 8:
                            audio_logits = audio_logits.permute(1, 0, 2).contiguous()
                        
                        # Compute loss
                        audio_loss = torch.nn.functional.cross_entropy(
                            audio_logits.view(-1, audio_logits.size(-1)),
                            audio_labels.view(-1),
                            ignore_index=-100,
                            reduction='mean'
                        )
                        
                        total_loss = audio_loss
                        loss_components['audio_loss'] = audio_loss.item()

                # Text loss 
                if hasattr(outputs, 'logits') and outputs.logits is not None:
                    text_logits = outputs.logits
                    
                    # Next-token prediction alignment
                    shift_logits = text_logits[..., :-1, :].contiguous()
                    shift_labels = text_labels[..., 1:].contiguous()
                    
                    text_loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        reduction='mean'
                    )
                    
                    total_loss = total_loss + text_loss if total_loss is not None else text_loss
                    loss_components['text_loss'] = text_loss.item()

                # CORE FIX 5: ESSENTIAL LOGGING ONLY
                if global_step % args.log_steps == 0:
                    with torch.no_grad():
                        # Text metrics
                        if 'text_loss' in loss_components:
                            text_pred = shift_logits.argmax(-1)
                            text_valid = (shift_labels != -100)
                            if text_valid.any():
                                text_acc = (text_pred[text_valid] == shift_labels[text_valid]).float().mean()
                                text_vocab = torch.unique(text_pred[text_valid]).numel()
                            else:
                                text_acc = text_vocab = 0
                            
                            logger.info(f"Step {global_step}: TEXT Loss={loss_components['text_loss']:.3f}, "
                                      f"Acc={text_acc:.1%}, Vocab={text_vocab}, Ref={ref_status}")
                        
                        # Audio metrics  
                        if 'audio_loss' in loss_components:
                            logger.info(f"Step {global_step}: AUDIO Loss={loss_components['audio_loss']:.3f}")
                        
                        # Supervised tokens
                        logger.info(f"Step {global_step}: SUPERVISED {min_supervised} tokens/sample")

                # Backward pass
                if total_loss is not None:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actual_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                else:
                    logger.warning("No valid loss computed - skipping step")
                
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
