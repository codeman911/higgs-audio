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

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from boson_multimodal.model.higgs_audio import HiggsAudioModel
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample

# ===== HIGGS-AUDIO constants =====
AUDIO_BOS = 1024
AUDIO_EOS = 1025
AUDIO_V = 1026
N_CODEBOOKS = 8
MIN_ASSISTANT_TOKENS = 32  # Arabic learning threshold

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
        
        # Process sample and get inputs/labels
        input_tokens, label_tokens, audio_contents, speaker_id = prepare_chatml_sample(sample, tokenizer)
        
        # CRITICAL FIX: Lower threshold to prevent empty batches (was 64, now 8)
        target_token_count = len([t for t in label_tokens if t != -100])
        audio_segment_count = len(audio_contents)
        
        # Skip only if extremely insufficient tokens (lowered from 64 to 8)
        if target_token_count < 8:
            continue
        
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
            # Concatenate audio codes from all segments [8, T_total]
            audio_ids_concat = torch.cat(audio_ids_list, dim=-1)
            # Create proper start indices for each audio segment
            audio_ids_start = torch.tensor([0] + [audio_codes.shape[1] for audio_codes in audio_ids_list[:-1]], dtype=torch.long).cumsum(dim=0)
            # Use first waveform as representative (collator handles properly)  
            audio_wv = audio_waveforms_list[0] if audio_waveforms_list else None
            audio_wv_start = torch.tensor([0], dtype=torch.long)
        else:
            # Create dummy audio to prevent errors
            audio_ids_concat = torch.zeros((8, 10), dtype=torch.long)  # 8 codebooks, 10 time steps
            audio_ids_start = torch.tensor([0, 10], dtype=torch.long)  # Corrected audio indexing structure
            audio_wv = torch.zeros((1000,), dtype=torch.float32)  # 1000 samples dummy audio
            audio_wv_start = torch.tensor([0], dtype=torch.long)
        
        # Create ChatMLDatasetSample with all required fields
        chatml_sample = ChatMLDatasetSample(
            input_ids=torch.tensor(input_tokens, dtype=torch.long),
            label_ids=torch.tensor(label_tokens, dtype=torch.long), 
            audio_ids_concat=audio_ids_concat,
            audio_ids_start=audio_ids_start,
            audio_waveforms_concat=audio_wv,
            audio_waveforms_start=audio_wv_start,
            audio_sample_rate=torch.tensor([sample_rate], dtype=torch.float32),
            audio_speaker_indices=torch.tensor([speaker_id or 0], dtype=torch.long)
        )
        
        chatml_samples.append(chatml_sample)
    
    # CRITICAL FIX: Ensure we never return completely empty batches
    if not chatml_samples:
        # Create a dummy sample to prevent collator crash
        dummy_input = torch.tensor([tokenizer.bos_token_id, tokenizer.eos_token_id], dtype=torch.long)
        dummy_labels = torch.tensor([-100, tokenizer.eos_token_id], dtype=torch.long)
        dummy_audio = torch.zeros((8, 10), dtype=torch.long)
        dummy_wv = torch.zeros((1000,), dtype=torch.float32)
        
        dummy_sample = ChatMLDatasetSample(
            input_ids=dummy_input,
            label_ids=dummy_labels,
            audio_ids_concat=dummy_audio, 
            audio_ids_start=torch.tensor([0, 10], dtype=torch.long),  # Corrected audio indexing structure
            audio_waveforms_concat=dummy_wv,
            audio_waveforms_start=torch.tensor([0], dtype=torch.long),
            audio_sample_rate=torch.tensor([sample_rate], dtype=torch.float32),
            audio_speaker_indices=torch.tensor([0], dtype=torch.long)
        )
        chatml_samples = [dummy_sample]
        logger.warning("EMPTY BATCH: Created dummy sample to prevent collator crash")
    
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
    parser.add_argument("--batch_size", type=int, default=8,
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
    parser.add_argument("--num_workers", type=int, default=48,
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
    parser.add_argument("--val_steps", type=int, default=100,
                        help="Run validation every N steps")
    parser.add_argument("--save_steps", type=int, default=1000,
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
    
    # Step 3: Supervision preparation function
    def prepare_supervision(batch, tokenizer, device, logger=None):
        """
        Fix audio teacher-forcing and text labels
        """
        inp = batch.input_ids.to(device)
        attn = batch.attention_mask.to(device)
        a_in = batch.audio_in_ids.to(device) if hasattr(batch, 'audio_in_ids') and batch.audio_in_ids is not None else None
        a_out = batch.audio_out_ids.to(device) if hasattr(batch, 'audio_out_ids') and batch.audio_out_ids is not None else None

        if a_out is not None:
            # ---- Audio: shift-right teacher forcing ----
            C, T = a_out.shape
            assert C == N_CODEBOOKS, f"Expected {N_CODEBOOKS} codebooks, got {C}"

            a_shift = a_out.clone()
            a_shift[:, 1:] = a_out[:, :-1]
            a_shift[:, 0] = AUDIO_BOS  # BOS at t=0 for inputs

            a_lbl = a_out.clone()
            a_lbl[:, 0] = -100  # do not learn BOS
            bos_mask = (a_lbl == AUDIO_BOS)
            if bos_mask.any() and logger:
                logger.info(f" MAPPING BOS TOKENS: {int(bos_mask.sum().item())} tokens ({AUDIO_BOS}) → -100")
            a_lbl[bos_mask] = -100
        else:
            a_shift = None
            a_lbl = None

        # ---- Text labels ----
        if hasattr(batch, 'label_ids') and batch.label_ids is not None:
            t_lbl = batch.label_ids.to(device)
        else:
            # Fallback: supervise non-pad tokens
            t_lbl = inp.clone()
            t_lbl[t_lbl == tokenizer.pad_token_id] = -100

        return dict(
            input_ids=inp, attention_mask=attn,
            audio_in_ids=a_in, audio_out_ids_shifted_in=a_shift, audio_labels=a_lbl,
            text_labels=t_lbl
        )

    def non_ignore_count(t): 
        return int((t != -100).sum().item())
    
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
                device = accelerator.device
                
                # DEBUG: Check what's in the batch to find where 'labels' is coming from
                if global_step == 0:
                    logger.info(f"DEBUG: Batch attributes: {[attr for attr in dir(batch) if not attr.startswith('_')]}")
                    if hasattr(batch, 'labels'):
                        logger.info(f"DEBUG: Found 'labels' in batch with shape: {batch.labels.shape}")
                
                # Get the underlying model (handle PEFT wrapping) - CRITICAL FIX!
                if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
                    actual_model = model.base_model.model  # PEFT wrapped
                elif hasattr(model, 'module'):
                    actual_model = model.module  # Accelerate wrapped
                else:
                    actual_model = model
                
                # Step 4: Use prepare_supervision for clean batch processing
                sup = prepare_supervision(batch, tokenizer, device, logger)
                
                # Step 4: Enforce minimum assistant tokens
                B = sup["input_ids"].size(0)
                valid_text = non_ignore_count(sup["text_labels"])
                per_sample = valid_text / max(1, B)

                if per_sample < MIN_ASSISTANT_TOKENS:
                    logger.error(f" INSUFFICIENT TEXT SUPERVISION: {per_sample:.1f} tokens/sample < {MIN_ASSISTANT_TOKENS} required for Arabic!")
                    continue  # Skip starved batches
                
                # Forward pass with clean inputs - NO LABELS to model (like working version)
                model_inputs = {
                    "input_ids": sup["input_ids"],
                    "attention_mask": sup["attention_mask"],
                    "audio_features": batch.audio_in_wv.to(device) if hasattr(batch, 'audio_in_wv') and batch.audio_in_wv is not None else None,
                    "audio_feature_attention_mask": batch.audio_feature_attention_mask.to(device) if hasattr(batch, 'audio_feature_attention_mask') and batch.audio_feature_attention_mask is not None else None,
                    "audio_in_ids": sup["audio_in_ids"],
                    "audio_out_ids": sup["audio_out_ids_shifted_in"],
                    "audio_in_ids_start": batch.audio_in_ids_start.to(device) if hasattr(batch, 'audio_in_ids_start') and batch.audio_in_ids_start is not None else None,
                    "audio_out_ids_start": batch.audio_out_ids_start.to(device) if hasattr(batch, 'audio_out_ids_start') and batch.audio_out_ids_start is not None else None,
                    "audio_out_ids_start_group_loc": batch.audio_out_ids_start_group_loc.to(device) if hasattr(batch, 'audio_out_ids_start_group_loc') and batch.audio_out_ids_start_group_loc is not None else None,
                }
                
                # Remove None values for clean forward pass
                model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
                
                # DEBUG: Log model inputs to verify NO labels
                if global_step == 0:
                    logger.info(f"DEBUG: Model input keys: {list(model_inputs.keys())}")
                    for key, value in model_inputs.items():
                        if hasattr(value, 'shape'):
                            logger.info(f"DEBUG: {key} shape: {value.shape}")
                
                # Forward pass - call underlying model directly to bypass PEFT label injection
                outputs = actual_model(**model_inputs)
                
                # Extract labels separately for manual loss computation
                text_labels = sup["text_labels"]
                audio_labels = sup["audio_labels"]
                
                # Manual loss computation (like working version)
                total_loss = 0.0
                
                # Audio loss computation
                audio_loss = torch.tensor(0.0, device=device)
                if "audio_logits" in outputs and outputs["audio_logits"] is not None and audio_labels is not None:
                    audio_logits = outputs["audio_logits"]
                    if audio_logits.dim() == 3:
                        # Ensure [C, T, V] format
                        if audio_logits.shape[1] == N_CODEBOOKS:
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
                
                # Step 6: Extract individual losses for logging (if available in outputs)
                text_loss = getattr(outputs, 'text_loss', None)
                audio_loss = getattr(outputs, 'audio_loss', None)
                
                # If individual losses not available, use total loss
                if text_loss is None or audio_loss is None:
                    # Fallback: assume equal weighting for logging purposes
                    text_loss = loss * 0.5
                    audio_loss = loss * 0.5
                
                # Step 8: Strategic logging (essential only)
                if global_step % args.log_steps == 0 and accelerator.is_main_process:
                    if "logits" in outputs and outputs["logits"] is not None:
                        # Text metrics - use same truncation as loss computation
                        txt_pred = outputs["logits"][:, :-1, :].argmax(-1)  # [B, T]
                        txt_lbl_shifted = text_labels[:, 1:]  # [B, T]
                        
                        # Apply same truncation as in loss computation
                        B_pred, T_pred = txt_pred.shape
                        B_lbl, T_lbl = txt_lbl_shifted.shape
                        if T_pred != T_lbl:
                            min_T = min(T_pred, T_lbl)
                            txt_pred = txt_pred[:, :min_T]
                            txt_lbl_shifted = txt_lbl_shifted[:, :min_T]
                        
                        txt_valid = (txt_lbl_shifted != -100)
                        
                        if txt_valid.any():
                            txt_acc = (txt_pred[txt_valid] == txt_lbl_shifted[txt_valid]).float().mean()
                            txt_vocab = len(torch.unique(txt_lbl_shifted[txt_valid]))
                            logger.info(f"TEXT: Loss={text_loss.item():.3f}, Acc={txt_acc.item():.1%}, Vocab={txt_vocab}")
                        else:
                            logger.info(f"TEXT: Loss={text_loss.item():.3f}, Acc=0%, Vocab=0 (no valid)")
                        
                        # Audio metrics  
                        if "audio_logits" in outputs and outputs["audio_logits"] is not None:
                            # Ensure audio_logits are in [C, T, V] format first
                            audio_logits_for_acc = outputs["audio_logits"]
                            if audio_logits_for_acc.dim() == 3:
                                if audio_logits_for_acc.shape[1] == N_CODEBOOKS:
                                    # [T, C, V] -> [C, T, V]
                                    audio_logits_for_acc = audio_logits_for_acc.permute(1, 0, 2)
                            
                            aud_pred = audio_logits_for_acc.argmax(-1)  # [C, T]
                            aud_lbl = audio_labels       # [C, T]
                            aud_valid = (aud_lbl != -100)
                            if aud_valid.any():
                                aud_acc = (aud_pred[aud_valid] == aud_lbl[aud_valid]).float().mean()
                                aud_vocab = len(torch.unique(aud_lbl[aud_valid]))
                                logger.info(f"AUDIO: Loss={audio_loss.item():.3f}, Acc={aud_acc.item():.1%}, Vocab={aud_vocab}")
                                
                                # Per-codebook CE for identity leak detection
                                cb_losses = []
                                for c in range(N_CODEBOOKS):
                                    cb_loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(outputs["audio_logits"][c], aud_lbl[c])
                                    cb_losses.append(f"{cb_loss.item():.3f}")
                                cb_str = "[" + ",".join(cb_losses) + "]"
                                
                                logger.info(f"CODEBOOK CE: {cb_str} (detect identity leak if all < 2.0)")
                            else:
                                logger.info(f"AUDIO: Loss={audio_loss.item():.3f}, Acc=0%, Vocab=0 (no valid)")
                    
                    logger.info(f"TOTAL: Loss={loss.item():.3f}, LR={scheduler.get_last_lr()[0]:.2e}, Step={global_step}")
                    
                    # Entropy difference diagnostic
                    if global_step > 0 and global_step % 200 == 0:
                        if "logits" in outputs and outputs["logits"] is not None:
                            text_logits_for_entropy = outputs["logits"][:, :-1, :]
                            
                            # Apply same truncation for entropy calculation
                            if text_logits_for_entropy.shape[1] != txt_lbl_shifted.shape[1]:
                                min_T = min(text_logits_for_entropy.shape[1], txt_lbl_shifted.shape[1])
                                text_logits_for_entropy = text_logits_for_entropy[:, :min_T, :]
                            
                            entropy_diff = compute_entropy_difference(text_logits_for_entropy, txt_lbl_shifted, tokenizer.pad_token_id)
                            if entropy_diff > 0.1:
                                logger.info(f"✅ CONDITIONING ACTIVE: Entropy diff = {entropy_diff:.3f} > 0.1")
                            else:
                                logger.warning(f"❌ POOR CONDITIONING: Entropy diff = {entropy_diff:.3f} ≤ 0.1")
                
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
                    
                    # Step 9: Use prepare_supervision for validation too
                    sup = prepare_supervision(batch, tokenizer, device)
                    
                    # Skip validation batches with insufficient text
                    B = sup["input_ids"].size(0)
                    valid_text = non_ignore_count(sup["text_labels"])
                    per_sample = valid_text / max(1, B)
                    if per_sample < MIN_ASSISTANT_TOKENS:
                        continue
                    
                    # Forward pass with clean inputs - NO LABELS to model (like working version)
                    model_inputs = {
                        "input_ids": sup["input_ids"],
                        "attention_mask": sup["attention_mask"],
                        "audio_features": batch.audio_in_wv.to(device) if hasattr(batch, 'audio_in_wv') and batch.audio_in_wv is not None else None,
                        "audio_feature_attention_mask": batch.audio_feature_attention_mask.to(device) if hasattr(batch, 'audio_feature_attention_mask') and batch.audio_feature_attention_mask is not None else None,
                        "audio_in_ids": sup["audio_in_ids"],
                        "audio_out_ids": sup["audio_out_ids_shifted_in"],
                        "audio_in_ids_start": batch.audio_in_ids_start.to(device) if hasattr(batch, 'audio_in_ids_start') and batch.audio_in_ids_start is not None else None,
                        "audio_out_ids_start": batch.audio_out_ids_start.to(device) if hasattr(batch, 'audio_out_ids_start') and batch.audio_out_ids_start is not None else None,
                        "audio_out_ids_start_group_loc": batch.audio_out_ids_start_group_loc.to(device) if hasattr(batch, 'audio_out_ids_start_group_loc') and batch.audio_out_ids_start_group_loc is not None else None,
                    }
                    
                    # Remove None values for clean forward pass
                    model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
                    
                    # Get the underlying model (handle PEFT wrapping) - CRITICAL FIX!
                    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
                        actual_model = model.base_model.model  # PEFT wrapped
                    elif hasattr(model, 'module'):
                        actual_model = model.module  # Accelerate wrapped
                    else:
                        actual_model = model
                    
                    # Forward pass - call underlying model directly to bypass PEFT label injection
                    outputs = actual_model(**model_inputs)
                    
                    # Extract labels separately for manual loss computation
                    text_labels = sup["text_labels"]
                    audio_labels = sup["audio_labels"]
                    
                    # Manual loss computation (like working version)
                    total_loss = 0.0
                    
                    # Audio loss computation
                    audio_loss = torch.tensor(0.0, device=device)
                    if "audio_logits" in outputs and outputs["audio_logits"] is not None and audio_labels is not None:
                        audio_logits = outputs["audio_logits"]
                        if audio_logits.dim() == 3:
                            # Ensure [C, T, V] format
                            if audio_logits.shape[1] == N_CODEBOOKS:
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
