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
                

                
                # CRITICAL DEBUG: Log model inputs to diagnose T=0 audio logits issue
                if global_step % 10 == 0:
                    logger.info(f" HIGGS-AUDIO TRAINING DIAGNOSTICS (Step {global_step}):")
                    for key, value in model_inputs.items():
                        if value is not None and hasattr(value, 'shape'):
                            logger.info(f"  {key}: {value.shape}")
                        elif value is not None:
                            logger.info(f"  {key}: {type(value)} (non-tensor)")
                        else:
                            logger.info(f"  {key}: None")
                    
                    # CRITICAL: Verify teacher-forcing structure (NOT leakage)
                    if 'audio_out_ids' in model_inputs and model_inputs['audio_out_ids'] is not None:
                        audio_out_shape = model_inputs['audio_out_ids'].shape
                        logger.info(f" TEACHER-FORCING STRUCTURE: audio_out_ids {audio_out_shape} (for alignment, NOT leakage)")
                        if len(audio_out_shape) > 1 and audio_out_shape[1] == 0:
                            logger.error(f" CRITICAL: Empty teacher-forcing structure! This breaks audio generation!")
                    else:
                        logger.error(f" CRITICAL: Missing teacher-forcing structure! Model cannot align audio generation!")
                
                # DECISIVE TEXT CONDITIONING DIAGNOSTICS (D1-D3 from analysis)
                if global_step % 50 == 0:  # Run comprehensive diagnostics
                    logger.info(f" TEXT CONDITIONING DIAGNOSTICS (Step {global_step}):")
                    
                    with torch.no_grad():
                        # D1: Text conditioning A/B test - CRITICAL for Arabic learning
                        try:
                            # Baseline forward pass
                            baseline_outputs = actual_model(**model_inputs)
                            
                            # Scramble text tokens to test dependency
                            scrambled_inputs = model_inputs.copy()
                            if 'input_ids' in scrambled_inputs and scrambled_inputs['input_ids'] is not None:
                                orig_input_ids = scrambled_inputs['input_ids'].clone()
                                # Scramble only text tokens (preserve special tokens)
                                for b in range(orig_input_ids.shape[0]):
                                    text_mask = (orig_input_ids[b] < 128000) & (orig_input_ids[b] > 1000)  # Text token range
                                    if text_mask.any():
                                        text_indices = torch.where(text_mask)[0]
                                        scrambled_order = text_indices[torch.randperm(len(text_indices))]
                                        orig_input_ids[b, text_indices] = orig_input_ids[b, scrambled_order]
                                scrambled_inputs['input_ids'] = orig_input_ids
                                scrambled_outputs = actual_model(**scrambled_inputs)
                                
                                # Compare audio logits - if similar, text NOT used for audio generation
                                if hasattr(baseline_outputs, 'audio_logits') and hasattr(scrambled_outputs, 'audio_logits'):
                                    # Calculate entropy manually: -sum(p * log(p))
                                    baseline_probs = torch.nn.functional.softmax(baseline_outputs.audio_logits, dim=-1)
                                    scrambled_probs = torch.nn.functional.softmax(scrambled_outputs.audio_logits, dim=-1)
                                    baseline_entropy = -(baseline_probs * torch.log(baseline_probs + 1e-8)).sum(dim=-1).mean()
                                    scrambled_entropy = -(scrambled_probs * torch.log(scrambled_probs + 1e-8)).sum(dim=-1).mean()
                                    entropy_diff = abs(baseline_entropy - scrambled_entropy)
                                    logger.info(f"  🎯 TEXT CONDITIONING TEST: entropy_diff={entropy_diff:.4f}")
                                    if entropy_diff < 0.1:
                                        logger.error(f"🚨 CRITICAL: Model NOT using text for audio generation! (entropy_diff < 0.1)")
                                    else:
                                        logger.info(f"  ✅ Model DOES use text for audio generation (entropy_diff >= 0.1)")
                            
                        except Exception as e:
                            logger.info(f"  📝 Text conditioning test failed: {str(e)[:100]}")
                
                # 🚨 PRE-FORWARD TEXT SUPERVISION GUARDRAIL
                if global_step % 10 == 0:
                    logger.info(f"🔍 PRE-FORWARD CHECKS: Ensuring text supervision adequacy...")
                    
                    # CRITICAL: Fail fast if batch lacks sufficient text for Arabic learning
                    if hasattr(batch, 'label_ids') and batch.label_ids is not None:
                        temp_text_labels = to_device(batch.label_ids)
                        temp_nonignore = (temp_text_labels != -100).sum().item()
                        batch_size = temp_text_labels.shape[0]
                        per_sample_text = temp_nonignore / batch_size
                        logger.info(f"  📊 BATCH TEXT SUPERVISION: {temp_nonignore} total, {per_sample_text:.1f} per sample")
                        
                        # GUARDRAIL: Demand adequate text supervision for Arabic learning
                        if per_sample_text < 32:
                            logger.error(f"🚨 INSUFFICIENT TEXT SUPERVISION: {per_sample_text:.1f} tokens/sample < 32 required for Arabic!")
                            logger.error(f"   This will cause 'mumbling' - model learns voice style but ignores Arabic text content!")
                        elif per_sample_text >= 50:
                            logger.info(f"  ✅ ADEQUATE TEXT SUPERVISION: {per_sample_text:.1f} tokens/sample for Arabic learning")
                        else:
                            logger.warning(f"  ⚠️  MARGINAL TEXT SUPERVISION: {per_sample_text:.1f} tokens/sample - may need improvement")

                # Forward pass - call model directly WITHOUT labels
                outputs = actual_model(**model_inputs)
                
                # 🚨 D2: ATTENTION MASK SANITY CHECK - CRITICAL for Arabic text conditioning
                if global_step % 50 == 0:
                    logger.info(f"🔍 POST-FORWARD ATTENTION DIAGNOSTICS (Step {global_step}):")
                    
                    # Check expanded_input_ids structure
                    if hasattr(outputs, 'expanded_input_ids'):
                        exp_shape = outputs.expanded_input_ids.shape
                        logger.info(f"  📝 expanded_input_ids shape: {exp_shape}")
                        
                        # Analyze sequence composition for one sample
                        if exp_shape[0] > 0:
                            sample_seq = outputs.expanded_input_ids[0]  # First sample
                            
                            # Identify text vs audio regions (rough heuristic)
                            text_tokens = ((sample_seq < 128000) & (sample_seq > 1000)).sum().item()
                            audio_tokens = ((sample_seq >= 1024) & (sample_seq <= 1025)).sum().item()
                            total_tokens = (sample_seq != 0).sum().item()  # Non-pad tokens
                            
                            logger.info(f"  📊 SEQUENCE COMPOSITION: {text_tokens} text, {audio_tokens} audio, {total_tokens} total")
                            
                            # CRITICAL: Verify audio_out positions can attend to text
                            if hasattr(outputs, 'attention_mask') and outputs.attention_mask is not None:
                                attn_mask = outputs.attention_mask[0]  # First sample
                                seq_len = attn_mask.shape[-1]
                                
                                # Check if latter positions (audio_out region) can attend to early positions (text region)
                                if seq_len > 50:  # Ensure sufficient length
                                    # Sample a mid-sequence position (likely audio_out region)
                                    mid_pos = seq_len // 2
                                    early_pos = min(20, seq_len // 4)  # Early position (likely text region)
                                    
                                    can_attend_early = attn_mask[mid_pos, early_pos].item() if attn_mask.dim() >= 2 else 1.0
                                    can_attend_recent = attn_mask[mid_pos, max(0, mid_pos-5)].item() if attn_mask.dim() >= 2 else 1.0
                                    
                                    logger.info(f"  🎯 ATTENTION CHECK: pos{mid_pos}→pos{early_pos} = {can_attend_early:.3f} (text access)")
                                    logger.info(f"  🎯 ATTENTION CHECK: pos{mid_pos}→pos{mid_pos-5} = {can_attend_recent:.3f} (recent access)")
                                    
                                    # CRITICAL: Audio positions MUST attend to text for Arabic learning
                                    if can_attend_early < 0.5:
                                        logger.error(f"🚨 ATTENTION MASK BROKEN: Audio positions cannot attend to text! This causes mumbling!")
                                    else:
                                        logger.info(f"  ✅ ATTENTION MASK HEALTHY: Audio can attend to text for Arabic learning")
                    else:
                        logger.info(f"  📝 No expanded_input_ids in outputs - using input_ids structure analysis")
                        
                        # Fallback: analyze input_ids structure
                        if 'input_ids' in model_inputs and model_inputs['input_ids'] is not None:
                            input_seq = model_inputs['input_ids'][0]
                            text_tokens = ((input_seq < 128000) & (input_seq > 1000)).sum().item()
                            logger.info(f"  📊 INPUT STRUCTURE: {text_tokens} text tokens in input_ids")
                
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

                # 🚨 D3: COMPREHENSIVE POST-EXTRACTION DIAGNOSTICS
                if global_step % 10 == 0:
                    logger.info(f"🔍 POST-EXTRACTION LABEL ANALYSIS (Step {global_step}):")
                    
                    # D3: Non-ignore text count - CRITICAL for Arabic learning
                    if text_labels is not None:
                        total_text_labels = text_labels.numel()
                        text_non_ignore = (text_labels != -100).sum().item()
                        batch_size = text_labels.shape[0]
                        per_sample_text = text_non_ignore / batch_size
                        text_supervision_ratio = text_non_ignore / total_text_labels if total_text_labels > 0 else 0.0
                        
                        logger.info(f"  📊 FINAL TEXT LABELS: {text_non_ignore} non-ignore / {total_text_labels} total ({text_supervision_ratio:.1%})")
                        logger.info(f"  📊 PER-SAMPLE TEXT: {per_sample_text:.1f} tokens/sample (need ≥32 for Arabic)")
                        
                        # CRITICAL: Final text supervision validation
                        if per_sample_text < 20:
                            logger.error(f"🚨 FATAL: Only {per_sample_text:.1f} text tokens/sample - Arabic learning IMPOSSIBLE!")
                            logger.error(f"   → Model will learn voice style but produce gibberish Arabic content!")
                        elif per_sample_text < 32:
                            logger.error(f"🚨 CRITICAL: Only {per_sample_text:.1f} text tokens/sample - insufficient for robust Arabic!")
                        elif per_sample_text >= 50:
                            logger.info(f"  ✅ EXCELLENT TEXT SUPERVISION: {per_sample_text:.1f} tokens/sample for Arabic learning")
                        else:
                            logger.warning(f"  ⚠️  MARGINAL TEXT SUPERVISION: {per_sample_text:.1f} tokens/sample")
                        
                        # Sample text tokens to verify Arabic tokenization quality
                        if text_non_ignore > 0:
                            non_ignore_mask = text_labels != -100
                            sample_tokens = text_labels[non_ignore_mask][:10].tolist()
                            logger.info(f"  🔤 Sample text tokens: {sample_tokens}")
                            
                            # Check for Arabic token patterns (rough heuristic)
                            arabic_range_count = sum(1 for tok in sample_tokens if 50000 <= tok <= 70000)
                            if arabic_range_count > 0:
                                logger.info(f"  🔤 Arabic token indicators: {arabic_range_count}/10 in potential Arabic range")
                            else:
                                logger.warning(f"  ⚠️  No clear Arabic token patterns detected - check tokenization!")
                    else:
                        logger.info(f"  📝 No text labels in batch - text-only mode disabled")
                    
                    # Audio label validation and BOS masking check
                    if audio_labels is not None:
                        audio_non_ignore = (audio_labels != -100).sum().item()
                        audio_total = audio_labels.numel()
                        logger.info(f"  📊 AUDIO LABELS: {audio_non_ignore} non-ignore / {audio_total} total")
                        
                        # Verify BOS masking (critical for autoregressive training)
                        if audio_labels.dim() >= 2 and audio_labels.shape[0] >= 8 and audio_labels.shape[1] > 0:
                            first_tokens_masked = (audio_labels[:8, 0] == -100).sum().item()
                            logger.info(f"  🔒 BOS masking: {first_tokens_masked}/8 codebooks have -100 at t=0")
                            if first_tokens_masked == 8:
                                logger.info(f"  ✅ PERFECT BOS masking for autoregressive training")
                            else:
                                logger.error(f"🚨 BROKEN BOS masking: {first_tokens_masked}/8 - will break audio generation!")

                # 🎯 HIGGS-AUDIO DUAL-FFN OPTIMIZED LOSS COMPUTATION
                # Based on Higgs-Audio architecture: separate text and audio processing with dual FFN layers
                total_loss = None  # use None sentinel; keep this a Tensor

                loss_components = {}
                

                # 🎯 1. AUDIO LOSS (PRIMARY) - Higgs-Audio Dual-FFN Optimized
                # The audio FFN pathway handles discrete audio token prediction with teacher forcing
                if hasattr(outputs, 'audio_logits') and outputs.audio_logits is not None and audio_labels is not None:
                    audio_logits = outputs.audio_logits
                    
                    # 🚨 CRITICAL: Higgs-Audio tensor alignment for 8-codebook structure
                    # Model dual-FFN outputs: [T, 8, V] (time-major, 8 codebooks, vocab_size)
                    # Teacher-forcing labels:  [8, T]    (codebook-major, 8 codebooks, time)
                    # MUST align before loss computation to prevent random cross-entropy!
                    
                    original_shape = audio_logits.shape
                    if audio_logits.dim() == 3 and audio_logits.shape[1] == 8:
                        # Permute to [8, T, V] to match Higgs-Audio codebook-major label order
                        audio_logits = audio_logits.permute(1, 0, 2).contiguous()
                        if global_step % 100 == 0:
                            logger.info(f"🔧 HIGGS DUAL-FFN ALIGNMENT: {original_shape} → {audio_logits.shape} (codebook-major)")
                    
                    # 🎯 HIGGS-AUDIO CODEBOOK-AWARE LOSS COMPUTATION
                    # Each of the 8 codebooks contributes to the final audio quality
                    # Teacher forcing ensures stable training across all codebook streams
                    audio_loss_fct = torch.nn.CrossEntropyLoss(
                        ignore_index=-100,  # Mask BOS/EOS/invalid tokens
                        label_smoothing=args.audio_label_smoothing,  # Prevent overconfidence
                        reduction='mean'  # Average across codebooks and time
                    )
                    
                    # 🚨 HIGGS-AUDIO MULTI-CODEBOOK LOSS VALIDATION
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
                            logger.error(f"🚨 INVALID CODEBOOK LABELS: range [{min_label}, {max_label}] ≠ [0, 1025]")
                            logger.error(f"   This breaks Higgs-Audio discrete code training!")
                        else:
                            if global_step % 100 == 0:
                                logger.info(f"  ✅ CODEBOOK LABELS VALID: range [{min_label}, {max_label}] within [0, 1025]")
                    
                    # Compute primary audio loss for zero-shot voice cloning
                    audio_loss = audio_loss_fct(logits_for_loss, labels_for_loss)
                    total_loss = audio_loss if total_loss is None else total_loss + audio_loss
                    loss_components['audio_loss'] = audio_loss.item()
                    
                    # 📊 HIGGS-AUDIO CODEBOOK-SPECIFIC DIAGNOSTICS
                    if global_step % 100 == 0:
                        # Compute per-codebook cross-entropy for detailed analysis
                        per_codebook_losses = []
                        for cb in range(8):
                            cb_logits = audio_logits[cb].view(-1, audio_logits.size(-1))  # [T, V]
                            cb_labels = audio_labels[cb].contiguous().view(-1)           # [T]
                            cb_valid_mask = cb_labels != -100
                            if cb_valid_mask.any():
                                cb_loss = torch.nn.functional.cross_entropy(
                                    cb_logits[cb_valid_mask], cb_labels[cb_valid_mask], reduction='mean'
                                )
                                per_codebook_losses.append(f"{cb_loss.item():.3f}")
                            else:
                                per_codebook_losses.append("N/A")
                        
                        logger.info(f"  📊 PER-CODEBOOK CE: {per_codebook_losses}")
                        
                        # Check for codebook collapse (all predictions similar)
                        unique_preds = torch.unique(logits_for_loss.argmax(dim=-1)).numel()
                        logger.info(f"  🎯 PREDICTION DIVERSITY: {unique_preds}/1026 unique tokens")
                        if unique_preds < 100:
                            logger.warning(f"  ⚠️  LOW DIVERSITY: Model may be collapsing to few tokens!")
                    
                    # 🎯 HIGGS-AUDIO TRAINING HEALTH MONITORING
                    if global_step % args.log_steps == 0:
                        # 🎯 HIGGS-AUDIO TEACHER-FORCING ACCURACY
                        predictions = logits_for_loss.argmax(dim=-1)
                        valid_predictions = labels_for_loss != -100
                        correct_predictions = (predictions == labels_for_loss) & valid_predictions
                        accuracy = correct_predictions.sum().float() / valid_predictions.sum().float() if valid_predictions.any() else 0.0
                        
                        logger.info(f"🔊 AUDIO LOSS (Step {global_step}): {audio_loss:.4f}")
                        logger.info(f"🎯 TEACHER-FORCING ACCURACY: {accuracy:.1%} (across 8 codebooks)")
                        
                        # 🔍 CODEBOOK PATTERN ANALYSIS (detect copying/mumbling)
                        sample_size = min(20, len(predictions))
                        first_preds = predictions[:sample_size].tolist()
                        first_true = labels_for_loss[:sample_size].tolist()
                        last_preds = predictions[-sample_size:].tolist() 
                        last_true = labels_for_loss[-sample_size:].tolist()
                        
                        logger.info(f"🎯 SEQUENCE START: pred={first_preds[:10]} | true={first_true[:10]}")
                        logger.info(f"🎯 SEQUENCE END:   pred={last_preds[-10:]} | true={last_true[-10:]}")
                        
                        # 🎯 HIGGS-AUDIO VOCABULARY DIVERSITY ANALYSIS
                        # Critical for Arabic: model must use diverse tokens, not just common ones
                        pred_tokens = predictions[valid_predictions]
                        label_tokens = labels_for_loss[valid_predictions]
                        
                        unique_preds = torch.unique(pred_tokens).numel()
                        unique_labels = torch.unique(label_tokens).numel()
                        logger.info(f"🔍 VOCABULARY DIVERSITY: pred={unique_preds}, labels={unique_labels} (of 1026 possible)")
                        
                        # Check for vocabulary collapse (Arabic mumbling indicator)
                        vocab_coverage = unique_preds / 1026.0
                        if vocab_coverage < 0.1:
                            logger.error(f"🚨 VOCABULARY COLLAPSE: Only {vocab_coverage:.1%} of vocab used - Arabic will be gibberish!")
                        elif vocab_coverage > 0.3:
                            logger.info(f"✅ RICH VOCABULARY: {vocab_coverage:.1%} coverage - good for Arabic diversity")
                        
                        # 🎯 LEARNING PROGRESS ASSESSMENT
                        baseline_ce = 6.933  # log(1026) for uniform distribution
                        progress_ratio = (baseline_ce - audio_loss.item()) / baseline_ce
                        
                        if progress_ratio < 0.1:
                            logger.warning(f"⚠️  MINIMAL LEARNING: {progress_ratio:.1%} progress from random baseline")
                            logger.warning(f"   Check reference audio conditioning and text supervision!")
                        elif progress_ratio > 0.6:
                            logger.info(f"✅ EXCELLENT PROGRESS: {progress_ratio:.1%} improvement from baseline")
                        else:
                            logger.info(f"🎯 STEADY LEARNING: {progress_ratio:.1%} progress from random baseline")
                            
                        # 🚨 SANITY CHECK 2: Audio token comparison (first/last 10)
                        pred = logits_for_loss.argmax(-1)  # [8, T] 
                        valid_mask = (labels_for_loss != -100)
                        

                    elif audio_ce > 6.0:
                        logger.warning(f"🚨 HIGH AUDIO CE ({audio_ce:.4f}) - Model struggling to learn audio")
                        

                
                # 🎯 2. TEXT LOSS (CRITICAL for Arabic Content Learning)
                # The text FFN pathway must learn Arabic text-to-phonetic mapping
                # WITHOUT sufficient text supervision, model produces correct voice but gibberish Arabic
                if hasattr(outputs, 'logits') and outputs.logits is not None and text_labels is not None:
                    text_logits = outputs.logits
                    
                    # 🚨 CRITICAL FOR ARABIC: Validate text supervision adequacy
                    text_nonignore_count = (text_labels != -100).sum().item()
                    batch_size = text_labels.shape[0]
                    per_sample_text = text_nonignore_count / batch_size
                    
                    if global_step % 50 == 0:
                        logger.info(f"🔍 TEXT FFN PATHWAY: Processing {text_nonignore_count} supervised text tokens")
                        logger.info(f"   Per-sample: {per_sample_text:.1f} tokens (Arabic needs ≥32 for quality learning)")
                        
                        if per_sample_text < 20:
                            logger.error(f"🚨 TEXT FFN STARVED: {per_sample_text:.1f} tokens/sample - Arabic learning IMPOSSIBLE!")
                        elif per_sample_text < 32:
                            logger.warning(f"⚠️  TEXT FFN LIMITED: {per_sample_text:.1f} tokens/sample - Arabic quality may suffer")
                    
                    # Only use text loss if we have reasonable dimensions
                    min_seq_len = min(text_logits.size(1), text_labels.size(1))

                    if min_seq_len > 1:  # Need at least 2 tokens for shifting
                        # Trim to matching sequence length 
                        text_logits = text_logits[:, :min_seq_len, :]
                        text_labels = text_labels[:, :min_seq_len]
                        
                        # Shift for next-token prediction
                        shift_logits = text_logits[..., :-1, :].contiguous()
                        shift_labels = text_labels[..., 1:].contiguous()
                        
                        # 🚨 CRITICAL: Compute text loss (was missing!)
                        text_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
                        text_loss = text_loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )
                        
                        # Weight text loss for Arabic learning
                        weighted_text_loss = args.text_loss_weight * text_loss
                        
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
                                    logger.info(f"🎯 HEALTHY TEXT LEARNING: {text_loss:.4f} - Arabic quality developing")
                            
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
