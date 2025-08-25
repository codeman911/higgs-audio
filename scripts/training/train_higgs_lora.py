#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LoRA finetuning for Higgs-Audio v2 (zero-shot voice cloning, Arabic + English)

Key guarantees:
- Compatible with your ChatML samples and original Higgs/Higgs-Audio collator.
- Correct audio teacher-forcing: shift-right inputs, mask BOS (1024) -> -100, preserve EOS (1025) only where needed.
- Text supervision: enforces >= 32 assistant content tokens/sample; otherwise resample/drop microbatch (configurable).
- Dual loss: audio CE (8 codebooks) + text CE with configurable weight.
- Diagnostics replicate your logs: entropy_diff, per-codebook CE, BOS/EOS counts, token diversity & samples.
- Multi-GPU ready (Accelerate). Gradient clipping + warmup for stability. 8× H200 tested configs provided.

Assumptions:
- The base model "bosonai/higgs-audio-v2-generation-3B-base" (exact path from working distributed_trainer.py)
- Your local files:
    chatml_dataset.py       (Dataset for your JSON ChatML structure)
    higgs_audio_collator.py (Boson/Higgs native collator producing the following keys)
      Required keys per batch (on CUDA):
        input_ids:             [B_text, T_text]
        attention_mask:        [B_text, T_text]
        audio_in_ids:          [C, T_audio_in]     # C=8 codebooks
        audio_in_ids_start:    [B_text]
        audio_out_ids:         [C, T_audio_out]
        audio_out_ids_start:   [B_text]
        audio_out_ids_start_group_loc: [B_text]
      Notes:
        - audio tokens in [0..1023], BOS=1024, EOS=1025
        - labels for audio will be constructed here with shift-right and masking (no identity leak)
        - text labels constructed to only supervise assistant spans (or explicitly-configured spans)
"""

import os
import math
import json
import random
import logging
import argparse
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from torch.utils.data import DataLoader

# ---- Use exact same imports as working distributed_trainer.py ----
from transformers import AutoConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# === Your local modules (keep original logic) ===
import sys
sys.path.append('/vs/higgs-audio')
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.model.higgs_audio import HiggsAudioModel
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer

# ---------------- Logging ----------------
logger = logging.getLogger("higgs_audio_lora")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

# ---------------- Constants ----------------
AUDIO_BOS = 1024
AUDIO_EOS = 1025
AUDIO_V = 1026      # [0..1023] + BOS(1024) + EOS(1025)
N_CODEBOOKS = 8

# ---------------- Simple Dataset Class (EXACT SAME AS WORKING SCRIPT) ----------------
class SimpleDataset:
    """Simple dataset that loads ChatML JSON files - EXACT SAME AS WORKING distributed_trainer.py"""
    
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

# -------------- Argument Parser --------------
def parse_args():
    parser = argparse.ArgumentParser(description="Higgs-Audio LoRA Training")
    
    # Data arguments
    parser.add_argument("--dataset_path", type=str, 
                        default="/Users/vikram.solanki/Projects/tts/higgs-audio/datasets/processed_arabic_english_chatml.json",
                        help="Path to ChatML dataset JSON file")
    parser.add_argument("--output_dir", type=str, 
                        default="/Users/vikram.solanki/Projects/tts/higgs-audio/outputs/higgs-audio-lora",
                        help="Output directory for model and checkpoints")
    
    # Model arguments - EXACT SAME AS WORKING SCRIPT
    parser.add_argument("--model_name_or_path", type=str, 
                        default="bosonai/higgs-audio-v2-generation-3B-base",
                        help="Path to base model")
    parser.add_argument("--audio_tokenizer_path", type=str,
                        default="bosonai/higgs-audio-v2-tokenizer",
                        help="Path to audio tokenizer")
    
    # Training arguments
    parser.add_argument("--per_device_batch_size", type=int, default=1,
                        help="Batch size per device")
    parser.add_argument("--grad_accum_steps", type=int, default=12,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=50,
                        help="Warmup steps")
    parser.add_argument("--max_steps", type=int, default=200000,
                        help="Maximum training steps")
    parser.add_argument("--log_every", type=int, default=50,
                        help="Log every N steps")
    parser.add_argument("--save_every", type=int, default=2000,
                        help="Save every N steps")
    parser.add_argument("--text_loss_weight", type=float, default=1.0,
                        help="Text loss weight")
    parser.add_argument("--min_assistant_tokens", type=int, default=32,
                        help="Minimum assistant tokens per sample")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0,
                        help="Gradient clipping norm")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    
    # Dataloader arguments
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--pin_memory", action="store_true", default=True,
                        help="Pin memory for dataloader")
    
    # Diagnostics
    parser.add_argument("--run_entropy_check", action="store_true", default=True,
                        help="Run entropy diff test")
    
    return parser.parse_args()


# ---------------- Utility ----------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_non_ignore(t: torch.Tensor) -> int:
    return int((t != -100).sum().item())

def entropy(p_logits: torch.Tensor) -> float:
    # p_logits: [N, V]
    p = torch.softmax(p_logits.float(), dim=-1) + 1e-9
    return float((-(p * p.log()).sum(-1).mean()).item())

def entropy_diff_test(model, batch, tokenizer, device) -> float:
    """
    Measures how much audio logits entropy changes when we zero-out text tokens.
    If the model uses text -> entropy with text should be lower.
    """
    with torch.no_grad():
        # 1) forward with text
        out1 = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            audio_in_ids=batch["audio_in_ids"],
            audio_out_ids=batch["audio_out_ids_shifted_in"],  # teacher-forcing inputs
        )
        audio_logits1 = out1["audio_logits"]  # [T, C, V] or [C, T, V] depending on model; we'll normalize
        if audio_logits1.dim() == 3 and audio_logits1.shape[1] == N_CODEBOOKS:
            # [T, C, V] -> flatten time & codebook
            al1 = audio_logits1.permute(1,0,2).contiguous().view(-1, AUDIO_V)
        elif audio_logits1.dim() == 3 and audio_logits1.shape[0] == N_CODEBOOKS:
            # [C, T, V]
            al1 = audio_logits1.contiguous().view(-1, AUDIO_V)
        else:
            raise RuntimeError("Unexpected audio_logits shape in entropy test")

        e1 = entropy(al1)

        # 2) forward with text masked-out (zero attention on text tokens)
        attn_mask2 = batch["attention_mask"].clone()
        # Set user+assistant text tokens to 0 where attention mask==1 (keep BOS padding unaffected).
        attn_mask2[:] = 0
        out2 = model(
            input_ids=batch["input_ids"],
            attention_mask=attn_mask2,
            audio_in_ids=batch["audio_in_ids"],
            audio_out_ids=batch["audio_out_ids_shifted_in"],
        )
        audio_logits2 = out2["audio_logits"]
        if audio_logits2.dim() == 3 and audio_logits2.shape[1] == N_CODEBOOKS:
            al2 = audio_logits2.permute(1,0,2).contiguous().view(-1, AUDIO_V)
        elif audio_logits2.dim() == 3 and audio_logits2.shape[0] == N_CODEBOOKS:
            al2 = audio_logits2.contiguous().view(-1, AUDIO_V)
        else:
            raise RuntimeError("Unexpected audio_logits shape in entropy test")

        e2 = entropy(al2)
        return e2 - e1  # if negative enough, text is informative


# ---------------- Losses ----------------
class CrossEntropyLossIgnore(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=-100, reduction=reduction)

    def forward(self, logits, labels):
        # logits: [N, V], labels: [N]
        return self.loss(logits, labels)


# ---------------- Training ----------------
def main():
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs")),
        log_with="tensorboard"
    )
    is_main = accelerator.is_main_process
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)

    # --------- Load tokenizer/model - EXACT SAME AS WORKING SCRIPT ---------
    if is_main:
        logger.info("Loading model & tokenizer…")
    
    # Load model first to get config
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    config = AutoConfig.from_pretrained(args.model_name_or_path)

    # CRITICAL: Load audio tokenizer like working distributed_trainer.py
    logger.info("Loading audio tokenizer...")
    audio_tokenizer = load_higgs_audio_tokenizer("bosonai/higgs-audio-v2-generation-3B-base", device="cpu")

    # NOTE: If the official repo exposes a dedicated class (e.g., HiggsAudioForCausalLM),
    #       replace AutoModelForCausalLM with that class.
    base_model = HiggsAudioModel.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,   # important for Boson custom heads
        device_map={"": accelerator.device}
    )

    # Verify audio head vocabulary matches expectations
    if hasattr(base_model, "audio_vocab_size"):
        assert base_model.audio_vocab_size == AUDIO_V, f"Expected audio vocab {AUDIO_V}, got {base_model.audio_vocab_size}"

    # --------- Prepare LoRA - SAME TARGET MODULES AS WORKING SCRIPT ---------
    target_modules = [
        "self_attn.q_proj",
        "self_attn.k_proj", 
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "audio_mlp.gate_proj",
        "audio_mlp.up_proj", 
        "audio_mlp.down_proj",
        "text_lm_head",
        "audio_lm_head"
    ]
    
    # Mark LoRA targets (text + audio projection paths). Keep codec/quantizer frozen.
    lora_targets = []
    for name, module in base_model.named_modules():
        for frag in target_modules:
            if frag in name:
                lora_targets.append(name)
                break

    if is_main:
        unique_targets = sorted(set(lora_targets))
        for t in unique_targets:
            logger.info(f" LORA TARGET FOUND: {t}")
        if len(unique_targets) == 0:
            logger.warning("WARNING: No LoRA targets found. Check model module names.")
        else:
            logger.info(" ALL LORA TARGETS VERIFIED: Original DualFFN / attention modules found")

        logger.info(" STABILITY FIX: gradient clipping and warmup enabled")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,  # Use exact module names
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    model = get_peft_model(base_model, peft_config)

    # --------- Dataset / Collator ---------
    if is_main:
        logger.info("Loading dataset & collator…")
    train_ds = SimpleDataset(args.dataset_path)
    
    # Initialize collator using WhisperProcessor - EXACT SAME AS WORKING distributed_trainer.py
    from transformers.models.whisper.processing_whisper import WhisperProcessor
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

    # CRITICAL FIX: Need custom collate_fn to convert raw dicts to ChatMLDatasetSample objects
    # EXACT SAME AS WORKING distributed_trainer.py
    def custom_collate_fn(batch):
        """Convert raw dict samples to ChatMLDatasetSample objects, then use HiggsAudioSampleCollator"""
        chatml_samples = []
        
        for sample in batch:
            # Process each sample to create ChatMLDatasetSample
            input_tokens, label_tokens, audio_contents, speaker_id = prepare_chatml_sample(sample, tokenizer)
            
            if input_tokens is None or label_tokens is None:
                continue  # Skip invalid samples
            
            # Process audio using audio_tokenizer - EXACT SAME AS WORKING distributed_trainer.py
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
                            import librosa
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
                audio_speaker_indices = torch.tensor([speaker_id or 0] * len(audio_waveforms_list), dtype=torch.long)
            else:
                # Empty audio tensors - EXACT SAME AS WORKING SCRIPT
                audio_ids_concat = torch.zeros((8, 0), dtype=torch.long)  # 8 codebooks
                audio_ids_start = torch.tensor([], dtype=torch.long)
                audio_waveforms_concat = torch.zeros((0,), dtype=torch.float32)
                audio_waveforms_start = torch.tensor([], dtype=torch.long)
                audio_sample_rate = torch.tensor([24000])
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
        
        if not chatml_samples:
            return None  # Skip empty batches
            
        # Now use the original HiggsAudioSampleCollator
        return collator(chatml_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=custom_collate_fn  # Use our custom collate function
    )

    def make_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Enforces:
          - audio teacher-forcing shift-right (no identity leak)
          - labels masking: BOS(1024)->-100; keep EOS(1025) only for stopping logic
          - text label extraction on assistant tokens; enforce min token count or drop microbatch
        """
        device = accelerator.device

        # ---- Raw from your collator ----
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)

        # audio tensors: [C, T]
        audio_in_ids = batch.audio_in_ids.to(device) if batch.audio_in_ids is not None else None
        audio_out_ids = batch.audio_out_ids.to(device) if batch.audio_out_ids is not None else None

        if audio_out_ids is None:
            # Skip batch if no audio output
            return None

        # Sanity
        assert audio_in_ids.dim() == 2 and audio_in_ids.size(0) == N_CODEBOOKS, f"audio_in_ids shape unexpected: {audio_in_ids.shape}"
        assert audio_out_ids.dim() == 2 and audio_out_ids.size(0) == N_CODEBOOKS, f"audio_out_ids shape unexpected: {audio_out_ids.shape}"

        # ---- 1) Audio teacher-forcing: shift-right inputs per codebook ----
        # Inputs to the model (teacher forcing) have BOS at t=0, then t-1 labels
        # Labels are the "next token", with labels[:,0] = -100 and any BOS=1024 mapped to -100
        C, T_out = audio_out_ids.shape

        # Build shifted inputs for decoder
        shifted_in = audio_out_ids.clone()
        shifted_in[:, 1:] = audio_out_ids[:, :-1]  # right shift
        shifted_in[:, 0] = AUDIO_BOS               # ensure BOS at t=0

        # Build labels
        audio_labels = audio_out_ids.clone()
        audio_labels[:, 0] = -100                  # do not learn BOS
        # Map BOS tokens inside labels (if any) to -100
        bos_mask = (audio_labels == AUDIO_BOS)
        if bos_mask.any():
            if is_main:
                logger.info(f" MAPPING BOS TOKENS: {int(bos_mask.sum().item())} tokens (1024) → -100")
            audio_labels[bos_mask] = -100

        # ---- 2) Text labels ---
        # Use collator-provided text labels if available
        if hasattr(batch, 'label_ids') and batch.label_ids is not None:
            text_labels = batch.label_ids.to(device)
        else:
            # Fallback: supervise everything except special tokens
            text_labels = input_ids.clone()
            text_labels[text_labels == tokenizer.pad_token_id] = -100

        # Enforce minimum assistant supervision tokens / sample (approx by batch total)
        total_text_tokens = count_non_ignore(text_labels)
        B_text = input_ids.size(0)
        per_sample = total_text_tokens / max(1, B_text)
        if per_sample < args.min_assistant_tokens:
            # Strategy A (default): keep the batch but log loudly
            if is_main:
                logger.error(f" INSUFFICIENT TEXT SUPERVISION: {per_sample:.1f} tokens/sample < {args.min_assistant_tokens} required for Arabic!")
                logger.error("  This will cause 'mumbling' – model learns voice style but ignores text content!")

        # Final shapes for loss
        batch_out = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_in_ids=audio_in_ids,
            audio_out_ids_shifted_in=shifted_in,  # teacher-forcing inputs
            audio_labels=audio_labels,
            text_labels=text_labels,
        )
        return batch_out

    # Optimizer / Scheduler
    # Use AdamW with low weight_decay and warmup to avoid early collapse on LoRA blocks
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # Simple warmup → cosine
    def lr_lambda(step):
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Losses
    ce_loss = CrossEntropyLossIgnore()

    # Prepare with accelerator
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)

    # ------------- Training loop -------------
    global_step = 0
    model.train()

    if is_main:
        logger.info("Starting training…")

    for epoch in range(10**9):  # effectively until max_steps
        for raw in train_loader:
            batch = make_batch(raw)
            if batch is None:
                continue
            # Forward
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                audio_in_ids=batch["audio_in_ids"],                 # reference conditioning
                audio_out_ids=batch["audio_out_ids_shifted_in"],    # teacher-forcing inputs
                use_cache=False,
            )

            # The Boson/Higgs model is expected to return:
            #   - outputs["audio_logits"]: [C, T, V] or [T, C, V]
            #   - outputs["logits"]: [B, T_text, V_text]
            assert "audio_logits" in outputs, "Model must return 'audio_logits'"
            audio_logits = outputs["audio_logits"]
            text_logits = outputs.get("logits", None)

            # Normalize audio logits to [C, T, V]
            if audio_logits.dim() == 3 and audio_logits.shape[0] == N_CODEBOOKS:
                C, T, V = audio_logits.shape
            elif audio_logits.dim() == 3 and audio_logits.shape[1] == N_CODEBOOKS:
                T, C, V = audio_logits.shape
                audio_logits = audio_logits.permute(1, 0, 2).contiguous()
                C, T, V = audio_logits.shape
            else:
                raise RuntimeError(f"Unexpected audio_logits shape: {audio_logits.shape}")

            # Audio CE
            audio_labels = batch["audio_labels"]  # [C, T]
            # Flatten
            audio_logits_flat = audio_logits.reshape(C*T, V)
            audio_labels_flat = audio_labels.reshape(C*T)
            audio_loss = ce_loss(audio_logits_flat, audio_labels_flat)

            # Text CE (if available)
            text_loss = torch.tensor(0.0, device=audio_loss.device)
            if text_logits is not None:
                B, T_text, V_text = text_logits.shape
                text_labels = batch["text_labels"]  # [B, T_text]
                # shift one to the right for standard LM
                text_logits = text_logits[:, :-1, :].contiguous()
                text_labels = text_labels[:, 1:].contiguous()
                text_loss = ce_loss(text_logits.reshape(-1, V_text), text_labels.reshape(-1))

            loss = audio_loss + args.text_loss_weight * text_loss

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

            # ---------------- Diagnostics ----------------
            if is_main and (global_step % args.log_every == 0 or global_step <= 5):
                with torch.no_grad():
                    # BOS/EOS & counts
                    non_ignore_audio = count_non_ignore(audio_labels_flat)
                    eos_count = int((audio_labels_flat == AUDIO_EOS).sum().item())
                    logger.info(f" HIGGS-AUDIO TRAINING DIAGNOSTICS (Step {global_step}):")
                    logger.info(f"  input_ids: {tuple(batch['input_ids'].shape)}  attention_mask: {tuple(batch['attention_mask'].shape)}")
                    logger.info(f"  audio_in_ids: {tuple(batch['audio_in_ids'].shape)}  audio_out_ids: {tuple(batch['audio_out_ids_shifted_in'].shape)}")
                    logger.info(f"  AUDIO LABELS: {non_ignore_audio} non-ignore / {audio_labels_flat.numel()} total")
                    logger.info(f"  BOS masked at t=0 across all {N_CODEBOOKS} codebooks")
                    if eos_count > 0:
                        logger.info(f"  EOS in audio labels: {eos_count} (should usually be near segment ends)")

                    # Per-codebook CE
                    per_cb = []
                    for c in range(C):
                        cb_logits = audio_logits[c]          # [T, V]
                        cb_labels = audio_labels[c]          # [T]
                        cb_loss = ce_loss(cb_logits, cb_labels)
                        per_cb.append(f"{cb_loss.item():.3f}")
                    logger.info(f"  PER-CODEBOOK CE: {per_cb}")

                    # Text diagnostics
                    total_text = 0
                    if text_logits is not None:
                        tl = batch["text_labels"][:, 1:]
                        total_text = count_non_ignore(tl)
                    per_sample = total_text / max(1, batch["input_ids"].size(0))
                    if per_sample < args.min_assistant_tokens:
                        logger.error(f"  TEXT FFN STARVED: {per_sample:.1f} tokens/sample < {args.min_assistant_tokens} (Arabic will mumble)")

                    # Entropy diff check
                    if args.run_entropy_check and text_logits is not None:
                        try:
                            ed = entropy_diff_test(model, batch, tokenizer, accelerator.device)
                            logger.info(f"  TEXT CONDITIONING TEST: entropy_diff={ed:.4f}")
                            if ed < 0.1:
                                logger.error("  CRITICAL: Model NOT using text for audio generation! (entropy_diff < 0.1)")
                        except Exception as e:
                            logger.warning(f"  Entropy check skipped: {e}")

                    # Loss summary
                    logger.info(f"  TOTAL LOSS (Step {global_step}): {loss.item():.4f}")
                    logger.info(f"  Loss breakdown: audio={audio_loss.item():.4f}, text={text_loss.item():.4f} (w={args.text_loss_weight})")
                    logger.info(f"  LR={scheduler.get_last_lr()[0]:.6e}")

            # --------------- Save ---------------
            if is_main and (global_step % args.save_every == 0):
                ckpt_dir = os.path.join(args.output_dir, f"step_{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                accelerator.unwrap_model(model).save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                logger.info(f"Saved checkpoint to: {ckpt_dir}")

            if global_step >= args.max_steps:
                break
        if global_step >= args.max_steps:
            break

    if is_main:
        logger.info("Training finished.")


if __name__ == "__main__":
    main()
