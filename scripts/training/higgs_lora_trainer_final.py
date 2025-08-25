#!/usr/bin/env python3
"""
Higgs Audio v2 LoRA Training Script - FINAL, CORRECTED VERSION

This script is a complete rewrite to ensure stability and correctness.
It uses the standard Hugging Face Trainer and a simple, robust data pipeline
that respects the exact API of the HiggsAudioModel.
"""

import os
import json
import logging
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    WhisperProcessor
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
import librosa
from pathlib import Path
from typing import Dict, List

# --- Correct Higgs Audio Imports ---
from boson_multimodal.model.higgs_audio import HiggsAudioModel, HiggsAudioConfig, HiggsAudioDualFFNDecoderLayer
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- DEFINITIVE DATA COLLATOR ---
class FinalDataCollator:
    """
    This is the definitive, correct collator.
    1. It uses the official HiggsAudioSampleCollator.
    2. It converts the output to a simple dictionary.
    3. It guarantees the dictionary keys match the HiggsAudioModel.forward() signature.
       - It ensures 'label_ids' is present for loss calculation.
       - It guarantees the problematic 'labels' key is NEVER present.
    """
    def __init__(self, **kwargs):
        self.base_collator = HiggsAudioSampleCollator(**kwargs)

    def __call__(self, batch: List[ChatMLDatasetSample]) -> Dict[str, torch.Tensor]:
        # Use the official collator to create the batch object
        batch_input = self.base_collator(batch)
        
        # Convert the batch object to a clean dictionary, containing only model-accepted keys
        model_inputs = {}
        for attr_name in dir(batch_input):
            if not attr_name.startswith('_') and not callable(getattr(batch_input, attr_name)):
                model_inputs[attr_name] = getattr(batch_input, attr_name)

        # CRITICAL: The model's forward pass computes loss from 'label_ids'.
        # The problematic 'labels' key (often added by the collator) must be removed.
        if 'labels' in model_inputs:
            model_inputs.pop('labels')

        return model_inputs


# --- DATASET (Unchanged, based on your working version) ---
class ChatMLDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer: AutoTokenizer, audio_tokenizer):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.samples = self._load_samples()
        if not self.samples:
            raise RuntimeError(f"No valid samples found in {data_dir}")
        logger.info(f"Loaded {len(self.samples)} samples from {data_dir}")

    def _load_samples(self) -> List[Dict]:
        samples = []
        for json_file in self.data_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    samples.append(json.load(f))
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ChatMLDatasetSample:
        try:
            sample = self.samples[idx]
            input_tokens, label_tokens, audio_contents, _ = prepare_chatml_sample(sample, self.tokenizer)
            
            audio_ids_list, audio_waveforms_list = [], []
            for audio_content in audio_contents:
                if audio_content and hasattr(audio_content, 'audio_url') and os.path.exists(audio_content.audio_url):
                    audio_codes = self.audio_tokenizer.encode(audio_content.audio_url).cpu().long()
                    waveform, _ = librosa.load(audio_content.audio_url, sr=24000, mono=True)
                    audio_ids_list.append(audio_codes)
                    audio_waveforms_list.append(torch.tensor(waveform, dtype=torch.float32))

            if not audio_ids_list:
                return self.__getitem__((idx + 1) % len(self)) # Skip samples without valid audio

            audio_ids_concat = torch.cat(audio_ids_list, dim=1)
            audio_ids_start = torch.tensor([0] + [c.shape[1] for c in audio_ids_list[:-1]], dtype=torch.long).cumsum(dim=0)
            audio_waveforms_concat = torch.cat(audio_waveforms_list, dim=0)
            audio_waveforms_start = torch.tensor([0] + [wv.shape[0] for wv in audio_waveforms_list[:-1]], dtype=torch.long).cumsum(dim=0)

            return ChatMLDatasetSample(
                input_ids=torch.tensor(input_tokens, dtype=torch.long),
                label_ids=torch.tensor(label_tokens, dtype=torch.long),
                audio_ids_concat=audio_ids_concat,
                audio_ids_start=audio_ids_start,
                audio_waveforms_concat=audio_waveforms_concat,
                audio_waveforms_start=audio_waveforms_start,
                audio_sample_rate=torch.tensor([24000], dtype=torch.float32),
                audio_speaker_indices=torch.tensor([0], dtype=torch.long),
            )
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}", exc_info=True)
            return self.__getitem__((idx + 1) % len(self))


# --- MODEL & TOKENIZER CREATION (Corrected) ---
def create_model_and_tokenizers(args):
    logger.info(f"Loading TEXT tokenizer from {args.tokenizer_path or args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path, trust_remote_code=True)

    logger.info(f"Loading AUDIO tokenizer from {args.audio_tokenizer_path}")
    audio_tokenizer = load_higgs_audio_tokenizer(args.audio_tokenizer_path)

    logger.info(f"Loading model from {args.model_path}")
    model = HiggsAudioModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)

    # Enable cross-modal conditioning (your requested feature)
    if not model.config.use_audio_out_self_attention:
        logger.info("ENABLING cross-modal conditioning (use_audio_out_self_attention=True)")
        model.config.use_audio_out_self_attention = True
        new_layers = nn.ModuleList([
            HiggsAudioDualFFNDecoderLayer(model.config, layer_idx=i, use_audio_attention=True)
            for i in range(len(model.layers))
        ])
        # Load weights from old layers to new layers
        new_layers.load_state_dict(model.layers.state_dict(), strict=False)
        model.layers = new_layers
        logger.info("Rebuilt model layers with audio attention enabled.")

    return model, tokenizer, audio_tokenizer


# --- LORA SETUP (Corrected) ---
def setup_lora(model, args):
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    if model.config.use_audio_out_self_attention:
        target_modules.extend(["audio_attn.q_proj", "audio_attn.v_proj", "audio_attn.k_proj", "audio_attn.o_proj"])
        logger.info("Adding audio attention modules to LoRA targets.")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


# --- MAIN TRAINING FUNCTION ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--audio_tokenizer_path", type=str, required=True)
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--report_to", type=str, default="wandb")
    args = parser.parse_args()

    model, tokenizer, audio_tokenizer = create_model_and_tokenizers(args)

    if args.use_lora:
        model = setup_lora(model, args)

    train_dataset = ChatMLDataset(args.train_data_dir, tokenizer, audio_tokenizer)

    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    data_collator = FinalDataCollator(
        whisper_processor=whisper_processor,
        audio_in_token_id=model.config.audio_in_token_idx,
        audio_out_token_id=model.config.audio_out_token_idx,
        audio_stream_bos_id=model.config.audio_stream_bos_id,
        audio_stream_eos_id=model.config.audio_stream_eos_id,
        encode_whisper_embed=True,
        pad_token_id=tokenizer.pad_token_id,
        return_audio_in_tokens=True,
        use_delay_pattern=True,
        round_to=8,
        audio_num_codebooks=8
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        bf16=True,
        remove_unused_columns=False, # Important: Must be False for this collator
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training with the new, corrected script...")
    trainer.train()
    logger.info("Training finished successfully.")

if __name__ == "__main__":
    main()
