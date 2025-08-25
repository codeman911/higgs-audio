#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for Arabic Zero-Shot Voice Cloning

This script implements LoRA (Low-Rank Adaptation) fine-tuning of Higgs Audio v2 
for Arabic zero-shot voice cloning using ChatML format data.
"""

import os
import json
import torch
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger
import warnings

# Transformers and PEFT imports
from transformers import (
    AutoTokenizer, 
    AutoConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)

# Higgs Audio imports
from boson_multimodal.model.higgs_audio import HiggsAudioModel, HiggsAudioConfig
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer

# Custom imports
from arabic_chatml_dataset import create_arabic_chatml_datasets, ArabicChatMLDataset

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class LoRATrainingArguments:
    """Configuration for LoRA training."""
    
    # Data
    chatml_file: str = field(metadata={"help": "Path to Arabic ChatML JSON file"})
    audio_base_path: Optional[str] = field(default=None, metadata={"help": "Base path for audio files"})
    
    # Model paths
    model_path: str = field(default="bosonai/higgs-audio-v2-generation-3B-base")
    audio_tokenizer_path: str = field(default="bosonai/higgs-audio-v2-tokenizer") 
    
    # LoRA configuration
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})
    lora_bias: str = field(default="none", metadata={"help": "LoRA bias type"})
    
    # Training configuration
    output_dir: str = field(default="./arabic_lora_checkpoints")
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=8)
    learning_rate: float = field(default=1e-4)
    weight_decay: float = field(default=0.01)
    warmup_steps: int = field(default=100)
    max_grad_norm: float = field(default=1.0)
    
    # Data splits
    train_ratio: float = field(default=0.8)
    val_ratio: float = field(default=0.1)
    max_audio_length: float = field(default=30.0)
    
    # Optimization
    use_gradient_checkpointing: bool = field(default=True)
    use_mixed_precision: bool = field(default=True)
    dataloader_num_workers: int = field(default=0)
    
    # Logging and saving
    logging_steps: int = field(default=10)
    save_steps: int = field(default=500)
    eval_steps: int = field(default=500)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_loss")
    greater_is_better: bool = field(default=False)
    
    # Early stopping
    early_stopping_patience: int = field(default=3)
    early_stopping_threshold: float = field(default=0.001)
    
    # Device configuration
    device: str = field(default="auto")
    local_rank: int = field(default=-1)
    
    # Validation
    validate_files: bool = field(default=True)
    normalize_text: bool = field(default=True)


class ArabicLoRATrainer:
    """Trainer class for Arabic LoRA fine-tuning."""
    
    def __init__(self, args: LoRATrainingArguments):
        self.args = args
        self.device = self._setup_device()
        
        # Initialize components
        self.tokenizer = None
        self.audio_tokenizer = None
        self.model = None
        self.collator = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        logger.info(f"Arabic LoRA Trainer initialized on device: {self.device}")
    
    def _setup_device(self) -> str:
        """Setup compute device."""
        if self.args.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = self.args.device
        
        logger.info(f"Using device: {device}")
        return device
    
    def setup_tokenizers(self):
        """Load and setup tokenizers."""
        logger.info("Loading tokenizers...")
        
        # Load text tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_path,
            trust_remote_code=True
        )
        
        # Load audio tokenizer
        audio_device = "cpu" if self.device == "mps" else self.device
        self.audio_tokenizer = load_higgs_audio_tokenizer(
            self.args.audio_tokenizer_path,
            device=audio_device
        )
        
        logger.info("Tokenizers loaded successfully")
    
    def setup_datasets(self):
        """Setup training datasets."""
        logger.info("Setting up datasets...")
        
        # Create datasets
        self.train_dataset, self.val_dataset, self.test_dataset = create_arabic_chatml_datasets(
            chatml_file=self.args.chatml_file,
            audio_tokenizer=self.audio_tokenizer,
            text_tokenizer=self.tokenizer,
            train_ratio=self.args.train_ratio,
            val_ratio=self.args.val_ratio,
            audio_base_path=self.args.audio_base_path,
            max_audio_length=self.args.max_audio_length,
            normalize_text=self.args.normalize_text,
            validate_files=self.args.validate_files,
        )
        
        logger.info(f"Dataset sizes - Train: {len(self.train_dataset)}, "
                   f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
        
        # Print speaker statistics
        train_speakers = self.train_dataset.get_speaker_stats()
        logger.info(f"Training speakers: {train_speakers}")
    
    def setup_model(self):
        """Setup model with LoRA configuration."""
        logger.info("Setting up model...")
        
        # Load base model
        logger.info(f"Loading base model from {self.args.model_path}")
        self.model = HiggsAudioModel.from_pretrained(
            self.args.model_path,
            torch_dtype=torch.bfloat16 if self.args.use_mixed_precision else torch.float32,
            device_map=self.device if self.device != "auto" else None,
        )
        
        # Prepare model for training
        if self.args.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Configure LoRA target modules for Higgs Audio
        target_modules = self._get_lora_target_modules()
        
        # Create LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # Higgs Audio is based on causal LM
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            bias=self.args.lora_bias,
            target_modules=target_modules,
            inference_mode=False,
        )
        
        logger.info(f"LoRA config: {lora_config}")
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        logger.info("Model setup complete")
    
    def _get_lora_target_modules(self) -> list:
        """Get target modules for LoRA based on Higgs Audio architecture."""
        target_modules = [
            # Text model attention layers
            "language_model.model.layers.*.self_attn.q_proj",
            "language_model.model.layers.*.self_attn.k_proj", 
            "language_model.model.layers.*.self_attn.v_proj",
            "language_model.model.layers.*.self_attn.o_proj",
            
            # Text model FFN layers
            "language_model.model.layers.*.mlp.gate_proj",
            "language_model.model.layers.*.mlp.up_proj",
            "language_model.model.layers.*.mlp.down_proj",
        ]
        
        # Add audio-specific layers if using dual FFN
        config = AutoConfig.from_pretrained(self.args.model_path)
        if hasattr(config, 'audio_adapter_type') and config.audio_adapter_type == "dual_ffn":
            if hasattr(config, 'audio_dual_ffn_layers') and config.audio_dual_ffn_layers:
                for layer_idx in config.audio_dual_ffn_layers:
                    target_modules.extend([
                        f"language_model.model.layers.{layer_idx}.audio_mlp.gate_proj",
                        f"language_model.model.layers.{layer_idx}.audio_mlp.up_proj",
                        f"language_model.model.layers.{layer_idx}.audio_mlp.down_proj",
                    ])
        
        # Audio projection layers
        target_modules.extend([
            "audio_projector.linear",
            "audio_decoder_projector.*",
        ])
        
        logger.info(f"LoRA target modules: {target_modules}")
        return target_modules
    
    def setup_collator(self):
        """Setup data collator."""
        logger.info("Setting up data collator...")
        
        config = AutoConfig.from_pretrained(self.args.model_path)
        
        self.collator = HiggsAudioSampleCollator(
            whisper_processor=None,
            audio_in_token_id=config.audio_in_token_idx,
            audio_out_token_id=config.audio_out_token_idx,
            audio_stream_bos_id=config.audio_stream_bos_id,
            audio_stream_eos_id=config.audio_stream_eos_id,
            encode_whisper_embed=config.encode_whisper_embed,
            pad_token_id=config.pad_token_id,
            return_audio_in_tokens=config.encode_audio_in_tokens,
            use_delay_pattern=config.use_delay_pattern,
            round_to=8,  # Round to multiple of 8 for efficiency
            audio_num_codebooks=config.audio_num_codebooks,
        )
        
        logger.info("Data collator setup complete")
    
    def train(self):
        """Run LoRA training."""
        logger.info("Starting LoRA training...")
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            num_train_epochs=self.args.num_train_epochs,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            warmup_steps=self.args.warmup_steps,
            max_grad_norm=self.args.max_grad_norm,
            
            # Mixed precision
            fp16=self.args.use_mixed_precision and self.device == "cuda",
            bf16=self.args.use_mixed_precision and self.device == "cuda",
            
            # Logging and evaluation
            logging_steps=self.args.logging_steps,
            eval_steps=self.args.eval_steps,
            evaluation_strategy="steps",
            save_steps=self.args.save_steps,
            save_strategy="steps",
            save_total_limit=self.args.save_total_limit,
            
            # Best model selection
            load_best_model_at_end=self.args.load_best_model_at_end,
            metric_for_best_model=self.args.metric_for_best_model,
            greater_is_better=self.args.greater_is_better,
            
            # Data loading
            dataloader_num_workers=self.args.dataloader_num_workers,
            dataloader_pin_memory=True,
            
            # Optimization
            gradient_checkpointing=self.args.use_gradient_checkpointing,
            
            # Reporting
            report_to="none",  # Disable wandb/tensorboard for now
            
            # Remove deprecated
            remove_unused_columns=False,
        )
        
        # Create callbacks
        callbacks = []
        if self.args.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.args.early_stopping_patience,
                    early_stopping_threshold=self.args.early_stopping_threshold
                )
            )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self.collator,
            tokenizer=self.tokenizer,
            callbacks=callbacks,
        )
        
        # Start training
        logger.info("Beginning training...")
        train_result = trainer.train()
        
        # Save final model
        logger.info("Saving final model...")
        trainer.save_model()
        
        # Save training metrics
        metrics_file = os.path.join(self.args.output_dir, "training_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        
        logger.info(f"Training complete! Model saved to {self.args.output_dir}")
        return train_result
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the trained model."""
        logger.info("Evaluating trained model...")
        
        # Create trainer for evaluation
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,
            dataloader_num_workers=self.args.dataloader_num_workers,
            remove_unused_columns=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            eval_dataset=self.test_dataset,
            data_collator=self.collator,
            tokenizer=self.tokenizer,
        )
        
        # Run evaluation
        eval_results = trainer.evaluate()
        
        # Save evaluation results
        eval_file = os.path.join(self.args.output_dir, "evaluation_results.json")
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        logger.info(f"Evaluation results: {eval_results}")
        return eval_results
    
    def run_full_training(self):
        """Run the complete training pipeline."""
        try:
            # Setup all components
            self.setup_tokenizers()
            self.setup_datasets()
            self.setup_model()
            self.setup_collator()
            
            # Run training
            train_result = self.train()
            
            # Run evaluation
            eval_result = self.evaluate()
            
            logger.info("Training pipeline completed successfully!")
            return train_result, eval_result
            
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Arabic LoRA Fine-tuning for Higgs Audio")
    
    # Required arguments
    parser.add_argument("--chatml_file", type=str, required=True,
                       help="Path to Arabic ChatML JSON file")
    
    # Model configuration
    parser.add_argument("--model_path", type=str, 
                       default="bosonai/higgs-audio-v2-generation-3B-base",
                       help="Path to base Higgs Audio model")
    parser.add_argument("--audio_tokenizer_path", type=str,
                       default="bosonai/higgs-audio-v2-tokenizer", 
                       help="Path to audio tokenizer")
    
    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    
    # Training configuration
    parser.add_argument("--output_dir", type=str, default="./arabic_lora_checkpoints",
                       help="Output directory for checkpoints")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, 
                       help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    
    # Data configuration
    parser.add_argument("--audio_base_path", type=str, default=None,
                       help="Base path for audio files")
    parser.add_argument("--max_audio_length", type=float, default=30.0,
                       help="Maximum audio length in seconds")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training data ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation data ratio")
    
    # Device configuration
    parser.add_argument("--device", type=str, default="auto", 
                       choices=["auto", "cuda", "mps", "cpu"], help="Device to use")
    
    # Other options
    parser.add_argument("--no_mixed_precision", action="store_true",
                       help="Disable mixed precision training")
    parser.add_argument("--no_gradient_checkpointing", action="store_true", 
                       help="Disable gradient checkpointing")
    parser.add_argument("--no_validate_files", action="store_true",
                       help="Skip audio file validation")
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Convert to dataclass
    training_args = LoRATrainingArguments(
        chatml_file=args.chatml_file,
        audio_base_path=args.audio_base_path,
        model_path=args.model_path,
        audio_tokenizer_path=args.audio_tokenizer_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_audio_length=args.max_audio_length,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        device=args.device,
        use_mixed_precision=not args.no_mixed_precision,
        use_gradient_checkpointing=not args.no_gradient_checkpointing,
        validate_files=not args.no_validate_files,
    )
    
    # Create output directory
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Save training configuration
    config_file = os.path.join(training_args.output_dir, "training_config.json")
    with open(config_file, 'w') as f:
        json.dump(training_args.__dict__, f, indent=2)
    
    # Initialize trainer
    trainer = ArabicLoRATrainer(training_args)
    
    # Run training
    train_result, eval_result = trainer.run_full_training()
    
    logger.info("Arabic LoRA fine-tuning completed successfully!")


if __name__ == "__main__":
    main()