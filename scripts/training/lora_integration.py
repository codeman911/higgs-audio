#!/usr/bin/env python3
"""
LoRA Integration for Higgs-Audio V2 Model
Integrates PEFT LoRA adapters with the existing Higgs-Audio architecture.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from peft.tuners.lora import LoraLayer
import re
from pathlib import Path
import sys

# Robust import handling for both CLI and module usage
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from boson_multimodal.model.higgs_audio.modeling_higgs_audio import HiggsAudioDualFFNDecoderLayer
    from boson_multimodal.model.higgs_audio import HiggsAudioModel, HiggsAudioConfig
except ImportError:
    # Fallback for different project structures
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    sys.path.insert(0, project_root)
    from boson_multimodal.model.higgs_audio.modeling_higgs_audio import HiggsAudioDualFFNDecoderLayer
    from boson_multimodal.model.higgs_audio import HiggsAudioModel, HiggsAudioConfig


class HiggsAudioLoRAConfig:
    """Configuration for LoRA fine-tuning of Higgs-Audio model"""
    
    def __init__(
        self,
        # LoRA hyperparameters
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        
        # Target modules for LoRA
        target_modules: Optional[List[str]] = None,
        
        # Training strategy
        freeze_base_model: bool = True,
        freeze_audio_tower: bool = True,
        freeze_audio_encoder_proj: bool = False,
        
        # Audio-specific LoRA
        enable_audio_lora: bool = True,
        audio_lora_r: int = 8,
        audio_lora_alpha: int = 16,
        
        # Language-specific settings
        enable_multilingual_lora: bool = True,
        arabic_specific_modules: Optional[List[str]] = None,
    ):
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Default target modules if not specified
        if target_modules is None:
            self.target_modules = [
                # LLaMA attention modules
                "self_attn.q_proj",
                "self_attn.k_proj", 
                "self_attn.v_proj",
                "self_attn.o_proj",
                
                # LLaMA MLP modules
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
                
                # Audio-specific modules
                "audio_mlp.gate_proj",
                "audio_mlp.up_proj", 
                "audio_mlp.down_proj",
                
                # Audio attention (if enabled)
                "audio_attn.q_proj",
                "audio_attn.k_proj",
                "audio_attn.v_proj",
                "audio_attn.o_proj",
                
                # Audio projection layers
                "audio_encoder_proj.linear",
                "audio_decoder_proj.audio_lm_head",
            ]
        else:
            self.target_modules = target_modules
        
        self.freeze_base_model = freeze_base_model
        self.freeze_audio_tower = freeze_audio_tower
        self.freeze_audio_encoder_proj = freeze_audio_encoder_proj
        
        self.enable_audio_lora = enable_audio_lora
        self.audio_lora_r = audio_lora_r
        self.audio_lora_alpha = audio_lora_alpha
        
        self.enable_multilingual_lora = enable_multilingual_lora
        self.arabic_specific_modules = arabic_specific_modules or []


class HiggsAudioLoRAWrapper:
    """Wrapper class for applying LoRA to Higgs-Audio model"""
    
    def __init__(self, model: HiggsAudioModel, config: HiggsAudioLoRAConfig):
        self.base_model = model
        self.lora_config = config
        self.peft_model = None
        
    def apply_lora(self) -> PeftModel:
        """Apply LoRA adapters to the model"""
        
        # Step 1: Freeze base model components as specified
        self._freeze_base_components()
        
        # Step 2: Create LoRA configuration
        peft_config = self._create_peft_config()
        
        # Step 3: Apply LoRA to the model
        self.peft_model = get_peft_model(self.base_model, peft_config)
        
        # Step 4: Apply custom LoRA configurations for audio modules
        if self.lora_config.enable_audio_lora:
            self._apply_audio_specific_lora()
        
        # Step 5: Setup multilingual LoRA if enabled
        if self.lora_config.enable_multilingual_lora:
            self._setup_multilingual_lora()
        
        return self.peft_model
    
    def _freeze_base_components(self):
        """Freeze specified components of the base model"""
        
        if self.lora_config.freeze_base_model:
            # Freeze LLM backbone
            self.base_model.freeze_llm(freeze_embed=True)
        
        if self.lora_config.freeze_audio_tower:
            # Freeze audio tower (Whisper encoder)
            self.base_model.freeze_audio_tower()
        
        if self.lora_config.freeze_audio_encoder_proj:
            # Freeze audio encoder projection
            self.base_model.freeze_audio_encoder_proj()
    
    def _create_peft_config(self) -> LoraConfig:
        """Create PEFT LoRA configuration"""
        
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_config.lora_r,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            target_modules=self.lora_config.target_modules,
            bias="none",
            fan_in_fan_out=False,
            init_lora_weights=True,
        )
    
    def _apply_audio_specific_lora(self):
        """Apply specialized LoRA configuration for audio modules"""
        
        # Find audio-specific modules and apply different LoRA settings
        for name, module in self.peft_model.named_modules():
            if any(audio_pattern in name for audio_pattern in ["audio_mlp", "audio_attn", "audio_decoder_proj"]):
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    # Adjust LoRA rank for audio modules
                    if hasattr(module, 'r'):
                        module.r = self.lora_config.audio_lora_r
                    if hasattr(module, 'lora_alpha'):
                        module.lora_alpha = self.lora_config.audio_lora_alpha
    
    def _setup_multilingual_lora(self):
        """Setup language-specific LoRA adapters"""
        
        # This is a placeholder for future multilingual LoRA implementation
        # Could involve language-specific adapter routing
        pass
    
    def get_trainable_parameters(self) -> Dict[str, int]:
        """Get count of trainable parameters"""
        
        if self.peft_model is None:
            raise ValueError("LoRA not applied yet. Call apply_lora() first.")
        
        total_params = 0
        trainable_params = 0
        
        for param in self.peft_model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": (trainable_params / total_params) * 100
        }
    
    def save_lora_adapters(self, save_path: str):
        """Save only the LoRA adapters"""
        
        if self.peft_model is None:
            raise ValueError("LoRA not applied yet. Call apply_lora() first.")
        
        self.peft_model.save_pretrained(save_path)
    
    def load_lora_adapters(self, adapter_path: str):
        """Load LoRA adapters"""
        
        self.peft_model = PeftModel.from_pretrained(self.base_model, adapter_path)
        return self.peft_model


class HiggsAudioLoRATrainer:
    """Trainer class for LoRA fine-tuning"""
    
    def __init__(
        self,
        model: HiggsAudioModel,
        lora_config: HiggsAudioLoRAConfig,
        tokenizer,
        audio_tokenizer,
    ):
        self.model = model
        self.lora_config = lora_config
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        
        # Apply LoRA
        self.lora_wrapper = HiggsAudioLoRAWrapper(model, lora_config)
        self.peft_model = self.lora_wrapper.apply_lora()
        
    def prepare_for_training(self):
        """Prepare model for training"""
        
        # Try to enable gradient checkpointing for memory efficiency (optional)
        gc_enable = getattr(self.peft_model, "gradient_checkpointing_enable", None)
        if callable(gc_enable):
            try:
                gc_enable()
            except Exception as e:
                print(f"[WARN] Gradient checkpointing not enabled: {e}. Continuing without it.")
        else:
            print("[WARN] Gradient checkpointing API not found on model. Continuing without it.")
         
        # Print trainable parameters
        param_stats = self.lora_wrapper.get_trainable_parameters()
        print(f"Trainable parameters: {param_stats['trainable_parameters']:,}")
        print(f"Total parameters: {param_stats['total_parameters']:,}")
        print(f"Trainable percentage: {param_stats['trainable_percentage']:.2f}%")
        
        return self.peft_model
    
    def compute_loss(self, batch, outputs):
        """
        FIXED: Compute loss only on target audio tokens, not reference audio tokens.
        Reference audio tokens are used for conditioning (like inference).
        Target audio tokens are used for prediction labels.
        """
        try:
            # Extract logits and labels
            text_logits = outputs.logits
            audio_logits = outputs.audio_logits
            
            text_labels = batch.get('labels')
            # CRITICAL FIX: Get target audio tokens from correct batch key
            audio_labels = batch.get('label_audio_ids')  # Target audio tokens
            if audio_labels is None:
                # Fallback to check other possible keys
                audio_labels = batch.get('audio_labels')
                if audio_labels is None:
                    audio_labels = batch.get('audio_out_ids')  # Last resort
            
            text_loss = 0.0
            audio_loss = 0.0
            
            # Debug logging
            if hasattr(self, 'step_count'):
                self.step_count += 1
            else:
                self.step_count = 1
                
            if self.step_count % 20 == 0:
                print(f"DEBUG Step {self.step_count}:")
                print(f"  text_labels shape: {text_labels.shape if text_labels is not None else 'None'}")
                print(f"  audio_labels shape: {audio_labels.shape if audio_labels is not None else 'None'}")
                print(f"  audio_logits shape: {audio_logits.shape if audio_logits is not None else 'None'}")
                
                # CRITICAL DEBUG: Check reference audio processing
                reference_audio = batch.get('audio_in_ids')
                if reference_audio is not None:
                    print(f"  REFERENCE AUDIO: {reference_audio.shape} tokens for conditioning")
                else:
                    print(f"  MISSING REFERENCE AUDIO: No audio_in_ids found!")
                
                # Check if audio_ids_concat is present (should map to audio_in_ids)
                audio_ids_concat = batch.get('audio_ids_concat')
                if audio_ids_concat is not None:
                    print(f"  AUDIO_IDS_CONCAT: {audio_ids_concat.shape} (should become audio_in_ids)")
                else:
                    print(f"  MISSING AUDIO_IDS_CONCAT: No reference audio in batch!")
            
            # CRITICAL: Add NaN/Inf detection for model outputs
            if text_logits is not None:
                if torch.isnan(text_logits).any() or torch.isinf(text_logits).any():
                    print(f"WARNING: Invalid text_logits detected!")
                    return {
                        'text_loss': torch.tensor(1e-6, device=text_logits.device, requires_grad=True),
                        'audio_loss': torch.tensor(0.0, device=text_logits.device, requires_grad=True),
                        'combined_loss': torch.tensor(1e-6, device=text_logits.device, requires_grad=True)
                    }
            
            if audio_logits is not None:
                if torch.isnan(audio_logits).any() or torch.isinf(audio_logits).any():
                    print(f"WARNING: Invalid audio_logits detected!")
                    return {
                        'text_loss': torch.tensor(0.0, device=audio_logits.device, requires_grad=True),
                        'audio_loss': torch.tensor(1e-6, device=audio_logits.device, requires_grad=True),
                        'combined_loss': torch.tensor(1e-6, device=audio_logits.device, requires_grad=True)
                    }
            
            # Text generation loss
            if text_labels is not None and text_logits is not None:
                batch_size = text_labels.shape[0]
                label_seq_len = text_labels.shape[1]
                logit_seq_len = text_logits.shape[1]
                
                if logit_seq_len != label_seq_len:
                    text_logits = text_logits[:, :label_seq_len, :]
                
                text_logits_flat = text_logits.contiguous().view(-1, text_logits.size(-1))
                text_labels_flat = text_labels.contiguous().view(-1)
                
                if text_logits_flat.shape[0] != text_labels_flat.shape[0]:
                    min_len = min(text_logits_flat.shape[0], text_labels_flat.shape[0])
                    text_logits_flat = text_logits_flat[:min_len]
                    text_labels_flat = text_labels_flat[:min_len]
                
                # Check for valid tokens
                valid_mask = text_labels_flat != -100
                if valid_mask.sum() > 0:
                    text_loss = nn.functional.cross_entropy(
                        text_logits_flat,
                        text_labels_flat,
                        ignore_index=-100,
                        reduction='mean'
                    )
                    
                    if torch.isnan(text_loss) or torch.isinf(text_loss):
                        print(f"WARNING: Invalid text_loss detected")
                        text_loss = torch.tensor(0.0, device=text_loss.device, requires_grad=True)
            
            # CRITICAL FIX: Audio generation loss (multi-codebook) - ONLY on target audio
            if audio_labels is not None and audio_logits is not None and audio_labels.numel() > 0:
                if self.step_count % 20 == 0:
                    print(f"  Processing audio loss with {audio_labels.numel()} audio label tokens")
                
                valid_codebook_count = 0
                
                # Handle tensor dimensions - audio_labels should be (num_codebooks, seq_len)
                if audio_labels.dim() == 2:
                    # Already correct shape (num_codebooks, seq_len)
                    pass
                elif audio_labels.dim() == 3:
                    # Remove batch dimension if present (batch_size, num_codebooks, seq_len)
                    if audio_labels.shape[0] == 1:
                        audio_labels = audio_labels.squeeze(0)
                    else:
                        print(f"WARNING: Unexpected audio_labels batch dimension: {audio_labels.shape}")
                        # Take first batch item
                        audio_labels = audio_labels[0]
                else:
                    print(f"WARNING: Unexpected audio_labels shape: {audio_labels.shape}")
                    return {
                        'text_loss': text_loss,
                        'audio_loss': torch.tensor(1e-6, device=audio_labels.device, requires_grad=True),
                        'combined_loss': text_loss + torch.tensor(1e-6, device=audio_labels.device, requires_grad=True)
                    }
                
                # Handle audio_logits dimensions
                if audio_logits is not None:
                    if audio_logits.dim() == 3:
                        # CRITICAL FIX: Model outputs (seq_len, num_codebooks, vocab_size)
                        # But we need (num_codebooks, seq_len, vocab_size)
                        if audio_logits.shape[1] == 8 and audio_logits.shape[2] > 1000:
                            # Transpose from (seq_len, num_codebooks, vocab_size) to (num_codebooks, seq_len, vocab_size)
                            audio_logits = audio_logits.transpose(0, 1)
                            if self.step_count % 20 == 0:
                                print(f"  FIXED: Transposed audio_logits from {audio_logits.transpose(0, 1).shape} to {audio_logits.shape}")
                    elif audio_logits.dim() == 4:
                        # (batch_size, seq_len, num_codebooks, vocab_size) or similar
                        if audio_logits.shape[0] == 1:
                            audio_logits = audio_logits.squeeze(0)
                            # Apply same transpose fix
                            if audio_logits.shape[1] == 8 and audio_logits.shape[2] > 1000:
                                audio_logits = audio_logits.transpose(0, 1)
                        else:
                            audio_logits = audio_logits[0]  # Take first batch
                            # Apply same transpose fix
                            if audio_logits.shape[1] == 8 and audio_logits.shape[2] > 1000:
                                audio_logits = audio_logits.transpose(0, 1)
                    else:
                        print(f"WARNING: Unexpected audio_logits shape: {audio_logits.shape}")
                        return {
                            'text_loss': text_loss,
                            'audio_loss': torch.tensor(1e-6, device=audio_logits.device, requires_grad=True),
                            'combined_loss': text_loss + torch.tensor(1e-6, device=audio_logits.device, requires_grad=True)
                        }
                
                num_codebooks = min(audio_labels.shape[0], audio_logits.shape[0])
                
                for codebook_idx in range(num_codebooks):
                    try:
                        codebook_logits = audio_logits[codebook_idx]  # (seq_len, vocab_size)
                        codebook_labels = audio_labels[codebook_idx]  # (seq_len,)
                        
                        if codebook_logits.shape[0] != codebook_labels.shape[0]:
                            min_seq_len = min(codebook_logits.shape[0], codebook_labels.shape[0])
                            codebook_logits = codebook_logits[:min_seq_len]
                            codebook_labels = codebook_labels[:min_seq_len]
                        
                        valid_mask = codebook_labels != -100
                        if valid_mask.sum() > 0:
                            codebook_loss = nn.functional.cross_entropy(
                                codebook_logits,
                                codebook_labels,
                                ignore_index=-100,
                                reduction='mean'
                            )
                            
                            if torch.isnan(codebook_loss) or torch.isinf(codebook_loss):
                                print(f"WARNING: Invalid codebook_loss for codebook {codebook_idx}")
                                continue
                            
                            audio_loss += codebook_loss
                            valid_codebook_count += 1
                            
                    except Exception as e:
                        print(f"ERROR: Failed to compute loss for codebook {codebook_idx}: {e}")
                        continue
                
                if valid_codebook_count > 0:
                    audio_loss = audio_loss / valid_codebook_count
                    if self.step_count % 20 == 0:
                        print(f"  Computed audio_loss: {audio_loss.item():.4f} from {valid_codebook_count} codebooks")
                else:
                    if self.step_count % 20 == 0:
                        print(f"  No valid audio tokens found for loss computation")
            else:
                if self.step_count % 20 == 0:
                    print(f"  No audio labels or logits available for loss computation")
        
            # Combine losses with adaptive weighting to prevent one modality from dominating
            if text_loss > 0 and audio_loss > 0:
                # Normalize losses to similar scales before combining
                text_weight = 1.0
                audio_weight = min(text_loss.item() / max(audio_loss.item(), 1e-6), 2.0)  # Cap at 2.0
                combined_loss = text_loss * text_weight + audio_loss * audio_weight
            else:
                # Fallback for edge cases
                combined_loss = text_loss + audio_loss
        
            # Final validation
            if torch.isnan(combined_loss) or torch.isinf(combined_loss):
                print(f"CRITICAL: Final combined_loss is invalid: {combined_loss}")
                combined_loss = torch.tensor(1e-6, requires_grad=True)
        
            return {
                'text_loss': text_loss,
                'audio_loss': audio_loss,
                'combined_loss': combined_loss
            }
            
        except Exception as e:
            print(f"ERROR: Exception in compute_loss: {e}")
            # Return safe fallback losses
            device = next(iter(batch.values())).device if batch else 'cpu'
            return {
                'text_loss': torch.tensor(1e-6, device=device, requires_grad=True),
                'audio_loss': torch.tensor(1e-6, device=device, requires_grad=True),
                'combined_loss': torch.tensor(1e-6, device=device, requires_grad=True)
            }
    
    def save_lora_adapters(self, save_path: str):
        """Save only the LoRA adapters"""
        
        if self.peft_model is None:
            raise ValueError("LoRA not applied yet. Call apply_lora() first.")
        
        self.peft_model.save_pretrained(save_path)
    
    def load_lora_adapters(self, adapter_path: str):
        """Load LoRA adapters"""
        
        self.peft_model = PeftModel.from_pretrained(self.base_model, adapter_path)
        return self.peft_model


def create_lora_model(
    model_path: str,
    lora_config: HiggsAudioLoRAConfig,
    device: str = "cuda",
    model_config: Optional[Any] = None
) -> HiggsAudioLoRATrainer:
    """Create a LoRA-enabled Higgs-Audio model"""
    
    # Load base model
    print("Loading base Higgs-Audio model...")
    model = HiggsAudioModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None
    )
    
    # Apply corrected model configuration if provided
    if model_config is not None:
        print(f"Applying corrected model configuration...")
        print(f"  • Original audio_num_codebooks: {model.config.audio_num_codebooks}")
        print(f"  • Corrected audio_num_codebooks: {model_config.audio_num_codebooks}")
        
        # Update the model's configuration to match the corrected config
        model.config.audio_num_codebooks = model_config.audio_num_codebooks
        model.audio_num_codebooks = model_config.audio_num_codebooks
        
        # Update other relevant configuration parameters
        for attr in ['audio_in_token_idx', 'audio_out_token_idx', 'audio_stream_bos_id', 
                     'audio_stream_eos_id', 'use_delay_pattern', 'encode_whisper_embed']:
            if hasattr(model_config, attr):
                setattr(model.config, attr, getattr(model_config, attr))
        
        print(f"Model configuration updated successfully!")
    
    # Load tokenizers
    from transformers import AutoTokenizer
    from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    audio_tokenizer = load_higgs_audio_tokenizer("bosonai/higgs-audio-v2-tokenizer", device=device)
    
    # Create LoRA trainer
    trainer = HiggsAudioLoRATrainer(
        model=model,
        lora_config=lora_config,
        tokenizer=tokenizer,
        audio_tokenizer=audio_tokenizer
    )
    
    return trainer


def main():
    """Example usage"""
    
    # Create LoRA configuration optimized for Arabic+English voice cloning
    lora_config = HiggsAudioLoRAConfig(
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        freeze_base_model=True,
        freeze_audio_tower=True,
        freeze_audio_encoder_proj=False,
        enable_audio_lora=True,
        audio_lora_r=8,
        audio_lora_alpha=16,
        enable_multilingual_lora=True
    )
    
    # Create LoRA model
    trainer = create_lora_model(
        model_path="bosonai/higgs-audio-v2-generation-3B-base",
        lora_config=lora_config,
        device="cuda"
    )
    
    # Prepare for training
    model = trainer.prepare_for_training()
    
    print("LoRA model ready for training!")
    
    return trainer


if __name__ == "__main__":
    main()
