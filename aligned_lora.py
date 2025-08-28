"""
Optimized LoRA configuration for Higgs Audio training.
Uses PEFT library with precise module targeting similar to train-higgs-audio.
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)


def get_target_modules(model):
    """Dynamically discover target modules from actual model structure."""
    target_modules = []
    
    # Scan model for attention and MLP modules
    for name, module in model.named_modules():
        # Target standard attention projections
        if any(proj in name for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            target_modules.append(name)
        
        # Target standard MLP layers
        if any(mlp in name for mlp in ["gate_proj", "up_proj", "down_proj"]):
            target_modules.append(name)
        
        # Target audio-specific DualFFN modules
        if "audio_mlp" in name and any(proj in name for proj in ["gate_proj", "up_proj", "down_proj"]):
            target_modules.append(name)
            
        # Target audio attention modules (CRITICAL FIX: Add audio attention targeting)
        if "audio_attn" in name and any(proj in name for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            target_modules.append(name)
    
    return list(set(target_modules))  # Remove duplicates


def create_aligned_lora_config(r: int = 16, 
                              lora_alpha: int = 32, 
                              lora_dropout: float = 0.05,
                              target_modules: list = None):
    """Create LoRA configuration for Higgs Audio DualFFN."""
    
    if target_modules is None:
        # Default targeting - will be resolved dynamically
        target_modules = [
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
            "audio_mlp.gate_proj", "audio_mlp.up_proj", "audio_mlp.down_proj",
            # CRITICAL FIX: Add audio attention targeting for cross-modal learning
            "audio_attn.q_proj", "audio_attn.k_proj", "audio_attn.v_proj", "audio_attn.o_proj"
        ]
    
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False
    )


def apply_aligned_lora(model, lora_config=None):
    """Apply LoRA to model with proper module targeting."""
    
    if lora_config is None:
        # Dynamically discover target modules
        discovered_modules = get_target_modules(model)
        lora_config = create_aligned_lora_config(target_modules=discovered_modules)
    
    # Apply LoRA directly to the model (our wrapper in this case)
    # This is a simpler approach that should work
    lora_model = get_peft_model(model, lora_config)
    return lora_model


def save_aligned_lora_adapters(model, output_dir: str):
    """Save only LoRA adapters with enhanced error handling."""
    try:
        import os
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"Saving LoRA adapters to {output_dir}")
        # Check if output directory exists and is writable
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created directory: {output_dir}")
            
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"No write permission for directory: {output_dir}")
            
        # Save the model
        logger.info("Calling model.save_pretrained...")
        model.save_pretrained(output_dir)
        logger.info(f"Successfully saved LoRA adapters to {output_dir}")
        
        # Verify files were created
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            logger.info(f"Saved files: {files}")
            if not files:
                logger.warning(f"No files were saved to {output_dir}")
        else:
            logger.error(f"Directory was not created: {output_dir}")
            
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to save LoRA adapters to {output_dir}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def load_aligned_lora_adapters(base_model, adapter_path: str):
    """Load LoRA adapters onto base model."""
    from peft import PeftModel
    return PeftModel.from_pretrained(base_model, adapter_path)


class HiggsAudioModelWrapper(torch.nn.Module):
    """Wrapper for Higgs Audio model to enable training with optimized loss computation"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config
    
    @property
    def device(self):
        return self.model.device
          
    def forward(self, **kwargs):
        # Check if HIGGS_AVAILABLE pattern should be used
        try:
            from boson_multimodal.model.higgs_audio import HiggsAudioModel
            HIGGS_AVAILABLE = True
        except ImportError:
            HIGGS_AVAILABLE = False
        
        # Filter out parameters that the Higgs Audio model doesn't expect
        # Hugging Face Trainer may pass a "labels" parameter that we need to filter out
        model_kwargs = {}
        for key, value in kwargs.items():
            # Skip the "labels" parameter that Hugging Face Trainer might add
            if key == "labels":
                continue
            model_kwargs[key] = value
        
        # Match train-higgs-audio approach as closely as possible
        if HIGGS_AVAILABLE:
            # When HIGGS is available, pass filtered kwargs to avoid the labels parameter issue
            # The HiggsAudioModel should be able to handle the label_ids parameter
            if self.model.device != kwargs['input_ids'].device:
                self.model = self.model.to(kwargs['input_ids'].device)
            result = self.model(**model_kwargs)
            
            # Always extract and return just the loss tensor when available
            # This prevents the Trainer from trying to divide a dictionary/object by an integer
            if hasattr(result, 'loss') and result.loss is not None:
                return result.loss
            else:
                return result
        else:
            # Fallback logic similar to train-higgs-audio
            input_ids = model_kwargs.get('input_ids')
            attention_mask = model_kwargs.get('attention_mask')
            labels = model_kwargs.get('label_ids')
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            loss = None
            if labels is not None:
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
            # Always return just the loss when computing loss
            if loss is not None:
                return loss
            else:
                return {"logits": outputs.logits}
    
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped model for PEFT compatibility"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            # Delegate to the wrapped model
            return getattr(self.model, name)
