"""
Minimal LoRA configuration targeting DualFFN architecture.
Uses PEFT library with precise module targeting.
"""

import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer


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
    
    return list(set(target_modules))  # Remove duplicates


def create_lora_config(r: int = 16, 
                      lora_alpha: int = 32, 
                      lora_dropout: float = 0.05,
                      target_modules: list = None):
    """Create LoRA configuration for Higgs Audio DualFFN."""
    
    if target_modules is None:
        # Default targeting - will be resolved dynamically
        target_modules = [
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
            "audio_mlp.gate_proj", "audio_mlp.up_proj", "audio_mlp.down_proj"
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


def apply_lora(model, lora_config=None):
    """Apply LoRA to model with proper module targeting."""
    
    if lora_config is None:
        # Dynamically discover target modules
        discovered_modules = get_target_modules(model)
        lora_config = create_lora_config(target_modules=discovered_modules)
    
    # Apply LoRA
    lora_model = get_peft_model(model, lora_config)
    
    return lora_model


def save_lora_adapters(model, output_dir: str):
    """Save only LoRA adapters."""
    model.save_pretrained(output_dir)


def load_lora_adapters(base_model, adapter_path: str):
    """Load LoRA adapters onto base model."""
    from peft import PeftModel
    return PeftModel.from_pretrained(base_model, adapter_path)