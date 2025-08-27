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