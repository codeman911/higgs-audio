#!/usr/bin/env python3
"""Script to inspect available modules in HiggsAudioModel for LoRA targeting."""

import sys
sys.path.append('.')

from boson_multimodal.model.higgs_audio.modeling_higgs_audio import HiggsAudioModel
from transformers import AutoConfig
import torch

# Load model config
config = AutoConfig.from_pretrained(
    "bosonai/higgs-audio-v2",
    trust_remote_code=True
)

# Initialize model
model = HiggsAudioModel.from_pretrained(
    "bosonai/higgs-audio-v2",
    config=config,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

print("Model structure and available modules for LoRA targeting:\n")
print("="*80)

# Print all named modules
all_modules = []
for name, module in model.named_modules():
    if name:  # Skip the root module
        module_type = type(module).__name__
        all_modules.append((name, module_type))

# Sort by name for easier reading
all_modules.sort(key=lambda x: x[0])

# Group by type
from collections import defaultdict
modules_by_type = defaultdict(list)
for name, module_type in all_modules:
    modules_by_type[module_type].append(name)

# Print Linear layers (most common target for LoRA)
print("\nLinear layers (potential LoRA targets):")
print("-"*40)
for name in sorted(modules_by_type.get('Linear', [])):
    print(f"  {name}")

# Print attention-related modules
print("\nAttention-related modules:")
print("-"*40)
attention_modules = []
for name, module_type in all_modules:
    if any(keyword in name.lower() for keyword in ['attention', 'attn', 'q_proj', 'k_proj', 'v_proj', 'o_proj']):
        attention_modules.append(name)
for name in sorted(set(attention_modules)):
    print(f"  {name}")

# Print audio-related modules
print("\nAudio-specific modules:")
print("-"*40)
audio_modules = []
for name, module_type in all_modules:
    if 'audio' in name.lower():
        audio_modules.append(name)
for name in sorted(set(audio_modules)):
    print(f"  {name}")

# Check for common projection layer patterns
print("\nProjection layers:")
print("-"*40)
projection_modules = []
for name, module_type in all_modules:
    if any(keyword in name.lower() for keyword in ['proj', 'projection', 'projector', 'mlp']):
        if module_type == 'Linear':
            projection_modules.append(name)
for name in sorted(set(projection_modules)):
    print(f"  {name}")

# Print recommended LoRA targets
print("\n" + "="*80)
print("RECOMMENDED LoRA TARGET MODULES:")
print("="*80)

# Look for audio encoder/decoder specific layers
recommended = []
for name, module_type in all_modules:
    if module_type == 'Linear':
        # Focus on audio-related layers
        if 'audio' in name.lower():
            recommended.append(name)
        # Also include MLP/projection layers in decoder
        elif any(kw in name for kw in ['gate_proj', 'up_proj', 'down_proj']):
            # But only from certain layers (customize as needed)
            if 'decoder_layers' in name:
                layer_num = int(name.split('.')[1]) if name.startswith('decoder_layers.') else -1
                if layer_num >= 0:  # Can filter to specific layers if needed
                    recommended.append(name)

# Remove duplicates and sort
recommended = sorted(set(recommended))
if recommended:
    for name in recommended[:20]:  # Limit output for readability
        print(f"  '{name}',")
else:
    print("  No audio-specific linear layers found. Defaulting to standard attention layers.")
    
print("\n" + "="*80)
