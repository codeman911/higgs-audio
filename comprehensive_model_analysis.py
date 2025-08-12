#!/usr/bin/env python3
"""
🔬 COMPREHENSIVE HIGGS AUDIO MODEL ANALYSIS SCRIPT
Emergency diagnostic tool to analyze model architecture for perfect LoRA training
This script logs EVERY detail needed to fix the audio loss plateau issue.
"""

import os
import sys
import json
import torch
import torch.nn as nn
from collections import defaultdict, OrderedDict
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Any, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def safe_import():
    """Safely import Higgs Audio modules with fallback"""
    try:
        from boson_multimodal.model.higgs_audio import HiggsAudioModel, HiggsAudioConfig
        from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
        return HiggsAudioModel, HiggsAudioConfig, load_higgs_audio_tokenizer, True
    except ImportError as e:
        print(f"❌ Failed to import Higgs Audio modules: {e}")
        return None, None, None, False

def analyze_model_structure(model: nn.Module) -> Dict[str, Any]:
    """Comprehensive model structure analysis"""
    analysis = {
        "top_level_modules": {},
        "all_named_modules": {},
        "all_named_parameters": {},
        "module_types": defaultdict(list),
        "parameter_counts": {},
        "linear_layers": {},
        "attention_layers": {},
        "audio_specific_modules": {},
        "text_specific_modules": {},
        "shared_modules": {},
        "total_parameters": 0,
        "trainable_parameters": 0,
    }
    
    # 1. Top-level modules
    print("🔍 ANALYZING TOP-LEVEL MODULES...")
    for name, module in model.named_children():
        analysis["top_level_modules"][name] = {
            "type": type(module).__name__,
            "has_children": len(list(module.children())) > 0,
            "parameter_count": sum(p.numel() for p in module.parameters())
        }
        print(f"  📁 {name}: {type(module).__name__} ({analysis['top_level_modules'][name]['parameter_count']:,} params)")
    
    # 2. All named modules (deep analysis)
    print("\n🔍 ANALYZING ALL NAMED MODULES...")
    for name, module in model.named_modules():
        module_type = type(module).__name__
        analysis["all_named_modules"][name] = {
            "type": module_type,
            "parameter_count": sum(p.numel() for p in module.parameters() if p.requires_grad),
            "has_bias": hasattr(module, 'bias') and module.bias is not None,
        }
        analysis["module_types"][module_type].append(name)
        
        # Identify specific layer types
        if isinstance(module, nn.Linear):
            analysis["linear_layers"][name] = {
                "in_features": getattr(module, 'in_features', None),
                "out_features": getattr(module, 'out_features', None),
                "has_bias": module.bias is not None,
                "parameter_count": sum(p.numel() for p in module.parameters())
            }
        
        # Look for attention patterns
        if 'attn' in name.lower() or 'attention' in name.lower():
            analysis["attention_layers"][name] = {
                "type": module_type,
                "parameter_count": sum(p.numel() for p in module.parameters())
            }
        
        # Audio-specific detection
        if any(keyword in name.lower() for keyword in ['audio', 'mel', 'whisper', 'codebook', 'acoustic']):
            analysis["audio_specific_modules"][name] = {
                "type": module_type,
                "parameter_count": sum(p.numel() for p in module.parameters())
            }
        
        # Text-specific detection
        if any(keyword in name.lower() for keyword in ['text', 'token', 'embed', 'vocab']):
            analysis["text_specific_modules"][name] = {
                "type": module_type,
                "parameter_count": sum(p.numel() for p in module.parameters())
            }
    
    # 3. All named parameters
    print("\n🔍 ANALYZING ALL NAMED PARAMETERS...")
    for name, param in model.named_parameters():
        analysis["all_named_parameters"][name] = {
            "shape": list(param.shape),
            "dtype": str(param.dtype),
            "requires_grad": param.requires_grad,
            "numel": param.numel(),
            "is_leaf": param.is_leaf,
        }
        analysis["total_parameters"] += param.numel()
        if param.requires_grad:
            analysis["trainable_parameters"] += param.numel()
    
    # 4. Parameter count by module type
    for module_type, module_names in analysis["module_types"].items():
        total_params = sum(
            analysis["all_named_modules"][name]["parameter_count"] 
            for name in module_names
        )
        analysis["parameter_counts"][module_type] = {
            "count": len(module_names),
            "total_parameters": total_params
        }
    
    return analysis

def find_potential_lora_targets(analysis: Dict[str, Any]) -> Dict[str, List[str]]:
    """Find potential LoRA target modules based on architecture analysis"""
    targets = {
        "standard_attention": [],
        "linear_layers": [],
        "audio_generation": [],
        "text_processing": [],
        "mlp_layers": [],
        "projection_layers": [],
        "embedding_layers": [],
        "decoder_layers": [],
        "high_parameter_modules": [],
    }
    
    # Standard attention patterns
    for name in analysis["all_named_modules"]:
        if any(pattern in name for pattern in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            targets["standard_attention"].append(name)
        
        if any(pattern in name for pattern in ['gate_proj', 'up_proj', 'down_proj']):
            targets["mlp_layers"].append(name)
        
        if 'proj' in name and 'audio' in name:
            targets["projection_layers"].append(name)
        
        if 'embed' in name:
            targets["embedding_layers"].append(name)
        
        if 'decoder' in name or 'lm_head' in name:
            targets["decoder_layers"].append(name)
        
        # High parameter count modules (potential bottlenecks)
        param_count = analysis["all_named_modules"][name]["parameter_count"]
        if param_count > 1000000:  # > 1M parameters
            targets["high_parameter_modules"].append(name)
    
    # Audio-specific targets
    for name in analysis["audio_specific_modules"]:
        if any(pattern in name for pattern in ['lm_head', 'decoder', 'proj']):
            targets["audio_generation"].append(name)
    
    # Text-specific targets
    for name in analysis["text_specific_modules"]:
        if any(pattern in name for pattern in ['lm_head', 'decoder', 'proj']):
            targets["text_processing"].append(name)
    
    # All linear layers
    targets["linear_layers"] = list(analysis["linear_layers"].keys())
    
    return targets

def analyze_gradient_flow(model: nn.Module) -> Dict[str, Any]:
    """Analyze potential gradient flow issues"""
    gradient_analysis = {
        "frozen_modules": [],
        "trainable_modules": [],
        "gradient_checkpointing": [],
        "potential_bottlenecks": [],
    }
    
    for name, module in model.named_modules():
        # Check if module is frozen
        if hasattr(module, 'requires_grad'):
            if not module.requires_grad:
                gradient_analysis["frozen_modules"].append(name)
        
        # Check parameters
        trainable_params = sum(1 for p in module.parameters() if p.requires_grad)
        total_params = sum(1 for p in module.parameters())
        
        if trainable_params > 0:
            gradient_analysis["trainable_modules"].append({
                "name": name,
                "trainable_params": trainable_params,
                "total_params": total_params,
                "trainable_ratio": trainable_params / total_params if total_params > 0 else 0
            })
        
        # Look for gradient checkpointing
        if hasattr(module, 'gradient_checkpointing'):
            gradient_analysis["gradient_checkpointing"].append(name)
    
    return gradient_analysis

def analyze_audio_pipeline(model: nn.Module) -> Dict[str, Any]:
    """Analyze the audio processing pipeline"""
    audio_pipeline = {
        "audio_tokenizer_modules": [],
        "whisper_modules": [],
        "audio_embedding_modules": [],
        "audio_decoder_modules": [],
        "codebook_modules": [],
        "mel_processing_modules": [],
    }
    
    for name, module in model.named_modules():
        if 'whisper' in name.lower():
            audio_pipeline["whisper_modules"].append(name)
        
        if 'tokenizer' in name.lower() and 'audio' in name.lower():
            audio_pipeline["audio_tokenizer_modules"].append(name)
        
        if 'embed' in name.lower() and 'audio' in name.lower():
            audio_pipeline["audio_embedding_modules"].append(name)
        
        if 'decoder' in name.lower() and 'audio' in name.lower():
            audio_pipeline["audio_decoder_modules"].append(name)
        
        if 'codebook' in name.lower():
            audio_pipeline["codebook_modules"].append(name)
        
        if 'mel' in name.lower():
            audio_pipeline["mel_processing_modules"].append(name)
    
    return audio_pipeline

def check_config_compatibility(config) -> Dict[str, Any]:
    """Check model configuration for LoRA compatibility"""
    config_analysis = {
        "audio_config": {},
        "text_config": {},
        "special_tokens": {},
        "architecture_details": {},
    }
    
    # Extract audio configuration
    for attr in dir(config):
        if 'audio' in attr.lower() and not attr.startswith('_'):
            config_analysis["audio_config"][attr] = getattr(config, attr, None)
        
        if 'text' in attr.lower() and not attr.startswith('_'):
            config_analysis["text_config"][attr] = getattr(config, attr, None)
        
        if 'token' in attr.lower() and not attr.startswith('_'):
            config_analysis["special_tokens"][attr] = getattr(config, attr, None)
    
    # Architecture details
    config_analysis["architecture_details"] = {
        "model_type": getattr(config, 'model_type', None),
        "architectures": getattr(config, 'architectures', None),
        "hidden_size": getattr(config, 'hidden_size', None),
        "num_hidden_layers": getattr(config, 'num_hidden_layers', None),
        "num_attention_heads": getattr(config, 'num_attention_heads', None),
    }
    
    return config_analysis

def generate_lora_recommendations(analysis: Dict[str, Any], targets: Dict[str, List[str]]) -> Dict[str, Any]:
    """Generate LoRA configuration recommendations"""
    recommendations = {
        "primary_targets": [],
        "secondary_targets": [],
        "avoid_targets": [],
        "config_suggestions": {},
        "reasoning": {},
    }
    
    # Primary targets (most important for audio generation)
    audio_generation = targets.get("audio_generation", [])
    decoder_layers = targets.get("decoder_layers", [])
    
    # Focus on audio-specific layers first
    recommendations["primary_targets"] = list(set(audio_generation + decoder_layers))
    
    # Secondary targets (general attention and MLP)
    standard_attention = targets.get("standard_attention", [])
    mlp_layers = targets.get("mlp_layers", [])
    
    recommendations["secondary_targets"] = list(set(standard_attention + mlp_layers))
    
    # Avoid embedding layers (usually frozen)
    recommendations["avoid_targets"] = targets.get("embedding_layers", [])
    
    # Configuration suggestions
    total_params = analysis.get("total_parameters", 0)
    
    if total_params > 10_000_000_000:  # > 10B parameters
        rank = 64
        alpha = 128
    elif total_params > 1_000_000_000:  # > 1B parameters
        rank = 32
        alpha = 64
    else:
        rank = 16
        alpha = 32
    
    recommendations["config_suggestions"] = {
        "rank": rank,
        "alpha": alpha,
        "dropout": 0.1,
        "target_modules": recommendations["primary_targets"][:10],  # Top 10
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }
    
    # Reasoning
    recommendations["reasoning"] = {
        "primary_focus": "Audio generation layers are critical for fixing the audio loss plateau",
        "rank_choice": f"Rank {rank} chosen based on model size ({total_params:,} parameters)",
        "alpha_choice": f"Alpha {alpha} set to 2x rank for stable training",
        "target_selection": "Prioritizing audio-specific modules over general attention"
    }
    
    return recommendations

def save_analysis_report(analysis_data: Dict[str, Any], output_file: str) -> None:
    """Save comprehensive analysis report"""
    print(f"\n💾 SAVING ANALYSIS REPORT TO: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, default=str)
    
    # Also save a human-readable summary
    summary_file = output_file.replace('.json', '_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("🔬 HIGGS AUDIO MODEL ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        # Model overview
        f.write("📊 MODEL OVERVIEW:\n")
        f.write(f"Total Parameters: {analysis_data['model_structure']['total_parameters']:,}\n")
        f.write(f"Trainable Parameters: {analysis_data['model_structure']['trainable_parameters']:,}\n")
        f.write(f"Top-level Modules: {len(analysis_data['model_structure']['top_level_modules'])}\n")
        f.write(f"All Named Modules: {len(analysis_data['model_structure']['all_named_modules'])}\n\n")
        
        # LoRA recommendations
        f.write("🎯 LORA RECOMMENDATIONS:\n")
        recommendations = analysis_data['lora_recommendations']
        f.write(f"Rank: {recommendations['config_suggestions']['rank']}\n")
        f.write(f"Alpha: {recommendations['config_suggestions']['alpha']}\n")
        f.write(f"Primary Targets: {len(recommendations['primary_targets'])}\n")
        f.write(f"Secondary Targets: {len(recommendations['secondary_targets'])}\n\n")
        
        # Audio pipeline
        f.write("🎵 AUDIO PIPELINE MODULES:\n")
        audio_pipeline = analysis_data['audio_pipeline']
        for category, modules in audio_pipeline.items():
            f.write(f"{category}: {len(modules)} modules\n")
        f.write("\n")
        
        # Critical findings
        f.write("🚨 CRITICAL FINDINGS FOR AUDIO LOSS PLATEAU:\n")
        if not recommendations['primary_targets']:
            f.write("❌ NO AUDIO GENERATION MODULES FOUND - This could explain the plateau!\n")
        else:
            f.write(f"✅ Found {len(recommendations['primary_targets'])} audio generation modules\n")
        
        if len(analysis_data['gradient_flow']['frozen_modules']) > 0:
            f.write(f"⚠️  {len(analysis_data['gradient_flow']['frozen_modules'])} frozen modules detected\n")
        
        f.write("\n")
        f.write("💡 NEXT STEPS:\n")
        f.write("1. Use primary_targets for LoRA configuration\n")
        f.write("2. Verify gradient flow to audio generation modules\n")
        f.write("3. Test with recommended rank/alpha values\n")
        f.write("4. Monitor audio loss specifically during training\n")

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Higgs Audio Model Analysis")
    parser.add_argument("--model_path", type=str, 
                       default="/Users/vikram.solanki/Projects/tts/higgs-audio/model_ckpt",
                       help="Path to Higgs Audio model")
    parser.add_argument("--output_dir", type=str, 
                       default="/Users/vikram.solanki/Projects/tts/higgs-audio",
                       help="Output directory for analysis reports")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to load model on")
    
    args = parser.parse_args()
    
    print("🔬 STARTING COMPREHENSIVE HIGGS AUDIO MODEL ANALYSIS")
    print("=" * 60)
    
    # Import modules
    HiggsAudioModel, HiggsAudioConfig, load_higgs_audio_tokenizer, available = safe_import()
    
    if not available:
        print("❌ Higgs Audio modules not available. Exiting.")
        return
    
    try:
        # Load model configuration
        print(f"📁 Loading model from: {args.model_path}")
        config = HiggsAudioConfig.from_pretrained(args.model_path)
        
        # Load model
        model = HiggsAudioModel.from_pretrained(
            config=config,
            pretrained_model_name_or_path=args.model_path,
            torch_dtype=torch.bfloat16,
            device_map=args.device
        )
        
        print(f"✅ Model loaded successfully on {args.device}")
        
        # Comprehensive analysis
        analysis_data = {}
        
        print("\n🔍 STEP 1: MODEL STRUCTURE ANALYSIS")
        analysis_data["model_structure"] = analyze_model_structure(model)
        
        print("\n🔍 STEP 2: FINDING LORA TARGETS")
        analysis_data["lora_targets"] = find_potential_lora_targets(analysis_data["model_structure"])
        
        print("\n🔍 STEP 3: GRADIENT FLOW ANALYSIS")
        analysis_data["gradient_flow"] = analyze_gradient_flow(model)
        
        print("\n🔍 STEP 4: AUDIO PIPELINE ANALYSIS")
        analysis_data["audio_pipeline"] = analyze_audio_pipeline(model)
        
        print("\n🔍 STEP 5: CONFIG COMPATIBILITY CHECK")
        analysis_data["config_analysis"] = check_config_compatibility(config)
        
        print("\n🔍 STEP 6: GENERATING LORA RECOMMENDATIONS")
        analysis_data["lora_recommendations"] = generate_lora_recommendations(
            analysis_data["model_structure"], 
            analysis_data["lora_targets"]
        )
        
        # Save results
        output_file = os.path.join(args.output_dir, "higgs_audio_analysis.json")
        save_analysis_report(analysis_data, output_file)
        
        print("\n✅ ANALYSIS COMPLETE!")
        print(f"📊 Full report saved to: {output_file}")
        print(f"📝 Summary saved to: {output_file.replace('.json', '_summary.txt')}")
        
        # Print critical findings
        print("\n🚨 CRITICAL FINDINGS FOR AUDIO LOSS PLATEAU:")
        recommendations = analysis_data["lora_recommendations"]
        
        if not recommendations["primary_targets"]:
            print("❌ NO AUDIO GENERATION MODULES FOUND!")
            print("   This likely explains the audio loss plateau!")
        else:
            print(f"✅ Found {len(recommendations['primary_targets'])} audio generation modules:")
            for target in recommendations["primary_targets"][:5]:
                print(f"   - {target}")
        
        print(f"\n💡 RECOMMENDED LORA CONFIG:")
        config_suggestions = recommendations["config_suggestions"]
        print(f"   Rank: {config_suggestions['rank']}")
        print(f"   Alpha: {config_suggestions['alpha']}")
        print(f"   Target Modules: {len(config_suggestions['target_modules'])} modules")
        
    except Exception as e:
        print(f"❌ ERROR during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
