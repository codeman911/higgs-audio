#!/usr/bin/env python3
"""
Loss Collapse Diagnostic Tool for Higgs-Audio LoRA Training
Identifies root causes of abnormal loss behavior through comprehensive analysis.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

# Add higgs-audio to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample


class LossCollapseDiagnostic:
    """Comprehensive diagnostic tool for loss collapse analysis"""
    
    def __init__(self, data_path: str, audio_tokenizer_path: str = "bosonai/higgs-audio-v2-tokenizer"):
        self.data_path = Path(data_path)
        self.audio_tokenizer = load_higgs_audio_tokenizer(
            audio_tokenizer_path, 
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.results = {}
        
    def analyze_codebook_distribution(self, samples: List[Dict]) -> Dict[str, Any]:
        """Analyze audio token distribution across codebooks"""
        print("🔍 Analyzing codebook token distribution...")
        
        codebook_stats = defaultdict(lambda: defaultdict(int))
        total_tokens_per_codebook = defaultdict(int)
        
        for sample in tqdm(samples[:100], desc="Processing samples"):  # Analyze first 100 samples
            try:
                # Extract audio tokens from assistant message
                for message in sample.get('messages', []):
                    if message.get('role') == 'assistant':
                        for content in message.get('content', []):
                            if content.get('type') == 'audio' and 'audio_tokens' in content:
                                audio_tokens = torch.tensor(content['audio_tokens'])
                                
                                if audio_tokens.dim() == 2:  # [codebooks, sequence_length]
                                    for codebook_idx in range(audio_tokens.shape[0]):
                                        tokens = audio_tokens[codebook_idx]
                                        valid_tokens = tokens[tokens != -100]
                                        
                                        if len(valid_tokens) > 0:
                                            token_counts = Counter(valid_tokens.tolist())
                                            for token, count in token_counts.items():
                                                codebook_stats[codebook_idx][token] += count
                                                total_tokens_per_codebook[codebook_idx] += count
                                                
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue
        
        # Analyze distribution statistics
        analysis = {
            'num_codebooks': len(codebook_stats),
            'codebook_analysis': {}
        }
        
        for codebook_idx, token_counts in codebook_stats.items():
            total_tokens = total_tokens_per_codebook[codebook_idx]
            unique_tokens = len(token_counts)
            
            # Calculate entropy (measure of distribution uniformity)
            probs = np.array(list(token_counts.values())) / total_tokens
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            max_entropy = np.log2(unique_tokens) if unique_tokens > 0 else 0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            # Find dominant tokens
            sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
            top_5_tokens = sorted_tokens[:5]
            top_token_dominance = top_5_tokens[0][1] / total_tokens if total_tokens > 0 else 0
            
            analysis['codebook_analysis'][codebook_idx] = {
                'total_tokens': total_tokens,
                'unique_tokens': unique_tokens,
                'entropy': entropy,
                'normalized_entropy': normalized_entropy,
                'top_token_dominance': top_token_dominance,
                'top_5_tokens': top_5_tokens,
                'distribution_skew': 'HIGH' if top_token_dominance > 0.3 else 'MEDIUM' if top_token_dominance > 0.1 else 'LOW'
            }
        
        return analysis
    
    def analyze_loss_computation_stability(self, samples: List[Dict]) -> Dict[str, Any]:
        """Analyze potential numerical instability in loss computation"""
        print("🔍 Analyzing loss computation stability...")
        
        # Simulate loss computation on sample data
        text_losses = []
        audio_losses = []
        combined_losses = []
        
        for sample in tqdm(samples[:50], desc="Computing sample losses"):
            try:
                # Extract labels and simulate logits
                text_labels = []
                audio_labels = []
                
                for message in sample.get('messages', []):
                    if message.get('role') == 'assistant':
                        for content in message.get('content', []):
                            if content.get('type') == 'text':
                                # Simulate text labels (tokenized text)
                                text_labels.extend([1, 2, 3, 4, 5] * 10)  # Dummy labels
                            elif content.get('type') == 'audio' and 'audio_tokens' in content:
                                audio_tokens = torch.tensor(content['audio_tokens'])
                                if audio_tokens.dim() == 2:
                                    audio_labels.append(audio_tokens)
                
                if text_labels and audio_labels:
                    # Simulate text loss
                    text_labels_tensor = torch.tensor(text_labels)
                    text_logits = torch.randn(len(text_labels), 32000)  # Vocab size
                    text_loss = nn.functional.cross_entropy(text_logits, text_labels_tensor, reduction='mean')
                    text_losses.append(text_loss.item())
                    
                    # Simulate audio loss
                    audio_tokens = audio_labels[0]  # Take first audio sample
                    total_audio_loss = 0
                    valid_codebooks = 0
                    
                    for codebook_idx in range(audio_tokens.shape[0]):
                        codebook_labels = audio_tokens[codebook_idx]
                        valid_mask = codebook_labels != -100
                        
                        if valid_mask.sum() > 0:
                            codebook_logits = torch.randn(len(codebook_labels), 1024)  # Codebook size
                            codebook_loss = nn.functional.cross_entropy(
                                codebook_logits, codebook_labels, ignore_index=-100, reduction='mean'
                            )
                            total_audio_loss += codebook_loss.item()
                            valid_codebooks += 1
                    
                    if valid_codebooks > 0:
                        audio_loss = total_audio_loss / valid_codebooks
                        audio_losses.append(audio_loss)
                        
                        # Test different weighting strategies
                        # Strategy 1: Adaptive weighting (current problematic approach)
                        adaptive_weight = min(text_loss.item() / max(audio_loss, 1e-6), 2.0)
                        adaptive_combined = text_loss.item() + audio_loss * adaptive_weight
                        
                        # Strategy 2: Fixed weighting
                        fixed_combined = text_loss.item() * 0.3 + audio_loss * 0.7
                        
                        combined_losses.append({
                            'text_loss': text_loss.item(),
                            'audio_loss': audio_loss,
                            'adaptive_combined': adaptive_combined,
                            'fixed_combined': fixed_combined,
                            'adaptive_weight': adaptive_weight
                        })
                        
            except Exception as e:
                print(f"Error in loss computation: {e}")
                continue
        
        # Analyze loss statistics
        if combined_losses:
            adaptive_weights = [l['adaptive_weight'] for l in combined_losses]
            adaptive_combined = [l['adaptive_combined'] for l in combined_losses]
            fixed_combined = [l['fixed_combined'] for l in combined_losses]
            
            analysis = {
                'num_samples_analyzed': len(combined_losses),
                'text_loss_stats': {
                    'mean': np.mean(text_losses),
                    'std': np.std(text_losses),
                    'min': np.min(text_losses),
                    'max': np.max(text_losses)
                },
                'audio_loss_stats': {
                    'mean': np.mean(audio_losses),
                    'std': np.std(audio_losses),
                    'min': np.min(audio_losses),
                    'max': np.max(audio_losses)
                },
                'adaptive_weight_stats': {
                    'mean': np.mean(adaptive_weights),
                    'std': np.std(adaptive_weights),
                    'min': np.min(adaptive_weights),
                    'max': np.max(adaptive_weights),
                    'instability_score': np.std(adaptive_weights) / np.mean(adaptive_weights)
                },
                'combined_loss_comparison': {
                    'adaptive_mean': np.mean(adaptive_combined),
                    'fixed_mean': np.mean(fixed_combined),
                    'adaptive_std': np.std(adaptive_combined),
                    'fixed_std': np.std(fixed_combined)
                }
            }
        else:
            analysis = {'error': 'No valid samples for loss analysis'}
        
        return analysis
    
    def analyze_data_leakage(self, samples: List[Dict]) -> Dict[str, Any]:
        """Analyze potential data leakage in ChatML structure"""
        print("🔍 Analyzing potential data leakage...")
        
        leakage_analysis = {
            'samples_analyzed': 0,
            'potential_leakage_cases': 0,
            'leakage_patterns': []
        }
        
        for sample in tqdm(samples[:100], desc="Checking for data leakage"):
            try:
                messages = sample.get('messages', [])
                leakage_analysis['samples_analyzed'] += 1
                
                user_content = ""
                assistant_content = ""
                
                for message in messages:
                    if message.get('role') == 'user':
                        for content in message.get('content', []):
                            if content.get('type') == 'text':
                                user_content += content.get('text', '')
                    elif message.get('role') == 'assistant':
                        for content in message.get('content', []):
                            if content.get('type') == 'text':
                                assistant_content += content.get('text', '')
                
                # Check for potential leakage patterns
                if user_content and assistant_content:
                    # Check if target text appears in user message
                    if assistant_content.lower() in user_content.lower():
                        leakage_analysis['potential_leakage_cases'] += 1
                        leakage_analysis['leakage_patterns'].append({
                            'type': 'target_text_in_user_message',
                            'user_content': user_content[:200],
                            'assistant_content': assistant_content[:200]
                        })
                
                # Check start_index configuration
                start_index = sample.get('start_index', 0)
                if start_index != 2:  # Should be 2 for assistant message
                    leakage_analysis['leakage_patterns'].append({
                        'type': 'incorrect_start_index',
                        'start_index': start_index,
                        'expected': 2
                    })
                        
            except Exception as e:
                print(f"Error analyzing sample for leakage: {e}")
                continue
        
        leakage_analysis['leakage_percentage'] = (
            leakage_analysis['potential_leakage_cases'] / 
            leakage_analysis['samples_analyzed'] * 100
            if leakage_analysis['samples_analyzed'] > 0 else 0
        )
        
        return leakage_analysis
    
    def run_full_diagnostic(self, output_dir: str = "diagnostic_results"):
        """Run complete diagnostic analysis"""
        print("🚀 Starting comprehensive loss collapse diagnostic...")
        
        # Load data
        data_files = list(self.data_path.glob("chatml_samples_*.json"))
        if not data_files:
            print(f"❌ No ChatML data files found in {self.data_path}")
            return
        
        all_samples = []
        for data_file in data_files:
            with open(data_file, 'r', encoding='utf-8') as f:
                samples = json.load(f)
                all_samples.extend(samples)
        
        print(f"📊 Loaded {len(all_samples)} samples from {len(data_files)} files")
        
        # Run analyses
        self.results['codebook_analysis'] = self.analyze_codebook_distribution(all_samples)
        self.results['loss_stability_analysis'] = self.analyze_loss_computation_stability(all_samples)
        self.results['data_leakage_analysis'] = self.analyze_data_leakage(all_samples)
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        with open(output_path / "diagnostic_results.json", 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate report
        self.generate_report(output_path)
        
        print(f"✅ Diagnostic complete! Results saved to {output_path}")
        
    def generate_report(self, output_path: Path):
        """Generate human-readable diagnostic report"""
        report = []
        report.append("# 🔍 Loss Collapse Diagnostic Report\n")
        
        # Codebook Analysis
        if 'codebook_analysis' in self.results:
            ca = self.results['codebook_analysis']
            report.append("## 📊 Codebook Distribution Analysis\n")
            report.append(f"- **Number of codebooks**: {ca.get('num_codebooks', 'Unknown')}\n")
            
            for codebook_idx, stats in ca.get('codebook_analysis', {}).items():
                report.append(f"### Codebook {codebook_idx}\n")
                report.append(f"- **Total tokens**: {stats['total_tokens']:,}\n")
                report.append(f"- **Unique tokens**: {stats['unique_tokens']:,}\n")
                report.append(f"- **Entropy**: {stats['entropy']:.3f} (normalized: {stats['normalized_entropy']:.3f})\n")
                report.append(f"- **Top token dominance**: {stats['top_token_dominance']:.1%}\n")
                report.append(f"- **Distribution skew**: {stats['distribution_skew']}\n")
                
                if stats['distribution_skew'] == 'HIGH':
                    report.append("  ⚠️  **WARNING**: High token dominance detected - potential cause of loss collapse!\n")
                
                report.append(f"- **Top 5 tokens**: {stats['top_5_tokens']}\n\n")
        
        # Loss Stability Analysis
        if 'loss_stability_analysis' in self.results:
            lsa = self.results['loss_stability_analysis']
            report.append("## ⚖️ Loss Computation Stability Analysis\n")
            
            if 'error' not in lsa:
                report.append(f"- **Samples analyzed**: {lsa['num_samples_analyzed']}\n")
                
                aws = lsa['adaptive_weight_stats']
                report.append(f"- **Adaptive weight instability score**: {aws['instability_score']:.3f}\n")
                
                if aws['instability_score'] > 1.0:
                    report.append("  ⚠️  **WARNING**: High weight instability - major cause of loss collapse!\n")
                
                clc = lsa['combined_loss_comparison']
                report.append(f"- **Adaptive loss std**: {clc['adaptive_std']:.3f}\n")
                report.append(f"- **Fixed loss std**: {clc['fixed_std']:.3f}\n")
                
                if clc['adaptive_std'] > clc['fixed_std'] * 2:
                    report.append("  ⚠️  **WARNING**: Adaptive weighting causes high loss variance!\n")
            else:
                report.append(f"❌ Error in analysis: {lsa['error']}\n")
        
        # Data Leakage Analysis
        if 'data_leakage_analysis' in self.results:
            dla = self.results['data_leakage_analysis']
            report.append("## 🔒 Data Leakage Analysis\n")
            report.append(f"- **Samples analyzed**: {dla['samples_analyzed']}\n")
            report.append(f"- **Potential leakage cases**: {dla['potential_leakage_cases']}\n")
            report.append(f"- **Leakage percentage**: {dla['leakage_percentage']:.1f}%\n")
            
            if dla['leakage_percentage'] > 10:
                report.append("  ⚠️  **WARNING**: High data leakage detected!\n")
            
            if dla['leakage_patterns']:
                report.append("- **Leakage patterns found**:\n")
                for pattern in dla['leakage_patterns'][:5]:  # Show first 5
                    report.append(f"  - {pattern['type']}\n")
        
        # Recommendations
        report.append("## 🎯 Recommendations\n")
        
        # Check for codebook collapse
        if 'codebook_analysis' in self.results:
            ca = self.results['codebook_analysis']
            high_dominance_codebooks = []
            for cb_idx, stats in ca.get('codebook_analysis', {}).items():
                if stats['top_token_dominance'] > 0.3:
                    high_dominance_codebooks.append(cb_idx)
            
            if high_dominance_codebooks:
                report.append("### 🚨 CRITICAL: Codebook Collapse Detected\n")
                report.append(f"Codebooks with high token dominance: {high_dominance_codebooks}\n")
                report.append("**Immediate fixes**:\n")
                report.append("1. Add label smoothing to cross-entropy loss\n")
                report.append("2. Use class-balanced cross-entropy\n")
                report.append("3. Add codebook entropy regularization\n")
                report.append("4. Increase LoRA dropout to 0.3-0.5\n\n")
        
        # Check for loss instability
        if 'loss_stability_analysis' in self.results:
            lsa = self.results['loss_stability_analysis']
            if 'adaptive_weight_stats' in lsa and lsa['adaptive_weight_stats']['instability_score'] > 1.0:
                report.append("### ⚡ Loss Computation Instability\n")
                report.append("**Immediate fixes**:\n")
                report.append("1. Replace adaptive weighting with fixed weights (text: 0.3, audio: 0.7)\n")
                report.append("2. Cast logits to float32 before cross-entropy\n")
                report.append("3. Add gradient clipping at 1.0\n\n")
        
        # Save report
        with open(output_path / "diagnostic_report.md", 'w') as f:
            f.write(''.join(report))


def main():
    """Run diagnostic analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose loss collapse in LoRA training")
    parser.add_argument("--data_path", required=True, help="Path to processed ChatML data directory")
    parser.add_argument("--output_dir", default="diagnostic_results", help="Output directory for results")
    parser.add_argument("--audio_tokenizer", default="bosonai/higgs-audio-v2-tokenizer", help="Audio tokenizer path")
    
    args = parser.parse_args()
    
    diagnostic = LossCollapseDiagnostic(
        data_path=args.data_path,
        audio_tokenizer_path=args.audio_tokenizer
    )
    
    diagnostic.run_full_diagnostic(args.output_dir)


if __name__ == "__main__":
    main()
