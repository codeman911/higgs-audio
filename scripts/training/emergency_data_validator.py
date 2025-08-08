#!/usr/bin/env python3
"""
Emergency Data Validator for Higgs-Audio Training
Identifies and fixes critical data pipeline issues causing training failure.
"""

import json
import torch
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any
import sys

# Add higgs-audio to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class EmergencyDataValidator:
    """Emergency validator to identify critical data pipeline issues"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.issues = []
        self.stats = defaultdict(int)
        
    def validate_sample(self, sample: Dict) -> Dict[str, Any]:
        """Validate a single ChatML sample"""
        issues = []
        sample_stats = defaultdict(int)
        
        try:
            messages = sample.get('messages', [])
            if not messages:
                issues.append("No messages in sample")
                return {'issues': issues, 'stats': sample_stats}
            
            # Check for assistant message with audio content
            assistant_audio_found = False
            for message in messages:
                if message.get('role') == 'assistant':
                    content = message.get('content', [])
                    for content_item in content:
                        if content_item.get('type') == 'audio':
                            assistant_audio_found = True
                            
                            # Check audio tokens
                            audio_tokens = content_item.get('audio_tokens')
                            if audio_tokens is None:
                                issues.append("Missing audio_tokens in assistant message")
                            else:
                                try:
                                    tokens_tensor = torch.tensor(audio_tokens)
                                    sample_stats['audio_tokens_shape'] = list(tokens_tensor.shape)
                                    
                                    if tokens_tensor.dim() != 2:
                                        issues.append(f"Invalid audio_tokens shape: {tokens_tensor.shape} (expected 2D)")
                                    else:
                                        num_codebooks, seq_len = tokens_tensor.shape
                                        sample_stats['num_codebooks'] = num_codebooks
                                        sample_stats['sequence_length'] = seq_len
                                        
                                        # Check for valid tokens per codebook
                                        for cb_idx in range(num_codebooks):
                                            cb_tokens = tokens_tensor[cb_idx]
                                            valid_tokens = cb_tokens[cb_tokens != -100]
                                            sample_stats[f'codebook_{cb_idx}_valid_tokens'] = len(valid_tokens)
                                            
                                            if len(valid_tokens) == 0:
                                                issues.append(f"Codebook {cb_idx} has no valid tokens (all -100)")
                                            elif len(valid_tokens) < 5:
                                                issues.append(f"Codebook {cb_idx} has very few valid tokens: {len(valid_tokens)}")
                                            
                                            # Check token range
                                            if len(valid_tokens) > 0:
                                                min_token = valid_tokens.min().item()
                                                max_token = valid_tokens.max().item()
                                                sample_stats[f'codebook_{cb_idx}_token_range'] = (min_token, max_token)
                                                
                                                if min_token < 0 or max_token > 1024:
                                                    issues.append(f"Codebook {cb_idx} has invalid token range: {min_token}-{max_token}")
                                                
                                                # Check for token dominance
                                                token_counts = Counter(valid_tokens.tolist())
                                                most_common = token_counts.most_common(1)[0]
                                                dominance = most_common[1] / len(valid_tokens)
                                                if dominance > 0.8:
                                                    issues.append(f"Codebook {cb_idx} shows severe token dominance: {dominance:.1%}")
                                        
                                except Exception as e:
                                    issues.append(f"Error processing audio_tokens: {e}")
                            
                            # Check raw audio
                            raw_audio = content_item.get('raw_audio')
                            if raw_audio is None:
                                issues.append("Missing raw_audio in assistant message")
                            
                            break
            
            if not assistant_audio_found:
                issues.append("No audio content found in assistant message")
            
            # Check start_index
            start_index = sample.get('start_index', 0)
            if start_index != 2:
                issues.append(f"Incorrect start_index: {start_index} (expected 2)")
                
        except Exception as e:
            issues.append(f"Critical error validating sample: {e}")
        
        return {'issues': issues, 'stats': sample_stats}
    
    def validate_dataset(self, max_samples: int = 100) -> Dict[str, Any]:
        """Validate the entire dataset"""
        print("🚨 EMERGENCY DATA VALIDATION STARTING...")
        
        # Find data files
        data_files = list(self.data_path.glob("*chatml*.json"))
        if not data_files:
            return {'error': f'No ChatML files found in {self.data_path}'}
        
        print(f"📁 Found {len(data_files)} data files")
        
        total_samples = 0
        valid_samples = 0
        critical_issues = []
        all_stats = defaultdict(list)
        
        for data_file in data_files:
            print(f"🔍 Validating {data_file.name}...")
            
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    samples = json.load(f)
                
                samples_to_check = min(len(samples), max_samples)
                print(f"  Checking {samples_to_check}/{len(samples)} samples...")
                
                for i, sample in enumerate(samples[:samples_to_check]):
                    total_samples += 1
                    validation_result = self.validate_sample(sample)
                    
                    if not validation_result['issues']:
                        valid_samples += 1
                    else:
                        # Collect critical issues
                        for issue in validation_result['issues']:
                            if any(keyword in issue.lower() for keyword in ['missing', 'no valid tokens', 'invalid', 'critical']):
                                critical_issues.append(f"Sample {i}: {issue}")
                    
                    # Collect stats
                    for key, value in validation_result['stats'].items():
                        all_stats[key].append(value)
                    
                    # Stop if we find too many critical issues
                    if len(critical_issues) > 50:
                        print(f"⚠️  Too many critical issues found, stopping early...")
                        break
                        
            except Exception as e:
                critical_issues.append(f"Error reading {data_file}: {e}")
        
        # Analyze statistics
        analysis = {
            'total_samples_checked': total_samples,
            'valid_samples': valid_samples,
            'invalid_samples': total_samples - valid_samples,
            'validity_rate': valid_samples / total_samples if total_samples > 0 else 0,
            'critical_issues': critical_issues[:20],  # Show first 20
            'total_critical_issues': len(critical_issues)
        }
        
        # Analyze codebook statistics
        codebook_analysis = {}
        for key, values in all_stats.items():
            if 'codebook_' in key and 'valid_tokens' in key:
                codebook_analysis[key] = {
                    'mean': np.mean(values) if values else 0,
                    'min': np.min(values) if values else 0,
                    'max': np.max(values) if values else 0,
                    'samples_with_zero': sum(1 for v in values if v == 0)
                }
        
        analysis['codebook_analysis'] = codebook_analysis
        
        # Determine severity
        if analysis['validity_rate'] < 0.1:
            analysis['severity'] = 'CATASTROPHIC'
            analysis['recommendation'] = 'COMPLETE DATA REPROCESSING REQUIRED'
        elif analysis['validity_rate'] < 0.5:
            analysis['severity'] = 'CRITICAL'
            analysis['recommendation'] = 'MAJOR DATA PIPELINE FIXES REQUIRED'
        elif analysis['validity_rate'] < 0.8:
            analysis['severity'] = 'MODERATE'
            analysis['recommendation'] = 'DATA PIPELINE IMPROVEMENTS NEEDED'
        else:
            analysis['severity'] = 'MINOR'
            analysis['recommendation'] = 'MINOR FIXES REQUIRED'
        
        return analysis
    
    def generate_emergency_report(self, analysis: Dict[str, Any], output_path: str = "emergency_data_report.md"):
        """Generate emergency data validation report"""
        report = []
        report.append("# 🚨 EMERGENCY DATA VALIDATION REPORT\n")
        report.append(f"**Severity**: {analysis['severity']}\n")
        report.append(f"**Recommendation**: {analysis['recommendation']}\n\n")
        
        report.append("## 📊 Validation Summary\n")
        report.append(f"- **Total samples checked**: {analysis['total_samples_checked']:,}\n")
        report.append(f"- **Valid samples**: {analysis['valid_samples']:,}\n")
        report.append(f"- **Invalid samples**: {analysis['invalid_samples']:,}\n")
        report.append(f"- **Validity rate**: {analysis['validity_rate']:.1%}\n")
        report.append(f"- **Critical issues found**: {analysis['total_critical_issues']:,}\n\n")
        
        if analysis['validity_rate'] < 0.5:
            report.append("## 🚨 CRITICAL FAILURE\n")
            report.append("Your data pipeline has **CATASTROPHIC ISSUES** that prevent training:\n\n")
        
        report.append("## 🔍 Critical Issues (First 20)\n")
        for issue in analysis['critical_issues']:
            report.append(f"- {issue}\n")
        report.append("\n")
        
        if 'codebook_analysis' in analysis:
            report.append("## 📈 Codebook Analysis\n")
            for codebook, stats in analysis['codebook_analysis'].items():
                report.append(f"### {codebook}\n")
                report.append(f"- **Mean valid tokens**: {stats['mean']:.1f}\n")
                report.append(f"- **Min valid tokens**: {stats['min']}\n")
                report.append(f"- **Max valid tokens**: {stats['max']}\n")
                report.append(f"- **Samples with zero tokens**: {stats['samples_with_zero']}\n\n")
        
        report.append("## 🛠️ Immediate Actions Required\n")
        if analysis['severity'] == 'CATASTROPHIC':
            report.append("1. **STOP ALL TRAINING IMMEDIATELY**\n")
            report.append("2. **Completely reprocess your dataset** using the corrected data pipeline\n")
            report.append("3. **Verify audio tokenizer is working correctly**\n")
            report.append("4. **Check audio file paths and accessibility**\n")
            report.append("5. **Validate ChatML format generation**\n")
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(''.join(report))
        
        print(f"📄 Emergency report saved to: {output_path}")


def main():
    """Run emergency data validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Emergency data validation for failed training")
    parser.add_argument("--data_path", required=True, help="Path to ChatML data directory")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum samples to check per file")
    parser.add_argument("--output", default="emergency_data_report.md", help="Output report file")
    
    args = parser.parse_args()
    
    validator = EmergencyDataValidator(args.data_path)
    analysis = validator.validate_dataset(args.max_samples)
    
    if 'error' in analysis:
        print(f"❌ Error: {analysis['error']}")
        return
    
    validator.generate_emergency_report(analysis, args.output)
    
    print(f"\n🚨 EMERGENCY VALIDATION COMPLETE")
    print(f"Severity: {analysis['severity']}")
    print(f"Validity Rate: {analysis['validity_rate']:.1%}")
    print(f"Critical Issues: {analysis['total_critical_issues']:,}")
    
    if analysis['severity'] in ['CATASTROPHIC', 'CRITICAL']:
        print(f"\n⚠️  {analysis['recommendation']}")
        print("DO NOT CONTINUE TRAINING UNTIL DATA ISSUES ARE RESOLVED!")


if __name__ == "__main__":
    main()
