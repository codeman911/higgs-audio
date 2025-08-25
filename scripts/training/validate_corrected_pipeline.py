#!/usr/bin/env python3
"""
Comprehensive validation script for the corrected Higgs-Audio training pipeline.
This script validates that the training pipeline now correctly separates reference 
and target audio processing to match inference behavior.
"""

import os
import sys
import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent, TextContent
from boson_multimodal.audio_processing.higgs_audio_tokenizer import HiggsAudioTokenizer
from transformers import AutoTokenizer

# Import the corrected dataset
from scripts.training.distributed_trainer import ArabicEnglishDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineValidator:
    """Validates the corrected training pipeline"""
    
    def __init__(self, data_path: str, tokenizer_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
        self.data_path = data_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.audio_tokenizer = HiggsAudioTokenizer.from_pretrained("bosonai/higgs-audio-v2-tokenizer")
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_test_data(self) -> List[Dict]:
        """Load test data samples"""
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return data[:5]  # Test with first 5 samples
            else:
                logger.error(f"Expected list, got {type(data)}")
                return []
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return []
    
    def validate_sample_structure(self, sample: Dict) -> bool:
        """Validate that sample has expected structure"""
        required_fields = ['messages']
        for field in required_fields:
            if field not in sample:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Check messages structure
        if not isinstance(sample['messages'], list):
            logger.error("Messages should be a list")
            return False
        
        for i, msg in enumerate(sample['messages']):
            if 'role' not in msg or 'content' not in msg:
                logger.error(f"Message {i} missing role or content")
                return False
        
        return True
    
    def analyze_audio_separation(self, dataset_sample: ChatMLDatasetSample) -> Dict[str, Any]:
        """Analyze how audio is separated in the dataset sample"""
        analysis = {
            'reference_audio_tokens': dataset_sample.audio_ids_concat.shape if dataset_sample.audio_ids_concat.numel() > 0 else (0, 0),
            'reference_audio_count': len(dataset_sample.audio_ids_start),
            'target_audio_tokens': dataset_sample.audio_label_ids_concat.shape if dataset_sample.audio_label_ids_concat is not None and dataset_sample.audio_label_ids_concat.numel() > 0 else (0, 0),
            'reference_waveform_length': len(dataset_sample.audio_waveforms_concat),
            'text_tokens': len(dataset_sample.input_ids),
            'label_tokens': len(dataset_sample.label_ids),
            'non_ignored_labels': (dataset_sample.label_ids != -100).sum().item(),
        }
        
        return analysis
    
    def validate_inference_compatibility(self, sample: Dict) -> Dict[str, Any]:
        """Validate that the sample structure is compatible with inference"""
        compatibility = {
            'has_user_audio': False,
            'has_assistant_audio': False,
            'user_audio_paths': [],
            'assistant_audio_paths': [],
            'structure_valid': True
        }
        
        try:
            for msg in sample['messages']:
                role = msg['role']
                content = msg['content']
                
                if isinstance(content, list):
                    for item in content:
                        if item.get('type') == 'audio':
                            audio_path = item.get('audio_url', item.get('raw_audio', ''))
                            if role == 'user':
                                compatibility['has_user_audio'] = True
                                compatibility['user_audio_paths'].append(audio_path)
                            elif role == 'assistant':
                                compatibility['has_assistant_audio'] = True
                                compatibility['assistant_audio_paths'].append(audio_path)
        except Exception as e:
            logger.error(f"Error validating inference compatibility: {e}")
            compatibility['structure_valid'] = False
        
        return compatibility
    
    def test_single_sample(self, sample: Dict, sample_idx: int) -> Dict[str, Any]:
        """Test a single sample through the corrected pipeline"""
        logger.info(f"\n=== Testing Sample {sample_idx} ===")
        
        results = {
            'sample_idx': sample_idx,
            'structure_valid': False,
            'dataset_creation_success': False,
            'audio_separation_analysis': {},
            'inference_compatibility': {},
            'errors': []
        }
        
        try:
            # Validate sample structure
            if not self.validate_sample_structure(sample):
                results['errors'].append("Invalid sample structure")
                return results
            
            results['structure_valid'] = True
            
            # Test inference compatibility
            results['inference_compatibility'] = self.validate_inference_compatibility(sample)
            
            # Create dataset and test the corrected pipeline
            test_dataset = ArabicEnglishDataset([sample], self.tokenizer, self.audio_tokenizer)
            
            # Get dataset sample
            dataset_sample = test_dataset[0]
            results['dataset_creation_success'] = True
            
            # Analyze audio separation
            results['audio_separation_analysis'] = self.analyze_audio_separation(dataset_sample)
            
            # Log detailed analysis
            logger.info(f"Sample {sample_idx} Analysis:")
            logger.info(f"  Reference audio tokens: {results['audio_separation_analysis']['reference_audio_tokens']}")
            logger.info(f"  Target audio tokens: {results['audio_separation_analysis']['target_audio_tokens']}")
            logger.info(f"  Reference audio count: {results['audio_separation_analysis']['reference_audio_count']}")
            logger.info(f"  Text tokens: {results['audio_separation_analysis']['text_tokens']}")
            logger.info(f"  Non-ignored labels: {results['audio_separation_analysis']['non_ignored_labels']}")
            
            # Validate the key fix: reference vs target separation
            has_reference = results['audio_separation_analysis']['reference_audio_tokens'][1] > 0
            has_target = results['audio_separation_analysis']['target_audio_tokens'][1] > 0
            
            if has_reference and has_target:
                logger.info(f"✅ CORRECT: Sample {sample_idx} has both reference and target audio properly separated")
            elif has_reference and not has_target:
                logger.info(f"⚠️  PARTIAL: Sample {sample_idx} has reference audio but no target audio")
            elif not has_reference and has_target:
                logger.info(f"⚠️  PARTIAL: Sample {sample_idx} has target audio but no reference audio")
            else:
                logger.info(f"❌ ISSUE: Sample {sample_idx} has no audio tokens")
            
        except Exception as e:
            error_msg = f"Error processing sample {sample_idx}: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        return results
    
    def run_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of the corrected pipeline"""
        logger.info("🚀 Starting Higgs-Audio Pipeline Validation")
        logger.info("=" * 60)
        
        # Load test data
        test_samples = self.load_test_data()
        if not test_samples:
            return {'error': 'No test data loaded'}
        
        logger.info(f"Loaded {len(test_samples)} test samples")
        
        # Test each sample
        sample_results = []
        for i, sample in enumerate(test_samples):
            result = self.test_single_sample(sample, i)
            sample_results.append(result)
        
        # Aggregate results
        total_samples = len(sample_results)
        successful_samples = sum(1 for r in sample_results if r['dataset_creation_success'])
        samples_with_reference = sum(1 for r in sample_results if r['audio_separation_analysis'].get('reference_audio_tokens', (0, 0))[1] > 0)
        samples_with_target = sum(1 for r in sample_results if r['audio_separation_analysis'].get('target_audio_tokens', (0, 0))[1] > 0)
        samples_with_both = sum(1 for r in sample_results if (
            r['audio_separation_analysis'].get('reference_audio_tokens', (0, 0))[1] > 0 and
            r['audio_separation_analysis'].get('target_audio_tokens', (0, 0))[1] > 0
        ))
        
        validation_summary = {
            'total_samples': total_samples,
            'successful_samples': successful_samples,
            'samples_with_reference_audio': samples_with_reference,
            'samples_with_target_audio': samples_with_target,
            'samples_with_both_audio_types': samples_with_both,
            'success_rate': successful_samples / total_samples if total_samples > 0 else 0,
            'audio_separation_rate': samples_with_both / total_samples if total_samples > 0 else 0,
            'sample_results': sample_results
        }
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("🎯 VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total samples tested: {total_samples}")
        logger.info(f"Successfully processed: {successful_samples} ({validation_summary['success_rate']:.1%})")
        logger.info(f"Samples with reference audio: {samples_with_reference}")
        logger.info(f"Samples with target audio: {samples_with_target}")
        logger.info(f"Samples with proper audio separation: {samples_with_both} ({validation_summary['audio_separation_rate']:.1%})")
        
        if validation_summary['audio_separation_rate'] > 0.8:
            logger.info("✅ PIPELINE VALIDATION PASSED: Audio separation is working correctly!")
        elif validation_summary['success_rate'] > 0.8:
            logger.info("⚠️  PIPELINE PARTIALLY WORKING: Processing succeeds but audio separation needs review")
        else:
            logger.info("❌ PIPELINE VALIDATION FAILED: Major issues detected")
        
        return validation_summary

def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate corrected Higgs-Audio training pipeline")
    parser.add_argument("--data_path", required=True, help="Path to test data JSON file")
    parser.add_argument("--tokenizer", default="meta-llama/Llama-3.2-3B-Instruct", help="Tokenizer name")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        return 1
    
    validator = PipelineValidator(args.data_path, args.tokenizer)
    results = validator.run_validation()
    
    # Save results
    output_path = "pipeline_validation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n📊 Detailed results saved to: {output_path}")
    
    return 0 if results.get('audio_separation_rate', 0) > 0.8 else 1

if __name__ == "__main__":
    exit(main())
