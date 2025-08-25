#!/usr/bin/env python3
"""
Validation and Testing Pipeline for Arabic Voice Cloning Training

This module provides comprehensive validation utilities to ensure training
correctness, data quality, and model performance.
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import time
import psutil
from loguru import logger

from arabic_voice_cloning_dataset import ArabicVoiceCloningDataset, ArabicVoiceCloningDatasetConfig
from arabic_voice_cloning_training_collator import ArabicVoiceCloningTrainingCollator
from arabic_voice_cloning_lora_config import create_higgs_audio_lora_model, HiggsAudioLoRATrainingConfig
from arabic_voice_cloning_loss_function import create_loss_function, LossConfig
from boson_multimodal.model.higgs_audio.configuration_higgs_audio import HiggsAudioConfig
from transformers import AutoTokenizer


class DataValidationSuite:
    """Comprehensive data validation for training pipeline."""
    
    def __init__(self, dataset_config: ArabicVoiceCloningDatasetConfig):
        self.dataset_config = dataset_config
        self.validation_results = {}
    
    def validate_chatml_data(self) -> Dict[str, Any]:
        """Validate ChatML data format and structure."""
        logger.info("Validating ChatML data format...")
        
        results = {
            "total_samples": 0,
            "valid_samples": 0,
            "missing_audio": 0,
            "invalid_structure": 0
        }
        
        try:
            with open(self.dataset_config.chatml_file, 'r') as f:
                if self.dataset_config.chatml_file.endswith('.jsonl'):
                    data = [json.loads(line) for line in f if line.strip()]
                else:
                    data = json.load(f)
            
            results["total_samples"] = len(data)
            
            for i, sample in enumerate(data[:1000]):  # Test first 1000 samples
                try:
                    if self._validate_sample_structure(sample):
                        audio_paths = self._extract_audio_paths(sample)
                        if self._validate_audio_files(audio_paths):
                            results["valid_samples"] += 1
                        else:
                            results["missing_audio"] += 1
                    else:
                        results["invalid_structure"] += 1
                except Exception:
                    results["invalid_structure"] += 1
            
            valid_percentage = (results["valid_samples"] / min(1000, len(data))) * 100
            
            logger.info(f"Data validation results:")
            logger.info(f"  Valid samples: {results['valid_samples']}/1000 ({valid_percentage:.1f}%)")
            
            self.validation_results["data_validation"] = results
            return results
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            raise
    
    def _validate_sample_structure(self, sample: Dict) -> bool:
        """Validate ChatML sample structure."""
        if "messages" not in sample:
            return False
        
        messages = sample["messages"]
        if not isinstance(messages, list) or len(messages) < 2:
            return False
        
        has_user = any(msg.get("role") == "user" for msg in messages)
        has_assistant = any(msg.get("role") == "assistant" for msg in messages)
        
        return has_user and has_assistant
    
    def _extract_audio_paths(self, sample: Dict) -> List[str]:
        """Extract audio paths from ChatML sample using direct paths."""
        audio_paths = []
        for message in sample["messages"]:
            content = message.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "audio":
                        audio_url = item.get("audio_url")
                        if audio_url:
                            # Use direct path from ChatML - no base path concatenation
                            audio_paths.append(audio_url)
        return audio_paths
    
    def _validate_audio_files(self, audio_paths: List[str]) -> bool:
        """Validate audio file accessibility."""
        for path in audio_paths:
            if not Path(path).exists():
                return False
            try:
                waveform, sample_rate = torchaudio.load(path)
                duration = len(waveform[0]) / sample_rate
                if duration < 0.5 or duration > 30.0:
                    return False
            except Exception:
                return False
        return True


class TrainingPipelineValidator:
    """Validate training pipeline correctness."""
    
    def __init__(self, model_path: str = "bosonai/higgs-audio-v2-generation-3B-base"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.validation_results = {}
    
    def test_model_loading(self) -> Dict[str, Any]:
        """Test model loading and LoRA application."""
        logger.info("Testing model loading...")
        
        results = {"success": False, "error": None, "model_info": {}}
        
        try:
            lora_config = HiggsAudioLoRATrainingConfig(r=8, target_modules_mode="attention_only")
            
            model, config, _ = create_higgs_audio_lora_model(
                model_path=self.model_path,
                custom_config=lora_config,
                device_map="cpu",  # Use CPU for testing
                torch_dtype=torch.float32
            )
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            results["model_info"] = {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "trainable_percentage": (trainable_params / total_params) * 100
            }
            
            results["success"] = True
            logger.info(f"Model loading successful: {trainable_params:,} trainable parameters")
            
        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Model loading failed: {e}")
        
        self.validation_results["model_loading"] = results
        return results
    
    def test_data_pipeline(self, dataset_config: ArabicVoiceCloningDatasetConfig) -> Dict[str, Any]:
        """Test data loading and collation pipeline."""
        logger.info("Testing data pipeline...")
        
        results = {"success": False, "error": None}
        
        try:
            text_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            dataset = ArabicVoiceCloningDataset(
                config=dataset_config,
                audio_tokenizer=None,
                text_tokenizer=text_tokenizer
            )
            
            if len(dataset) == 0:
                raise ValueError("Dataset is empty")
            
            sample = dataset[0]
            
            config = HiggsAudioConfig()
            collator = ArabicVoiceCloningTrainingCollator(
                config=config,
                whisper_processor=None,
                enable_teacher_forcing=True
            )
            
            batch = collator([sample])
            
            results["success"] = True
            logger.info(f"Data pipeline test successful: {len(dataset)} samples, batch shape {batch.input_ids.shape}")
            
        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Data pipeline test failed: {e}")
        
        self.validation_results["data_pipeline"] = results
        return results
    
    def test_loss_computation(self) -> Dict[str, Any]:
        """Test loss function computation."""
        logger.info("Testing loss computation...")
        
        results = {"success": False, "error": None}
        
        try:
            config = HiggsAudioConfig()
            loss_config = LossConfig()
            loss_fn = create_loss_function(config, vocab_size=128256, loss_config=loss_config)
            
            # Create dummy data
            batch_size, seq_len = 2, 10
            text_logits = torch.randn(batch_size, seq_len, 128256)
            audio_logits = torch.randn(12, seq_len, 1024)
            
            from arabic_voice_cloning_training_collator import HiggsAudioTrainingBatch
            dummy_batch = HiggsAudioTrainingBatch(
                input_ids=torch.randint(0, 128256, (batch_size, seq_len)),
                attention_mask=torch.ones(batch_size, seq_len),
                labels=torch.randint(0, 128256, (batch_size, seq_len)),
                audio_features=None,
                audio_feature_attention_mask=None,
                audio_out_ids=torch.randint(0, 1024, (12, seq_len)),
                audio_out_ids_start=torch.tensor([0]),
                audio_out_ids_start_group_loc=None,
                audio_in_ids=None,
                audio_in_ids_start=None,
                audio_labels=torch.randint(0, 1024, (12, seq_len))
            )
            
            loss_dict = loss_fn(
                text_logits=text_logits,
                audio_logits=audio_logits,
                batch=dummy_batch
            )
            
            total_loss = loss_dict['losses']['total_loss'].item()
            
            results["success"] = True
            logger.info(f"Loss computation test successful: total loss {total_loss:.6f}")
            
        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Loss computation test failed: {e}")
        
        self.validation_results["loss_computation"] = results
        return results


class PerformanceBenchmark:
    """Benchmark training performance."""
    
    def __init__(self):
        self.benchmark_results = {}
    
    def benchmark_system_resources(self) -> Dict[str, Any]:
        """Benchmark system resource usage."""
        logger.info("Benchmarking system resources...")
        
        results = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1e9,
            "memory_available_gb": psutil.virtual_memory().available / 1e9,
            "gpu_count": 0,
            "gpus": []
        }
        
        if torch.cuda.is_available():
            results["gpu_count"] = torch.cuda.device_count()
            
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                gpu_info = {
                    "name": gpu_props.name,
                    "memory_total_gb": gpu_props.total_memory / 1e9,
                    "compute_capability": f"{gpu_props.major}.{gpu_props.minor}"
                }
                results["gpus"].append(gpu_info)
        
        logger.info(f"System resources: {results['cpu_count']} CPUs, {results['memory_total_gb']:.1f}GB RAM, {results['gpu_count']} GPUs")
        
        self.benchmark_results["system_resources"] = results
        return results


def generate_validation_report(
    data_validator: DataValidationSuite,
    pipeline_validator: TrainingPipelineValidator,
    benchmark: PerformanceBenchmark,
    output_dir: str
) -> str:
    """Generate comprehensive validation report."""
    report_path = Path(output_dir) / "validation_report.json"
    
    full_report = {
        "validation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_validation": data_validator.validation_results,
        "pipeline_validation": pipeline_validator.validation_results,
        "performance_benchmark": benchmark.benchmark_results
    }
    
    with open(report_path, 'w') as f:
        json.dump(full_report, f, indent=2, default=str)
    
    # Generate summary
    logger.info("🔍 VALIDATION REPORT SUMMARY")
    logger.info("=" * 50)
    
    if "data_validation" in data_validator.validation_results:
        data_results = data_validator.validation_results["data_validation"]
        logger.info(f"📊 Data: {data_results['valid_samples']}/1000 samples valid")
    
    pipeline_success = all(
        result.get("success", False) 
        for result in pipeline_validator.validation_results.values()
    )
    logger.info(f"🔧 Pipeline: {'✅ All tests passed' if pipeline_success else '❌ Some tests failed'}")
    
    logger.info(f"📄 Full report: {report_path}")
    
    return str(report_path)


def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Arabic Voice Cloning Validation Suite")
    parser.add_argument("--data_path", required=True, help="Path to ChatML data file with direct audio paths")
    parser.add_argument("--output_dir", default="./validation_output", help="Output directory")
    parser.add_argument("--model_path", default="bosonai/higgs-audio-v2-generation-3B-base", help="Model path")
    parser.add_argument("--skip_model_tests", action="store_true", help="Skip model tests")
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 Starting Validation Suite")
    
    try:
        dataset_config = ArabicVoiceCloningDatasetConfig(
            chatml_file=args.data_path,
            validate_on_init=False
        )
        
        # Data validation
        data_validator = DataValidationSuite(dataset_config)
        data_validator.validate_chatml_data()
        
        # Pipeline validation
        if not args.skip_model_tests:
            pipeline_validator = TrainingPipelineValidator(args.model_path)
            pipeline_validator.test_model_loading()
            pipeline_validator.test_data_pipeline(dataset_config)
            pipeline_validator.test_loss_computation()
        else:
            pipeline_validator = TrainingPipelineValidator()
        
        # Performance benchmark
        benchmark = PerformanceBenchmark()
        benchmark.benchmark_system_resources()
        
        # Generate report
        report_path = generate_validation_report(
            data_validator, pipeline_validator, benchmark, args.output_dir
        )
        
        logger.info("✅ Validation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        raise


if __name__ == "__main__":
    main()