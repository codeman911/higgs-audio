#!/usr/bin/env python3
"""
Validation and Testing Utilities for Arabic Voice Cloning

Core validation and testing tools for the Arabic voice cloning pipeline.
"""

import os
import json
import torch
import librosa
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from loguru import logger

from arabic_text_preprocessing import detect_arabic_content, validate_arabic_text


@dataclass
class AudioQualityMetrics:
    """Audio quality metrics."""
    duration: float
    sample_rate: int
    rms_energy: float
    is_clipped: bool = False
    silence_ratio: float = 0.0


@dataclass
class ChatMLValidationResult:
    """Validation result for a ChatML sample."""
    sample_id: str
    is_valid: bool
    issues: List[str]
    warnings: List[str]


@dataclass
class ModelTestResult:
    """Result from model testing."""
    test_name: str
    passed: bool
    error_message: Optional[str] = None
    execution_time: Optional[float] = None


class AudioQualityAnalyzer:
    """Analyzer for audio quality metrics."""
    
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
    
    def analyze(self, audio_path: str) -> Optional[AudioQualityMetrics]:
        """Analyze audio quality metrics."""
        try:
            if not os.path.exists(audio_path):
                return None
            
            y, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
            if len(y) == 0:
                return None
            
            duration = len(y) / sr
            rms_energy = float(np.sqrt(np.mean(y**2)))
            is_clipped = bool(np.any(np.abs(y) >= 0.99))
            silence_ratio = float(np.sum(np.abs(y) < 0.01) / len(y))
            
            return AudioQualityMetrics(
                duration=duration,
                sample_rate=sr,
                rms_energy=rms_energy,
                is_clipped=is_clipped,
                silence_ratio=silence_ratio,
            )
            
        except Exception as e:
            logger.error(f"Audio analysis failed for {audio_path}: {e}")
            return None


class ChatMLDataValidator:
    """Validator for ChatML format data."""
    
    def __init__(
        self,
        check_audio_files: bool = True,
        min_audio_duration: float = 1.0,
        max_audio_duration: float = 30.0,
        min_text_length: int = 10,
    ):
        self.check_audio_files = check_audio_files
        self.min_audio_duration = min_audio_duration
        self.max_audio_duration = max_audio_duration
        self.min_text_length = min_text_length
        self.audio_analyzer = AudioQualityAnalyzer()
    
    def validate_sample(self, sample: Dict[str, Any], audio_base_path: Optional[str] = None) -> ChatMLValidationResult:
        """Validate a single ChatML sample."""
        sample_id = sample.get("speaker", "unknown")
        issues = []
        warnings = []
        
        # Extract components
        ref_audio_path, ref_text, target_text, target_audio_path = self._extract_components(sample, audio_base_path)
        
        # Validate text
        if not self._validate_text(ref_text, "reference", issues):
            pass
        if not self._validate_text(target_text, "target", issues):
            pass
        
        # Validate audio
        if self.check_audio_files:
            if not self._validate_audio(ref_audio_path, "reference", issues):
                pass
            if not self._validate_audio(target_audio_path, "target", issues):
                pass
        
        is_valid = len(issues) == 0
        
        return ChatMLValidationResult(
            sample_id=sample_id,
            is_valid=is_valid,
            issues=issues,
            warnings=warnings,
        )
    
    def _extract_components(self, sample: Dict[str, Any], audio_base_path: Optional[str]) -> Tuple:
        """Extract components from ChatML sample."""
        try:
            messages = sample.get("messages", [])
            ref_audio_path = ref_text = target_text = target_audio_path = None
            
            for message in messages:
                role = message["role"]
                content = message["content"]
                
                if role == "user" and isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if item["type"] == "text":
                            text_parts.append(item["text"])
                        elif item["type"] == "audio" and ref_audio_path is None:
                            ref_audio_path = item["audio_url"]
                    
                    if len(text_parts) >= 2:
                        ref_text = text_parts[0]
                        target_text = text_parts[-1].split(":")[-1].strip() if ":" in text_parts[-1] else text_parts[-1]
                
                elif role == "assistant" and isinstance(content, list):
                    for item in content:
                        if item["type"] == "audio":
                            target_audio_path = item["audio_url"]
                            break
            
            # Resolve paths
            if ref_audio_path and audio_base_path and not os.path.isabs(ref_audio_path):
                ref_audio_path = os.path.join(audio_base_path, ref_audio_path)
            if target_audio_path and audio_base_path and not os.path.isabs(target_audio_path):
                target_audio_path = os.path.join(audio_base_path, target_audio_path)
            
            return ref_audio_path, ref_text, target_text, target_audio_path
            
        except Exception:
            return None, None, None, None
    
    def _validate_text(self, text: Optional[str], text_type: str, issues: List[str]) -> bool:
        """Validate text component."""
        if not text:
            issues.append(f"Missing {text_type} text")
            return False
        
        if len(text.strip()) < self.min_text_length:
            issues.append(f"{text_type} text too short")
            return False
        
        # Check Arabic content
        content_analysis = detect_arabic_content(text)
        if not content_analysis["is_arabic"]:
            issues.append(f"{text_type} text has low Arabic content")
            return False
        
        return True
    
    def _validate_audio(self, audio_path: Optional[str], audio_type: str, issues: List[str]) -> bool:
        """Validate audio file."""
        if not audio_path:
            issues.append(f"Missing {audio_type} audio path")
            return False
        
        if not os.path.exists(audio_path):
            issues.append(f"{audio_type} audio file not found")
            return False
        
        try:
            quality = self.audio_analyzer.analyze(audio_path)
            if not quality:
                issues.append(f"Cannot analyze {audio_type} audio")
                return False
            
            if quality.duration < self.min_audio_duration:
                issues.append(f"{audio_type} audio too short ({quality.duration:.1f}s)")
                return False
            
            if quality.duration > self.max_audio_duration:
                issues.append(f"{audio_type} audio too long ({quality.duration:.1f}s)")
                return False
            
        except Exception as e:
            issues.append(f"Cannot load {audio_type} audio: {e}")
            return False
        
        return True
    
    def validate_dataset(self, chatml_file: str, audio_base_path: Optional[str] = None) -> Dict[str, Any]:
        """Validate entire ChatML dataset."""
        logger.info(f"Validating dataset: {chatml_file}")
        
        with open(chatml_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            data = [data]
        
        results = []
        valid_count = 0
        
        for i, sample in enumerate(data):
            try:
                result = self.validate_sample(sample, audio_base_path)
                result.sample_id = f"sample_{i}"
                results.append(result)
                
                if result.is_valid:
                    valid_count += 1
            
            except Exception as e:
                logger.error(f"Validation failed for sample {i}: {e}")
                results.append(ChatMLValidationResult(
                    sample_id=f"sample_{i}",
                    is_valid=False,
                    issues=[f"Validation error: {e}"],
                    warnings=[],
                ))
        
        valid_ratio = valid_count / len(results) if results else 0
        
        return {
            "total_samples": len(results),
            "valid_samples": valid_count,
            "valid_ratio": valid_ratio,
            "detailed_results": results,
        }


class ModelTester:
    """Tester for Arabic voice cloning model."""
    
    def test_model_loading(self, model_path: str) -> ModelTestResult:
        """Test model loading."""
        try:
            import time
            start_time = time.time()
            
            from boson_multimodal.model.higgs_audio import HiggsAudioModel
            model = HiggsAudioModel.from_pretrained(model_path)
            
            execution_time = time.time() - start_time
            
            return ModelTestResult(
                test_name="model_loading",
                passed=True,
                execution_time=execution_time,
            )
            
        except Exception as e:
            return ModelTestResult(
                test_name="model_loading",
                passed=False,
                error_message=str(e),
            )
    
    def test_tokenizers(self, text_tokenizer_path: str, audio_tokenizer_path: str) -> List[ModelTestResult]:
        """Test tokenizer loading."""
        results = []
        
        # Test text tokenizer
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_path)
            test_text = "مرحبا بكم في نظام التعرف على الصوت"
            tokens = tokenizer.encode(test_text)
            
            results.append(ModelTestResult(
                test_name="text_tokenizer",
                passed=True,
            ))
        except Exception as e:
            results.append(ModelTestResult(
                test_name="text_tokenizer",
                passed=False,
                error_message=str(e),
            ))
        
        # Test audio tokenizer
        try:
            from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
            audio_tokenizer = load_higgs_audio_tokenizer(audio_tokenizer_path)
            
            results.append(ModelTestResult(
                test_name="audio_tokenizer",
                passed=True,
            ))
        except Exception as e:
            results.append(ModelTestResult(
                test_name="audio_tokenizer",
                passed=False,
                error_message=str(e),
            ))
        
        return results
    
    def test_dataset_loading(self, chatml_file: str) -> ModelTestResult:
        """Test dataset loading."""
        try:
            from arabic_chatml_dataset import ArabicChatMLDataset
            from transformers import AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")
            
            dataset = ArabicChatMLDataset(
                chatml_file=chatml_file,
                audio_tokenizer="bosonai/higgs-audio-v2-tokenizer",
                text_tokenizer=tokenizer,
                validate_files=False,
            )
            
            return ModelTestResult(
                test_name="dataset_loading",
                passed=True,
            )
            
        except Exception as e:
            return ModelTestResult(
                test_name="dataset_loading",
                passed=False,
                error_message=str(e),
            )
    
    def run_all_tests(self, chatml_file: str, model_path: str = "bosonai/higgs-audio-v2-generation-3B-base") -> Dict[str, Any]:
        """Run comprehensive test suite."""
        logger.info("Running test suite...")
        
        all_results = []
        
        # Test model loading
        all_results.append(self.test_model_loading(model_path))
        
        # Test tokenizers
        all_results.extend(self.test_tokenizers(model_path, "bosonai/higgs-audio-v2-tokenizer"))
        
        # Test dataset
        all_results.append(self.test_dataset_loading(chatml_file))
        
        passed_tests = [r for r in all_results if r.passed]
        
        return {
            "total_tests": len(all_results),
            "passed_tests": len(passed_tests),
            "success_rate": len(passed_tests) / len(all_results) if all_results else 0,
            "detailed_results": all_results,
        }


def validate_chatml_file(chatml_file: str, audio_base_path: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to validate a ChatML file."""
    validator = ChatMLDataValidator()
    return validator.validate_dataset(chatml_file, audio_base_path)


def test_model_components(chatml_file: str) -> Dict[str, Any]:
    """Convenience function to test model components."""
    tester = ModelTester()
    return tester.run_all_tests(chatml_file)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python validation_testing.py <chatml_file> [audio_base_path]")
        sys.exit(1)
    
    chatml_file = sys.argv[1]
    audio_base_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Run validation
    print("Running ChatML validation...")
    validation_results = validate_chatml_file(chatml_file, audio_base_path)
    print(f"Validation: {validation_results['valid_samples']}/{validation_results['total_samples']} samples valid")
    
    # Run model tests
    print("\nRunning model tests...")
    test_results = test_model_components(chatml_file)
    print(f"Tests: {test_results['passed_tests']}/{test_results['total_tests']} passed")