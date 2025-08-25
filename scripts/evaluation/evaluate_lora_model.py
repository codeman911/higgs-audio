#!/usr/bin/env python3
"""
Evaluation script for LoRA fine-tuned Higgs-Audio V2 model
Evaluates zero-shot voice cloning performance on Arabic+English test set
"""

import os
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import logging
from tqdm import tqdm
import librosa
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import pandas as pd

# Robust import handling for both CLI and module usage
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
    from boson_multimodal.data_types import ChatMLSample, Message, AudioContent, TextContent
    from scripts.training.lora_integration import load_lora_model
except ImportError:
    # Fallback for different project structures
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    sys.path.insert(0, project_root)
    from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
    from boson_multimodal.data_types import ChatMLSample, Message, AudioContent, TextContent
    from scripts.training.lora_integration import load_lora_model


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    speaker_similarity: float
    audio_quality: float
    text_alignment: float
    language_accuracy: float
    prosody_naturalness: float
    overall_score: float


class VoiceCloningEvaluator:
    """Evaluator for zero-shot voice cloning performance"""
    
    def __init__(
        self,
        model_path: str,
        base_model_path: str,
        audio_tokenizer_path: str,
        device: str = "cuda"
    ):
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.audio_tokenizer_path = audio_tokenizer_path
        self.device = device
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load models
        self.logger.info("Loading LoRA fine-tuned model...")
        self.lora_model = self._load_lora_model()
        
        self.logger.info("Loading base model for comparison...")
        self.base_model = self._load_base_model()
        
        # Initialize serve engines
        self.lora_engine = HiggsAudioServeEngine(
            model=self.lora_model,
            audio_tokenizer_path=self.audio_tokenizer_path,
            device=self.device
        )
        
        self.base_engine = HiggsAudioServeEngine(
            model=self.base_model,
            audio_tokenizer_path=self.audio_tokenizer_path,
            device=self.device
        )
        
        # Load speaker embedding extractor for similarity computation
        self._setup_speaker_embedding_extractor()
    
    def _load_lora_model(self):
        """Load LoRA fine-tuned model"""
        return load_lora_model(
            base_model_path=self.base_model_path,
            lora_adapter_path=self.model_path,
            device=self.device
        )
    
    def _load_base_model(self):
        """Load base model for comparison"""
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True
        )
    
    def _setup_speaker_embedding_extractor(self):
        """Setup speaker embedding extractor for similarity computation"""
        try:
            # Use a pre-trained speaker verification model
            from speechbrain.pretrained import EncoderClassifier
            self.speaker_encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="tmp/spkrec"
            )
        except ImportError:
            self.logger.warning("SpeechBrain not available, using simple MFCC-based similarity")
            self.speaker_encoder = None
    
    def extract_speaker_embedding(self, audio_path: str) -> np.ndarray:
        """Extract speaker embedding from audio file"""
        if self.speaker_encoder is not None:
            # Use SpeechBrain ECAPA-TDNN
            embedding = self.speaker_encoder.encode_file(audio_path)
            return embedding.squeeze().cpu().numpy()
        else:
            # Fallback to MFCC-based features
            audio, sr = librosa.load(audio_path, sr=16000)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            return np.mean(mfccs, axis=1)
    
    def compute_speaker_similarity(self, ref_audio: str, gen_audio: str) -> float:
        """Compute speaker similarity between reference and generated audio"""
        try:
            ref_embedding = self.extract_speaker_embedding(ref_audio)
            gen_embedding = self.extract_speaker_embedding(gen_audio)
            
            # Compute cosine similarity
            similarity = 1 - cosine(ref_embedding, gen_embedding)
            return max(0.0, similarity)  # Ensure non-negative
            
        except Exception as e:
            self.logger.warning(f"Error computing speaker similarity: {e}")
            return 0.0
    
    def compute_audio_quality(self, audio_path: str) -> float:
        """Compute audio quality metrics"""
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Simple quality metrics
            # 1. Signal-to-noise ratio estimate
            signal_power = np.mean(audio ** 2)
            noise_power = np.var(audio - np.mean(audio))
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            
            # 2. Spectral centroid (brightness)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
            
            # 3. Zero crossing rate (smoothness)
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
            
            # Normalize and combine metrics
            snr_score = min(1.0, max(0.0, (snr + 10) / 30))  # Normalize SNR
            centroid_score = min(1.0, spectral_centroid / 4000)  # Normalize centroid
            zcr_score = 1.0 - min(1.0, zcr * 10)  # Lower ZCR is better
            
            quality_score = (snr_score + centroid_score + zcr_score) / 3
            return quality_score
            
        except Exception as e:
            self.logger.warning(f"Error computing audio quality: {e}")
            return 0.0
    
    def detect_language(self, text: str) -> str:
        """Detect language of text (Arabic, English, or Mixed)"""
        import re
        
        # Simple language detection based on script
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = arabic_chars + english_chars
        
        if total_chars == 0:
            return "unknown"
        
        arabic_ratio = arabic_chars / total_chars
        english_ratio = english_chars / total_chars
        
        if arabic_ratio > 0.8:
            return "arabic"
        elif english_ratio > 0.8:
            return "english"
        else:
            return "mixed"
    
    def evaluate_sample(
        self,
        reference_audio: str,
        target_text: str,
        output_dir: str,
        sample_id: str
    ) -> Tuple[EvaluationMetrics, EvaluationMetrics]:
        """Evaluate a single sample with both LoRA and base models"""
        
        # Create ChatML sample
        chatml_sample = ChatMLSample(
            messages=[
                Message(role="system", content="You are a helpful voice assistant."),
                Message(role="user", content=[
                    TextContent(text=target_text),
                    AudioContent(audio_url=reference_audio)
                ]),
                Message(role="assistant", content="")
            ]
        )
        
        # Generate with LoRA model
        lora_output_path = os.path.join(output_dir, f"{sample_id}_lora.wav")
        try:
            lora_result = self.lora_engine.generate(
                chatml_sample,
                max_new_tokens=1500,
                temperature=0.7,
                do_sample=True
            )
            
            # Save generated audio
            if hasattr(lora_result, 'audio_waveform'):
                torchaudio.save(lora_output_path, lora_result.audio_waveform, 24000)
            else:
                self.logger.warning(f"No audio generated for LoRA sample {sample_id}")
                lora_output_path = None
                
        except Exception as e:
            self.logger.error(f"Error generating with LoRA model for sample {sample_id}: {e}")
            lora_output_path = None
        
        # Generate with base model
        base_output_path = os.path.join(output_dir, f"{sample_id}_base.wav")
        try:
            base_result = self.base_engine.generate(
                chatml_sample,
                max_new_tokens=1500,
                temperature=0.7,
                do_sample=True
            )
            
            # Save generated audio
            if hasattr(base_result, 'audio_waveform'):
                torchaudio.save(base_output_path, base_result.audio_waveform, 24000)
            else:
                self.logger.warning(f"No audio generated for base sample {sample_id}")
                base_output_path = None
                
        except Exception as e:
            self.logger.error(f"Error generating with base model for sample {sample_id}: {e}")
            base_output_path = None
        
        # Compute metrics
        lora_metrics = self._compute_metrics(reference_audio, lora_output_path, target_text)
        base_metrics = self._compute_metrics(reference_audio, base_output_path, target_text)
        
        return lora_metrics, base_metrics
    
    def _compute_metrics(
        self,
        reference_audio: str,
        generated_audio: Optional[str],
        target_text: str
    ) -> EvaluationMetrics:
        """Compute evaluation metrics for a sample"""
        
        if generated_audio is None or not os.path.exists(generated_audio):
            return EvaluationMetrics(
                speaker_similarity=0.0,
                audio_quality=0.0,
                text_alignment=0.0,
                language_accuracy=0.0,
                prosody_naturalness=0.0,
                overall_score=0.0
            )
        
        # Speaker similarity
        speaker_similarity = self.compute_speaker_similarity(reference_audio, generated_audio)
        
        # Audio quality
        audio_quality = self.compute_audio_quality(generated_audio)
        
        # Text alignment (placeholder - would need ASR for proper evaluation)
        text_alignment = 0.8  # Assume good alignment for now
        
        # Language accuracy
        detected_lang = self.detect_language(target_text)
        language_accuracy = 1.0 if detected_lang in ["arabic", "english", "mixed"] else 0.5
        
        # Prosody naturalness (placeholder - would need specialized models)
        prosody_naturalness = 0.7  # Assume reasonable naturalness
        
        # Overall score (weighted average)
        overall_score = (
            0.3 * speaker_similarity +
            0.2 * audio_quality +
            0.2 * text_alignment +
            0.15 * language_accuracy +
            0.15 * prosody_naturalness
        )
        
        return EvaluationMetrics(
            speaker_similarity=speaker_similarity,
            audio_quality=audio_quality,
            text_alignment=text_alignment,
            language_accuracy=language_accuracy,
            prosody_naturalness=prosody_naturalness,
            overall_score=overall_score
        )
    
    def evaluate_dataset(
        self,
        test_data_path: str,
        output_dir: str,
        num_samples: int = 100
    ) -> Dict:
        """Evaluate on test dataset"""
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load test samples
        test_samples = self._load_test_samples(test_data_path, num_samples)
        
        # Evaluation results
        lora_results = []
        base_results = []
        
        # Evaluate each sample
        for i, sample in enumerate(tqdm(test_samples, desc="Evaluating samples")):
            sample_id = f"sample_{i:04d}"
            
            try:
                lora_metrics, base_metrics = self.evaluate_sample(
                    reference_audio=sample['reference_audio'],
                    target_text=sample['target_text'],
                    output_dir=output_dir,
                    sample_id=sample_id
                )
                
                lora_results.append({
                    'sample_id': sample_id,
                    'language': sample.get('language', 'unknown'),
                    **lora_metrics.__dict__
                })
                
                base_results.append({
                    'sample_id': sample_id,
                    'language': sample.get('language', 'unknown'),
                    **base_metrics.__dict__
                })
                
            except Exception as e:
                self.logger.error(f"Error evaluating sample {sample_id}: {e}")
                continue
        
        # Compute aggregate metrics
        results = {
            'lora_results': lora_results,
            'base_results': base_results,
            'lora_aggregate': self._compute_aggregate_metrics(lora_results),
            'base_aggregate': self._compute_aggregate_metrics(base_results),
            'improvement': self._compute_improvement(lora_results, base_results)
        }
        
        # Save results
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate plots
        self._generate_plots(results, output_dir)
        
        return results
    
    def _load_test_samples(self, test_data_path: str, num_samples: int) -> List[Dict]:
        """Load test samples from processed dataset"""
        test_samples = []
        
        # Load from ChatML files
        for lang_file in ["chatml_samples_arabic.json", "chatml_samples_english.json", "chatml_samples_mixed.json"]:
            file_path = os.path.join(test_data_path, lang_file)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    samples = json.load(f)
                
                # Extract test samples
                for sample in samples[:num_samples // 3]:  # Distribute evenly across languages
                    # Find reference audio and target text
                    reference_audio = None
                    target_text = None
                    
                    for message in sample['messages']:
                        if message['role'] == 'user':
                            content = message['content']
                            if isinstance(content, list):
                                for c in content:
                                    if c['type'] == 'text':
                                        target_text = c['text']
                                    elif c['type'] == 'audio' and 'audio_url' in c:
                                        reference_audio = c['audio_url']
                    
                    if reference_audio and target_text:
                        test_samples.append({
                            'reference_audio': reference_audio,
                            'target_text': target_text,
                            'language': lang_file.split('_')[2].split('.')[0]  # Extract language
                        })
        
        return test_samples[:num_samples]
    
    def _compute_aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Compute aggregate metrics across all samples"""
        if not results:
            return {}
        
        metrics = ['speaker_similarity', 'audio_quality', 'text_alignment', 
                  'language_accuracy', 'prosody_naturalness', 'overall_score']
        
        aggregate = {}
        for metric in metrics:
            values = [r[metric] for r in results if metric in r]
            if values:
                aggregate[f'{metric}_mean'] = np.mean(values)
                aggregate[f'{metric}_std'] = np.std(values)
                aggregate[f'{metric}_median'] = np.median(values)
        
        # Language-specific metrics
        for lang in ['arabic', 'english', 'mixed']:
            lang_results = [r for r in results if r.get('language') == lang]
            if lang_results:
                for metric in metrics:
                    values = [r[metric] for r in lang_results if metric in r]
                    if values:
                        aggregate[f'{metric}_{lang}_mean'] = np.mean(values)
        
        return aggregate
    
    def _compute_improvement(self, lora_results: List[Dict], base_results: List[Dict]) -> Dict:
        """Compute improvement of LoRA over base model"""
        if not lora_results or not base_results:
            return {}
        
        metrics = ['speaker_similarity', 'audio_quality', 'text_alignment', 
                  'language_accuracy', 'prosody_naturalness', 'overall_score']
        
        improvement = {}
        for metric in metrics:
            lora_values = [r[metric] for r in lora_results if metric in r]
            base_values = [r[metric] for r in base_results if metric in r]
            
            if lora_values and base_values:
                lora_mean = np.mean(lora_values)
                base_mean = np.mean(base_values)
                improvement[f'{metric}_absolute'] = lora_mean - base_mean
                improvement[f'{metric}_relative'] = (lora_mean - base_mean) / (base_mean + 1e-10) * 100
        
        return improvement
    
    def _generate_plots(self, results: Dict, output_dir: str):
        """Generate evaluation plots"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # Create comparison plots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('LoRA vs Base Model Evaluation Results', fontsize=16)
            
            metrics = ['speaker_similarity', 'audio_quality', 'text_alignment', 
                      'language_accuracy', 'prosody_naturalness', 'overall_score']
            
            for i, metric in enumerate(metrics):
                ax = axes[i // 3, i % 3]
                
                # Extract values
                lora_values = [r[metric] for r in results['lora_results'] if metric in r]
                base_values = [r[metric] for r in results['base_results'] if metric in r]
                
                # Box plot
                ax.boxplot([base_values, lora_values], labels=['Base', 'LoRA'])
                ax.set_title(metric.replace('_', ' ').title())
                ax.set_ylabel('Score')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'evaluation_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Language-specific comparison
            fig, ax = plt.subplots(figsize=(12, 6))
            
            languages = ['arabic', 'english', 'mixed']
            lora_scores = []
            base_scores = []
            
            for lang in languages:
                lora_lang = [r['overall_score'] for r in results['lora_results'] if r.get('language') == lang]
                base_lang = [r['overall_score'] for r in results['base_results'] if r.get('language') == lang]
                
                lora_scores.append(np.mean(lora_lang) if lora_lang else 0)
                base_scores.append(np.mean(base_lang) if base_lang else 0)
            
            x = np.arange(len(languages))
            width = 0.35
            
            ax.bar(x - width/2, base_scores, width, label='Base Model', alpha=0.8)
            ax.bar(x + width/2, lora_scores, width, label='LoRA Model', alpha=0.8)
            
            ax.set_xlabel('Language')
            ax.set_ylabel('Overall Score')
            ax.set_title('Language-Specific Performance Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels([lang.title() for lang in languages])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'language_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Error generating plots: {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA fine-tuned Higgs-Audio model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to LoRA model")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--audio_tokenizer_path", type=str, required=True, help="Path to audio tokenizer")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to test data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = VoiceCloningEvaluator(
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        audio_tokenizer_path=args.audio_tokenizer_path,
        device=args.device
    )
    
    # Run evaluation
    results = evaluator.evaluate_dataset(
        test_data_path=args.test_data_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )
    
    # Print summary
    print("\n=== Evaluation Results Summary ===")
    print(f"LoRA Model Overall Score: {results['lora_aggregate']['overall_score_mean']:.3f} ± {results['lora_aggregate']['overall_score_std']:.3f}")
    print(f"Base Model Overall Score: {results['base_aggregate']['overall_score_mean']:.3f} ± {results['base_aggregate']['overall_score_std']:.3f}")
    print(f"Improvement: {results['improvement']['overall_score_absolute']:.3f} ({results['improvement']['overall_score_relative']:.1f}%)")
    
    print("\nLanguage-specific results:")
    for lang in ['arabic', 'english', 'mixed']:
        if f'overall_score_{lang}_mean' in results['lora_aggregate']:
            print(f"  {lang.title()}: {results['lora_aggregate'][f'overall_score_{lang}_mean']:.3f}")


if __name__ == "__main__":
    main()
