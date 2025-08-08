#!/usr/bin/env python3
"""
FIXED Arabic+English Dataset Processor for Higgs-Audio V2 Zero-Shot Voice Cloning
Corrected version with proper audio tokenization validation and error handling.
"""

import json
import os
import sys
import torch
import torchaudio
import librosa
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import arabic_reshaper
from bidi.algorithm import get_display
import re
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add higgs-audio to path
sys.path.append('/workspace/higgs-audio')

from boson_multimodal.data_types import ChatMLSample, Message, AudioContent, TextContent
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
import soundfile as sf


@dataclass
class DatasetSample:
    """Single sample from the Arabic+English dataset"""
    id: str
    audio_file: str
    transcript_file: str
    ref_audio_file: str
    ref_transcript: str
    duration: float
    language: str = "auto"  # Will be detected


class ZeroShotVoiceCloningProcessorFixed:
    """FIXED Processor for zero-shot voice cloning dataset with ChatML conversion"""
    
    def __init__(
        self,
        dataset_dir: str,
        output_dir: str,
        audio_tokenizer_path: str = "bosonai/higgs-audio-v2-tokenizer",
        target_sample_rate: int = 16000,
        max_audio_duration: float = 30.0,
        min_audio_duration: float = 0.5,
        num_workers: int = 8,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_audio_duration = max_audio_duration
        self.min_audio_duration = min_audio_duration
        self.target_sample_rate = target_sample_rate
        self.num_workers = num_workers
        
        # Load audio tokenizer with validation
        print("Loading audio tokenizer...")
        self.audio_tokenizer = load_higgs_audio_tokenizer(
            audio_tokenizer_path, 
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # CRITICAL: Validate audio tokenizer configuration
        print(f"Audio tokenizer configuration:")
        print(f"  - Number of codebooks (n_q): {self.audio_tokenizer.n_q}")
        print(f"  - Codebook size: {getattr(self.audio_tokenizer, 'codebook_size', 'Unknown')}")
        print(f"  - Sample rate: {getattr(self.audio_tokenizer, 'sample_rate', 'Unknown')}")
        
        if self.audio_tokenizer.n_q != 8:
            raise ValueError(f"Expected 8 codebooks, got {self.audio_tokenizer.n_q}")
        
        # Language detection patterns
        self.arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
        self.english_pattern = re.compile(r'[a-zA-Z]')
        
    def detect_language(self, text: str) -> str:
        """Detect if text is Arabic, English, or mixed"""
        arabic_chars = len(self.arabic_pattern.findall(text))
        english_chars = len(self.english_pattern.findall(text))
        
        if arabic_chars > english_chars:
            return "arabic"
        elif english_chars > arabic_chars:
            return "english"
        else:
            return "mixed"
    
    def normalize_arabic_text(self, text: str) -> str:
        """Normalize Arabic text for better processing"""
        # Remove diacritics (optional - you might want to keep them)
        text = re.sub(r'[\u064B-\u0652\u0670\u0640]', '', text)
        
        # Normalize Arabic letters
        text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
        text = text.replace('ة', 'ه')
        text = text.replace('ى', 'ي')
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def load_metadata(self) -> List[DatasetSample]:
        """Load dataset metadata"""
        metadata_path = self.dataset_dir / "metadata.json"
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        samples = []
        for sample_data in tqdm(metadata['samples'], desc="Loading metadata"):
            # Detect language from transcript
            transcript_path = self.dataset_dir / sample_data['transcript_file']
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = f.read().strip()
            
            language = self.detect_language(transcript)
            
            samples.append(DatasetSample(
                id=sample_data['id'],
                audio_file=str(self.dataset_dir / sample_data['audio_file']),
                transcript_file=str(transcript_path),
                ref_audio_file=str(self.dataset_dir / sample_data['ref_audio_file']),
                ref_transcript=sample_data['ref_transcript'],
                duration=sample_data['duration'],
                language=language
            ))
        
        print(f"Loaded {len(samples)} samples")
        return samples
    
    def process_audio(self, audio_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Process audio file to target format"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.target_sample_rate, mono=True)
            
            # Validate audio
            if len(audio) == 0:
                print(f"Warning: Empty audio file {audio_path}")
                return None, None
            
            # Check duration
            duration = len(audio) / sr
            if not (self.min_audio_duration <= duration <= self.max_audio_duration):
                print(f"Warning: Audio duration {duration:.2f}s outside range [{self.min_audio_duration}, {self.max_audio_duration}]")
                return None, None
            
            return audio, sr
            
        except Exception as e:
            print(f"Error processing audio {audio_path}: {e}")
            return None, None
    
    def validate_audio_tokens(self, audio_tokens: torch.Tensor, audio_file: str) -> bool:
        """CRITICAL: Validate audio tokens for corruption"""
        try:
            # Check tensor shape
            if audio_tokens.dim() != 2:
                print(f"ERROR: Invalid audio tokens shape {audio_tokens.shape} for {audio_file}")
                return False
            
            num_codebooks, seq_len = audio_tokens.shape
            
            # Check codebook count
            if num_codebooks != self.audio_tokenizer.n_q:
                print(f"ERROR: Expected {self.audio_tokenizer.n_q} codebooks, got {num_codebooks} for {audio_file}")
                return False
            
            # Check sequence length
            if seq_len < 10:  # Minimum reasonable sequence length
                print(f"ERROR: Sequence too short ({seq_len}) for {audio_file}")
                return False
            
            # Check for valid tokens per codebook
            for cb_idx in range(num_codebooks):
                cb_tokens = audio_tokens[cb_idx]
                valid_tokens = cb_tokens[cb_tokens != -100]
                
                if len(valid_tokens) < 5:  # Minimum valid tokens per codebook
                    print(f"ERROR: Codebook {cb_idx} has only {len(valid_tokens)} valid tokens for {audio_file}")
                    return False
                
                # Check token range
                if len(valid_tokens) > 0:
                    min_token = valid_tokens.min().item()
                    max_token = valid_tokens.max().item()
                    
                    if min_token < 0 or max_token > 1024:  # Reasonable token range
                        print(f"ERROR: Invalid token range [{min_token}, {max_token}] in codebook {cb_idx} for {audio_file}")
                        return False
                    
                    # Check for excessive token dominance (collapse indicator)
                    from collections import Counter
                    token_counts = Counter(valid_tokens.tolist())
                    most_common_count = token_counts.most_common(1)[0][1]
                    dominance = most_common_count / len(valid_tokens)
                    
                    if dominance > 0.8:  # More than 80% dominance is suspicious
                        print(f"WARNING: High token dominance ({dominance:.1%}) in codebook {cb_idx} for {audio_file}")
                        # Don't fail, but warn
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to validate audio tokens for {audio_file}: {e}")
            return False
    
    def tokenize_audio_safely(self, audio_file: str) -> Optional[torch.Tensor]:
        """CRITICAL: Safely tokenize audio with comprehensive validation"""
        try:
            # Check if file exists
            if not Path(audio_file).exists():
                print(f"ERROR: Audio file not found: {audio_file}")
                return None
            
            # Tokenize audio
            print(f"Tokenizing: {audio_file}")
            audio_tokens = self.audio_tokenizer.encode(audio_file)
            
            # Validate the result
            if audio_tokens is None:
                print(f"ERROR: Audio tokenizer returned None for {audio_file}")
                return None
            
            if not isinstance(audio_tokens, torch.Tensor):
                print(f"ERROR: Audio tokenizer returned non-tensor ({type(audio_tokens)}) for {audio_file}")
                return None
            
            # Comprehensive validation
            if not self.validate_audio_tokens(audio_tokens, audio_file):
                print(f"ERROR: Audio tokens failed validation for {audio_file}")
                return None
            
            print(f"✅ Successfully tokenized {audio_file}: shape {audio_tokens.shape}")
            return audio_tokens
            
        except Exception as e:
            print(f"ERROR: Exception during audio tokenization for {audio_file}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_zero_shot_chatml_sample(self, sample: DatasetSample) -> Optional[ChatMLSample]:
        """Create ChatML sample for zero-shot voice cloning with FIXED tokenization"""
        try:
            # Load transcript
            with open(sample.transcript_file, 'r', encoding='utf-8') as f:
                target_text = f.read().strip()
            
            # Normalize text based on language
            if sample.language in ['arabic', 'mixed']:
                target_text = self.normalize_arabic_text(target_text)
                ref_transcript = self.normalize_arabic_text(sample.ref_transcript)
            else:
                ref_transcript = sample.ref_transcript
            
            # Process reference audio
            ref_audio, ref_sr = self.process_audio(sample.ref_audio_file)
            if ref_audio is None:
                print(f"Failed to process reference audio: {sample.ref_audio_file}")
                return None
            
            # Process target audio
            target_audio, target_sr = self.process_audio(sample.audio_file)
            if target_audio is None:
                print(f"Failed to process target audio: {sample.audio_file}")
                return None
            
            # CRITICAL FIX: Safe audio tokenization with validation
            ref_audio_tokens = self.tokenize_audio_safely(sample.ref_audio_file)
            if ref_audio_tokens is None:
                print(f"Failed to tokenize reference audio: {sample.ref_audio_file}")
                return None
            
            target_audio_tokens = self.tokenize_audio_safely(sample.audio_file)
            if target_audio_tokens is None:
                print(f"Failed to tokenize target audio: {sample.audio_file}")
                return None
            
            # Create system message for zero-shot voice cloning
            system_message = self._create_system_message(sample.language)
            
            # Create user message with reference audio and transcript (for voice cloning context)
            user_content = [
                TextContent(text=f"Here is a reference audio sample and its transcript: '{ref_transcript}'. Please clone this voice and generate speech for: '{target_text}'"),
                AudioContent(
                    audio_url=sample.ref_audio_file,
                    raw_audio=self._encode_audio_to_base64(ref_audio, ref_sr),
                    audio_tokens=ref_audio_tokens.tolist()  # Reference audio tokens for voice cloning
                )
            ]
            
            # Create assistant message with ONLY target audio (no text leakage)
            assistant_content = [
                AudioContent(
                    audio_url=sample.audio_file,
                    raw_audio=self._encode_audio_to_base64(target_audio, target_sr),
                    audio_tokens=target_audio_tokens.tolist()  # Target audio tokens to generate
                )
            ]
            
            # Create ChatML sample
            messages = [
                Message(role="system", content=system_message),
                Message(role="user", content=user_content),
                Message(role="assistant", content=assistant_content)
            ]
            
            chatml_sample = ChatMLSample(
                messages=messages,
                start_index=2,  # Start training from assistant message
                speaker=f"speaker_{sample.id}",
                misc={
                    "sample_id": sample.id,
                    "language": sample.language,
                    "ref_transcript": ref_transcript,
                    "target_text": target_text,
                    "ref_duration": len(ref_audio) / ref_sr,
                    "target_duration": len(target_audio) / target_sr,
                    "ref_audio_tokens_shape": list(ref_audio_tokens.shape),
                    "target_audio_tokens_shape": list(target_audio_tokens.shape),
                    "codebook_count": self.audio_tokenizer.n_q,
                    "validation_passed": True  # Mark as validated
                }
            )
            
            print(f"✅ Created valid ChatML sample for {sample.id}")
            return chatml_sample
            
        except Exception as e:
            print(f"ERROR: Failed to create ChatML sample for {sample.id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_system_message(self, language: str) -> str:
        """Create system message based on language"""
        if language == "arabic":
            return "You are an AI assistant specialized in Arabic zero-shot voice cloning. Generate natural-sounding Arabic speech that matches the provided reference voice."
        elif language == "english":
            return "You are an AI assistant specialized in English zero-shot voice cloning. Generate natural-sounding English speech that matches the provided reference voice."
        else:
            return "You are an AI assistant specialized in multilingual zero-shot voice cloning. Generate natural-sounding speech that matches the provided reference voice."
    
    def _encode_audio_to_base64(self, audio: np.ndarray, sample_rate: int) -> str:
        """Encode audio to base64 string"""
        try:
            import base64
            import io
            
            # Convert to 16-bit PCM
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # Create WAV in memory
            buffer = io.BytesIO()
            sf.write(buffer, audio_int16, sample_rate, format='WAV')
            buffer.seek(0)
            
            # Encode to base64
            audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            return audio_base64
            
        except Exception as e:
            print(f"Error encoding audio to base64: {e}")
            return ""
    
    def process_single_sample(self, sample: DatasetSample) -> Optional[ChatMLSample]:
        """Process a single sample (for multiprocessing)"""
        return self.create_zero_shot_chatml_sample(sample)
    
    def process_dataset(self) -> Dict[str, List]:
        """Process the entire dataset with FIXED validation"""
        samples = self.load_metadata()
        
        language_groups = {
            'arabic': [],
            'english': [],
            'mixed': []
        }
        
        failed_samples = []
        validation_stats = {
            'total_samples': len(samples),
            'processed_samples': 0,
            'failed_samples': 0,
            'tokenization_failures': 0,
            'validation_failures': 0
        }
        
        print(f"\n🔄 Processing {len(samples)} samples with FIXED validation...")
        
        for sample in tqdm(samples, desc="Processing samples"):
            try:
                chatml_sample = self.process_single_sample(sample)
                
                if chatml_sample is not None:
                    language_groups[sample.language].append(chatml_sample)
                    validation_stats['processed_samples'] += 1
                    
                    # Additional validation check
                    if chatml_sample.misc.get('validation_passed', False):
                        print(f"✅ Sample {sample.id} passed all validations")
                    else:
                        print(f"⚠️  Sample {sample.id} processed but validation uncertain")
                else:
                    failed_samples.append(sample.id)
                    validation_stats['failed_samples'] += 1
                    print(f"❌ Failed to process sample {sample.id}")
                    
            except Exception as e:
                failed_samples.append(sample.id)
                validation_stats['failed_samples'] += 1
                print(f"❌ Exception processing sample {sample.id}: {e}")
        
        # Print validation statistics
        print(f"\n📊 Processing Statistics:")
        print(f"  • Total samples: {validation_stats['total_samples']}")
        print(f"  • Successfully processed: {validation_stats['processed_samples']}")
        print(f"  • Failed samples: {validation_stats['failed_samples']}")
        print(f"  • Success rate: {validation_stats['processed_samples']/validation_stats['total_samples']*100:.1f}%")
        
        # Save processing statistics
        stats_file = self.output_dir / "validation_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(validation_stats, f, indent=2)
        
        # Save processed data
        self._save_processed_data(language_groups, failed_samples)
        
        return language_groups
    
    def _save_processed_data(self, language_groups: Dict[str, List], failed_samples: List[str]):
        """Save processed data to files"""
        # Save each language group
        for language, samples in language_groups.items():
            if not samples:
                continue
                
            output_file = self.output_dir / f"chatml_samples_{language}.json"
            
            # Convert ChatML samples to serializable format
            serializable_samples = []
            for sample in samples:
                serializable_samples.append({
                    'messages': [
                        {
                            'role': msg.role,
                            'content': msg.content if isinstance(msg.content, str) else [
                                {'type': c.type, **{k: v for k, v in c.__dict__.items() if k != 'type'}}
                                for c in msg.content
                            ]
                        }
                        for msg in sample.messages
                    ],
                    'start_index': sample.start_index,
                    'speaker': sample.speaker,
                    'misc': sample.misc
                })
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_samples, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Saved {len(samples)} validated {language} samples to {output_file}")
        
        # Save failed samples log
        failed_file = self.output_dir / "failed_samples.json"
        with open(failed_file, 'w') as f:
            json.dump(failed_samples, f, indent=2)
        
        print(f"📄 Processing complete! Check {self.output_dir} for results.")


def main():
    """Main processing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FIXED Arabic+English dataset processor for Higgs-Audio training")
    parser.add_argument("--dataset_dir", required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", required=True, help="Path to output directory")
    parser.add_argument("--audio_tokenizer_path", default="bosonai/higgs-audio-v2-tokenizer", help="Audio tokenizer path")
    parser.add_argument("--max_duration", type=float, default=30.0, help="Maximum audio duration")
    parser.add_argument("--min_duration", type=float, default=0.5, help="Minimum audio duration")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes")
    
    args = parser.parse_args()
    
    processor = ZeroShotVoiceCloningProcessorFixed(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        audio_tokenizer_path=args.audio_tokenizer_path,
        max_audio_duration=args.max_duration,
        min_audio_duration=args.min_duration,
        num_workers=args.num_workers
    )
    
    language_groups = processor.process_dataset()
    
    print("\n=== FIXED Processing Complete ===")
    for language, samples in language_groups.items():
        print(f"{language.capitalize()}: {len(samples)} validated samples")
    
    print("\n🎉 Data processing completed with comprehensive validation!")
    print("Your data should now be free of tokenization corruption.")


if __name__ == "__main__":
    main()
