#!/usr/bin/env python3
"""
Arabic+English Dataset Processor for Higgs-Audio V2 Zero-Shot Voice Cloning
Converts the provided dataset format to ChatML format for training.
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


class ZeroShotVoiceCloningProcessor:
    """Processor for zero-shot voice cloning dataset with ChatML conversion"""
    
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
        
        # Load audio tokenizer
        print("Loading audio tokenizer...")
        self.audio_tokenizer = load_higgs_audio_tokenizer(
            audio_tokenizer_path, 
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
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
            
            sample = DatasetSample(
                id=sample_data['id'],
                audio_file=str(self.dataset_dir / sample_data['audio_file']),
                transcript_file=str(transcript_path),
                ref_audio_file=str(self.dataset_dir / sample_data['ref_audio_file']),
                ref_transcript=sample_data['ref_transcript'],
                duration=sample_data['duration'],
                language=language
            )
            samples.append(sample)
        
        return samples
    
    def process_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Process audio file to target format"""
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, 
                    new_freq=self.target_sample_rate
                )
                waveform = resampler(waveform)
            
            # Convert to numpy
            waveform_np = waveform.squeeze().numpy()
            
            return waveform_np, self.target_sample_rate
            
        except Exception as e:
            print(f"Error processing audio {audio_path}: {e}")
            return None, None
    
    def create_zero_shot_chatml_sample(self, sample: DatasetSample) -> Optional[ChatMLSample]:
        """Create ChatML sample for zero-shot voice cloning"""
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
                return None
            
            # Check duration constraints
            ref_duration = len(ref_audio) / ref_sr
            if not (self.min_audio_duration <= ref_duration <= self.max_audio_duration):
                return None
            
            # CRITICAL FIX: Generate audio tokens using the audio tokenizer
            # This ensures we get proper 8-codebook audio tokens
            print(f"Tokenizing reference audio: {sample.ref_audio_file}")
            try:
                ref_audio_tokens = self.audio_tokenizer.encode(sample.ref_audio_file)
                print(f"Reference audio tokens shape: {ref_audio_tokens.shape}")
                
                # Validate codebook dimensions
                if ref_audio_tokens.shape[0] != self.audio_tokenizer.n_q:
                    raise ValueError(f"Reference audio tokens have {ref_audio_tokens.shape[0]} codebooks, expected {self.audio_tokenizer.n_q}")
                
            except Exception as e:
                print(f"Error tokenizing reference audio {sample.ref_audio_file}: {e}")
                return None
            
            # Process target audio and generate tokens
            target_audio, target_sr = self.process_audio(sample.audio_file)
            if target_audio is None:
                return None
            
            target_duration = len(target_audio) / target_sr
            if not (self.min_audio_duration <= target_duration <= self.max_audio_duration):
                return None
            
            # CRITICAL FIX: Generate target audio tokens using the audio tokenizer
            print(f"Tokenizing target audio: {sample.audio_file}")
            try:
                target_audio_tokens = self.audio_tokenizer.encode(sample.audio_file)
                print(f"Target audio tokens shape: {target_audio_tokens.shape}")
                
                # Validate codebook dimensions
                if target_audio_tokens.shape[0] != self.audio_tokenizer.n_q:
                    raise ValueError(f"Target audio tokens have {target_audio_tokens.shape[0]} codebooks, expected {self.audio_tokenizer.n_q}")
                    
            except Exception as e:
                print(f"Error tokenizing target audio {sample.audio_file}: {e}")
                return None
            
            # Create system message for zero-shot voice cloning
            system_message = self._create_system_message(sample.language)
            
            # Create user message with reference audio AND reference transcript
            user_content = [
                TextContent(text=f"<ref_text>{ref_transcript}</ref_text>\n\n<text>{target_text}</text>"),
                AudioContent(
                    audio_url=sample.ref_audio_file,
                    raw_audio=self._encode_audio_to_base64(ref_audio, ref_sr),
                    audio_tokens=ref_audio_tokens.tolist()  # Add tokenized audio
                )
            ]
            
            # Create assistant message with tokenized target audio
            assistant_content = [
                AudioContent(
                    audio_url=sample.audio_file,
                    raw_audio=self._encode_audio_to_base64(target_audio, target_sr),
                    audio_tokens=target_audio_tokens.tolist()  # Add tokenized audio
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
                    "ref_duration": ref_duration,
                    "target_duration": target_duration,
                    "ref_audio_tokens_shape": list(ref_audio_tokens.shape),
                    "target_audio_tokens_shape": list(target_audio_tokens.shape),
                    "codebook_count": self.audio_tokenizer.n_q
                }
            )
            
            print(f"✅ Created ChatML sample with {self.audio_tokenizer.n_q}-codebook audio tokens")
            return chatml_sample
            
        except Exception as e:
            print(f"Error creating ChatML sample for {sample.id}: {e}")
            return None
    
    def _create_system_message(self, language: str) -> str:
        """Create system message based on language"""
        if language == "arabic":
            return """أنت مساعد ذكي مصمم لتحويل النص إلى كلام. مهمتك هي توليد كلام طبيعي وواضح باستخدام الصوت المرجعي المقدم. احرص على الحفاظ على خصائص الصوت الأصلي مثل النبرة والسرعة والأسلوب."""
        elif language == "english":
            return """You are an AI assistant designed to convert text into speech. Your task is to generate natural and clear speech using the provided reference voice. Maintain the original voice characteristics such as tone, pace, and style."""
        else:  # mixed
            return """You are an AI assistant designed to convert text into speech in both Arabic and English. Your task is to generate natural and clear speech using the provided reference voice, maintaining the original voice characteristics such as tone, pace, and style. Handle code-switching between languages naturally."""
    
    def _encode_audio_to_base64(self, audio: np.ndarray, sample_rate: int) -> str:
        """Encode audio to base64 string"""
        import base64
        import io
        
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Create WAV in memory
        buffer = io.BytesIO()
        sf.write(buffer, audio_int16, sample_rate, format='WAV')
        buffer.seek(0)
        
        # Encode to base64
        audio_bytes = buffer.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return audio_base64
    
    def process_single_sample(self, sample: DatasetSample) -> Optional[Dict]:
        """Process a single sample (for multiprocessing)"""
        chatml_sample = self.create_zero_shot_chatml_sample(sample)
        if chatml_sample is None:
            return None
        
        return {
            'sample_id': sample.id,
            'chatml_sample': chatml_sample,
            'language': sample.language
        }
    
    def process_dataset(self) -> Dict[str, List]:
        """Process the entire dataset"""
        print("Loading dataset metadata...")
        samples = self.load_metadata()
        
        print(f"Found {len(samples)} samples")
        
        # Filter samples by duration
        valid_samples = []
        for sample in samples:
            if self.min_audio_duration <= sample.duration <= self.max_audio_duration:
                valid_samples.append(sample)
        
        print(f"Filtered to {len(valid_samples)} valid samples")
        
        # Process samples
        processed_samples = []
        failed_samples = []
        
        print("Processing samples...")
        for sample in tqdm(valid_samples):
            result = self.process_single_sample(sample)
            if result is not None:
                processed_samples.append(result)
            else:
                failed_samples.append(sample.id)
        
        print(f"Successfully processed: {len(processed_samples)}")
        print(f"Failed samples: {len(failed_samples)}")
        
        # Group by language
        language_groups = {
            'arabic': [],
            'english': [],
            'mixed': []
        }
        
        for result in processed_samples:
            language_groups[result['language']].append(result['chatml_sample'])
        
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
            
            print(f"Saved {len(samples)} {language} samples to {output_file}")
        
        # Save failed samples log
        failed_file = self.output_dir / "failed_samples.json"
        with open(failed_file, 'w') as f:
            json.dump(failed_samples, f, indent=2)
        
        # Save processing statistics
        stats = {
            'total_processed': sum(len(samples) for samples in language_groups.values()),
            'arabic_samples': len(language_groups['arabic']),
            'english_samples': len(language_groups['english']),
            'mixed_samples': len(language_groups['mixed']),
            'failed_samples': len(failed_samples),
            'processing_config': {
                'max_audio_duration': self.max_audio_duration,
                'min_audio_duration': self.min_audio_duration,
                'target_sample_rate': self.target_sample_rate
            }
        }
        
        stats_file = self.output_dir / "processing_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Processing statistics saved to {stats_file}")


def main():
    """Main processing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process Arabic+English dataset for Higgs-Audio training")
    parser.add_argument("--dataset_dir", required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", required=True, help="Path to output directory")
    parser.add_argument("--audio_tokenizer_path", default="bosonai/higgs-audio-v2-tokenizer", help="Audio tokenizer path")
    parser.add_argument("--max_duration", type=float, default=30.0, help="Maximum audio duration")
    parser.add_argument("--min_duration", type=float, default=0.5, help="Minimum audio duration")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes")
    
    args = parser.parse_args()
    
    processor = ZeroShotVoiceCloningProcessor(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        audio_tokenizer_path=args.audio_tokenizer_path,
        max_audio_duration=args.max_duration,
        min_audio_duration=args.min_duration,
        num_workers=args.num_workers
    )
    
    language_groups = processor.process_dataset()
    
    print("\n=== Processing Complete ===")
    for language, samples in language_groups.items():
        print(f"{language.capitalize()}: {len(samples)} samples")


if __name__ == "__main__":
    main()
