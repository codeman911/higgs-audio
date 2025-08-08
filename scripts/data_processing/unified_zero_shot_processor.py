#!/usr/bin/env python3
"""
UNIFIED Zero-Shot Voice Cloning Data Processor for Higgs-Audio V2
Uses the SAME approach as the working inference script but supports Arabic+English datasets.

CRITICAL: This processor does NOT perform audio tokenization during preprocessing.
Audio tokenization happens during training/inference, just like the working inference pipeline.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from tqdm import tqdm
import librosa
import soundfile as sf
import numpy as np
import re

# Robust import handling for both CLI and module usage
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from boson_multimodal.data_types import ChatMLSample, Message, AudioContent, TextContent
except ImportError:
    # Fallback for different project structures
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    sys.path.insert(0, project_root)
    from boson_multimodal.data_types import ChatMLSample, Message, AudioContent, TextContent


class UnifiedZeroShotProcessor:
    """
    UNIFIED processor that uses the SAME approach as working inference script.
    
    Key Principles:
    1. NO audio tokenization during preprocessing (just like inference)
    2. Store only file paths and metadata
    3. Audio tokenization happens during training/inference
    4. Support Arabic+English dataset format
    """
    
    def __init__(
        self,
        dataset_path: str,
        output_dir: str,
        target_sample_rate: int = 24000,
        max_duration: float = 30.0,
        min_duration: float = 0.5
    ):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.target_sample_rate = target_sample_rate
        self.max_duration = max_duration
        self.min_duration = min_duration
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Language detection patterns
        self.arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
        self.english_pattern = re.compile(r'[a-zA-Z]')
        
        self.logger.info("🚀 UNIFIED Zero-Shot Processor initialized")
        self.logger.info("📝 Using SAME approach as working inference script")
        self.logger.info("⚠️  NO audio tokenization during preprocessing")
    
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
    
    def load_metadata(self) -> Dict:
        """Load the metadata.json file"""
        metadata_file = self.dataset_path / "metadata.json"
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"metadata.json not found in {self.dataset_path}")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        self.logger.info(f"📊 Loaded metadata with {len(metadata['samples'])} samples")
        
        return metadata
    
    def read_transcript_file(self, transcript_file: str) -> str:
        """Read text from transcript file"""
        transcript_path = self.dataset_path / transcript_file
        
        if not transcript_path.exists():
            self.logger.warning(f"Transcript file not found: {transcript_file}")
            return ""
        
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            return text
        except Exception as e:
            self.logger.warning(f"Error reading transcript {transcript_file}: {e}")
            return ""
    
    def validate_audio_file(self, audio_file: str) -> bool:
        """Validate audio file exists and has reasonable duration"""
        audio_path = self.dataset_path / audio_file
        
        if not audio_path.exists():
            self.logger.warning(f"Audio file not found: {audio_file}")
            return False
        
        try:
            # Quick duration check without full loading
            info = sf.info(audio_path)
            duration = info.frames / info.samplerate
            
            if duration < self.min_duration or duration > self.max_duration:
                self.logger.warning(f"Audio duration {duration:.2f}s outside range [{self.min_duration}, {self.max_duration}] for {audio_file}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Error validating audio {audio_file}: {e}")
            return False
    
    def create_system_message(self, language: str) -> str:
        """Create system message based on language"""
        if language == "arabic":
            return "You are an AI assistant specialized in Arabic zero-shot voice cloning. Generate natural-sounding Arabic speech that matches the provided reference voice."
        elif language == "english":
            return "You are an AI assistant specialized in English zero-shot voice cloning. Generate natural-sounding English speech that matches the provided reference voice."
        else:
            return "You are an AI assistant specialized in multilingual zero-shot voice cloning. Generate natural-sounding speech that matches the provided reference voice."
    
    def create_chatml_sample(self, sample: Dict) -> Optional[ChatMLSample]:
        """
        Convert a single sample to ChatML format using the SAME approach as inference script.
        
        CRITICAL: NO audio tokenization here - just like the working inference script!
        FIXED: Set raw_audio to file path for compatibility with distributed trainer
        """
        
        # Read target text from file
        target_text = self.read_transcript_file(sample['transcript_file'])
        if not target_text:
            self.logger.warning(f"No target text for sample {sample['id']}")
            return None
        
        # Get reference transcript
        ref_transcript = sample.get('ref_transcript', '')
        
        # Detect language and normalize text
        language = self.detect_language(target_text)
        if language in ['arabic', 'mixed']:
            target_text = self.normalize_arabic_text(target_text)
            ref_transcript = self.normalize_arabic_text(ref_transcript)
        
        # Validate audio files exist and have reasonable duration
        if not self.validate_audio_file(sample['ref_audio_file']):
            self.logger.warning(f"Invalid reference audio for sample {sample['id']}")
            return None
            
        if not self.validate_audio_file(sample['audio_file']):
            self.logger.warning(f"Invalid target audio for sample {sample['id']}")
            return None
        
        # Get absolute paths for audio files
        ref_audio_path = str(self.dataset_path / sample['ref_audio_file'])
        target_audio_path = str(self.dataset_path / sample['audio_file'])
        
        # Create system message based on language
        system_message = self.create_system_message(language)
        
        # Create ChatML messages using the SAME structure as working inference script
        # FIXED: Set raw_audio to actual file path for distributed trainer compatibility
        messages = [
            # System message: Instructions for voice cloning
            Message(
                role="system",
                content=system_message
            ),
            
            # User message: Target text + Reference audio (voice to clone)
            Message(
                role="user", 
                content=[
                    TextContent(text=f"Here is a reference audio sample and its transcript: '{ref_transcript}'. Please clone this voice and generate speech for: '{target_text}'"),
                    AudioContent(
                        audio_url=ref_audio_path,
                        raw_audio=ref_audio_path,  # FIXED: Set to actual path for trainer compatibility
                        duration=None  # Will be computed during training - SAME as inference
                        # NO audio_tokens here - SAME as inference script!
                    )
                ]
            ),
            
            # Assistant message: Target audio (what model should generate)
            Message(
                role="assistant",
                content=[
                    AudioContent(
                        audio_url=target_audio_path,
                        raw_audio=target_audio_path,  # FIXED: Set to actual path for trainer compatibility
                        duration=sample.get('duration')
                        # NO audio_tokens here - SAME as inference script!
                    )
                ]
            )
        ]
        
        # Create ChatML sample using SAME structure as inference script
        chatml_sample = ChatMLSample(
            messages=messages,
            start_index=2,  # Start training from assistant message
            speaker=f"speaker_{sample['id']}",  # Use sample ID as speaker identifier
            misc={
                'sample_id': sample['id'],
                'language': language,
                'ref_transcript': ref_transcript,
                'target_text': target_text,
                'duration': sample.get('duration', 0.0),
                'processing_approach': 'unified_no_tokenization',  # Mark the approach used
                'compatible_with_inference': True  # Mark as compatible
            }
        )
        
        return chatml_sample
    
    def _serialize_content(self, content):
        """Serialize message content for JSON storage - SAME as inference script"""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            serialized = []
            for item in content:
                if isinstance(item, TextContent):
                    serialized.append({
                        'type': 'text',
                        'text': item.text
                    })
                elif isinstance(item, AudioContent):
                    serialized.append({
                        'type': 'audio',
                        'audio_url': item.audio_url,
                        'raw_audio': item.raw_audio,  # FIXED: Now contains actual file path
                        'duration': item.duration,
                        'offset': getattr(item, 'offset', None)
                        # NO audio_tokens - SAME as inference script!
                    })
            return serialized
        else:
            return str(content)
    
    def create_train_val_split(self, chatml_samples: List[Dict], train_ratio: float = 0.9) -> Tuple[List[Dict], List[Dict]]:
        """Create train/validation split compatible with distributed trainer"""
        import random
        
        # Shuffle samples for random split
        samples_copy = chatml_samples.copy()
        random.shuffle(samples_copy)
        
        # Calculate split point
        split_point = int(len(samples_copy) * train_ratio)
        
        train_samples = samples_copy[:split_point]
        val_samples = samples_copy[split_point:]
        
        self.logger.info(f"� Created train/val split:")
        self.logger.info(f"   • Training samples: {len(train_samples)}")
        self.logger.info(f"   • Validation samples: {len(val_samples)}")
        
        return train_samples, val_samples
    
    def save_chatml_samples(self, chatml_samples: List[Dict], filename: str = "chatml_samples.json"):
        """Save ChatML samples to JSON file - FIXED: Compatible with distributed trainer"""
        
        # Create train/val split for distributed trainer compatibility
        train_samples, val_samples = self.create_train_val_split(chatml_samples, train_ratio=0.9)
        
        # Save train samples (distributed trainer expects this file)
        train_file = self.output_dir / "train_chatml_samples.json"
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_samples, f, ensure_ascii=False, indent=2)
        self.logger.info(f"💾 Saved {len(train_samples)} training samples to {train_file}")
        
        # Save validation samples (distributed trainer expects this file)
        val_file = self.output_dir / "val_chatml_samples.json"
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_samples, f, ensure_ascii=False, indent=2)
        self.logger.info(f"💾 Saved {len(val_samples)} validation samples to {val_file}")
        
        # Also save combined file for backward compatibility
        output_file = self.output_dir / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chatml_samples, f, ensure_ascii=False, indent=2)
        self.logger.info(f"💾 Saved {len(chatml_samples)} combined samples to {output_file}")
        
        # Save summary statistics
        self._save_processing_summary(chatml_samples)
        
        # Create processing stats file for distributed trainer
        self._save_processing_stats_for_trainer(chatml_samples)
    
    def _save_processing_stats_for_trainer(self, chatml_samples: List[Dict]):
        """Save processing stats in format expected by distributed trainer"""
        
        # Calculate language distribution
        language_counts = {}
        total_duration = 0
        
        for sample in chatml_samples:
            lang = sample['misc'].get('language', 'unknown')
            language_counts[lang] = language_counts.get(lang, 0) + 1
            total_duration += sample['misc'].get('duration', 0)
        
        # Create stats in format expected by distributed trainer
        processing_stats = {
            'manifest_metadata': {
                'total_samples': len(chatml_samples),
                'total_duration_hours': total_duration / 3600.0,  # Convert to hours
                'directories_processed': 1,  # Single dataset directory
                'language_distribution': language_counts,
                'processing_approach': 'unified_no_tokenization',
                'compatible_with_inference': True,
                'compatible_with_distributed_trainer': True
            },
            'processing_config': {
                'target_sample_rate': self.target_sample_rate,
                'max_duration': self.max_duration,
                'min_duration': self.min_duration
            }
        }
        
        stats_file = self.output_dir / "processing_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(processing_stats, f, indent=2)
        
        self.logger.info(f"📄 Processing stats for trainer saved to {stats_file}")
    
    def process_dataset(self, max_samples: Optional[int] = None) -> List[Dict]:
        """Process the entire dataset and convert to ChatML format"""
        
        # Load metadata
        metadata = self.load_metadata()
        samples = metadata['samples']
        
        if max_samples:
            samples = samples[:max_samples]
            self.logger.info(f"🔢 Processing only first {max_samples} samples for testing")
        
        chatml_samples = []
        failed_samples = []
        language_stats = {'arabic': 0, 'english': 0, 'mixed': 0}
        
        self.logger.info(f"🔄 Processing {len(samples)} samples...")
        
        for sample in tqdm(samples, desc="Converting to ChatML"):
            try:
                chatml_sample = self.create_chatml_sample(sample)
                
                if chatml_sample:
                    # Convert to serializable format
                    serializable_sample = {
                        'messages': [
                            {
                                'role': msg.role,
                                'content': self._serialize_content(msg.content)
                            }
                            for msg in chatml_sample.messages
                        ],
                        'start_index': chatml_sample.start_index,
                        'speaker': chatml_sample.speaker,
                        'misc': chatml_sample.misc
                    }
                    
                    chatml_samples.append(serializable_sample)
                    language_stats[chatml_sample.misc['language']] += 1
                    
                else:
                    failed_samples.append(sample['id'])
                    
            except Exception as e:
                self.logger.error(f"Error processing sample {sample['id']}: {e}")
                failed_samples.append(sample['id'])
        
        # Log statistics
        self.logger.info(f"✅ Successfully processed {len(chatml_samples)} samples")
        self.logger.info(f"❌ Failed to process {len(failed_samples)} samples")
        self.logger.info(f"📊 Language distribution:")
        for lang, count in language_stats.items():
            self.logger.info(f"   • {lang.capitalize()}: {count} samples")
        
        return chatml_samples
    
    def _save_processing_summary(self, chatml_samples: List[Dict]):
        """Save processing summary and statistics - SAME as inference script"""
        
        # Calculate language distribution
        language_counts = {}
        total_duration = 0
        
        for sample in chatml_samples:
            lang = sample['misc'].get('language', 'unknown')
            language_counts[lang] = language_counts.get(lang, 0) + 1
            total_duration += sample['misc'].get('duration', 0)
        
        summary = {
            'total_samples': len(chatml_samples),
            'processing_approach': 'unified_no_tokenization',
            'compatible_with_inference': True,
            'language_distribution': language_counts,
            'processing_config': {
                'target_sample_rate': self.target_sample_rate,
                'max_duration': self.max_duration,
                'min_duration': self.min_duration
            },
            'sample_statistics': {
                'total_duration': total_duration,
                'avg_duration': total_duration / len(chatml_samples) if chatml_samples else 0,
                'unique_speakers': len(set([s['speaker'] for s in chatml_samples]))
            }
        }
        
        summary_file = self.output_dir / "processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"📄 Processing summary saved to {summary_file}")
        self.logger.info(f"📊 Total samples: {summary['total_samples']}")
        self.logger.info(f"⏱️  Total duration: {summary['sample_statistics']['total_duration']:.2f} seconds")
        
        # Save language-specific files
        self._save_language_specific_files(chatml_samples)
    
    def _save_language_specific_files(self, chatml_samples: List[Dict]):
        """Save separate files for each language"""
        language_groups = {}
        
        for sample in chatml_samples:
            lang = sample['misc'].get('language', 'unknown')
            if lang not in language_groups:
                language_groups[lang] = []
            language_groups[lang].append(sample)
        
        # Save each language group
        for language, samples in language_groups.items():
            if samples:
                lang_file = self.output_dir / f"chatml_samples_{language}.json"
                with open(lang_file, 'w', encoding='utf-8') as f:
                    json.dump(samples, f, ensure_ascii=False, indent=2)
                
                self.logger.info(f"💾 Saved {len(samples)} {language} samples to {lang_file}")


def main():
    """Main processing function"""
    parser = argparse.ArgumentParser(description="UNIFIED zero-shot voice cloning dataset processor (compatible with inference)")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed data")
    parser.add_argument("--max_samples", type=int, help="Maximum samples to process (for testing)")
    parser.add_argument("--target_sample_rate", type=int, default=24000, help="Target sample rate")
    parser.add_argument("--max_duration", type=float, default=30.0, help="Maximum audio duration")
    parser.add_argument("--min_duration", type=float, default=0.5, help="Minimum audio duration")
    
    args = parser.parse_args()
    
    print("🚀 UNIFIED Zero-Shot Voice Cloning Processor")
    print("=" * 50)
    print("✅ Uses SAME approach as working inference script")
    print("✅ NO audio tokenization during preprocessing")
    print("✅ Audio tokenization happens during training/inference")
    print("✅ Compatible with Arabic+English datasets")
    print("=" * 50)
    
    # Create processor
    processor = UnifiedZeroShotProcessor(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        target_sample_rate=args.target_sample_rate,
        max_duration=args.max_duration,
        min_duration=args.min_duration
    )
    
    # Process dataset
    chatml_samples = processor.process_dataset(max_samples=args.max_samples)
    
    # Save results
    processor.save_chatml_samples(chatml_samples)
    
    print(f"\n🎉 UNIFIED Processing completed!")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📊 Processed samples: {len(chatml_samples)}")
    print(f"✅ Data is now compatible with inference pipeline")
    print(f"✅ Audio tokenization will happen during training")


if __name__ == "__main__":
    main()
