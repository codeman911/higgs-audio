#!/usr/bin/env python3
"""
Zero-Shot Voice Cloning Data Processor for Higgs-Audio V2
Converts your specific dataset format to ChatML for training.

Dataset Format Expected:
{
  "dataset_info": {...},
  "samples": [
    {
      "id": "sample_00000000",
      "audio_file": "target_audio_*.wav",           # What model should generate
      "transcript_file": "target_text_*.txt",       # Text to speak
      "ref_audio_file": "ref_audio_*.wav",          # Reference voice
      "ref_transcript": "reference text...",        # Reference text
      "duration": 2.13
    }
  ]
}
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


class ZeroShotDataProcessor:
    """Simple processor for zero-shot voice cloning dataset"""
    
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
    
    def load_metadata(self) -> Dict:
        """Load the metadata.json file"""
        metadata_file = self.dataset_path / "metadata.json"
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"metadata.json not found in {self.dataset_path}")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        self.logger.info(f"Loaded metadata with {len(metadata['samples'])} samples")
        self.logger.info(f"Total duration: {metadata['dataset_info']['total_duration']:.2f} seconds")
        
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
    
    def process_audio_file(self, audio_file: str) -> Optional[Tuple[np.ndarray, int]]:
        """Process audio file: load, resample, validate"""
        audio_path = self.dataset_path / audio_file
        
        if not audio_path.exists():
            self.logger.warning(f"Audio file not found: {audio_file}")
            return None
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.target_sample_rate, mono=True)
            
            # Validate duration
            duration = len(audio) / sr
            if duration < self.min_duration or duration > self.max_duration:
                self.logger.warning(f"Audio duration {duration:.2f}s outside range [{self.min_duration}, {self.max_duration}]")
                return None
            
            return audio, sr
            
        except Exception as e:
            self.logger.warning(f"Error processing audio {audio_file}: {e}")
            return None
    
    def create_chatml_sample(self, sample: Dict) -> Optional[ChatMLSample]:
        """
        Convert a single sample to ChatML format for zero-shot voice cloning
        
        CORRECTED ChatML Structure for DualFFN:
        - System: Instructions for voice cloning
        - User: Reference transcript + Reference audio + Request for target text
        - Assistant: Target text + Target audio (enables DualFFN learning)
        """
        
        # Read target text from file
        target_text = self.read_transcript_file(sample['transcript_file'])
        if not target_text:
            return None
        
        # Get reference transcript (what's actually spoken in reference audio)
        ref_transcript = sample.get('ref_transcript', '')
        if not ref_transcript:
            self.logger.warning(f"Missing ref_transcript for sample {sample['id']}")
            return None
        
        # Validate audio files exist
        ref_audio_path = self.dataset_path / sample['ref_audio_file']
        target_audio_path = self.dataset_path / sample['audio_file']
        
        if not ref_audio_path.exists() or not target_audio_path.exists():
            self.logger.warning(f"Missing audio files for sample {sample['id']}")
            return None
        
        # Create ChatML messages - CORRECTED STRUCTURE
        messages = [
            # System message: Instructions for voice cloning
            Message(
                role="system",
                content="You are a helpful assistant capable of generating speech in the voice of the provided reference audio."
            ),
            
            # User message: Reference transcript + Reference audio + Request for target text
            Message(
                role="user", 
                content=[
                    TextContent(text=ref_transcript),  # What's spoken in reference audio
                    AudioContent(
                        audio_url=str(ref_audio_path),
                        raw_audio="",  # Will be loaded during training
                        duration=None  # Will be computed during training
                    ),
                    TextContent(text=f"Please generate speech for given text in reference audio's voice: {target_text}")
                ]
            ),
            
            # Assistant message: Target text + Target audio (CRITICAL FOR DUALFFN!)
            Message(
                role="assistant",
                content=[
                    TextContent(text=target_text),  # Text MLP learns to predict this
                    AudioContent(
                        audio_url=str(target_audio_path),
                        raw_audio="",  # Will be loaded during training
                        duration=sample.get('duration')  # Audio MLP learns to predict this
                    )
                ]
            )
        ]
        
        # Create ChatML sample
        chatml_sample = ChatMLSample(
            messages=messages,
            start_index=0,
            speaker=sample.get('id', 'unknown'),  # Use sample ID as speaker identifier
            misc={
                'sample_id': sample['id'],
                'ref_transcript': ref_transcript,
                'target_transcript': target_text,  # Store both for clarity
                'duration': sample.get('duration', 0.0)
            }
        )
        
        return chatml_sample
    
    def process_dataset(self, max_samples: Optional[int] = None) -> List[Dict]:
        """Process the entire dataset and convert to ChatML format"""
        
        # Load metadata
        metadata = self.load_metadata()
        samples = metadata['samples']
        
        if max_samples:
            samples = samples[:max_samples]
            self.logger.info(f"Processing first {max_samples} samples for testing")
        
        # Process samples
        chatml_samples = []
        failed_samples = 0
        
        for sample in tqdm(samples, desc="Processing samples"):
            try:
                chatml_sample = self.create_chatml_sample(sample)
                
                if chatml_sample:
                    # Convert to serializable format
                    chatml_dict = {
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
                    
                    chatml_samples.append(chatml_dict)
                else:
                    failed_samples += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
                failed_samples += 1
        
        self.logger.info(f"Successfully processed {len(chatml_samples)} samples")
        self.logger.info(f"Failed to process {failed_samples} samples")
        
        return chatml_samples
    
    def _serialize_content(self, content):
        """Serialize message content for JSON storage"""
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
                        'raw_audio': item.raw_audio,
                        'duration': item.duration,
                        'offset': getattr(item, 'offset', None)
                    })
            return serialized
        else:
            return str(content)
    
    def save_chatml_samples(self, chatml_samples: List[Dict], filename: str = "chatml_samples.json"):
        """Save ChatML samples to JSON file"""
        output_file = self.output_dir / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chatml_samples, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Saved {len(chatml_samples)} ChatML samples to {output_file}")
        
        # Save summary statistics
        self._save_processing_summary(chatml_samples)
    
    def _save_processing_summary(self, chatml_samples: List[Dict]):
        """Save processing summary and statistics"""
        summary = {
            'total_samples': len(chatml_samples),
            'processing_config': {
                'target_sample_rate': self.target_sample_rate,
                'max_duration': self.max_duration,
                'min_duration': self.min_duration
            },
            'sample_statistics': {
                'avg_duration': np.mean([s['misc']['duration'] for s in chatml_samples if 'duration' in s['misc']]),
                'total_duration': sum([s['misc']['duration'] for s in chatml_samples if 'duration' in s['misc']]),
                'unique_speakers': len(set([s['speaker'] for s in chatml_samples]))
            }
        }
        
        summary_file = self.output_dir / "processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Processing summary saved to {summary_file}")
        self.logger.info(f"Total samples: {summary['total_samples']}")
        self.logger.info(f"Total duration: {summary['sample_statistics']['total_duration']:.2f} seconds")


def main():
    """Main processing function"""
    parser = argparse.ArgumentParser(description="Process zero-shot voice cloning dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed data")
    parser.add_argument("--max_samples", type=int, help="Maximum samples to process (for testing)")
    parser.add_argument("--target_sample_rate", type=int, default=24000, help="Target sample rate")
    parser.add_argument("--max_duration", type=float, default=30.0, help="Maximum audio duration")
    parser.add_argument("--min_duration", type=float, default=0.5, help="Minimum audio duration")
    
    args = parser.parse_args()
    
    # Create processor
    processor = ZeroShotDataProcessor(
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
    
    print(f"\n✅ Processing completed!")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📊 Processed samples: {len(chatml_samples)}")


if __name__ == "__main__":
    main()
