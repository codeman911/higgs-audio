#!/usr/bin/env python3
"""
Manifest-based ChatML Processor for LoRA Training - Higgs-Audio V2
Processes unified manifest file to create ChatML samples for distributed training
Uses the proven ChatML format from successful inference pipeline
"""

import os
import json
import random
import argparse
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from boson_multimodal.audio_processing.higgs_audio_tokenizer import HiggsAudioTokenizer

class ManifestChatMLProcessor:
    """
    Process unified manifest file to create ChatML samples for LoRA training
    Uses the exact ChatML format validated in our inference pipeline
    """
    
    def __init__(self, 
                 manifest_path: str,
                 output_dir: str,
                 max_samples: Optional[int] = None,
                 train_split: float = 0.9,
                 random_seed: int = 42):
        """
        Initialize manifest-based ChatML processor
        
        Args:
            manifest_path: Path to unified manifest file
            output_dir: Output directory for ChatML files
            max_samples: Maximum number of samples to process (None for all)
            train_split: Fraction of data for training (rest for validation)
            random_seed: Random seed for reproducible splits
        """
        self.manifest_path = manifest_path
        self.output_dir = output_dir
        self.max_samples = max_samples
        self.train_split = train_split
        self.random_seed = random_seed
        
        # Load manifest
        self.manifest_data = self.load_manifest()
        
        # Statistics
        self.stats = {
            'total_samples': 0,
            'processed_samples': 0,
            'train_samples': 0,
            'val_samples': 0,
            'errors': []
        }
        
        # Initialize audio tokenizer
        self.audio_tokenizer = HiggsAudioTokenizer()
        
        logger.info(f"Initialized ManifestChatMLProcessor")
        logger.info(f"Manifest: {manifest_path}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Train split: {train_split}")

    def load_manifest(self) -> Dict[str, Any]:
        """Load and validate unified manifest file"""
        logger.info(f"Loading manifest from: {self.manifest_path}")
        
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
        
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        # Validate manifest structure
        if 'samples' not in manifest:
            raise ValueError("Invalid manifest format: missing 'samples' key")
        
        samples = manifest['samples']
        logger.info(f"Loaded manifest with {len(samples)} samples")
        
        if 'metadata' in manifest:
            metadata = manifest['metadata']
            logger.info(f"Manifest metadata: {metadata.get('total_duration_hours', 0):.2f} hours")
            logger.info(f"Total samples: {metadata.get('total_samples', 0):,}")
        
        return manifest

    def create_chatml_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create ChatML sample from manifest entry with audio tokenization
        Uses the exact format validated in our inference/training pipeline
        """
        try:
            # Extract sample information
            sample_id = sample['sample_id']
            ref_audio_path = sample['ref_audio_path']
            target_audio_path = sample['target_audio_path']
            ref_transcript = sample['ref_transcript']
            target_transcript = sample['target_transcript']
            
            # Validate file paths exist
            if not os.path.exists(ref_audio_path):
                raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")
            if not os.path.exists(target_audio_path):
                raise FileNotFoundError(f"Target audio not found: {target_audio_path}")
            
            # Generate 8-codebook audio tokens
            logger.debug(f"Tokenizing reference audio: {ref_audio_path}")
            try:
                ref_audio_tokens = self.audio_tokenizer.encode(ref_audio_path)
                logger.debug(f"Reference audio tokens shape: {ref_audio_tokens.shape}")
                if ref_audio_tokens.shape[0] != 8:
                    raise ValueError(f"Reference audio tokens have {ref_audio_tokens.shape[0]} codebooks, expected 8")
            except Exception as e:
                raise Exception(f"Error tokenizing reference audio {ref_audio_path}: {e}")
            
            logger.debug(f"Tokenizing target audio: {target_audio_path}")
            try:
                target_audio_tokens = self.audio_tokenizer.encode(target_audio_path)
                logger.debug(f"Target audio tokens shape: {target_audio_tokens.shape}")
                if target_audio_tokens.shape[0] != 8:
                    raise ValueError(f"Target audio tokens have {target_audio_tokens.shape[0]} codebooks, expected 8")
            except Exception as e:
                raise Exception(f"Error tokenizing target audio {target_audio_path}: {e}")
            
            # Create ChatML structure with audio tokens and urls
            chatml_sample = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant capable of generating speech in the voice of the provided reference audio."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": target_transcript},
                            {"type": "audio", "audio_url": ref_audio_path, "audio_tokens": ref_audio_tokens.tolist()}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "audio", "audio_url": target_audio_path, "audio_tokens": target_audio_tokens.tolist()}
                        ]
                    }
                ],
                "metadata": {
                    "sample_id": sample_id,
                    "ref_transcript": ref_transcript,
                    "target_transcript": target_transcript,
                    "duration": sample.get('duration', 0.0),
                    "sample_rate": sample.get('sample_rate', 24000),
                    "source_directory": sample.get('source_directory', ''),
                    "ref_audio_tokens_shape": list(ref_audio_tokens.shape),
                    "target_audio_tokens_shape": list(target_audio_tokens.shape),
                    "codebook_count": 8
                }
            }
            
            return chatml_sample
            
        except Exception as e:
            error_msg = f"Error creating ChatML for sample {sample.get('sample_id', 'unknown')}: {str(e)}"
            logger.warning(error_msg)
            self.stats['errors'].append(error_msg)
            return None

    def process_samples(self) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process all samples from manifest and create train/val splits
        
        Returns:
            (train_samples, val_samples)
        """
        logger.info("Processing samples from manifest")
        
        samples = self.manifest_data['samples']
        
        # Limit samples if specified
        if self.max_samples and len(samples) > self.max_samples:
            logger.info(f"Limiting to {self.max_samples} samples (from {len(samples)})")
            # Use random sampling for better representation
            random.seed(self.random_seed)
            samples = random.sample(samples, self.max_samples)
        
        self.stats['total_samples'] = len(samples)
        
        # Process each sample
        chatml_samples = []
        
        for sample in tqdm(samples, desc="Creating ChatML samples"):
            chatml_sample = self.create_chatml_sample(sample)
            
            if chatml_sample is not None:
                chatml_samples.append(chatml_sample)
                self.stats['processed_samples'] += 1
        
        logger.info(f"Successfully processed {len(chatml_samples)} ChatML samples")
        
        # Create train/val split
        random.seed(self.random_seed)
        random.shuffle(chatml_samples)
        
        split_idx = int(len(chatml_samples) * self.train_split)
        train_samples = chatml_samples[:split_idx]
        val_samples = chatml_samples[split_idx:]
        
        self.stats['train_samples'] = len(train_samples)
        self.stats['val_samples'] = len(val_samples)
        
        logger.info(f"Split: {len(train_samples)} train, {len(val_samples)} validation")
        
        return train_samples, val_samples

    def save_chatml_files(self, train_samples: List[Dict[str, Any]], val_samples: List[Dict[str, Any]]) -> None:
        """Save ChatML samples to files"""
        logger.info(f"Saving ChatML files to: {self.output_dir}")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save training samples
        train_path = os.path.join(self.output_dir, "train_chatml_samples.json")
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_samples, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Training samples saved: {train_path}")
        
        # Save validation samples
        val_path = os.path.join(self.output_dir, "val_chatml_samples.json")
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_samples, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Validation samples saved: {val_path}")
        
        # Save statistics
        stats_path = os.path.join(self.output_dir, "processing_stats.json")
        stats_data = {
            'processing_statistics': self.stats,
            'manifest_metadata': self.manifest_data.get('metadata', {}),
            'output_files': {
                'train_samples': train_path,
                'val_samples': val_path,
                'train_count': len(train_samples),
                'val_count': len(val_samples)
            }
        }
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Statistics saved: {stats_path}")

    def print_statistics(self) -> None:
        """Print comprehensive processing statistics"""
        logger.info("\n" + "="*60)
        logger.info("CHATML PROCESSING STATISTICS")
        logger.info("="*60)
        
        logger.info(f"📊 Total Samples: {self.stats['total_samples']:,}")
        logger.info(f"✅ Processed Samples: {self.stats['processed_samples']:,}")
        logger.info(f"🚂 Training Samples: {self.stats['train_samples']:,}")
        logger.info(f"🔍 Validation Samples: {self.stats['val_samples']:,}")
        
        if self.stats['errors']:
            logger.warning(f"\n⚠️  Errors encountered: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:5]:  # Show first 5 errors
                logger.warning(f"   • {error}")
            if len(self.stats['errors']) > 5:
                logger.warning(f"   • ... and {len(self.stats['errors']) - 5} more")

    def process_manifest(self) -> str:
        """
        Main method to process manifest and create ChatML files
        
        Returns:
            Output directory path
        """
        logger.info("🚀 Starting manifest-based ChatML processing")
        
        # Process samples
        train_samples, val_samples = self.process_samples()
        
        if not train_samples:
            logger.error("❌ No valid training samples created!")
            return None
        
        # Save ChatML files
        self.save_chatml_files(train_samples, val_samples)
        
        # Print statistics
        self.print_statistics()
        
        logger.info("🎉 ChatML processing completed successfully!")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info("🚀 Ready for LoRA training!")
        
        return self.output_dir


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description="Manifest-based ChatML Processor for LoRA Training")
    parser.add_argument("--manifest_path", type=str, required=True,
                       help="Path to unified manifest file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for ChatML files")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process")
    parser.add_argument("--train_split", type=float, default=0.9,
                       help="Fraction of data for training")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducible splits")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = ManifestChatMLProcessor(
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        train_split=args.train_split,
        random_seed=args.random_seed
    )
    
    # Process manifest
    output_dir = processor.process_manifest()
    
    if output_dir:
        logger.info(f"✅ ChatML files created in: {output_dir}")
        logger.info("🚀 Ready for LoRA training!")
    else:
        logger.error("❌ Failed to process manifest")


if __name__ == "__main__":
    main()
