#!/usr/bin/env python3
"""
Unified LoRA Data Processing Pipeline - Higgs-Audio V2
Complete end-to-end pipeline for processing 500-hour dataset
Combines manifest creation and ChatML processing for LoRA training
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

# Add project root to Python path for robust imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import our custom processors
from scripts.data_processing.unified_data_loader import UnifiedDataLoader
from scripts.data_processing.manifest_chatml_processor import ManifestChatMLProcessor


class UnifiedLoRADataPipeline:
    """
    Complete pipeline for processing multi-directory dataset into LoRA-ready ChatML format
    Handles 500-hour dataset with robust processing and validation
    """
    
    def __init__(self, 
                 data_directories: List[str],
                 output_base_dir: str,
                 max_samples: Optional[int] = None,
                 train_split: float = 0.9,
                 target_sample_rate: int = 24000,
                 max_duration: float = 30.0,
                 min_duration: float = 0.5,
                 random_seed: int = 42):
        """
        Initialize unified LoRA data processing pipeline
        
        Args:
            data_directories: List of paths to data directories
            output_base_dir: Base output directory for all processed files
            max_samples: Maximum number of samples to process (None for all)
            train_split: Fraction of data for training
            target_sample_rate: Target sample rate for audio processing
            max_duration: Maximum audio duration in seconds
            min_duration: Minimum audio duration in seconds
            random_seed: Random seed for reproducible processing
        """
        self.data_directories = data_directories
        self.output_base_dir = output_base_dir
        self.max_samples = max_samples
        self.train_split = train_split
        self.target_sample_rate = target_sample_rate
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.random_seed = random_seed
        
        # Create output directory structure
        self.manifest_dir = os.path.join(output_base_dir, "manifest")
        self.chatml_dir = os.path.join(output_base_dir, "chatml")
        self.logs_dir = os.path.join(output_base_dir, "logs")
        
        # File paths
        self.manifest_path = os.path.join(self.manifest_dir, "unified_manifest.json")
        
        # Create directories
        for dir_path in [self.manifest_dir, self.chatml_dir, self.logs_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Setup logging
        log_file = os.path.join(self.logs_dir, "lora_data_pipeline.log")
        logger.add(log_file, rotation="10 MB", retention="7 days")
        
        logger.info("🚀 Initialized Unified LoRA Data Processing Pipeline")
        logger.info(f"📁 Data directories: {len(data_directories)}")
        logger.info(f"📁 Output base: {output_base_dir}")
        logger.info(f"🎯 Max samples: {max_samples or 'All'}")
        logger.info(f"📊 Train split: {train_split}")

    def validate_input_directories(self) -> bool:
        """Validate that all input directories exist and contain data"""
        logger.info("🔍 Validating input directories...")
        
        valid_dirs = []
        for directory in self.data_directories:
            if not os.path.exists(directory):
                logger.error(f"❌ Directory not found: {directory}")
                continue
            
            # Check for audio files
            audio_files = []
            for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
                audio_files.extend(Path(directory).rglob(ext))
            
            if not audio_files:
                logger.warning(f"⚠️  No audio files found in: {directory}")
                continue
            
            logger.info(f"✅ Valid directory: {directory} ({len(audio_files)} audio files)")
            valid_dirs.append(directory)
        
        if not valid_dirs:
            logger.error("❌ No valid directories found!")
            return False
        
        self.data_directories = valid_dirs
        logger.info(f"✅ Validated {len(valid_dirs)} directories")
        return True

    def create_unified_manifest(self) -> bool:
        """Step 1: Create unified manifest from all directories"""
        logger.info("\n" + "="*60)
        logger.info("STEP 1: CREATING UNIFIED MANIFEST")
        logger.info("="*60)
        
        try:
            # Initialize data loader
            loader = UnifiedDataLoader(
                data_directories=self.data_directories,
                manifest_output_path=self.manifest_path,
                target_sample_rate=self.target_sample_rate,
                max_duration=self.max_duration,
                min_duration=self.min_duration
            )
            
            # Process all directories
            manifest_path = loader.process_all_directories()
            
            if not manifest_path or not os.path.exists(manifest_path):
                logger.error("❌ Failed to create unified manifest")
                return False
            
            logger.info(f"✅ Unified manifest created: {manifest_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating manifest: {str(e)}")
            return False

    def process_chatml_samples(self) -> bool:
        """Step 2: Process manifest into ChatML format"""
        logger.info("\n" + "="*60)
        logger.info("STEP 2: PROCESSING CHATML SAMPLES")
        logger.info("="*60)
        
        try:
            # Check manifest exists
            if not os.path.exists(self.manifest_path):
                logger.error(f"❌ Manifest not found: {self.manifest_path}")
                return False
            
            # Initialize ChatML processor
            processor = ManifestChatMLProcessor(
                manifest_path=self.manifest_path,
                output_dir=self.chatml_dir,
                max_samples=self.max_samples,
                train_split=self.train_split,
                random_seed=self.random_seed
            )
            
            # CRITICAL: Validate that the processor generates 8-codebook audio tokens
            logger.info("🔍 Validating audio tokenizer configuration...")
            
            # Load the same tokenizer to verify configuration
            from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
            audio_tokenizer = load_higgs_audio_tokenizer("bosonai/higgs-audio-v2-tokenizer")
            
            logger.info(f"Audio tokenizer configuration:")
            logger.info(f"  - num_codebooks (n_q): {audio_tokenizer.n_q}")
            logger.info(f"  - codebook_size: {getattr(audio_tokenizer, 'codebook_size', 'N/A')}")
            logger.info(f"  - sample_rate: {getattr(audio_tokenizer, 'sample_rate', 'N/A')}")
            
            if audio_tokenizer.n_q != 8:
                logger.error(f"❌ Audio tokenizer has {audio_tokenizer.n_q} codebooks, expected 8")
                logger.error("This will cause tensor size mismatch during training!")
                return False
            
            logger.info(f"✅ Audio tokenizer validated: {audio_tokenizer.n_q} codebooks")
            
            success = processor.process_manifest()
            
            if success:
                # Validate that processed data has correct codebook dimensions
                logger.info("🔍 Validating processed ChatML data...")
                
                # Check a sample from the processed data
                train_file = os.path.join(self.chatml_dir, "train_chatml_samples.json")
                if os.path.exists(train_file):
                    with open(train_file, 'r', encoding='utf-8') as f:
                        samples = json.load(f)
                    
                    if samples:
                        sample = samples[0]
                        codebook_count = None
                        if 'metadata' in sample and 'codebook_count' in sample['metadata']:
                            codebook_count = sample['metadata']['codebook_count']
                        elif 'misc' in sample and 'codebook_count' in sample['misc']:
                            # Backward-compat fallback
                            codebook_count = sample['misc']['codebook_count']
                        if codebook_count is not None:
                            logger.info(f"Processed data codebook count: {codebook_count}")
                            
                            if codebook_count != 8:
                                logger.error(f"❌ Processed data has {codebook_count} codebooks, expected 8")
                                logger.error("Data processing failed to generate correct codebook dimensions!")
                                return False
                            
                            logger.info(f"✅ Processed data validated: {codebook_count} codebooks")
                        else:
                            logger.warning("⚠️  Could not validate codebook count from processed data (missing metadata.misc.codebook_count)")
                
                logger.info("✅ ChatML processing completed successfully")
                return True
            else:
                logger.error("❌ ChatML processing failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error processing ChatML samples: {str(e)}")
            return False

    def validate_output_files(self) -> bool:
        """Step 3: Validate all output files are created correctly"""
        logger.info("\n" + "="*60)
        logger.info("STEP 3: VALIDATING OUTPUT FILES")
        logger.info("="*60)
        
        required_files = [
            self.manifest_path,
            os.path.join(self.chatml_dir, "train_chatml_samples.json"),
            os.path.join(self.chatml_dir, "val_chatml_samples.json"),
            os.path.join(self.chatml_dir, "processing_stats.json")
        ]
        
        all_valid = True
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                logger.error(f"❌ Missing file: {file_path}")
                all_valid = False
                continue
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size < 100:  # Less than 100 bytes
                logger.error(f"❌ File too small: {file_path} ({file_size} bytes)")
                all_valid = False
                continue
            
            logger.info(f"✅ Valid file: {Path(file_path).name} ({file_size:,} bytes)")
        
        return all_valid

    def generate_summary_report(self) -> None:
        """Generate comprehensive summary report"""
        logger.info("\n" + "="*60)
        logger.info("PIPELINE SUMMARY REPORT")
        logger.info("="*60)
        
        try:
            # Load manifest metadata
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            
            manifest_meta = manifest.get('metadata', {})
            
            # Load processing stats
            stats_path = os.path.join(self.chatml_dir, "processing_stats.json")
            with open(stats_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            
            processing_stats = stats.get('processing_statistics', {})
            output_files = stats.get('output_files', {})
            
            # Print comprehensive report
            logger.info(f"📊 DATASET STATISTICS:")
            logger.info(f"   • Total Duration: {manifest_meta.get('total_duration_hours', 0):.2f} hours")
            logger.info(f"   • Total Samples: {manifest_meta.get('total_samples', 0):,}")
            logger.info(f"   • Directories Processed: {manifest_meta.get('directories_processed', 0)}")
            
            logger.info(f"\n🚂 TRAINING SPLIT:")
            logger.info(f"   • Training Samples: {output_files.get('train_count', 0):,}")
            logger.info(f"   • Validation Samples: {output_files.get('val_count', 0):,}")
            
            logger.info(f"\n📁 OUTPUT FILES:")
            logger.info(f"   • Manifest: {self.manifest_path}")
            logger.info(f"   • Training Data: {output_files.get('train_samples', 'N/A')}")
            logger.info(f"   • Validation Data: {output_files.get('val_samples', 'N/A')}")
            logger.info(f"   • Statistics: {stats_path}")
            
            # Save summary report
            summary_report = {
                'pipeline_config': {
                    'data_directories': self.data_directories,
                    'output_base_dir': self.output_base_dir,
                    'max_samples': self.max_samples,
                    'train_split': self.train_split,
                    'target_sample_rate': self.target_sample_rate,
                    'max_duration': self.max_duration,
                    'min_duration': self.min_duration
                },
                'manifest_metadata': manifest_meta,
                'processing_statistics': processing_stats,
                'output_files': output_files,
                'ready_for_training': True
            }
            
            summary_path = os.path.join(self.output_base_dir, "pipeline_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📋 Summary report saved: {summary_path}")
            
        except Exception as e:
            logger.error(f"❌ Error generating summary: {str(e)}")

    def run_pipeline(self) -> bool:
        """Run the complete LoRA data processing pipeline"""
        logger.info("🚀 Starting Unified LoRA Data Processing Pipeline")
        logger.info(f"Processing {len(self.data_directories)} directories...")
        
        # Step 1: Validate input directories
        if not self.validate_input_directories():
            logger.error("❌ Input validation failed")
            return False
        
        # Step 2: Create unified manifest
        if not self.create_unified_manifest():
            logger.error("❌ Manifest creation failed")
            return False
        
        # Step 3: Process ChatML samples
        if not self.process_chatml_samples():
            logger.error("❌ ChatML processing failed")
            return False
        
        # Step 4: Validate output files
        if not self.validate_output_files():
            logger.error("❌ Output validation failed")
            return False
        
        # Step 5: Generate summary report
        self.generate_summary_report()
        
        logger.info("\n" + "🎉" * 20)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("🎉" * 20)
        logger.info(f"📁 All files ready in: {self.output_base_dir}")
        logger.info("🚀 Ready for LoRA training on 8x H200 GPUs!")
        
        return True


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description="Unified LoRA Data Processing Pipeline")
    parser.add_argument("--data_dirs", nargs='+', required=True,
                       help="List of data directories to process")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Base output directory for all processed files")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process (None for all)")
    parser.add_argument("--train_split", type=float, default=0.9,
                       help="Fraction of data for training")
    parser.add_argument("--sample_rate", type=int, default=24000,
                       help="Target sample rate")
    parser.add_argument("--max_duration", type=float, default=30.0,
                       help="Maximum audio duration in seconds")
    parser.add_argument("--min_duration", type=float, default=0.5,
                       help="Minimum audio duration in seconds")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducible processing")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = UnifiedLoRADataPipeline(
        data_directories=args.data_dirs,
        output_base_dir=args.output_dir,
        max_samples=args.max_samples,
        train_split=args.train_split,
        target_sample_rate=args.sample_rate,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
        random_seed=args.random_seed
    )
    
    # Run pipeline
    success = pipeline.run_pipeline()
    
    if success:
        logger.info("✅ Pipeline completed successfully!")
        logger.info("🚀 Ready for LoRA training!")
        exit(0)
    else:
        logger.error("❌ Pipeline failed!")
        exit(1)


if __name__ == "__main__":
    main()
