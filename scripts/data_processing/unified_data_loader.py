#!/usr/bin/env python3
"""
Unified Data Loader for LoRA Training - Higgs-Audio V2
Combines multiple data directories into a single manifest for 500-hour Arabic+English dataset
Simple approach: just merge existing metadata.json files from each directory
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
from tqdm import tqdm


class UnifiedDataLoader:
    """
    Simple unified data loader that combines multiple directories into a single manifest
    Works with existing directory structure and metadata.json files
    """
    
    def __init__(self, 
                 data_directories: List[str],
                 manifest_output_path: str,
                 target_sample_rate: int = 24000,
                 max_duration: float = 30.0,
                 min_duration: float = 0.5):
        """
        Initialize unified data loader
        
        Args:
            data_directories: List of paths to data directories (each with metadata.json)
            manifest_output_path: Path to save unified manifest file
            target_sample_rate: Target sample rate for audio processing
            max_duration: Maximum audio duration in seconds
            min_duration: Minimum audio duration in seconds
        """
        self.data_directories = data_directories
        self.manifest_output_path = manifest_output_path
        self.target_sample_rate = target_sample_rate
        self.max_duration = max_duration
        self.min_duration = min_duration
        
        # Statistics tracking
        self.stats = {
            'total_samples': 0,
            'valid_samples': 0,
            'total_duration': 0.0,
            'directories_processed': 0,
            'errors': []
        }
        
        logger.info(f"Initialized UnifiedDataLoader for {len(data_directories)} directories")
        logger.info(f"Target sample rate: {target_sample_rate} Hz")
        logger.info(f"Duration range: {min_duration}-{max_duration} seconds")

    def load_directory_metadata(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Load metadata.json from a single directory and convert to unified format
        """
        logger.info(f"Loading metadata from: {directory_path}")
        
        if not os.path.exists(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return []
        
        # Look for metadata file
        metadata_file = Path(directory_path) / "metadata.json"
        
        if not metadata_file.exists():
            logger.error(f"metadata.json not found in: {directory_path}")
            return []
        
        # Load metadata
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata with {len(metadata)} entries")
        except Exception as e:
            logger.error(f"Error loading metadata from {metadata_file}: {str(e)}")
            return []
        
        # Convert to unified format
        unified_samples = []
        directory_name = Path(directory_path).name
        
        for sample_id, sample_data in tqdm(metadata.items(), desc=f"Processing {directory_name}"):
            try:
                # Extract sample information (adjust field names as needed)
                ref_audio_file = sample_data.get('ref_audio_file', sample_data.get('audio_file', ''))
                target_audio_file = sample_data.get('audio_file', sample_data.get('target_audio_file', ''))
                ref_transcript = sample_data.get('ref_transcript', '')
                target_transcript = sample_data.get('transcript', sample_data.get('target_transcript', ''))
                
                # Skip if missing essential data
                if not ref_audio_file or not target_audio_file or not target_transcript:
                    continue
                
                # Build full paths
                ref_audio_path = os.path.join(directory_path, ref_audio_file)
                target_audio_path = os.path.join(directory_path, target_audio_file)
                
                # Validate files exist
                if not os.path.exists(ref_audio_path) or not os.path.exists(target_audio_path):
                    continue
                
                # Create unified sample entry
                unified_sample = {
                    'sample_id': f"{directory_name}_{sample_id}",
                    'ref_audio_path': ref_audio_path,
                    'target_audio_path': target_audio_path,
                    'ref_transcript': ref_transcript,
                    'target_transcript': target_transcript,
                    'duration': sample_data.get('duration', 0.0),
                    'sample_rate': sample_data.get('sample_rate', self.target_sample_rate),
                    'source_directory': directory_path
                }
                
                unified_samples.append(unified_sample)
                
                # Update statistics
                self.stats['total_duration'] += unified_sample['duration']
                
            except Exception as e:
                error_msg = f"Error processing sample {sample_id}: {str(e)}"
                logger.warning(error_msg)
                self.stats['errors'].append(error_msg)
        
        logger.info(f"Successfully processed {len(unified_samples)} samples from {directory_path}")
        return unified_samples

    def create_unified_manifest(self) -> List[Dict[str, Any]]:
        """
        Create unified manifest from all data directories
        """
        logger.info("Creating unified manifest from all directories")
        all_samples = []
        
        for directory in self.data_directories:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing directory: {directory}")
            logger.info(f"{'='*60}")
            
            directory_samples = self.load_directory_metadata(directory)
            all_samples.extend(directory_samples)
            self.stats['directories_processed'] += 1
        
        self.stats['total_samples'] = len(all_samples)
        self.stats['valid_samples'] = len(all_samples)
        
        return all_samples

    def save_manifest(self, manifest_data: List[Dict[str, Any]]) -> None:
        """
        Save unified manifest to file
        """
        logger.info(f"Saving unified manifest to: {self.manifest_output_path}")
        
        # Create output directory
        os.makedirs(os.path.dirname(self.manifest_output_path), exist_ok=True)
        
        # Prepare final manifest structure
        final_manifest = {
            'metadata': {
                'total_samples': len(manifest_data),
                'total_duration_hours': self.stats['total_duration'] / 3600,
                'directories_processed': self.stats['directories_processed'],
                'target_sample_rate': self.target_sample_rate,
                'created_by': 'UnifiedDataLoader',
                'version': '1.0'
            },
            'samples': manifest_data
        }
        
        # Save manifest
        with open(self.manifest_output_path, 'w', encoding='utf-8') as f:
            json.dump(final_manifest, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Manifest saved successfully!")
        
        # Print statistics
        self.print_statistics()

    def print_statistics(self) -> None:
        """
        Print comprehensive statistics about the unified dataset
        """
        logger.info("\n" + "="*60)
        logger.info("UNIFIED DATASET STATISTICS")
        logger.info("="*60)
        
        logger.info(f"📊 Total Samples: {self.stats['total_samples']:,}")
        logger.info(f"⏱️  Total Duration: {self.stats['total_duration']/3600:.2f} hours")
        logger.info(f"📁 Directories Processed: {self.stats['directories_processed']}")
        
        if self.stats['errors']:
            logger.warning(f"\n⚠️  Errors encountered: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:5]:  # Show first 5 errors
                logger.warning(f"   • {error}")
            if len(self.stats['errors']) > 5:
                logger.warning(f"   • ... and {len(self.stats['errors']) - 5} more")

    def process_all_directories(self) -> str:
        """
        Main method to process all directories and create unified manifest
        
        Returns:
            Path to created manifest file
        """
        logger.info("🚀 Starting unified data processing for LoRA training")
        
        # Create unified manifest
        manifest_data = self.create_unified_manifest()
        
        if not manifest_data:
            logger.error("❌ No valid samples found in any directory!")
            return None
        
        # Save manifest
        self.save_manifest(manifest_data)
        
        logger.info("🎉 Unified data processing completed successfully!")
        return self.manifest_output_path


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description="Unified Data Loader for LoRA Training")
    parser.add_argument("--data_dirs", nargs='+', required=True,
                       help="List of data directories to process")
    parser.add_argument("--manifest_output", type=str, required=True,
                       help="Output path for unified manifest file")
    parser.add_argument("--sample_rate", type=int, default=24000,
                       help="Target sample rate")
    parser.add_argument("--max_duration", type=float, default=30.0,
                       help="Maximum audio duration in seconds")
    parser.add_argument("--min_duration", type=float, default=0.5,
                       help="Minimum audio duration in seconds")
    
    args = parser.parse_args()
    
    # Initialize loader
    loader = UnifiedDataLoader(
        data_directories=args.data_dirs,
        manifest_output_path=args.manifest_output,
        target_sample_rate=args.sample_rate,
        max_duration=args.max_duration,
        min_duration=args.min_duration
    )
    
    # Process all directories
    manifest_path = loader.process_all_directories()
    
    if manifest_path:
        logger.info(f"✅ Unified manifest created: {manifest_path}")
        logger.info("🚀 Ready for LoRA training!")
    else:
        logger.error("❌ Failed to create unified manifest")


if __name__ == "__main__":
    main()
