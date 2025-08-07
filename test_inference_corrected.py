#!/usr/bin/env python3
"""
Corrected Zero-Shot Voice Cloning Inference Test Script for Higgs-Audio V2
Includes both reference audio AND reference transcript for proper voice conditioning.
"""

import json
import os
import random
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from loguru import logger
from typing import List, Dict, Any, Optional, Tuple

# Import from examples/generation.py - exact working imports
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent, TextContent
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern
from boson_multimodal.serve.higgs_audio_model_client import HiggsAudioModelClient


class CorrectedInferenceTest:
    """Corrected inference test with both reference audio and transcript."""
    
    def __init__(self, model_path: str = "bosonai/higgs-audio-v2-generation-3B-base",
                 tokenizer_path: str = "bosonai/higgs-audio-v2-tokenizer"):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load audio tokenizer
        logger.info("Loading audio tokenizer...")
        self.audio_tokenizer = load_higgs_audio_tokenizer(tokenizer_path, device=self.device)
        
        # Initialize model client
        logger.info("Loading Higgs-Audio model...")
        self.model_client = HiggsAudioModelClient(
            model_path=model_path,
            audio_tokenizer=self.audio_tokenizer,
            device=self.device,
            device_id=0 if self.device.startswith("cuda") else None,
            max_new_tokens=2048,
            use_static_kv_cache=True if self.device.startswith("cuda") else False,
        )
        logger.info("Model loaded successfully!")

    def load_chatml_samples(self, json_file: str, max_samples: int = 10) -> List[Dict]:
        """Load ChatML samples from JSON file."""
        logger.info(f"Loading ChatML samples from: {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = data.get('samples', [])
        if not samples:
            raise ValueError(f"No samples found in {json_file}")
        
        # Randomly select samples
        selected_samples = random.sample(samples, min(max_samples, len(samples)))
        logger.info(f"Selected {len(selected_samples)} samples for testing")
        
        return selected_samples

    def prepare_corrected_chatml_messages(self, ref_audio_path: str, ref_transcript: str, 
                                        target_text: str) -> Tuple[List[Message], List[torch.Tensor]]:
        """
        Prepare corrected ChatML messages with BOTH reference audio and transcript.
        This follows the proper zero-shot voice cloning format.
        """
        messages = []
        audio_ids = []
        
        # System message
        system_message = Message(
            role="system",
            content="You are a helpful assistant that can generate speech with the voice characteristics of the provided reference audio."
        )
        messages.append(system_message)
        
        # User message with reference audio AND its transcript
        try:
            # Load and tokenize reference audio
            logger.info(f"Loading reference audio: {ref_audio_path}")
            
            # Check if file exists and is readable
            if not os.path.exists(ref_audio_path):
                logger.error(f"Reference audio file not found: {ref_audio_path}")
                return messages, audio_ids
                
            # Get file size for validation
            file_size = os.path.getsize(ref_audio_path)
            logger.info(f"Reference audio file size: {file_size} bytes")
            
            if file_size < 1000:  # Less than 1KB
                logger.warning(f"Reference audio file very small ({file_size} bytes), might be corrupted")
                return messages, audio_ids
            
            # Tokenize reference audio (pass file path, not tensor)
            ref_audio_tokens = self.audio_tokenizer.encode(ref_audio_path)
            logger.info(f"✅ Reference audio tokenized successfully: {ref_audio_tokens.shape}")
            
            # Create user message with BOTH reference audio and its transcript
            user_content = [
                AudioContent(audio=ref_audio_tokens),
                TextContent(text=f"Reference transcript: \"{ref_transcript}\"\n\nNow please generate speech for the following text using the same voice: \"{target_text}\"")
            ]
            
            user_message = Message(role="user", content=user_content)
            messages.append(user_message)
            audio_ids.append(ref_audio_tokens)
            
            logger.info("✅ ChatML messages prepared with reference audio AND transcript")
            
        except Exception as e:
            logger.error(f"❌ Error processing reference audio {ref_audio_path}: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            
            # Fallback: text-only generation without reference
            user_message = Message(
                role="user", 
                content=[TextContent(text=f"Please generate speech for: \"{target_text}\"")]
            )
            messages.append(user_message)
            logger.info("⚠️ Falling back to text-only generation (no voice cloning)")
        
        return messages, audio_ids

    def test_corrected_inference(self, chatml_samples: List[Dict], output_dir: str = "corrected_samples"):
        """Test corrected zero-shot voice cloning inference."""
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        success_count = 0
        
        for i, sample in enumerate(chatml_samples):
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing Sample {i+1}/{len(chatml_samples)}")
            logger.info(f"{'='*60}")
            
            try:
                # Extract sample data
                sample_id = f"sample_{i:08d}"
                target_text = sample.get('target_text', '')
                ref_audio_path = sample.get('reference_audio', '')
                ref_transcript = sample.get('ref_transcript', '')  # This is the key addition!
                
                logger.info(f"Sample ID: {sample_id}")
                logger.info(f"Target text: {target_text[:100]}...")
                logger.info(f"Reference audio: {ref_audio_path}")
                logger.info(f"Reference transcript: {ref_transcript[:100]}...")  # Log reference transcript
                
                # Prepare corrected ChatML messages with BOTH audio and transcript
                messages, audio_ids = self.prepare_corrected_chatml_messages(
                    ref_audio_path, ref_transcript, target_text
                )
                
                if not audio_ids:
                    logger.warning("⚠️ No reference audio available, performing text-only generation")
                    generation_type = "text_to_speech"
                else:
                    logger.info("🎯 Performing zero-shot voice cloning with reference audio + transcript")
                    generation_type = "voice_cloning"
                
                # Prepare chunked text (simple approach)
                chunked_text = [target_text]
                
                # Generate audio
                logger.info("🚀 Starting generation...")
                waveform, sample_rate, generated_text = self.model_client.generate(
                    messages=messages,
                    audio_ids=audio_ids,
                    chunked_text=chunked_text,
                    generation_chunk_buffer_size=None,
                    temperature=1.0,
                    top_k=50,
                    top_p=0.95,
                    ras_win_len=7,
                    ras_win_max_num_repeat=2,
                    seed=42
                )
                
                # Save generated audio
                output_audio_path = os.path.join(output_dir, f"{sample_id}_generated.wav")
                sf.write(output_audio_path, waveform, sample_rate)
                
                # Copy reference audio for comparison
                if os.path.exists(ref_audio_path):
                    ref_output_path = os.path.join(output_dir, f"{sample_id}_reference.wav")
                    ref_waveform, ref_sr = torchaudio.load(ref_audio_path)
                    sf.write(ref_output_path, ref_waveform.numpy().T, ref_sr)
                
                # Save metadata
                metadata = {
                    "sample_id": sample_id,
                    "target_text": target_text,
                    "reference_audio": ref_audio_path,
                    "reference_transcript": ref_transcript,  # Include reference transcript in metadata
                    "generated_audio": output_audio_path,
                    "sampling_rate": sample_rate,
                    "generated_text": generated_text,
                    "generation_type": generation_type,
                    "success": True
                }
                
                metadata_path = os.path.join(output_dir, f"{sample_id}_metadata.json")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                results.append(metadata)
                success_count += 1
                
                logger.info(f"✅ Sample {sample_id} completed successfully!")
                logger.info(f"   • Generation type: {generation_type}")
                logger.info(f"   • Audio saved: {output_audio_path}")
                logger.info(f"   • Duration: {len(waveform) / sample_rate:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ Error processing sample {i}: {str(e)}")
                error_metadata = {
                    "sample_id": f"sample_{i:08d}",
                    "error": str(e),
                    "success": False
                }
                results.append(error_metadata)
        
        # Save summary
        summary = {
            "total_samples": len(chatml_samples),
            "successful_samples": success_count,
            "success_rate": success_count / len(chatml_samples) if chatml_samples else 0,
            "results": results
        }
        
        summary_path = os.path.join(output_dir, "inference_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"CORRECTED INFERENCE TEST COMPLETED!")
        logger.info(f"{'='*60}")
        logger.info(f"Total samples: {len(chatml_samples)}")
        logger.info(f"Successful: {success_count}")
        logger.info(f"Success rate: {success_count / len(chatml_samples) * 100:.1f}%")
        logger.info(f"Results saved to: {output_dir}")
        
        return results


def main():
    """Main function to run corrected inference test."""
    logger.info("Starting Corrected Zero-Shot Voice Cloning Inference Test")
    
    # Initialize inference test
    inference_test = CorrectedInferenceTest()
    
    # Load ChatML samples
    chatml_file = "test_processed/test_chatml_samples.json"
    
    if not os.path.exists(chatml_file):
        logger.error(f"ChatML file not found: {chatml_file}")
        logger.info("Please run the data processing script first to generate ChatML samples.")
        return
    
    # Load and test samples
    samples = inference_test.load_chatml_samples(chatml_file, max_samples=5)
    results = inference_test.test_corrected_inference(samples, output_dir="corrected_samples")
    
    logger.info("✅ Corrected inference test completed!")
    logger.info("Check the 'corrected_samples' directory for results.")
    logger.info("Compare the audio quality with the previous 'tested_samples' results.")


if __name__ == "__main__":
    main()
