#!/usr/bin/env python3
"""
Final Corrected Zero-Shot Voice Cloning Inference Test Script for Higgs-Audio V2
Uses correct imports and matches actual ChatML format from training data.
"""

import json
import os
import random
import torch
import torchaudio
import soundfile as sf
import argparse
from pathlib import Path
from loguru import logger
from typing import List, Dict, Any, Optional, Tuple

# Import from examples/generation.py - exact working imports
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent, TextContent
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern

# Import the HiggsAudioModelClient class directly from examples/generation.py
import sys
sys.path.append('examples')
from generation import HiggsAudioModelClient


class FinalInferenceTest:
    """Final corrected inference test matching actual ChatML format."""
    
    def __init__(self, model_path: str = "bosonai/higgs-audio-v2-generation-3B-base",
                 tokenizer_path: str = "bosonai/higgs-audio-v2-tokenizer"):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load audio tokenizer
        logger.info("Loading audio tokenizer...")
        self.audio_tokenizer = load_higgs_audio_tokenizer(tokenizer_path, device=self.device)
        
        # Initialize model client using the correct class from examples
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

    def convert_mp3_to_wav(self, audio_path: str) -> str:
        """Convert MP3 to WAV with mono channel and 24kHz sample rate."""
        if not audio_path.lower().endswith('.mp3'):
            return audio_path
        
        wav_path = audio_path.replace('.mp3', '.wav')
        
        if os.path.exists(wav_path):
            return wav_path  # Use existing WAV
        
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                logger.info("Converted stereo to mono")
            
            # Resample to 24kHz if needed
            if sample_rate != 24000:
                resampler = torchaudio.transforms.Resample(sample_rate, 24000)
                waveform = resampler(waveform)
                logger.info(f"Resampled from {sample_rate}Hz to 24kHz")
                sample_rate = 24000
            
            torchaudio.save(wav_path, waveform, sample_rate)
            logger.info(f"Converted: {audio_path} -> {wav_path} (mono, 24kHz)")
            return wav_path
        except:
            return audio_path  # Use original on error

    def chunk_text_for_generation(self, text: str) -> List[str]:
        """Split text into chunks for better pronunciation and pacing."""
        import re
        
        # Split on Arabic and English punctuation
        # Pattern matches: . ! ? ، (Arabic comma) ؛ (Arabic semicolon) ؟ (Arabic question mark)
        sentences = re.split(r'([.!?،؛؟]+)', text)
        
        # Combine sentence with its punctuation and filter empty chunks
        chunked_text = []
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                chunk = (sentences[i] + sentences[i + 1]).strip()
            else:
                chunk = sentences[i].strip()
            
            if chunk:
                chunked_text.append(chunk)
        
        # If no chunks or only one very short chunk, return original text
        if not chunked_text or (len(chunked_text) == 1 and len(chunked_text[0]) < 10):
            return [text]
        
        return chunked_text

    def load_chatml_samples(self, json_file: str, max_samples: int = 10, seed: Optional[int] = None) -> List[Dict]:
        """Load ChatML samples from JSON file with robust structure handling and shuffling."""
        logger.info(f"Loading ChatML samples from: {json_file}")
        
        # Set random seed for reproducibility if provided
        if seed is not None:
            random.seed(seed)
            logger.info(f"Using random seed: {seed}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            # JSON file contains array of samples directly
            samples = data
            logger.info(f"Loaded JSON as direct array with {len(samples)} samples")
        elif isinstance(data, dict):
            # JSON file contains dictionary with 'samples' key
            samples = data.get('samples', [])
            if not samples:
                # Try other common keys
                samples = data.get('data', [])
                if not samples:
                    samples = data.get('items', [])
            logger.info(f"Loaded JSON as dictionary with {len(samples)} samples")
        else:
            raise ValueError(f"Unexpected JSON structure in {json_file}. Expected list or dict, got {type(data)}")
        
        if not samples:
            raise ValueError(f"No samples found in {json_file}")
        
        # Explicitly shuffle the entire dataset first
        samples_copy = samples.copy()  # Don't modify original list
        random.shuffle(samples_copy)
        logger.info(f"Shuffled {len(samples_copy)} samples")
        
        # Select the first max_samples after shuffling
        if len(samples_copy) <= max_samples:
            selected_samples = samples_copy
        else:
            selected_samples = samples_copy[:max_samples]
        
        logger.info(f"Selected {len(selected_samples)} samples for testing after shuffling")
        
        return selected_samples

    def convert_chatml_to_messages(self, chatml_sample: Dict) -> Tuple[List[Message], List[torch.Tensor]]:
        """
        Convert ChatML sample to proper Message format matching training data structure.
        
        Expected ChatML structure:
        {
          "messages": [
            {"role": "system", "content": "voice cloning instruction"},
            {"role": "user", "content": [
              {"type": "text", "text": "target_text"},
              {"type": "audio", "audio_url": "reference_audio.wav"}
            ]},
            {"role": "assistant", "content": [
              {"type": "audio", "audio_url": "target_audio.wav"}
            ]}
          ]
        }
        """
        messages = []
        audio_ids = []
        
        chatml_messages = chatml_sample.get('messages', [])
        if not chatml_messages:
            raise ValueError("No messages found in ChatML sample")
        
        for msg in chatml_messages:
            role = msg.get('role')
            content = msg.get('content')
            
            if role == 'system':
                # System message
                messages.append(Message(role="system", content=content))
                
            elif role == 'user':
                # User message with text and reference audio
                if isinstance(content, list):
                    message_content = []
                    
                    for item in content:
                        if item.get('type') == 'text':
                            # Add text content
                            text = item.get('text', '')
                            message_content.append(TextContent(text=text))
                            logger.info(f"Added text: {text[:50]}...")
                            
                        elif item.get('type') == 'audio':
                            # Add reference audio
                            audio_url = item.get('audio_url', '')
                            if audio_url and os.path.exists(audio_url):
                                try:
                                    logger.info(f"Loading reference audio: {audio_url}")
                                    
                                    # Convert MP3 to WAV if needed
                                    wav_audio_url = self.convert_mp3_to_wav(audio_url)
                                    
                                    # Check file size
                                    file_size = os.path.getsize(wav_audio_url)
                                    logger.info(f"Audio file size: {file_size} bytes")
                                    
                                    if file_size < 1000:
                                        logger.warning(f"Audio file very small ({file_size} bytes), might be corrupted")
                                        continue
                                    
                                    # Tokenize reference audio for audio_ids
                                    ref_audio_tokens = self.audio_tokenizer.encode(wav_audio_url)
                                    logger.info(f"✅ Reference audio tokenized: {ref_audio_tokens.shape}")
                                    
                                    # Add to message content using WAV audio_url (not tokens)
                                    message_content.append(AudioContent(audio_url=wav_audio_url))
                                    # Add tokens to audio_ids for generation
                                    audio_ids.append(ref_audio_tokens)
                                    
                                except Exception as e:
                                    logger.error(f"❌ Error processing reference audio {audio_url}: {str(e)}")
                                    continue
                            else:
                                logger.warning(f"Reference audio not found or empty: {audio_url}")
                    
                    if message_content:
                        messages.append(Message(role="user", content=message_content))
                else:
                    # Simple text content
                    messages.append(Message(role="user", content=[TextContent(text=content)]))
            
            elif role == 'assistant':
                # We don't need to process assistant messages for inference
                # They contain the target audio which we're trying to generate
                pass
        
        return messages, audio_ids

    def extract_target_text_from_chatml(self, chatml_sample: Dict) -> str:
        """Extract target text from ChatML sample with robust error handling."""
        chatml_messages = chatml_sample.get('messages', [])
        
        # If no messages, try direct sample structure
        if not chatml_messages and 'role' in chatml_sample:
            chatml_messages = [chatml_sample]
        
        for msg in chatml_messages:
            if msg.get('role') == 'user':
                content = msg.get('content', [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            text = item.get('text', '')
                            if text:
                                return text
                elif isinstance(content, str):
                    return content
        
        logger.warning("No target text found in ChatML sample")
        return "Hello, this is a test."  # Fallback text

    def test_final_inference(self, chatml_samples: List[Dict], output_dir: str = "final_samples"):
        """Test final corrected zero-shot voice cloning inference."""
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        success_count = 0
        
        for i, sample in enumerate(chatml_samples):
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing Sample {i+1}/{len(chatml_samples)}")
            logger.info(f"{'='*60}")
            
            try:
                sample_id = f"sample_{i:08d}"
                
                # Extract target text from ChatML structure
                target_text = self.extract_target_text_from_chatml(sample)
                logger.info(f"Sample ID: {sample_id}")
                logger.info(f"Target text: {target_text[:100]}...")
                
                # Convert ChatML to proper Message format
                messages, audio_ids = self.convert_chatml_to_messages(sample)
                
                if not audio_ids:
                    logger.warning("⚠️ No reference audio available, performing text-only generation")
                    generation_type = "text_to_speech"
                    audio_ids = None  # Use None instead of empty list
                else:
                    logger.info("🎯 Performing zero-shot voice cloning with reference audio")
                    generation_type = "voice_cloning"
                
                # NOTE: Text is already in messages, no need for separate chunked_text
                # The chunked_text parameter duplicates the text and causes confusion
                logger.info("🚀 Starting generation...")
                waveform, sample_rate, generated_text = self.model_client.generate(
                    messages=messages,
                    audio_ids=audio_ids,
                    chunked_text=None,  # Let model use text from messages only
                    generation_chunk_buffer_size=128,  # Official parameter for long texts
                    temperature=0.6,  # Lower temperature for more faithful text reading
                    top_k=40,  # Official parameter
                    top_p=0.9,  # Lower top_p for better text fidelity
                    ras_win_len=7,  # Repetition-aware sampling
                    ras_win_max_num_repeat=2,
                    seed=42
                )
                
                # Save generated audio
                output_audio_path = os.path.join(output_dir, f"{sample_id}_generated.wav")
                sf.write(output_audio_path, waveform, sample_rate)
                
                # Find and copy reference audio for comparison
                ref_audio_path = None
                chatml_messages = sample.get('messages', [])
                for msg in chatml_messages:
                    if msg.get('role') == 'user':
                        content = msg.get('content', [])
                        if isinstance(content, list):
                            for item in content:
                                if item.get('type') == 'audio':
                                    ref_audio_path = item.get('audio_url', '')
                                    break
                
                if ref_audio_path and os.path.exists(ref_audio_path):
                    ref_output_path = os.path.join(output_dir, f"{sample_id}_reference.wav")
                    ref_waveform, ref_sr = torchaudio.load(ref_audio_path)
                    sf.write(ref_output_path, ref_waveform.numpy().T, ref_sr)
                
                # Save metadata
                metadata = {
                    "sample_id": sample_id,
                    "target_text": target_text,
                    "reference_audio": ref_audio_path,
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
        logger.info(f"FINAL INFERENCE TEST COMPLETED!")
        logger.info(f"{'='*60}")
        logger.info(f"Total samples: {len(chatml_samples)}")
        logger.info(f"Successful: {success_count}")
        logger.info(f"Success rate: {success_count / len(chatml_samples) * 100:.1f}%")
        logger.info(f"Results saved to: {output_dir}")
        
        return results


def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description="Final Zero-Shot Voice Cloning Inference Test")
    parser.add_argument("--model_path", type=str, default="bosonai/higgs-audio-v2-generation-3B-base",
                       help="Path to model (base model or merged LoRA model)")
    parser.add_argument("--tokenizer_path", type=str, default="bosonai/higgs-audio-v2-tokenizer",
                       help="Path to audio tokenizer")
    parser.add_argument("--chatml_file", type=str, default="test_processed/test_chatml_samples.json",
                       help="Path to ChatML samples JSON file")
    parser.add_argument("--output_dir", type=str, default="final_samples",
                       help="Output directory for generated audio files")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Number of samples to test")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducible sample selection (optional)")
    
    args = parser.parse_args()
    
    logger.info("Starting Final Zero-Shot Voice Cloning Inference Test")
    logger.info(f"Using model: {args.model_path}")
    logger.info(f"Using tokenizer: {args.tokenizer_path}")
    
    # Initialize inference test with specified model
    inference_test = FinalInferenceTest(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path
    )
    
    if not os.path.exists(args.chatml_file):
        logger.error(f"ChatML file not found: {args.chatml_file}")
        logger.info("Please run the data processing script first to generate ChatML samples.")
        return
    
    # Load and test samples
    samples = inference_test.load_chatml_samples(args.chatml_file, max_samples=args.num_samples, seed=args.seed)
    results = inference_test.test_final_inference(samples, output_dir=args.output_dir)
    
    logger.info("✅ Final inference test completed!")
    logger.info(f"Check the '{args.output_dir}' directory for results.")


if __name__ == "__main__":
    main()
