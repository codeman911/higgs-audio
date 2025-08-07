#!/usr/bin/env python3
"""
Zero-shot voice cloning inference test using Higgs-Audio pretrained model.
This script tests the pretrained model on processed ChatML samples to validate
the data processing pipeline before LoRA fine-tuning.
"""

import sys
import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any
import torch
import torchaudio
import soundfile as sf
from dataclasses import asdict
import re
import shutil
from loguru import logger
import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent
while not (project_root / "boson_multimodal").exists() and project_root != project_root.parent:
    project_root = project_root.parent

if project_root not in [Path(p) for p in sys.path]:
    sys.path.insert(0, str(project_root))

try:
    # Use exact imports from examples/generation.py that work
    from boson_multimodal.model.higgs_audio import HiggsAudioModel, HiggsAudioConfig
    from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent
    from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
    from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
    from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
    from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern
    from transformers import AutoConfig, AutoTokenizer
    from transformers.cache_utils import StaticCache
    from dataclasses import asdict
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class HiggsAudioModelClient:
    """Simplified client based on examples/generation.py"""
    
    def __init__(
        self,
        model_path,
        audio_tokenizer,
        device=None,
        max_new_tokens=2048,
        use_static_kv_cache=False,
    ):
        # Device setup
        if device is not None:
            self._device = device
        else:
            if torch.cuda.is_available():
                self._device = "cuda:0"
            else:
                self._device = "cpu"

        logger.info(f"Using device: {self._device}")
        
        # Load audio tokenizer
        if isinstance(audio_tokenizer, str):
            audio_tokenizer_device = "cpu" if self._device == "mps" else self._device
            self._audio_tokenizer = load_higgs_audio_tokenizer(audio_tokenizer, device=audio_tokenizer_device)
        else:
            self._audio_tokenizer = audio_tokenizer

        # Load model
        self._model = HiggsAudioModel.from_pretrained(
            model_path,
            device_map=self._device,
            torch_dtype=torch.bfloat16,
        )
        self._model.eval()
        self._use_static_kv_cache = use_static_kv_cache

        # Load tokenizer and config
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._config = AutoConfig.from_pretrained(model_path)
        self._max_new_tokens = max_new_tokens
        
        # Setup collator
        self._collator = HiggsAudioSampleCollator(
            whisper_processor=None,
            audio_in_token_id=self._config.audio_in_token_idx,
            audio_out_token_id=self._config.audio_out_token_idx,
            audio_stream_bos_id=self._config.audio_stream_bos_id,
            audio_stream_eos_id=self._config.audio_stream_eos_id,
            encode_whisper_embed=self._config.encode_whisper_embed,
            pad_token_id=self._config.pad_token_id,
            return_audio_in_tokens=self._config.encode_audio_in_tokens,
            use_delay_pattern=self._config.use_delay_pattern,
            round_to=1,
            audio_num_codebooks=self._config.audio_num_codebooks,
        )
        self.kv_caches = None

    @torch.inference_mode()
    def generate(
        self,
        messages,
        audio_ids,
        chunked_text,
        generation_chunk_buffer_size=None,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        ras_win_len=7,
        ras_win_max_num_repeat=2,
        seed=123,
    ):
        """Generate audio from messages and audio context"""
        if ras_win_len is not None and ras_win_len <= 0:
            ras_win_len = None
        sr = 24000
        audio_out_ids_l = []
        generated_audio_ids = []
        generation_messages = []
        
        for idx, chunk_text in tqdm.tqdm(
            enumerate(chunked_text), desc="Generating audio chunks", total=len(chunked_text)
        ):
            generation_messages.append(
                Message(
                    role="user",
                    content=chunk_text,
                )
            )
            chatml_sample = ChatMLSample(messages=messages + generation_messages)
            input_tokens, _, _, _ = prepare_chatml_sample(chatml_sample, self._tokenizer)
            postfix = self._tokenizer.encode(
                "<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False
            )
            input_tokens.extend(postfix)

            logger.info(f"========= Chunk {idx} Input =========")
            logger.info(self._tokenizer.decode(input_tokens))
            context_audio_ids = audio_ids + generated_audio_ids

            curr_sample = ChatMLDatasetSample(
                input_ids=torch.LongTensor(input_tokens),
                label_ids=None,
                audio_ids_concat=torch.concat([ele.cpu() for ele in context_audio_ids], dim=1)
                if context_audio_ids
                else None,
                audio_ids_start=torch.cumsum(
                    torch.tensor([0] + [ele.shape[1] for ele in context_audio_ids], dtype=torch.long), dim=0
                )
                if context_audio_ids
                else None,
                audio_waveforms_concat=None,
                audio_waveforms_start=None,
                audio_sample_rate=None,
                audio_speaker_indices=None,
            )

            batch_data = self._collator([curr_sample])
            batch = asdict(batch_data)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.contiguous().to(self._device)

            # Generate audio
            outputs = self._model.generate(
                **batch,
                max_new_tokens=self._max_new_tokens,
                use_cache=True,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                past_key_values_buckets=self.kv_caches,
                ras_win_len=ras_win_len,
                ras_win_max_num_repeat=ras_win_max_num_repeat,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                tokenizer=self._tokenizer,
                seed=seed,
            )

            step_audio_out_ids_l = []
            for ele in outputs[1]:
                audio_out_ids = ele
                if self._config.use_delay_pattern:
                    audio_out_ids = revert_delay_pattern(audio_out_ids)
                step_audio_out_ids_l.append(audio_out_ids.clip(0, self._audio_tokenizer.codebook_size - 1)[:, 1:-1])
            audio_out_ids = torch.concat(step_audio_out_ids_l, dim=1)
            audio_out_ids_l.append(audio_out_ids)
            generated_audio_ids.append(audio_out_ids)

            generation_messages.append(
                Message(
                    role="assistant",
                    content=AudioContent(audio_url=""),
                )
            )
            if generation_chunk_buffer_size is not None and len(generated_audio_ids) > generation_chunk_buffer_size:
                generated_audio_ids = generated_audio_ids[-generation_chunk_buffer_size:]
                generation_messages = generation_messages[(-2 * generation_chunk_buffer_size) :]

        logger.info(f"========= Final Text output =========")
        logger.info(self._tokenizer.decode(outputs[0][0]))
        concat_audio_out_ids = torch.concat(audio_out_ids_l, dim=1)

        # Fix MPS compatibility: detach and move to CPU before decoding
        if concat_audio_out_ids.device.type == "mps":
            concat_audio_out_ids_cpu = concat_audio_out_ids.detach().cpu()
        else:
            concat_audio_out_ids_cpu = concat_audio_out_ids

        concat_wv = self._audio_tokenizer.decode(concat_audio_out_ids_cpu.unsqueeze(0))[0, 0]
        text_result = self._tokenizer.decode(outputs[0][0])
        return concat_wv, sr, text_result


class InferenceTest:
    """Test inference with pretrained model"""
    
    def __init__(self, model_path: str, audio_tokenizer_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.audio_tokenizer_path = audio_tokenizer_path
        self.device = device
        
        print(f"🤖 Loading Higgs-Audio model: {model_path}")
        print(f"🎵 Loading audio tokenizer: {audio_tokenizer_path}")
        print(f"🔧 Device: {device}")
        
        # Load model client
        try:
            self.model_client = HiggsAudioModelClient(
                model_path=model_path,
                audio_tokenizer=audio_tokenizer_path,
                device=device,
                max_new_tokens=1500,
                use_static_kv_cache=True if device == "cuda" else False,
            )
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

    def load_processed_samples(self, chatml_file: str, num_samples: int = 10):
        """Load random samples from processed ChatML data"""
        
        print(f"📂 Loading samples from: {chatml_file}")
        
        with open(chatml_file, 'r', encoding='utf-8') as f:
            all_samples = json.load(f)
        
        print(f"📊 Total available samples: {len(all_samples)}")
        
        # Select random samples
        selected_samples = random.sample(all_samples, min(num_samples, len(all_samples)))
        print(f"🎲 Selected {len(selected_samples)} random samples for testing")
        
        return selected_samples

    def test_inference(self, chatml_file: str, output_dir: str, num_samples: int = 10):
        """Run inference test on random samples using correct API"""
        
        print("🧪 Starting Inference Test")
        print("=" * 50)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load samples
        samples = self.load_processed_samples(chatml_file, num_samples)
        
        results = []
        
        for i, sample_data in enumerate(samples):
            sample_id = sample_data.get('speaker', f'sample_{i:05d}')
            print(f"\n🔄 Processing sample {i+1}/{len(samples)}: {sample_id}")
            
            try:
                # Extract target text and reference audio info
                target_text = ""
                ref_audio_path = ""
                
                for msg in sample_data['messages']:
                    if msg['role'] == 'user':
                        # Extract text and reference audio from user message
                        if isinstance(msg['content'], list):
                            for content in msg['content']:
                                if content['type'] == 'text':
                                    target_text = content['text']
                                elif content['type'] == 'audio':
                                    ref_audio_path = content['audio_url']
                        else:
                            target_text = msg['content']
                
                print(f"   🎯 Target text: {target_text[:50]}...")
                print(f"   🎤 Reference audio: {os.path.basename(ref_audio_path)}")
                
                # Skip samples with very short audio (< 1 second) to avoid kernel size errors
                if ref_audio_path and os.path.exists(ref_audio_path):
                    try:
                        audio_info = sf.info(ref_audio_path)
                        if audio_info.duration < 1.0:
                            print(f"   ⚠️  Skipping sample with very short audio ({audio_info.duration:.2f}s)")
                            continue
                    except:
                        print(f"   ⚠️  Could not read audio file info, skipping")
                        continue
                
                # Prepare messages and audio_ids like in generation.py
                messages = []
                audio_ids = []
                
                # System message
                messages.append(Message(role="system", content="You are a helpful assistant."))
                
                # Add reference audio if available
                if ref_audio_path and os.path.exists(ref_audio_path):
                    try:
                        print(f"   🔍 Loading reference audio: {ref_audio_path}")
                        
                        # Check file size and permissions
                        file_stat = os.stat(ref_audio_path)
                        print(f"   📏 Audio file size: {file_stat.st_size} bytes")
                        
                        if file_stat.st_size == 0:
                            print(f"   ⚠️  Audio file is empty, skipping reference audio")
                        else:
                            # Load and tokenize reference audio
                            waveform, sample_rate = torchaudio.load(ref_audio_path)
                            print(f"   🎵 Loaded audio: {waveform.shape}, sample_rate: {sample_rate}")
                            
                            if sample_rate != 24000:
                                waveform = torchaudio.functional.resample(waveform, sample_rate, 24000)
                                print(f"   🔄 Resampled to 24kHz: {waveform.shape}")
                            
                            # Tokenize audio
                            audio_tokens = self.model_client._audio_tokenizer.encode(waveform.unsqueeze(0))
                            audio_ids.append(audio_tokens[0])
                            print(f"   🎯 Audio tokenized: {audio_tokens[0].shape}")
                            
                            # Add reference audio message
                            messages.append(Message(
                                role="user",
                                content=AudioContent(audio_url=ref_audio_path)
                            ))
                            print(f"   ✅ Reference audio loaded successfully")
                            
                    except Exception as e:
                        print(f"   ❌ Error processing reference audio: {type(e).__name__}: {str(e)}")
                        print(f"   🔄 Continuing with text-only generation (no voice cloning)")
                else:
                    if ref_audio_path:
                        print(f"   ⚠️  Reference audio file not found: {ref_audio_path}")
                    else:
                        print(f"   ℹ️  No reference audio path provided")
                    print(f"   🔄 Proceeding with text-only generation")
                
                # Prepare chunked text (simple approach)
                chunked_text = [target_text] if target_text else ["Generate speech."]
                
                # Run generation using the correct API
                concat_wv, sr, text_result = self.model_client.generate(
                    messages=messages,
                    audio_ids=audio_ids,
                    chunked_text=chunked_text,
                    generation_chunk_buffer_size=None,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    seed=123
                )
                
                # Save generated audio
                if concat_wv is not None and len(concat_wv) > 0:
                    # Save generated audio
                    generated_audio_path = output_path / f"{sample_id}_generated.wav"
                    sf.write(generated_audio_path, concat_wv, sr)
                    
                    # Copy reference audio for comparison
                    if ref_audio_path and os.path.exists(ref_audio_path):
                        ref_audio_dest = output_path / f"{sample_id}_reference.wav"
                        shutil.copy2(ref_audio_path, ref_audio_dest)
                    
                    # Save metadata
                    metadata = {
                        'sample_id': sample_id,
                        'target_text': target_text,
                        'reference_audio': ref_audio_path,
                        'generated_audio': str(generated_audio_path),
                        'sampling_rate': sr,
                        'generated_text': text_result,
                        'success': True
                    }
                    
                    with open(output_path / f"{sample_id}_metadata.json", 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                    
                    print(f"   ✅ Generated audio saved: {generated_audio_path}")
                    results.append(metadata)
                    
                else:
                    print(f"   ❌ No audio generated for sample {sample_id}")
                    results.append({
                        'sample_id': sample_id,
                        'target_text': target_text,
                        'success': False,
                        'error': 'No audio generated'
                    })
                    
            except Exception as e:
                print(f"   ❌ Error processing sample {sample_id}: {e}")
                results.append({
                    'sample_id': sample_id,
                    'success': False,
                    'error': str(e)
                })
        
        # Save summary
        successful = sum(1 for r in results if r.get('success', False))
        summary = {
            'total_samples': len(results),
            'successful': successful,
            'failed': len(results) - successful,
            'success_rate': successful / len(results) if results else 0,
            'results': results
        }
        
        with open(output_path / 'test_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Inference Test Results:")
        print(f"   • Total samples: {summary['total_samples']}")
        print(f"   • Successful: {summary['successful']}")
        print(f"   • Failed: {summary['failed']}")
        print(f"   • Success rate: {summary['success_rate']:.1%}")
        print(f"   • Results saved to: {output_dir}")
        
        if summary['successful'] == 0:
            print("\n⚠️  Inference test had issues.")
        else:
            print("\n🎉 Inference test completed successfully!")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Test zero-shot voice cloning inference")
    parser.add_argument(
        "--chatml_file", 
        type=str, 
        required=True,
        help="Path to processed ChatML samples JSON file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./tested_samples",
        help="Output directory for test results"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=10,
        help="Number of random samples to test"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="bosonai/higgs-audio-v2-generation-3B-base",
        help="Higgs-Audio model path"
    )
    parser.add_argument(
        "--audio_tokenizer_path", 
        type=str, 
        default="bosonai/higgs-audio-v2-tokenizer",
        help="Audio tokenizer path"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = InferenceTest(
        model_path=args.model_path,
        audio_tokenizer_path=args.audio_tokenizer_path,
        device=args.device
    )
    
    # Run test
    summary = tester.test_inference(
        chatml_file=args.chatml_file,
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )
    
    # Exit with appropriate code
    exit(0 if summary['successful'] > 0 else 1)


if __name__ == "__main__":
    main()
