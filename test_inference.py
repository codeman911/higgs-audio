#!/usr/bin/env python3
"""
Test zero-shot voice cloning inference with pretrained Higgs-Audio model
Takes random samples from processed ChatML data and generates audio
"""

import os
import json
import random
import shutil
from pathlib import Path
import sys
import argparse
import soundfile as sf
import torch

# Robust import handling
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

try:
    from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
    from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent
    from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running from the higgs-audio project root")
    sys.exit(1)


class InferenceTest:
    """Test inference with pretrained model"""
    
    def __init__(
        self,
        model_path: str = "bosonai/higgs-audio-v2-generation-3B-base",
        audio_tokenizer_path: str = "bosonai/higgs-audio-v2-tokenizer",
        device: str = "cuda"
    ):
        self.model_path = model_path
        self.audio_tokenizer_path = audio_tokenizer_path
        self.device = device
        
        print(f"🤖 Loading Higgs-Audio model: {model_path}")
        print(f"🎵 Loading audio tokenizer: {audio_tokenizer_path}")
        print(f"🔧 Device: {device}")
        
        # Load serve engine
        try:
            self.serve_engine = HiggsAudioServeEngine(
                model_name_or_path=model_path,
                audio_tokenizer_name_or_path=audio_tokenizer_path,
                device=device
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
    
    def create_chatml_sample(self, sample_data):
        """Convert processed sample back to ChatMLSample object"""
        
        messages = []
        for msg_data in sample_data['messages']:
            content = msg_data['content']
            
            if isinstance(content, str):
                # Simple text content
                message_content = content
            else:
                # Multimodal content
                message_content = []
                for c in content:
                    if c['type'] == 'text':
                        message_content.append(TextContent(text=c['text']))
                    elif c['type'] == 'audio':
                        message_content.append(AudioContent(
                            audio_url=c['audio_url'],
                            raw_audio=c.get('raw_audio', ''),
                            duration=c.get('duration')
                        ))
            
            messages.append(Message(
                role=msg_data['role'],
                content=message_content
            ))
        
        return ChatMLSample(
            messages=messages,
            start_index=sample_data.get('start_index', 0),
            speaker=sample_data.get('speaker'),
            misc=sample_data.get('misc', {})
        )
    
    def test_inference(self, chatml_file: str, output_dir: str, num_samples: int = 10):
        """Run inference test on random samples"""
        
        print("🧪 Starting Inference Test")
        print("=" * 50)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load samples
        samples = self.load_processed_samples(chatml_file, num_samples)
        
        results = []
        
        for i, sample_data in enumerate(samples):
            sample_id = sample_data.get('speaker', f'sample_{i}')
            print(f"\n🔄 Processing sample {i+1}/{len(samples)}: {sample_id}")
            
            try:
                # Create sample directory
                sample_dir = output_path / f"sample_{i:03d}_{sample_id}"
                sample_dir.mkdir(exist_ok=True)
                
                # Extract information from sample
                user_msg = sample_data['messages'][1]  # User message
                assistant_msg = sample_data['messages'][2]  # Assistant message
                
                # Find reference audio, target text, and target audio
                ref_audio_path = None
                target_text = None
                target_audio_path = None
                
                # From user message
                for content in user_msg['content']:
                    if content['type'] == 'text':
                        target_text = content['text']
                    elif content['type'] == 'audio':
                        ref_audio_path = content['audio_url']
                
                # From assistant message
                for content in assistant_msg['content']:
                    if content['type'] == 'audio':
                        target_audio_path = content['audio_url']
                
                if not all([ref_audio_path, target_text, target_audio_path]):
                    print(f"⚠️  Missing data for sample {sample_id}")
                    continue
                
                # Copy reference and target audio files
                ref_audio_dest = sample_dir / "reference_audio.wav"
                target_audio_dest = sample_dir / "target_audio.wav"
                
                shutil.copy2(ref_audio_path, ref_audio_dest)
                shutil.copy2(target_audio_path, target_audio_dest)
                
                # Save target text
                with open(sample_dir / "target_text.txt", 'w', encoding='utf-8') as f:
                    f.write(target_text)
                
                # Create ChatML sample for inference
                chatml_sample = self.create_chatml_sample(sample_data)
                
                # Run inference
                print(f"   🎯 Target text: {target_text[:50]}...")
                print(f"   🎤 Reference audio: {Path(ref_audio_path).name}")
                
                response = self.serve_engine.generate(
                    chatml_sample,
                    max_new_tokens=1500,
                    temperature=0.7,
                    top_p=0.95
                )
                
                # Save generated audio
                if hasattr(response, 'audio_waveform') and response.audio_waveform is not None:
                    generated_audio_path = sample_dir / "generated_audio.wav"
                    
                    # Convert tensor to numpy if needed
                    if torch.is_tensor(response.audio_waveform):
                        audio_data = response.audio_waveform.cpu().numpy()
                    else:
                        audio_data = response.audio_waveform
                    
                    # Get sample rate (default to 24000 if not available)
                    sample_rate = getattr(response, 'sample_rate', 24000)
                    
                    sf.write(generated_audio_path, audio_data, sample_rate)
                    print(f"   ✅ Generated audio saved: {generated_audio_path.name}")
                    
                    # Save metadata
                    metadata = {
                        'sample_id': sample_id,
                        'target_text': target_text,
                        'reference_audio': str(ref_audio_dest),
                        'target_audio': str(target_audio_dest),
                        'generated_audio': str(generated_audio_path),
                        'sample_rate': sample_rate,
                        'model_path': self.model_path,
                        'success': True
                    }
                    
                    results.append(metadata)
                    
                else:
                    print(f"   ❌ No audio generated for sample {sample_id}")
                    metadata = {
                        'sample_id': sample_id,
                        'target_text': target_text,
                        'success': False,
                        'error': 'No audio generated'
                    }
                    results.append(metadata)
                
                # Save sample metadata
                with open(sample_dir / "metadata.json", 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                    
            except Exception as e:
                print(f"   ❌ Error processing sample {sample_id}: {e}")
                results.append({
                    'sample_id': sample_id,
                    'success': False,
                    'error': str(e)
                })
        
        # Save overall results
        results_file = output_path / "inference_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Print summary
        successful = sum(1 for r in results if r.get('success', False))
        total = len(results)
        
        print(f"\n📊 Inference Test Results:")
        print(f"   • Total samples: {total}")
        print(f"   • Successful: {successful}")
        print(f"   • Failed: {total - successful}")
        print(f"   • Success rate: {successful/total*100:.1f}%")
        print(f"   • Results saved to: {output_dir}")
        
        if successful > 0:
            print(f"\n🎉 Inference test PASSED!")
            print(f"The pretrained model works with your data format.")
            print(f"You can now proceed with LoRA fine-tuning.")
        else:
            print(f"\n⚠️  Inference test had issues.")
            print(f"Please check the errors and data format.")
        
        return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test inference with pretrained model")
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
        help="Path to pretrained model"
    )
    parser.add_argument(
        "--audio_tokenizer_path", 
        type=str, 
        default="bosonai/higgs-audio-v2-tokenizer",
        help="Path to audio tokenizer"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    # Create inference tester
    tester = InferenceTest(
        model_path=args.model_path,
        audio_tokenizer_path=args.audio_tokenizer_path,
        device=args.device
    )
    
    # Run test
    results = tester.test_inference(
        chatml_file=args.chatml_file,
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )
    
    return 0 if any(r.get('success', False) for r in results) else 1


if __name__ == "__main__":
    exit(main())
