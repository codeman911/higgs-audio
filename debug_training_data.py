#!/usr/bin/env python3

import json
import torch
import librosa
from transformers import AutoTokenizer, AutoProcessor

# Import exact components from boson_multimodal
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer

def debug_training_data():
    # Load a sample from the manifest
    manifest_path = "/Users/vikram.solanki/Projects/exp/level1/higgs-audio/data/train_manifest.json"
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        print(f"Total samples in manifest: {len(samples)}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")
        
        # Process the first sample
        sample = samples[0]
        print(f"\nSample keys: {sample.keys()}")
        print(f"Sample messages: {sample['messages']}")
        
        # Process the sample
        input_tokens, label_tokens, audio_contents, speaker_id = prepare_chatml_sample(sample, tokenizer)
        
        print(f"\nInput tokens length: {len(input_tokens)}")
        print(f"Label tokens length: {len(label_tokens)}")
        print(f"First 20 input tokens: {input_tokens[:20]}")
        print(f"First 20 label tokens: {label_tokens[:20]}")
        
        # Count masked vs unmasked tokens
        masked_count = label_tokens.count(-100)
        unmasked_count = len(label_tokens) - masked_count
        print(f"Masked tokens: {masked_count}, Unmasked tokens: {unmasked_count}")
        
        # Check for audio contents
        print(f"Audio contents: {len(audio_contents)}")
        for i, audio_content in enumerate(audio_contents):
            print(f"  Audio {i}: {audio_content}")
        
        # Create a ChatMLDatasetSample
        if audio_contents:
            # Process audio using audio_tokenizer
            audio_tokenizer = load_higgs_audio_tokenizer("bosonai/higgs-audio-v2-tokenizer", device='cpu')
            
            audio_ids_list = []
            audio_waveforms_list = []
            
            for audio_content in audio_contents:
                if audio_content and hasattr(audio_content, 'audio_url'):
                    audio_path = audio_content.audio_url
                    if audio_path and audio_path.endswith('.wav') and len(audio_path) < 100:  # Simple validation
                        try:
                            # Tokenize audio
                            audio_codes = audio_tokenizer.encode(audio_path)
                            # Load waveform at exact sample rate (24000Hz matches inference)
                            waveform, sr = librosa.load(audio_path, sr=24000, mono=True)
                            waveform = torch.tensor(waveform, dtype=torch.float32)
                            
                            audio_ids_list.append(audio_codes)
                            audio_waveforms_list.append(waveform)
                            print(f"Processed audio: {audio_path}")
                        except Exception as e:
                            print(f"Failed to process audio {audio_path}: {e}")
        
            if audio_ids_list:
                # Concatenate audio data
                audio_ids_concat = torch.cat(audio_ids_list, dim=1)
                audio_ids_start = torch.tensor([0] + [c.shape[1] for c in audio_ids_list[:-1]], dtype=torch.long).cumsum(dim=0)
                audio_waveforms_concat = torch.cat(audio_waveforms_list, dim=0)
                audio_waveforms_start = torch.tensor([0] + [wv.shape[0] for wv in audio_waveforms_list[:-1]], dtype=torch.long).cumsum(dim=0)
                audio_sample_rate = torch.tensor([24000])
            else:
                # Empty audio tensors
                audio_ids_concat = torch.zeros((8, 0), dtype=torch.long)
                audio_ids_start = torch.tensor([], dtype=torch.long)
                audio_waveforms_concat = torch.zeros((0,), dtype=torch.float32)
                audio_waveforms_start = torch.tensor([], dtype=torch.long)
                audio_sample_rate = torch.tensor([24000])
        else:
            # Empty audio tensors
            audio_ids_concat = torch.zeros((8, 0), dtype=torch.long)
            audio_ids_start = torch.tensor([], dtype=torch.long)
            audio_waveforms_concat = torch.zeros((0,), dtype=torch.float32)
            audio_waveforms_start = torch.tensor([], dtype=torch.long)
            audio_sample_rate = torch.tensor([24000])
        
        # Handle speaker ID conversion
        numeric_speaker_id = 0 if isinstance(speaker_id, str) or speaker_id is None else int(speaker_id)
        audio_speaker_indices = torch.tensor([numeric_speaker_id], dtype=torch.long)
        
        # Create ChatMLDatasetSample
        chatml_sample = ChatMLDatasetSample(
            input_ids=torch.tensor(input_tokens, dtype=torch.long),
            label_ids=torch.tensor(label_tokens, dtype=torch.long),
            audio_ids_concat=audio_ids_concat,
            audio_ids_start=audio_ids_start,
            audio_waveforms_concat=audio_waveforms_concat,
            audio_waveforms_start=audio_waveforms_start,
            audio_sample_rate=audio_sample_rate,
            audio_speaker_indices=audio_speaker_indices
        )
        
        print(f"\nChatMLDatasetSample created:")
        print(f"  input_ids shape: {chatml_sample.input_ids.shape}")
        print(f"  label_ids shape: {chatml_sample.label_ids.shape}")
        print(f"  audio_ids_concat shape: {chatml_sample.audio_ids_concat.shape}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_training_data()