"""
Minimal dataset implementation that mirrors inference preprocessing exactly.
Strictly reuses boson_multimodal components without modifications.
"""

import os
import json
import torch
import librosa
from typing import List, Dict, Any
from torch.utils.data import Dataset

# Import exact components from boson_multimodal
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer


class HiggsAudioDataset(Dataset):
    """Dataset that mirrors inference preprocessing exactly."""
    
    def __init__(self, 
                 manifest_path: str,
                 tokenizer,
                 audio_tokenizer):
        """
        Args:
            manifest_path: Path to ChatML JSON manifest
            tokenizer: Text tokenizer from HiggsAudioModel
            audio_tokenizer: Audio tokenizer from load_higgs_audio_tokenizer
        """
        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
        
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> ChatMLDatasetSample:
        """Process sample using exact inference pipeline logic."""
        sample = self.samples[idx]
        
        # Use EXACT prepare_chatml_sample from boson_multimodal
        input_tokens, label_tokens, audio_contents, speaker_id = prepare_chatml_sample(
            sample, self.tokenizer
        )
        
        if input_tokens is None or label_tokens is None:
            # Skip invalid samples, try next one
            return self.__getitem__((idx + 1) % len(self.samples))
        
        # Process audio using audio_tokenizer - EXACT pattern from training scripts
        audio_ids_list = []
        audio_waveforms_list = []
        
        for audio_content in audio_contents:
            if audio_content and hasattr(audio_content, 'audio_url'):
                audio_path = audio_content.audio_url
                if audio_path and os.path.exists(audio_path):
                    # Tokenize audio
                    audio_codes = self.audio_tokenizer.encode(audio_path)
                    # Load waveform at exact sample rate
                    waveform, sr = librosa.load(audio_path, sr=24000, mono=True)
                    waveform = torch.tensor(waveform, dtype=torch.float32)
                    
                    audio_ids_list.append(audio_codes)
                    audio_waveforms_list.append(waveform)
        
        if audio_ids_list:
            # Concatenate audio data - EXACT pattern from working scripts
            audio_ids_concat = torch.cat(audio_ids_list, dim=1)
            audio_ids_start = torch.tensor([0] + [c.shape[1] for c in audio_ids_list[:-1]], dtype=torch.long).cumsum(dim=0)
            audio_waveforms_concat = torch.cat(audio_waveforms_list, dim=0)
            audio_waveforms_start = torch.tensor([0] + [wv.shape[0] for wv in audio_waveforms_list[:-1]], dtype=torch.long).cumsum(dim=0)
            audio_sample_rate = torch.tensor([24000])
        else:
            # Empty audio tensors - EXACT pattern from working scripts
            audio_ids_concat = torch.zeros((8, 0), dtype=torch.long)  # 8 codebooks
            audio_ids_start = torch.tensor([], dtype=torch.long)
            audio_waveforms_concat = torch.zeros((0,), dtype=torch.float32)
            audio_waveforms_start = torch.tensor([], dtype=torch.long)
            audio_sample_rate = torch.tensor([24000])
        
        # Handle speaker ID conversion
        numeric_speaker_id = 0 if isinstance(speaker_id, str) or speaker_id is None else int(speaker_id)
        audio_speaker_indices = torch.tensor([numeric_speaker_id], dtype=torch.long)
        
        # Create ChatMLDatasetSample with EXACT field structure
        return ChatMLDatasetSample(
            input_ids=torch.tensor(input_tokens, dtype=torch.long),
            label_ids=torch.tensor(label_tokens, dtype=torch.long),
            audio_ids_concat=audio_ids_concat,
            audio_ids_start=audio_ids_start,
            audio_waveforms_concat=audio_waveforms_concat,
            audio_waveforms_start=audio_waveforms_start,
            audio_sample_rate=audio_sample_rate,
            audio_speaker_indices=audio_speaker_indices
        )


def create_collator(config, whisper_processor):
    """Create collator with EXACT parameters from serve_engine.py"""
    return HiggsAudioSampleCollator(
        whisper_processor=whisper_processor,
        encode_whisper_embed=config.encode_whisper_embed,
        audio_in_token_id=config.audio_in_token_idx,
        audio_out_token_id=config.audio_out_token_idx,
        audio_stream_bos_id=config.audio_stream_bos_id,
        audio_stream_eos_id=config.audio_stream_eos_id,
        pad_token_id=config.pad_token_id,
        return_audio_in_tokens=False,  # EXACT from serve_engine.py
        use_delay_pattern=config.use_delay_pattern,
        audio_num_codebooks=config.audio_num_codebooks,
        round_to=1,  # EXACT from serve_engine.py
    )