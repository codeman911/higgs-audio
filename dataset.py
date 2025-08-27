import json
import os
import torch
import librosa
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List

from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample


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
        # Updated to handle audio labels properly
        result = prepare_chatml_sample(sample, self.tokenizer)
        
        # Handle different versions of prepare_chatml_sample
        if len(result) == 4:
            input_tokens, label_tokens, audio_contents, speaker_id = result
            # For the old version, we need to manually identify audio labels
            # We'll assume that audio contents in assistant responses are labels
            audio_label_contents = []
            # Extract audio label contents from assistant messages
            messages = sample['messages'] if isinstance(sample, dict) else sample.messages
            for message in messages:
                role = message['role'] if isinstance(message, dict) else message.role
                content = message['content'] if isinstance(message, dict) else message.content
                
                if role == 'assistant':
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'audio':
                                audio_label_contents.append(item)
                            elif hasattr(item, 'type') and getattr(item, 'type', None) == 'audio':
                                audio_label_contents.append(item)
                    elif isinstance(content, dict) and content.get('type') == 'audio':
                        audio_label_contents.append(content)
                    elif hasattr(content, 'type') and getattr(content, 'type', None) == 'audio':
                        audio_label_contents.append(content)
        elif len(result) == 5:
            input_tokens, label_tokens, audio_contents, audio_label_contents, speaker_id = result
        else:
            # Skip invalid samples, try next one
            return self.__getitem__((idx + 1) % len(self.samples))
        
        if input_tokens is None or label_tokens is None:
            # Skip invalid samples, try next one
            return self.__getitem__((idx + 1) % len(self.samples))
        
        # Process audio using audio_tokenizer - EXACT pattern from training scripts
        audio_ids_list = []
        audio_waveforms_list = []
        label_audio_ids_list = []  # For audio labels
        
        # Ensure audio_contents is a list
        if not isinstance(audio_contents, list):
            audio_contents = []
            
        # Ensure audio_label_contents is a list
        if not isinstance(audio_label_contents, list):
            audio_label_contents = [None] * len(audio_contents)
        elif len(audio_label_contents) < len(audio_contents):
            # Pad audio_label_contents if it's shorter than audio_contents
            audio_label_contents.extend([None] * (len(audio_contents) - len(audio_label_contents)))
        
        # CRITICAL FIX: Process reference audio (for conditioning) and target audio (for labels) separately
        # Process reference audio (goes to audio_ids_list for conditioning)
        for i, audio_content in enumerate(audio_contents):
            if audio_content and (hasattr(audio_content, 'audio_url') or isinstance(audio_content, dict)):
                audio_path = audio_content.audio_url if hasattr(audio_content, 'audio_url') else audio_content.get('audio_url')
                if audio_path and os.path.exists(audio_path):
                    # Tokenize audio - EXACT pattern from inference
                    try:
                        audio_codes = self.audio_tokenizer.encode(audio_path)
                        # Load waveform at exact sample rate (24000Hz matches inference)
                        waveform, sr = librosa.load(audio_path, sr=24000, mono=True)
                        waveform = torch.tensor(waveform, dtype=torch.float32)
                        
                        audio_ids_list.append(audio_codes)
                        audio_waveforms_list.append(waveform)
                    except Exception as e:
                        pass
        
        # CRITICAL FIX: Process target audio labels (goes to label_audio_ids_list for training)
        # This is the key fix - we need to process audio_label_contents to create label_audio_ids
        for i, audio_label_content in enumerate(audio_label_contents):
            if audio_label_content is not None:
                # Check if it's an AudioContent object with audio_url attribute
                if hasattr(audio_label_content, 'audio_url'):
                    label_audio_path = audio_label_content.audio_url
                    if label_audio_path and os.path.exists(label_audio_path):
                        try:
                            label_audio_codes = self.audio_tokenizer.encode(label_audio_path)
                            label_audio_ids_list.append(label_audio_codes)
                        except Exception as e:
                            pass
                # Check if it's a dict with audio_url key
                elif isinstance(audio_label_content, dict) and 'audio_url' in audio_label_content:
                    label_audio_path = audio_label_content['audio_url']
                    if label_audio_path and os.path.exists(label_audio_path):
                        try:
                            label_audio_codes = self.audio_tokenizer.encode(label_audio_path)
                            label_audio_ids_list.append(label_audio_codes)
                        except Exception as e:
                            pass
        
        if audio_ids_list:
            # Concatenate audio data - EXACT pattern from working scripts
            audio_ids_concat = torch.cat(audio_ids_list, dim=1)
            audio_ids_start = torch.tensor([0] + [c.shape[1] for c in audio_ids_list[:-1]], dtype=torch.long).cumsum(dim=0)
            audio_waveforms_concat = torch.cat(audio_waveforms_list, dim=0)
            audio_waveforms_start = torch.tensor([0] + [wv.shape[0] for wv in audio_waveforms_list[:-1]], dtype=torch.long).cumsum(dim=0)
            audio_sample_rate = torch.tensor([24000])
            
            # CRITICAL FIX: Concatenate audio label data if available
            if label_audio_ids_list:
                label_audio_ids_concat = torch.cat(label_audio_ids_list, dim=1)
            else:
                label_audio_ids_concat = None
        else:
            # Empty audio tensors - EXACT pattern from working scripts  
            audio_ids_concat = torch.zeros((8, 0), dtype=torch.long)  # 8 codebooks (matches bosonai/higgs-audio-v2-tokenizer)
            audio_ids_start = torch.tensor([], dtype=torch.long)
            audio_waveforms_concat = torch.zeros((0,), dtype=torch.float32)
            audio_waveforms_start = torch.tensor([], dtype=torch.long)
            audio_sample_rate = torch.tensor([24000])
            label_audio_ids_concat = None
        
        # Handle speaker ID conversion
        numeric_speaker_id = 0 if isinstance(speaker_id, str) or speaker_id is None else int(speaker_id)
        audio_speaker_indices = torch.tensor([numeric_speaker_id], dtype=torch.long)
        
        # Create ChatMLDatasetSample with EXACT field structure
        return ChatMLDatasetSample(
            input_ids=torch.tensor(input_tokens, dtype=torch.long),
            label_ids=torch.tensor(label_tokens, dtype=torch.long),
            audio_ids_concat=audio_ids_concat.long() if audio_ids_concat is not None else torch.zeros((8, 0), dtype=torch.long),
            audio_ids_start=audio_ids_start.long() if audio_ids_start is not None else torch.tensor([], dtype=torch.long),
            audio_waveforms_concat=audio_waveforms_concat,
            audio_waveforms_start=audio_waveforms_start.long() if audio_waveforms_start is not None else torch.tensor([], dtype=torch.long),
            audio_sample_rate=audio_sample_rate.float() if audio_sample_rate is not None else torch.tensor([24000], dtype=torch.float32),
            audio_speaker_indices=audio_speaker_indices.long() if audio_speaker_indices is not None else torch.tensor([0], dtype=torch.long),
            audio_label_ids_concat=label_audio_ids_concat.long() if label_audio_ids_concat is not None else None
        )


def create_collator(config, whisper_processor):
    """Create collator with EXACT parameters from working implementation"""
    return ExtendedHiggsAudioSampleCollator(
        whisper_processor=whisper_processor,
        encode_whisper_embed=True,  # Always enable for training
        audio_in_token_id=config.audio_in_token_idx,
        audio_out_token_id=config.audio_out_token_idx,
        audio_stream_bos_id=config.audio_stream_bos_id,
        audio_stream_eos_id=config.audio_stream_eos_id,
        pad_token_id=config.pad_token_id,
        return_audio_in_tokens=True,  # CRITICAL FIX: Enable for proper audio handling
        use_delay_pattern=False,      # Match working implementation
        audio_num_codebooks=8,        # Explicitly set to 8 codebooks
        round_to=8,                   # Match working implementation
        mask_audio_out_token_label=False,  # CRITICAL FIX: Disable over-masking
    )


@dataclass
class ExtendedHiggsAudioBatchInput:
    """
    Extended HiggsAudioBatchInput with __len__ method for Trainer compatibility
    """
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __len__(self):
        """Return the batch size based on input_ids"""
        if hasattr(self, 'input_ids') and getattr(self, 'input_ids', None) is not None:
            return self.input_ids.shape[0]
        else:
            return 0
    
    def __getitem__(self, key):
        """Allow dictionary-style access for compatibility"""
        return getattr(self, key, None)
    
    def __contains__(self, key):
        """Check if attribute exists"""
        return hasattr(self, key)
    
    def keys(self):
        """Return all attribute names for compatibility"""
        return [attr for attr in dir(self) if not attr.startswith('_') and not callable(getattr(self, attr))]


class ExtendedHiggsAudioSampleCollator:
    """
    Extended collator that returns our custom batch input class
    """
    
    def __init__(self, **kwargs):
        from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
        self.base_collator = HiggsAudioSampleCollator(**kwargs)
    
    def __call__(self, batch: List[ChatMLDatasetSample]):
        # 1. Call the official base collator to do all the complex padding and alignment work
        batch_input = self.base_collator(batch)
        
        # 2. Convert to our extended class and pass in the perfectly aligned labels
        extended_batch = ExtendedHiggsAudioBatchInput(
            input_ids=batch_input.input_ids,
            attention_mask=batch_input.attention_mask,
            audio_features=batch_input.audio_features,
            audio_feature_attention_mask=batch_input.audio_feature_attention_mask,
            audio_out_ids=batch_input.audio_out_ids,
            audio_out_ids_start=batch_input.audio_out_ids_start,
            audio_out_ids_start_group_loc=batch_input.audio_out_ids_start_group_loc,
            audio_in_ids=batch_input.audio_in_ids,
            audio_in_ids_start=batch_input.audio_in_ids_start,
            label_ids=batch_input.label_ids,
            label_audio_ids=batch_input.label_audio_ids, # Use the properly aligned labels from base collator
            reward=batch_input.reward,
        )
        
        return extended_batch