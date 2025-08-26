#!/usr/bin/env python3

import json
import torch
from transformers import AutoTokenizer, AutoProcessor
from transformers.models.whisper.processing_whisper import WhisperProcessor

# Import exact components from boson_multimodal
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent

def debug_collator():
    # Create a simple test sample
    sample = {
        "messages": [
            {
                "role": "user",
                "content": "Hello, how are you?"
            },
            {
                "role": "assistant",
                "content": "I'm doing well, thank you for asking!"
            }
        ]
    }
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")
    
    # Process the sample
    input_tokens, label_tokens, audio_contents, speaker_id = prepare_chatml_sample(sample, tokenizer)
    
    print("Before collator:")
    print("Input tokens:", input_tokens)
    print("Label tokens:", label_tokens)
    
    # Create a ChatMLDatasetSample
    chatml_sample = ChatMLDatasetSample(
        input_ids=torch.tensor(input_tokens, dtype=torch.long),
        label_ids=torch.tensor(label_tokens, dtype=torch.long),
        audio_ids_concat=torch.zeros((8, 0), dtype=torch.long),  # Empty audio tensors
        audio_ids_start=torch.tensor([], dtype=torch.long),
        audio_waveforms_concat=torch.zeros((0,), dtype=torch.float32),
        audio_waveforms_start=torch.tensor([], dtype=torch.long),
        audio_sample_rate=torch.tensor([24000]),
        audio_speaker_indices=torch.tensor([0], dtype=torch.long)
    )
    
    # Load Whisper processor
    whisper_processor = AutoProcessor.from_pretrained("openai/whisper-large-v3", trust_remote_code=True)
    
    # Create collator with mask_audio_out_token_label=False
    collator = HiggsAudioSampleCollator(
        whisper_processor=whisper_processor,
        audio_in_token_id=128015,  # <|AUDIO|>
        audio_out_token_id=128016, # <|AUDIO_OUT|>
        audio_stream_bos_id=128013, # <|audio_bos|>
        audio_stream_eos_id=128014, # <|audio_eos|>
        pad_token_id=128004,       # <|pad|>
        encode_whisper_embed=True,
        return_audio_in_tokens=False,
        use_delay_pattern=False,
        round_to=1,
        audio_num_codebooks=8,
        mask_audio_out_token_label=False,  # This should prevent over-masking
    )
    
    # Process with collator
    batch_input = collator([chatml_sample])
    
    print("\nAfter collator:")
    print("Input IDs shape:", batch_input.input_ids.shape)
    print("Label IDs shape:", batch_input.label_ids.shape)
    print("Input IDs:", batch_input.input_ids[0].tolist())
    print("Label IDs:", batch_input.label_ids[0].tolist())
    
    # Count masked vs unmasked tokens after collator
    label_list = batch_input.label_ids[0].tolist()
    masked_count = label_list.count(-100)
    unmasked_count = len(label_list) - masked_count
    print(f"Masked tokens: {masked_count}, Unmasked tokens: {unmasked_count}")

if __name__ == "__main__":
    debug_collator()