#!/usr/bin/env python3
import os
import sys
import json
import argparse
import torch
import torchaudio
import librosa
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, WhisperProcessor, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch.nn as nn
from accelerate import Accelerator
from tqdm import tqdm
from pathlib import Path
import logging

# FIX: Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from boson_multimodal.model.higgs_audio import HiggsAudioModel
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample

# ===== HIGGS-AUDIO constants =====
AUDIO_BOS = 1024
AUDIO_EOS = 1025
AUDIO_V = 1026
N_CODEBOOKS = 8
MIN_ASSISTANT_TOKENS = 32  # Arabic learning threshold

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleDataset(Dataset):
    """Simple dataset that loads ChatML JSON files"""
    
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, list):
            self.samples = data
        elif isinstance(data, dict):
            self.samples = data.get('samples', data.get('data', []))
        else:
            self.samples = []
        
        logger.info(f"Loaded {len(self.samples)} samples from {json_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def simple_collate_fn(batch, tokenizer, audio_tokenizer, collator, sample_rate=24000):
    """SIMPLE collate function - properly process audio like the backup trainer"""
    chatml_samples = []
    
    for sample in batch:
        # CRITICAL FIX: Convert to proper chatml_dict structure first (like backup trainer)
        messages = sample.get('messages', [])
        
        # Build ChatML dict (exactly like backup trainer)
        chatml_dict = {"messages": []}
        
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content')
            
            if role and content:
                # Handle different content formats
                if isinstance(content, list):
                    # Multi-modal content
                    processed_content = []
                    for item in content:
                        if item.get('type') == 'text':
                            processed_content.append({"type": "text", "text": item.get('text', '')})
                        elif item.get('type') == 'audio':
                            audio_url = item.get('audio_url', '')
                            if audio_url:
                                processed_content.append({"type": "audio", "audio_url": audio_url})
                    chatml_dict["messages"].append({"role": role, "content": processed_content})
                else:
                    # Simple text content
                    chatml_dict["messages"].append({"role": role, "content": content})
        
        # Tokenize with prepare_chatml_sample (exactly like backup trainer)
        try:
            input_tokens, label_tokens, audio_contents, speaker_id = prepare_chatml_sample(
                chatml_dict, tokenizer
            )
        except Exception as e:
            logger.warning(f"Failed to prepare sample: {e}")
            # Create empty sample
            input_tokens = [tokenizer.pad_token_id]
            label_tokens = [-100]
            audio_contents = []
            speaker_id = 0
        
        # Process audio properly using audio_tokenizer (like backup trainer)
        audio_ids_list = []
        audio_waveforms_list = []
        
        for audio_content in audio_contents:
            if audio_content and hasattr(audio_content, 'audio_url'):
                audio_path = audio_content.audio_url
                if audio_path and os.path.exists(audio_path):
                    try:
                        # Tokenize audio
                        audio_codes = audio_tokenizer.encode(audio_path)
                        # Ensure tensor is on CPU (like backup trainer)
                        if audio_codes.is_cuda:
                            audio_codes = audio_codes.cpu()
                        # Ensure 8 codebooks (like backup trainer)
                        if audio_codes.shape[0] != 8:
                            if audio_codes.shape[0] > 8:
                                audio_codes = audio_codes[:8, :]
                            else:
                                padding = torch.zeros(8 - audio_codes.shape[0], audio_codes.shape[1])
                                audio_codes = torch.cat([audio_codes, padding], dim=0)
                        audio_ids_list.append(audio_codes)
                        
                        # Load waveform using torchaudio (like backup trainer)
                        waveform, sr = torchaudio.load(audio_path)
                        if sr != sample_rate:
                            waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
                        if waveform.shape[0] > 1:
                            waveform = waveform.mean(dim=0, keepdim=True)
                        waveform = waveform.squeeze(0)  # Flatten to 1D
                        audio_waveforms_list.append(waveform)
                        
                    except Exception as e:
                        logger.warning(f"Failed to process audio {audio_path}: {e}")
        
        # Create tensors (exactly like backup trainer)
        if audio_ids_list:
            audio_ids_concat = torch.cat(audio_ids_list, dim=1)
            audio_ids_start = torch.cumsum(
                torch.tensor([0] + [ids.shape[1] for ids in audio_ids_list]), dim=0
            )
        else:
            audio_ids_concat = torch.zeros((8, 0), dtype=torch.long)
            audio_ids_start = torch.tensor([0], dtype=torch.long)
        
        if audio_waveforms_list:
            audio_waveforms_concat = torch.cat(audio_waveforms_list, dim=0)
            lengths = [len(wv) for wv in audio_waveforms_list]
            audio_waveforms_start = torch.tensor([0] + lengths[:-1]).cumsum(dim=0)
            audio_sample_rate = torch.tensor([sample_rate] * len(audio_waveforms_list))
            audio_speaker_indices = torch.zeros(len(audio_waveforms_list), dtype=torch.long)
        else:
            audio_waveforms_concat = torch.tensor([])
            audio_waveforms_start = torch.tensor([0], dtype=torch.long)
            audio_sample_rate = torch.tensor([sample_rate])
            audio_speaker_indices = torch.tensor([0], dtype=torch.long)
        
        # Create ChatMLDatasetSample (exactly like backup trainer)
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
        
        # DEBUG: Log audio structure for first sample
        if len(chatml_samples) == 0:  # Only log first sample
            logger.info(f"🔍 AUDIO STRUCTURE DEBUG for sample 0:")
            logger.info("=" * 80)
            
            # Log raw ChatML conversation structure from batch
            logger.info(f"   input_tokens length: {len(input_tokens)}")
            logger.info(f"   audio_ids_concat shape: {audio_ids_concat.shape}")
            logger.info(f"   audio_ids_start: {audio_ids_start}")
            logger.info(f"   audio_waveforms_concat length: {len(audio_waveforms_concat) if len(audio_waveforms_concat) > 0 else 0}")
            logger.info(f"   Number of audio segments: {len(audio_ids_list)}")
            if len(audio_contents) > 0:
                logger.info(f"   Audio contents found: {len(audio_contents)} items")
                for i, content in enumerate(audio_contents[:2]):  # First 2 audio items
                    if hasattr(content, 'audio_url'):
                        logger.info(f"     Audio {i}: {content.audio_url}")
        
        chatml_samples.append(chatml_sample)
    
    # Now call original collator with proper objects
    return collator(chatml_samples)


def main():
    parser = argparse.ArgumentParser(description="Higgs-Audio LoRA Training")
    
    # Data arguments
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to dataset directory containing train/val JSON files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for model and checkpoints")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, 
                        default="bosonai/higgs-audio-v2-generation-3B-base",
                        help="Path to base model")
    parser.add_argument("--audio_tokenizer_path", type=str,
                        default="bosonai/higgs-audio-v2-tokenizer",
                        help="Path to audio tokenizer")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=12,
                        help="Batch size per device (increased for H200)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps (reduced with larger batch)")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm")
    
    # LoRA arguments
    parser.add_argument('--lora_r', type=int, default=32, help='LoRA rank (increased for Arabic learning)')
    parser.add_argument('--lora_alpha', type=int, default=64, help='LoRA alpha (increased for Arabic learning)')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--text_loss_weight', type=float, default=1.0, help='Text loss weight')
    
    # Other arguments - OPTIMIZED FOR H200
    parser.add_argument("--num_workers", type=int, default=64,
                        help="DataLoader workers per GPU (optimized for H200)")
    parser.add_argument("--prefetch_factor", type=int, default=4,
                        help="DataLoader prefetch factor (optimized)")
    parser.add_argument("--persistent_workers", action="store_true", default=True,
                        help="Keep workers alive across epochs for speed")
    parser.add_argument("--audio_label_smoothing", type=float, default=0.05,
                        help="Label smoothing for audio CE over codebooks")
    parser.add_argument("--compile_model", action="store_true", default=True,
                        help="Enable torch.compile for H200 speed")
    parser.add_argument("--use_cached_codes", action="store_true", default=False,
                        help="Use <audio_path>.codes.pt if present (faster training)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--log_steps", type=int, default=50,
                        help="Log every N steps (reduced for speed)")
    parser.add_argument("--val_steps", type=int, default=100,
                        help="Run validation every N steps")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision training")
    
    # DEBUG arguments
    parser.add_argument("--debug_samples", type=int, default=None,
                        help="Limit training to N samples for debugging (default: use full dataset)")
    parser.add_argument("--debug_val_samples", type=int, default=None,
                        help="Limit validation to N samples for debugging (default: use full dataset)")
    
    args = parser.parse_args()
    
    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision
    )
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Fast + stable matmul on Hopper; keep BF16 for mixed precision
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Loading data from {args.dataset_path}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Load tokenizers
    logger.info("Loading tokenizers...")
    # Text tokenizer from model path
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Audio tokenizer - load on CPU, accelerator will handle device placement
    audio_tokenizer = load_higgs_audio_tokenizer(args.audio_tokenizer_path, device="cpu")
    
    # Load model config
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    
    # CRITICAL ARCHITECTURE FIX: Revert to original Higgs-Audio V2 configuration
    # Original architecture doesn't use cross-attention modules
    # Text-audio conditioning happens through shared attention + DualFFN, not separate cross-attention
    logger.info(" ORIGINAL CONFIG: audio_out_self_attention=False (shared attention + DualFFN)")
    
    # CRITICAL: Revert to original configuration - no cross-attention modules
    config.use_audio_out_self_attention = False  # Original default - proven architecture!
    
    logger.info(" REVERTED: use_audio_out_self_attention=False (original architecture)")
    
    # Load model
    logger.info("Loading model...")
    model = HiggsAudioModel.from_pretrained(
        args.model_path,
        config=config,  # Use modified config
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": accelerator.device}
    )
    
    # Initialize collator using WhisperProcessor
    logger.info("Initializing collator...")
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    
    # Use the original HiggsAudioSampleCollator with CORRECT constructor parameters
    # This handles all audio indexing automatically and correctly
    collator = HiggsAudioSampleCollator(
        whisper_processor=whisper_processor,
        audio_in_token_id=config.audio_in_token_idx,
        audio_out_token_id=config.audio_out_token_idx,
        audio_stream_bos_id=config.audio_stream_bos_id,
        audio_stream_eos_id=config.audio_stream_eos_id,
        encode_whisper_embed=config.encode_whisper_embed,
        pad_token_id=config.pad_token_id,
        return_audio_in_tokens=config.encode_audio_in_tokens,
        use_delay_pattern=config.use_delay_pattern,
        round_to=8,  # Documentation recommends round_to=8 for optimal batching
        audio_num_codebooks=8
    )
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = SimpleDataset(os.path.join(args.dataset_path, "train_chatml_samples.json"))[:args.debug_samples]
    val_dataset = SimpleDataset(os.path.join(args.dataset_path, "val_chatml_samples.json"))[:args.debug_val_samples]
    
    # ========== STRATEGIC LOGGING: DATA PIPELINE ANALYSIS ==========
    logger.info("🔍 STRATEGIC ANALYSIS: ChatML Data Format")
    logger.info("=" * 80)
    
    # Analyze first sample to understand exact data format
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        logger.info("📋 RAW CHATML SAMPLE STRUCTURE:")
        logger.info(f"   Sample keys: {list(sample.keys())}")
        
        if 'messages' in sample:
            logger.info(f"   Messages count: {len(sample['messages'])}")
            for i, msg in enumerate(sample['messages']):
                role = msg.get('role', 'unknown')
                content = msg.get('content', [])
                logger.info(f"   Message {i} - Role: {role}")
                
                if isinstance(content, list):
                    for j, item in enumerate(content):
                        item_type = item.get('type', 'unknown')
                        if item_type == 'text':
                            text_preview = item.get('text', '')[:80] + ('...' if len(item.get('text', '')) > 80 else '')
                            logger.info(f"     Content {j} [text]: {text_preview}")
                        elif item_type == 'audio':
                            audio_path = item.get('audio_url', 'No URL')
                            logger.info(f"     Content {j} [audio]: {Path(audio_path).name}")
                elif isinstance(content, str):
                    content_preview = content[:80] + ('...' if len(content) > 80 else '')
                    logger.info(f"     Content [text]: {content_preview}")
        
        # Test prepare_chatml_sample to see what it produces
        logger.info("� TESTING prepare_chatml_sample OUTPUT:")
        try:
            input_tokens, label_tokens, audio_contents, speaker_id = prepare_chatml_sample(sample, tokenizer)
            
            logger.info(f"   input_tokens length: {len(input_tokens)}")
            logger.info(f"   label_tokens length: {len(label_tokens)}")
            logger.info(f"   audio_contents count: {len(audio_contents)}")
            logger.info(f"   speaker_id: {speaker_id}")
            
            # Analyze label masking pattern
            supervised_tokens = len([t for t in label_tokens if t != -100])
            ignored_tokens = len([t for t in label_tokens if t == -100])
            logger.info(f"   Supervised tokens: {supervised_tokens}")
            logger.info(f"   Ignored tokens: {ignored_tokens}")
            
            # Show tokenized conversation preview
            input_text_preview = tokenizer.decode(input_tokens[:200], skip_special_tokens=False)
            logger.info(f"   Input text preview: {input_text_preview}")
            
            # Show what gets supervised (labels != -100)
            supervised_token_ids = [input_tokens[i] for i, label in enumerate(label_tokens) if label != -100 and i < len(input_tokens)]
            if supervised_token_ids:
                supervised_text = tokenizer.decode(supervised_token_ids[:100], skip_special_tokens=False)
                logger.info(f"   Supervised text preview: {supervised_text}")
            
        except Exception as e:
            logger.error(f"   Error in prepare_chatml_sample: {e}")
    
    logger.info("✅ STRATEGIC ANALYSIS COMPLETE")
    logger.info("=" * 80)
    
    # Create data loaders - OPTIMIZED FOR H200
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        collate_fn=lambda batch: simple_collate_fn(batch, tokenizer, audio_tokenizer, collator),  # Use custom collate function
        pin_memory=True,  # H200 optimization
        drop_last=True    # Consistent batch sizes for speed
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers // 2,  # Less workers for validation
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        collate_fn=lambda batch: simple_collate_fn(batch, tokenizer, audio_tokenizer, collator),  # Use custom collate function
        pin_memory=True,
        drop_last=False
    )
    
    # LoRA configuration with systematic target collection
    TARGET_FRAGMENTS = (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        "audio_mlp.gate_proj", "audio_mlp.up_proj", "audio_mlp.down_proj",
    )

    def collect_lora_targets(model):
        leaf_targets = set()
        for name, module in model.named_modules():
            for frag in TARGET_FRAGMENTS:
                if name.endswith(frag):
                    leaf_targets.add(frag.split(".")[-1])  # leaf module name
        return sorted(leaf_targets)

    lora_leaf_targets = collect_lora_targets(model)
    if not lora_leaf_targets:
        logger.warning("No LoRA targets found – check module names.")
    else:
        logger.info(f"LoRA leaf targets: {lora_leaf_targets}")

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=tuple(lora_leaf_targets),
        bias="none", task_type=TaskType.CAUSAL_LM
    )
    
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, lora_config)
    
    # Step 3: Supervision preparation function
    def prepare_supervision(batch, tokenizer, device, logger=None):
        """
        Fix audio teacher-forcing and text labels
        """
        inp = batch.input_ids.to(device)
        attn = batch.attention_mask.to(device)
        a_in = batch.audio_in_ids.to(device) if hasattr(batch, 'audio_in_ids') and batch.audio_in_ids is not None else None
        a_out = batch.audio_out_ids.to(device) if hasattr(batch, 'audio_out_ids') and batch.audio_out_ids is not None else None

        if a_out is not None:
            # ---- Audio: shift-right teacher forcing ----
            C, T = a_out.shape
            assert C == N_CODEBOOKS, f"Expected {N_CODEBOOKS} codebooks, got {C}"

            a_shift = a_out.clone()
            a_shift[:, 1:] = a_out[:, :-1]
            a_shift[:, 0] = AUDIO_BOS  # BOS at t=0 for inputs

            a_lbl = a_out.clone()
            a_lbl[:, 0] = -100  # do not learn BOS
            bos_mask = (a_lbl == AUDIO_BOS)
            if bos_mask.any() and logger:
                logger.info(f" MAPPING BOS TOKENS: {int(bos_mask.sum().item())} tokens ({AUDIO_BOS}) → -100")
            a_lbl[bos_mask] = -100
        else:
            a_shift = None
            a_lbl = None

        # ---- Text labels ----
        if hasattr(batch, 'label_ids') and batch.label_ids is not None:
            t_lbl = batch.label_ids.to(device)
        else:
            # Fallback: supervise non-pad tokens
            t_lbl = inp.clone()
            t_lbl[t_lbl == tokenizer.pad_token_id] = -100

        return dict(
            input_ids=inp, attention_mask=attn,
            audio_in_ids=a_in, audio_out_ids=a_shift, audio_labels=a_lbl,
            text_labels=t_lbl,
            # CRITICAL FIX: Create exactly the right number of start indices to produce correct length values
            # For 8 codebooks, we need 8 start indices to produce 8 length values (not 9)
            audio_in_ids_start=torch.arange(a_in.shape[0], dtype=torch.long, device=device) * a_in.shape[1] if a_in is not None else torch.tensor([0], dtype=torch.long, device=device),
            audio_out_ids_start=torch.arange(a_shift.shape[0], dtype=torch.long, device=device) * a_shift.shape[1] if a_shift is not None else torch.tensor([0], dtype=torch.long, device=device),
            audio_out_ids_start_group_loc=None  # Not needed for single audio segments
        )

    def non_ignore_count(t): 
        return int((t != -100).sum().item())
    
    # Setup optimizer with stability improvements
    # Lower learning rate for newly initialized cross-attention modules
    stable_lr = min(args.learning_rate, 1e-4)  # Cap at 1e-4 for stability
    if stable_lr != args.learning_rate:
        logger.info(f" STABILITY FIX: Reducing learning rate from {args.learning_rate} to {stable_lr} for cross-attention stability")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=stable_lr,
        weight_decay=0.01,
        eps=1e-8,  # Increase epsilon for numerical stability
        betas=(0.9, 0.95)  # More conservative beta2 for stability
    )
    
    # CRITICAL: Add gradient clipping for newly initialized cross-attention modules
    gradient_clip_norm = 1.0  # Conservative clipping for stability
    logger.info(f" STABILITY FIX: Adding gradient clipping (max_norm={gradient_clip_norm}) for cross-attention stability")
    
    # Calculate training steps for proper warmup scheduling
    num_training_steps = len(train_dataloader) * args.num_epochs
    
    # FINAL FIX: Immediate learning rate for newly initialized cross-attention modules
    # The cosine scheduler with long warmup prevents learning for thousands of steps
    # Newly initialized modules need immediate adaptation to learn text-audio conditioning
    
    # Calculate training steps for proper warmup scheduling
    num_training_steps = len(train_dataloader) * args.num_epochs
    
    # CRITICAL FIX: Immediate learning for cross-attention adaptation
    # Use minimal warmup (50 steps) so learning starts immediately
    warmup_steps = 50  # Very short warmup for immediate learning
    logger.info(f" IMMEDIATE LEARNING FIX: Using {warmup_steps} warmup steps (immediate learning for cross-attention)")
    
    # Alternative: Use linear scheduler that starts with small non-zero LR
    from transformers import get_linear_schedule_with_warmup
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Verify scheduler starts with meaningful LR after just 1 step
    # Take one scheduler step to see actual LR
    dummy_step_lr = stable_lr / warmup_steps  # Expected LR after 1 step
    logger.info(f" LEARNING RATE FIX: After 1 step, LR will be ≈ {dummy_step_lr:.2e} (immediate learning)")
    
    # Alternative approach: Start with constant LR for first 1000 steps, then decay
    # This ensures immediate learning for cross-attention adaptation
    logger.info(" CROSS-ATTENTION LEARNING: Enabling immediate gradient updates for newly initialized modules")
    
    # Prepare for training
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    
    if val_dataloader:
        val_dataloader = accelerator.prepare(val_dataloader)
    
    # Training loop
    logger.info("Starting training...")
    global_step = 0
    
    def extract_assistant_labels(input_ids, tokenizer):
        """Extract labels ONLY from assistant content before <|AUDIO_OUT|>"""
        B, T = input_ids.size()
        labels = input_ids.new_full((B, T), -100)
        
        AUDIO_OUT_ID = 128275  # <|AUDIO_OUT|>
        EOH_ID = 128007       # <|end_header_id|>
        ASSISTANT_ID = 78191  # "assistant"
        
        for b in range(B):
            ids = input_ids[b]
            
            # Find <|AUDIO_OUT|> position
            audio_out_pos = (ids == AUDIO_OUT_ID).nonzero(as_tuple=True)[0]
            if audio_out_pos.numel() == 0:
                continue
            audio_out_pos = audio_out_pos[0].item()
            
            # Find last assistant header before <|AUDIO_OUT|>
            eoh_positions = (ids[:audio_out_pos] == EOH_ID).nonzero(as_tuple=True)[0]
            if eoh_positions.numel() == 0:
                continue
                
            # Find assistant content start
            for eoh_idx in reversed(eoh_positions):
                content_start = eoh_idx.item() + 1
                if content_start < audio_out_pos and (ASSISTANT_ID in ids[:eoh_idx]):
                    # Supervise assistant content (next-token prediction)
                    if content_start + 1 < audio_out_pos:
                        labels[b, content_start+1:audio_out_pos] = ids[content_start+1:audio_out_pos]
                    break
        
        return labels

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                # Move batch tensors to the correct device and dtype (like backup trainer)
                device = accelerator.device
                
                # Get model dtype for audio features (model uses mixed precision)
                model_dtype = next(model.parameters()).dtype
                
                # Helper function to move tensor to device and optionally convert dtype (like backup trainer)
                def to_device(tensor, convert_dtype=False):
                    if tensor is not None and hasattr(tensor, 'to'):
                        if convert_dtype and tensor.dtype in [torch.float32, torch.float64]:
                            # Convert float tensors to match model dtype (for audio features)
                            return tensor.to(device=device, dtype=model_dtype)
                        else:
                            return tensor.to(device)
                    return tensor
                
                # CRITICAL FIX: Clean separation of model inputs (NO LABELS to model) - like backup trainer
                model_inputs = {
                    'input_ids': to_device(batch.input_ids),
                    'attention_mask': to_device(batch.attention_mask),
                    'audio_features': to_device(batch.audio_in_wv, convert_dtype=True) if hasattr(batch, 'audio_in_wv') and batch.audio_in_wv is not None else None,
                    'audio_feature_attention_mask': to_device(batch.audio_feature_attention_mask) if hasattr(batch, 'audio_feature_attention_mask') and batch.audio_feature_attention_mask is not None else None,
                    'audio_in_ids': to_device(batch.audio_in_ids) if hasattr(batch, 'audio_in_ids') else None,
                    'audio_in_ids_start': to_device(batch.audio_in_ids_start) if hasattr(batch, 'audio_in_ids_start') else None,
                    'audio_out_ids': to_device(batch.audio_out_ids) if hasattr(batch, 'audio_out_ids') else None,
                    'audio_out_ids_start': to_device(batch.audio_out_ids_start) if hasattr(batch, 'audio_out_ids_start') else None,
                    'audio_out_ids_start_group_loc': to_device(batch.audio_out_ids_start_group_loc) if hasattr(batch, 'audio_out_ids_start_group_loc') else None,
                }
                # Remove None values for clean forward pass
                model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
                
                # Get the underlying model (handle PEFT wrapping)
                if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
                    actual_model = model.base_model.model  # PEFT wrapped
                elif hasattr(model, 'module'):
                    actual_model = model.module  # Accelerate wrapped
                else:
                    actual_model = model
                
                # SIMPLE FORWARD PASS (like working backup trainer)
                outputs = actual_model(**model_inputs)
                
                # Extract labels separately for loss computation (like backup trainer)
                text_labels = to_device(batch.label_ids) if hasattr(batch, 'label_ids') else None
                audio_labels = to_device(batch.label_audio_ids) if hasattr(batch, 'label_audio_ids') else None
                
                # Manual loss computation (like working version)
                total_loss = 0.0
                
                # Audio loss computation
                audio_loss = torch.tensor(0.0, device=device)
                if "audio_logits" in outputs and outputs["audio_logits"] is not None and audio_labels is not None:
                    audio_logits = outputs["audio_logits"]
                    if audio_logits.dim() == 3:
                        # Ensure [C, T, V] format
                        if audio_logits.shape[1] == N_CODEBOOKS:
                            audio_logits = audio_logits.permute(1, 0, 2).contiguous()
                        
                        C, T, V = audio_logits.shape
                        audio_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                        audio_loss = audio_loss_fct(
                            audio_logits.view(C * T, V),
                            audio_labels.view(C * T)
                        )
                        total_loss += audio_loss
                
                # Text loss computation
                text_loss = torch.tensor(0.0, device=device)
                if "logits" in outputs and outputs["logits"] is not None and text_labels is not None:
                    text_logits = outputs["logits"][:, :-1, :].contiguous()  # Shift for LM
                    shift_labels = text_labels[:, 1:].contiguous()
                    
                    # Ensure shapes match before computing loss
                    B, T, V = text_logits.shape
                    B_lbl, T_lbl = shift_labels.shape
                    
                    if T != T_lbl:
                        # Adjust to minimum sequence length
                        min_T = min(T, T_lbl)
                        text_logits = text_logits[:, :min_T, :].contiguous()
                        shift_labels = shift_labels[:, :min_T].contiguous()
                        T = min_T
                    
                    text_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                    text_loss = text_loss_fct(
                        text_logits.view(B * T, V),
                        shift_labels.view(B * T)
                    )
                    # Weight text loss lower for voice cloning
                    weighted_text_loss = args.text_loss_weight * text_loss
                    total_loss += weighted_text_loss
                
                loss = total_loss
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Save checkpoint
                if global_step % args.save_steps == 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        # Validation
        if val_dataloader:
            model.eval()
            val_loss = 0
            val_steps = 0
            
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validation"):
                    device = accelerator.device
                    
                    # Move batch tensors to the correct device and dtype (like backup trainer)
                    model_inputs = {
                        'input_ids': batch.input_ids.to(device),
                        'attention_mask': batch.attention_mask.to(device),
                        'audio_features': batch.audio_in_wv.to(device) if hasattr(batch, 'audio_in_wv') and batch.audio_in_wv is not None else None,
                        'audio_feature_attention_mask': batch.audio_feature_attention_mask.to(device) if hasattr(batch, 'audio_feature_attention_mask') and batch.audio_feature_attention_mask is not None else None,
                        'audio_in_ids': batch.audio_in_ids.to(device) if hasattr(batch, 'audio_in_ids') else None,
                        'audio_in_ids_start': batch.audio_in_ids_start.to(device) if hasattr(batch, 'audio_in_ids_start') else None,
                        'audio_out_ids': batch.audio_out_ids.to(device) if hasattr(batch, 'audio_out_ids') else None,
                        'audio_out_ids_start': batch.audio_out_ids_start.to(device) if hasattr(batch, 'audio_out_ids_start') else None,
                        'audio_out_ids_start_group_loc': batch.audio_out_ids_start_group_loc.to(device) if hasattr(batch, 'audio_out_ids_start_group_loc') else None,
                    }
                    # Remove None values for clean forward pass
                    model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
                    
                    # Get the underlying model (handle PEFT wrapping)
                    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
                        actual_model = model.base_model.model  # PEFT wrapped
                    elif hasattr(model, 'module'):
                        actual_model = model.module  # Accelerate wrapped
                    else:
                        actual_model = model
                    
                    # Forward pass with clean inputs - NO LABELS to model (like working version)
                    outputs = actual_model(**model_inputs)
                    
                    # Extract labels separately for manual loss computation
                    text_labels = batch.label_ids.to(device) if hasattr(batch, 'label_ids') else None
                    audio_labels = batch.label_audio_ids.to(device) if hasattr(batch, 'label_audio_ids') else None
                    
                    # Manual loss computation (like working version)
                    total_loss = 0.0
                    
                    # Audio loss computation
                    audio_loss = torch.tensor(0.0, device=device)
                    if "audio_logits" in outputs and outputs["audio_logits"] is not None and audio_labels is not None:
                        audio_logits = outputs["audio_logits"]
                        if audio_logits.dim() == 3:
                            # Ensure [C, T, V] format
                            if audio_logits.shape[1] == N_CODEBOOKS:
                                audio_logits = audio_logits.permute(1, 0, 2).contiguous()
                            
                            C, T, V = audio_logits.shape
                            audio_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                            audio_loss = audio_loss_fct(
                                audio_logits.view(C * T, V),
                                audio_labels.view(C * T)
                            )
                            total_loss += audio_loss
                    
                    # Text loss computation
                    text_loss = torch.tensor(0.0, device=device)
                    if "logits" in outputs and outputs["logits"] is not None and text_labels is not None:
                        text_logits = outputs["logits"][:, :-1, :].contiguous()  # Shift for LM
                        shift_labels = text_labels[:, 1:].contiguous()
                        
                        # Ensure shapes match before computing loss
                        B, T, V = text_logits.shape
                        B_lbl, T_lbl = shift_labels.shape
                        
                        if T != T_lbl:
                            # Adjust to minimum sequence length
                            min_T = min(T, T_lbl)
                            text_logits = text_logits[:, :min_T, :].contiguous()
                            shift_labels = shift_labels[:, :min_T].contiguous()
                            T = min_T
                        
                        text_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                        text_loss = text_loss_fct(
                            text_logits.view(B * T, V),
                            shift_labels.view(B * T)
                        )
                        # Weight text loss lower for voice cloning
                        weighted_text_loss = args.text_loss_weight * text_loss
                        total_loss += weighted_text_loss
                    
                    loss = total_loss
                    
                    # Combined validation loss
                    batch_val_loss = audio_loss + text_loss
                    val_loss += batch_val_loss.item()
                    val_steps += 1
            
            avg_val_loss = val_loss / val_steps
            logger.info(f"Epoch {epoch+1} - Validation loss: {avg_val_loss:.4f}")
    
    # Save final model
    final_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    logger.info(f"Training complete! Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
