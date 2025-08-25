# Multi-Node Training Commands

## Node 0 (Main Node - IP: 10.0.0.1)
```bash
accelerate launch \
    --multi_gpu \
    --machine_rank 0 \
    --num_machines 2 \
    --main_process_ip 10.0.0.1 \
    --main_process_port 29500 \
    --num_processes 16 \
    --mixed_precision bf16 \
    scripts/training/distributed_trainer.py \
    --dataset_path lora_training_data_zr/chatml/ \
    --output_dir exp/lora_16gpu \
    --num_epochs 1 \
    --batch_size 6 \
    --learning_rate 1e-4 \
    --lora_r 32 \
    --lora_alpha 64 \
    --num_workers 12 \
    --save_steps 500 \
    --warmup_steps 100 \
    --use_cached_codes
```

## Node 1 (Worker Node - IP: 10.0.0.2)  
```bash
accelerate launch \
    --multi_gpu \
    --machine_rank 1 \
    --num_machines 2 \
    --main_process_ip 10.0.0.1 \
    --main_process_port 29500 \
    --num_processes 16 \
    --mixed_precision bf16 \
    scripts/training/distributed_trainer.py \
    --dataset_path lora_training_data_zr/chatml/ \
    --output_dir exp/lora_16gpu \
    --num_epochs 1 \
    --batch_size 6 \
    --learning_rate 1e-4 \
    --lora_r 32 \
    --lora_alpha 64 \
    --num_workers 12 \
    --save_steps 500 \
    --warmup_steps 100 \
    --use_cached_codes
```

## Key Changes for 16 GPU Scaling:
- **Total batch size**: 6 × 16 = 96 samples/step (excellent for stable training)
- **Learning rate**: Keep at 1e-4 (proven to work well)  
- **Num workers**: 12 per GPU (optimal for H200 bandwidth)
- **Save steps**: 500 (less frequent saves due to faster training)
- **Warmup**: 100 steps (good for large batch training)

## Performance Expectations:
- **2× faster training** (16 vs 8 GPUs)
- **Better gradient stability** (larger effective batch size)
- **Linear scaling efficiency** with H200 NVLink
