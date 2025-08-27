# Higgs Audio Training Instructions

## Prerequisites

1. Ensure you have the required dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure you have access to the training data or create your own dataset.

## Training Data Format

The training data should be in JSON format with the following structure:
```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": "User message text"
      },
      {
        "role": "assistant",
        "content": "Assistant response text"
      }
    ]
  }
]
```

## Running Training

To run the training with 8 GPUs:

```bash
torchrun --nproc_per_node=8 trainer.py \
  --train_manifest path/to/your/train_data.json \
  --output_dir expmt_v1 \
  --base_ckpt bosonai/higgs-audio-v2-generation-3B-base \
  --batch_size 8 \
  --lr 2e-4 \
  --epochs 1 \
  --grad_accum 8 \
  --val_steps 500 \
  --save_steps 250
```

## Performance Optimization

The trainer has been optimized with the following settings:
- Moderate `num_workers` (8 for training, 4 for validation) to balance CPU utilization and overhead
- `persistent_workers=False` to prevent deadlocks in distributed training
- `pin_memory=True` for faster GPU memory transfer

## Troubleshooting

If training is slow or hanging:
1. Check that your training data file exists and is properly formatted
2. Verify that you have sufficient disk space and memory
3. Ensure your GPU drivers and CUDA installation are correct
4. Try reducing `num_workers` if you have a lower-core CPU