#!/bin/bash
#SBATCH --job-name=tts_v1
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128

set -euo pipefail

# Rendezvous config for 2 nodes / 16 H200 GPUs total
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29500
NNODES=$SLURM_NNODES
RDZV_ID=$SLURM_JOB_ID

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=16

# One launcher per node; each spawns 8 local ranks => 16 total
srun --ntasks="$NNODES" --ntasks-per-node=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash -lc '
  eval "$(conda shell.bash hook)"
  conda activate tts_env
  python -m torch.distributed.run \
    --nnodes='"$NNODES"' \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint='"$MASTER_ADDR:$MASTER_PORT"' \
    --rdzv_id='"$RDZV_ID"' \
    trainer.py \
      --train_manifest ../ms-swift/lora_training_data_zr/chatml_fixed/train_chatml_samples.json \
      --output_dir expmt_v1 \
      --base_ckpt bosonai/higgs-audio-v2-generation-3B-base \
      --batch_size 4 \
      --lr 1e-5 \
      --epochs 1 \
      --grad_accum 4 \
      --val_steps 500 \
      --save_steps 200
'
