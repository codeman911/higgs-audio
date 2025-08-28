#!/bin/bash

# Environment variables for optimized Higgs Audio training performance

# NCCL settings for better distributed training
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1

# PyTorch memory management optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=128,expandable_segments=True

# CUDA settings
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Additional optimizations
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU

echo "Environment variables for optimized training have been set."
echo "TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
echo "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=128,expandable_segments=True"
echo "CUDA_DEVICE_MAX_CONNECTIONS=1"