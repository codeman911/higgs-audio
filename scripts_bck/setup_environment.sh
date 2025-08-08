#!/bin/bash
# Environment setup script for Higgs-Audio V2 LoRA training
# Supports both Docker and native installation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_VERSION="3.10"
CUDA_VERSION="12.1"

echo -e "${BLUE}=== Higgs-Audio V2 LoRA Training Environment Setup ===${NC}"
echo "Project Root: $PROJECT_ROOT"
echo "Python Version: $PYTHON_VERSION"
echo "CUDA Version: $CUDA_VERSION"
echo "=================================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check GPU availability
check_gpu() {
    echo -e "${BLUE}Checking GPU availability...${NC}"
    if command_exists nvidia-smi; then
        nvidia-smi
        echo -e "${GREEN}✓ NVIDIA GPUs detected${NC}"
        return 0
    else
        echo -e "${RED}✗ nvidia-smi not found. Please install NVIDIA drivers.${NC}"
        return 1
    fi
}

# Function to setup conda environment
setup_conda_env() {
    echo -e "${BLUE}Setting up Conda environment...${NC}"
    
    if ! command_exists conda; then
        echo -e "${RED}✗ Conda not found. Please install Miniconda or Anaconda.${NC}"
        exit 1
    fi
    
    # Create environment
    ENV_NAME="higgs-audio-lora"
    if conda env list | grep -q "$ENV_NAME"; then
        echo -e "${YELLOW}Environment $ENV_NAME already exists. Removing...${NC}"
        conda env remove -n "$ENV_NAME" -y
    fi
    
    echo "Creating new environment: $ENV_NAME"
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    
    # Activate environment
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    
    echo -e "${GREEN}✓ Conda environment created and activated${NC}"
}

# Function to install PyTorch with CUDA support
install_pytorch() {
    echo -e "${BLUE}Installing PyTorch with CUDA support...${NC}"
    
    # Install PyTorch with CUDA 12.1
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify installation
    python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Number of GPUs: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"
    
    echo -e "${GREEN}✓ PyTorch installed successfully${NC}"
}

# Function to install dependencies
install_dependencies() {
    echo -e "${BLUE}Installing Python dependencies...${NC}"
    
    # Install base requirements
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        pip install -r "$PROJECT_ROOT/requirements.txt"
    fi
    
    # Install LoRA training requirements
    if [ -f "$PROJECT_ROOT/requirements_lora_training.txt" ]; then
        pip install -r "$PROJECT_ROOT/requirements_lora_training.txt"
    fi
    
    # Install Higgs-Audio in development mode
    cd "$PROJECT_ROOT"
    pip install -e .
    
    echo -e "${GREEN}✓ Dependencies installed${NC}"
}

# Function to setup data directories
setup_directories() {
    echo -e "${BLUE}Setting up directories...${NC}"
    
    mkdir -p "$PROJECT_ROOT/data/raw_dataset"
    mkdir -p "$PROJECT_ROOT/data/processed_chatml"
    mkdir -p "$PROJECT_ROOT/outputs"
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/checkpoints"
    mkdir -p "$PROJECT_ROOT/.cache/huggingface"
    
    echo -e "${GREEN}✓ Directories created${NC}"
}

# Function to download models
download_models() {
    echo -e "${BLUE}Downloading pre-trained models...${NC}"
    
    python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer

print('Downloading Higgs-Audio V2 base model...')
try:
    tokenizer = AutoTokenizer.from_pretrained('bosonai/higgs-audio-v2-generation-3B-base')
    model = AutoModelForCausalLM.from_pretrained('bosonai/higgs-audio-v2-generation-3B-base')
    print('✓ Base model downloaded')
except Exception as e:
    print(f'✗ Error downloading base model: {e}')

print('Downloading audio tokenizer...')
try:
    audio_tokenizer = load_higgs_audio_tokenizer('bosonai/higgs-audio-v2-tokenizer')
    print('✓ Audio tokenizer downloaded')
except Exception as e:
    print(f'✗ Error downloading audio tokenizer: {e}')
"
    
    echo -e "${GREEN}✓ Models downloaded${NC}"
}

# Function to test installation
test_installation() {
    echo -e "${BLUE}Testing installation...${NC}"
    
    python -c "
import sys
import torch
import transformers
import peft
import deepspeed
import accelerate
import librosa
import soundfile
import arabic_reshaper
import bidi.algorithm

print('Testing imports...')
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'DeepSpeed: {deepspeed.__version__}')
print(f'Accelerate: {accelerate.__version__}')

# Test CUDA
if torch.cuda.is_available():
    print(f'CUDA: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
    
    # Test tensor operations
    x = torch.randn(10, 10).cuda()
    y = torch.randn(10, 10).cuda()
    z = torch.matmul(x, y)
    print('✓ CUDA tensor operations working')
else:
    print('✗ CUDA not available')

# Test Arabic text processing
arabic_text = 'مرحبا بكم في نظام الذكاء الاصطناعي'
reshaped_text = arabic_reshaper.reshape(arabic_text)
bidi_text = bidi.algorithm.get_display(reshaped_text)
print(f'✓ Arabic text processing: {bidi_text}')

print('✓ All tests passed!')
"
    
    echo -e "${GREEN}✓ Installation test completed${NC}"
}

# Function to setup Docker environment
setup_docker() {
    echo -e "${BLUE}Setting up Docker environment...${NC}"
    
    if ! command_exists docker; then
        echo -e "${RED}✗ Docker not found. Please install Docker.${NC}"
        exit 1
    fi
    
    if ! command_exists docker-compose; then
        echo -e "${RED}✗ Docker Compose not found. Please install Docker Compose.${NC}"
        exit 1
    fi
    
    # Build Docker image
    cd "$PROJECT_ROOT"
    docker-compose -f docker/docker-compose.yml build higgs-audio-training
    
    echo -e "${GREEN}✓ Docker environment setup completed${NC}"
    echo -e "${YELLOW}To start training with Docker:${NC}"
    echo "cd $PROJECT_ROOT"
    echo "docker-compose -f docker/docker-compose.yml up higgs-audio-training"
}

# Function to create sample dataset
create_sample_dataset() {
    echo -e "${BLUE}Creating sample dataset for testing...${NC}"
    
    python -c "
import os
import json
import numpy as np
import soundfile as sf
from pathlib import Path

# Create sample data directory
data_dir = Path('$PROJECT_ROOT/data/raw_dataset')
data_dir.mkdir(parents=True, exist_ok=True)

# Create sample metadata
metadata = {
    'dataset_name': 'Sample Arabic-English Dataset',
    'total_duration_hours': 0.1,
    'languages': ['arabic', 'english', 'mixed'],
    'speakers': ['speaker_001', 'speaker_002'],
    'samples': []
}

# Create sample audio files and metadata
for i in range(10):
    # Generate dummy audio (1 second of sine wave)
    sample_rate = 24000
    duration = 1.0
    frequency = 440 + i * 50
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # Save audio file
    audio_file = data_dir / f'sample_{i:03d}.wav'
    sf.write(audio_file, audio, sample_rate)
    
    # Create metadata entry
    texts = [
        'مرحبا، كيف حالك اليوم؟',  # Arabic
        'Hello, how are you today?',  # English
        'مرحبا Hello, كيف حالك how are you؟'  # Mixed
    ]
    
    sample_metadata = {
        'audio_file': str(audio_file.name),
        'text': texts[i % 3],
        'language': ['arabic', 'english', 'mixed'][i % 3],
        'speaker_id': f'speaker_{(i % 2) + 1:03d}',
        'duration': duration,
        'sample_rate': sample_rate
    }
    
    metadata['samples'].append(sample_metadata)

# Save metadata
with open(data_dir / 'metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f'✓ Sample dataset created in {data_dir}')
print(f'  - {len(metadata[\"samples\"])} samples')
print(f'  - Languages: {metadata[\"languages\"]}')
print(f'  - Speakers: {metadata[\"speakers\"]}')
"
    
    echo -e "${GREEN}✓ Sample dataset created${NC}"
}

# Main installation function
main() {
    echo -e "${BLUE}Choose installation method:${NC}"
    echo "1) Native installation (Conda)"
    echo "2) Docker installation"
    echo "3) Test existing installation"
    echo "4) Create sample dataset only"
    read -p "Enter choice (1-4): " choice
    
    case $choice in
        1)
            echo -e "${BLUE}Starting native installation...${NC}"
            check_gpu
            setup_conda_env
            install_pytorch
            install_dependencies
            setup_directories
            download_models
            create_sample_dataset
            test_installation
            
            echo -e "${GREEN}=== Installation completed successfully! ===${NC}"
            echo -e "${YELLOW}To activate the environment:${NC}"
            echo "conda activate higgs-audio-lora"
            echo ""
            echo -e "${YELLOW}To start training:${NC}"
            echo "cd $PROJECT_ROOT"
            echo "./scripts/launch_training.sh"
            ;;
        2)
            echo -e "${BLUE}Starting Docker installation...${NC}"
            check_gpu
            setup_directories
            create_sample_dataset
            setup_docker
            ;;
        3)
            echo -e "${BLUE}Testing existing installation...${NC}"
            test_installation
            ;;
        4)
            echo -e "${BLUE}Creating sample dataset...${NC}"
            setup_directories
            create_sample_dataset
            ;;
        *)
            echo -e "${RED}Invalid choice. Exiting.${NC}"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
