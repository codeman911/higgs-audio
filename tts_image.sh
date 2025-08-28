#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96

BASE_IMAGE="docker://huggingface/transformers-pytorch-gpu:latest"
ENROOT_CONTAINER="tts_container"
ENROOT_IMAGE="${ENROOT_CONTAINER}.sqsh"

# Check if enroot is installed
if ! command -v enroot &> /dev/null; then
    echo "Error: Enroot is not installed. Please install it before running this script."
    exit 1
fi

# Clean up any existing files/containers
if [ -f "$ENROOT_IMAGE" ]; then
    echo "Removing existing image: $ENROOT_IMAGE"
    rm -f "$ENROOT_IMAGE"
fi

if enroot list | grep -q "$ENROOT_CONTAINER"; then
    echo "Removing existing container: $ENROOT_CONTAINER"
    enroot remove -f "$ENROOT_CONTAINER"
fi

echo "Pulling and converting $BASE_IMAGE..."
enroot import -o ${ENROOT_IMAGE} "$BASE_IMAGE"

echo "Extracting the container..."
enroot create --name $ENROOT_CONTAINER ${ENROOT_IMAGE}

echo "Installing required packages..."
enroot start --root $ENROOT_CONTAINER /bin/bash -c "
    # Update pip
    pip install --upgrade pip &&
    
    # Install requirements from both files
    pip install -r requirements.txt &&
    pip install -r requirements_training.txt &&
    
    # Install additional packages
    pip install wandb &&
    pip install peft &&
    pip install datasets &&
    
    # Install the project in development mode
    pip install -e .
"

echo "Saving the modified container..."
enroot export -f -o tts.sqsh $ENROOT_CONTAINER

echo "Cleaning up..."
enroot remove -f $ENROOT_CONTAINER

echo "Custom TTS container created: tts.sqsh"