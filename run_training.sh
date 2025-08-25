#!/bin/bash
# Higgs-Audio LoRA Training Startup Script
# This script ensures the training pipeline runs from the correct directory
# and provides helpful guidance for common issues.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🎵 Higgs-Audio LoRA Training Startup Script${NC}"
echo "=================================================="

# Get the script directory (should be higgs-audio root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINER_DIR="$SCRIPT_DIR/trainer"

echo -e "${BLUE}📍 Script location: ${NC}$SCRIPT_DIR"
echo -e "${BLUE}📍 Trainer location: ${NC}$TRAINER_DIR"

# Verify we're in the correct directory
if [[ ! -d "$SCRIPT_DIR/boson_multimodal" ]]; then
    echo -e "${RED}❌ Error: boson_multimodal directory not found${NC}"
    echo -e "${YELLOW}💡 This script should be run from the higgs-audio root directory${NC}"
    echo -e "${YELLOW}💡 Make sure you're in the correct repository location${NC}"
    exit 1
fi

if [[ ! -f "$TRAINER_DIR/train.py" ]]; then
    echo -e "${RED}❌ Error: trainer/train.py not found${NC}"
    echo -e "${YELLOW}💡 Make sure the trainer directory is properly set up${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Directory structure validated${NC}"

# Change to the higgs-audio root directory
cd "$SCRIPT_DIR"
echo -e "${BLUE}📂 Working directory: ${NC}$(pwd)"

# Validate Python environment
echo -e "${BLUE}🔍 Validating Python environment...${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 not found${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python3 available: ${NC}$(python3 --version)"

# Check key Python packages
echo -e "${BLUE}🔍 Checking Python dependencies...${NC}"

python3 -c "import torch; print('✅ PyTorch:', torch.__version__)" 2>/dev/null || {
    echo -e "${RED}❌ PyTorch not available${NC}"
    exit 1
}

python3 -c "import transformers; print('✅ Transformers:', transformers.__version__)" 2>/dev/null || {
    echo -e "${RED}❌ Transformers not available${NC}"
    exit 1
}

python3 -c "import peft; print('✅ PEFT:', peft.__version__)" 2>/dev/null || {
    echo -e "${RED}❌ PEFT not available${NC}"
    exit 1
}

# Check if training data path is provided
if [[ $# -eq 0 ]]; then
    echo -e "${YELLOW}⚠️  No arguments provided${NC}"
    echo -e "${BLUE}Usage examples:${NC}"
    echo ""
    echo "  # Basic training:"
    echo "  $0 --train_data path/to/training_data.json"
    echo ""
    echo "  # Advanced training:"
    echo "  $0 --train_data path/to/training_data.json --batch_size 2 --learning_rate 1e-4"
    echo ""
    echo "  # Quick test:"
    echo "  $0 --train_data path/to/training_data.json --quick_test"
    echo ""
    echo "  # Validate data only:"
    echo "  $0 --validate_data_only --train_data path/to/training_data.json"
    echo ""
    exit 0
fi

# Pass all arguments to the training script
echo -e "${BLUE}🚀 Starting training pipeline...${NC}"
echo -e "${BLUE}Command: ${NC}python3 trainer/train.py $*"
echo ""

# Execute the training script with all passed arguments
exec python3 trainer/train.py "$@"