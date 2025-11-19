#!/bin/bash

# PTv3 Inference Setup Script for UniDock-PointCept
# This script helps you set up PTv3 for inference

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "PTv3 Inference Setup for UniDock-PointCept"
echo "=========================================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Step 1: Check if conda is available
print_info "Step 1: Checking conda installation..."
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed or not in PATH"
    exit 1
fi
print_info "Conda found: $(which conda)"

# Step 2: Check/Create environment
print_info "Step 2: Checking Python environment..."
if conda env list | grep -q "^pointcept "; then
    print_info "Environment 'pointcept' already exists"
else
    print_warning "Environment 'pointcept' not found"
    echo "Would you like to create it using environment.yml? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        conda env create -f environment.yml --verbose
        print_info "Environment created successfully"
    else
        print_info "Please create the environment manually and re-run this script"
        exit 0
    fi
fi

# Step 3: Activate environment (instruction)
print_info "Step 3: Please activate the environment:"
echo "    conda activate pointcept"
echo ""

# Step 4: Check if pointops is installed
print_info "Step 4: Checking if pointops is installed..."
if [ -d "libs/pointops/build" ]; then
    print_info "Pointops appears to be installed"
else
    print_warning "Pointops not found. Install it with:"
    echo "    cd libs/pointops && python setup.py install && cd ../.."
fi

# Step 5: Create data directory
print_info "Step 5: Setting up data directory..."
if [ ! -d "data" ]; then
    mkdir -p data
    print_info "Created data directory"
else
    print_info "Data directory already exists"
fi

# Step 6: Create exp directory
print_info "Step 6: Setting up experiment directory..."
if [ ! -d "exp" ]; then
    mkdir -p exp
    print_info "Created exp directory"
else
    print_info "Experiment directory already exists"
fi

# Step 7: Check available datasets
print_info "Step 7: Checking available PTv3 configs..."
echo ""
echo "Available PTv3 configurations:"
echo "  Indoor datasets:"
echo "    - ScanNet:      configs/scannet/semseg-pt-v3m1-0-base.py"
echo "    - ScanNet200:   configs/scannet200/semseg-pt-v3m1-0-base.py"
echo "    - S3DIS:        configs/s3dis/semseg-pt-v3m1-1-rpe.py"
echo "    - Matterport3D: configs/matterport3d/semseg-pt-v3m1-0-base.py"
echo ""
echo "  Outdoor datasets:"
echo "    - nuScenes:     configs/nuscenes/semseg-pt-v3m1-0-base.py"
echo "    - Waymo:        configs/waymo/semseg-pt-v3m1-0-base.py"
echo ""

# Step 8: Instructions for dataset preparation
print_info "Step 8: Dataset preparation"
echo ""
echo "To run inference, you need:"
echo "  1. Download and preprocess a dataset (see PTv3_INFERENCE_GUIDE.md)"
echo "  2. Link it to data/ directory. Example:"
echo "     ln -s /path/to/processed/scannet data/scannet"
echo ""
echo "Or download preprocessed data from:"
echo "  https://huggingface.co/datasets/Pointcept/scannet-compressed"
echo ""

# Step 9: Instructions for model weights
print_info "Step 9: Model weights"
echo ""
echo "Download pre-trained PTv3 weights from:"
echo "  https://huggingface.co/Pointcept/PointTransformerV3"
echo ""
echo "Example for ScanNet:"
echo "  mkdir -p exp/scannet/semseg-pt-v3m1-0-base/model"
echo "  # Download model_best.pth to:"
echo "  # exp/scannet/semseg-pt-v3m1-0-base/model/model_best.pth"
echo ""

# Step 10: Quick inference command
print_info "Step 10: Running inference"
echo ""
echo "Once you have dataset and weights ready, run inference with:"
echo ""
echo "  # Using script (recommended):"
echo "  sh scripts/test.sh -p python -g 4 -d scannet -n semseg-pt-v3m1-0-base -w model_best"
echo ""
echo "  # Or directly:"
echo "  export PYTHONPATH=./"
echo "  python tools/test.py \\"
echo "    --config-file configs/scannet/semseg-pt-v3m1-0-base.py \\"
echo "    --num-gpus 4 \\"
echo "    --options save_path=exp/scannet/semseg-pt-v3m1-0-base \\"
echo "              weight=exp/scannet/semseg-pt-v3m1-0-base/model/model_best.pth"
echo ""

# Summary
echo ""
echo "=========================================="
print_info "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate environment: conda activate pointcept"
echo "  2. Install pointops if needed: cd libs/pointops && python setup.py install"
echo "  3. Prepare your dataset (see PTv3_INFERENCE_GUIDE.md)"
echo "  4. Download pre-trained weights"
echo "  5. Run inference with scripts/test.sh"
echo ""
echo "For detailed instructions, see: PTv3_INFERENCE_GUIDE.md"
echo ""

