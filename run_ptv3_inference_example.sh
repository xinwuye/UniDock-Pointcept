#!/bin/bash

# Example script for running PTv3 inference
# This demonstrates how to run inference on different datasets

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}=========================================="
    echo -e "$1"
    echo -e "==========================================${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if PYTHONPATH is set
if [ -z "$PYTHONPATH" ] || [ "$PYTHONPATH" != "./" ]; then
    print_info "Setting PYTHONPATH=./"
    export PYTHONPATH=./
fi

# Check conda environment
print_info "Checking Python environment..."
if ! python -c "import torch; import spconv" 2>/dev/null; then
    print_error "Required packages not found. Please activate the correct environment:"
    echo "    conda activate pointcept"
    exit 1
fi

print_header "PTv3 Inference Examples"

# Function to run inference
run_inference() {
    local DATASET=$1
    local CONFIG=$2
    local EXP_NAME=$3
    local NUM_GPU=${4:-4}
    local WEIGHT=${5:-model_best}
    
    print_info "Running inference for $DATASET"
    echo "  Config: $CONFIG"
    echo "  Experiment: $EXP_NAME"
    echo "  GPUs: $NUM_GPU"
    echo "  Weight: $WEIGHT"
    echo ""
    
    # Check if data exists
    if [ ! -d "data/$DATASET" ]; then
        print_warning "Dataset not found: data/$DATASET"
        echo "  Please prepare the dataset first (see PTv3_INFERENCE_GUIDE.md)"
        echo ""
        return
    fi
    
    # Check if model weight exists
    WEIGHT_PATH="exp/$DATASET/$EXP_NAME/model/${WEIGHT}.pth"
    if [ ! -f "$WEIGHT_PATH" ]; then
        print_warning "Model weight not found: $WEIGHT_PATH"
        echo "  Please download pre-trained weights from:"
        echo "  https://huggingface.co/Pointcept/PointTransformerV3"
        echo ""
        return
    fi
    
    # Run inference
    print_info "Starting inference..."
    sh scripts/test.sh -p python -g $NUM_GPU -d $DATASET -n $EXP_NAME -w $WEIGHT
    
    print_info "Inference complete! Results saved to: exp/$DATASET/$EXP_NAME/result/"
    echo ""
}

# Display menu
echo ""
echo "Available PTv3 inference examples:"
echo ""
echo "1. ScanNet (Indoor) - 20 classes"
echo "2. ScanNet200 (Indoor) - 200 classes"
echo "3. S3DIS Area5 (Indoor) - 13 classes"
echo "4. nuScenes (Outdoor) - 16 classes"
echo "5. Custom (Manual input)"
echo ""
echo "0. Show all available configs"
echo ""

read -p "Select an option (1-5, or 0 for help): " choice

case $choice in
    1)
        print_header "PTv3 Inference on ScanNet"
        run_inference "scannet" "semseg-pt-v3m1-0-base" "semseg-pt-v3m1-0-base" 4
        ;;
    2)
        print_header "PTv3 Inference on ScanNet200"
        run_inference "scannet200" "semseg-pt-v3m1-0-base" "semseg-pt-v3m1-0-base" 4
        ;;
    3)
        print_header "PTv3 Inference on S3DIS Area5"
        echo "Note: S3DIS uses RPE (Relative Position Encoding)"
        run_inference "s3dis" "semseg-pt-v3m1-1-rpe" "semseg-pt-v3m1-1-rpe" 4
        ;;
    4)
        print_header "PTv3 Inference on nuScenes"
        run_inference "nuscenes" "semseg-pt-v3m1-0-base" "semseg-pt-v3m1-0-base" 4
        ;;
    5)
        print_header "Custom PTv3 Inference"
        echo ""
        read -p "Dataset name (e.g., scannet): " dataset
        read -p "Config name (without .py): " config
        read -p "Experiment name: " exp_name
        read -p "Number of GPUs [4]: " num_gpu
        num_gpu=${num_gpu:-4}
        read -p "Weight name [model_best]: " weight
        weight=${weight:-model_best}
        
        run_inference "$dataset" "$config" "$exp_name" "$num_gpu" "$weight"
        ;;
    0)
        print_header "Available PTv3 Configurations"
        echo ""
        echo "Indoor Datasets:"
        echo "  ScanNet (20 classes):"
        echo "    - configs/scannet/semseg-pt-v3m1-0-base.py"
        echo "    - configs/scannet/semseg-pt-v3m1-1-ppt-extreme.py (with PPT)"
        echo ""
        echo "  ScanNet200 (200 classes):"
        echo "    - configs/scannet200/semseg-pt-v3m1-0-base.py"
        echo "    - configs/scannet200/semseg-pt-v3m1-1-ppt-ft.py (fine-tuned from PPT)"
        echo ""
        echo "  ScanNet++:"
        echo "    - configs/scannetpp/semseg-pt-v3m1-0-base.py"
        echo "    - configs/scannetpp/semseg-pt-v3m1-1-submit.py (test submission)"
        echo ""
        echo "  S3DIS (13 classes):"
        echo "    - configs/s3dis/semseg-pt-v3m1-0-base.py"
        echo "    - configs/s3dis/semseg-pt-v3m1-1-rpe.py (with RPE)"
        echo "    - configs/s3dis/semseg-pt-v3m1-2-ppt-extreme.py (with PPT)"
        echo ""
        echo "  Matterport3D:"
        echo "    - configs/matterport3d/semseg-pt-v3m1-0-base.py"
        echo ""
        echo "Outdoor Datasets:"
        echo "  nuScenes (16 classes):"
        echo "    - configs/nuscenes/semseg-pt-v3m1-0-base.py"
        echo ""
        echo "  Waymo:"
        echo "    - configs/waymo/semseg-pt-v3m1-0-base.py"
        echo ""
        echo "For detailed information, see: PTv3_INFERENCE_GUIDE.md"
        echo ""
        ;;
    *)
        print_error "Invalid option"
        exit 1
        ;;
esac

print_header "PTv3 Inference Script Complete"
echo ""
echo "For more information:"
echo "  - Full guide: PTv3_INFERENCE_GUIDE.md"
echo "  - Repository relationship: RELATIONSHIP_PTv3_POINTCEPT.md"
echo "  - Setup help: ./setup_ptv3_inference.sh"
echo ""

