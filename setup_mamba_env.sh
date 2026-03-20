#!/usr/bin/env bash
#
# Setup script for tdlu mamba environment
# Creates the environment and installs all necessary packages for the mammography TDLU project.
# Compatible with NVIDIA L40S GPUs (CUDA 12.x/13.x driver).
#

set -e

ENV_NAME="${TDLU_ENV_NAME:-tdlu}"
PYTHON_VERSION="3.11"

echo "=========================================="
echo "Creating mamba environment: ${ENV_NAME}"
echo "Python version: ${PYTHON_VERSION}"
echo "=========================================="

# Create environment with Python
mamba create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

# Activate and install packages (mamba run avoids shell activation issues)
echo ""
echo "Installing PyTorch with CUDA 12.4 support..."
mamba run -n "${ENV_NAME}" mamba install -y \
    pytorch torchvision pytorch-cuda=12.4 \
    -c pytorch -c nvidia

echo ""
echo "Installing deep learning & training packages..."
mamba run -n "${ENV_NAME}" mamba install -y \
    pytorch-lightning \
    torchmetrics \
    -c conda-forge

echo ""
echo "Installing medical imaging packages..."
mamba run -n "${ENV_NAME}" mamba install -y \
    torchio \
    pydicom \
    gdcm \
    -c conda-forge

echo ""
echo "Installing scientific & data packages..."
mamba run -n "${ENV_NAME}" mamba install -y \
    numpy scipy pandas scikit-learn \
    matplotlib pillow \
    pyyaml tqdm \
    -c conda-forge

echo ""
echo "Installing OpenCV..."
mamba run -n "${ENV_NAME}" mamba install -y \
    opencv \
    -c conda-forge

echo ""
echo "Installing Hugging Face transformers (for Swin models)..."
mamba run -n "${ENV_NAME}" pip install transformers

echo ""
echo "=========================================="
echo "Environment '${ENV_NAME}' created successfully!"
echo "=========================================="
echo ""
echo "Activate with:  mamba activate ${ENV_NAME}"
echo ""
echo "Verify installation:"
echo "  mamba run -n ${ENV_NAME} python -c \"import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())\""
echo ""
echo "Run training (from tdlu directory):"
echo "  cd tdlu && mamba run -n ${ENV_NAME} python main.py --config_path config/config.yaml"
echo ""
