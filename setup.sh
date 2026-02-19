#!/usr/bin/env bash
# EVS Navigation System - WSL/Linux Setup Script (Conda)
# Usage: bash setup.sh              - Setup (skip if env exists)
#        bash setup.sh --reinstall  - Remove env and redo full setup

set -euo pipefail
cd "$(dirname "$0")"

# ============================================================
# Parse arguments
# ============================================================
REINSTALL=0
for arg in "$@"; do
    case "$arg" in
        --reinstall|-r) REINSTALL=1 ;;
    esac
done

echo "============================================================"
echo "  EVS Navigation System - Setup Script [Conda / WSL]"
echo "============================================================"
if [ "$REINSTALL" -eq 1 ]; then
    echo "  Mode: Force Reinstall"
fi
echo ""

# ============================================================
# Step 1 — Check Conda
# ============================================================
echo "[1/4] Checking Conda installation..."

if command -v conda &>/dev/null; then
    echo "      Conda found: $(conda --version)"
else
    echo ""
    echo "[ERROR] Conda not found."
    echo "        Please install Miniconda or Anaconda first:"
    echo "          https://docs.conda.io/en/latest/miniconda.html"
    echo ""
    echo "        Quick install (Miniconda):"
    echo "          wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "          bash Miniconda3-latest-Linux-x86_64.sh"
    echo "          source ~/.bashrc"
    echo ""
    exit 1
fi

# ============================================================
# Step 2 — Check FFmpeg (system)
# ============================================================
echo ""
echo "[2/4] Checking FFmpeg installation..."

if command -v ffmpeg &>/dev/null; then
    echo "      FFmpeg found: $(ffmpeg -version 2>&1 | head -1)"
else
    echo "      FFmpeg not found on system PATH."
    echo "      Will attempt to install via conda in the next step."
fi

# ============================================================
# Step 3 — Create / activate Conda environment
# ============================================================
echo ""
echo "[3/4] Setting up Conda environment [cw_evs_app]..."

# Source conda so we can use `conda activate`
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

ENV_EXISTS=$(conda env list | grep -c "^cw_evs_app" || true)

if [ "$ENV_EXISTS" -gt 0 ]; then
    if [ "$REINSTALL" -eq 1 ]; then
        echo "      Removing existing environment..."
        conda deactivate 2>/dev/null || true
        conda env remove -n cw_evs_app -y
        echo "      Creating new environment [Python 3.11]..."
        conda create -n cw_evs_app python=3.11 -y
    else
        echo "      Environment 'cw_evs_app' already exists. Skipping creation."
        echo "      Use 'bash setup.sh --reinstall' to recreate."
    fi
else
    echo "      Creating new environment [Python 3.11]..."
    conda create -n cw_evs_app python=3.11 -y
fi

echo "      Activating environment..."
conda activate cw_evs_app

if [ "$CONDA_DEFAULT_ENV" != "cw_evs_app" ]; then
    echo ""
    echo "[ERROR] Failed to activate conda environment 'cw_evs_app'."
    echo "        Try running: conda init bash"
    echo "        Then reopen your terminal and run setup.sh again."
    exit 1
fi
echo "      Environment activated: $CONDA_DEFAULT_ENV"

# ============================================================
# Install FFmpeg via conda
# ============================================================
echo "      Installing FFmpeg via conda..."
conda install -c conda-forge ffmpeg -y

echo "      Verifying FFmpeg installation..."
if command -v ffmpeg &>/dev/null; then
    echo "      FFmpeg installed successfully."
else
    echo ""
    echo "[WARNING] FFmpeg could not be verified after conda install."
    echo "          Audio processing and CrisperWhisper transcription will not work."
    echo ""
    echo "          Please install FFmpeg manually using one of these methods:"
    echo ""
    echo "          Option 1 - apt (Ubuntu/Debian):"
    echo "            sudo apt update && sudo apt install -y ffmpeg"
    echo ""
    echo "          Option 2 - conda (retry):"
    echo "            conda install -c conda-forge ffmpeg"
    echo ""
    echo "          Option 3 - snap:"
    echo "            sudo snap install ffmpeg"
    echo ""
    echo "          After installing FFmpeg, re-run: bash setup.sh"
    echo ""
fi

# ============================================================
# Install PyTorch via conda
# ============================================================
echo ""
echo "      Detecting GPU..."
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    echo "      NVIDIA GPU detected. Installing PyTorch with CUDA via conda..."
    if ! conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y; then
        echo "      CUDA 12.4 failed, trying CUDA 12.1..."
        conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
    fi
else
    echo "      No NVIDIA GPU detected. Installing CPU-only PyTorch via conda..."
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
fi

# ============================================================
# Install pip dependencies
# ============================================================
echo ""
echo "      Installing pip dependencies..."
pip install -r requirements.txt || echo "[WARNING] Some pip requirements failed. Check errors above."

# ============================================================
# Verify key packages
# ============================================================
echo ""
echo "      Verifying installation..."
python -c "import torch; gpu='CUDA '+torch.version.cuda if torch.cuda.is_available() else 'CPU only'; print(f'PyTorch {torch.__version__} ({gpu})')"
python -c "import streamlit; import transformers; import funasr; import plotly; import numpy; import scipy; print('All key packages verified.')" \
    || echo "[WARNING] Some packages failed to import. Check errors above."

# ============================================================
# Step 4 — Initialize Database
# ============================================================
echo ""
echo "[4/4] Initializing database..."

if [ "$REINSTALL" -eq 1 ]; then
    echo "      Reinitializing database..."
    rm -f data/evs_repository.db
    python init_database.py && echo "      Database reinitialized successfully." \
        || echo "[WARNING] Database reinitialization had issues."
elif [ -f "data/evs_repository.db" ]; then
    echo "      Database already exists."
else
    python init_database.py && echo "      Database initialized successfully." \
        || echo "[WARNING] Database initialization had issues."
fi

# ============================================================
# Done
# ============================================================
echo ""
echo "============================================================"
echo "  Setup Complete!"
echo "============================================================"
echo ""
echo "To start the application:"
echo "  bash start.sh"
echo "  Or manually:"
echo "    conda activate cw_evs_app"
echo "    python -m streamlit run app.py"
echo ""
echo "If packages are missing, run:"
echo "  bash setup.sh --reinstall"
echo ""
echo "The application will be available at:"
echo "  http://localhost:8501"
echo ""
