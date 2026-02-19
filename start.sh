#!/usr/bin/env bash
# EVS Navigation System - WSL/Linux Start Script (Conda)
# Usage: bash start.sh

set -euo pipefail
cd "$(dirname "$0")"

echo "============================================================"
echo "  EVS Navigation System - Starting Application"
echo "============================================================"
echo ""

# ============================================================
# Check Conda
# ============================================================
if ! command -v conda &>/dev/null; then
    echo "[ERROR] Conda not found. Please install Anaconda/Miniconda."
    echo "        Then run: bash setup.sh"
    exit 1
fi

# Source conda so we can use `conda activate`
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ============================================================
# Check environment exists
# ============================================================
if ! conda env list | grep -q "^cw_evs_app"; then
    echo "[ERROR] Conda environment 'cw_evs_app' not found."
    echo "        Please run setup.sh first:"
    echo "          bash setup.sh"
    exit 1
fi

# ============================================================
# Activate environment
# ============================================================
echo "Activating conda environment 'cw_evs_app'..."
conda activate cw_evs_app

if [ "$CONDA_DEFAULT_ENV" != "cw_evs_app" ]; then
    echo "[ERROR] Failed to activate conda environment 'cw_evs_app'."
    echo "        Try running: conda init bash"
    echo "        Then reopen your terminal."
    exit 1
fi

# ============================================================
# Find an available port (8501–8510)
# ============================================================
PORT=8501
while [ "$PORT" -le 8510 ]; do
    if ! ss -ltn 2>/dev/null | grep -q ":${PORT} " && \
       ! netstat -ltn 2>/dev/null | grep -q ":${PORT} "; then
        break
    fi
    echo "[INFO] Port $PORT is in use, trying next port..."
    PORT=$((PORT + 1))
done

if [ "$PORT" -gt 8510 ]; then
    echo "[ERROR] No available ports found in range 8501-8510."
    exit 1
fi

# ============================================================
# Start application
# ============================================================
echo "Starting EVS Navigation System on port $PORT..."
echo ""
echo "The application will be available at:"
echo "  http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

python -m streamlit run app.py --server.address localhost --server.port "$PORT"
