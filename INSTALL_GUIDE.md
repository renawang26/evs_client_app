# EVS Navigation System - Installation Guide

This guide helps you install the EVS Navigation System on a new computer.

## System Requirements

- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **RAM**: 8GB minimum, 16GB recommended (for ASR models)
- **Disk Space**: 5GB minimum (more for ASR models)
- **GPU** (Optional): NVIDIA GPU with CUDA support for faster ASR processing

---

## Quick Start

### Windows
```batch
setup.bat
start.bat
```

### Linux/macOS
```bash
chmod +x setup.sh start.sh
./setup.sh
./start.sh
```

---

## Detailed Installation

### Step 1: Install Python

#### Windows

**Option A: Official Installer (Recommended)**
1. Download Python 3.10+ from https://www.python.org/downloads/
2. Run the installer
3. **IMPORTANT**: Check "Add Python to PATH" at the bottom of the installer
4. Click "Install Now"

**Option B: Using winget**
```batch
winget install Python.Python.3.10
```

**Option C: Using Conda**
```batch
conda create -n evs python=3.10
conda activate evs
```

#### macOS

**Using Homebrew (Recommended)**
```bash
brew install python@3.10
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
```

### Step 2: Install FFmpeg

FFmpeg is required for audio file processing.

#### Windows

**Option A: Using winget (Recommended)**
```batch
winget install Gyan.FFmpeg
```

**Option B: Manual Installation**
1. Download from https://ffmpeg.org/download.html#build-windows
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to your system PATH:
   - Press Win+X, select "System"
   - Click "Advanced system settings"
   - Click "Environment Variables"
   - Under "System variables", find "Path", click Edit
   - Click "New" and add `C:\ffmpeg\bin`
   - Click OK to save

#### macOS
```bash
brew install ffmpeg
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt install ffmpeg
```

### Step 3: Install Dependencies

```bash
# If using conda
conda activate evs

# Install all packages
pip install -r requirements.txt
```

### Step 4: Initialize Database

```bash
python init_database.py
```

### Step 5: Start the Application

```bash
streamlit run app.py
```

The application will be available at: **http://localhost:8501**

---

## Optional: GPU Acceleration

For faster ASR processing with CrisperWhisper and FunASR:

### Windows/Linux with NVIDIA GPU

1. Install NVIDIA CUDA Toolkit 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive

2. Install CUDA-enabled PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Verify GPU Support
```python
import torch
print(torch.cuda.is_available())     # Should print: True
print(torch.cuda.get_device_name(0)) # Shows your GPU name
```

---

## Optional: Configure ASR Providers

### CrisperWhisper + FunASR (Default)

These are the default providers and run **locally** without API keys:
- **CrisperWhisper** for English — verbatim ASR with filler word detection
- **FunASR (paraformer-zh)** for Chinese — Alibaba's high-accuracy Chinese ASR

Models download automatically on first use (~2-4GB). Requires CUDA GPU for best performance.

| Provider | Language | Model Size | VRAM Required |
|----------|----------|------------|---------------|
| CrisperWhisper | English | ~3GB | ~4GB |
| FunASR paraformer-zh | Chinese | ~1GB | ~2GB |

### Tencent Cloud ASR (Optional, for Chinese)

1. Create account at https://cloud.tencent.com/
2. Get SecretId and SecretKey from https://console.cloud.tencent.com/cam/capi
3. Create `keys/tencent_cloud_config.json`:
```json
{
    "secret_id": "YOUR_SECRET_ID",
    "secret_key": "YOUR_SECRET_KEY"
}
```

### Google Cloud Speech-to-Text (Optional)

1. Create project at https://console.cloud.google.com/
2. Enable Speech-to-Text API
3. Create service account and download JSON key
4. Save as `keys/google_credentials.json`
5. Set environment variable:
```bash
# Windows
set GOOGLE_APPLICATION_CREDENTIALS=keys\google_credentials.json

# Linux/macOS
export GOOGLE_APPLICATION_CREDENTIALS=keys/google_credentials.json
```

---

## Optional: Configure LLM (for Auto Align)

The Auto Align feature uses an LLM to automatically pair EN-ZH words. Configure in `config/llm_config.json`.

### Ollama (Recommended for local use)

1. Install Ollama from https://ollama.com/
2. Pull a model:
```bash
ollama pull gemma3:27b
```
3. The default config already points to local Ollama — no changes needed.

### vLLM Server

For high-throughput inference:
```bash
# Start vLLM server
./start_vllm.sh   # Linux/macOS
.\start_vllm.ps1  # Windows
```

Update `config/llm_config.json`:
```json
{
    "active_llm_provider": "vllm",
    "llm_configs": {
        "vllm": {
            "llm_provider": "vllm",
            "llm_api_base": "http://localhost:8000/v1",
            "llm_model": "Qwen/Qwen2.5-7B-Instruct"
        }
    }
}
```

---

## Verify Installation

Run the verification script:

```bash
python scripts/verify_env.py
```

Expected output:
```
============================================================
  EVS Navigation System - Environment Verification
============================================================

[Python Version]
  Version: 3.10.x
  Status: OK

[FFmpeg]
  ffmpeg version x.x.x
  Status: OK

[Core Packages]
  streamlit: OK
  pandas: OK
  ...

[Database]
  Status: OK

Summary
============================================================
  Environment is ready!
```

---

## Troubleshooting

### Python not found after installation

**Windows**: Ensure Python was added to PATH during installation. You may need to:
1. Uninstall Python
2. Reinstall and check "Add Python to PATH"
3. Restart your terminal/command prompt

**Linux/macOS**: Try using `python3` instead of `python`:
```bash
python3 --version
```

### FFmpeg not found

Open a new terminal after installation and verify:
```bash
ffmpeg -version
```

### pip install fails with SSL errors

```bash
python -m pip install --upgrade pip
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

### HanLP import errors (protobuf version)

```bash
pip install protobuf==3.19.6 --force-reinstall
```

### Out of memory errors with CrisperWhisper

- Use Google Cloud or Tencent Cloud ASR instead (cloud-based, no local GPU needed)
- Or reduce audio file length before processing

### Database initialization fails

```bash
mkdir data
python init_database.py
```

### Port 8501 already in use

```bash
streamlit run app.py --server.port 8502
```

---

## File Structure

```
evs_client_app/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── init_database.py          # Database initialization
├── create_evs_tables.sql     # Database schema
├── setup.bat / start.bat     # Windows scripts
├── config/                   # Configuration files
│   └── llm_config.json       # LLM provider settings
├── utils/                    # Utility modules
│   └── asr_utils.py          # ASR processing
├── clients/                  # LLM client adapters
├── components/               # UI components
├── evs_annotator/            # EVS annotation component
├── pages/                    # Multi-page app pages
├── data/                     # Database storage (auto-created)
│   └── evs_repository.db     # SQLite database
└── keys/                     # API credentials (create as needed)
```

---

## Updating

To update to a new version:

1. Backup your database:
   ```bash
   cp data/evs_repository.db data/evs_repository_backup.db
   ```
2. Pull latest code:
   ```bash
   git pull
   ```
3. Update dependencies:
   ```bash
   pip install -r requirements.txt --upgrade
   ```
4. Restart the application
