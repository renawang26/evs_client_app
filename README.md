# EVS Navigation System

A Streamlit-based application for analyzing **Ear-Voice Span (EVS)** in simultaneous interpretation. Supports multi-language audio transcription (English/Chinese), EVS annotation, corpus analysis, SI quality metrics, and LLM-powered automatic alignment.

## Features

### Audio Transcription
- Multiple ASR providers:
  - **CrisperWhisper** (English) — verbatim ASR with filler/stutter detection
  - **FunASR** (Chinese) — Alibaba's paraformer-zh model
  - **Google Cloud Speech-to-Text**
  - **Tencent Cloud ASR**
  - **IBM Watson Speech-to-Text**
- Dual-channel processing (source/target separation)
- Word-level timestamps
- Multiple audio formats (WAV, MP3, M4A, FLAC)

### Transcription Editing
- Time-aligned bilingual display (EN/ZH side by side)
- Inline word editing with change tracking
- 30-second segment grouping with audio playback

### EVS Annotation
- Interactive word-pair selection (click EN word + ZH word)
- **Auto Align** — LLM-powered automatic EN-ZH word pairing
- Visual color-coded pair display
- EVS calculation with pair preview
- Batch save to database

### Corpus Analysis
- Word frequency lists
- Keyword extraction (log-likelihood)
- Collocate analysis (window-based)
- N-gram / word cluster generation

### SI Quality Analysis
- LLM-based semantic pair analysis
- Prosody analysis (filled pauses, unfilled pauses, speech rate, repairs)
- Quality metrics with export support

---

## Prerequisites

| Requirement | Windows | WSL (Linux) |
|---|---|---|
| Python 3.11 | via Conda | via Conda |
| Conda (Miniconda / Anaconda) | required | required |
| FFmpeg | auto-installed by setup | auto-installed by setup |
| CUDA GPU | optional | optional |
| LLM Server | optional | optional |

---

## Quick Start

### Windows

```bat
REM 1. Clone the repository
git clone https://github.com/renawang26/evs_client_app.git
cd evs_client_app

REM 2. Run setup (first time)
setup.bat

REM 3. Start the application
start.bat
```

To reinstall from scratch:
```bat
setup.bat --reinstall
```

---

### WSL (Linux)

```bash
# 1. Clone the repository
git clone https://github.com/renawang26/evs_client_app.git
cd evs_client_app

# 2. Run setup (first time)
bash setup.sh

# 3. Start the application
bash start.sh
```

To reinstall from scratch:
```bash
bash setup.sh --reinstall
```

---

### What the setup scripts do

Both `setup.bat` (Windows) and `setup.sh` (WSL) perform the same steps:

1. **Check Conda** — exits with install instructions if not found
2. **Check FFmpeg** — reports status; installs via conda in the next step
3. **Create conda environment** `cw_evs_app` with Python 3.11
4. **Install FFmpeg** via `conda install -c conda-forge ffmpeg`; if verification fails, shows platform-specific manual install instructions
5. **Install PyTorch** — detects NVIDIA GPU and installs CUDA or CPU-only build accordingly
6. **Install pip dependencies** from `requirements.txt`
7. **Verify** key packages (torch, streamlit, transformers, funasr, plotly)
8. **Initialize the database** at `data/evs_repository.db`

---

## Windows vs. WSL Differences

| | Windows (`setup.bat` / `start.bat`) | WSL (`setup.sh` / `start.sh`) |
|---|---|---|
| Script format | Batch (`.bat`) | Bash (`.sh`) |
| Run command | `setup.bat` | `bash setup.sh` |
| FFmpeg manual install | winget / Chocolatey / manual download | `sudo apt install ffmpeg` |
| Port check | `netstat` (Windows) | `ss` / `netstat` (Linux) |
| Conda activation | `call conda activate` | `source conda.sh && conda activate` |
| GPU support | CUDA via nvidia-smi | CUDA via nvidia-smi (WSL2 with GPU passthrough) |
| Browser | Opens automatically | Navigate to http://localhost:8501 manually |

> **WSL GPU note**: CUDA acceleration in WSL2 requires Windows 11 or Windows 10 21H2+
> with the [WSL2 GPU driver](https://developer.nvidia.com/cuda/wsl) installed on the Windows host.

---

## Manual Setup (without scripts)

If you prefer to set up manually:

```bash
# Create environment
conda create -n cw_evs_app python=3.11
conda activate cw_evs_app

# Install FFmpeg
conda install -c conda-forge ffmpeg

# Install PyTorch (CUDA — adjust cuda version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Install PyTorch (CPU only)
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install pip dependencies
pip install -r requirements.txt

# Initialize database
python init_database.py

# Run
python -m streamlit run app.py
```

---

## Optional Configuration

### ASR Providers

**CrisperWhisper + FunASR (Default, Local)**

No API keys needed. Models download automatically on first use.
- CrisperWhisper requires ~3–4 GB VRAM (falls back to CPU if unavailable)
- FunASR requires ~2 GB VRAM

**Google Cloud Speech-to-Text**
1. Enable the API at https://console.cloud.google.com/
2. Create a service account and download the JSON key
3. Save as `keys/google_credentials.json`
4. Set environment variable:
   ```bash
   # Linux / WSL
   export GOOGLE_APPLICATION_CREDENTIALS=keys/google_credentials.json

   # Windows (Command Prompt)
   set GOOGLE_APPLICATION_CREDENTIALS=keys\google_credentials.json
   ```

**Tencent Cloud ASR**
1. Get credentials from https://console.cloud.tencent.com/cam/capi
2. Create `keys/tencent_cloud_config.json`:
   ```json
   {
       "secret_id": "YOUR_SECRET_ID",
       "secret_key": "YOUR_SECRET_KEY"
   }
   ```

### LLM (for Auto Align and SI Analysis)

Configure in `config/llm_config.json`. Default uses local Ollama:

```json
{
    "active_llm_provider": "ollama",
    "llm_configs": {
        "ollama": {
            "llm_provider": "ollama",
            "llm_model": "gemma3:27b",
            "llm_base_url": "http://localhost:11434"
        }
    }
}
```

Install Ollama and pull a model:
```bash
# Linux / WSL
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma3:27b

# Windows — download from https://ollama.com/
ollama pull gemma3:27b
```

### vLLM Server

```bash
# Linux / WSL
bash start_vllm.sh

# Windows (PowerShell)
.\start_vllm.ps1
```

---

## Database

SQLite database stored at `data/evs_repository.db`.

```bash
python init_database.py          # Create tables (safe to re-run)
python init_database.py --reset  # Drop and recreate all tables
```

---

## Project Structure

```
evs_client_app/
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── init_database.py              # Database initialization
├── create_evs_tables.sql         # Database schema
│
├── setup.bat / setup.sh          # Setup scripts (Windows / WSL)
├── start.bat / start.sh          # Launch scripts (Windows / WSL)
├── start_vllm.sh / start_vllm.ps1  # vLLM server launcher
│
├── config/                       # Configuration
│   ├── config.py                 # App settings (SMTP, auth)
│   ├── asr_language_config.py    # ASR model recommendations per language
│   ├── database_config.py        # Database paths
│   ├── display_config.py         # UI display settings
│   └── llm_config.json           # LLM provider configs (Ollama/vLLM/OpenAI)
│
├── utils/                        # Utility modules
│   ├── asr_utils.py              # ASR processing (CrisperWhisper, FunASR, etc.)
│   ├── analysis_utils.py         # Analysis helpers
│   └── si_analysis_integration.py
│
├── clients/                      # LLM client adapters
│   ├── ollama_client.py
│   └── openai_client.py
│
├── components/
│   └── session_state.py          # Session state management
│
├── evs_annotator/                # Custom Streamlit component for EVS annotation
├── word_aligner/                 # Custom Streamlit component for word alignment
│
├── pages/                        # Multi-page app
│   ├── admin_panel.py
│   └── download_audio.py
│
├── db_manager.py                 # Database connection management
├── db_utils.py                   # Database operations
├── paraverbal_process.py         # Prosody and EVS calculation
├── concordance_utils.py          # Corpus analysis
├── chinese_nlp_unified.py        # Chinese NLP processing
├── user_utils.py                 # Authentication and user management
├── report_generator.py           # Excel / PDF report generation
├── cache_utils.py                # Streamlit caching helpers
├── file_alias_manager.py         # Privacy file name aliasing
├── privacy_settings.py           # Privacy controls UI
├── email_queue.py                # Email notification queue
│
├── data/                         # Database storage (gitignored)
│   └── evs_repository.db
└── keys/                         # API credentials (gitignored)
```

---

## License

This project is for academic research purposes.
