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

## Prerequisites

- **Python**: 3.10+ recommended
- **FFmpeg**: Required for audio processing
- **CUDA GPU** (Optional): For faster CrisperWhisper/FunASR inference
- **LLM Server** (Optional): Ollama, vLLM, or OpenAI-compatible API for Auto Align and SI analysis

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/renawang26/evs_client_app.git
cd evs_client_app

# Create virtual environment
conda create -n evs python=3.10
conda activate evs

# Install dependencies
pip install -r requirements.txt
```

### 2. Install FFmpeg

```bash
# Windows
winget install Gyan.FFmpeg

# macOS
brew install ffmpeg

# Linux
sudo apt install ffmpeg
```

### 3. Initialize database

```bash
python init_database.py
```

### 4. Run

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Project Structure

```
evs_client_app/
├── app.py                    # Main Streamlit application
├── main.py                   # Alternative entry point
├── requirements.txt          # Python dependencies
├── init_database.py          # Database initialization
├── create_evs_tables.sql     # Database schema
│
├── config/                   # Configuration
│   ├── config.py             # App settings (SMTP, auth)
│   ├── asr_language_config.py # ASR model recommendations per language
│   ├── database_config.py    # Database paths
│   ├── display_config.py     # UI display settings
│   └── llm_config.json       # LLM provider configs (Ollama/vLLM/OpenAI)
│
├── utils/                    # Utility modules
│   ├── asr_utils.py          # ASR processing (CrisperWhisper, FunASR, etc.)
│   ├── analysis_utils.py     # Analysis helpers
│   └── si_analysis_integration.py  # SI quality analysis
│
├── clients/                  # LLM client adapters
│   ├── ollama_client.py      # Ollama integration
│   └── openai_client.py      # OpenAI-compatible API client
│
├── components/               # UI components
│   └── session_state.py      # Session state management
│
├── evs_annotator/            # Custom Streamlit component for EVS annotation
│   └── index.html
│
├── word_aligner/             # Word alignment component
│   └── index.html
│
├── pages/                    # Multi-page app pages
│   ├── admin_panel.py        # Admin configuration
│   └── download_audio.py     # Audio download
│
├── db_utils.py               # Database operations (EVSDataUtils)
├── db_manager.py             # Database connection management
├── paraverbal_process.py     # Prosody and EVS calculation
├── concordance_utils.py      # Corpus analysis
├── chinese_nlp_unified.py    # Chinese NLP processing
├── user_utils.py             # Authentication and user management
├── cache_utils.py            # Streamlit caching helpers
├── file_alias_manager.py     # File display name management
├── privacy_settings.py       # Privacy configuration
├── email_queue.py            # Email notification queue
├── report_generator.py       # Report generation
│
├── data/                     # Database storage (gitignored)
│   └── evs_repository.db    # SQLite database
├── keys/                     # API credentials (gitignored)
├── setup.bat / start.bat     # Windows scripts
└── start_vllm.sh / .ps1     # vLLM server launcher
```

## Optional Configuration

### ASR Providers

**CrisperWhisper + FunASR (Default, Local)**
No API keys needed. Models download automatically on first use. Requires ~4GB VRAM for CrisperWhisper.

**Google Cloud Speech-to-Text**
1. Enable the API at https://console.cloud.google.com/
2. Create a service account and download the JSON key
3. Save as `keys/google_credentials.json`
4. Set environment variable:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=keys/google_credentials.json
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

Install Ollama from https://ollama.com/ and pull a model:
```bash
ollama pull gemma3:27b
```

### GPU Acceleration

For CUDA-enabled PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Database

SQLite database at `data/evs_repository.db`. Initialize or reset:

```bash
python init_database.py          # Create tables (safe to re-run)
python init_database.py --reset  # Drop and recreate all tables
```

## License

This project is for academic research purposes.
