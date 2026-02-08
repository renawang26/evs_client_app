# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EVS Navigation System is a Streamlit-based application for analyzing Ear-Voice Span (EVS) in simultaneous interpretation. It supports multi-language audio transcription (English/Chinese), corpus analysis, and quality metrics using various ASR providers.

## Commands

### Run Application
```bash
streamlit run app.py
```
Access at http://localhost:8501

### Install Dependencies
```bash
conda create -n evs python
conda activate evs
pip install -r requirements.txt
```

### Initialize Database
```bash
python init_database.py          # Create tables (safe to re-run)
python init_database.py --reset  # Drop and recreate all tables
```

### Lint
```bash
flake8 . --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

### Test
```bash
pytest
```

### Start vLLM Server (for LLM features)
```bash
# Linux/Mac
./start_vllm.sh
# Windows
.\start_vllm.ps1
```

## Architecture

### Entry Point
- `app.py` - Main Streamlit application (~8,600 lines, monolithic - see REFACTORING_GUIDE.md for planned modularization)

### Configuration Layer (`config/`)
- `database_config.py` - Database paths, `get_db_path()`, `ensure_data_dir()`
- `asr_language_config.py` - ASR model recommendations by language (Whisper, FunASR, etc.)
- `display_config.py` - UI display and file aliasing settings
- `llm_config.json` - LLM provider configs (Ollama, OpenAI, vLLM)

### Core Modules
- `db_utils.py` - `EVSDataUtils` class with static methods for all database operations
- `utils/asr_utils.py` - `ASRUtils` class supporting multiple ASR providers:
  - CrisperWhisper (nyrahealth - verbatim English ASR with filler/stutter detection)
  - FunASR (Alibaba - excellent for Chinese)
  - Google Cloud Speech-to-Text
  - Tencent Cloud ASR
  - IBM Watson
- `config/asr_language_config.py` - Language-specific ASR model recommendations
  - Separate provider/model selection for English vs Chinese
  - FunASR paraformer-zh recommended for Chinese
  - CrisperWhisper for English (verbatim with filler detection)
- `user_utils.py` - `UserUtils` class for authentication, password hashing, email verification
- `paraverbal_process.py` - EVS calculation and prosody analysis
- `concordance_utils.py` - Corpus analysis (word frequency, keywords, collocates, n-grams)

### Data Layer
- Database: SQLite at `data/evs_repository.db`
- Schema: `create_evs_tables.sql`
- Key tables: `asr_results_segments`, `asr_results_words`, `si_analysis_results`, `users`

### Caching
- `cache_utils.py` - Streamlit `@st.cache_resource` decorators for DB connections and common queries

### Multi-page Structure (`pages/`)
- `admin_panel.py` - Admin configuration panel (partially extracted)
- Future pages planned: login, asr_processing, edit_transcription, evs_annotation, word_analysis, si_analysis

## Key Patterns

### Database Access
```python
from db_utils import EVSDataUtils
# All methods are static
results = EVSDataUtils.get_asr_results(file_id)
```

### ASR Processing
```python
from utils.asr_utils import ASRUtils
# Supports channel separation, format conversion, word-level timestamps
```

### Configuration Import
```python
from config import get_db_path, ASR_MODELS, DISPLAY_CONFIG
```

## Dependencies

- **Runtime**: Python 3.10+, FFmpeg (latest Python recommended)
- **Web**: Streamlit 1.31.1, streamlit-aggrid
- **Audio**: pydub, ffmpeg-python, torch, torchaudio
- **ASR**: transformers (CrisperWhisper), funasr (FunASR), google-cloud-speech, tencentcloud-sdk-python, ibm-watson
- **NLP**: jieba, hanlp (Chinese), protobuf==3.19.6 (HanLP compatibility)
- **Data**: pandas, numpy<2.0, plotly

## Environment Setup

- Google Cloud credentials: `export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"`
- Store API keys in `keys/` directory (gitignored)
- CUDA support available via conda for GPU acceleration
