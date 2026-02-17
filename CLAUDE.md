# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EVS Navigation System is a Streamlit-based application for analyzing Ear-Voice Span (EVS) in simultaneous interpretation. It supports bilingual audio transcription (English/Chinese), EVS annotation, corpus analysis, SI quality metrics, and LLM-powered automatic EN-ZH word alignment using various ASR and LLM providers.

## Commands

### Run Application
```bash
streamlit run app.py
```
Access at http://localhost:8501

### Install Dependencies
```bash
# Automated setup (recommended):
setup.bat                # First-time setup (skip if env exists)
setup.bat --reinstall    # Remove env and redo full setup

# Manual setup:
conda create -n cw_evs_app python=3.11
conda activate cw_evs_app
conda install -c conda-forge ffmpeg
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Initialize Database
```bash
python init_database.py          # Create tables (safe to re-run)
python init_database.py --reset  # Drop and recreate all tables
```

### Lint
```bash
# Critical errors only (blocks CI)
flake8 . --select=E9,F63,F7,F82 --show-source --statistics
# All warnings (non-blocking)
flake8 . --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

### Test
```bash
pytest
```
Note: No test infrastructure exists yet (no test files, conftest.py, or tests/ directory).

### Start vLLM Server (for LLM features)
```bash
# Linux/Mac
./start_vllm.sh
# Windows
.\start_vllm.ps1
```

### Verify Environment
```bash
python scripts/verify_env.py
```

## Architecture

### Entry Points
- `app.py` — Main Streamlit application (~8,600 lines, monolithic). This is the primary entry point.
- `main.py` — Word aligner demo component (not the main app)

### Configuration Layer (`config/`)
- `__init__.py` — Re-exports all config symbols. Use `from config import get_db_path, DISPLAY_CONFIG, ...`
- `database_config.py` — Database paths (`PROJECT_ROOT`, `DATA_DIR`, `DB_PATH`), `get_db_path()`, `ensure_data_dir()`
- `asr_language_config.py` — Language-specific ASR model recommendations with `ASRModelInfo` dataclass
- `display_config.py` — UI display and file aliasing settings
- `config.py` — App settings (SMTP, auth: `AUTH_CONFIG`, `GOOGLE_CLOUD_CONFIG`, `BASE_URL`)
- `llm_config.json` — LLM provider configs (Ollama, OpenAI, vLLM)
- `agent_instruction.json` — Agent instruction templates
- `analysis_rules.json` — SI analysis rules

### Database Layer
- `db_manager.py` — Singleton `DBManager.get_instance()` for connection management. Uses `check_same_thread=False` for thread safety, `row_factory = sqlite3.Row` for dict-like access.
- `db_utils.py` — `EVSDataUtils` class with all-static methods for database operations
- Schema: `create_evs_tables.sql` — SQLite at `data/evs_repository.db`
- Key tables: `asr_files`, `asr_results_segments`, `asr_results_words`, `si_analysis_results`, `si_error_details`, `users`, `file_alias_mapping`

### LLM Clients (`clients/`)
- Factory pattern: `from clients import get_llm_client` — returns the active LLM client based on `llm_config.json`
- `ollama_client.py` — `OllamaClient` (singleton, supports `/api/generate` and `/api/chat`)
- `openai_client.py` — `OpenAIClient` (singleton, OpenAI-compatible API including vLLM)

### ASR Processing
- `utils/asr_utils.py` — `ASRUtils` class supporting CrisperWhisper (English), FunASR (Chinese), Google Cloud, Tencent Cloud, IBM Watson
- `config/asr_language_config.py` — Separate provider/model recommendations per language

### NLP & Analysis
- `chinese_nlp_unified.py` — `ChineseNLPUnified` class with `NLPEngine` enum (JIEBA or HANLP). Supports custom dictionaries via `chinese_custom_dict.txt`.
- `paraverbal_process.py` — EVS calculation and prosody analysis
- `concordance_utils.py` — Corpus analysis (word frequency, keywords, collocates, n-grams)
- `utils/si_analysis_integration.py` — SI quality analysis integration

### UI & Session
- `components/session_state.py` — `SessionState` data class and `initialize_session_state()` for Streamlit session management
- `file_alias_manager.py` — Privacy-protected file name anonymization (SHA256 hash, generates `file_1`, `file_2` aliases)
- `privacy_settings.py` — Streamlit sidebar UI for privacy controls and alias management
- `cache_utils.py` — `@st.cache_resource` decorators for DB connections and common queries
- `evs_annotator/` and `word_aligner/` — Custom Streamlit HTML components

### Multi-page Structure (`pages/`)
- `admin_panel.py` — Admin configuration panel
- `download_audio.py` — Audio download utility

## Key Patterns

### Database Access
```python
from db_utils import EVSDataUtils
# All methods are static
results = EVSDataUtils.get_asr_results(file_id)
```

### Database Connection
```python
from db_manager import DBManager
db = DBManager.get_instance()
result = db.execute_query("SELECT ...", params)
```

### LLM Client
```python
from clients import get_llm_client
client = get_llm_client()  # Returns OllamaClient or OpenAIClient based on config
```

### Configuration Import
```python
from config import get_db_path, DISPLAY_CONFIG, LANGUAGE_ASR_RECOMMENDATIONS
```

## CI/CD

GitHub Actions workflow (`.github/workflows/python-app.yml`) runs on push/PR to `main`:
- Python 3.11, Ubuntu, with FFmpeg
- Lint: flake8 (syntax errors block, warnings don't)
- Test: pytest

## Dependencies & Constraints

- **Python**: 3.11 (CI and setup.bat both use 3.11)
- **FFmpeg**: Required for audio processing
- **protobuf**: Must be `>=3.20,<4` for HanLP compatibility
- **numpy**: Must be `<2.0`
- **CUDA GPU**: Optional, needed for local CrisperWhisper (~4GB VRAM) and FunASR (~2GB VRAM)

## Environment Setup

- Google Cloud credentials: `export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"`
- Tencent Cloud: Create `keys/tencent_cloud_config.json` with `secret_id` and `secret_key`
- API keys go in `keys/` directory (gitignored)
- LLM config: Edit `config/llm_config.json` to switch between Ollama, OpenAI, or vLLM providers
- Docker: `Dockerfile` available for containerized deployment
