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
conda create -n cw_evs_app python=3.11 -y
conda activate cw_evs_app
# Install FFmpeg as a system binary (NOT via conda):
#   Windows: winget install --id=Gyan.FFmpeg -e
#   Ubuntu:  sudo apt install ffmpeg
#   macOS:   brew install ffmpeg
# Install PyTorch (GPU):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
# Install PyTorch (CPU only):
#   pip install torch torchaudio
pip install -r requirements.txt
# Optional: for advanced interpretation analysis (semantic similarity, spaCy NLP)
pip install -r requirements_advanced.txt
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
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

### Pre-download ASR Models
```bash
python download_models.py                # Download all models
python download_models.py crisperwhisper # CrisperWhisper only (~3GB)
python download_models.py funasr         # FunASR only (~1GB)
```

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

### Database Diagnostics
```bash
python analyze_db.py  # Prints schema, row counts, and sample rows for all tables
```

## Architecture

### Entry Points
- `app.py` — Main Streamlit application (~9,800 lines, monolithic). This is the primary entry point.
- `main.py` — Word aligner demo component (not the main app). Run with `streamlit run main.py` to test EN-ZH alignment UI in isolation.

### Configuration Layer (`config/`)
- `__init__.py` — Re-exports all config symbols. Use `from config import get_db_path, DISPLAY_CONFIG, ...`
- `database_config.py` — Database paths (`PROJECT_ROOT`, `DATA_DIR`, `DB_PATH`, `SCHEMA_FILE`, `REPOSITORY_ROOT`), `get_db_path()`, `get_db_path_str()`, `ensure_data_dir()`, `get_schema_file()`
- `asr_language_config.py` — Language-specific ASR model recommendations with `ASRModelInfo` dataclass. Exports `FUNASR_MODELS`, `CRISPERWHISPER_MODELS`, `LANGUAGE_ASR_RECOMMENDATIONS`, and helper functions (`get_available_models_for_language`, `get_recommended_model_for_language`, etc.)
- `display_config.py` — UI display and file aliasing settings (`ENABLE_FILE_ALIASING`, `DISPLAY_CONFIG`)
- `config.py` — App settings (SMTP, auth: `AUTH_CONFIG`, `GOOGLE_CLOUD_CONFIG`, `BASE_URL`)
- `llm_config.json` — LLM provider configs (Ollama, OpenAI, vLLM)
- `agent_instruction.json` — Agent instruction templates
- `analysis_rules.json` — SI analysis rules
- `chinese_custom_dict.txt` — Custom dictionary entries for Chinese NLP segmentation

### Database Layer
- `db_manager.py` — Singleton `DBManager.get_instance()` for connection management. Uses `check_same_thread=False` for thread safety, `row_factory = sqlite3.Row` for dict-like access.
- `db_utils.py` — `EVSDataUtils` class with all-static methods for database operations
- `save_asr_results.py` — Persistence module for ASR outputs (`save_asr_result_to_database()`, `get_asr_results()`)
- Schema: `create_evs_tables.sql` — SQLite at `data/evs_repository.db`
- Key tables: `asr_files`, `asr_results_segments`, `asr_results_words`, `si_analysis_results`, `si_error_details`, `si_cultural_issues`, `si_timing_issues`, `si_analysis_config`, `si_corrected_versions`, `users`, `login_history`, `email_metrics`, `email_metrics_summary`, `asr_config`
- Migration: `migrate_to_asr_results.py` — One-time migration script to add `asr_results_words` table schema

### LLM Clients (`clients/`)
- Factory pattern: `from clients import get_llm_client` — returns the active LLM client based on `llm_config.json`
- `ollama_client.py` — `OllamaClient` (singleton, supports `/api/generate` and `/api/chat`)
- `openai_client.py` — `OpenAIClient` (singleton, OpenAI-compatible API including vLLM)
- `llm_config.py` (top-level) — Programmatic LLM config management with `DEFAULT_LLM_CONFIG` and `DEFAULT_ANALYSIS_RULES`. Loads/saves `config/llm_config.json` with fallback defaults.

### ASR Processing
- `utils/asr_utils.py` — `ASRUtils` class supporting CrisperWhisper (English), FunASR (Chinese), Google Cloud, Tencent Cloud, IBM Watson
- `config/asr_language_config.py` — Separate provider/model recommendations per language
- `asr_config.py` (top-level) — Static dictionaries for ASR providers, languages, and per-provider config (older config predating `config/asr_language_config.py`)

### NLP & Analysis
- `chinese_nlp_unified.py` — `ChineseNLPUnified` class with `NLPEngine` enum (JIEBA or HANLP). Supports custom dictionaries via `config/chinese_custom_dict.txt`.
- `paraverbal_process.py` — EVS calculation and prosody analysis
- `concordance_utils.py` — Corpus analysis (word frequency, keywords, collocates, n-grams)
- `utils/si_analysis_integration.py` — SI quality analysis integration
- `utils/analysis_utils.py` — Time-based EN/ZH segment alignment via `create_time_based_segment_mapping()`. Pairs English and Chinese ASR segments by temporal overlap and midpoint proximity.
- `advanced_interpretation_analysis.py` — Optional advanced NLP module requiring `requirements_advanced.txt`. Provides semantic similarity, contextual coherence scoring, anaphora resolution, and cultural/pragmatic analysis. Gracefully degrades if dependencies missing.

### Email System
- `email_queue.py` — `EmailQueue` class for rate-limited, reliable email delivery. Background thread worker, configurable rate (default 20/min), retry logic (default 3 attempts), SQLite persistence. Provides `initialize_email_queue()` and `get_email_queue()`.
- `email_metrics.py` — `EmailMetrics` class for delivery monitoring. In-memory LRU cache (100 entries), thread-safe. Generates matplotlib charts (base64 PNG). Stores records in `email_metrics` and `email_metrics_summary` tables.

### User Management
- `user_utils.py` — `UserUtils` class with static methods for registration, login/logout, email verification (UUID tokens), password hashing (hashlib). Routes verification emails through `EmailQueue`. Reads SMTP/auth settings from `config.AUTH_CONFIG`.

### Reporting
- `report_generator.py` — `ReportGenerator` class for exporting analysis results as formatted Excel (openpyxl) and PDF (reportlab/kaleido) reports with embedded Plotly charts.
- `results_view.py` — Streamlit UI module for displaying ASR transcription results (file selector, segment display, metrics). Legacy module referencing older Whisper-specific methods.

### UI & Session
- `components/session_state.py` — `SessionState` data class, `initialize_session_state()`, and `clear_session_data()` for Streamlit session management
- `file_alias_manager.py` — Privacy-protected file name anonymization (SHA256 hash, generates `file_1`, `file_2` aliases)
- `file_display_utils.py` — Streamlit helper for resolving display names respecting privacy/aliasing settings (`get_file_display_name()`, `get_original_filename_from_display()`)
- `privacy_settings.py` — Streamlit sidebar UI for privacy controls and alias management
- `cache_utils.py` — `@st.cache_resource` decorators for DB connections and common queries
- `styles.py` — UI color constants (`COLOR_LEGEND` for EVS pair types), CSS inline styles, and HTML legend snippets for EVS and fluency annotations
- `evs_annotator/` — Custom Streamlit HTML component for EVS annotation (index.html + __init__.py)
- `word_aligner/` — Custom Streamlit HTML component for EN-ZH word alignment (index.html + __init__.py)

### Tools (`tools/`)
- `si-analysis-tools.py` — Google ADK (Agent Development Kit) tool collection for SI quality analysis. Includes `ExtractFileContentTool` (parses .json, .srt, .vtt, plain text ASR files). No `__init__.py` — standalone script, not a package.

### Multi-page Structure (`pages/`)
- `admin_panel.py` — Admin configuration panel (exported as `render_admin_panel`)
- `download_audio.py` — Audio download utility (exported as `render_download_audio_tab`)
- Both are callable render functions invoked from `app.py`

### Scripts (`scripts/`)
- `verify_env.py` — Environment verification utility
- `deploy.py` — Deployment automation script
- `install_python.bat` / `start.bat` — Windows setup and launch scripts

### Logging
- `logger_config.py` — `setup_logger(name)` function. Creates `./logs/` directory, attaches DEBUG-level rotating file handler (`logs/evs_app_YYYY-MM-DD.log`) and INFO-level console handler. Idempotent (safe on Streamlit reruns).

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

### Logging
```python
from logger_config import setup_logger
logger = setup_logger(__name__)
```

### Email
```python
from email_queue import get_email_queue
queue = get_email_queue()
```

## CI/CD

GitHub Actions workflow (`.github/workflows/python-app.yml`) runs on push/PR to `main`:
- Python 3.11, Ubuntu, with FFmpeg
- Lint: flake8 (syntax errors block, warnings don't)
- Test: pytest (with `GOOGLE_APPLICATION_CREDENTIALS` from repository secrets)
- Note: `requirements_advanced.txt` is not installed in CI

## Dependencies & Constraints

- **Python**: 3.11 (CI and setup.bat both use 3.11)
- **FFmpeg**: Required for audio processing
- **protobuf**: Must be `>=3.20,<4` for HanLP compatibility
- **numpy**: Must be `<2.0`
- **CUDA GPU**: Optional, needed for local CrisperWhisper (~4GB VRAM) and FunASR (~2GB VRAM)
- **Key packages**: streamlit, pandas, plotly, matplotlib, transformers, funasr, jieba, hanlp, openpyxl, reportlab, yt-dlp, tenacity
- **Optional (advanced analysis)**: sentence-transformers, spacy, scikit-learn (see `requirements_advanced.txt`)

## Environment Setup

- Google Cloud credentials: `export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"`
- Tencent Cloud: Create `keys/tencent_cloud_config.json` with `secret_id` and `secret_key`
- API keys go in `keys/` directory (gitignored)
- LLM config: Edit `config/llm_config.json` to switch between Ollama, OpenAI, or vLLM providers
- Docker: `Dockerfile` available (stub — base image and system deps only, needs completion)
- Logs are written to `logs/` directory (gitignored)
- Database stored at `data/evs_repository.db` (gitignored)

## File Organization Notes

- Top-level Python files are a mix of core modules and utility scripts. The app is monolithic (`app.py`), with supporting modules at the top level rather than in packages.
- `config/` is the only well-structured package with a proper `__init__.py` re-exporting all symbols.
- `clients/`, `components/`, and `pages/` are small packages with 2-3 modules each.
- `utils/` has no `__init__.py` — import directly: `from utils.asr_utils import ASRUtils`
- `tools/` has no `__init__.py` — standalone script, not importable as a package.
- Gitignored directories: `keys/`, `logs/`, `models/`, `data/*.db`, `backup/`, `.claude/`
