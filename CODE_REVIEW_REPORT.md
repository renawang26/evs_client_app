# EVS Navigation System — Code Review Report

**Date**: 2026-02-19
**Scope**: Full project review by 5 parallel specialist agents
**Reviewed by**: Security · Database · LLM/ASR · UI/Components · Architecture

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Critical Issues](#critical-issues)
3. [Security Review](#security-review)
4. [Database Layer Review](#database-layer-review)
5. [LLM & ASR Integration Review](#llm--asr-integration-review)
6. [UI & Component Review](#ui--component-review)
7. [Architecture & app.py Review](#architecture--apppy-review)
8. [Refactoring Roadmap](#refactoring-roadmap)
9. [Finding Index](#finding-index)

---

## Executive Summary

The EVS Navigation System is a functional research tool with a solid feature set, but it has grown into a 9,500-line monolith with several security vulnerabilities that must be addressed before any shared or production deployment. The five review areas each surfaced distinct concerns; collectively they tell a consistent story: **the application was built quickly for local research use, but its configuration and defaults are unsafe for any networked deployment.**

### Severity Counts

| Category | Critical | High | Medium | Low | Total |
|----------|----------|------|--------|-----|-------|
| Security | 1 | 5 | 6 | 4 | 16 |
| Database | 1 | 5 | 7 | 5 | 18 |
| LLM / ASR | 1 | 4 | 5 | 4 | 14 |
| UI / Components | 3 | 4 | 4 | 4 | 15 |
| Architecture | 2 | 5 | 7 | 3 | 17 |
| **Total** | **8** | **23** | **29** | **20** | **80** |

### Top Priorities Before Any Networked Deployment

1. **Disable auth bypass** — `BYPASS_LOGIN: True` in `config/config.py` exposes the entire app with no authentication
2. **Replace SHA-256 password hashing** with bcrypt or argon2
3. **Remove the hardcoded Windows credential path** in `app.py:86`
4. **Add login rate limiting** — no brute-force protection exists
5. **Move SMTP and LLM API keys out of source code** to environment variables
6. **Fix the broken singleton context-manager pattern** in `db_utils.py` that silently closes the shared DB connection
7. **Fix the `ReferenceError` in `evs_annotator`** that breaks the "Clear All" button

---

## Critical Issues

These issues represent broken behavior or security vulnerabilities that must be fixed before any non-local deployment.

### CRIT-1 — Authentication Bypass Enabled by Default
**Area**: Security · `config/config.py:8`

`BYPASS_LOGIN: True` completely disables authentication. Any user can access the application as admin without credentials. This is not a development-only risk: a researcher who runs `streamlit run app.py` on a server reachable over a network exposes all user data and admin functions.

**Fix**: Set `BYPASS_LOGIN: False`. Control this via environment variable:
```python
BYPASS_LOGIN: os.environ.get("BYPASS_LOGIN", "false").lower() == "true"
```

---

### CRIT-2 — Hardcoded Windows Absolute Path Overwrites Credentials at Startup
**Area**: Architecture · `app.py:86`

```python
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/myap/PhD_Project/key/google_credentials.json"
```
This line runs unconditionally on every startup on every machine, clobbering any properly configured credential path in the environment.

**Fix**: Remove this line entirely. Use `os.environ.setdefault()` with a relative path under `keys/` if a default is needed.

---

### CRIT-3 — Singleton DB Connection Destroyed by Context Manager
**Area**: Database · `db_utils.py:95, 126, 169` (~15 call sites)

`EVSDataUtils.get_db_connection()` returns the `DBManager` singleton connection. Every `with EVSDataUtils.get_db_connection() as conn:` block calls `conn.__exit__()`, which closes the underlying `sqlite3.Connection` permanently. All subsequent database operations in the session fail silently or raise `ProgrammingError: Cannot operate on a closed database`.

**Fix**: Either (a) have `get_db_connection()` create a short-lived connection rather than returning the singleton, or (b) wrap the singleton in a proxy that overrides `__exit__` to commit/rollback without calling `close()`.

---

### CRIT-4 — Runtime ReferenceError in EVS Annotator
**Area**: UI · `evs_annotator/index.html:516`

`onClearSelections()` calls `Streamlit.setComponentValue({...})` but `Streamlit` is never defined in this component — it uses a custom `sendToStreamlit()` wrapper. This throws a `ReferenceError` at runtime, silently breaking all "Clear All" annotation deletion.

**Fix**: Replace `Streamlit.setComponentValue({...})` with `sendToStreamlit({...})`.

---

### CRIT-5 — SQL Injection in `cleanup_old_aliases()`
**Area**: UI · `file_alias_manager.py:313–314`

```python
"AND last_accessed < datetime('now', '-{} days')".format(days_old)
```
`days_old` is interpolated directly via `.format()`. Though currently sourced from a Streamlit slider, this pattern is fragile and injectable if called programmatically.

**Fix**: Use parameterized binding: `datetime('now', ? || ' days')` with `('-' + str(int(days_old)),)`.

---

### CRIT-6 — Wildcard `postMessage` targetOrigin in Both Custom Components
**Area**: UI · `evs_annotator/index.html:239`, `word_aligner/index.html:175`

Both components send all annotation data via `window.parent.postMessage(outData, "*")`. Any origin can receive the data if the app is embedded in an iframe.

**Fix**: Use `window.parent.postMessage(outData, window.location.origin)`.

---

### CRIT-7 — Non-Thread-Safe LLM Client Singleton
**Area**: LLM/ASR · `clients/ollama_client.py:25`, `clients/openai_client.py:24`

`get_instance()` uses `if cls._instance is None: cls._instance = cls()` with no threading lock. Streamlit handles multiple users concurrently; a TOCTOU race can create duplicate client instances with separate connection pools, causing pool exhaustion and inconsistent config.

**Fix**: Add a `threading.Lock()` class variable and wrap the check+assign:
```python
_lock = threading.Lock()
with cls._lock:
    if cls._instance is None:
        cls._instance = cls()
```

---

### CRIT-8 — Direct SQLite Access Bypassing DB Abstraction Layer
**Area**: Architecture · `app.py:26, 209–210, 2259, 2726, 9264, 9391`

Multiple render functions import `sqlite3` and run raw SQL directly, bypassing `DBManager` and `EVSDataUtils`. Notably `render_login_page` (line 209) performs a cursor query for email verification status inside UI rendering code.

**Fix**: Encapsulate each raw SQL call in a named method on `EVSDataUtils` or `UserUtils`. Render functions should call named methods, not bare SQL.

---

## Security Review

*Reviewer*: security-reviewer | *Files*: `user_utils.py`, `email_queue.py`, `db_manager.py`, `db_utils.py`, `config/config.py`

### High

**[SEC-H1] Unsalted SHA-256 Password Hashing**
- **Location**: `user_utils.py:91–92`
- **Issue**: `hashlib.sha256(password.encode()).hexdigest()` uses no salt. Rainbow table attacks can crack all passwords at once; identical passwords produce identical hashes across accounts.
- **Fix**: Replace with `bcrypt` (`pip install bcrypt`) or `argon2-cffi`. At minimum use `hashlib.pbkdf2_hmac` with a per-user random salt stored in the `users` table.

**[SEC-H2] No Login Rate Limiting or Account Lockout**
- **Location**: `user_utils.py:367–411`
- **Issue**: Login attempts are logged to `login_history` but never throttled. An attacker can make unlimited password guesses.
- **Fix**: After N consecutive failures for an email (e.g., 5), lock the account for 15 minutes using the existing `login_history` table.

**[SEC-H3] SMTP Credentials Stored as String Literals in Source**
- **Location**: `config/config.py:30–35`
- **Issue**: SMTP username, password, and server are hardcoded strings. Even as placeholders, this pattern leads developers to replace them in-source.
- **Fix**: Load from environment variables: `os.environ.get("SMTP_PASSWORD")`.

**[SEC-H4] Verification Token Persisted in `email_queue` Table**
- **Location**: `user_utils.py:171–173`
- **Issue**: The `verification_token` is stored inside the `metadata` dict in the `email_queue` table — a second storage location alongside `users`. A DB leak exposes tokens without needing to crack passwords.
- **Fix**: Remove the token from queue metadata. It already exists in `users`; the duplicate is unnecessary.

**[SEC-H5] User Email Reflected into HTML Email Body Without Escaping**
- **Location**: `user_utils.py:146–160`
- **Issue**: `email` and `verification_token` are interpolated into an HTML email body via f-string with no `html.escape()`. A crafted registration email address can inject HTML into the email body.
- **Fix**: `html.escape(email)` and `html.escape(verification_token)` before interpolation.

### Medium

**[SEC-M1] Dynamic SQL Field Concatenation in `email_queue.py`**
- **Location**: `email_queue.py:219–221`
- **Issue**: SET clause built via `", ".join(...)` over field name strings concatenated into SQL. Field names are programmer-controlled here but the pattern is fragile.
- **Fix**: Use a fixed SQL statement per status transition or validate field names against a whitelist.

**[SEC-M2] Password Complexity Requirements Too Weak**
- **Location**: `config/config.py:43–47`
- **Issue**: `MIN_LENGTH: 6`, `REQUIRE_SPECIAL_CHAR: False`, `REQUIRE_NUMBER: False`. Below OWASP recommendations (12+ chars minimum).
- **Fix**: `MIN_LENGTH: 12`, enable number and special char requirements.

**[SEC-M3] Email Verification Disabled by Default**
- **Location**: `config/config.py:17`
- **Issue**: `ENABLE_EMAIL_VERIFICATION: False` combined with `BYPASS_LOGIN: True` means all auth is disabled out of the box.
- **Fix**: Both flags should be secure by default; use environment variables to loosen for local dev.

**[SEC-M4] Shared SQLite Connection Across Threads Without Mutex**
- **Location**: `db_manager.py:42`
- **Issue**: Single `sqlite3.Connection` shared with `check_same_thread=False` but no `threading.Lock` around execution.
- **Fix**: Add a `threading.RLock` around `execute_query` / `get_connection`, or use WAL mode with per-thread connections.

**[SEC-M5] `BASE_URL` Hardcoded to `localhost`**
- **Location**: `config/config.py:68`
- **Issue**: Verification emails contain links to `http://localhost:8501`. In any networked deployment, links are broken.
- **Fix**: `BASE_URL = os.environ.get("APP_BASE_URL", "http://localhost:8501")`.

**[SEC-M6] Internal DB Error Details Logged to Login History**
- **Location**: `user_utils.py:407`
- **Issue**: `record_login_attempt(email, f"ERROR: {str(e)[:50]}", ...)` logs exception text that may reveal schema details.
- **Fix**: Log only `"FAILED_DB_ERROR"` to login history; send full details to the application log.

### Low

**[SEC-L1] `keys/` Directory Absent — No Default Structure**
- **Location**: `.gitignore`
- **Fix**: Add `keys/.gitkeep` with a README noting what belongs there.

**[SEC-L2] Naive Datetime Comparison for Token Expiry**
- **Location**: `user_utils.py:344`
- **Fix**: Use `datetime.utcnow()` consistently throughout; store all timestamps in UTC.

**[SEC-L3] Full Email Bodies with Active Tokens Stored in DB**
- **Location**: `email_queue.py:113–128`
- **Fix**: Store only metadata; clear queue records immediately after delivery.

**[SEC-L4] No CSRF Protection on Streamlit Sessions**
- **Location**: General (`app.py`)
- **Note**: Streamlit framework limitation. For sensitive admin actions, add server-side nonce validation.

---

## Database Layer Review

*Reviewer*: db-reviewer | *Files*: `db_manager.py`, `db_utils.py`, `save_asr_results.py`, `create_evs_tables.sql`, `init_database.py`

*(CRIT-3 — singleton closed by context manager — is the critical finding for this area; see Critical Issues above.)*

### High

**[DB-H1] `save_asr_results.py` Creates Its Own Connection, Bypassing DBManager**
- **Location**: `save_asr_results.py:23–31`
- **Issue**: Defines its own `get_db_connection()` via `sqlite3.connect()` with no `row_factory`, creating a third connection path. Changes may not be visible across connections; lock contention increases.
- **Fix**: Remove `save_asr_results.py:get_db_connection()` and import from `db_manager`.

**[DB-H2] Rollback in `except` Block Is Unreachable**
- **Location**: `save_asr_results.py:131–132`
- **Issue**: `conn` is defined inside the `with` block; the outer `except`'s `'conn' in locals()` is always `False`. Failed transactions are never explicitly rolled back.
- **Fix**: Move rollback inside the `with` block or use explicit `try/except/finally`.

**[DB-H3] Missing Indexes on Primary Query Columns**
- **Location**: `create_evs_tables.sql:52–90`
- **Issue**: `asr_results_words` and `asr_results_segments` are queried by `(file_name, lang, asr_provider)` on every fetch but have no indexes on these columns.
- **Fix**:
  ```sql
  CREATE INDEX idx_words_file_lang_provider ON asr_results_words(file_name, lang, asr_provider);
  CREATE INDEX idx_segs_file_lang_provider ON asr_results_segments(file_name, lang, asr_provider);
  ```

**[DB-H4] Extra SELECT Round-Trip After Upsert**
- **Location**: `db_utils.py:128–151`
- **Issue**: After `INSERT ... ON CONFLICT DO UPDATE`, a second `SELECT id FROM asr_files WHERE ...` retrieves the row ID. `cursor.lastrowid` is available directly.
- **Fix**: Replace the SELECT with `cursor.lastrowid` after the upsert.

**[DB-H5] `register_asr_file` Called Outside the Main Save Transaction**
- **Location**: `save_asr_results.py:101–119`
- **Issue**: Words/segments are saved and committed; then `register_asr_file()` is called after the `with` block. A failure here leaves `asr_files` out of sync with the data tables.
- **Fix**: Include `register_asr_file` inside the same transaction.

### Medium

**[DB-M1] `ensure_asr_files_table()` Called on Every Operation**
- **Location**: `db_utils.py:125, 168, 195, 212, 244`
- **Issue**: Issues `CREATE TABLE IF NOT EXISTS` + 2 index statements before every query — redundant on every UI interaction.
- **Fix**: Call once during `DBManager.__init__` or application startup.

**[DB-M2] `get_file_stats()` Issues 3 Queries Where 1 Would Do**
- **Location**: `db_utils.py:518–593`
- **Fix**: Combine into a single query using subquery SELECTs.

**[DB-M3] `get_evs()` Queries a Non-Existent Table**
- **Location**: `db_utils.py:294–305`
- **Issue**: Queries `pd_interpret_words` which is not in `create_evs_tables.sql`. Any call raises `OperationalError: no such table`.
- **Fix**: Remove or update this method to use current schema tables.

**[DB-M4] DataFrame Used for a Scalar `COUNT(*)` Result**
- **Location**: `db_utils.py:608–619`
- **Fix**: Use `conn.execute(query, params).fetchone()[0]` directly.

**[DB-M5] `sqlite_master` Existence Check on Every Delete**
- **Location**: `db_utils.py:654–662`
- **Issue**: Queries `sqlite_master` before deleting from `chinese_nlp_results`, indicating schema drift.
- **Fix**: Define `chinese_nlp_results` in the schema so it always exists; remove the check.

**[DB-M6] Schema Initialization Commented Out in DBManager**
- **Location**: `db_manager.py:37`
- **Issue**: `# self._initialize_tables_from_sql()` — if `init_database.py` has not been run, the app fails silently on missing tables with no clear error.
- **Fix**: Re-enable or add a startup validation that checks required tables exist.

**[DB-M7] `pandas.to_sql()` Inside a Manually-Opened Transaction**
- **Location**: `save_asr_results.py:85–94`
- **Issue**: `to_sql` is called after `conn.execute("BEGIN TRANSACTION")`. `to_sql` may issue its own transaction management, causing undefined interaction.
- **Fix**: Let `to_sql` manage its own transactions, or use chunked manual inserts within an explicit context.

### Low

**[DB-L1] Timestamp in `UNIQUE` Constraint on `si_analysis_results`**
- **Location**: `create_evs_tables.sql:145`
- **Fix**: Remove `analysis_timestamp` from the unique constraint; use `UNIQUE(file_name, asr_provider, analysis_type)`.

**[DB-L2] No `ON DELETE CASCADE` and Foreign Keys Not Enforced**
- **Location**: `create_evs_tables.sql:162, 176, 190, 219`
- **Fix**: Add `ON DELETE CASCADE` to child table FKs; add `PRAGMA foreign_keys = ON` to `DBManager._initialize_connection`.

**[DB-L3] Streamlit Imported in the Database Layer**
- **Location**: `save_asr_results.py:5, 54, 130`
- **Fix**: Raise exceptions instead of calling `st.error()`. Let UI code handle display.

**[DB-L4] `asr_config` Table Uses `TEXT` for Timestamp Columns**
- **Location**: `create_evs_tables.sql:283–284`
- **Fix**: Change to `TIMESTAMP DEFAULT CURRENT_TIMESTAMP` for consistency.

**[DB-L5] Bare `except:` Clauses**
- **Location**: `db_utils.py` (multiple, in `_convert_slice_duration`)
- **Fix**: Use `except (struct.error, ValueError):` specifically.

---

## LLM & ASR Integration Review

*Reviewer*: llm-reviewer | *Files*: `clients/`, `utils/asr_utils.py`, `llm_config.py`, `config/llm_config.json`, `asr_config.py`

*(CRIT-7 — non-thread-safe singleton — is the critical finding for this area; see Critical Issues above.)*

### High

**[LLM-H1] No Retry Logic or Exponential Backoff**
- **Location**: `clients/ollama_client.py:51`, `clients/openai_client.py:68`
- **Issue**: A single network failure permanently drops the user's work. No retry for transient errors (429, 503, timeout, connect error).
- **Fix**: Implement 3-retry exponential backoff (1s/2s/4s) for `httpx.TimeoutException`, `httpx.ConnectError`, HTTP 429, and HTTP 503.

**[LLM-H2] Prompt Injection via Unsanitized Transcription Content**
- **Location**: `app.py:7591–7594, 7812, 7819`
- **Issue**: ASR-derived word content is interpolated directly into LLM prompts via f-strings. A crafted audio recording can inject instructions into the prompt.
- **Fix**: Wrap user-derived content in explicit delimiters (`<word>...</word>`) in the prompt template; add a system instruction forbidding overrides; cap content length before passing to LLM.

**[LLM-H3] LLM API Key in Plaintext Config File Not in `.gitignore`**
- **Location**: `config/llm_config.json:13`
- **Issue**: `"llm_api_key": "sk-xxxxxxx"` in a tracked JSON file. Developers will replace the placeholder in-source.
- **Fix**: Load from `os.environ.get("OPENAI_API_KEY")`; add `config/llm_config.json` to `.gitignore`; provide `config/llm_config.example.json`.

**[LLM-H4] vLLM Provider Silently Falls Back to Ollama Client**
- **Location**: `clients/__init__.py:24–27`
- **Issue**: The factory only handles `"openai"`, defaulting everything else (including `"vllm"`) to `OllamaClient`. vLLM uses an OpenAI-compatible API, so the Ollama client will fail against it.
- **Fix**: Add `elif active_provider == "vllm": return OpenAIClient.get_instance()`.

### Medium

**[LLM-M1] Tencent Polling Timeout Too Short for Long Audio**
- **Location**: `utils/asr_utils.py:510–511`
- **Issue**: `max_retries=10` × `retry_delay=2s` = 20s timeout. Long recordings (30+ min) can take several minutes to process.
- **Fix**: Increase to 60+ retries or derive from audio duration. Add a clear "timed out" error distinct from "task failed."

**[LLM-M2] Language Detection Model Cache Not Thread-Safe**
- **Location**: `utils/asr_utils.py:652`
- **Issue**: `_get_language_detect_model()` accesses `_MODEL_CACHE` without acquiring `_MODEL_CACHE_LOCK`, unlike sibling model loaders.
- **Fix**: Wrap `_MODEL_CACHE` access inside `_MODEL_CACHE_LOCK`.

**[LLM-M3] IBM Watson Service URL Is a Hardcoded Placeholder**
- **Location**: `utils/asr_utils.py:290`
- **Issue**: `set_service_url('...watson.cloud.ibm.com/instances/your-instance-id')` — the function is broken by default regardless of API key.
- **Fix**: Remove `process_audio_file_with_ibm()` or accept `service_url` as a parameter.

**[LLM-M4] No GPU Memory Check Before Model Loading**
- **Location**: `utils/asr_utils.py:235, 144`
- **Issue**: CrisperWhisper (~3GB) and FunASR (~2GB) are loaded without checking available VRAM. OOM errors produce cryptic CUDA exceptions.
- **Fix**: Call `get_gpu_info()` before loading; warn the user and fall back to CPU if VRAM is insufficient.

**[LLM-M5] `llm_config.json` Read from Disk on Every Request**
- **Location**: `clients/__init__.py:19–22`
- **Issue**: `load_llm_config()` parses the JSON file on every `get_llm_client()` call.
- **Fix**: Cache with `@st.cache_resource` or a module-level variable; re-read only when the file modification time changes.

### Low

**[LLM-L1] `traceback` Imported but Unused in Both Clients**
- **Location**: `clients/ollama_client.py:10`, `clients/openai_client.py:9`
- **Fix**: Remove the dead imports.

**[LLM-L2] `set_language_asr_config` Mutates Module-Level Dict**
- **Location**: `config/asr_language_config.py:238`
- **Issue**: One user's ASR config selection mutates a shared module-level dict, affecting other sessions.
- **Fix**: Store per-session ASR config in Streamlit session state.

**[LLM-L3] Relative `CONFIG_FILE` Path Breaks When CWD ≠ Project Root**
- **Location**: `llm_config.py:111`
- **Fix**: Use `Path(__file__).parent / "llm_config.json"` anchored to the config package directory.

**[LLM-L4] Sensitive Tencent API Responses Logged at INFO Level**
- **Location**: `utils/asr_utils.py:495, 530`
- **Fix**: Downgrade to `logger.debug()` or log only `TaskId` and `Status` fields.

---

## UI & Component Review

*Reviewer*: ui-reviewer | *Files*: `styles.py`, `file_alias_manager.py`, `file_display_utils.py`, `privacy_settings.py`, `cache_utils.py`, `evs_annotator/`, `word_aligner/`

*(CRIT-4, CRIT-5, CRIT-6 are the critical findings for this area; see Critical Issues above.)*

### High

**[UI-H1] `st.expander()` Used in `if` Statement — Always Renders**
- **Location**: `privacy_settings.py:34`
- **Issue**: `if st.expander(...):` — expanders are always truthy; `render_alias_management()` is called regardless of whether the user opens the expander.
- **Fix**: `with st.expander("📂 File Alias Management", expanded=False): render_alias_management()`.

**[UI-H2] Unguarded `st.session_state.file_alias_manager` Access**
- **Location**: `privacy_settings.py:39, 93, 130, 152, 220, 278, 301, 370`
- **Issue**: Direct attribute access without existence check. Raises `AttributeError` if session state is not initialized.
- **Fix**: Use `st.session_state.get('file_alias_manager')` with a guard, or ensure `initialize_session_state()` runs on all entry paths.

**[UI-H3] No `postMessage` Origin Validation in Inbound Handlers**
- **Location**: `evs_annotator/index.html:592–643`, `word_aligner/index.html:373–388`
- **Issue**: Message handlers check only `event.data.type === "streamlit:render"`, never `event.origin`. Forged render events from a malicious parent can inject word data.
- **Fix**: Capture the Streamlit origin from the first render event and reject messages from other origins.

**[UI-H4] Race Condition in Alias Name Generation**
- **Location**: `file_alias_manager.py:71–90`
- **Issue**: COUNT query and INSERT happen in separate connections, creating a window where concurrent requests generate the same alias, causing `UNIQUE` constraint failures.
- **Fix**: Use `DBManager.get_instance()` singleton, wrap count+insert in a single transaction, use `INSERT OR IGNORE` with a timestamp-based fallback.

### Medium

**[UI-M1] Duplicate Display-Name Logic (DRY Violation)**
- **Location**: `file_display_utils.py:16–42` and `privacy_settings.py:112–131`
- **Issue**: `get_file_display_name()` and `get_display_filename()` are functionally identical in two modules.
- **Fix**: Consolidate in `file_display_utils.py`; remove the duplicate from `privacy_settings.py`.

**[UI-M2] `@st.cache_resource` DB Connection Used by `@st.cache_data` Functions**
- **Location**: `cache_utils.py:22–34, 45–66`
- **Issue**: Shared cached connection passed to `@st.cache_data` functions. On server restart, cached functions will fail on a closed connection. `cache_utils.py` also maintains its own `DB_PATH` constant duplicating `config/database_config.py`.
- **Fix**: Use `DBManager.get_instance()` inside cache functions; import `DB_PATH` from `config/database_config.py`.

**[UI-M3] Silent Exception Swallowing in Display Utilities**
- **Location**: `file_display_utils.py:40, 65`
- **Fix**: Add `logger.warning(f"Alias lookup failed: {e}", exc_info=True)` before the fallback return.

**[UI-M4] `st.button()` Called With Invalid `width='stretch'` Parameter**
- **Location**: `file_display_utils.py:205`
- **Fix**: Remove `width='stretch'`; this argument is silently ignored.

### Low

**[UI-L1] Malformed CSS in `COLOR_LEGEND`**
- **Location**: `styles.py:32–39`
- **Issue**: `padding: 5px; 10px 5px 10px;` (semicolon mid-shorthand) and `color:C4E1FF` (missing `#`).
- **Fix**: `padding: 5px 10px;` and `color: #C4E1FF`.

**[UI-L2] Original Filenames Written to Application Logs**
- **Location**: `file_alias_manager.py:145, 262, 289`
- **Issue**: Log lines include the original (potentially PII) filename, defeating the privacy alias system.
- **Fix**: Log only the alias and file hash, never the original filename.

**[UI-L3] Privacy UI Makes Inaccurate Isolation Claim**
- **Location**: `privacy_settings.py:199`
- **Issue**: "Only you can see the mapping" is false — all users share the same SQLite DB; admins can see all mappings.
- **Fix**: Update copy to accurately describe the isolation model.

**[UI-L4] `invalidate_cache_on_write` Clears All Caches**
- **Location**: `cache_utils.py:248–259`
- **Issue**: Calls `clear_cache('all')` after any write, clearing all `@st.cache_data` caches system-wide unnecessarily.
- **Fix**: Clear only the relevant cache subset (e.g., `clear_cache('files')` or `clear_cache('asr_data')`).

---

## Architecture & app.py Review

*Reviewer*: arch-reviewer | *Files*: `app.py`, `main.py`, `components/session_state.py`, `config/`

*(CRIT-2 and CRIT-8 are the critical findings for this area; see Critical Issues above.)*

### High

**[ARCH-H1] Monolith — 9,579-Line `app.py` With No Enforced Boundaries**
- **Location**: `app.py` (entire file)
- **Issue**: Contains auth logic (~123–263), an 1,800-line admin panel (~265–2010), session-state management (~2012–2132), NLP business logic (~2133–2321), five tab renderers (~2322–4824), and SI/LLM integration (~6002–9579). Functions are defined after `main()`, making reading non-linear.
- **Fix**: See Refactoring Roadmap below.

**[ARCH-H2] `SessionState` Defined in Two Places — `components/` Version Is Dead Code**
- **Location**: `app.py:2012–2034`, `components/session_state.py:11–41`
- **Issue**: Two diverging implementations of the same class. `app.py` never imports from `components/session_state.py`, making the component a dead module.
- **Fix**: Delete duplicates from `app.py`; add `from components.session_state import SessionState, initialize_session_state, clear_session_data`.

**[ARCH-H3] Business Logic Entangled in UI Render Functions**
- **Location**: `app.py:2133, 3397, 6192, 8302` and others
- **Issue**: Functions like `perform_si_analysis`, `analyze_translation_accuracy`, `analyze_discourse_structure` perform pure computation but call `st.progress()`, `st.write()`, `st.empty()`, mixing output with computation.
- **Fix**: Move computation to `services/` modules. Render functions receive results and display them.

**[ARCH-H4] Parallel ASR Config Systems — Root-Level `asr_config.py` and `config/asr_language_config.py`**
- **Location**: `asr_config.py` (root), `app.py:100–106`, `config/asr_language_config.py`
- **Issue**: Three places define or imply ASR providers; CLAUDE.md itself warns the root config is "older" and should be avoided.
- **Fix**: Consolidate into `config/asr_language_config.py`; remove `asr_config.py` and the inline dict from `app.py`.

**[ARCH-H5] `render_login_page` Runs Raw DB Queries**
- **Location**: `app.py:209–213`
- **Issue**: UI render function directly calls `UserUtils.get_db_connection()` and executes raw SQL for email verification status.
- **Fix**: Add `UserUtils.get_email_verification_status(email) -> bool` and call that instead.

### Medium

**[ARCH-M1] Duplicate `plotly.graph_objects` Import**
- **Location**: `app.py:5, 27`
- **Fix**: Remove the duplicate at line 27.

**[ARCH-M2] Mixed Chinese/English Comments and Inline UI Strings**
- **Location**: `app.py:108, 122, 248, 264` and throughout
- **Fix**: Standardize code comments to English; use a localization mechanism for Chinese UI strings.

**[ARCH-M3] Dead Commented-Out Code Blocks in `main()`**
- **Location**: `app.py:4864–4869, 4880, 120`
- **Fix**: Delete commented-out blocks; they are preserved in git history.

**[ARCH-M4] Deferred Imports Inside `main()` and Conditional Blocks**
- **Location**: `app.py:4842–4843, 4851–4852, 4884`
- **Fix**: Move all imports to module top-level; use session state guards only for initialization logic.

**[ARCH-M5] `set_file_aliasing_enabled` Mutates Module-Level Global**
- **Location**: `config/display_config.py:33–42`
- **Issue**: `global ENABLE_FILE_ALIASING` mutation at runtime causes cross-session interference in multi-user deployments.
- **Fix**: Store in `st.session_state` or persist to a config file and reload.

**[ARCH-M6] `main.py` Name Causes Entry-Point Confusion**
- **Location**: `main.py`
- **Fix**: Rename to `demo_word_aligner.py` or move to `demos/`.

**[ARCH-M7] Indentation Error — Potentially Unreachable Code in `main()`**
- **Location**: `app.py:4876–4877`
- **Issue**: `st.title(...)` indentation makes it appear inside an `if/return` block.
- **Fix**: Verify and correct indentation; confirm the title renders on every code path.

### Low

**[ARCH-L1] `torch` Imported at Module Load Always**
- **Location**: `app.py:13`
- **Fix**: Lazy-import torch inside the ASR provider initialization path to reduce cold startup time.

**[ARCH-L2] `REPOSITORY_ROOT` Alias Points to Data Directory, Not Project Root**
- **Location**: `config/database_config.py:68`
- **Fix**: Migrate callers to `DATA_DIR` or `PROJECT_ROOT`; remove the misleading alias.

**[ARCH-L3] `components/session_state.py` Is Dead Code**
- **Location**: `components/session_state.py`
- **Fix**: Verify no imports with `grep -r "from components.session_state"`. If unused, delete it (or make it canonical per ARCH-H2).

---

## Refactoring Roadmap

Ordered by impact vs. effort, based on the architecture review. Each step reduces `app.py` line count and improves testability.

### Step 1 — Fix Critical Regressions First (1–2 days)
- Fix CRIT-3 (DB singleton context manager) — data loss risk
- Fix CRIT-4 (evs_annotator ReferenceError) — broken feature
- Fix CRIT-1 (auth bypass default) — security
- Fix CRIT-2 (Windows path overwrite) — portability

### Step 2 — Extract Auth (~140 lines, quick win)
Pull `check_authentication()`, `render_login_page()`, `render_user_menu()` (lines 123–263) into `auth/auth_ui.py`. Add `UserUtils.get_email_verification_status()` to eliminate the raw SQL call from the UI.

### Step 3 — Delete `SessionState` Duplicate and Wire `components/`
Remove the duplicate class from `app.py` and add the import. Zero functionality change; ~40 lines removed.

### Step 4 — Extract Admin Panel (~1,750 lines)
Move `render_admin_panel()` and its sub-panels to `pages/admin_panel.py` (a pages module already exists). This is the single highest line-count reduction available.

### Step 5 — Extract SI Analysis Service Layer (~3,500 lines)
Move `perform_si_analysis`, `analyze_*`, `calculate_advanced_performance_metrics`, `generate_improvement_recommendations` to `services/si_analysis.py`. Render functions become thin callers. This also eliminates `st.*` calls from computation logic.

### Step 6 — Extract LLM Pairing Logic (~700 lines)
Move `create_llm_based_pairs`, `parse_llm_pairing_response`, `call_ollama_llm`, `call_openai_llm`, `improve_evs_analysis_with_llm` to `services/llm_pairing.py`.

### Step 7 — Extract EVS Calculation Utilities (~400 lines)
Move `get_evs_data`, `calculate_evs`, `slice_audio`, `update_pairs_from_selections` to `services/evs_calculator.py` with no Streamlit dependency — enabling unit testing.

### Step 8 — Consolidate Config (~low effort, high clarity)
- Remove root-level `asr_config.py`; migrate to `config/asr_language_config.py`
- Remove inline `ASR_PROVIDERS` dict from `app.py`
- Add `config/llm_config.json` to `.gitignore`; create `.example` version
- Rename `main.py` → `demo_word_aligner.py`

### Target State
After these refactors, `app.py` should shrink to ~1,500–2,000 lines of pure tab orchestration. All data access goes through `EVSDataUtils`/`DBManager`. Computation logic is independently testable. Security defaults are safe for deployment.

---

## Finding Index

| ID | Severity | Area | File | Summary |
|----|----------|------|------|---------|
| CRIT-1 | Critical | Security | config/config.py:8 | Auth bypass enabled by default |
| CRIT-2 | Critical | Architecture | app.py:86 | Hardcoded Windows path overwrites credentials |
| CRIT-3 | Critical | Database | db_utils.py:95+ | Singleton connection closed by context manager |
| CRIT-4 | Critical | UI | evs_annotator/index.html:516 | ReferenceError breaks Clear All button |
| CRIT-5 | Critical | UI | file_alias_manager.py:313 | SQL injection in cleanup_old_aliases |
| CRIT-6 | Critical | UI | evs_annotator + word_aligner | Wildcard postMessage targetOrigin |
| CRIT-7 | Critical | LLM/ASR | clients/ollama_client.py:25 | Non-thread-safe LLM client singleton |
| CRIT-8 | Critical | Architecture | app.py:209+ | Raw SQLite in UI render functions |
| SEC-H1 | High | Security | user_utils.py:91 | Unsalted SHA-256 password hashing |
| SEC-H2 | High | Security | user_utils.py:367 | No login rate limiting |
| SEC-H3 | High | Security | config/config.py:30 | SMTP credentials in source |
| SEC-H4 | High | Security | user_utils.py:171 | Token persisted in email_queue DB |
| SEC-H5 | High | Security | user_utils.py:146 | Email reflected in HTML without escaping |
| DB-H1 | High | Database | save_asr_results.py:23 | Split connection bypasses DBManager |
| DB-H2 | High | Database | save_asr_results.py:131 | Rollback unreachable |
| DB-H3 | High | Database | create_evs_tables.sql:52 | Missing indexes on search columns |
| DB-H4 | High | Database | db_utils.py:128 | Extra SELECT round-trip after upsert |
| DB-H5 | High | Database | save_asr_results.py:101 | register_asr_file outside transaction |
| LLM-H1 | High | LLM/ASR | clients/ollama_client.py:51 | No retry / backoff |
| LLM-H2 | High | LLM/ASR | app.py:7591 | Prompt injection via ASR content |
| LLM-H3 | High | LLM/ASR | config/llm_config.json:13 | API key in untracked JSON |
| LLM-H4 | High | LLM/ASR | clients/__init__.py:24 | vLLM falls back to wrong client |
| UI-H1 | High | UI | privacy_settings.py:34 | st.expander() in if — always renders |
| UI-H2 | High | UI | privacy_settings.py:39+ | Unguarded session state access |
| UI-H3 | High | UI | evs_annotator + word_aligner | No postMessage origin validation |
| UI-H4 | High | UI | file_alias_manager.py:71 | Race condition in alias generation |
| ARCH-H1 | High | Architecture | app.py | 9,579-line monolith |
| ARCH-H2 | High | Architecture | app.py:2012 | SessionState defined twice |
| ARCH-H3 | High | Architecture | app.py:2133+ | Business logic in render functions |
| ARCH-H4 | High | Architecture | asr_config.py | Parallel ASR config systems |
| ARCH-H5 | High | Architecture | app.py:209 | DB queries in render_login_page |
| SEC-M1–M6 | Medium | Security | various | See Security section |
| DB-M1–M7 | Medium | Database | various | See Database section |
| LLM-M1–M5 | Medium | LLM/ASR | various | See LLM/ASR section |
| UI-M1–M4 | Medium | UI | various | See UI section |
| ARCH-M1–M7 | Medium | Architecture | various | See Architecture section |
| SEC-L1–L4 | Low | Security | various | See Security section |
| DB-L1–L5 | Low | Database | various | See Database section |
| LLM-L1–L4 | Low | LLM/ASR | various | See LLM/ASR section |
| UI-L1–L4 | Low | UI | various | See UI section |
| ARCH-L1–L3 | Low | Architecture | various | See Architecture section |

---

*Report generated by 5-agent parallel code review team · EVS Navigation System · 2026-02-19*
