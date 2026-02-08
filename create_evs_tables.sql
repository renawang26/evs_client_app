-- asr_files definition (stores file + model combinations for fast lookup)
CREATE TABLE IF NOT EXISTS asr_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT NOT NULL,
    lang TEXT NOT NULL,  -- 'en' or 'zh'
    asr_provider TEXT NOT NULL,
    model TEXT NOT NULL,
    slice_duration REAL,
    channel_num INTEGER,
    audio_file TEXT,  -- Path to the original audio file
    total_segments INTEGER DEFAULT 0,
    total_words INTEGER DEFAULT 0,
    total_duration REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Unique constraint to prevent duplicates
    UNIQUE(file_name, lang, asr_provider, model)
);

-- Index for fast file lookups
CREATE INDEX IF NOT EXISTS idx_asr_files_file_name ON asr_files(file_name);
CREATE INDEX IF NOT EXISTS idx_asr_files_lang ON asr_files(file_name, lang);

-- asr_results_segments definition

CREATE TABLE IF NOT EXISTS asr_results_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asr_file_id INTEGER,  -- Reference to asr_files table
    asr_provider TEXT,
    file_name TEXT NOT NULL,
    masked_file_name TEXT,
    lang TEXT NOT NULL,
    segment_id INTEGER,
    start_time REAL,
    end_time REAL,
    text TEXT,
    edit_text TEXT,
    speaker TEXT,
    duration REAL,
    slice_duration REAL,
    model TEXT,
    channel_num INTEGER,
    audio_file TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    chunk_start_time NUMERIC,
    chunk_end_time NUMERIC,
    chunk_id INTEGER,
    FOREIGN KEY (asr_file_id) REFERENCES asr_files(id)
);


-- Create the words table
CREATE TABLE IF NOT EXISTS asr_results_words (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asr_file_id INTEGER,  -- Reference to asr_files table
    asr_provider TEXT,
    file_name TEXT NOT NULL,
    masked_file_name TEXT,
    lang TEXT NOT NULL,
    word TEXT,
    edit_word TEXT,
    start_time REAL,
    end_time REAL,
    confidence REAL,
    speaker TEXT,
    segment_id INTEGER,
    word_seq_no INTEGER,
    duration REAL,
    slice_duration REAL,
    model TEXT,
    channel_num INTEGER,
    audio_file TEXT,
    pair_type TEXT,
    annotate TEXT,
    pair_seq INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    chunk_id INTEGER,
    chunk_start_time NUMERIC,
    chunk_end_time NUMERIC,
    -- NLP processing columns
    nlp_word TEXT,
    nlp_pos TEXT,
    nlp_confidence REAL,
    nlp_processed_at TEXT,
    nlp_comparison TEXT,
    nlp_engine TEXT,
    nlp_engine_info TEXT,
    UNIQUE(file_name, chunk_id, lang, segment_id, word_seq_no, asr_provider),
    FOREIGN KEY (asr_file_id) REFERENCES asr_files(id)
);

-- Simultaneous Interpretation Analysis Results table
CREATE TABLE IF NOT EXISTS si_analysis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT NOT NULL,
    asr_provider TEXT NOT NULL,
    analysis_type TEXT NOT NULL, -- 'complete_analysis', 'quality', 'errors', 'timing', 'cultural', 'suggestions', 'corrected'
    analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Quality metrics (verified and reliable)
    overall_score REAL,        -- English WPM or primary metric
    accuracy_score REAL,       -- Chinese WPM or secondary metric
    fluency_score REAL,        -- Speed ratio or fluency metric
    completeness_score REAL,   -- Coverage rate or completeness metric
    quality_level TEXT,        -- Overall quality assessment

    -- Speech rate analysis (verified metrics)
    en_wpm REAL,              -- English words per minute
    zh_wpm REAL,              -- Chinese words per minute
    speed_ratio REAL,         -- Chinese/English speed ratio
    pace_assessment TEXT,     -- Pace evaluation (Slow/Moderate/Fast)
    balance_assessment TEXT,  -- Speed balance evaluation

    -- Segment coverage analysis (verified metrics)
    bilingual_segments INTEGER,  -- Number of bilingual segments
    coverage_rate REAL,          -- Segment coverage percentage

    -- Confidence analysis (when available)
    confidence_mean REAL,        -- Average confidence score
    confidence_std REAL,         -- Confidence standard deviation
    confidence_words_count INTEGER, -- Number of words with confidence scores

    -- Legacy fields (for future EVS analysis)
    total_errors INTEGER,
    error_statistics TEXT, -- JSON string of error counts by type
    error_density REAL,
    average_delay REAL,
    max_delay REAL,
    sync_quality TEXT,
    sync_issue_count INTEGER,
    adaptation_score REAL,
    cultural_issue_count INTEGER,
    adaptation_level TEXT,

    -- Analysis configuration and results
    analysis_config TEXT, -- JSON string of analysis parameters
    analysis_results TEXT, -- JSON string of detailed results

    -- Metadata
    total_segments INTEGER,
    processing_time_ms INTEGER,
    llm_model TEXT,
    created_by TEXT,

    UNIQUE(file_name, asr_provider, analysis_type, analysis_timestamp)
);

-- Detailed error records table
CREATE TABLE IF NOT EXISTS si_error_details (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL,
    error_type TEXT NOT NULL, -- 'omission', 'mistranslation', 'tense_error', etc.
    severity TEXT NOT NULL, -- 'high', 'medium', 'low'
    segment_index INTEGER,
    timestamp REAL,
    source_text TEXT,
    target_text TEXT,
    description TEXT,
    suggestion TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (analysis_id) REFERENCES si_analysis_results(id)
);

-- Cultural issues table
CREATE TABLE IF NOT EXISTS si_cultural_issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL,
    issue_type TEXT NOT NULL, -- 'idiom_literal', 'politeness', etc.
    severity TEXT NOT NULL,
    segment_index INTEGER,
    timestamp REAL,
    description TEXT,
    suggestion TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (analysis_id) REFERENCES si_analysis_results(id)
);

-- Timing sync issues table
CREATE TABLE IF NOT EXISTS si_timing_issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL,
    segment_index INTEGER,
    timestamp REAL,
    delay REAL,
    severity TEXT,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (analysis_id) REFERENCES si_analysis_results(id)
);

-- Analysis configuration table
CREATE TABLE IF NOT EXISTS si_analysis_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_name TEXT UNIQUE NOT NULL,
    config_type TEXT NOT NULL, -- 'quality_metrics', 'error_patterns', 'cultural_rules'
    config_data TEXT NOT NULL, -- JSON configuration data
    is_active INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT
);

-- Corrected versions table
CREATE TABLE IF NOT EXISTS si_corrected_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_analysis_id INTEGER NOT NULL,
    file_name TEXT NOT NULL,
    asr_provider TEXT NOT NULL,
    corrected_segments TEXT NOT NULL, -- JSON array of corrected segments
    corrections_made INTEGER,
    correction_rate REAL,
    correction_summary TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT,

    FOREIGN KEY (original_analysis_id) REFERENCES si_analysis_results(id)
);

-- email_metrics definition

CREATE TABLE IF NOT EXISTS email_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email_id TEXT,
    email_type TEXT,
    recipient TEXT,
    queued_at TIMESTAMP,
    sent_at TIMESTAMP NULL,
    status TEXT,
    error_message TEXT NULL,
    priority INTEGER,
    attempt_count INTEGER,
    queue_time_ms INTEGER,
    metadata TEXT
);

-- email_metrics_summary definition

CREATE TABLE IF NOT EXISTS email_metrics_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT,
    hour INTEGER,
    total_sent INTEGER,
    total_failed INTEGER,
    total_queued INTEGER,
    avg_queue_time_ms REAL,
    p95_queue_time_ms REAL,
    common_errors TEXT
);

-- login_history definition

CREATE TABLE IF NOT EXISTS login_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address TEXT,
    user_agent TEXT,
    login_status TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id)
);

-- users definition
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    verification_token TEXT,
    verified INTEGER DEFAULT 0,
    token_expiry TIMESTAMP,
    user_type TEXT
);

-- ASR Configuration Table
CREATE TABLE IF NOT EXISTS asr_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    config TEXT NOT NULL,
    created_at TEXT,
    updated_at TEXT,
    UNIQUE(provider)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_si_analysis_file ON si_analysis_results(file_name);
CREATE INDEX IF NOT EXISTS idx_si_analysis_provider ON si_analysis_results(asr_provider);
CREATE INDEX IF NOT EXISTS idx_si_analysis_type ON si_analysis_results(analysis_type);
CREATE INDEX IF NOT EXISTS idx_si_analysis_timestamp ON si_analysis_results(analysis_timestamp);
CREATE INDEX IF NOT EXISTS idx_si_error_analysis ON si_error_details(analysis_id);
CREATE INDEX IF NOT EXISTS idx_si_cultural_analysis ON si_cultural_issues(analysis_id);
CREATE INDEX IF NOT EXISTS idx_si_timing_analysis ON si_timing_issues(analysis_id);
CREATE INDEX IF NOT EXISTS idx_si_config_name ON si_analysis_config(config_name);
CREATE INDEX IF NOT EXISTS idx_si_config_type ON si_analysis_config(config_type);