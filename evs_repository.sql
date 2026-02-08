```sql
-- asr_results_segments table schema
CREATE TABLE asr_results_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asr_provider TEXT,
    file_name TEXT NOT NULL,
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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- asr_results_words table schema
CREATE TABLE asr_results_words (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asr_provider TEXT,
    file_name TEXT NOT NULL,
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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```