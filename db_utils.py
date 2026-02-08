import sqlite3
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# 导入DBManager
logger = logging.getLogger(__name__)

try:
    from db_manager import get_db_connection as get_unified_db_connection
    USE_UNIFIED_DB = True
    logger.info("使用统一数据库管理器")
except ImportError:
    USE_UNIFIED_DB = False
    logger.warning("未找到统一数据库管理器，使用默认连接")

# Try to import cache utilities for Streamlit caching
try:
    from cache_utils import (
        get_cached_db_connection,
        get_cached_interpret_files,
        get_cached_processed_files,
        get_cached_asr_data,
        get_cached_word_frequency,
        get_cached_asr_config,
        clear_cache
    )
    USE_CACHE = False  # Disabled - SQLite is fast enough, avoids stale data bugs
    logger.info("缓存工具已加载（缓存已禁用）")
except ImportError:
    USE_CACHE = False
    logger.warning("缓存工具未找到，使用标准数据库连接")

# Constants
REPOSITORY_ROOT = "./data"

def _convert_slice_duration(value, default=30) -> float:
    """Convert slice_duration from various types to float.

    SQLite may return BLOB/bytes instead of numeric for some data,
    especially when data was inserted with incorrect type.
    """
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, bytes):
        # Handle bytes - try to interpret as little-endian integer
        try:
            import struct
            if len(value) == 8:
                return float(struct.unpack('<q', value)[0])
            elif len(value) == 4:
                return float(struct.unpack('<i', value)[0])
            else:
                return default
        except:
            return default
    if isinstance(value, str):
        try:
            return float(value)
        except:
            return default
    return default

class EVSDataUtils:
    @staticmethod
    def get_db_path() -> Path:
        return Path(REPOSITORY_ROOT) / 'evs_repository.db'

    @staticmethod
    def get_db_connection():
        """Create and return a database connection."""
        try:
            # 如果存在统一数据库管理器，则使用它
            if USE_UNIFIED_DB:
                return get_unified_db_connection()

            # 否则使用原始连接方式
            conn = sqlite3.connect(EVSDataUtils.get_db_path(), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {str(e)}")
            raise

    # ============ ASR Files Table Management ============

    @staticmethod
    def ensure_asr_files_table():
        """Create the asr_files table if it doesn't exist."""
        with EVSDataUtils.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS asr_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_name TEXT NOT NULL,
                    lang TEXT NOT NULL,
                    asr_provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    slice_duration REAL,
                    channel_num INTEGER,
                    audio_file TEXT,
                    total_segments INTEGER DEFAULT 0,
                    total_words INTEGER DEFAULT 0,
                    total_duration REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(file_name, lang, asr_provider, model)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_asr_files_file_name ON asr_files(file_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_asr_files_lang ON asr_files(file_name, lang)")
            conn.commit()

    @staticmethod
    def register_asr_file(file_name: str, lang: str, asr_provider: str, model: str,
                          slice_duration: float = None, channel_num: int = None,
                          audio_file: str = None, total_segments: int = 0,
                          total_words: int = 0, total_duration: float = None) -> int:
        """Register a file with its ASR model info. Returns the asr_file_id."""
        EVSDataUtils.ensure_asr_files_table()
        with EVSDataUtils.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO asr_files (file_name, lang, asr_provider, model, slice_duration,
                                       channel_num, audio_file, total_segments, total_words,
                                       total_duration, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(file_name, lang, asr_provider, model) DO UPDATE SET
                    slice_duration = excluded.slice_duration,
                    channel_num = excluded.channel_num,
                    audio_file = excluded.audio_file,
                    total_segments = excluded.total_segments,
                    total_words = excluded.total_words,
                    total_duration = excluded.total_duration,
                    updated_at = CURRENT_TIMESTAMP
            """, (file_name, lang, asr_provider, model, slice_duration, channel_num,
                  audio_file, total_segments, total_words, total_duration))
            conn.commit()

            # Get the ID
            cursor.execute("""
                SELECT id FROM asr_files
                WHERE file_name = ? AND lang = ? AND asr_provider = ? AND model = ?
            """, (file_name, lang, asr_provider, model))
            row = cursor.fetchone()
            return row[0] if row else None

    @staticmethod
    def get_asr_file_id(file_name: str, lang: str, asr_provider: str, model: str) -> Optional[int]:
        """Get the asr_file_id for a file/lang/provider/model combination."""
        with EVSDataUtils.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id FROM asr_files
                WHERE file_name = ? AND lang = ? AND asr_provider = ? AND model = ?
            """, (file_name, lang, asr_provider, model))
            row = cursor.fetchone()
            return row[0] if row else None

    @staticmethod
    def migrate_asr_files_from_existing_data():
        """Populate asr_files table from existing asr_results_words data."""
        EVSDataUtils.ensure_asr_files_table()
        with EVSDataUtils.get_db_connection() as conn:
            cursor = conn.cursor()
            # Get unique file/lang/provider/model combinations from words table
            cursor.execute("""
                INSERT OR IGNORE INTO asr_files (file_name, lang, asr_provider, model, slice_duration,
                                                  total_segments, total_words)
                SELECT
                    w.file_name,
                    w.lang,
                    w.asr_provider,
                    w.model,
                    MAX(w.slice_duration) as slice_duration,
                    COUNT(DISTINCT w.segment_id) as total_segments,
                    COUNT(*) as total_words
                FROM asr_results_words w
                WHERE w.asr_provider IS NOT NULL AND w.model IS NOT NULL
                GROUP BY w.file_name, w.lang, w.asr_provider, w.model
            """)
            rows_inserted = cursor.rowcount
            conn.commit()
            logger.info(f"Migrated {rows_inserted} records to asr_files table")
            return rows_inserted

    @staticmethod
    def get_all_asr_files() -> pd.DataFrame:
        """Get all files from asr_files table (fast lookup)."""
        EVSDataUtils.ensure_asr_files_table()
        with EVSDataUtils.get_db_connection() as conn:
            query = """
                SELECT id, file_name, lang, asr_provider, model, slice_duration,
                       total_segments, total_words, created_at, updated_at
                FROM asr_files
                ORDER BY file_name, lang
            """
            return pd.read_sql_query(query, conn)

    @staticmethod
    def get_files_with_models_fast() -> pd.DataFrame:
        """Get all files with their EN and ZH model info from asr_files table (fast).

        Returns a DataFrame with columns:
        - file_name, en_provider, en_model, zh_provider, zh_model, slice_duration
        """
        EVSDataUtils.ensure_asr_files_table()

        # First check if asr_files has data, if not migrate
        with EVSDataUtils.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM asr_files")
            count = cursor.fetchone()[0]
            if count == 0:
                logger.info("asr_files table is empty, migrating from existing data...")
                EVSDataUtils.migrate_asr_files_from_existing_data()

            # Now query the asr_files table
            query = """
                SELECT
                    file_name,
                    MAX(CASE WHEN lang = 'en' THEN asr_provider END) as en_provider,
                    MAX(CASE WHEN lang = 'en' THEN model END) as en_model,
                    MAX(CASE WHEN lang = 'zh' THEN asr_provider END) as zh_provider,
                    MAX(CASE WHEN lang = 'zh' THEN model END) as zh_model,
                    MAX(slice_duration) as slice_duration
                FROM asr_files
                GROUP BY file_name
                ORDER BY file_name
            """
            return pd.read_sql_query(query, conn)

    @staticmethod
    def get_file_models_fast(file_name: str) -> list:
        """Get all model pairs for a file from asr_files table (fast).

        Returns a list of dicts with en_provider, en_model, zh_provider, zh_model, display_name.
        """
        EVSDataUtils.ensure_asr_files_table()
        with EVSDataUtils.get_db_connection() as conn:
            query = """
                SELECT id, lang, asr_provider, model, slice_duration
                FROM asr_files
                WHERE file_name = ?
                ORDER BY lang, asr_provider, model
            """
            df = pd.read_sql_query(query, conn, params=[file_name])

            if df.empty:
                return []

            # Separate EN and ZH
            en_combos = df[df['lang'] == 'en'][['asr_provider', 'model', 'slice_duration']].drop_duplicates()
            zh_combos = df[df['lang'] == 'zh'][['asr_provider', 'model', 'slice_duration']].drop_duplicates()

            result = []
            if en_combos.empty and not zh_combos.empty:
                for _, zh in zh_combos.iterrows():
                    result.append({
                        'en_provider': None, 'en_model': None,
                        'zh_provider': zh['asr_provider'], 'zh_model': zh['model'],
                        'slice_duration': _convert_slice_duration(zh['slice_duration']),
                        'display_name': f"ZH: {zh['asr_provider']}/{zh['model']}"
                    })
            elif zh_combos.empty and not en_combos.empty:
                for _, en in en_combos.iterrows():
                    result.append({
                        'en_provider': en['asr_provider'], 'en_model': en['model'],
                        'zh_provider': None, 'zh_model': None,
                        'slice_duration': _convert_slice_duration(en['slice_duration']),
                        'display_name': f"EN: {en['asr_provider']}/{en['model']}"
                    })
            else:
                for _, en in en_combos.iterrows():
                    for _, zh in zh_combos.iterrows():
                        display = f"EN: {en['asr_provider']}/{en['model']} | ZH: {zh['asr_provider']}/{zh['model']}"
                        sd = _convert_slice_duration(en['slice_duration']) or _convert_slice_duration(zh['slice_duration'])
                        result.append({
                            'en_provider': en['asr_provider'], 'en_model': en['model'],
                            'zh_provider': zh['asr_provider'], 'zh_model': zh['model'],
                            'slice_duration': sd,
                            'display_name': display
                        })

            return result

    # ============ Original Methods ============

    @staticmethod
    def get_evs() -> pd.DataFrame:
        """Get EVS data from database."""
        with EVSDataUtils.get_db_connection() as conn:
            query = """
            SELECT interpret_file, speech, sentence_seq_no, word_seq_no,
                   edit_word, speak_start_time, speak_end_time
            FROM pd_interpret_words
            WHERE service_provider = 'ibm' AND lang = 'zh'
            ORDER BY interpret_id, speak_start_timestamp ASC
            """
            return pd.read_sql_query(query, conn)

    @staticmethod
    def get_interpret_files(asr_provider: str) -> pd.DataFrame:
        """Get interpretation files from database."""
        # Use cached version if available
        if USE_CACHE:
            return get_cached_interpret_files(asr_provider)

        with EVSDataUtils.get_db_connection() as conn:
            query = """
            SELECT DISTINCT(file_name), slice_duration
            FROM asr_results_segments
            WHERE asr_provider = ?
            """
            return pd.read_sql_query(query, conn, params=[asr_provider])

    @staticmethod
    def get_all_files_with_transcriptions() -> pd.DataFrame:
        """Get all interpretation files with their EN and ZH transcription info.

        Uses the fast asr_files table lookup. Falls back to slow query if needed.

        Returns a DataFrame with columns:
        - file_name: The interpretation file name
        - en_provider, en_model: English transcription provider and model (or None)
        - zh_provider, zh_model: Chinese transcription provider and model (or None)
        - slice_duration: The slice duration used for the file
        """
        try:
            # Use the fast method from asr_files table
            result = EVSDataUtils.get_files_with_models_fast()
            if not result.empty:
                return result

            # Fall back to slow method if asr_files is empty
            logger.warning("asr_files table empty, using slow query")
            return EVSDataUtils._get_all_files_with_transcriptions_slow()
        except Exception as e:
            logger.error(f"Error getting files with transcriptions: {str(e)}")
            # Try slow method as fallback
            try:
                return EVSDataUtils._get_all_files_with_transcriptions_slow()
            except:
                return pd.DataFrame(columns=['file_name', 'en_provider', 'en_model',
                                              'zh_provider', 'zh_model', 'slice_duration'])

    @staticmethod
    def _get_all_files_with_transcriptions_slow() -> pd.DataFrame:
        """Slow fallback: Get files from asr_results_words table."""
        with EVSDataUtils.get_db_connection() as conn:
            query = """
            SELECT DISTINCT
                w.file_name,
                w.lang,
                w.asr_provider,
                w.model,
                s.slice_duration
            FROM asr_results_words w
            LEFT JOIN asr_results_segments s
                ON w.file_name = s.file_name
                AND w.asr_provider = s.asr_provider
                AND w.model = s.model
            ORDER BY w.file_name, w.lang
            """
            df = pd.read_sql_query(query, conn)

            if df.empty:
                return pd.DataFrame(columns=['file_name', 'en_provider', 'en_model',
                                              'zh_provider', 'zh_model', 'slice_duration'])

            result = []
            for file_name in df['file_name'].unique():
                file_data = df[df['file_name'] == file_name]
                en_data = file_data[file_data['lang'] == 'en']
                zh_data = file_data[file_data['lang'] == 'zh']

                slice_duration = None
                if not en_data.empty and pd.notna(en_data.iloc[0]['slice_duration']):
                    slice_duration = en_data.iloc[0]['slice_duration']
                elif not zh_data.empty and pd.notna(zh_data.iloc[0]['slice_duration']):
                    slice_duration = zh_data.iloc[0]['slice_duration']

                result.append({
                    'file_name': file_name,
                    'en_provider': en_data.iloc[0]['asr_provider'] if not en_data.empty else None,
                    'en_model': en_data.iloc[0]['model'] if not en_data.empty else None,
                    'zh_provider': zh_data.iloc[0]['asr_provider'] if not zh_data.empty else None,
                    'zh_model': zh_data.iloc[0]['model'] if not zh_data.empty else None,
                    'slice_duration': slice_duration
                })

            return pd.DataFrame(result)

    @staticmethod
    def get_file_model_pairs(file_name: str) -> list:
        """Get all available model pairs for a specific file.

        Uses the fast asr_files table lookup. Falls back to slow query if needed.

        Returns a list of dicts, each containing:
        - en_provider, en_model: English transcription info
        - zh_provider, zh_model: Chinese transcription info
        - slice_duration: The slice duration used
        - display_name: Human-readable name for the dropdown
        """
        try:
            # Use the fast method from asr_files table
            result = EVSDataUtils.get_file_models_fast(file_name)
            if result:
                return result

            # Fall back to slow method if asr_files doesn't have this file
            logger.debug(f"File {file_name} not in asr_files, using slow query")
            return EVSDataUtils._get_file_model_pairs_slow(file_name)
        except Exception as e:
            logger.error(f"Error getting model pairs for file {file_name}: {str(e)}")
            # Try slow method as fallback
            try:
                return EVSDataUtils._get_file_model_pairs_slow(file_name)
            except:
                return []

    @staticmethod
    def _get_file_model_pairs_slow(file_name: str) -> list:
        """Slow fallback: Get model pairs from asr_results_words table."""
        with EVSDataUtils.get_db_connection() as conn:
            query = """
            SELECT DISTINCT
                w.lang,
                w.asr_provider,
                w.model,
                s.slice_duration
            FROM asr_results_words w
            LEFT JOIN asr_results_segments s
                ON w.file_name = s.file_name
                AND w.asr_provider = s.asr_provider
                AND w.model = s.model
            WHERE w.file_name = ?
            ORDER BY w.lang, w.asr_provider, w.model
            """
            df = pd.read_sql_query(query, conn, params=[file_name])

            if df.empty:
                return []

            en_combos = df[df['lang'] == 'en'][['asr_provider', 'model', 'slice_duration']].drop_duplicates()
            zh_combos = df[df['lang'] == 'zh'][['asr_provider', 'model', 'slice_duration']].drop_duplicates()

            result = []
            if en_combos.empty and not zh_combos.empty:
                for _, zh in zh_combos.iterrows():
                    result.append({
                        'en_provider': None, 'en_model': None,
                        'zh_provider': zh['asr_provider'], 'zh_model': zh['model'],
                        'slice_duration': _convert_slice_duration(zh['slice_duration']),
                        'display_name': f"ZH: {zh['asr_provider']}/{zh['model']}"
                    })
            elif zh_combos.empty and not en_combos.empty:
                for _, en in en_combos.iterrows():
                    result.append({
                        'en_provider': en['asr_provider'], 'en_model': en['model'],
                        'zh_provider': None, 'zh_model': None,
                        'slice_duration': _convert_slice_duration(en['slice_duration']),
                        'display_name': f"EN: {en['asr_provider']}/{en['model']}"
                    })
            else:
                for _, en in en_combos.iterrows():
                    for _, zh in zh_combos.iterrows():
                        display = f"EN: {en['asr_provider']}/{en['model']} | ZH: {zh['asr_provider']}/{zh['model']}"
                        sd = _convert_slice_duration(en['slice_duration']) or _convert_slice_duration(zh['slice_duration'])
                        result.append({
                            'en_provider': en['asr_provider'], 'en_model': en['model'],
                            'zh_provider': zh['asr_provider'], 'zh_model': zh['model'],
                            'slice_duration': sd,
                            'display_name': display
                        })

            return result

    @staticmethod
    def get_file_asr_info(file_name: str) -> dict:
        """Get ASR provider and model info for each language in a file.

        Returns:
            dict: {
                'en': {'provider': str, 'model': str} or None,
                'zh': {'provider': str, 'model': str} or None
            }
        """
        try:
            with EVSDataUtils.get_db_connection() as conn:
                query = """
                SELECT DISTINCT lang, asr_provider, model
                FROM asr_results_words
                WHERE file_name = ?
                ORDER BY lang
                """
                cursor = conn.execute(query, [file_name])
                rows = cursor.fetchall()

                result = {'en': None, 'zh': None}
                for row in rows:
                    lang, provider, model = row
                    if lang in result:
                        result[lang] = {'provider': provider, 'model': model}

                return result
        except Exception as e:
            logger.error(f"Error getting file ASR info: {str(e)}")
            return {'en': None, 'zh': None}

    @staticmethod
    def get_file_stats(file_name: str, asr_provider: str, model: str = None) -> Dict[str, Any]:
        """
        Get statistics for a file (optionally filtered by model).

        Args:
            file_name: Name of the file
            asr_provider: ASR provider name
            model: Model name (if None, returns stats for all models)

        Returns:
            Dict with file statistics
        """
        try:
            with EVSDataUtils.get_db_connection() as conn:
                if model:
                    # Count words for specific model
                    words_query = """
                        SELECT COUNT(*) as count FROM asr_results_words
                        WHERE file_name = ? AND asr_provider = ? AND model = ?
                    """
                    words_count = pd.read_sql_query(words_query, conn, params=[file_name, asr_provider, model])

                    # Count segments for specific model
                    segments_query = """
                        SELECT COUNT(*) as count FROM asr_results_segments
                        WHERE file_name = ? AND asr_provider = ? AND model = ?
                    """
                    segments_count = pd.read_sql_query(segments_query, conn, params=[file_name, asr_provider, model])

                    # Get languages for specific model
                    lang_query = """
                        SELECT DISTINCT lang FROM asr_results_words
                        WHERE file_name = ? AND asr_provider = ? AND model = ?
                    """
                    languages = pd.read_sql_query(lang_query, conn, params=[file_name, asr_provider, model])
                else:
                    # Count words for all models
                    words_query = """
                        SELECT COUNT(*) as count FROM asr_results_words
                        WHERE file_name = ? AND asr_provider = ?
                    """
                    words_count = pd.read_sql_query(words_query, conn, params=[file_name, asr_provider])

                    # Count segments for all models
                    segments_query = """
                        SELECT COUNT(*) as count FROM asr_results_segments
                        WHERE file_name = ? AND asr_provider = ?
                    """
                    segments_count = pd.read_sql_query(segments_query, conn, params=[file_name, asr_provider])

                    # Get languages for all models
                    lang_query = """
                        SELECT DISTINCT lang FROM asr_results_words
                        WHERE file_name = ? AND asr_provider = ?
                    """
                    languages = pd.read_sql_query(lang_query, conn, params=[file_name, asr_provider])

                return {
                    'file_name': file_name,
                    'asr_provider': asr_provider,
                    'model': model,
                    'words_count': int(words_count['count'].iloc[0]) if not words_count.empty else 0,
                    'segments_count': int(segments_count['count'].iloc[0]) if not segments_count.empty else 0,
                    'languages': languages['lang'].tolist() if not languages.empty else []
                }

        except Exception as e:
            logger.error(f"Error getting file stats for '{file_name}': {str(e)}")
            return {
                'file_name': file_name,
                'asr_provider': asr_provider,
                'model': model,
                'words_count': 0,
                'segments_count': 0,
                'languages': []
            }

    @staticmethod
    def check_file_model_exists(file_name: str, model: str, asr_provider: str) -> bool:
        """
        Check if a file + model combination already exists in the database.

        Args:
            file_name: Name of the file to check
            model: Model name to check
            asr_provider: ASR provider name

        Returns:
            bool: True if data exists, False otherwise
        """
        try:
            with EVSDataUtils.get_db_connection() as conn:
                query = """
                    SELECT COUNT(*) as count FROM asr_results_words
                    WHERE file_name = ? AND model = ? AND asr_provider = ?
                """
                result = pd.read_sql_query(query, conn, params=[file_name, model, asr_provider])
                return int(result['count'].iloc[0]) > 0 if not result.empty else False

        except Exception as e:
            logger.error(f"Error checking file model exists: {str(e)}")
            return False

    @staticmethod
    def delete_file(file_name: str, asr_provider: str, model: str = None) -> bool:
        """
        Delete a file and its associated data from the database.
        If model is specified, only delete data for that model.

        Args:
            file_name: Name of the file to delete
            asr_provider: ASR provider name
            model: Model name (if None, deletes all models for the file)

        Returns:
            bool: True if deletion successful, False otherwise
        """
        try:
            with EVSDataUtils.get_db_connection() as conn:
                cursor = conn.cursor()

                if model:
                    # Delete only specific model's data
                    cursor.execute("""
                        DELETE FROM asr_results_words
                        WHERE file_name = ? AND asr_provider = ? AND model = ?
                    """, (file_name, asr_provider, model))
                    words_deleted = cursor.rowcount

                    cursor.execute("""
                        DELETE FROM asr_results_segments
                        WHERE file_name = ? AND asr_provider = ? AND model = ?
                    """, (file_name, asr_provider, model))
                    segments_deleted = cursor.rowcount

                    # Delete from chinese_nlp_results if exists (table doesn't have model column)
                    cursor.execute("""
                        SELECT name FROM sqlite_master
                        WHERE type='table' AND name='chinese_nlp_results'
                    """)
                    if cursor.fetchone():
                        cursor.execute("""
                            DELETE FROM chinese_nlp_results
                            WHERE file_name = ? AND asr_provider = ?
                        """, (file_name, asr_provider))

                    conn.commit()
                    logger.info(f"Deleted file '{file_name}' model '{model}' (provider: {asr_provider}): "
                               f"{words_deleted} words, {segments_deleted} segments")
                else:
                    # Delete all models for this file
                    cursor.execute("""
                        DELETE FROM asr_results_words
                        WHERE file_name = ? AND asr_provider = ?
                    """, (file_name, asr_provider))
                    words_deleted = cursor.rowcount

                    cursor.execute("""
                        DELETE FROM asr_results_segments
                        WHERE file_name = ? AND asr_provider = ?
                    """, (file_name, asr_provider))
                    segments_deleted = cursor.rowcount

                    # Delete from chinese_nlp_results if exists
                    cursor.execute("""
                        SELECT name FROM sqlite_master
                        WHERE type='table' AND name='chinese_nlp_results'
                    """)
                    if cursor.fetchone():
                        cursor.execute("""
                            DELETE FROM chinese_nlp_results
                            WHERE file_name = ? AND asr_provider = ?
                        """, (file_name, asr_provider))

                    conn.commit()
                    logger.info(f"Deleted file '{file_name}' all models (provider: {asr_provider}): "
                               f"{words_deleted} words, {segments_deleted} segments")

                # Clear cache if available
                if USE_CACHE:
                    clear_cache('all')

                return True

        except Exception as e:
            logger.error(f"Error deleting file '{file_name}': {str(e)}")
            return False

    @staticmethod
    def get_asr_data(interpret_file: str, asr_provider: str = None, model: str = None,
                     en_provider: str = None, en_model: str = None,
                     zh_provider: str = None, zh_model: str = None) -> pd.DataFrame:
        """Get ASR data for a given interpretation file.

        Supports two modes:
        1. Same provider/model for both languages (legacy): use asr_provider and model params
        2. Different provider/model per language: use en_provider/en_model and zh_provider/zh_model

        If per-language params are provided, they take precedence over legacy params.
        If no provider/model specified, returns all data for the file.
        """
        # Only use cache for legacy queries (same provider for both languages)
        # Skip cache when per-language params are provided
        use_per_lang_params = en_provider or en_model or zh_provider or zh_model
        if USE_CACHE and not use_per_lang_params and asr_provider:
            return get_cached_asr_data(interpret_file, asr_provider)

        try:
            with EVSDataUtils.get_db_connection() as conn:
                # Determine provider/model for each language
                actual_en_provider = en_provider or asr_provider
                actual_en_model = en_model or model
                actual_zh_provider = zh_provider or asr_provider
                actual_zh_model = zh_model or model

                # Build filter conditions for English
                en_conditions = ["file_name = ?", "lang = 'en'"]
                en_params = [interpret_file]
                if actual_en_provider:
                    en_conditions.append("asr_provider = ?")
                    en_params.append(actual_en_provider)
                if actual_en_model:
                    en_conditions.append("model = ?")
                    en_params.append(actual_en_model)

                # Build filter conditions for Chinese
                zh_conditions = ["file_name = ?", "lang = 'zh'"]
                zh_params = [interpret_file]
                if actual_zh_provider:
                    zh_conditions.append("asr_provider = ?")
                    zh_params.append(actual_zh_provider)
                if actual_zh_model:
                    zh_conditions.append("model = ?")
                    zh_params.append(actual_zh_model)

                en_where = " AND ".join(en_conditions)
                zh_where = " AND ".join(zh_conditions)

                query = f"""
                SELECT * FROM (
                    SELECT id as interpret_id, file_name, lang,
                        segment_id, word_seq_no,
                        COALESCE(edit_word, '') || '&&' ||
                        COALESCE(pair_type, '') || '&&' ||
                        COALESCE(word, '') || '&&' ||
                        COALESCE(annotate, '') AS combined_word,
                        (end_time - start_time) AS duration, start_time, end_time,
                        slice_duration, confidence, word, edit_word, pair_type, pair_seq, annotate,
                        asr_provider, model
                    FROM asr_results_words
                    WHERE {en_where}
                    ORDER BY start_time ASC
                )
                UNION ALL
                SELECT * FROM (
                    SELECT id as interpret_id, file_name, lang,
                        segment_id, word_seq_no,
                        COALESCE(edit_word, '') || '&&' ||
                        COALESCE(pair_type, '') || '&&' ||
                        COALESCE(word, '') || '&&' ||
                        COALESCE(annotate, '') AS combined_word,
                        (end_time - start_time) AS duration, start_time, end_time,
                        slice_duration, confidence, word, edit_word, pair_type, pair_seq, annotate,
                        asr_provider, model
                    FROM asr_results_words
                    WHERE {zh_where}
                    ORDER BY start_time ASC
                )
                ORDER BY lang, start_time ASC;
                """

                params = en_params + zh_params
                return pd.read_sql_query(query, conn, params=params)
        except Exception as e:
            logger.error(f"Error getting ASR data: {str(e)}")
            raise

    @staticmethod
    def update_edit_word(interpret_file: str, lang: str,
                        segment_id: int, word_seq_no: int, edit_word: str, asr_provider: str) -> bool:
        """Update edited word in ASR results database."""
        with EVSDataUtils.get_db_connection() as conn:
            query = """UPDATE asr_results_words
            SET edit_word = ?
            WHERE file_name = ?
                AND segment_id = ?
                AND word_seq_no = ?
                AND lang = ?
                AND asr_provider = ?
            """
            conn.execute(query, [
                edit_word.replace("'", "''"),
                interpret_file,
                segment_id,
                word_seq_no,
                lang,
                asr_provider
            ])
            conn.commit()
            return True

    @staticmethod
    def get_word_frequency(interpret_file: str, lang: str, asr_provider: str) -> pd.DataFrame:
        # Use cached version if available
        if USE_CACHE:
            return get_cached_word_frequency(interpret_file, lang, asr_provider)

        df = None
        try:
            # Build query
            base_query = """
                    SELECT edit_word, lang, COUNT(*) as frequency
                    FROM asr_results_words
                    WHERE edit_word IS NOT NULL
                    """

            params = []
            conditions = []

            if interpret_file != 'All':
                conditions.append("file_name = ?")
                params.append(interpret_file)

            if lang != 'All':
                conditions.append("lang = ?")
                params.append(lang)

            if asr_provider != 'All':
                conditions.append("asr_provider = ?")
                params.append(asr_provider)

            final_query = base_query
            if conditions:
                final_query = f"{base_query} AND {' AND '.join(conditions)}"

            final_query += " GROUP BY edit_word, lang ORDER BY frequency DESC"


            with EVSDataUtils.get_db_connection() as conn:
                df = pd.read_sql_query(final_query, conn, params=params)

        except Exception as e:
            logger.error(f"Word list error: {str(e)}", exc_info=True)

        return df


    @staticmethod
    def update_evs(file_name: str, asr_provider: str, evs_start: List[str], evs_end: List[str]) -> bool:
        """Update EVS data in database."""
        try:
            # 输出详细日志
            logger.info(f"开始更新EVS数据, 文件: {file_name}")
            logger.info(f"evs_start 数据: {evs_start}")
            logger.info(f"evs_end 数据: {evs_end}")

            # 重置现有的EVS数据
            with EVSDataUtils.get_db_connection() as conn:

                # 重置现有数据
                reset_query = f"""
                UPDATE asr_results_words
                SET pair_type = NULL, pair_seq = NULL
                WHERE file_name = ? AND asr_provider = ?
                """
                res = conn.execute(reset_query, [file_name, asr_provider])
                logger.info(f"重置了 {res.rowcount} 行数据")

                # 主查询 - 使用精确匹配时间和单词
                update_query = f"""
                UPDATE asr_results_words
                SET pair_type = ?, pair_seq = ?
                WHERE file_name = ?
                AND asr_provider = ?
                AND start_time = ?
                AND edit_word = ?
                """

                success_count = 0
                for i, (start, end) in enumerate(zip(evs_start, evs_end), start=1):
                    if pd.isna(start) or pd.isna(end):
                        logger.warning(f"跳过空值配对: start={start}, end={end}")
                        continue

                    # 拆分时间戳和词语
                    start_parts = start.split(' ', 1)  # 最多拆分一次，保留带空格的词语
                    end_parts = end.split(' ', 1)      # 最多拆分一次，保留带空格的词语

                    if len(start_parts) < 2 or len(end_parts) < 2:
                        logger.warning(f"跳过格式错误的配对: start={start}, end={end}")
                        continue

                    # 提取时间戳和词语
                    try:
                        start_time = float(start_parts[0])
                        end_time = float(end_parts[0])
                        start_word = start_parts[1]
                        end_word = end_parts[1]

                        # 调试输出
                        logger.info(f"准备更新配对 {i}: EN={start_time} '{start_word}', ZH={end_time} '{end_word}'")

                        # 尝试更新英文 - 主查询
                        res1 = conn.execute(update_query, [
                            'S',  # 起点
                            i,
                            file_name,
                            asr_provider,
                            start_time,
                            start_word
                        ])

                        # 尝试更新中文 - 主查询
                        res2 = conn.execute(update_query, [
                            'E',  # 终点
                            i,
                            file_name,
                            asr_provider,
                            end_time,
                            end_word
                        ])


                        # Record the number of affected rows
                        logger.info(f"Update results: EN={res1.rowcount}, ZH={res2.rowcount}")

                        if res1.rowcount > 0 and res2.rowcount > 0:
                            success_count += 1
                        else:
                            # Try querying the database again to check if the words exist
                            check_en = conn.execute(
                                f"SELECT COUNT(*) FROM asr_results_words WHERE file_name = ? AND asr_provider = ? AND lang = 'en' AND edit_word = ?",
                                [file_name, asr_provider, start_word]
                            ).fetchone()[0]

                            check_zh = conn.execute(
                                f"SELECT COUNT(*) FROM asr_results_words WHERE file_name = ? AND asr_provider = ? AND lang = 'zh' AND edit_word = ?",
                                [file_name, asr_provider, end_word]
                            ).fetchone()[0]

                            logger.warning(f"Pair {i} update failed. English word in database: {check_en > 0}, Chinese word in database: {check_zh > 0}")

                    except Exception as e:
                        logger.error(f"Error processing pair {i}: {str(e)}", exc_info=True)
                        continue

                # 提交事务
                conn.commit()
                logger.info(f"成功更新了 {success_count} 对配对")
                return success_count > 0

        except Exception as e:
            logger.error(f"更新EVS数据时出错: {str(e)}", exc_info=True)
            if 'conn' in locals():
                try:
                    conn.rollback()
                except sqlite3.Error as rollback_error:
                    logger.warning(f"Rollback failed: {rollback_error}")
            return False

    @staticmethod
    def update_evs_by_ids(file_name: str, asr_provider: str, pairs: List[Dict],
                          source_provider: str = None, target_provider: str = None) -> bool:
        """
        Update EVS data using segment_id and word_seq_no for reliable matching.

        Args:
            file_name: The interpretation file name
            asr_provider: The default ASR provider name (used for reset and fallback)
            pairs: List of dicts with keys:
                - source_segment_id, source_word_seq_no
                - target_segment_id, target_word_seq_no
            source_provider: ASR provider for source language (optional, uses asr_provider if not set)
            target_provider: ASR provider for target language (optional, uses asr_provider if not set)

        Returns:
            bool: True if at least one pair was updated successfully
        """
        # Use provided providers or fall back to default
        src_provider = source_provider or asr_provider
        tgt_provider = target_provider or asr_provider

        try:
            logger.info(f"Updating EVS data by IDs, file: {file_name}, pairs: {len(pairs)}, "
                       f"source_provider: {src_provider}, target_provider: {tgt_provider}")

            with EVSDataUtils.get_db_connection() as conn:
                # Reset existing EVS data for both providers
                reset_query = """
                UPDATE asr_results_words
                SET pair_type = NULL, pair_seq = NULL
                WHERE file_name = ? AND asr_provider IN (?, ?)
                """
                res = conn.execute(reset_query, [file_name, src_provider, tgt_provider])
                logger.info(f"Reset {res.rowcount} rows")

                # Update query using segment_id and word_seq_no
                update_query = """
                UPDATE asr_results_words
                SET pair_type = ?, pair_seq = ?
                WHERE file_name = ?
                AND asr_provider = ?
                AND segment_id = ?
                AND word_seq_no = ?
                """

                success_count = 0
                for i, pair in enumerate(pairs, start=1):
                    try:
                        # Convert to int to handle numpy types
                        src_seg_id = int(pair['source_segment_id'])
                        src_word_seq = int(pair['source_word_seq_no'])
                        tgt_seg_id = int(pair['target_segment_id'])
                        tgt_word_seq = int(pair['target_word_seq_no'])

                        logger.info(f"Pair {i}: source=({src_seg_id}, {src_word_seq}) provider={src_provider}, "
                                   f"target=({tgt_seg_id}, {tgt_word_seq}) provider={tgt_provider}")

                        # Update source word (S = Start) using source provider
                        res1 = conn.execute(update_query, [
                            'S',
                            i,
                            file_name,
                            src_provider,
                            src_seg_id,
                            src_word_seq
                        ])

                        # Update target word (E = End) using target provider
                        res2 = conn.execute(update_query, [
                            'E',
                            i,
                            file_name,
                            tgt_provider,
                            tgt_seg_id,
                            tgt_word_seq
                        ])

                        logger.info(f"Pair {i}: source updated={res1.rowcount}, target updated={res2.rowcount}")

                        if res1.rowcount > 0 and res2.rowcount > 0:
                            success_count += 1
                        else:
                            logger.warning(f"Pair {i} update incomplete: source={res1.rowcount}, target={res2.rowcount}")

                    except Exception as e:
                        logger.error(f"Error processing pair {i}: {str(e)}", exc_info=True)
                        continue

                conn.commit()
                logger.info(f"Successfully updated {success_count} pairs")
                return success_count > 0

        except Exception as e:
            logger.error(f"Error updating EVS data by IDs: {str(e)}", exc_info=True)
            return False

    @staticmethod
    def delete_specific_pairs(file_name: str, words_to_unpair: list) -> bool:
        """
        Remove pair_type/pair_seq from specific words.

        Args:
            file_name: The interpretation file name
            words_to_unpair: List of dicts with keys:
                - segment_id (int)
                - word_seq_no (int)

        Returns:
            bool: True if at least one word was updated
        """
        if not words_to_unpair:
            return False
        try:
            logger.info(f"Deleting {len(words_to_unpair)} specific pair entries for {file_name}")
            with EVSDataUtils.get_db_connection() as conn:
                query = """
                UPDATE asr_results_words
                SET pair_type = NULL, pair_seq = NULL
                WHERE file_name = ? AND segment_id = ? AND word_seq_no = ?
                """
                total = 0
                for w in words_to_unpair:
                    res = conn.execute(query, [file_name, int(w['segment_id']), int(w['word_seq_no'])])
                    total += res.rowcount
                conn.commit()
                logger.info(f"Cleared pair data from {total} word rows")
                return total > 0
        except Exception as e:
            logger.error(f"Error deleting specific pairs: {str(e)}", exc_info=True)
            return False

    @staticmethod
    def reset_evs(file_name: str, asr_provider: str = None) -> bool:
        """
        重置指定文件的所有EVS配对数据

        Args:
            file_name: 口译文件名
            asr_provider: ASR提供商 (可选，如果不指定则重置所有提供商的数据)

        Returns:
            bool: 操作是否成功
        """
        try:
            logger.info(f"Resetting EVS data for file {file_name}")

            with EVSDataUtils.get_db_connection() as conn:
                cursor = conn.cursor()

                # 重置配对数据
                if asr_provider:
                    reset_query = """
                    UPDATE asr_results_words
                    SET pair_type = NULL, pair_seq = NULL
                    WHERE file_name = ? AND asr_provider = ?
                    """
                    cursor.execute(reset_query, (file_name, asr_provider))
                else:
                    # Reset all providers for this file
                    reset_query = """
                    UPDATE asr_results_words
                    SET pair_type = NULL, pair_seq = NULL
                    WHERE file_name = ?
                    """
                    cursor.execute(reset_query, (file_name,))

                rows_affected = cursor.rowcount
                conn.commit()

                logger.info(f"Reset {rows_affected} rows")
                return True

        except Exception as e:
            logger.error(f"Error resetting EVS data: {str(e)}")
            return False

    @staticmethod
    def save_analysis(analysis_data: Dict[str, Any], asr_provider: str) -> bool:
        """Save analysis results to database.

        Args:
            analysis_data: Dictionary containing analysis data with keys:
                - timestamp: ISO format timestamp
                - file_name: Name of the audio file
                - transcription: Transcribed text

        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            conn = sqlite3.connect('audio_analysis.db', check_same_thread=False)
            c = conn.cursor()

            # Create table if it doesn't exist
            c.execute('''
                CREATE TABLE IF NOT EXISTS analysis_history
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                file_name TEXT,
                asr_provider TEXT,
                transcription TEXT)
            ''')

            # Save analysis data
            c.execute('''
                INSERT INTO analysis_history (timestamp, file_name, asr_provider, transcription)
                VALUES (?, ?, ?, ?)
            ''', (
                analysis_data['timestamp'],
                analysis_data['file_name'],
                asr_provider,
                analysis_data['transcription']
            ))

            conn.commit()
            logger.info(f"Analysis saved for file: {analysis_data['file_name']}")
            return True

        except sqlite3.Error as e:
            logger.error(f"Error saving analysis: {str(e)}")
            if 'conn' in locals():
                conn.rollback()
            return False
        finally:
            if 'conn' in locals():
                conn.close()

    @staticmethod
    def get_analysis_history(asr_provider: str) -> List[Dict[str, Any]]:
        """Retrieve analysis history from database.

        Returns:
            List[Dict]: List of dictionaries containing analysis history
        """
        try:
            conn = sqlite3.connect('audio_analysis.db', check_same_thread=False)
            c = conn.cursor()

            # Create table if it doesn't exist
            c.execute('''
                CREATE TABLE IF NOT EXISTS analysis_history
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                file_name TEXT,
                asr_provider TEXT,
                transcription TEXT)
            ''')

            # Get analysis history
            c.execute('''
                SELECT timestamp, file_name, asr_provider, transcription
                FROM analysis_history
                ORDER BY timestamp DESC
            ''')

            results = []
            for row in c.fetchall():
                results.append({
                    'timestamp': row[0],
                    'file_name': row[1],
                    'asr_provider': row[2],
                    'transcription': row[3]
                })

            return results

        except sqlite3.Error as e:
            logger.error(f"Error retrieving analysis history: {str(e)}")
            return []
        finally:
            if 'conn' in locals():
                conn.close()


    @staticmethod
    def get_lang_file_words(interpret_file: str, lang: str, asr_provider: str) -> pd.DataFrame:
        try:
            # Get data from database
            query = """
                SELECT *
                FROM asr_results_words
                WHERE edit_word IS NOT NULL
                AND asr_provider = ?
            """

            params = []
            conditions = []

            if interpret_file != 'All':
                conditions.append("file_name = ?")
                params.append(interpret_file)

            if lang != 'All':
                conditions.append("lang = ?")
                params.append(lang)

            if conditions:
                query += " AND " + " AND ".join(conditions)

            query += " ORDER BY file_name, segment_id, word_seq_no"

            # Execute query
            with EVSDataUtils.get_db_connection() as conn:
                df = pd.read_sql_query(query, conn, params=[asr_provider] + params)
            return df
        except Exception as e:
            logger.error(f"Error getting search words: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def get_file_content(interpret_file: str, asr_provider: str) -> pd.DataFrame:
        # Get data from database
        query = """
        SELECT *
        FROM asr_results_words
        WHERE file_name = ?
        AND asr_provider = ?
        ORDER BY segment_id, word_seq_no
        """

        with EVSDataUtils.get_db_connection() as conn:
            df = pd.read_sql_query(query, conn, params=[interpret_file, asr_provider])
            return df



    @staticmethod
    def get_search_words(search_term: str, interpret_file: str, lang: str, asr_provider: str) -> pd.DataFrame:
        try:
            query = """
                WITH file_stats AS (
                    SELECT
                        file_name as FileName,
                        COUNT(*) as FileTokens
                    FROM asr_results_words
                    GROUP BY file_name
                ),
                hit_stats AS (
                    SELECT
                        w1.file_name as FileName,
                        COUNT(*) as Freq,
                        GROUP_CONCAT(w1.start_time, ',') as positions,
                        MAX(w1.start_time) as max_pos
                    FROM asr_results_words w1
                    WHERE CAST(edit_word AS TEXT) LIKE ?
            """

            params = [f'%{search_term}%']
            conditions = []

            if interpret_file != 'All':
                conditions.append("AND w1.file_name = ?")
                params.append(interpret_file)

            if lang != 'All':
                conditions.append("AND w1.lang = ?")
                params.append(lang)

            if asr_provider != 'All':
                conditions.append("AND w1.asr_provider = ?")
                params.append(asr_provider)

            # 添加WHERE条件到查询
            if conditions:
                query += " " + " ".join(conditions)

            # 完成hit_stats子查询
            query += """
                    GROUP BY file_name
                )
                SELECT
                    fs.FileName,
                    fs.FileTokens,
                    COALESCE(hs.Freq, 0) as Freq,
                    ROUND(CAST(COALESCE(hs.Freq, 0) AS FLOAT) * 1000 / fs.FileTokens, 3) as NormFreq,
                    hs.positions,
                    hs.max_pos
                FROM file_stats fs
                LEFT JOIN hit_stats hs ON fs.FileName = hs.FileName
                ORDER BY fs.FileName
            """

            with EVSDataUtils.get_db_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)

                return df

        except Exception as e:
            logger.error(f"Error getting interpreted words: {str(e)}")
            return pd.DataFrame()



    @staticmethod
    def search_interpreted_words(search_term, interpret_file, lang, asr_provider):
        query = """
                WITH matches AS (
                    SELECT DISTINCT
                        segment_id,
                        file_name,
                        word_seq_no,
                        lang
                    FROM asr_results_words
                    WHERE CAST(edit_word AS TEXT) LIKE ?
                )
                SELECT
                    w.edit_word,
                    w.lang,
                    w.file_name,
                    w.segment_id,
                    w.word_seq_no,
                    w.start_time
                FROM asr_results_words w
                INNER JOIN matches m
                    ON w.segment_id = m.segment_id
                    AND w.file_name = m.file_name
                    AND w.lang = m.lang
                """

        params = [f'%{search_term}%']
        conditions = []

        if interpret_file != 'All':
            conditions.append("w.file_name = ?")
            params.append(interpret_file)

        if lang != 'All':
            conditions.append("w.lang = ?")
            params.append(lang)

        if asr_provider != 'All':
            conditions.append("w.asr_provider = ?")
            params.append(asr_provider)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY w.file_name, w.segment_id, w.start_time"

        with EVSDataUtils.get_db_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    @staticmethod
    def get_asr_pair_evs(interpret_file: str, asr_provider: str = None) -> pd.DataFrame:
        """Get EVS pair data from database.

        Returns pairs where pair_type='S' is source and pair_type='E' is target,
        regardless of language or provider (supports different providers for each language).

        Args:
            interpret_file: The interpretation file name
            asr_provider: Optional ASR provider filter (if None, returns pairs from all providers)
        """
        try:          
            with EVSDataUtils.get_db_connection() as conn:

                # Query that supports different providers for source and target
                query = """
                    SELECT
                    s.pair_seq,
                    s.start_time AS en_start_time,
                    t.start_time AS zh_start_time,
                    COALESCE(s.edit_word, s.word) AS en_edit_word,
                    COALESCE(t.edit_word, t.word) AS zh_edit_word,
                    (t.start_time - s.start_time) AS EVS,
                    s.segment_id AS en_segment_id,
                    s.word_seq_no AS en_word_seq_no,
                    t.segment_id AS zh_segment_id,
                    t.word_seq_no AS zh_word_seq_no,
                    s.lang AS source_lang,
                    t.lang AS target_lang,
                    s.asr_provider AS source_provider,
                    t.asr_provider AS target_provider
                FROM asr_results_words s
                INNER JOIN asr_results_words t
                    ON s.file_name = t.file_name
                    AND s.pair_seq = t.pair_seq
                WHERE s.file_name = ?
                    AND s.pair_type = 'S'
                    AND t.pair_type = 'E'
                ORDER BY s.pair_seq;
                """
                return pd.read_sql_query(query, conn, params=[interpret_file])
        except Exception as e:
            logger.error(f"获取配对数据失败: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def debug_word_records(file_name, word=None, lang=None):
        """查询数据库中的单词记录，用于调试。

        参数:
            file_name: 文件名
            word: 可选，要查询的特定单词
            lang: 可选，语言 ('en' 或 'zh')

        返回:
            DataFrame: 包含查询结果的DataFrame
        """
        try:
            with EVSDataUtils.get_db_connection() as conn:
                query = """
                SELECT
                    file_name,
                    lang,
                    start_time,
                    end_time,
                    edit_word,
                    pair_type,
                    pair_seq
                FROM asr_results_words
                WHERE file_name = ?
                """

                params = [file_name]

                if word:
                    query += " AND edit_word = ?"
                    params.append(word)

                if lang:
                    query += " AND lang = ?"
                    params.append(lang)

                # 添加排序
                query += " ORDER BY lang, start_time"

                df = pd.read_sql_query(query, conn, params=params)
                return df

        except Exception as e:
            logger.error(f"查询数据库记录时出错: {str(e)}", exc_info=True)
            return pd.DataFrame()

    def get_reference_and_target_corpus(target_file: str, reference_file: str, lang: str, asr_provider: str) -> pd.DataFrame:
        if target_file == reference_file:
            logger.error("Target and reference files must be different")
            return

        try:
            # Build target query with parameterized conditions
            target_query = """
                SELECT edit_word, lang, COUNT(*) as frequency
                FROM asr_results_words
                WHERE edit_word IS NOT NULL
                AND asr_provider = ?
            """
            target_params = [asr_provider]

            if target_file != 'All':
                target_query += " AND file_name = ?"
                target_params.append(target_file)
            if lang != 'All':
                target_query += " AND lang = ?"
                target_params.append(lang)
            target_query += " GROUP BY edit_word, lang"

            # Build reference query with parameterized conditions
            ref_query = """
                SELECT edit_word, lang, COUNT(*) as frequency
                FROM asr_results_words
                WHERE edit_word IS NOT NULL
                AND asr_provider = ?
            """
            ref_params = [asr_provider]

            if reference_file != 'All':
                ref_query += " AND file_name = ?"
                ref_params.append(reference_file)
            if lang != 'All':
                ref_query += " AND lang = ?"
                ref_params.append(lang)
            ref_query += " GROUP BY edit_word, lang"

            # Execute queries with parameterized values
            with EVSDataUtils.get_db_connection() as conn:
                target_corpus = pd.read_sql_query(target_query, conn, params=target_params)
                reference_corpus = pd.read_sql_query(ref_query, conn, params=ref_params)

                if target_corpus.empty or reference_corpus.empty:
                    logger.error("No data found in one or both corpora")
                    return

                return target_corpus, reference_corpus
        except Exception as e:
            logger.error(f"Error getting reference and target corpus: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()

    @staticmethod
    def get_cluster_data(file_name: str, lang: str, asr_provider: str) -> pd.DataFrame:
        try:
            # Get data from database
            query = """
                SELECT *
                FROM asr_results_words
                WHERE edit_word IS NOT NULL
                AND asr_provider = ?
            """

            params = []
            conditions = []

            if file_name != 'All':
                conditions.append("file_name = ?")
                params.append(file_name)

            if lang != 'All':
                conditions.append("lang = ?")
                params.append(lang)

            if conditions:
                query += " AND " + " AND ".join(conditions)

            query += " ORDER BY file_name, segment_id, word_seq_no"

            # Execute query
            with EVSDataUtils.get_db_connection() as conn:
                df = pd.read_sql_query(query, conn, params=[asr_provider] + params)

                return df
        except Exception as e:
            logger.error(f"Error getting cluster data: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def save_asr_config(provider, config_data):
        """保存ASR提供商的配置信息

        参数:
            provider: ASR提供商 (如 'crisperwhisper', 'funasr', 'google', 'tencent', 'ibm')
            config_data: 配置数据字典

        返回:
            bool: 保存是否成功
        """
        try:
            with EVSDataUtils.get_db_connection() as conn:
                # 先检查配置是否已存在
                c = conn.cursor()
                c.execute("SELECT COUNT(*) FROM asr_config WHERE provider = ?", (provider,))
                exists = c.fetchone()[0] > 0

                config_json = json.dumps(config_data, ensure_ascii=False)

                if exists:
                    # 更新现有配置
                    c.execute(
                        "UPDATE asr_config SET config = ?, updated_at = ? WHERE provider = ?",
                        (config_json, datetime.now().isoformat(), provider)
                    )
                else:
                    # 插入新配置
                    c.execute(
                        "INSERT INTO asr_config (provider, config, created_at, updated_at) VALUES (?, ?, ?, ?)",
                        (provider, config_json, datetime.now().isoformat(), datetime.now().isoformat())
                    )

                conn.commit()
                logger.info(f"ASR配置已保存: {provider}")
                return True

        except Exception as e:
            logger.error(f"保存ASR配置失败: {str(e)}", exc_info=True)
            return False

    @staticmethod
    def get_asr_config(provider=None):
        """获取ASR提供商的配置信息

        参数:
            provider: 可选, ASR提供商名称。如果为None，返回所有配置

        返回:
            dict: 配置数据字典，如果provider为None则返回所有配置的字典
        """
        try:
            with EVSDataUtils.get_db_connection() as conn:
                c = conn.cursor()

                if provider:
                    # 获取指定提供商的配置
                    c.execute("SELECT provider, config FROM asr_config WHERE provider = ?", (provider,))
                    row = c.fetchone()

                    if row:
                        return {
                            "provider": row[0],
                            "config": json.loads(row[1])
                        }
                    return None
                else:
                    # 获取所有配置
                    c.execute("SELECT provider, config FROM asr_config")
                    rows = c.fetchall()

                    result = {}
                    for row in rows:
                        result[row[0]] = json.loads(row[1])

                    return result

        except Exception as e:
            logger.error(f"获取ASR配置失败: {str(e)}", exc_info=True)
            return {} if provider is None else None

    # 同声传译分析相关方法
    @staticmethod
    def save_si_analysis_result(file_name: str, asr_provider: str, analysis_type: str,
                              analysis_data: Dict[str, Any], created_by: str = None) -> int:
        """
        保存同声传译分析结果到数据库

        Args:
            file_name: 文件名
            asr_provider: ASR提供商
            analysis_type: 分析类型 ('quality', 'errors', 'timing', 'cultural', 'suggestions', 'corrected')
            analysis_data: 分析结果数据
            created_by: 创建者

        Returns:
            int: 插入记录的ID，失败返回-1
        """
        try:
            with EVSDataUtils.get_db_connection() as conn:
                cursor = conn.cursor()

                # 准备插入数据 - 使用REPLACE INTO来处理唯一键冲突
                insert_query = """
                REPLACE INTO si_analysis_results (
                    file_name, asr_provider, analysis_type, analysis_timestamp,
                    overall_score, accuracy_score, fluency_score, completeness_score, quality_level,
                    en_wpm, zh_wpm, speed_ratio, pace_assessment, balance_assessment,
                    bilingual_segments, coverage_rate,
                    confidence_mean, confidence_std, confidence_words_count,
                    total_errors, error_statistics, error_density,
                    average_delay, max_delay, sync_quality, sync_issue_count,
                    adaptation_score, cultural_issue_count, adaptation_level,
                    analysis_config, analysis_results, total_segments,
                    processing_time_ms, llm_model, created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

                # 从analysis_data中提取字段值
                cursor.execute(insert_query, [
                    file_name,
                    asr_provider,
                    analysis_type,
                    analysis_data.get('analysis_timestamp') or datetime.now().isoformat(),
                    analysis_data.get('overall_score'),
                    analysis_data.get('accuracy_score'),
                    analysis_data.get('fluency_score'),
                    analysis_data.get('completeness_score'),
                    analysis_data.get('quality_level'),
                    # Speech rate metrics
                    analysis_data.get('en_wpm'),
                    analysis_data.get('zh_wpm'),
                    analysis_data.get('speed_ratio'),
                    analysis_data.get('pace_assessment'),
                    analysis_data.get('balance_assessment'),
                    # Coverage metrics
                    analysis_data.get('bilingual_segments'),
                    analysis_data.get('coverage_rate'),
                    # Confidence metrics
                    analysis_data.get('confidence_mean'),
                    analysis_data.get('confidence_std'),
                    analysis_data.get('confidence_words_count'),
                    # Legacy fields
                    analysis_data.get('total_errors'),
                    json.dumps(analysis_data.get('error_statistics', {})),
                    analysis_data.get('error_density'),
                    analysis_data.get('average_delay'),
                    analysis_data.get('max_delay'),
                    analysis_data.get('sync_quality'),
                    analysis_data.get('sync_issue_count'),
                    analysis_data.get('adaptation_score'),
                    analysis_data.get('cultural_issue_count'),
                    analysis_data.get('adaptation_level'),
                    # Configuration and results
                    json.dumps(analysis_data.get('analysis_config', {})),
                    json.dumps(analysis_data.get('analysis_results', {})),
                    analysis_data.get('total_segments'),
                    analysis_data.get('processing_time_ms'),
                    analysis_data.get('llm_model'),
                    created_by
                ])

                analysis_id = cursor.lastrowid
                conn.commit()

                logger.info(f"成功保存分析结果，ID: {analysis_id}")
                return analysis_id

        except Exception as e:
            logger.error(f"保存分析结果失败: {str(e)}", exc_info=True)
            return -1

    @staticmethod
    def save_si_error_details(analysis_id: int, errors: List[Dict[str, Any]]) -> bool:
        """
        保存错误详情到数据库

        Args:
            analysis_id: 分析结果ID
            errors: 错误列表

        Returns:
            bool: 保存是否成功
        """
        try:
            with EVSDataUtils.get_db_connection() as conn:
                cursor = conn.cursor()

                insert_query = """
                INSERT INTO si_error_details (
                    analysis_id, error_type, severity, segment_index,
                    timestamp, source_text, target_text, description, suggestion
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

                for error in errors:
                    cursor.execute(insert_query, [
                        analysis_id,
                        error.get('type'),
                        error.get('severity'),
                        error.get('segment_index'),
                        error.get('timestamp'),
                        error.get('source'),
                        error.get('target'),
                        error.get('description'),
                        error.get('suggestion', '')
                    ])

                conn.commit()
                logger.info(f"成功保存 {len(errors)} 个错误详情")
                return True

        except Exception as e:
            logger.error(f"保存错误详情失败: {str(e)}", exc_info=True)
            return False

    @staticmethod
    def save_si_cultural_issues(analysis_id: int, issues: List[Dict[str, Any]]) -> bool:
        """
        保存文化问题到数据库

        Args:
            analysis_id: 分析结果ID
            issues: 文化问题列表

        Returns:
            bool: 保存是否成功
        """
        try:
            with EVSDataUtils.get_db_connection() as conn:
                cursor = conn.cursor()

                insert_query = """
                INSERT INTO si_cultural_issues (
                    analysis_id, issue_type, severity, segment_index,
                    timestamp, description, suggestion
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """

                for issue in issues:
                    cursor.execute(insert_query, [
                        analysis_id,
                        issue.get('type'),
                        issue.get('severity'),
                        issue.get('segment_index'),
                        issue.get('timestamp'),
                        issue.get('description'),
                        issue.get('suggestion', '')
                    ])

                conn.commit()
                logger.info(f"成功保存 {len(issues)} 个文化问题")
                return True

        except Exception as e:
            logger.error(f"保存文化问题失败: {str(e)}", exc_info=True)
            return False

    @staticmethod
    def save_si_timing_issues(analysis_id: int, timing_issues: List[Dict[str, Any]]) -> bool:
        """
        保存时间同步问题到数据库

        Args:
            analysis_id: 分析结果ID
            timing_issues: 时间同步问题列表

        Returns:
            bool: 保存是否成功
        """
        try:
            with EVSDataUtils.get_db_connection() as conn:
                cursor = conn.cursor()

                insert_query = """
                INSERT INTO si_timing_issues (
                    analysis_id, segment_index, timestamp, delay, severity, description
                ) VALUES (?, ?, ?, ?, ?, ?)
                """

                for issue in timing_issues:
                    cursor.execute(insert_query, [
                        analysis_id,
                        issue.get('segment_index'),
                        issue.get('timestamp'),
                        issue.get('delay'),
                        issue.get('severity'),
                        issue.get('description')
                    ])

                conn.commit()
                logger.info(f"成功保存 {len(timing_issues)} 个时间同步问题")
                return True

        except Exception as e:
            logger.error(f"保存时间同步问题失败: {str(e)}", exc_info=True)
            return False

    @staticmethod
    def get_si_analysis_results(file_name: str = None, asr_provider: str = None,
                               analysis_type: str = None, limit: int = 100) -> pd.DataFrame:
        """
        获取同声传译分析结果

        Args:
            file_name: 可选，文件名筛选
            asr_provider: 可选，ASR提供商筛选
            analysis_type: 可选，分析类型筛选
            limit: 返回记录数限制

        Returns:
            DataFrame: 分析结果数据
        """
        try:
            with EVSDataUtils.get_db_connection() as conn:
                query = """
                SELECT * FROM si_analysis_results
                WHERE 1=1
                """
                params = []

                if file_name:
                    query += " AND file_name = ?"
                    params.append(file_name)

                if asr_provider:
                    query += " AND asr_provider = ?"
                    params.append(asr_provider)

                if analysis_type:
                    query += " AND analysis_type = ?"
                    params.append(analysis_type)

                query += " ORDER BY analysis_timestamp DESC LIMIT ?"
                params.append(limit)

                return pd.read_sql_query(query, conn, params=params)

        except Exception as e:
            logger.error(f"获取分析结果失败: {str(e)}", exc_info=True)
            return pd.DataFrame()

    @staticmethod
    def get_si_error_details(analysis_id: int) -> pd.DataFrame:
        """
        获取特定分析的错误详情

        Args:
            analysis_id: 分析结果ID

        Returns:
            DataFrame: 错误详情数据
        """
        try:
            with EVSDataUtils.get_db_connection() as conn:
                query = """
                SELECT * FROM si_error_details
                WHERE analysis_id = ?
                ORDER BY segment_index, timestamp
                """
                return pd.read_sql_query(query, conn, params=[analysis_id])

        except Exception as e:
            logger.error(f"获取错误详情失败: {str(e)}", exc_info=True)
            return pd.DataFrame()

    @staticmethod
    def save_si_analysis_config(config_name: str, config_type: str, config_data: Dict[str, Any],
                               created_by: str = None) -> bool:
        """
        保存分析配置

        Args:
            config_name: 配置名称
            config_type: 配置类型
            config_data: 配置数据
            created_by: 创建者

        Returns:
            bool: 保存是否成功
        """
        try:
            with EVSDataUtils.get_db_connection() as conn:
                cursor = conn.cursor()

                # 检查配置是否已存在
                check_query = "SELECT COUNT(*) FROM si_analysis_config WHERE config_name = ?"
                cursor.execute(check_query, [config_name])
                exists = cursor.fetchone()[0] > 0

                if exists:
                    # 更新现有配置
                    update_query = """
                    UPDATE si_analysis_config
                    SET config_type = ?, config_data = ?, updated_at = ?
                    WHERE config_name = ?
                    """
                    cursor.execute(update_query, [
                        config_type,
                        json.dumps(config_data, ensure_ascii=False),
                        datetime.now().isoformat(),
                        config_name
                    ])
                else:
                    # 插入新配置
                    insert_query = """
                    INSERT INTO si_analysis_config (
                        config_name, config_type, config_data, created_by
                    ) VALUES (?, ?, ?, ?)
                    """
                    cursor.execute(insert_query, [
                        config_name,
                        config_type,
                        json.dumps(config_data, ensure_ascii=False),
                        created_by
                    ])

                conn.commit()
                logger.info(f"成功保存分析配置: {config_name}")
                return True

        except Exception as e:
            logger.error(f"保存分析配置失败: {str(e)}", exc_info=True)
            return False

    @staticmethod
    def get_si_analysis_config(config_name: str = None, config_type: str = None) -> pd.DataFrame:
        """
        获取分析配置

        Args:
            config_name: 可选，配置名称筛选
            config_type: 可选，配置类型筛选

        Returns:
            DataFrame: 配置数据
        """
        try:
            with EVSDataUtils.get_db_connection() as conn:
                query = """
                SELECT * FROM si_analysis_config
                WHERE is_active = 1
                """
                params = []

                if config_name:
                    query += " AND config_name = ?"
                    params.append(config_name)

                if config_type:
                    query += " AND config_type = ?"
                    params.append(config_type)

                query += " ORDER BY created_at DESC"

                return pd.read_sql_query(query, conn, params=params)

        except Exception as e:
            logger.error(f"获取分析配置失败: {str(e)}", exc_info=True)
            return pd.DataFrame()

    @staticmethod
    def delete_si_analysis_result(analysis_id: int) -> bool:
        """
        删除分析结果及相关数据

        Args:
            analysis_id: 分析结果ID

        Returns:
            bool: 删除是否成功
        """
        try:
            with EVSDataUtils.get_db_connection() as conn:
                cursor = conn.cursor()

                # 删除相关的详细记录
                cursor.execute("DELETE FROM si_error_details WHERE analysis_id = ?", [analysis_id])
                cursor.execute("DELETE FROM si_cultural_issues WHERE analysis_id = ?", [analysis_id])
                cursor.execute("DELETE FROM si_timing_issues WHERE analysis_id = ?", [analysis_id])
                cursor.execute("DELETE FROM si_corrected_versions WHERE original_analysis_id = ?", [analysis_id])

                # 删除主记录
                cursor.execute("DELETE FROM si_analysis_results WHERE id = ?", [analysis_id])

                conn.commit()
                logger.info(f"成功删除分析结果: {analysis_id}")
                return True

        except Exception as e:
            logger.error(f"删除分析结果失败: {str(e)}", exc_info=True)
            return False

    @staticmethod
    def get_si_analysis_summary(file_name: str = None, asr_provider: str = None) -> Dict[str, Any]:
        """
        获取分析结果汇总统计

        Args:
            file_name: 可选，文件名筛选
            asr_provider: 可选，ASR提供商筛选

        Returns:
            Dict: 汇总统计数据
        """
        try:
            with EVSDataUtils.get_db_connection() as conn:
                # 基础查询条件
                where_clause = "WHERE 1=1"
                params = []

                if file_name:
                    where_clause += " AND file_name = ?"
                    params.append(file_name)

                if asr_provider:
                    where_clause += " AND asr_provider = ?"
                    params.append(asr_provider)

                # 获取各类分析的统计信息
                query = f"""
                SELECT
                    analysis_type,
                    COUNT(*) as count,
                    AVG(overall_score) as avg_score,
                    MAX(analysis_timestamp) as latest_analysis
                FROM si_analysis_results
                {where_clause}
                GROUP BY analysis_type
                """

                summary_df = pd.read_sql_query(query, conn, params=params)

                # 获取总体统计
                total_query = f"""
                SELECT
                    COUNT(DISTINCT file_name) as total_files,
                    COUNT(*) as total_analyses,
                    AVG(overall_score) as avg_overall_score
                FROM si_analysis_results
                {where_clause}
                """

                total_stats = pd.read_sql_query(total_query, conn, params=params)

                return {
                    'analysis_breakdown': summary_df.to_dict('records'),
                    'total_statistics': total_stats.to_dict('records')[0] if not total_stats.empty else {}
                }

        except Exception as e:
            logger.error(f"获取分析汇总失败: {str(e)}", exc_info=True)
            return {}

    @staticmethod
    def get_processed_files() -> List[str]:
        """
        获取已处理的文件列表

        Returns:
            List[str]: 已处理的文件名列表
        """
        # Use cached version if available
        if USE_CACHE:
            return get_cached_processed_files()

        try:
            with EVSDataUtils.get_db_connection() as conn:
                query = "SELECT DISTINCT file_name FROM asr_results_words ORDER BY file_name"
                result = pd.read_sql_query(query, conn)
                return result['file_name'].tolist()
        except Exception as e:
            logger.error(f"获取已处理文件列表失败: {str(e)}")
            return []

    @staticmethod
    def get_all_asr_results(asr_providers: List[str] = None) -> pd.DataFrame:
        """
        获取所有ASR识别结果

        Args:
            asr_providers: ASR提供商列表，如果为None则获取所有

        Returns:
            pd.DataFrame: ASR结果数据
        """
        try:
            with EVSDataUtils.get_db_connection() as conn:
                query = """
                SELECT * FROM asr_results_words
                """
                params = []

                if asr_providers:
                    placeholders = ','.join(['?' for _ in asr_providers])
                    query += f" WHERE asr_provider IN ({placeholders})"
                    params.extend(asr_providers)

                query += " ORDER BY file_name, segment_id, word_seq_no"

                return pd.read_sql_query(query, conn, params=params)
        except Exception as e:
            logger.error(f"获取所有ASR结果失败: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def get_all_evs_pairs(asr_providers: List[str] = None) -> pd.DataFrame:
        """
        获取所有EVS配对数据

        Args:
            asr_providers: ASR提供商列表，如果为None则获取所有

        Returns:
            pd.DataFrame: EVS配对数据
        """
        try:
            with EVSDataUtils.get_db_connection() as conn:
                query = """
                SELECT * FROM asr_results_words
                WHERE pair_type IS NOT NULL AND pair_type != ''
                """
                params = []

                if asr_providers:
                    placeholders = ','.join(['?' for _ in asr_providers])
                    query += f" AND asr_provider IN ({placeholders})"
                    params.extend(asr_providers)

                query += " ORDER BY file_name, pair_seq"

                return pd.read_sql_query(query, conn, params=params)
        except Exception as e:
            logger.error(f"获取所有EVS配对数据失败: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def get_all_edit_words(asr_providers: List[str] = None) -> pd.DataFrame:
        """
        获取所有编辑词汇数据

        Args:
            asr_providers: ASR提供商列表，如果为None则获取所有

        Returns:
            pd.DataFrame: 编辑词汇数据
        """
        try:
            with EVSDataUtils.get_db_connection() as conn:
                query = """
                SELECT * FROM asr_results_words
                WHERE edit_word IS NOT NULL AND edit_word != ''
                """
                params = []

                if asr_providers:
                    placeholders = ','.join(['?' for _ in asr_providers])
                    query += f" AND asr_provider IN ({placeholders})"
                    params.extend(asr_providers)

                query += " ORDER BY file_name, segment_id, word_seq_no"

                return pd.read_sql_query(query, conn, params=params)
        except Exception as e:
            logger.error(f"获取所有编辑词汇失败: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def get_all_nlp_results() -> pd.DataFrame:
        """
        获取所有NLP处理结果

        Returns:
            pd.DataFrame: NLP结果数据
        """
        try:
            with EVSDataUtils.get_db_connection() as conn:
                # 检查chinese_nlp_results表是否存在
                check_table_query = """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='chinese_nlp_results'
                """
                table_exists = pd.read_sql_query(check_table_query, conn)

                if table_exists.empty:
                    logger.warning("chinese_nlp_results表不存在")
                    return pd.DataFrame()

                query = """
                SELECT * FROM chinese_nlp_results
                ORDER BY file_name, segment_id
                """
                return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"获取所有NLP结果失败: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def get_all_si_analysis_results(asr_providers: List[str] = None) -> pd.DataFrame:
        """
        获取所有同传分析结果

        Args:
            asr_providers: ASR提供商列表，如果为None则获取所有

        Returns:
            pd.DataFrame: 同传分析结果数据
        """
        try:
            with EVSDataUtils.get_db_connection() as conn:
                query = """
                SELECT * FROM si_analysis_results
                """
                params = []

                if asr_providers:
                    placeholders = ','.join(['?' for _ in asr_providers])
                    query += f" WHERE asr_provider IN ({placeholders})"
                    params.extend(asr_providers)

                query += " ORDER BY analysis_timestamp DESC"

                return pd.read_sql_query(query, conn, params=params)
        except Exception as e:
            logger.error(f"获取所有同传分析结果失败: {str(e)}")
            return pd.DataFrame()


# Module-level functions for audio analysis
def save_analysis(analysis_data: Dict[str, Any]) -> bool:
    """Wrapper for DatabaseUtils.save_analysis"""
    return EVSDataUtils.save_analysis(analysis_data)

def get_analysis_history() -> List[Dict[str, Any]]:
    """Wrapper for DatabaseUtils.get_analysis_history"""
    return EVSDataUtils.get_analysis_history()

@staticmethod
def read_query(query: str) -> pd.DataFrame:
    """Execute a SELECT query and return results as DataFrame"""
    try:
        conn = sqlite3.connect('interpret.db', check_same_thread=False)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        raise e

