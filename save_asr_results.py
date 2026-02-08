import sqlite3
import pandas as pd
import logging
from pathlib import Path
import streamlit as st

logger = logging.getLogger(__name__)

# Import cache utilities
try:
    from cache_utils import clear_cache
    USE_CACHE = True
except ImportError:
    USE_CACHE = False
    logger.warning("Cache utilities not found, cache clearing disabled")

# Constants
REPOSITORY_ROOT = "./data"

def get_db_path() -> Path:
    return Path(REPOSITORY_ROOT) / 'evs_repository.db'

def get_db_connection():
    """Create and return a database connection."""
    try:
        conn = sqlite3.connect(get_db_path())
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {str(e)}")
        raise

def save_asr_result_to_database(words_df, segments_df):
    """
    Save ASR results to database, using file_name, slice_number, and asr_provider as composite key.

    Args:
        words_df: DataFrame containing word-level results
        segments_df: DataFrame containing segment-level results
        asr_provider: String identifying the ASR provider (e.g., 'crisperwhisper', 'funasr', 'google')

    Returns:
        bool: True if save successful, False otherwise
    """
    try:
        # Check if DataFrame is empty or missing required columns
        if words_df is None or words_df.empty:
            logger.warning("Words DataFrame is empty, no data to save")
            return

        # Check if 'word' column exists
        if 'word' not in words_df.columns:
            logger.error("Words DataFrame is missing the 'word' column")
            st.error("Transcription data format error: 'word' column not found")
            return

        # Initialize edit_word column with the original word
        words_df['edit_word'] = words_df['word']
        segments_df['edit_text'] = segments_df['text']

        # Get asr_provider, file_name and lang from the DataFrame
        asr_provider = words_df['asr_provider'].iloc[0]
        file_name = words_df['file_name'].iloc[0]
        lang = words_df['lang'].iloc[0]
        words_df['asr_provider'] = asr_provider
        segments_df['asr_provider'] = asr_provider

        with get_db_connection() as conn:
            # Start transaction
            conn.execute("BEGIN TRANSACTION")


            # Delete existing records for this file and provider at file level
            conn.execute(
                "DELETE FROM asr_results_words WHERE file_name = ? AND asr_provider = ? AND lang = ?",
                (file_name, asr_provider, lang)
            )
            conn.execute(
                "DELETE FROM asr_results_segments WHERE file_name = ? AND asr_provider = ? AND lang = ?",
                (file_name, asr_provider, lang)
            )
            logger.info(f"Deleted existing records for file: {file_name}, provider: {asr_provider}, language: {lang}")

            # Save new words data
            words_df.to_sql('asr_results_words',
                          conn,
                          if_exists='append',
                          index=False)

            # Save new segments data
            segments_df.to_sql('asr_results_segments',
                             conn,
                             if_exists='append',
                             index=False)

            # Commit transaction
            conn.commit()
            logger.info(f"{asr_provider} ASR results saved successfully for file: {file_name}")

            # Register file in asr_files table for fast lookup
            try:
                from db_utils import EVSDataUtils
                model = words_df['model'].iloc[0] if 'model' in words_df.columns else None
                slice_duration = words_df['slice_duration'].iloc[0] if 'slice_duration' in words_df.columns else None
                total_segments = len(segments_df) if segments_df is not None else 0
                total_words = len(words_df)

                EVSDataUtils.register_asr_file(
                    file_name=file_name,
                    lang=lang,
                    asr_provider=asr_provider,
                    model=model,
                    slice_duration=slice_duration,
                    total_segments=total_segments,
                    total_words=total_words
                )
                logger.info(f"Registered file in asr_files: {file_name}, {lang}, {asr_provider}/{model}")
            except Exception as reg_e:
                logger.warning(f"Failed to register file in asr_files table: {reg_e}")

            # Clear cache so new data is immediately available
            if USE_CACHE:
                clear_cache('all')
                logger.info("Cache cleared after saving ASR results")

            return True

    except Exception as e:
        logger.error(f"Error saving ASR results to database: {str(e)}")
        st.error(f"Failed to save ASR results to database: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
        return False

def get_asr_results(file_name=None, lang=None, asr_provider=None):
    """
    Get ASR transcription results.

    Args:
        file_name: File name (optional)
        lang: Language code (optional)
        asr_provider: ASR provider name (optional)

    Returns:
        tuple: (words_df, segments_df) containing word-level and segment-level results
    """
    try:
        with get_db_connection() as conn:
            # Build base queries
            words_query = """
            SELECT * FROM asr_results_words
            WHERE 1=1
            """
            segments_query = """
            SELECT * FROM asr_results_segments
            WHERE 1=1
            """

            params = []

            # Add filter conditions
            if file_name:
                words_query += " AND file_name = ?"
                segments_query += " AND file_name = ?"
                params.append(file_name)

            if lang:
                words_query += " AND lang = ?"
                segments_query += " AND lang = ?"
                params.append(lang)

            if asr_provider:
                words_query += " AND asr_provider = ?"
                segments_query += " AND asr_provider = ?"
                params.append(asr_provider)

            # Add sorting
            words_query += " ORDER BY start_time, word_seq_no"
            segments_query += " ORDER BY start_time, segment_id"

            # Execute queries
            words_df = pd.read_sql_query(words_query, conn, params=params)
            segments_df = pd.read_sql_query(segments_query, conn, params=params)

            return words_df, segments_df

    except Exception as e:
        logger.error(f"Error retrieving ASR results: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def update_edit_word(file_name, lang, segment_id, word_seq_no, edit_word, asr_provider):
    """
    Update edited word in database.

    Args:
        file_name: File name
        lang: Language code
        segment_id: Segment ID
        word_seq_no: Word sequence number
        edit_word: Edited word text
        asr_provider: ASR provider name

    Returns:
        bool: True if update successful, False otherwise
    """
    try:
        with get_db_connection() as conn:
            query = """
            UPDATE asr_results_words
            SET edit_word = ?
            WHERE file_name = ?
                AND lang = ?
                AND segment_id = ?
                AND word_seq_no = ?
                AND asr_provider = ?
            """
            conn.execute(query, [
                edit_word,
                file_name,
                lang,
                segment_id,
                word_seq_no,
                asr_provider
            ])
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Error updating edit word: {str(e)}")
        return False