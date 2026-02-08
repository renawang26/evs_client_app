import sqlite3
import pandas as pd
import logging
from pathlib import Path
import os
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("migration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("migration")

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

def create_asr_results_tables():
    """Create the new asr_results_words table."""
    logger.info("Creating asr_results_words table...")

    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS asr_results_words (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asr_provider TEXT,
                file_name TEXT NOT NULL,
                slice_number INTEGER,
                lang TEXT NOT NULL,
                word TEXT,
                edit_word TEXT,
                start_time REAL,
                end_time REAL,
                confidence REAL,
                speaker TEXT,
                segment_id INTEGER,
                word_seq_no INTEGER,
                pair_type TEXT,
                annotate TEXT,
                pair_seq INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for better performance
        conn.execute("CREATE INDEX IF NOT EXISTS idx_asr_file_name ON asr_results_words(file_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_asr_lang ON asr_results_words(lang)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_asr_provider ON asr_results_words(asr_provider)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_asr_pair_seq ON asr_results_words(pair_seq)")

        logger.info("Table and indexes created successfully")

def migrate_data():
    """Migrate data from pd_interpret_words to asr_results_words."""
    logger.info("Starting data migration...")

    try:
        with get_db_connection() as conn:
            # Get the total count for progress tracking
            count = conn.execute("SELECT COUNT(*) FROM pd_interpret_words").fetchone()[0]
            logger.info(f"Total records to migrate: {count}")

            # Start a transaction
            conn.execute("BEGIN TRANSACTION")

            # Get all unique interpret_file values
            files = conn.execute("SELECT DISTINCT interpret_file FROM pd_interpret_words").fetchall()

            total_migrated = 0

            # Process each file separately to manage memory usage
            for file_row in files:
                interpret_file = file_row[0]
                logger.info(f"Processing file: {interpret_file}")

                # Get data for this file
                query = """
                SELECT
                    interpret_file,
                    service_provider,
                    lang,
                    word,
                    edit_word,
                    speak_start_time,
                    speak_end_time,
                    NULL as confidence,
                    NULL as speaker,
                    sentence_seq_no as segment_id,
                    word_seq_no,
                    pair_type,
                    annotate,
                    pair_seq,
                    0 as slice_number
                FROM pd_interpret_words
                WHERE interpret_file = ?
                """

                cursor = conn.execute(query, (interpret_file,))

                # Process in batches
                batch_size = 1000
                batch = []

                for row in cursor:
                    # Convert row to dict
                    record = {
                        'file_name': row[0],
                        'asr_provider': row[1],
                        'lang': row[2],
                        'word': row[3],
                        'edit_word': row[4],
                        'start_time': row[5],
                        'end_time': row[6],
                        'confidence': row[7],
                        'speaker': row[8],
                        'segment_id': row[9],
                        'word_seq_no': row[10],
                        'pair_type': row[11],
                        'annotate': row[12],
                        'pair_seq': row[13],
                        'slice_number': row[14]
                    }

                    batch.append(record)

                    if len(batch) >= batch_size:
                        # Insert batch
                        placeholders = ', '.join(['?'] * 15)
                        insert_query = f"""
                        INSERT INTO asr_results_words (
                            file_name, asr_provider, lang, word, edit_word,
                            start_time, end_time, confidence, speaker, segment_id,
                            word_seq_no, pair_type, annotate, pair_seq, slice_number
                        ) VALUES ({placeholders})
                        """

                        conn.executemany(
                            insert_query,
                            [(
                                r['file_name'], r['asr_provider'], r['lang'], r['word'], r['edit_word'],
                                r['start_time'], r['end_time'], r['confidence'], r['speaker'], r['segment_id'],
                                r['word_seq_no'], r['pair_type'], r['annotate'], r['pair_seq'], r['slice_number']
                            ) for r in batch]
                        )

                        total_migrated += len(batch)
                        logger.info(f"Migrated {total_migrated}/{count} records ({(total_migrated/count)*100:.2f}%)")

                        batch = []

                # Insert any remaining records
                if batch:
                    placeholders = ', '.join(['?'] * 15)
                    insert_query = f"""
                    INSERT INTO asr_results_words (
                        file_name, asr_provider, lang, word, edit_word,
                        start_time, end_time, confidence, speaker, segment_id,
                        word_seq_no, pair_type, annotate, pair_seq, slice_number
                    ) VALUES ({placeholders})
                    """

                    conn.executemany(
                        insert_query,
                        [(
                            r['file_name'], r['asr_provider'], r['lang'], r['word'], r['edit_word'],
                            r['start_time'], r['end_time'], r['confidence'], r['speaker'], r['segment_id'],
                            r['word_seq_no'], r['pair_type'], r['annotate'], r['pair_seq'], r['slice_number']
                        ) for r in batch]
                    )

                    total_migrated += len(batch)
                    logger.info(f"Migrated {total_migrated}/{count} records ({(total_migrated/count)*100:.2f}%)")

            # Commit the transaction
            conn.commit()
            logger.info(f"Migration completed successfully. Total records migrated: {total_migrated}")

    except Exception as e:
        logger.error(f"Error during migration: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
        raise

def main():
    start_time = time.time()
    logger.info("Starting migration process...")

    try:
        # Create the new table
        create_asr_results_tables()

        # Migrate the data
        migrate_data()

        elapsed_time = time.time() - start_time
        logger.info(f"Migration completed in {elapsed_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        elapsed_time = time.time() - start_time
        logger.info(f"Migration failed after {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()