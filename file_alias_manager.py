#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Alias Manager - Privacy Protection Module
文件名别名管理器 - 隐私保护模块

This module provides functionality to create and manage anonymized file name aliases
to protect user privacy in the interpretation analysis system.
"""

import sqlite3
import hashlib
from typing import Dict, Optional, List, Tuple
import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class FileAliasManager:
    """Manager for file name anonymization and alias mapping"""

    def __init__(self, db_path: str = "./data/evs_repository.db"):
        self.db_path = db_path
        self._initialize_alias_table()

    def _initialize_alias_table(self):
        """Initialize the file alias mapping table in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create alias mapping table if not exists
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS file_alias_mapping (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        original_filename TEXT UNIQUE NOT NULL,
                        alias_name TEXT UNIQUE NOT NULL,
                        file_hash TEXT UNIQUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1
                    )
                """)

                # Create index for faster lookups
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_original_filename
                    ON file_alias_mapping(original_filename)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_alias_name
                    ON file_alias_mapping(alias_name)
                """)

                conn.commit()
                logger.info("File alias mapping table initialized successfully")

        except sqlite3.Error as e:
            logger.error(f"Error initializing alias table: {str(e)}")
            raise

    def _generate_file_hash(self, filename: str) -> str:
        """Generate a consistent hash for the filename"""
        return hashlib.sha256(filename.encode('utf-8')).hexdigest()[:16]

    def _generate_alias_name(self, filename: str) -> str:
        """Generate an alias name for the file"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Count existing aliases to determine next number
                cursor.execute("SELECT COUNT(*) FROM file_alias_mapping WHERE is_active = 1")
                count = cursor.fetchone()[0]

                # Generate alias in format: file_1, file_2, etc.
                alias_name = f"file_{count + 1}"

                # Ensure uniqueness (in case of gaps from deletions)
                while self._alias_exists(alias_name):
                    count += 1
                    alias_name = f"file_{count + 1}"

                return alias_name

        except sqlite3.Error as e:
            logger.error(f"Error generating alias name: {str(e)}")
            return f"file_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _alias_exists(self, alias_name: str) -> bool:
        """Check if alias name already exists"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM file_alias_mapping WHERE alias_name = ? AND is_active = 1",
                    (alias_name,)
                )
                return cursor.fetchone()[0] > 0
        except sqlite3.Error:
            return False

    def get_or_create_alias(self, original_filename: str) -> str:
        """
        Get existing alias or create new one for the given filename

        Args:
            original_filename: The original file name

        Returns:
            The alias name for the file
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Check if alias already exists
                cursor.execute(
                    "SELECT alias_name FROM file_alias_mapping WHERE original_filename = ? AND is_active = 1",
                    (original_filename,)
                )
                result = cursor.fetchone()

                if result:
                    # Update last accessed time
                    cursor.execute(
                        "UPDATE file_alias_mapping SET last_accessed = CURRENT_TIMESTAMP WHERE original_filename = ?",
                        (original_filename,)
                    )
                    conn.commit()
                    return result[0]

                # Create new alias
                alias_name = self._generate_alias_name(original_filename)
                file_hash = self._generate_file_hash(original_filename)

                cursor.execute("""
                    INSERT INTO file_alias_mapping (original_filename, alias_name, file_hash)
                    VALUES (?, ?, ?)
                """, (original_filename, alias_name, file_hash))

                conn.commit()
                logger.info(f"Created new alias '{alias_name}' for file '{original_filename}'")
                return alias_name

        except sqlite3.Error as e:
            logger.error(f"Error creating/retrieving alias: {str(e)}")
            # Fallback to a simple anonymized name
            return f"file_{abs(hash(original_filename)) % 10000}"

    def get_original_filename(self, alias_name: str) -> Optional[str]:
        """
        Get the original filename from alias

        Args:
            alias_name: The alias name

        Returns:
            The original filename or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT original_filename FROM file_alias_mapping WHERE alias_name = ? AND is_active = 1",
                    (alias_name,)
                )
                result = cursor.fetchone()
                return result[0] if result else None

        except sqlite3.Error as e:
            logger.error(f"Error retrieving original filename: {str(e)}")
            return None

    def get_all_aliases(self) -> List[Tuple[str, str, str]]:
        """
        Get all active file aliases

        Returns:
            List of tuples (alias_name, original_filename, created_at)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT alias_name, original_filename, created_at
                    FROM file_alias_mapping
                    WHERE is_active = 1
                    ORDER BY created_at DESC
                """)
                return cursor.fetchall()

        except sqlite3.Error as e:
            logger.error(f"Error retrieving aliases: {str(e)}")
            return []

    def get_alias_mapping_dict(self) -> Dict[str, str]:
        """
        Get mapping dictionary from original filename to alias

        Returns:
            Dictionary mapping original filename to alias name
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT original_filename, alias_name
                    FROM file_alias_mapping
                    WHERE is_active = 1
                """)
                return dict(cursor.fetchall())

        except sqlite3.Error as e:
            logger.error(f"Error retrieving alias mapping: {str(e)}")
            return {}

    def get_reverse_mapping_dict(self) -> Dict[str, str]:
        """
        Get reverse mapping dictionary from alias to original filename

        Returns:
            Dictionary mapping alias name to original filename
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT alias_name, original_filename
                    FROM file_alias_mapping
                    WHERE is_active = 1
                """)
                return dict(cursor.fetchall())

        except sqlite3.Error as e:
            logger.error(f"Error retrieving reverse mapping: {str(e)}")
            return {}

    def deactivate_alias(self, original_filename: str) -> bool:
        """
        Deactivate an alias (soft delete)

        Args:
            original_filename: The original filename to deactivate

        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE file_alias_mapping SET is_active = 0 WHERE original_filename = ?",
                    (original_filename,)
                )
                conn.commit()

                if cursor.rowcount > 0:
                    logger.info(f"Deactivated alias for file '{original_filename}'")
                    return True
                return False

        except sqlite3.Error as e:
            logger.error(f"Error deactivating alias: {str(e)}")
            return False

    def reactivate_alias(self, original_filename: str) -> bool:
        """
        Reactivate a deactivated alias

        Args:
            original_filename: The original filename to reactivate

        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE file_alias_mapping SET is_active = 1 WHERE original_filename = ?",
                    (original_filename,)
                )
                conn.commit()

                if cursor.rowcount > 0:
                    logger.info(f"Reactivated alias for file '{original_filename}'")
                    return True
                return False

        except sqlite3.Error as e:
            logger.error(f"Error reactivating alias: {str(e)}")
            return False

    def cleanup_old_aliases(self, days_old: int = 30) -> int:
        """
        Clean up old, inactive aliases

        Args:
            days_old: Number of days to consider as old

        Returns:
            Number of aliases cleaned up
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM file_alias_mapping
                    WHERE is_active = 0
                    AND last_accessed < datetime('now', '-{} days')
                """.format(days_old))

                deleted_count = cursor.rowcount
                conn.commit()

                logger.info(f"Cleaned up {deleted_count} old aliases")
                return deleted_count

        except sqlite3.Error as e:
            logger.error(f"Error cleaning up aliases: {str(e)}")
            return 0

    def get_file_statistics(self) -> Dict[str, int]:
        """
        Get statistics about file aliases

        Returns:
            Dictionary with statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Total active aliases
                cursor.execute("SELECT COUNT(*) FROM file_alias_mapping WHERE is_active = 1")
                active_count = cursor.fetchone()[0]

                # Total inactive aliases
                cursor.execute("SELECT COUNT(*) FROM file_alias_mapping WHERE is_active = 0")
                inactive_count = cursor.fetchone()[0]

                # Files accessed today
                cursor.execute("""
                    SELECT COUNT(*) FROM file_alias_mapping
                    WHERE is_active = 1 AND date(last_accessed) = date('now')
                """)
                accessed_today = cursor.fetchone()[0]

                return {
                    'total_active': active_count,
                    'total_inactive': inactive_count,
                    'accessed_today': accessed_today,
                    'total_files': active_count + inactive_count
                }

        except sqlite3.Error as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {'total_active': 0, 'total_inactive': 0, 'accessed_today': 0, 'total_files': 0}

def apply_alias_to_dataframe(df, filename_column: str = 'file_name') -> tuple:
    """
    Apply alias transformation to a pandas DataFrame

    Args:
        df: DataFrame containing file names
        filename_column: Name of the column containing file names

    Returns:
        Tuple of (modified_dataframe, alias_manager, mapping_dict)
    """
    if df.empty or filename_column not in df.columns:
        return df, None, {}

    alias_manager = FileAliasManager()
    mapping_dict = {}

    # Create aliases for unique filenames
    unique_files = df[filename_column].unique()
    for filename in unique_files:
        if filename and str(filename).strip():
            alias = alias_manager.get_or_create_alias(str(filename))
            mapping_dict[filename] = alias

    # Apply aliases to dataframe
    df_aliased = df.copy()
    df_aliased[filename_column] = df_aliased[filename_column].map(
        lambda x: mapping_dict.get(x, x) if x else x
    )

    return df_aliased, alias_manager, mapping_dict

def test_alias_manager():
    """Test function for the FileAliasManager"""
    print("🧪 Testing File Alias Manager")
    print("=" * 40)

    # Initialize manager
    manager = FileAliasManager()

    # Test file names (simulating real user files)
    test_files = [
        "BOOTH 2_1 Chen Huiwen.mp3",
        "BOOTH 2_2 Ni Chengjun.mp3",
        "BOOTH 6_2 Zheng Hanning.mp3",
        "Recording_John_Smith_20231201.mp3",
        "Meeting_ABC_Company_Internal.mp3"
    ]

    print("📝 Creating aliases for test files:")
    aliases = {}
    for filename in test_files:
        alias = manager.get_or_create_alias(filename)
        aliases[filename] = alias
        print(f"  {filename} → {alias}")

    print(f"\n📊 Statistics:")
    stats = manager.get_file_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print(f"\n🔄 Testing reverse lookup:")
    for original, alias in aliases.items():
        retrieved = manager.get_original_filename(alias)
        print(f"  {alias} → {retrieved} ({'✅' if retrieved == original else '❌'})")

    return manager

if __name__ == "__main__":
    # Run test if script is executed directly
    test_alias_manager()