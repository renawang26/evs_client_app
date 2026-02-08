"""
Database Configuration Module

Provides centralized database path configuration for the EVS Navigation System.
All database-related paths should be imported from this module to ensure consistency.
"""

import os
from pathlib import Path

# Project root is relative to this config file (config/database_config.py -> project root)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Data directory - use local ./data directory
DATA_DIR = PROJECT_ROOT / "data"

# Database file path
DB_PATH = DATA_DIR / "evs_repository.db"

# Schema file path
SCHEMA_FILE = PROJECT_ROOT / "schema" / "evs_schema.sql"


def get_db_path() -> Path:
    """
    Get the database file path, ensuring the data directory exists.

    Returns:
        Path: Absolute path to the database file
    """
    ensure_data_dir()
    return DB_PATH


def get_db_path_str() -> str:
    """
    Get the database file path as a string.

    Returns:
        str: Absolute path to the database file as string
    """
    return str(get_db_path())


def ensure_data_dir() -> None:
    """
    Ensure the data directory exists, creating it if necessary.
    """
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_schema_file() -> Path:
    """
    Get the schema file path.

    Returns:
        Path: Path to the schema SQL file
    """
    if SCHEMA_FILE.exists():
        return SCHEMA_FILE
    else:
        raise FileNotFoundError(f"Schema file not found. Expected at {SCHEMA_FILE}")


# For backwards compatibility - export as REPOSITORY_ROOT
# This allows existing code to import REPOSITORY_ROOT and get the data directory
REPOSITORY_ROOT = str(DATA_DIR)


__all__ = [
    'PROJECT_ROOT',
    'DATA_DIR',
    'DB_PATH',
    'SCHEMA_FILE',
    'REPOSITORY_ROOT',
    'get_db_path',
    'get_db_path_str',
    'ensure_data_dir',
    'get_schema_file'
]
