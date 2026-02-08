#!/usr/bin/env python
"""
Database Initialization Script for EVS Navigation System

Usage:
    python init_database.py [--reset]

Options:
    --reset    Drop and recreate all tables (WARNING: destroys existing data)
"""

import sqlite3
import sys
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "evs_repository.db"
SCHEMA_FILE = PROJECT_ROOT / "create_evs_tables.sql"


def init_database(reset: bool = False):
    """Initialize the database with schema."""

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if reset and DB_PATH.exists():
        print(f"Removing existing database: {DB_PATH}")
        DB_PATH.unlink()

    # Check schema file exists
    if not SCHEMA_FILE.exists():
        print(f"ERROR: Schema file not found: {SCHEMA_FILE}")
        sys.exit(1)

    print(f"Initializing database: {DB_PATH}")

    # Connect and execute schema
    conn = sqlite3.connect(DB_PATH)
    try:
        with open(SCHEMA_FILE, 'r', encoding='utf-8') as f:
            conn.executescript(f.read())
        print("Schema executed successfully.")

        # Verify tables
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        tables = [t for t in tables if t != 'sqlite_sequence']

        print(f"\nTables created ({len(tables)}):")
        for table in tables:
            # Get row count
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  - {table} ({count} rows)")

        print("\nDatabase initialization complete.")

    except sqlite3.Error as e:
        print(f"ERROR: Database error: {e}")
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    reset_flag = "--reset" in sys.argv

    if reset_flag:
        confirm = input("WARNING: This will delete all existing data. Continue? (yes/no): ")
        if confirm.lower() != "yes":
            print("Aborted.")
            sys.exit(0)

    init_database(reset=reset_flag)
