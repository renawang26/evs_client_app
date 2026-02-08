import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logger(name: str):
    """Set up logger with file and console handlers.

    Safe to call multiple times (e.g. on Streamlit reruns) — handlers
    are only added once per logger name.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on Streamlit reruns
    if logger.handlers:
        return logger

    # Create logs directory if it doesn't exist
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)

    logger.setLevel(logging.DEBUG)

    # Create formatters and add it to the handlers
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')

    # Create file handler
    today = datetime.now().strftime('%Y-%m-%d')
    file_handler = logging.FileHandler(f'logs/evs_app_{today}.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger