"""
Configuration package for EVS application.

This package contains configuration-related modules and utilities.
"""

from .config import AUTH_CONFIG, GOOGLE_CLOUD_CONFIG, BASE_URL
from .database_config import (
    PROJECT_ROOT,
    DATA_DIR,
    DB_PATH,
    SCHEMA_FILE,
    REPOSITORY_ROOT,
    get_db_path,
    get_db_path_str,
    ensure_data_dir,
    get_schema_file
)
from .display_config import (
    ENABLE_FILE_ALIASING,
    DISPLAY_CONFIG,
    get_file_aliasing_enabled,
    set_file_aliasing_enabled
)
from .asr_language_config import (
    ASRModelInfo,
    FUNASR_MODELS,
    CRISPERWHISPER_MODELS,
    LANGUAGE_ASR_RECOMMENDATIONS,
    ENABLE_LANGUAGE_SPECIFIC_ASR,
    DEFAULT_LANGUAGE_ASR,
    get_available_models_for_language,
    get_recommended_model_for_language,
    get_model_options_for_ui,
    get_language_asr_config,
    set_language_asr_config,
    get_chinese_asr_recommendations,
    check_funasr_available,
    check_crisperwhisper_available,
    get_available_providers
)

__all__ = [
    # Application config
    'AUTH_CONFIG',
    'GOOGLE_CLOUD_CONFIG',
    'BASE_URL',
    # Database config
    'PROJECT_ROOT',
    'DATA_DIR',
    'DB_PATH',
    'SCHEMA_FILE',
    'REPOSITORY_ROOT',
    'get_db_path',
    'get_db_path_str',
    'ensure_data_dir',
    'get_schema_file',
    # Display config
    'ENABLE_FILE_ALIASING',
    'DISPLAY_CONFIG',
    'get_file_aliasing_enabled',
    'set_file_aliasing_enabled',
    # ASR Language config
    'ASRModelInfo',
    'FUNASR_MODELS',
    'CRISPERWHISPER_MODELS',
    'LANGUAGE_ASR_RECOMMENDATIONS',
    'ENABLE_LANGUAGE_SPECIFIC_ASR',
    'DEFAULT_LANGUAGE_ASR',
    'get_available_models_for_language',
    'get_recommended_model_for_language',
    'get_model_options_for_ui',
    'get_language_asr_config',
    'set_language_asr_config',
    'get_chinese_asr_recommendations',
    'check_funasr_available',
    'check_crisperwhisper_available',
    'get_available_providers'
]
