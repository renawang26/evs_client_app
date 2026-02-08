"""
Display Configuration Module

Controls display-related settings for the EVS application.
"""

# File Aliasing Configuration
# Set to True to show files as "file_1", "file_2", etc. (privacy protection)
# Set to False to show original file names
ENABLE_FILE_ALIASING = True

# Display settings
DISPLAY_CONFIG = {
    # File name display
    'file_aliasing': ENABLE_FILE_ALIASING,

    # Default language pair for EVS annotation
    'default_source_lang': 'en',
    'default_target_lang': 'zh',
}


def get_file_aliasing_enabled() -> bool:
    """
    Check if file aliasing is enabled.

    Returns:
        bool: True if file aliasing is enabled, False otherwise
    """
    return DISPLAY_CONFIG.get('file_aliasing', False)


def set_file_aliasing_enabled(enabled: bool) -> None:
    """
    Set file aliasing enabled/disabled.

    Args:
        enabled: True to enable file aliasing, False to disable
    """
    global ENABLE_FILE_ALIASING
    ENABLE_FILE_ALIASING = enabled
    DISPLAY_CONFIG['file_aliasing'] = enabled


__all__ = [
    'ENABLE_FILE_ALIASING',
    'DISPLAY_CONFIG',
    'get_file_aliasing_enabled',
    'set_file_aliasing_enabled'
]
