"""
Pages package for EVS application.

This package contains Streamlit page modules.
"""

from .admin_panel import render_admin_panel
from .download_audio import render_download_audio_tab

__all__ = ['render_admin_panel', 'render_download_audio_tab']
