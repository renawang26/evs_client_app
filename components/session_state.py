"""
Session State Management Module

Handles Streamlit session state initialization and cleanup for the EVS application.
"""

import streamlit as st
from file_alias_manager import FileAliasManager


class SessionState:
    """Class to hold session state data for EVS annotation"""

    def __init__(self):
        self.mark_evs_start = {"data": [], "remove": False}
        self.mark_evs_end = {"data": [], "remove": False}
        self.mark_annotate = {"data": [], "remove": False}
        self.wordlist_data = None
        self.df = None


def initialize_session_state():
    """Initialize session state variables"""
    if 'session_state' not in st.session_state:
        st.session_state.session_state = SessionState()

    if "editing_enabled" not in st.session_state:
        st.session_state["editing_enabled"] = False

    if "edited_cells" not in st.session_state:
        st.session_state["edited_cells"] = []

    if "original_table_data" not in st.session_state:
        st.session_state["original_table_data"] = {}

    # Initialize file alias manager for privacy protection
    if 'file_alias_manager' not in st.session_state:
        st.session_state.file_alias_manager = FileAliasManager()

    if 'show_original_filenames' not in st.session_state:
        st.session_state.show_original_filenames = False


def clear_session_data():
    """Clear all session data related to table editing"""
    keys_to_remove = []

    for key in st.session_state:
        # Clear original data
        if key.startswith('original_'):
            keys_to_remove.append(key)
        # Clear key mappings
        elif key.startswith('key_map_'):
            keys_to_remove.append(key)
        # Clear table data
        elif key.startswith('loaded_data_'):
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del st.session_state[key]

    # Clear changes
    if 'changes' in st.session_state:
        del st.session_state['changes']

    if 'table_data' in st.session_state:
        del st.session_state['table_data']

    if 'session_state' in st.session_state and hasattr(st.session_state.session_state, 'wordlist_data'):
        st.session_state.session_state.wordlist_data = None

    if 'wordlist_file' in st.session_state:
        del st.session_state['wordlist_file']

    if 'wordlist_lang' in st.session_state:
        del st.session_state['wordlist_lang']
