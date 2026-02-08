#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Display Utilities
文件显示工具函数

Simple utilities for displaying anonymized file names in the interface.
"""

import pandas as pd
import streamlit as st
from file_alias_manager import FileAliasManager
from config.display_config import get_file_aliasing_enabled
from typing import List, Optional

def get_file_display_name(original_filename: str) -> str:
    """
    Get display name for a file (alias if enabled, original if not)

    Args:
        original_filename: The original file name

    Returns:
        Display name (alias or original based on config)
    """
    if not original_filename:
        return original_filename

    # Check if file aliasing is enabled
    if not get_file_aliasing_enabled():
        return original_filename

    try:
        alias_manager = st.session_state.file_alias_manager

        # Get or create alias for the file
        alias = alias_manager.get_or_create_alias(str(original_filename))
        return alias

    except Exception:
        # Fallback to original filename if any error
        return original_filename

def get_original_filename_from_display(display_name: str) -> str:
    """
    Get original filename from display name

    Args:
        display_name: The display name (might be alias)

    Returns:
        Original filename
    """
    if not display_name:
        return display_name

    # If aliasing is disabled, display_name is the original
    if not get_file_aliasing_enabled():
        return display_name

    try:
        alias_manager = st.session_state.file_alias_manager
        original = alias_manager.get_original_filename(display_name)
        return original if original else display_name
    except Exception:
        return display_name

def prepare_files_for_display(files_df: pd.DataFrame, filename_column: str = 'file_name') -> pd.DataFrame:
    """
    Prepare files DataFrame for display with aliases

    Args:
        files_df: DataFrame containing file information
        filename_column: Column name containing file names

    Returns:
        DataFrame with display names (aliases if enabled)
    """
    if files_df.empty or filename_column not in files_df.columns:
        return files_df

    # If aliasing is disabled, return original DataFrame
    if not get_file_aliasing_enabled():
        return files_df

    display_df = files_df.copy()

    # Apply display names (aliases)
    display_df[filename_column] = display_df[filename_column].apply(get_file_display_name)

    return display_df


def get_file_options_for_selectbox(files_df: pd.DataFrame, filename_column: str = 'file_name', include_empty: bool = True) -> tuple:
    """
    Get file options for selectbox with proper aliasing.
    Returns both display options and a mapping to original names.

    Args:
        files_df: DataFrame containing file information
        filename_column: Column name containing file names
        include_empty: Whether to include empty option

    Returns:
        Tuple of (display_options, display_to_original_mapping)
    """
    if files_df.empty or filename_column not in files_df.columns:
        return ([""] if include_empty else [], {})

    original_names = files_df[filename_column].unique().tolist()
    display_to_original = {}

    if get_file_aliasing_enabled():
        for original in original_names:
            display_name = get_file_display_name(original)
            display_to_original[display_name] = original
        display_options = list(display_to_original.keys())
    else:
        display_options = original_names
        display_to_original = {name: name for name in original_names}

    if include_empty:
        display_options = [""] + display_options

    return display_options, display_to_original

def create_file_selectbox(files_df: pd.DataFrame, label: str = "File", key: str = None, filename_column: str = 'file_name') -> str:
    """
    Create a file selectbox with anonymized display names

    Args:
        files_df: DataFrame containing file information
        label: Label for the selectbox
        key: Unique key for the selectbox
        filename_column: Column name containing file names

    Returns:
        Original filename (not the display name)
    """
    if files_df.empty:
        st.warning("No files available")
        return ""

    # Prepare display DataFrame
    display_df = prepare_files_for_display(files_df, filename_column)

    # Create selectbox with display names
    display_name = st.selectbox(
        label,
        options=[''] + display_df[filename_column].tolist(),
        key=key
    )

    if not display_name:
        return ""

    # Return original filename
    return get_original_filename_from_display(display_name)

def create_file_selectbox_with_all(files_df: pd.DataFrame, label: str = "File", key: str = None, filename_column: str = 'file_name') -> str:
    """
    Create a file selectbox with anonymized display names and "All" option

    Args:
        files_df: DataFrame containing file information
        label: Label for the selectbox
        key: Unique key for the selectbox
        filename_column: Column name containing file names

    Returns:
        Original filename or "All" (not the display name)
    """
    if files_df.empty:
        st.warning("No files available")
        return ""

    # Prepare display DataFrame
    display_df = prepare_files_for_display(files_df, filename_column)

    # Create selectbox with display names, including "All" option
    display_name = st.selectbox(
        label,
        options=['All'] + display_df[filename_column].tolist(),
        key=key
    )

    if not display_name or display_name == "All":
        return display_name

    # Return original filename
    return get_original_filename_from_display(display_name)

def show_file_alias_info():
    """Show file alias information in sidebar"""
    if st.session_state.get('file_alias_manager'):
        alias_manager = st.session_state.file_alias_manager
        stats = alias_manager.get_file_statistics()

        if stats['total_active'] > 0:
            with st.sidebar:
                st.markdown("---")
                st.markdown("**🔒 Privacy Protection Active**")
                st.caption(f"已保护 {stats['total_active']} 个文件")

                if st.button("Show File Mapping", type="secondary", width='stretch'):
                    show_alias_mapping_modal()

def show_alias_mapping_modal():
    """Show alias mapping in a modal dialog"""
    try:
        alias_manager = st.session_state.file_alias_manager
        aliases = alias_manager.get_all_aliases()

        if aliases:
            with st.expander("📂 File Alias Mapping", expanded=True):
                alias_df = pd.DataFrame(aliases, columns=['别名', '原始文件名', '创建时间'])
                alias_df['创建时间'] = pd.to_datetime(alias_df['创建时间']).dt.strftime('%Y-%m-%d %H:%M')

                st.dataframe(
                    alias_df,
                    width='stretch',
                    hide_index=True,
                    column_config={
                        "别名": st.column_config.TextColumn("别名", width="small"),
                        "原始文件名": st.column_config.TextColumn("原始文件名", width="large"),
                        "创建时间": st.column_config.TextColumn("创建时间", width="small")
                    }
                )
        else:
            st.info("暂无文件别名")
    except Exception as e:
        st.error(f"显示别名映射时出错: {str(e)}")

def display_privacy_status():
    """Display privacy protection status"""
    try:
        alias_manager = st.session_state.file_alias_manager
        stats = alias_manager.get_file_statistics()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔒 Protected Files", stats['total_active'])
        with col2:
            st.metric("📅 Today's Access", stats['accessed_today'])
        with col3:
            st.metric("📊 Total Files", stats['total_files'])

    except Exception as e:
        st.warning(f"无法显示隐私状态: {str(e)}")

def test_file_display_utils():
    """Test function for file display utilities"""
    print("🧪 Testing File Display Utils")
    print("=" * 40)

    # Test files
    test_files = [
        "BOOTH 2_1 Chen Huiwen.mp3",
        "BOOTH 2_2 Ni Chengjun.mp3",
        "Recording_Meeting_20231201.mp3"
    ]

    # Initialize manager
    from file_alias_manager import FileAliasManager
    manager = FileAliasManager()

    # Create test DataFrame
    files_df = pd.DataFrame({'file_name': test_files})

    print("📝 Original files:")
    for file in test_files:
        print(f"  {file}")

    print("\n🎭 Display names:")
    for file in test_files:
        display_name = get_file_display_name(file)
        print(f"  {file} → {display_name}")

    print("\n🔄 Reverse mapping:")
    for file in test_files:
        display_name = get_file_display_name(file)
        original = get_original_filename_from_display(display_name)
        print(f"  {display_name} → {original} ({'✅' if original == file else '❌'})")

    return manager

if __name__ == "__main__":
    test_file_display_utils()