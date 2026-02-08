#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Privacy Settings Component
隐私设置组件

This module provides UI components for managing file privacy settings
including alias display preferences and file management.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Tuple
from file_alias_manager import FileAliasManager

def render_privacy_settings_sidebar():
    """Render privacy settings in the sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.subheader("🔒 Privacy Settings")

        # File alias display toggle
        show_original = st.checkbox(
            "显示原始文件名 (Show Original Filenames)",
            value=st.session_state.get('show_original_filenames', False),
            help="Toggle between anonymized file names (file_1, file_2...) and original file names"
        )

        if show_original != st.session_state.get('show_original_filenames', False):
            st.session_state.show_original_filenames = show_original
            st.rerun()

        # File alias management
        if st.expander("📂 File Alias Management", expanded=False):
            render_alias_management()

def render_alias_management():
    """Render file alias management interface"""
    alias_manager = st.session_state.file_alias_manager

    # Display current aliases
    st.markdown("**Current File Aliases:**")
    aliases = alias_manager.get_all_aliases()

    if aliases:
        # Create DataFrame for display
        alias_df = pd.DataFrame(aliases, columns=['Alias', 'Original Filename', 'Created'])
        alias_df['Created'] = pd.to_datetime(alias_df['Created']).dt.strftime('%Y-%m-%d %H:%M')

        # Display as table
        st.dataframe(
            alias_df,
            width='stretch',
            hide_index=True,
            column_config={
                "Alias": st.column_config.TextColumn("Alias", width="small"),
                "Original Filename": st.column_config.TextColumn("Original", width="large"),
                "Created": st.column_config.TextColumn("Created", width="small")
            }
        )

        # Statistics
        stats = alias_manager.get_file_statistics()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Active Files", stats['total_active'])
        with col2:
            st.metric("Today's Access", stats['accessed_today'])
    else:
        st.info("No file aliases created yet")

def apply_privacy_filter_to_dataframe(df: pd.DataFrame, filename_column: str = 'file_name') -> pd.DataFrame:
    """
    Apply privacy filtering to DataFrame based on user settings

    Args:
        df: DataFrame containing file names
        filename_column: Name of the column containing file names

    Returns:
        DataFrame with appropriate file name display
    """
    if df.empty or filename_column not in df.columns:
        return df

    show_original = st.session_state.get('show_original_filenames', False)

    if show_original:
        # Show original filenames
        return df
    else:
        # Apply alias transformation
        alias_manager = st.session_state.file_alias_manager
        df_copy = df.copy()

        # Create mapping for unique filenames
        unique_files = df_copy[filename_column].dropna().unique()
        filename_mapping = {}

        for filename in unique_files:
            if filename and str(filename).strip():
                alias = alias_manager.get_or_create_alias(str(filename))
                filename_mapping[filename] = alias

        # Apply mapping
        df_copy[filename_column] = df_copy[filename_column].map(
            lambda x: filename_mapping.get(x, x) if x else x
        )

        return df_copy

def get_display_filename(original_filename: str) -> str:
    """
    Get the display filename based on privacy settings

    Args:
        original_filename: The original file name

    Returns:
        Display filename (alias or original based on settings)
    """
    if not original_filename:
        return original_filename

    show_original = st.session_state.get('show_original_filenames', False)

    if show_original:
        return original_filename
    else:
        alias_manager = st.session_state.file_alias_manager
        return alias_manager.get_or_create_alias(original_filename)

def get_original_filename(display_filename: str) -> str:
    """
    Get the original filename from display filename

    Args:
        display_filename: The display filename (might be alias)

    Returns:
        Original filename
    """
    if not display_filename:
        return display_filename

    show_original = st.session_state.get('show_original_filenames', False)

    if show_original:
        return display_filename
    else:
        alias_manager = st.session_state.file_alias_manager
        original = alias_manager.get_original_filename(display_filename)
        return original if original else display_filename

def create_privacy_aware_file_selector(files_df: pd.DataFrame, key: str, label: str = "File") -> str:
    """
    Create a file selector that respects privacy settings

    Args:
        files_df: DataFrame containing file information
        key: Unique key for the selectbox
        label: Label for the selectbox

    Returns:
        Original filename (regardless of display)
    """
    if files_df.empty:
        st.warning("No files available")
        return ""

    # Apply privacy filter to display
    display_df = apply_privacy_filter_to_dataframe(files_df.copy(), 'file_name')

    # Create selectbox with display names
    display_filename = st.selectbox(
        label,
        options=[''] + display_df['file_name'].tolist(),
        key=key
    )

    if not display_filename:
        return ""

    # Return original filename
    return get_original_filename(display_filename)

def render_privacy_info_panel():
    """Render privacy information panel"""
    with st.expander("🔒 Privacy Protection Information", expanded=False):
        st.markdown("""
        ### File Privacy Protection

        This system provides comprehensive privacy protection for your interpretation files:

        #### 🎭 Anonymized Display
        - File names are displayed as `file_1`, `file_2`, etc. by default
        - Original names containing personal information are hidden from view
        - Only you can see the mapping between aliases and original names

        #### 🗄️ Secure Storage
        - Alias mappings are stored securely in the database
        - Original filenames are preserved but not displayed publicly
        - Each file gets a permanent, consistent alias

        #### ⚙️ Flexible Settings
        - Toggle between anonymous and original file name display
        - Manage file aliases in the privacy settings panel
        - Statistics show your file usage patterns

        #### 🔐 Data Protection
        - File hashes ensure data integrity
        - Soft deletion preserves historical data
        - Automatic cleanup of old, inactive aliases

        **Note:** Administrators may have access to original filenames for system management purposes.
        """)

def render_file_alias_statistics():
    """Render file alias statistics dashboard"""
    alias_manager = st.session_state.file_alias_manager
    stats = alias_manager.get_file_statistics()

    st.subheader("📊 File Privacy Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Files",
            stats['total_files'],
            help="Total number of files with aliases"
        )

    with col2:
        st.metric(
            "Active Files",
            stats['total_active'],
            help="Currently active file aliases"
        )

    with col3:
        st.metric(
            "Accessed Today",
            stats['accessed_today'],
            help="Files accessed today"
        )

    with col4:
        st.metric(
            "Inactive Files",
            stats['total_inactive'],
            help="Deactivated file aliases"
        )

    # Usage chart
    if stats['total_active'] > 0:
        import plotly.graph_objects as go

        fig = go.Figure(data=[
            go.Bar(
                x=['Active', 'Inactive', 'Today\'s Access'],
                y=[stats['total_active'], stats['total_inactive'], stats['accessed_today']],
                marker_color=['green', 'orange', 'blue']
            )
        ])

        fig.update_layout(
            title="File Alias Usage",
            xaxis_title="Category",
            yaxis_title="Count",
            height=300
        )

        st.plotly_chart(fig, width='stretch')

def cleanup_old_aliases():
    """Cleanup old, inactive aliases"""
    alias_manager = st.session_state.file_alias_manager

    st.subheader("🧹 Cleanup Old Aliases")

    days_old = st.slider(
        "Days to consider as old",
        min_value=7,
        max_value=365,
        value=30,
        help="Files inactive for this many days will be cleaned up"
    )

    if st.button("🗑️ Cleanup Old Aliases", type="secondary"):
        with st.spinner("Cleaning up old aliases..."):
            deleted_count = alias_manager.cleanup_old_aliases(days_old)

        if deleted_count > 0:
            st.success(f"✅ Cleaned up {deleted_count} old aliases")
        else:
            st.info("No old aliases found to clean up")

def export_alias_mapping():
    """Export alias mapping for backup"""
    alias_manager = st.session_state.file_alias_manager

    st.subheader("📥 Export Alias Mapping")

    if st.button("📋 Export Mapping", type="secondary"):
        aliases = alias_manager.get_all_aliases()

        if aliases:
            import json
            from datetime import datetime

            mapping_data = {
                'export_date': datetime.now().isoformat(),
                'total_files': len(aliases),
                'mappings': [
                    {
                        'alias': alias[0],
                        'original': alias[1],
                        'created': alias[2]
                    }
                    for alias in aliases
                ]
            }

            json_str = json.dumps(mapping_data, indent=2, ensure_ascii=False)

            st.download_button(
                label="💾 Download Mapping",
                data=json_str,
                file_name=f"file_alias_mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        else:
            st.warning("No aliases to export")

def render_privacy_settings_admin_panel():
    """Render privacy settings for admin panel (not sidebar)"""
    st.subheader("🔒 User Privacy Settings Management")

    # Global privacy toggle for current session (for testing/admin purposes)
    st.write("**Current Session Settings**")

    col1, col2 = st.columns(2)
    with col1:
        # File alias display toggle for admin
        admin_show_original = st.checkbox(
            "Admin Show Original Filenames",
            value=st.session_state.get('admin_show_original_filenames', False),
            help="Administrators can choose to view original filenames or aliases"
        )

        if admin_show_original != st.session_state.get('admin_show_original_filenames', False):
            st.session_state.admin_show_original_filenames = admin_show_original

    with col2:
        # User default setting
        user_default_original = st.checkbox(
            "User Default Show Original",
            value=st.session_state.get('show_original_filenames', False),
            help="Set whether regular users default to showing original filenames"
        )

        if user_default_original != st.session_state.get('show_original_filenames', False):
            st.session_state.show_original_filenames = user_default_original

    # File alias management for admin
    st.markdown("---")
    st.write("**File Alias Management**")

    alias_manager = st.session_state.file_alias_manager
    aliases = alias_manager.get_all_aliases()

    if aliases:
        # Create DataFrame for display
        alias_df = pd.DataFrame(aliases, columns=['Alias', 'Original', 'Created'])
        alias_df['Created'] = pd.to_datetime(alias_df['Created']).dt.strftime('%Y-%m-%d %H:%M')

        # Quick stats
        stats = alias_manager.get_file_statistics()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Active Aliases", stats['total_active'])
        with col2:
            st.metric("Today's Access", stats['accessed_today'])
        with col3:
            st.metric("Total Aliases", stats['total_files'])

        # Compact view of aliases
        with st.expander("View All File Aliases", expanded=False):
            st.dataframe(
                alias_df,
                width='stretch',
                hide_index=True,
                column_config={
                    "Alias": st.column_config.TextColumn("Alias", width="small"),
                    "Original": st.column_config.TextColumn("Original Filename", width="large"),
                    "Created": st.column_config.TextColumn("Created", width="small")
                }
            )
    else:
        st.info("No file aliases created yet")

    # Quick actions
    st.markdown("---")
    st.write("**Quick Actions**")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Refresh Alias Data", type="secondary"):
            st.rerun()

    with col2:
        if st.button("📊 View Detailed Stats", type="secondary"):
            st.info("Detailed statistics available in 'Privacy Statistics' tab")

    with col3:
        if st.button("📥 Export Mapping", type="secondary"):
            st.info("Export functionality available in 'Data Export' tab")

def get_user_privacy_preference(user_email: str = None) -> bool:
    """
    Get user's privacy preference for file name display

    Args:
        user_email: User email (optional, uses session if not provided)

    Returns:
        True if user prefers original filenames, False for aliases
    """
    if not user_email:
        user_email = st.session_state.get('user_email', '')

    # For now, use session state. In future, could store in database per user
    return st.session_state.get('show_original_filenames', False)

def set_user_privacy_preference(show_original: bool, user_email: str = None):
    """
    Set user's privacy preference for file name display

    Args:
        show_original: Whether to show original filenames
        user_email: User email (optional, uses session if not provided)
    """
    if not user_email:
        user_email = st.session_state.get('user_email', '')

    # For now, use session state. In future, could store in database per user
    st.session_state.show_original_filenames = show_original

def render_user_privacy_controls():
    """
    Render privacy controls for regular users (non-admin)
    This function can be called in the main interface if needed
    """
    if not st.session_state.get('is_admin', False):
        with st.expander("🔒 Privacy Settings", expanded=False):
            show_original = st.checkbox(
                "Show Original Filenames",
                value=get_user_privacy_preference(),
                help="Toggle between displaying original filenames or anonymous aliases"
            )

            if show_original != get_user_privacy_preference():
                set_user_privacy_preference(show_original)
                st.success("Privacy settings updated")
                st.rerun()

            # Show current file count
            if st.session_state.get('file_alias_manager'):
                stats = st.session_state.file_alias_manager.get_file_statistics()
                if stats['total_active'] > 0:
                    st.info(f"🔒 Currently protecting {stats['total_active']} files")