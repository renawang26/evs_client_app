import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from db_utils import EVSDataUtils
from user_utils import UserUtils
import base64
from typing import List, Dict, Any, Optional, Tuple
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')
import torch
import warnings
import ffmpeg
from pydub import AudioSegment
import io
import tempfile
import subprocess
import os
from datetime import datetime
import time
import streamlit.components.v1 as components
from evs_annotator import evs_annotator
import json
import sqlite3
import plotly.graph_objects as go
from st_aggrid import AgGrid
from concordance_utils import ConcordanceUtils
from utils.asr_utils import ASRUtils  # import ASRUtils class
from results_view import render_whisper_results_tab  # Import results view functions
from pages.download_audio import render_download_audio_tab
from save_asr_results import save_asr_result_to_database
import shutil
import re
import uuid
# try to import librosa, if not exists, ignore
try:
    import librosa
except ImportError:
    pass
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass
from dataclasses import dataclass, field
from copy import deepcopy
import socket

# Import email queue
from email_queue import get_email_queue, initialize_email_queue

# Import file alias manager for privacy protection
from file_alias_manager import FileAliasManager, apply_alias_to_dataframe
from file_display_utils import (
    create_file_selectbox,
    create_file_selectbox_with_all,
    show_file_alias_info,
    get_file_display_name,
    get_original_filename_from_display,
    prepare_files_for_display,
    get_file_options_for_selectbox
)
from config.display_config import get_file_aliasing_enabled

# Import Chinese NLP processor
from chinese_nlp_unified import ChineseNLPUnified, NLPEngine, create_nlp_processor


from logger_config import setup_logger

# Set up logger
logger = setup_logger('evs_app')

# Configure page
st.set_page_config(page_title="EVS Navigation", layout="wide", initial_sidebar_state="collapsed")

# Hide sidebar completely
st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
</style>
""", unsafe_allow_html=True)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/myap/PhD_Project/key/google_credentials.json"

EVS_RESOURCES_PATH = r"./evs_resources"
SLICE_AUDIO_PATH = os.path.join(EVS_RESOURCES_PATH, "slice_audio_files")

# Import CSS from styles.py
from styles import COLOR_LEGEND, EVS_LEGEND, GREEN, YELLOW, PINK

 # language selection
LANGUAGES = {
    'English': 'en',
    'Chinese': 'zh',
}

ASR_PROVIDERS = {
    "CrisperWhisper": "crisperwhisper",
    "FunASR": "funasr",
    "Google": "google",
    "Tencent": "tencent",
    "IBM": "ibm",
}

# 添加配置文件导入
from config import AUTH_CONFIG, GOOGLE_CLOUD_CONFIG
from config.asr_language_config import (
    check_funasr_available,
    check_crisperwhisper_available,
    get_available_providers,
    get_model_options_for_ui,
    get_chinese_asr_recommendations,
    CRISPERWHISPER_MODELS,
    FUNASR_MODELS
)

# Email queue initialization moved to main() with session_state guard

# 修改身份验证检查函数
def check_authentication():
    """Check if user is already logged in or if login is bypassed"""
    if AUTH_CONFIG['BYPASS_LOGIN']:
        # Set default user session data
        st.session_state.user_authenticated = True
        st.session_state.user_email = AUTH_CONFIG['DEFAULT_USER']['email']
        st.session_state.is_admin = AUTH_CONFIG['DEFAULT_USER'].get('is_admin', False)  # 使用get方法安全获取is_admin值
        return True

    if 'user_authenticated' not in st.session_state:
        st.session_state.user_authenticated = False

    return st.session_state.user_authenticated

def render_login_page():
    """Render login page or bypass if configured"""
    if AUTH_CONFIG['BYPASS_LOGIN']:
        st.session_state.user_authenticated = True
        st.session_state.user_email = AUTH_CONFIG['DEFAULT_USER']['email']
        st.session_state.is_admin = AUTH_CONFIG['DEFAULT_USER'].get('is_admin', False)  # 使用get方法安全获取is_admin值
        st.rerun()
        return

    st.title("Welcome to EVS Navigation")

    # Check for verification parameters in URL
    if 'verify' in st.query_params and 'email' in st.query_params:
        verify_token = st.query_params['verify']
        email = st.query_params['email']

        if UserUtils.verify_email(email, verify_token):
            st.success(f"Email {email} has been verified successfully! You can now log in.")
            # Clear query parameters
            st.query_params.clear()
        else:
            st.error("Email verification failed. The link may be invalid or expired.")
            # Offer resend verification option
            if st.button("Resend Verification Email"):
                if UserUtils.resend_verification_email(email):
                    st.success(f"Verification email has been resent to {email}.")
                else:
                    st.error("Failed to resend verification email. Please try again later.")

    # Create tabs to switch between login and registration
    login_tab, register_tab = st.tabs(["Login", "Register"])

    # Login tab
    with login_tab:
        st.header("User Login")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")

        # Get client IP and User-Agent
        ip_address = None
        user_agent = None

        try:
            # Try to get client IP - in production this may need to be adjusted based on deployment
            ip_address = socket.gethostbyname(socket.gethostname())
        except socket.error as e:
            logger.debug(f"Unable to get client IP: {e}")
            ip_address = "127.0.0.1"  # Default if unable to determine

        # Get User-Agent from Streamlit if available
        if hasattr(st, 'request_headers') and callable(st.request_headers):
            headers = st.request_headers()
            if headers and 'User-Agent' in headers:
                user_agent = headers['User-Agent']

        login_col1, login_col2 = st.columns([1, 1])
        with login_col1:
            if st.button("Login", key="login_button"):
                if UserUtils.validate_user(email, password, ip_address, user_agent):
                    st.session_state.user_authenticated = True
                    st.session_state.user_email = email

                    # 检查是否是管理员用户
                    is_admin = UserUtils.is_admin_user(email)  # 假设UserUtils中有此方法
                    st.session_state.is_admin = is_admin

                    st.success("登录成功!")
                    time.sleep(1)
                    st.rerun()
                else:
                    if UserUtils.email_exists(email):
                        # Check if email exists but not verified
                        with UserUtils.get_db_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute("SELECT verified FROM users WHERE email = ?", (email,))
                            result = cursor.fetchone()
                            if result and result[0] == 0:
                                st.error("Email not verified. Please check your inbox for verification email.")
                                if st.button("Resend Verification Email", key="resend_login"):
                                    if UserUtils.resend_verification_email(email):
                                        st.success("Verification email has been resent to {email}.")
                                    else:
                                        st.error("Failed to resend verification email. Please try again later.")
                            else:
                                st.error("Invalid email or password. Please try again.")
                    else:
                        st.error("Invalid email or password. Please try again.")

        with login_col2:
            if st.button("Forgot Password?", key="forgot_password"):
                st.info("Please contact the administrator to reset your password.")

    # Registration tab
    with register_tab:
        st.header("User Registration")
        reg_email = st.text_input("Email", key="register_email")
        reg_password = st.text_input("Password", type="password", key="register_password")
        reg_confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm_password")

        if st.button("Register", key="register_button"):
            if not reg_email or not reg_password:
                st.error("Email and password are required.")
            elif reg_password != reg_confirm_password:
                st.error("Passwords do not match.")
            else:
                success, message = UserUtils.register_user(reg_email, reg_password)
                if success:
                    st.success(message)
                else:
                    st.error(message)

# 添加注销功能
def render_user_menu():
    """Render user menu"""
    if check_authentication():
        # Display user information
        st.sidebar.write(f"Current User: {st.session_state.user_email}")
        if st.session_state.get('is_admin', False):
            st.sidebar.write("Role: Administrator")
        else:
            st.sidebar.write("Role: Regular User")

        if st.sidebar.button("Logout"):
            # Clear all session states
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

def render_admin_panel():
    """Render admin panel"""
    admin_tabs = st.tabs(["User Management", "Email Queue", "Analysis History", "System Status", "ASR Configuration", "LLM & SI Analysis", "File Privacy Management", "Data Export"])

    # ASR Config tab
    with admin_tabs[4]:
        st.subheader("ASR Configuration")
        try:
            # Display current ASR configuration
            asr_config = EVSDataUtils.get_asr_config()

            if not asr_config:
                # Auto-initialize ASR configuration with defaults
                from asr_config import ASR_CONFIG
                for provider, config in ASR_CONFIG.items():
                    EVSDataUtils.save_asr_config(provider, config)
                asr_config = EVSDataUtils.get_asr_config()
                logger.info("ASR configuration auto-initialized with defaults")

            if asr_config:
                # Create selectbox for ASR providers
                provider_option = st.selectbox(
                    "Select ASR Provider",
                    options=list(ASR_PROVIDERS.keys()),
                    index=0
                )

                selected_provider = ASR_PROVIDERS[provider_option]

                if selected_provider in asr_config:
                    st.subheader(f"{provider_option} Configuration")

                    # Get configuration data
                    config_data = asr_config[selected_provider]

                    # Parse config data if it's a string
                    if isinstance(config_data, str):
                        try:
                            config_data = json.loads(config_data)
                        except json.JSONDecodeError:
                            st.error("Invalid configuration format")
                            return

                    # Configuration Form
                    with st.form(key=f"asr_config_form_{selected_provider}"):
                        st.markdown("### 📋 Configuration Settings")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("#### 🤖 Model Settings")

                            # Available models - editable multiselect
                            available_models = config_data.get("available_models", [])
                            # Normalize: if available_models is a dict, flatten to list
                            if isinstance(available_models, dict):
                                flat = []
                                for v in available_models.values():
                                    if isinstance(v, list):
                                        flat.extend(v)
                                    else:
                                        flat.append(str(v))
                                available_models = flat
                            # Normalize default_model to string
                            raw_default = config_data.get("default_model", "")
                            if isinstance(raw_default, dict):
                                raw_default = next(iter(raw_default.values()), "")
                            default_model_val = raw_default if raw_default in available_models else (available_models[0] if available_models else "")

                            if available_models:
                                # Model selection dropdown
                                default_model = st.selectbox(
                                    "Default Model",
                                    options=available_models,
                                    index=available_models.index(default_model_val) if default_model_val in available_models else 0,
                                    help="Select the default model for this ASR provider"
                                )

                                # Models management
                                st.markdown("**Available Models:**")

                                # Add new model
                                new_model = st.text_input("Add New Model", placeholder="Enter model name")
                                col_add, col_remove = st.columns(2)

                                with col_add:
                                    if st.form_submit_button("➕ Add Model", width='stretch'):
                                        if new_model and new_model not in available_models:
                                            available_models.append(new_model)
                                            st.success(f"Added model: {new_model}")

                                # Display current models with remove option
                                st.write("Current Models:")
                                for i, model in enumerate(available_models):
                                    st.write(f"• {model}")

                                # Remove model selection
                                if len(available_models) > 1:
                                    model_to_remove = st.selectbox(
                                        "Remove Model",
                                        options=[""] + available_models,
                                        help="Select a model to remove"
                                    )
                                    if model_to_remove and st.form_submit_button("🗑️ Remove Model", width='stretch'):
                                        available_models.remove(model_to_remove)
                                        st.success(f"Removed model: {model_to_remove}")
                            else:
                                st.info("No models configured for this provider")
                                new_model = st.text_input("Add First Model", placeholder="Enter model name")
                                if st.form_submit_button("Add Model") and new_model:
                                    available_models = [new_model]
                                    default_model = new_model

                        with col2:
                            st.markdown("#### 💬 Prompt Settings")

                            # Language prompts
                            prompts = config_data.get("prompts", {})

                            chinese_prompt = st.text_area(
                                "Chinese Prompt",
                                value=prompts.get("zh", "保留所有语气词如嗯、啊、呃等"),
                                height=100,
                                help="Instructions for Chinese transcription",
                                placeholder="Enter Chinese transcription instructions..."
                            )

                            english_prompt = st.text_area(
                                "English Prompt",
                                value=prompts.get("en", "Please preserve all filler words such as um, uh, er, hmm, like, you know, etc."),
                                height=100,
                                help="Instructions for English transcription",
                                placeholder="Enter English transcription instructions..."
                            )

                        # Advanced settings (provider-specific)
                        with st.expander("🔧 Advanced Settings", expanded=False):
                            st.markdown("#### Provider-Specific Configuration")

                            # Language codes
                            if "language_codes" in config_data:
                                st.markdown("**Language Codes:**")
                                lang_codes = config_data["language_codes"]
                                col_en, col_zh = st.columns(2)

                                with col_en:
                                    en_code = st.text_input("English Code", value=lang_codes.get("en", "en"))
                                with col_zh:
                                    zh_code = st.text_input("Chinese Code", value=lang_codes.get("zh", "zh"))

                            # Other provider-specific settings
                            if selected_provider == "google":
                                sample_rate = st.number_input("Sample Rate", value=config_data.get("sample_rate", 16000))
                                encoding = st.text_input("Encoding", value=config_data.get("encoding", "LINEAR16"))
                                credentials_file = st.text_input("Credentials File", value=config_data.get("credentials_file", ""))

                            elif selected_provider == "tencent":
                                engine_type_en = st.text_input("English Engine Type", value=config_data.get("engine_type", {}).get("en", "16k_en"))
                                engine_type_zh = st.text_input("Chinese Engine Type", value=config_data.get("engine_type", {}).get("zh", "16k_zh"))
                                api_config_file = st.text_input("API Config File", value=config_data.get("api_config_file", ""))

                        # Submit button
                        st.markdown("---")
                        submitted = st.form_submit_button("💾 Save Configuration", type="primary", width='stretch')

                        if submitted:
                            # Build new configuration
                            new_config = {
                                "available_models": available_models,
                                "default_model": default_model if 'default_model' in locals() else (available_models[0] if available_models else ""),
                                "prompts": {
                                    "zh": chinese_prompt,
                                    "en": english_prompt
                                }
                            }

                            # Add language codes if they exist
                            if "language_codes" in config_data:
                                new_config["language_codes"] = {
                                    "en": en_code if 'en_code' in locals() else config_data["language_codes"].get("en", "en"),
                                    "zh": zh_code if 'zh_code' in locals() else config_data["language_codes"].get("zh", "zh")
                                }

                            # Add provider-specific settings
                            if selected_provider == "google":
                                new_config.update({
                                    "sample_rate": sample_rate if 'sample_rate' in locals() else config_data.get("sample_rate", 16000),
                                    "encoding": encoding if 'encoding' in locals() else config_data.get("encoding", "LINEAR16"),
                                    "credentials_file": credentials_file if 'credentials_file' in locals() else config_data.get("credentials_file", "")
                                })
                            elif selected_provider == "tencent":
                                new_config.update({
                                    "engine_type": {
                                        "en": engine_type_en if 'engine_type_en' in locals() else config_data.get("engine_type", {}).get("en", "16k_en"),
                                        "zh": engine_type_zh if 'engine_type_zh' in locals() else config_data.get("engine_type", {}).get("zh", "16k_zh")
                                    },
                                    "api_config_file": api_config_file if 'api_config_file' in locals() else config_data.get("api_config_file", "")
                                })

                            # Save configuration
                            try:
                                if EVSDataUtils.save_asr_config(selected_provider, new_config):
                                    st.success("✅ Configuration saved successfully!")
                                    st.balloons()
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to save configuration")
                            except Exception as e:
                                st.error(f"❌ Error saving configuration: {str(e)}")

                    # Raw JSON view (for advanced users)
                    with st.expander("🔍 View Raw Configuration (Advanced)", expanded=False):
                        st.json(config_data)
                else:
                    st.info(f"No configuration found for {provider_option}")

                # Add button to create default configuration
                if st.button("Create Default Configuration"):
                    from asr_config import ASR_CONFIG
                    st.info("This will create default configurations for all ASR providers")

                    # Save each configuration to database
                    success = True
                    for provider, config in ASR_CONFIG.items():
                        if not EVSDataUtils.save_asr_config(provider, config):
                            success = False
                            st.error(f"Failed to save configuration for {provider}")

                    if success:
                        st.success("Default configurations created successfully")
                        st.rerun()
            else:
                st.error("ASR configuration could not be initialized. Check database connection.")

        except Exception as e:
            logger.error(f"Failed to load ASR configuration: {str(e)}")
            st.warning(f"ASR configuration error: {str(e)}. Re-initializing...")
            try:
                from asr_config import ASR_CONFIG
                for provider, config in ASR_CONFIG.items():
                    EVSDataUtils.save_asr_config(provider, config)
                st.success("ASR configuration re-initialized. Please reload.")
                st.rerun()
            except Exception as reinit_e:
                st.error(f"Failed to re-initialize ASR configuration: {str(reinit_e)}")

    # LLM & SI Analysis Configuration tab
    with admin_tabs[5]:
        st.subheader("LLM & Simultaneous Interpretation Analysis Configuration")

        # Create sub-tabs for different configuration areas
        llm_sub_tabs = st.tabs(["LLM Configuration", "Analysis Rules", "Error Patterns", "Cultural Rules", "Analysis History"])

        # LLM Configuration
        with llm_sub_tabs[0]:
            st.subheader("LLM Provider Configuration")

            try:
                from llm_config import load_llm_config, update_llm_config, get_active_llm_config

                # Load current LLM configuration
                llm_config = load_llm_config()
                active_provider = llm_config.get("active_llm_provider", "ollama")

                col1, col2 = st.columns([1, 3])

                with col1:
                    st.write("**Current Configuration:**")
                    st.write(f"Active Provider: {active_provider}")

                    # Provider selection
                    new_provider = st.selectbox(
                        "Select LLM Provider",
                        options=["ollama", "openai"],
                        index=0 if active_provider == "ollama" else 1,
                        key="llm_provider_select"
                    )

                with col2:
                    st.write("**Current Provider Configuration:**")
                    current_config = get_active_llm_config()

                    # Display current config
                    config_display = {
                        "Model": current_config.get("llm_model", "N/A"),
                        "Temperature": current_config.get("llm_temperature", "N/A"),
                        "Max Tokens": current_config.get("llm_max_tokens", "N/A"),
                        "Base URL": current_config.get("llm_base_url", "N/A"),
                        "Timeout": current_config.get("llm_request_timeout", "N/A")
                    }

                    for key, value in config_display.items():
                        st.write(f"- {key}: {value}")

                # Configuration editing
                st.write("---")
                st.subheader("Edit Configuration")

                config_to_edit = llm_config.get("llm_configs", {}).get(new_provider, {})

                col1, col2 = st.columns(2)

                with col1:
                    model = st.text_input(
                        "Model Name",
                        value=config_to_edit.get("llm_model"),
                        key="llm_model_input"
                    )

                    temperature = st.slider(
                        "Temperature",
                        min_value=0.0,
                        max_value=2.0,
                        value=float(config_to_edit.get("llm_temperature", 0.2)),
                        step=0.1,
                        key="llm_temperature_input"
                    )

                    max_tokens = st.number_input(
                        "Max Tokens",
                        min_value=1000,
                        max_value=100000,
                        value=int(config_to_edit.get("llm_max_tokens", 40960)),
                        step=1000,
                        key="llm_max_tokens_input"
                    )

                with col2:
                    base_url = st.text_input(
                        "Base URL",
                        value=config_to_edit.get("llm_base_url", "http://localhost:11434"),
                        key="llm_base_url_input"
                    )

                    timeout = st.number_input(
                        "Request Timeout (seconds)",
                        min_value=30,
                        max_value=600,
                        value=int(config_to_edit.get("llm_request_timeout", 300)),
                        step=30,
                        key="llm_timeout_input"
                    )

                    if new_provider == "openai":
                        api_key = st.text_input(
                            "API Key",
                            value=config_to_edit.get("llm_api_key", ""),
                            type="password",
                            key="llm_api_key_input"
                        )

                # Save configuration
                if st.button("Save LLM Configuration", type="primary"):
                    # Prepare new configuration
                    new_config = {
                        "llm_provider": new_provider,
                        "llm_model": model,
                        "llm_temperature": temperature,
                        "llm_max_tokens": max_tokens,
                        "llm_base_url": base_url,
                        "llm_request_timeout": timeout
                    }

                    if new_provider == "openai":
                        new_config["llm_api_key"] = api_key

                    # Update full configuration
                    updated_llm_config = llm_config.copy()
                    updated_llm_config["active_llm_provider"] = new_provider

                    if "llm_configs" not in updated_llm_config:
                        updated_llm_config["llm_configs"] = {}

                    updated_llm_config["llm_configs"][new_provider] = new_config

                    # Save to file
                    if update_llm_config(updated_llm_config):
                        st.success("LLM Configuration Saved Successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to save LLM Configuration")

            except Exception as e:
                st.error(f"Failed to load LLM Configuration: {str(e)}")

        # Analysis Rules Configuration
        with llm_sub_tabs[1]:
            st.subheader("Analysis Rules Configuration")

            try:
                from llm_config import load_analysis_rules, save_analysis_rules

                # Load current analysis rules
                analysis_rules = load_analysis_rules()

                # Quality metrics configuration
                st.write("**Quality Assessment Weights**")
                col1, col2, col3 = st.columns(3)

                quality_metrics = analysis_rules.get("quality_metrics", {})

                with col1:
                    accuracy_weight = st.slider(
                        "Accuracy Weight",
                        0.0, 1.0,
                        value=quality_metrics.get("accuracy_weight", 0.4),
                        step=0.1
                    )

                with col2:
                    fluency_weight = st.slider(
                        "Fluency Weight",
                        0.0, 1.0,
                        value=quality_metrics.get("fluency_weight", 0.3),
                        step=0.1
                    )

                with col3:
                    completeness_weight = st.slider(
                        "Completeness Weight",
                        0.0, 1.0,
                        value=quality_metrics.get("completeness_weight", 0.3),
                        step=0.1
                    )

                # Ensure weights sum to 1.0
                total_weight = accuracy_weight + fluency_weight + completeness_weight
                if abs(total_weight - 1.0) > 0.01:
                    st.warning(f"Weights should sum to 1.0, currently {total_weight:.2f}")

                # Timing analysis settings
                st.write("---")
                st.write("**Time Synchronization Analysis Settings**")

                timing_analysis = analysis_rules.get("timing_analysis", {})
                delay_thresholds = timing_analysis.get("delay_thresholds", {})

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    excellent_threshold = st.number_input(
                        "Excellent Threshold (seconds)",
                        min_value=0.1,
                        max_value=5.0,
                        value=delay_thresholds.get("excellent", 1.0),
                        step=0.1
                    )

                with col2:
                    good_threshold = st.number_input(
                        "Good Threshold (seconds)",
                        min_value=0.1,
                        max_value=10.0,
                        value=delay_thresholds.get("good", 2.0),
                        step=0.1
                    )

                with col3:
                    acceptable_threshold = st.number_input(
                        "Acceptable Threshold (seconds)",
                        min_value=0.1,
                        max_value=15.0,
                        value=delay_thresholds.get("acceptable", 3.0),
                        step=0.1
                    )

                with col4:
                    needs_improvement_threshold = st.number_input(
                        "Needs Improvement Threshold (seconds)",
                        min_value=0.1,
                        max_value=20.0,
                        value=delay_thresholds.get("needs_improvement", 5.0),
                        step=0.1
                    )

                # Analysis settings
                st.write("---")
                st.write("**Analysis Settings**")

                analysis_settings = analysis_rules.get("analysis_settings", {})

                col1, col2 = st.columns(2)

                with col1:
                    min_segment_length = st.number_input(
                        "Minimum Segment Length",
                        min_value=1,
                        max_value=10,
                        value=analysis_settings.get("min_segment_length", 3)
                    )

                    confidence_threshold = st.slider(
                        "Confidence Threshold",
                        0.0, 1.0,
                        value=analysis_settings.get("confidence_threshold", 0.7),
                        step=0.05
                    )

                with col2:
                    max_processing_time = st.number_input(
                        "Max Processing Time (seconds)",
                        min_value=60,
                        max_value=1800,
                        value=analysis_settings.get("max_processing_time", 300),
                        step=30
                    )

                    enable_llm_enhancement = st.checkbox(
                        "Enable LLM Enhancement Analysis",
                        value=analysis_settings.get("enable_llm_enhancement", True)
                    )

                # Save configuration
                if st.button("Save Analysis Rules Configuration", type="primary"):
                    # Update configuration
                    updated_rules = analysis_rules.copy()

                    updated_rules["quality_metrics"] = {
                        "accuracy_weight": accuracy_weight,
                        "fluency_weight": fluency_weight,
                        "completeness_weight": completeness_weight,
                        "scoring_thresholds": quality_metrics.get("scoring_thresholds", {})
                    }

                    updated_rules["timing_analysis"] = {
                        "delay_thresholds": {
                            "excellent": excellent_threshold,
                            "good": good_threshold,
                            "acceptable": acceptable_threshold,
                            "needs_improvement": needs_improvement_threshold
                        },
                        "speaking_rate": timing_analysis.get("speaking_rate", {}),
                        "processing_time_base": timing_analysis.get("processing_time_base", 0.5)
                    }

                    updated_rules["analysis_settings"] = {
                        "min_segment_length": min_segment_length,
                        "max_processing_time": max_processing_time,
                        "confidence_threshold": confidence_threshold,
                        "enable_llm_enhancement": enable_llm_enhancement,
                        "llm_models": analysis_settings.get("llm_models", {})
                    }

                    if save_analysis_rules(updated_rules):
                        st.success("Analysis Rules Configuration Saved Successfully!")

                        # Also save to database
                        try:
                            EVSDataUtils.save_si_analysis_config(
                                "default_analysis_rules",
                                "quality_metrics",
                                updated_rules,
                                st.session_state.get('user_email', 'admin')
                            )
                        except Exception as e:
                            st.warning(f"Error saving to database: {str(e)}")
                    else:
                        st.error("Failed to save Analysis Rules Configuration")

            except Exception as e:
                st.error(f"Failed to load Analysis Rules Configuration: {str(e)}")

        # Error Patterns Configuration
        with llm_sub_tabs[2]:
            st.subheader("Error Pattern Configuration")

            try:
                from llm_config import load_analysis_rules, save_analysis_rules

                analysis_rules = load_analysis_rules()
                error_patterns = analysis_rules.get("error_patterns", [])

                # Display existing error patterns
                st.write("**Current Error Patterns**")

                if error_patterns:
                    for i, pattern in enumerate(error_patterns):
                        with st.expander(f"Error Pattern {i+1}: {pattern.get('description', 'N/A')}"):
                            col1, col2 = st.columns(2)

                            with col1:
                                st.write(f"**ID:** {pattern.get('id', 'N/A')}")
                                st.write(f"**Type:** {pattern.get('type', 'N/A')}")
                                st.write(f"**Severity:** {pattern.get('severity', 'N/A')}")
                                st.write(f"**Enabled:** {'Yes' if pattern.get('enabled', False) else 'No'}")

                            with col2:
                                st.write(f"**Regular Expression:** `{pattern.get('pattern', 'N/A')}`")
                                st.write(f"**Description:** {pattern.get('description', 'N/A')}")
                                if pattern.get('suggestion'):
                                    st.write(f"**Suggestion:** {pattern.get('suggestion')}")
                else:
                    st.info("No error pattern configuration")

                # Add new error pattern
                st.write("---")
                st.write("**Add New Error Pattern**")

                with st.form("add_error_pattern"):
                    col1, col2 = st.columns(2)

                    with col1:
                        new_id = st.text_input("Error ID", placeholder="Example: tense_error_002")
                        new_type = st.selectbox(
                            "Error Type",
                            ["tense_error", "terminology_error", "accuracy_error", "omission", "mistranslation", "grammar_error", "cultural_error"]
                        )
                        new_severity = st.selectbox("Severity", ["low", "medium", "high"])

                    with col2:
                        new_pattern = st.text_input("Regular Expression", placeholder="Example: \\d+")
                        new_description = st.text_input("Description", placeholder="Error Description")
                        new_suggestion = st.text_input("Suggestion", placeholder="Optional correction suggestion")
                        new_enabled = st.checkbox("Enabled", value=True)

                    if st.form_submit_button("Add Error Pattern"):
                        if new_id and new_pattern and new_description:
                            new_error_pattern = {
                                "id": new_id,
                                "type": new_type,
                                "pattern": new_pattern,
                                "description": new_description,
                                "severity": new_severity,
                                "enabled": new_enabled
                            }

                            if new_suggestion:
                                new_error_pattern["suggestion"] = new_suggestion

                            # Add to existing patterns
                            updated_patterns = error_patterns + [new_error_pattern]
                            updated_rules = analysis_rules.copy()
                            updated_rules["error_patterns"] = updated_patterns

                            if save_analysis_rules(updated_rules):
                                st.success("New error pattern added successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to add error pattern")
                        else:
                            st.error("Please fill in all required fields")

            except Exception as e:
                st.error(f"Failed to load error pattern configuration: {str(e)}")

        # Cultural Rules Configuration
        with llm_sub_tabs[3]:
            st.subheader("Cultural Rules Configuration")

            try:
                from llm_config import load_analysis_rules, save_analysis_rules

                analysis_rules = load_analysis_rules()
                cultural_rules = analysis_rules.get("cultural_rules", {})

                # Idiom mappings
                st.write("**Idiom Mappings**")
                idiom_mappings = cultural_rules.get("idiom_mappings", {})

                if idiom_mappings:
                    for english, chinese in idiom_mappings.items():
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**English:** {english}")
                        with col2:
                            st.write(f"**Chinese:** {chinese}")
                else:
                    st.info("No idiom mapping configuration")

                # Add new idiom mapping
                st.write("---")
                st.write("**Add New Idiom Mapping**")

                with st.form("add_idiom_mapping"):
                    col1, col2 = st.columns(2)

                    with col1:
                        new_english_idiom = st.text_input("English Idiom", placeholder="例: break a leg")

                    with col2:
                        new_chinese_idiom = st.text_input("Chinese Translation", placeholder="例: 祝好运")

                    if st.form_submit_button("Add Idiom Mapping"):
                        if new_english_idiom and new_chinese_idiom:
                            updated_idioms = idiom_mappings.copy()
                            updated_idioms[new_english_idiom] = new_chinese_idiom

                            updated_rules = analysis_rules.copy()
                            if "cultural_rules" not in updated_rules:
                                updated_rules["cultural_rules"] = {}
                            updated_rules["cultural_rules"]["idiom_mappings"] = updated_idioms

                            if save_analysis_rules(updated_rules):
                                st.success("New idiom mapping added successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to add idiom mapping")
                        else:
                            st.error("Please fill in both English idiom and Chinese Translation")

                # Terminology dictionary
                st.write("---")
                st.write("**Terminology Dictionary**")

                terminology = cultural_rules.get("terminology_dictionary", {})

                for category, terms in terminology.items():
                    with st.expander(f"{category.replace('_', ' ').title()} Terminology"):
                        for en_term, zh_term in terms.items():
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**{en_term}**")
                            with col2:
                                st.write(f"{zh_term}")

            except Exception as e:
                st.error(f"Failed to load cultural rules configuration: {str(e)}")

        # Analysis History
        with llm_sub_tabs[4]:
            st.subheader("Analysis History")

            try:
                # Get analysis summary
                summary = EVSDataUtils.get_si_analysis_summary()

                if summary.get('total_statistics'):
                    stats = summary['total_statistics']

                    # Display overview metrics
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Total Analysis Files", stats.get('total_files', 0))

                    with col2:
                        st.metric("Total Analyses", stats.get('total_analyses', 0))

                    with col3:
                        avg_score = stats.get('avg_overall_score', 0)
                        st.metric("Average Quality Score", f"{avg_score:.2f}" if avg_score else "N/A")

                # Analysis breakdown by type
                if summary.get('analysis_breakdown'):
                    st.write("**Analysis Type Breakdown**")

                    breakdown_df = pd.DataFrame(summary['analysis_breakdown'])
                    if not breakdown_df.empty:
                        st.dataframe(
                            breakdown_df,
                            column_config={
                                'analysis_type': st.column_config.TextColumn('Analysis Type'),
                                'count': st.column_config.NumberColumn('Analysis Count'),
                                'avg_score': st.column_config.NumberColumn('Average Score', format="%.2f"),
                                'latest_analysis': st.column_config.DatetimeColumn('Latest Analysis Time')
                            },
                            hide_index=True
                        )

                # Recent analysis results
                st.write("---")
                st.write("**Recent Analysis Results**")

                recent_results = EVSDataUtils.get_si_analysis_results(limit=10)

                if not recent_results.empty:
                    # Select columns to display
                    display_columns = [
                        'file_name', 'asr_provider', 'analysis_type',
                        'overall_score', 'quality_level', 'analysis_timestamp'
                    ]

                    display_df = recent_results[display_columns].copy()

                    # Apply file name privacy protection
                    if not display_df.empty and 'file_name' in display_df.columns:
                        display_df['file_name'] = display_df['file_name'].apply(get_file_display_name)

                    st.dataframe(
                        display_df,
                        column_config={
                            'file_name': st.column_config.TextColumn('File Name'),
                            'asr_provider': st.column_config.TextColumn('ASR Provider'),
                            'analysis_type': st.column_config.TextColumn('Analysis Type'),
                            'overall_score': st.column_config.NumberColumn('Overall Score', format="%.2f"),
                            'quality_level': st.column_config.TextColumn('Quality Level'),
                            'analysis_timestamp': st.column_config.DatetimeColumn('Analysis Time')
                        },
                        hide_index=True
                    )
                else:
                    st.info("No analysis history")

            except Exception as e:
                st.error(f"Failed to load analysis history: {str(e)}")

    # File Privacy Management tab
    with admin_tabs[6]:  # File Privacy Management tab
        st.subheader("File Privacy Management")

        # Import required functions
        from privacy_settings import (
            render_file_alias_statistics,
            export_alias_mapping,
            cleanup_old_aliases,
            get_display_filename,
            get_original_filename,
            render_privacy_settings_admin_panel
        )
        from file_display_utils import get_file_display_name, display_privacy_status

        # Create sub-tabs for different privacy management areas
        privacy_sub_tabs = st.tabs([
            "File Mapping",
            "Privacy Statistics",
            "Admin Controls",
            "Settings",
            "Data Export"
        ])

        # File Mapping Management
        with privacy_sub_tabs[0]:
            st.subheader("📂 File Alias Mapping")

            try:
                alias_manager = st.session_state.file_alias_manager
                aliases = alias_manager.get_all_aliases()

                if aliases:
                    # Create DataFrame for display
                    alias_df = pd.DataFrame(aliases, columns=['Alias', 'Original', 'Created'])
                    alias_df['Created'] = pd.to_datetime(alias_df['Created']).dt.strftime('%Y-%m-%d %H:%M')

                    # Display statistics
                    stats = alias_manager.get_file_statistics()
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Files", stats['total_files'])
                    with col2:
                        st.metric("Active Files", stats['total_active'])
                    with col3:
                        st.metric("Today's Access", stats['accessed_today'])
                    with col4:
                        st.metric("Inactive Files", stats['total_inactive'])

                    st.markdown("---")

                    # File mapping table with search functionality
                    st.write("**File Mapping Table**")

                    # Search functionality
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        search_term = st.text_input(
                            "Search Files",
                            placeholder="Enter filename or alias to search...",
                            key="file_search"
                        )
                    with col2:
                        show_all = st.checkbox("Show All Files", value=True)

                    # Filter data based on search
                    display_df = alias_df.copy()
                    if search_term and not show_all:
                        mask = (
                            display_df['别名 (Alias)'].str.contains(search_term, case=False, na=False) |
                            display_df['原始文件名 (Original)'].str.contains(search_term, case=False, na=False)
                        )
                        display_df = display_df[mask]

                    # Display filtered results
                    if not display_df.empty:
                        st.dataframe(
                            display_df,
                            width='stretch',
                            hide_index=True,
                            column_config={
                                "别名 (Alias)": st.column_config.TextColumn("别名 (Alias)", width="small"),
                                "原始文件名 (Original)": st.column_config.TextColumn("原始文件名 (Original)", width="large"),
                                "创建时间 (Created)": st.column_config.TextColumn("创建时间 (Created)", width="small")
                            }
                        )

                        # File management actions
                        st.markdown("---")
                        st.write("**文件管理操作 (File Management Actions)**")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button("🔄 刷新数据 (Refresh Data)", type="secondary"):
                                st.rerun()

                        with col2:
                            # Bulk operations placeholder
                            st.write("批量操作功能计划中... (Bulk operations coming soon...)")

                        with col3:
                            # Privacy toggle for admin
                            admin_show_original = st.checkbox(
                                "管理员查看模式 (Admin View Mode)",
                                value=True,
                                help="管理员可以查看原始文件名"
                            )
                    else:
                        st.info("没有找到匹配的文件 (No matching files found)")

                else:
                    st.info("暂无文件别名记录 (No file aliases created yet)")
                    st.write("当用户上传音频文件进行转录时，系统会自动创建文件别名以保护隐私。")
                    st.write("(File aliases are automatically created when users upload audio files for transcription to protect privacy.)")

            except Exception as e:
                st.error(f"Error loading file mapping: {str(e)}")

        # Privacy Statistics
        with privacy_sub_tabs[1]:
            st.subheader("📊 隐私保护统计 (Privacy Protection Statistics)")

            try:
                # Display comprehensive privacy statistics
                render_file_alias_statistics()

                # Additional usage analytics
                st.markdown("---")
                st.write("**使用分析 (Usage Analytics)**")

                alias_manager = st.session_state.file_alias_manager

                # Get detailed statistics
                all_aliases = alias_manager.get_all_aliases()
                if all_aliases:
                    # Create usage analytics
                    import plotly.express as px
                    from datetime import datetime, timedelta

                    # Convert to DataFrame for analysis
                    df = pd.DataFrame(all_aliases, columns=['Alias', 'Original', 'Created'])
                    df['Created'] = pd.to_datetime(df['Created'])
                    df['Date'] = df['Created'].dt.date

                    # Files created per day
                    daily_counts = df.groupby('Date').size().reset_index(name='Count')

                    if len(daily_counts) > 1:
                        fig = px.line(
                            daily_counts,
                            x='Date',
                            y='Count',
                            title='每日文件创建趋势 (Daily File Creation Trend)',
                            labels={'Count': '文件数量 (File Count)', 'Date': '日期 (Date)'}
                        )
                        st.plotly_chart(fig, width='stretch')

                    # File type analysis (based on extension)
                    df['Extension'] = df['Original'].str.extract(r'\.([^.]+)$')
                    ext_counts = df['Extension'].value_counts()

                    if len(ext_counts) > 0:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**文件类型分布 (File Type Distribution)**")
                            st.bar_chart(ext_counts)

                        with col2:
                            st.write("**文件扩展名统计 (File Extension Statistics)**")
                            for ext, count in ext_counts.items():
                                if pd.notna(ext):
                                    st.write(f"- .{ext}: {count} 个文件")

            except Exception as e:
                st.error(f"加载统计数据时出错 (Error loading statistics): {str(e)}")

        # Admin Controls
        with privacy_sub_tabs[2]:
            render_privacy_settings_admin_panel()

        # Settings Management
        with privacy_sub_tabs[3]:
            st.subheader("⚙️ Privacy Settings Management")

            # Global privacy settings
            st.write("**Global Privacy Settings**")

            col1, col2 = st.columns(2)
            with col1:
                # Default privacy mode for new users
                default_privacy_mode = st.selectbox(
                    "新用户默认隐私模式 (Default Privacy Mode for New Users)",
                    options=[
                        ("anonymous", "匿名模式 (Anonymous Mode)"),
                        ("original", "原始文件名模式 (Original Filename Mode)")
                    ],
                    index=0,
                    format_func=lambda x: x[1]
                )

                # Auto-cleanup settings
                auto_cleanup_days = st.number_input(
                    "自动清理天数 (Auto Cleanup Days)",
                    min_value=7,
                    max_value=365,
                    value=90,
                    help="非活跃文件在指定天数后自动清理"
                )

            with col2:
                # Privacy notification settings
                show_privacy_notices = st.checkbox(
                    "显示隐私保护通知 (Show Privacy Notices)",
                    value=True,
                    help="在界面中显示隐私保护状态"
                )

                # File alias format
                alias_format = st.selectbox(
                    "别名格式 (Alias Format)",
                    options=[
                        "file_{number}",
                        "doc_{number}",
                        "audio_{number}",
                        "item_{number}"
                    ],
                    index=0
                )

            if st.button("保存全局设置 (Save Global Settings)", type="primary"):
                # Save settings logic would go here
                st.success("设置已保存 (Settings saved successfully)")

            # Manual cleanup section
            st.markdown("---")
            cleanup_old_aliases()

        # Data Export
        with privacy_sub_tabs[4]:
            st.subheader("📥 Data Export Management")

            # Export functionality
            export_alias_mapping()

            # Additional export options
            st.markdown("---")
            st.write("**高级导出选项 (Advanced Export Options)**")

            col1, col2 = st.columns(2)
            with col1:
                export_format = st.selectbox(
                    "导出格式 (Export Format)",
                    options=["JSON", "CSV", "Excel"],
                    index=0
                )

                include_inactive = st.checkbox(
                    "包含非活跃文件 (Include Inactive Files)",
                    value=False
                )

            with col2:
                date_range = st.date_input(
                    "日期范围 (Date Range)",
                    value=[],
                    help="选择要导出的日期范围"
                )

                include_statistics = st.checkbox(
                    "包含统计信息 (Include Statistics)",
                    value=True
                )

            if st.button("生成自定义导出 (Generate Custom Export)", type="secondary"):
                with st.spinner("生成导出文件中... (Generating export file...)"):
                    # Custom export logic would go here
                    st.success("导出文件已生成 (Export file generated successfully)")

            # Import functionality
            st.markdown("---")
            st.write("**Data Import**")
            st.info("Data import functionality is planned for restoring file mappings from backups.")

            uploaded_file = st.file_uploader(
                "Choose backup file",
                type=['json'],
                help="Upload previously exported JSON backup file"
            )

            if uploaded_file is not None:
                if st.button("Import Backup", type="secondary"):
                    # Import logic would go here
                    st.warning("Import functionality under development...")

    # Data Export tab
    with admin_tabs[7]:  # Data Export tab
        st.subheader("📊 数据导出 (Data Export)")

        # Create sub-tabs for different export types
        export_sub_tabs = st.tabs([
            "ASR Results",
            "Database Export",
            "Analysis Data",
            "System Logs"
        ])

        # ASR Results Export
        with export_sub_tabs[0]:
            st.markdown("### 🎤 ASR识别结果导出 (ASR Recognition Results Export)")

            try:
                # Get list of processed files
                processed_files = EVSDataUtils.get_processed_files()

                if processed_files:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### 📂 文件选择 (File Selection)")

                        # Create display names (aliases) for files if aliasing is enabled
                        if get_file_aliasing_enabled():
                            display_files = [get_file_display_name(f) for f in processed_files]
                            display_to_original = {get_file_display_name(f): f for f in processed_files}
                            original_to_display = {f: get_file_display_name(f) for f in processed_files}
                        else:
                            display_files = processed_files
                            display_to_original = {f: f for f in processed_files}
                            original_to_display = {f: f for f in processed_files}

                        # Initialize session state for selections if not exists
                        if 'selected_files_export' not in st.session_state:
                            st.session_state.selected_files_export = []
                        if 'selected_providers_export' not in st.session_state:
                            st.session_state.selected_providers_export = ["crisperwhisper"]
                        if 'selected_languages_export' not in st.session_state:
                            st.session_state.selected_languages_export = ["zh", "en"]

                        # Convert existing selections to display names
                        default_display = [original_to_display.get(f, f) for f in st.session_state.selected_files_export if f in original_to_display]

                        # File selection with select all option
                        col_file1, col_file2 = st.columns([3, 1])
                        with col_file1:
                            selected_display_files = st.multiselect(
                                "选择要导出的文件 (Select files to export)",
                                options=display_files,
                                default=default_display,
                                key="file_export_select",
                                help="可以选择多个文件进行批量导出"
                            )
                            # Convert back to original names and update session state
                            selected_files = [display_to_original.get(f, f) for f in selected_display_files]
                            st.session_state.selected_files_export = selected_files
                        with col_file2:
                            st.write("")  # Spacing
                            if st.button("🔘 全选文件 (Select All Files)", key="select_all_files"):
                                st.session_state.selected_files_export = processed_files
                                st.rerun()
                            if st.button("❌ 清空选择 (Clear Selection)", key="clear_files"):
                                st.session_state.selected_files_export = []
                                st.rerun()

                        # ASR Provider selection with select all option
                        asr_providers = ["crisperwhisper", "funasr", "google", "tencent", "ibm"]
                        col_asr1, col_asr2 = st.columns([3, 1])
                        with col_asr1:
                            selected_providers = st.multiselect(
                                "选择ASR提供商 (Select ASR Providers)",
                                options=asr_providers,
                                default=st.session_state.selected_providers_export,
                                key="provider_export_select"
                            )
                            # Update session state when selection changes
                            st.session_state.selected_providers_export = selected_providers
                        with col_asr2:
                            st.write("")  # Spacing
                            if st.button("🔘 全选提供商 (Select All Providers)", key="select_all_providers"):
                                st.session_state.selected_providers_export = asr_providers
                                st.rerun()
                            if st.button("❌ 清空选择 (Clear Selection)", key="clear_providers"):
                                st.session_state.selected_providers_export = []
                                st.rerun()

                        # Language selection with select all option
                        col_lang1, col_lang2 = st.columns([3, 1])
                        with col_lang1:
                            languages = st.multiselect(
                                "选择语言 (Select Languages)",
                                options=["zh", "en"],
                                default=st.session_state.selected_languages_export,
                                key="language_export_select"
                            )
                            # Update session state when selection changes
                            st.session_state.selected_languages_export = languages
                        with col_lang2:
                            st.write("")  # Spacing
                            if st.button("🔘 全选语言 (Select All Languages)", key="select_all_languages"):
                                st.session_state.selected_languages_export = ["zh", "en"]
                                st.rerun()
                            if st.button("❌ 清空选择 (Clear Selection)", key="clear_languages"):
                                st.session_state.selected_languages_export = []
                                st.rerun()

                    with col2:
                        st.markdown("#### ⚙️ 导出设置 (Export Settings)")

                        # Export format
                        export_format = st.selectbox(
                            "导出格式 (Export Format)",
                            options=["JSON", "Excel", "CSV"],
                            help="JSON格式包含完整的ASR识别数据结构"
                        )

                        # Include options
                        include_timestamps = st.checkbox("包含时间戳 (Include Timestamps)", value=True)
                        include_confidence = st.checkbox("包含置信度 (Include Confidence Scores)", value=True)
                        include_segments = st.checkbox("包含段落信息 (Include Segment Info)", value=True)

                        # Date range filter
                        date_filter = st.checkbox("按日期过滤 (Filter by Date)", value=False)
                        if date_filter:
                            date_range = st.date_input(
                                "选择日期范围 (Select Date Range)",
                                value=[]
                            )

                    st.markdown("---")

                    # Export options
                    st.markdown("#### 📋 导出选项 (Export Options)")
                    col_opt1, col_opt2 = st.columns(2)
                    with col_opt1:
                        create_individual_files = st.checkbox("每个文件单独的JSON (Individual JSON files)", value=False)
                    with col_opt2:
                        use_compression = st.checkbox("压缩导出文件 (Compress export files)", value=True)

                    # Export button
                    if st.button("🚀 开始导出 ASR 数据 (Start ASR Data Export)", type="primary", width='stretch'):
                        if selected_files and selected_providers:
                            with st.spinner("正在导出数据... (Exporting data...)"):
                                try:
                                    import json
                                    import tempfile
                                    import zipfile
                                    import py7zr
                                    from io import BytesIO

                                    # Get alias manager for filename mapping
                                    alias_manager = st.session_state.file_alias_manager

                                    export_data = {}
                                    individual_files = []  # Store individual JSON files for compression
                                    progress_bar = st.progress(0)
                                    total_operations = len(selected_files) * len(selected_providers)
                                    current_operation = 0

                                    for file_name in selected_files:
                                        # Get alias for the file
                                        file_alias = alias_manager.get_or_create_alias(file_name)
                                        export_data[file_alias] = {}

                                        for provider in selected_providers:
                                            current_operation += 1
                                            progress_bar.progress(current_operation / total_operations)

                                            # Get ASR data
                                            asr_data = EVSDataUtils.get_asr_data(file_name, provider)

                                            if not asr_data.empty:
                                                # Replace original file_name with alias in the data
                                                asr_data = asr_data.copy()  # Create a copy to avoid modifying original
                                                asr_data['file_name'] = file_alias

                                                # Remove combined_word field from export
                                                if 'combined_word' in asr_data.columns:
                                                    asr_data = asr_data.drop('combined_word', axis=1)

                                                # Filter by language
                                                if languages:
                                                    asr_data = asr_data[asr_data['lang'].isin(languages)]

                                                # Convert to export format
                                                if export_format == "JSON":
                                                    export_data[file_alias][provider] = asr_data.to_dict('records')
                                                else:
                                                    export_data[file_alias][provider] = asr_data

                                    # Handle export based on format and compression preferences
                                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

                                    if export_format == "JSON":
                                        if create_individual_files:
                                            # Create individual JSON files for each file-provider combination
                                            files_to_compress = []

                                            with tempfile.TemporaryDirectory() as temp_dir:
                                                for file_alias, providers_data in export_data.items():
                                                    for provider, data in providers_data.items():
                                                        if data:  # Only create file if data exists
                                                            individual_filename = f"{file_alias}_{provider}_{timestamp}.json"
                                                            individual_content = json.dumps(data, ensure_ascii=False, indent=2)
                                                            individual_filepath = os.path.join(temp_dir, individual_filename)

                                                            with open(individual_filepath, 'w', encoding='utf-8') as f:
                                                                f.write(individual_content)

                                                            files_to_compress.append((individual_filepath, individual_filename))

                                                if use_compression and files_to_compress:
                                                    # Create 7z compressed file
                                                    compressed_buffer = BytesIO()
                                                    with py7zr.SevenZipFile(compressed_buffer, 'w') as archive:
                                                        for file_path, file_name in files_to_compress:
                                                            archive.write(file_path, file_name)

                                                    compressed_filename = f"asr_export_individual_{timestamp}.7z"
                                                    st.download_button(
                                                        "📥 下载压缩文件 (Download 7z Archive)",
                                                        data=compressed_buffer.getvalue(),
                                                        file_name=compressed_filename,
                                                        mime="application/x-7z-compressed"
                                                    )

                                                elif files_to_compress:
                                                    # Show individual download buttons (if not compressed)
                                                    st.write("**Individual JSON Files:**")
                                                    for file_path, file_name in files_to_compress:
                                                        with open(file_path, 'r', encoding='utf-8') as f:
                                                            content = f.read()
                                                        st.download_button(
                                                            f"📥 {file_name}",
                                                            data=content,
                                                            file_name=file_name,
                                                            mime="application/json"
                                                        )
                                        else:
                                            # Single combined JSON file
                                            export_content = json.dumps(export_data, ensure_ascii=False, indent=2)
                                            combined_filename = f"asr_export_combined_{timestamp}.json"
                                            st.download_button(
                                                "📥 下载 JSON 文件 (Download JSON File)",
                                                data=export_content,
                                                file_name=combined_filename,
                                                mime="application/json"
                                            )

                                    elif export_format == "Excel":
                                        buffer = BytesIO()
                                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                            for file_alias, providers_data in export_data.items():
                                                for provider, data in providers_data.items():
                                                    if isinstance(data, pd.DataFrame) and not data.empty:
                                                        # Use alias in sheet name
                                                        sheet_name = f"{file_alias[:20]}_{provider}"[:31]  # Excel sheet name limit
                                                        data.to_excel(writer, sheet_name=sheet_name, index=False)

                                        excel_filename = f"asr_export_{timestamp}.xlsx"
                                        st.download_button(
                                            "📥 下载 Excel 文件 (Download Excel File)",
                                            data=buffer.getvalue(),
                                            file_name=excel_filename,
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                        )

                                    st.success("✅ 数据导出完成！(Data export completed!)")

                                except Exception as e:
                                    st.error(f"❌ 导出失败 (Export failed): {str(e)}")
                                    logger.error(f"Export error: {str(e)}", exc_info=True)
                        else:
                            st.warning("⚠️ 请选择文件和ASR提供商 (Please select files and ASR providers)")

                else:
                    st.info("📝 暂无已处理的文件 (No processed files available)")

            except Exception as e:
                st.error(f"❌ 加载文件列表失败 (Failed to load file list): {str(e)}")

        # Database Export
        with export_sub_tabs[1]:
            st.markdown("### 🗃️ 数据库数据导出 (Database Data Export)")

            try:
                # Available tables
                table_options = {
                    "asr_results": "ASR识别结果 (ASR Results)",
                    "evs_pairs": "EVS配对数据 (EVS Pairs)",
                    "edit_words": "编辑词汇 (Edited Words)",
                    "chinese_nlp_results": "中文NLP结果 (Chinese NLP Results)",
                    "si_analysis_results": "同传分析结果 (SI Analysis Results)",
                    "asr_config": "ASR配置 (ASR Configuration)"
                }

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 📋 表格选择 (Table Selection)")

                    # Table selection with buttons
                    col_table1, col_table2 = st.columns([3, 1])
                    with col_table1:
                        selected_tables = st.multiselect(
                            "选择要导出的数据表 (Select tables to export)",
                            options=list(table_options.keys()),
                            format_func=lambda x: table_options[x]
                        )
                    with col_table2:
                        st.write("")  # Spacing
                        if st.button("🔘 全选表格 (Select All Tables)", key="select_all_tables"):
                            st.session_state.selected_tables_export = list(table_options.keys())
                            st.rerun()
                        if st.button("❌ 清空选择 (Clear Selection)", key="clear_tables"):
                            st.session_state.selected_tables_export = []
                            st.rerun()

                    # Use session state for table selection
                    if 'selected_tables_export' in st.session_state:
                        selected_tables = st.session_state.selected_tables_export

                    # Quick selection presets
                    st.markdown("**快速选择预设 (Quick Selection Presets):**")
                    col_preset1, col_preset2, col_preset3 = st.columns(3)

                    with col_preset1:
                        if st.button("📊 核心数据 (Core Data)", key="preset_core", help="ASR结果 + EVS配对 + 编辑词汇"):
                            st.session_state.selected_tables_export = ["asr_results", "evs_pairs", "edit_words"]
                            st.rerun()

                    with col_preset2:
                        if st.button("🔬 分析数据 (Analysis Data)", key="preset_analysis", help="NLP结果 + SI分析结果"):
                            st.session_state.selected_tables_export = ["chinese_nlp_results", "si_analysis_results"]
                            st.rerun()

                    with col_preset3:
                        if st.button("⚙️ 配置数据 (Config Data)", key="preset_config", help="ASR配置"):
                            st.session_state.selected_tables_export = ["asr_config"]
                            st.rerun()

                    # Export all option (legacy support)
                    export_all = st.checkbox("导出所有表格 (Export All Tables)", value=False)
                    if export_all:
                        selected_tables = list(table_options.keys())

                with col2:
                    st.markdown("#### ⚙️ 导出设置 (Export Settings)")
                    db_export_format = st.selectbox(
                        "导出格式 (Export Format)",
                        options=["Excel", "CSV", "JSON"],
                        help="Excel格式将每个表格作为单独的工作表"
                    )

                    # ASR Provider Selection for relevant tables
                    st.markdown("#### 🎤 ASR Provider 过滤 (ASR Provider Filter)")

                    col_asr1, col_asr2 = st.columns([3, 1])
                    with col_asr1:
                        selected_asr_providers = st.multiselect(
                            "选择ASR提供商 (Select ASR Providers)",
                            options=list(ASR_PROVIDERS.values()),
                            default=list(ASR_PROVIDERS.values()),
                            format_func=lambda x: next(k for k, v in ASR_PROVIDERS.items() if v == x),
                            help="仅对包含ASR数据的表格有效 (Only applies to tables with ASR data)"
                        )
                    with col_asr2:
                        st.write("")  # Spacing
                        if st.button("🔘 全选ASR (Select All)", key="select_all_asr_providers"):
                            st.session_state.selected_asr_providers_db = list(ASR_PROVIDERS.values())
                            st.rerun()
                        if st.button("❌ 清空ASR (Clear ASR)", key="clear_asr_providers"):
                            st.session_state.selected_asr_providers_db = []
                            st.rerun()

                    # Use session state for ASR provider selection
                    if 'selected_asr_providers_db' in st.session_state:
                        selected_asr_providers = st.session_state.selected_asr_providers_db

                    include_metadata = st.checkbox("包含元数据 (Include Metadata)", value=True)
                    compress_output = st.checkbox("压缩输出 (Compress Output)", value=False)

                st.markdown("---")

                if st.button("🚀 开始导出数据库数据 (Start Database Export)", type="primary", width='stretch'):
                    if selected_tables or export_all:
                        with st.spinner("正在导出数据库数据... (Exporting database data...)"):
                            try:
                                export_data = {}
                                progress_bar = st.progress(0)

                                tables_to_export = selected_tables if not export_all else list(table_options.keys())

                                for i, table_name in enumerate(tables_to_export):
                                    progress_bar.progress((i + 1) / len(tables_to_export))

                                    # Get table data with ASR provider filtering where applicable
                                    if table_name == "asr_results":
                                        data = EVSDataUtils.get_all_asr_results(selected_asr_providers if selected_asr_providers else None)
                                    elif table_name == "evs_pairs":
                                        data = EVSDataUtils.get_all_evs_pairs(selected_asr_providers if selected_asr_providers else None)
                                    elif table_name == "edit_words":
                                        data = EVSDataUtils.get_all_edit_words(selected_asr_providers if selected_asr_providers else None)
                                    elif table_name == "chinese_nlp_results":
                                        # NLP results may also need ASR provider filtering if the table contains asr_provider column
                                        data = EVSDataUtils.get_all_nlp_results()
                                        if not data.empty and 'asr_provider' in data.columns and selected_asr_providers:
                                            data = data[data['asr_provider'].isin(selected_asr_providers)]
                                    elif table_name == "si_analysis_results":
                                        # Get SI analysis results with ASR provider filtering
                                        data = EVSDataUtils.get_all_si_analysis_results(selected_asr_providers if selected_asr_providers else None)

                                        # Filter to only include quality metrics columns for SI analysis
                                        if not data.empty:
                                            quality_metrics_columns = [
                                                'id', 'file_name', 'asr_provider', 'analysis_type', 'analysis_timestamp',
                                                # Core quality metrics
                                                'overall_score', 'accuracy_score', 'fluency_score', 'completeness_score', 'quality_level',
                                                # Speech rate metrics
                                                'en_wpm', 'zh_wpm', 'speed_ratio', 'pace_assessment', 'balance_assessment',
                                                # Coverage metrics
                                                'total_segments', 'bilingual_segments', 'coverage_rate',
                                                # Confidence metrics
                                                'confidence_mean', 'confidence_std', 'confidence_words_count',
                                                # Processing info
                                                'processing_time_ms', 'created_by'
                                            ]
                                            # Only keep columns that exist in the data
                                            available_columns = [col for col in quality_metrics_columns if col in data.columns]
                                            data = data[available_columns]
                                    elif table_name == "asr_config":
                                        config_data = EVSDataUtils.get_asr_config()
                                        data = pd.DataFrame([config_data]) if config_data else pd.DataFrame()
                                    else:
                                        data = pd.DataFrame()

                                    if not data.empty:
                                        # Replace original filenames with aliases for privacy protection
                                        data_copy = data.copy()
                                        if 'file_name' in data_copy.columns:
                                            alias_manager = st.session_state.file_alias_manager
                                            # Create a mapping of original names to aliases
                                            unique_files = data_copy['file_name'].unique()
                                            alias_mapping = {}
                                            for original_file in unique_files:
                                                if pd.notna(original_file) and original_file != '':
                                                    alias_mapping[original_file] = alias_manager.get_or_create_alias(str(original_file))

                                            # Apply alias mapping
                                            data_copy['file_name'] = data_copy['file_name'].map(alias_mapping).fillna(data_copy['file_name'])

                                        # Special handling for SI analysis results - focus on quality metrics
                                        if table_name == "si_analysis_results":
                                            # Add descriptive column names for better understanding
                                            if 'overall_score' in data_copy.columns:
                                                data_copy = data_copy.rename(columns={
                                                    'overall_score': 'english_wpm',
                                                    'accuracy_score': 'chinese_wpm',
                                                    'fluency_score': 'speed_ratio',
                                                    'completeness_score': 'coverage_rate_score'
                                                })

                                            # Sort by file_name and analysis_timestamp for better organization
                                            if 'analysis_timestamp' in data_copy.columns:
                                                data_copy = data_copy.sort_values(['file_name', 'analysis_timestamp'], ascending=[True, False])

                                        # Remove combined_word field from asr_results export
                                        if table_name == "asr_results" and 'combined_word' in data_copy.columns:
                                            data_copy = data_copy.drop('combined_word', axis=1)

                                        export_data[table_name] = data_copy

                                # Generate download
                                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

                                if db_export_format == "Excel":
                                    from io import BytesIO
                                    buffer = BytesIO()
                                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                        for table_name, data in export_data.items():
                                            if not data.empty:
                                                data.to_excel(writer, sheet_name=table_name, index=False)

                                    file_name = f"database_export_{timestamp}.xlsx"
                                    st.download_button(
                                        "📥 下载 Excel 文件 (Download Excel File)",
                                        data=buffer.getvalue(),
                                        file_name=file_name,
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )

                                elif db_export_format == "JSON":
                                    # Convert DataFrames to JSON
                                    json_data = {}
                                    for table_name, data in export_data.items():
                                        if not data.empty:
                                            json_data[table_name] = data.to_dict('records')

                                    export_content = json.dumps(json_data, ensure_ascii=False, indent=2)
                                    file_name = f"database_export_{timestamp}.json"
                                    st.download_button(
                                        "📥 下载 JSON 文件 (Download JSON File)",
                                        data=export_content,
                                        file_name=file_name,
                                        mime="application/json"
                                    )

                                st.success("✅ 数据库数据导出完成！(Database export completed!)")

                                # Show export summary
                                st.markdown("#### 📊 导出摘要 (Export Summary)")

                                # Display selected ASR providers
                                if selected_asr_providers:
                                    provider_names = [next(k for k, v in ASR_PROVIDERS.items() if v == provider) for provider in selected_asr_providers]
                                    st.write(f"**选择的ASR提供商 (Selected ASR Providers):** {', '.join(provider_names)}")
                                else:
                                    st.write("**ASR提供商过滤 (ASR Provider Filter):** 所有 (All)")

                                st.write("**导出的数据表 (Exported Tables):**")
                                for table_name, data in export_data.items():
                                    if not data.empty:
                                        # Show additional info for ASR-related tables
                                        if table_name in ["asr_results", "evs_pairs", "edit_words", "si_analysis_results"] and selected_asr_providers:
                                            st.write(f"- {table_options.get(table_name, table_name)}: {len(data)} 条记录 (已过滤ASR提供商)")
                                        else:
                                            st.write(f"- {table_options.get(table_name, table_name)}: {len(data)} 条记录")

                            except Exception as e:
                                st.error(f"❌ 数据库导出失败 (Database export failed): {str(e)}")
                    else:
                        st.warning("⚠️ 请选择要导出的表格 (Please select tables to export)")

            except Exception as e:
                st.error(f"❌ 加载数据库信息失败 (Failed to load database info): {str(e)}")

        # Analysis Report Export
        with export_sub_tabs[2]:
            st.markdown("### 📈 Analysis Report Generator")
            st.write("Generate comprehensive analysis reports with metrics, charts, and quality assessments.")

            # Get available files
            try:
                asr_files_df = EVSDataUtils.get_all_asr_files()
            except Exception:
                asr_files_df = pd.DataFrame()

            if asr_files_df.empty:
                st.info("No transcribed files available. Process audio files first.")
            else:
                # File selection
                file_options = asr_files_df['file_name'].unique().tolist()
                selected_file = st.selectbox(
                    "Select File",
                    options=file_options,
                    key="report_file_select"
                )

                if selected_file:
                    # Get available providers/languages for this file
                    file_records = asr_files_df[asr_files_df['file_name'] == selected_file]

                    col1, col2 = st.columns(2)
                    with col1:
                        lang_options = file_records['lang'].unique().tolist()
                        selected_lang = st.selectbox(
                            "Language",
                            options=lang_options,
                            format_func=lambda x: 'English' if x == 'en' else 'Chinese',
                            key="report_lang_select"
                        )
                    with col2:
                        provider_options = file_records[file_records['lang'] == selected_lang]['asr_provider'].unique().tolist()
                        selected_provider = st.selectbox(
                            "ASR Provider",
                            options=provider_options,
                            key="report_provider_select"
                        )

                    st.markdown("---")

                    # Generate report
                    if st.button("Generate Report", type="primary", key="generate_report_btn"):
                        from report_generator import ReportGenerator

                        with st.spinner("Generating report..."):
                            try:
                                rg = ReportGenerator(selected_file, selected_lang, selected_provider)
                                if rg.load_data():
                                    excel_bytes, pdf_bytes = rg.generate_full_report()

                                    st.session_state['report_excel'] = excel_bytes
                                    st.session_state['report_pdf'] = pdf_bytes
                                    st.session_state['report_file'] = selected_file
                                    st.session_state['report_metrics'] = rg.metrics
                                    st.success("Report generated successfully!")
                                else:
                                    st.warning("No transcription data found for this file/provider/language combination.")
                            except Exception as e:
                                logger.error(f"Report generation failed: {e}", exc_info=True)
                                st.error(f"Report generation failed: {str(e)}")

                    # Show download buttons if report is ready
                    if 'report_excel' in st.session_state and st.session_state.get('report_file') == selected_file:
                        st.markdown("---")
                        st.markdown("#### Downloads")

                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        safe_name = selected_file.replace(' ', '_').replace('.', '_')

                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            st.download_button(
                                label="Download Excel Report",
                                data=st.session_state['report_excel'],
                                file_name=f"EVS_Report_{safe_name}_{timestamp}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="download_excel_report"
                            )
                        with col_dl2:
                            st.download_button(
                                label="Download PDF Report",
                                data=st.session_state['report_pdf'],
                                file_name=f"EVS_Report_{safe_name}_{timestamp}.pdf",
                                mime="application/pdf",
                                key="download_pdf_report"
                            )

                        # Show preview metrics
                        metrics = st.session_state.get('report_metrics', {})
                        if metrics:
                            st.markdown("#### Report Preview")
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Total Words", metrics.get('total_words', 0))
                            m2.metric("Duration (min)", metrics.get('total_duration_min', 0))
                            m3.metric("Speech Rate (WPM)", metrics.get('wpm', 0))
                            m4.metric("Pauses", metrics.get('total_pauses', 0))

        # System Logs Export
        with export_sub_tabs[3]:
            st.markdown("### 📋 系统日志导出 (System Logs Export)")
            st.info("🚧 此功能正在开发中... (This feature is under development...)")

            # Placeholder for system logs export
            st.write("""
            **计划功能 (Planned Features):**
            - 应用程序日志 (Application Logs)
            - 错误日志 (Error Logs)
            - 用户操作日志 (User Activity Logs)
            - 系统性能日志 (System Performance Logs)
            """)

class SessionState:
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

def save_edit_word(interpret_file, changes, default_asr_provider):
    """Save table changes to database with improved error handling"""
    try:

        # Create a container for progress indicators
        with st.container():
            st.write("Saving changes...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            success = True
            total_changes = len(changes)

            for i, change in enumerate(changes, 1):
                try:
                    # Update progress without forcing rerun
                    progress = int((i / total_changes) * 100)
                    progress_bar.progress(progress)
                    status_text.write(f"Processing change {i} of {total_changes}...")

                    # Use provider from change dict if available, otherwise use default
                    asr_provider = change.get('asr_provider', default_asr_provider)

                    result = EVSDataUtils.update_edit_word(
                        interpret_file=interpret_file,
                        lang=change['lang'],
                        segment_id=int(change['segment_id']),
                        word_seq_no=int(change['word_seq_no']),
                        edit_word=change['new_value'],
                        asr_provider=asr_provider
                    )

                    if not result:
                        success = False
                        logger.error(f"Database update failed for change: {change}")

                except KeyError as ke:
                    # record specific key error
                    success = False
                    logger.error(f"Unexpected error processing change: {str(ke)}", exc_info=True)
                    status_text.error(f"Error processing change {i}: Missing required field - {str(ke)}")

                except ValueError as ve:
                    # value conversion error
                    success = False
                    logger.error(f"Value error processing change: {str(ve)}", exc_info=True)
                    status_text.error(f"Error processing change {i}: Invalid value - {str(ve)}")

                except Exception as e:
                    # other unexpected errors
                    success = False
                    logger.error(f"Unexpected error processing change: {str(e)}", exc_info=True)
                    status_text.error(f"Error processing change {i}: {str(e)}")

            # Final status update
            if success:
                status_text.success("All changes saved successfully!")
            else:
                status_text.warning("Some changes could not be saved. See log for details.")

            return success

    except Exception as e:
        logger.error(f"Error in save_edit_word: {str(e)}", exc_info=True)
        st.error(f"An error occurred while saving changes: {str(e)}")
        return False


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

def process_chinese_nlp_unified(file_name: str, asr_provider: str, engine: str = "jieba") -> bool:
    """
    Process Chinese text with unified NLP processor supporting multiple engines

    Args:
        file_name: Name of the file to process
        asr_provider: ASR provider name
        engine: NLP engine to use ('jieba' or 'hanlp')

    Returns:
        True if processing successful, False otherwise
    """
    try:
        # Initialize unified NLP processor
        nlp_processor = create_nlp_processor(engine)

        # Get engine info
        engine_info = nlp_processor.get_engine_info()
        logger.info(f"Using NLP engine: {engine_info}")

        # Get ASR data from database
        asr_data = EVSDataUtils.get_asr_data(file_name, asr_provider)

        if asr_data.empty:
            logger.warning(f"No ASR data found for file: {file_name}")
            return False

        # Filter Chinese data only
        chinese_data = asr_data[asr_data['lang'] == 'zh'].copy()

        if chinese_data.empty:
            logger.warning(f"No Chinese data found for file: {file_name}")
            return False

        # Combine all Chinese text for document-level processing
        all_chinese_text = " ".join(chinese_data['edit_word'].fillna(chinese_data['word']).astype(str))

        if not all_chinese_text.strip():
            logger.warning("No valid Chinese text found")
            return False

        logger.info(f"Processing document with {len(all_chinese_text)} characters using {engine}")

        # Process with NLP
        nlp_words = nlp_processor.segment_text(all_chinese_text)
        pos_tags = nlp_processor.pos_tag(all_chinese_text)

        # Create comparison data
        original_words = list(chinese_data['edit_word'].fillna(chinese_data['word']).astype(str))
        comparison = {
            'original_text': all_chinese_text,
            'original_words': original_words,
            'original_word_count': len(original_words),
            'nlp_words': nlp_words,
            'nlp_word_count': len(nlp_words),
            'pos_tags': pos_tags,
            'improvement_ratio': len(nlp_words) / len(original_words) if original_words else 0,
            'engine_used': engine,
            'engine_info': engine_info,
            'processed_at': datetime.now().isoformat()
        }

        # Update database with results
        pos_dict = {word: pos for word, pos in pos_tags}
        success = update_document_nlp_results_unified(file_name, asr_provider, chinese_data,
                                                    nlp_words, pos_dict, comparison, engine)

        if success:
            logger.info(f"Document-level Chinese NLP processing completed for file: {file_name} using {engine}")
            return True
        else:
            logger.error("Failed to update database with NLP results")
            return False

    except Exception as e:
        logger.error(f"Error in Chinese NLP processing with {engine}: {str(e)}")
        return False

def calculate_word_similarity(word1: str, word2: str) -> float:
    """
    Calculate similarity between two words

    Args:
        word1: First word
        word2: Second word

    Returns:
        Similarity score between 0 and 1
    """
    if not word1 or not word2:
        return 0.0

    if word1 == word2:
        return 1.0

    # Simple character overlap similarity
    set1 = set(word1)
    set2 = set(word2)

    if not set1 or not set2:
        return 0.0

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection / union if union > 0 else 0.0

def update_document_nlp_results_unified(file_name: str, asr_provider: str, chinese_data: pd.DataFrame,
                                       nlp_words: List[str], pos_dict: Dict[str, str],
                                       comparison: Dict, engine: str) -> bool:
    """
    Update database with unified NLP results

    Args:
        file_name: File name
        asr_provider: ASR provider
        chinese_data: All Chinese words DataFrame
        nlp_words: NLP segmented words for entire document
        pos_dict: POS tags dictionary
        comparison: Document-level comparison results
        engine: NLP engine used

    Returns:
        True if update successful, False otherwise
    """
    try:
        with EVSDataUtils.get_db_connection() as conn:
            cursor = conn.cursor()

            # Add new columns if they don't exist
            try:
                cursor.execute("""
                    ALTER TABLE asr_results_words
                    ADD COLUMN nlp_engine TEXT
                """)
                cursor.execute("""
                    ALTER TABLE asr_results_words
                    ADD COLUMN nlp_engine_info TEXT
                """)
            except sqlite3.OperationalError:
                pass  # Columns already exist

            # Update each word with NLP results
            update_query = """
                UPDATE asr_results_words
                SET nlp_word = ?, nlp_pos = ?, nlp_confidence = ?,
                    nlp_processed_at = ?, nlp_comparison = ?,
                    nlp_engine = ?, nlp_engine_info = ?
                WHERE file_name = ? AND asr_provider = ? AND segment_id = ? AND word_seq_no = ?
            """

            # Simple mapping strategy: distribute NLP words across original words
            total_original = len(chinese_data)
            total_nlp = len(nlp_words)

            for i, (idx, word_row) in enumerate(chinese_data.iterrows()):
                # Calculate which NLP word to assign
                nlp_index = min(int(i * total_nlp / total_original), total_nlp - 1) if total_nlp > 0 else 0

                if nlp_index < len(nlp_words):
                    nlp_word = nlp_words[nlp_index]
                    nlp_pos = pos_dict.get(nlp_word, 'x')

                    # Calculate confidence based on similarity
                    original_word = str(word_row.get('edit_word', word_row.get('word', '')))
                    confidence = calculate_word_similarity(original_word, nlp_word)

                    cursor.execute(update_query, [
                        nlp_word,
                        nlp_pos,
                        confidence,
                        datetime.now().isoformat(),
                        json.dumps(comparison, ensure_ascii=False, default=str),
                        engine,
                        json.dumps(comparison.get('engine_info', {}), ensure_ascii=False, default=str),
                        file_name,
                        asr_provider,
                        word_row['segment_id'],
                        word_row['word_seq_no']
                    ])

            conn.commit()
            logger.info(f"Updated {len(chinese_data)} words with {engine} NLP results")
            return True

    except Exception as e:
        logger.error(f"Error updating database with {engine} NLP results: {str(e)}")
        return False

def render_file_list_tab():
    """Render the File List tab showing all transcribed files with statistics."""
    st.subheader("📁 Transcribed Files")

    # Refresh button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🔄 Refresh", key="refresh_file_list", help="Refresh file list from database"):
            # Clear model pairs cache
            keys_to_delete = [k for k in st.session_state.keys() if k.startswith('evs_model_pairs_')]
            for k in keys_to_delete:
                del st.session_state[k]
            st.rerun()

    # Get all files from asr_files table
    try:
        files_df = EVSDataUtils.get_all_asr_files()

        if files_df.empty:
            st.info("No transcribed files found. Please transcribe some audio files first.")
            return

        # Summary statistics
        st.markdown("### Summary")
        col1, col2, col3, col4 = st.columns(4)

        unique_files = files_df['file_name'].nunique()
        total_segments = files_df['total_segments'].sum()
        total_words = files_df['total_words'].sum()
        total_records = len(files_df)

        with col1:
            st.metric("Total Files", unique_files)
        with col2:
            st.metric("Total Transcriptions", total_records, help="Each file can have multiple transcriptions (EN/ZH with different models)")
        with col3:
            st.metric("Total Segments", f"{total_segments:,}")
        with col4:
            st.metric("Total Words", f"{total_words:,}")

        st.divider()

        # Detailed file list
        st.markdown("### File Details")

        # Create a more readable display DataFrame
        display_df = files_df.copy()

        # Apply file aliasing for privacy protection
        display_df['file_name'] = display_df['file_name'].apply(get_file_display_name)

        # Format the display
        display_df['Language'] = display_df['lang'].map({'en': '🇬🇧 English', 'zh': '🇨🇳 Chinese'})
        display_df['Provider/Model'] = display_df['asr_provider'] + '/' + display_df['model']
        display_df['Segments'] = display_df['total_segments']
        display_df['Words'] = display_df['total_words']
        display_df['Duration (s)'] = display_df['slice_duration']
        display_df['Created'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')

        # Select columns to display
        display_cols = ['file_name', 'Language', 'Provider/Model', 'Segments', 'Words', 'Duration (s)', 'Created']
        display_df = display_df[display_cols]
        display_df.columns = ['File Name', 'Language', 'Provider/Model', 'Segments', 'Words', 'Duration (s)', 'Created']

        # Display the dataframe
        st.dataframe(
            display_df,
            column_config={
                'File Name': st.column_config.TextColumn('File Name', width='large'),
                'Language': st.column_config.TextColumn('Language', width='small'),
                'Provider/Model': st.column_config.TextColumn('Provider/Model', width='medium'),
                'Segments': st.column_config.NumberColumn('Segments', width='small'),
                'Words': st.column_config.NumberColumn('Words', width='small'),
                'Duration (s)': st.column_config.NumberColumn('Duration (s)', width='small'),
                'Created': st.column_config.TextColumn('Created', width='medium'),
            },
            hide_index=True,
            width='stretch'
        )

        st.divider()

        # Per-file summary (grouped)
        st.markdown("### Per-File Summary")

        # Group by file_name
        file_summary = files_df.groupby('file_name').agg({
            'lang': lambda x: ', '.join(sorted(x.unique())),
            'total_segments': 'sum',
            'total_words': 'sum',
            'asr_provider': lambda x: ', '.join(sorted(x.unique())),
            'model': lambda x: ', '.join(sorted(x.unique())),
            'created_at': 'max'
        }).reset_index()

        file_summary.columns = ['File Name', 'Languages', 'Total Segments', 'Total Words', 'Providers', 'Models', 'Last Updated']
        # Apply file aliasing for privacy protection
        file_summary['File Name'] = file_summary['File Name'].apply(get_file_display_name)
        file_summary['Languages'] = file_summary['Languages'].replace({'en': '🇬🇧', 'zh': '🇨🇳', 'en, zh': '🇬🇧 🇨🇳', 'zh, en': '🇬🇧 🇨🇳'})
        file_summary['Last Updated'] = pd.to_datetime(file_summary['Last Updated']).dt.strftime('%Y-%m-%d %H:%M')

        st.dataframe(
            file_summary,
            column_config={
                'File Name': st.column_config.TextColumn('File Name', width='large'),
                'Languages': st.column_config.TextColumn('Lang', width='small'),
                'Total Segments': st.column_config.NumberColumn('Segments', width='small'),
                'Total Words': st.column_config.NumberColumn('Words', width='small'),
                'Providers': st.column_config.TextColumn('Providers', width='medium'),
                'Models': st.column_config.TextColumn('Models', width='medium'),
                'Last Updated': st.column_config.TextColumn('Updated', width='medium'),
            },
            hide_index=True,
            width='stretch'
        )

        st.divider()

        # Audio Player
        st.markdown("### 🔊 Audio Player")

        audio_col1, audio_col2 = st.columns([2, 2])
        with audio_col1:
            audio_file_select = st.selectbox(
                "Select file",
                options=[""] + files_df['file_name'].unique().tolist(),
                format_func=lambda x: "-- Select a file --" if x == "" else get_file_display_name(x),
                key='audio_player_file'
            )

        if audio_file_select:
            file_stem = os.path.splitext(audio_file_select)[0]
            file_audio_dir = os.path.join(SLICE_AUDIO_PATH, file_stem)

            # Original audio
            original_path = os.path.join(file_audio_dir, "original.mp3")
            if os.path.exists(original_path):
                st.markdown("**Original Audio**")
                st.audio(original_path)
            else:
                st.info("Original audio not available (only saved for new uploads).")

            # Find all available sliced audio subdirectories (provider/model_folder)
            audio_folders = []
            if os.path.isdir(file_audio_dir):
                for provider in sorted(os.listdir(file_audio_dir)):
                    provider_dir = os.path.join(file_audio_dir, provider)
                    if os.path.isdir(provider_dir):
                        for model_folder in sorted(os.listdir(provider_dir)):
                            model_dir = os.path.join(provider_dir, model_folder)
                            if os.path.isdir(model_dir):
                                mp3_files = sorted(
                                    [f for f in os.listdir(model_dir) if f.endswith('.mp3')],
                                    key=lambda f: int(os.path.splitext(f)[0]) if os.path.splitext(f)[0].isdigit() else 0
                                )
                                if mp3_files:
                                    audio_folders.append({
                                        'label': f"{provider} / {model_folder}",
                                        'path': model_dir,
                                        'files': mp3_files
                                    })

            if audio_folders:
                with audio_col2:
                    folder_labels = [f['label'] for f in audio_folders]
                    selected_folder_label = st.selectbox(
                        "Provider / Model",
                        options=folder_labels,
                        index=0,
                        key='audio_player_folder'
                    )
                selected_folder = next(f for f in audio_folders if f['label'] == selected_folder_label)

                st.markdown(f"**Sliced Audio — {len(selected_folder['files'])} segments**")

                for mp3_file in selected_folder['files']:
                    mp3_path = os.path.join(selected_folder['path'], mp3_file)
                    slice_num = os.path.splitext(mp3_file)[0]
                    st.markdown(f"**Slice {slice_num}**")
                    st.audio(mp3_path)
            else:
                st.info("No sliced audio files found for this file.")

        st.divider()

        # File Content Viewer
        st.markdown("### 📄 View File Content")

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            view_file = st.selectbox(
                "Select file to view",
                options=[""] + files_df['file_name'].unique().tolist(),
                format_func=lambda x: "-- Select a file --" if x == "" else get_file_display_name(x),
                key='view_file_content'
            )

        # Get model pairs for selected file
        view_model_pair = None
        if view_file:
            view_model_pairs = EVSDataUtils.get_file_model_pairs(view_file)
            with col2:
                if view_model_pairs:
                    pair_options = [p['display_name'] for p in view_model_pairs]
                    selected_pair_display = st.selectbox(
                        "Model Pair",
                        options=pair_options,
                        index=0,
                        key='view_model_pair'
                    )
                    view_model_pair = next((p for p in view_model_pairs if p['display_name'] == selected_pair_display), view_model_pairs[0])

            with col3:
                view_language = st.selectbox(
                    "Language",
                    options=['All', 'en', 'zh'],
                    index=0,
                    key='view_language'
                )

        if view_file and view_model_pair:
            # Create tabs for Segments and Words
            content_tabs = st.tabs(["📝 Segments", "🔤 Words"])

            with content_tabs[0]:  # Segments tab
                st.markdown("#### Segments")
                try:
                    with EVSDataUtils.get_db_connection() as conn:
                        # Build query based on selections
                        query = """
                            SELECT segment_id, lang, text, edit_text, start_time, end_time, duration,
                                   asr_provider, model, created_at
                            FROM asr_results_segments
                            WHERE file_name = ?
                        """
                        params = [view_file]

                        # Filter by provider based on language
                        if view_language == 'en' and view_model_pair['en_provider']:
                            query += " AND asr_provider = ? AND lang = 'en'"
                            params.append(view_model_pair['en_provider'])
                        elif view_language == 'zh' and view_model_pair['zh_provider']:
                            query += " AND asr_provider = ? AND lang = 'zh'"
                            params.append(view_model_pair['zh_provider'])
                        elif view_language == 'All':
                            # Get both languages with their respective providers
                            providers = []
                            if view_model_pair['en_provider']:
                                providers.append(view_model_pair['en_provider'])
                            if view_model_pair['zh_provider'] and view_model_pair['zh_provider'] not in providers:
                                providers.append(view_model_pair['zh_provider'])
                            if providers:
                                placeholders = ','.join(['?' for _ in providers])
                                query += f" AND asr_provider IN ({placeholders})"
                                params.extend(providers)

                        query += " ORDER BY lang, segment_id"

                        segments_df = pd.read_sql_query(query, conn, params=params)

                    if segments_df.empty:
                        st.info("No segments found for the selected criteria.")
                    else:
                        st.write(f"Found {len(segments_df)} segments")

                        # Format for display
                        segments_df['Language'] = segments_df['lang'].map({'en': '🇬🇧', 'zh': '🇨🇳'})
                        segments_df['Time'] = segments_df.apply(
                            lambda x: f"{x['start_time']:.2f}s - {x['end_time']:.2f}s" if pd.notna(x['end_time']) else f"{x['start_time']:.2f}s",
                            axis=1
                        )

                        display_segments = segments_df[['segment_id', 'Language', 'text', 'edit_text', 'Time', 'duration', 'asr_provider', 'model']]
                        display_segments.columns = ['ID', 'Lang', 'Original Text', 'Edited Text', 'Time', 'Duration', 'Provider', 'Model']

                        st.dataframe(
                            display_segments,
                            column_config={
                                'ID': st.column_config.NumberColumn('ID', width='small'),
                                'Lang': st.column_config.TextColumn('Lang', width='small'),
                                'Original Text': st.column_config.TextColumn('Original Text', width='large'),
                                'Edited Text': st.column_config.TextColumn('Edited Text', width='large'),
                                'Time': st.column_config.TextColumn('Time', width='medium'),
                                'Duration': st.column_config.NumberColumn('Duration', format="%.2f", width='small'),
                                'Provider': st.column_config.TextColumn('Provider', width='small'),
                                'Model': st.column_config.TextColumn('Model', width='small'),
                            },
                            hide_index=True,
                            width='stretch',
                            height=400
                        )
                except Exception as e:
                    st.error(f"Error loading segments: {str(e)}")

            with content_tabs[1]:  # Words tab
                st.markdown("#### Words")
                try:
                    with EVSDataUtils.get_db_connection() as conn:
                        # Build query based on selections
                        query = """
                            SELECT segment_id, word_seq_no, lang, word, edit_word, start_time, end_time,
                                   duration, confidence, asr_provider, model
                            FROM asr_results_words
                            WHERE file_name = ?
                        """
                        params = [view_file]

                        # Filter by provider based on language
                        if view_language == 'en' and view_model_pair['en_provider']:
                            query += " AND asr_provider = ? AND lang = 'en'"
                            params.append(view_model_pair['en_provider'])
                        elif view_language == 'zh' and view_model_pair['zh_provider']:
                            query += " AND asr_provider = ? AND lang = 'zh'"
                            params.append(view_model_pair['zh_provider'])
                        elif view_language == 'All':
                            providers = []
                            if view_model_pair['en_provider']:
                                providers.append(view_model_pair['en_provider'])
                            if view_model_pair['zh_provider'] and view_model_pair['zh_provider'] not in providers:
                                providers.append(view_model_pair['zh_provider'])
                            if providers:
                                placeholders = ','.join(['?' for _ in providers])
                                query += f" AND asr_provider IN ({placeholders})"
                                params.extend(providers)

                        query += " ORDER BY lang, segment_id, word_seq_no"

                        words_df_view = pd.read_sql_query(query, conn, params=params)

                    if words_df_view.empty:
                        st.info("No words found for the selected criteria.")
                    else:
                        st.write(f"Found {len(words_df_view)} words")

                        # Format for display
                        words_df_view['Language'] = words_df_view['lang'].map({'en': '🇬🇧', 'zh': '🇨🇳'})
                        words_df_view['Time'] = words_df_view['start_time'].apply(lambda x: f"{x:.2f}s")
                        words_df_view['Confidence'] = words_df_view['confidence'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "-")

                        display_words = words_df_view[['segment_id', 'word_seq_no', 'Language', 'word', 'edit_word', 'Time', 'duration', 'Confidence', 'asr_provider', 'model']]
                        display_words.columns = ['Seg', 'Seq', 'Lang', 'Original', 'Edited', 'Time', 'Duration', 'Conf', 'Provider', 'Model']

                        st.dataframe(
                            display_words,
                            column_config={
                                'Seg': st.column_config.NumberColumn('Seg', width='small'),
                                'Seq': st.column_config.NumberColumn('Seq', width='small'),
                                'Lang': st.column_config.TextColumn('Lang', width='small'),
                                'Original': st.column_config.TextColumn('Original', width='medium'),
                                'Edited': st.column_config.TextColumn('Edited', width='medium'),
                                'Time': st.column_config.TextColumn('Time', width='small'),
                                'Duration': st.column_config.NumberColumn('Dur', format="%.3f", width='small'),
                                'Conf': st.column_config.TextColumn('Conf', width='small'),
                                'Provider': st.column_config.TextColumn('Provider', width='small'),
                                'Model': st.column_config.TextColumn('Model', width='small'),
                            },
                            hide_index=True,
                            width='stretch',
                            height=400
                        )

                        # Show word statistics
                        st.markdown("##### Word Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            en_words = len(words_df_view[words_df_view['lang'] == 'en'])
                            st.metric("English Words", en_words)
                        with col2:
                            zh_words = len(words_df_view[words_df_view['lang'] == 'zh'])
                            st.metric("Chinese Words", zh_words)
                        with col3:
                            avg_conf = words_df_view['confidence'].mean()
                            st.metric("Avg Confidence", f"{avg_conf:.2%}" if pd.notna(avg_conf) else "-")

                except Exception as e:
                    st.error(f"Error loading words: {str(e)}")

        elif view_file and not view_model_pair:
            st.warning("No model pairs found for this file.")

        # Delete functionality (admin only)
        if st.session_state.get('is_admin', False):
            st.divider()
            st.markdown("### 🗑️ Delete Transcription Data")
            st.warning("⚠️ This will permanently delete transcription data. Use with caution!")

            col1, col2 = st.columns([3, 1])
            with col1:
                file_to_delete = st.selectbox(
                    "Select file to delete",
                    options=[""] + files_df['file_name'].unique().tolist(),
                    format_func=lambda x: "-- Select a file --" if x == "" else get_file_display_name(x),
                    key='file_to_delete'
                )
            with col2:
                st.write("")  # Spacer
                st.write("")  # Spacer
                if st.button("🗑️ Delete", type="secondary", key="delete_file_btn", disabled=not file_to_delete):
                    if file_to_delete:
                        try:
                            with EVSDataUtils.get_db_connection() as conn:
                                cursor = conn.cursor()
                                # Delete from all tables
                                cursor.execute("DELETE FROM asr_results_words WHERE file_name = ?", (file_to_delete,))
                                words_deleted = cursor.rowcount
                                cursor.execute("DELETE FROM asr_results_segments WHERE file_name = ?", (file_to_delete,))
                                segments_deleted = cursor.rowcount
                                cursor.execute("DELETE FROM asr_files WHERE file_name = ?", (file_to_delete,))
                                files_deleted = cursor.rowcount
                                conn.commit()

                            st.success(f"Deleted: {words_deleted} words, {segments_deleted} segments, {files_deleted} file records")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting file: {str(e)}")

    except Exception as e:
        logger.error(f"Error rendering file list: {str(e)}")
        st.error(f"Error loading file list: {str(e)}")


def render_edit_transcription_tab():
    # Render the word edit tab content
    # Fixed source/target: EN (Source) -> ZH (Target)
    source_lang = 'en'
    target_lang = 'zh'

    # Get all files with their transcription info
    files_df = EVSDataUtils.get_all_files_with_transcriptions()

    if files_df.empty:
        st.warning("No transcription files found. Please upload and transcribe audio files first.")
        return

    # Get NLP engine from admin config
    nlp_config = EVSDataUtils.get_asr_config('nlp')
    if nlp_config and 'config' in nlp_config:
        nlp_cfg = nlp_config['config']
        if isinstance(nlp_cfg, str):
            try:
                nlp_cfg = json.loads(nlp_cfg)
            except:
                nlp_cfg = {}
        nlp_engine = nlp_cfg.get('engine', 'jieba')
    else:
        nlp_engine = 'jieba'

    # Row 1: File selection, model pair selection, NLP engine display, Save button
    col1, col2, col3, col4 = st.columns([1, 2, 1, 1])

    with col1:
        # File selection dropdown - no default selection
        file_options = [""] + files_df['file_name'].tolist()
        interpret_file = st.selectbox(
            "Select File",
            options=file_options,
            index=0,
            format_func=lambda x: "-- Select a file --" if x == "" else get_file_display_name(x),
            key='edit_transcription_file'
        )
        # Convert empty string to None for easier handling
        if interpret_file == "":
            interpret_file = None

    # Get available model pairs for selected file
    edit_model_pairs = []
    edit_selected_pair = None
    file_info = None

    with col2:
        if interpret_file:
            edit_model_pairs = EVSDataUtils.get_file_model_pairs(interpret_file)
            if edit_model_pairs:
                if len(edit_model_pairs) == 1:
                    edit_selected_pair = edit_model_pairs[0]
                    st.markdown(f"**Model:** {edit_selected_pair['display_name']}")
                else:
                    pair_options = [p['display_name'] for p in edit_model_pairs]
                    selected_display = st.selectbox(
                        "Select Model Pair",
                        options=pair_options,
                        index=0,
                        key='edit_model_pair_select'
                    )
                    edit_selected_pair = next((p for p in edit_model_pairs if p['display_name'] == selected_display), edit_model_pairs[0])

    # Convert to file_info dict for compatibility
    if edit_selected_pair:
        file_info = {
            'en_provider': edit_selected_pair['en_provider'],
            'en_model': edit_selected_pair['en_model'],
            'zh_provider': edit_selected_pair['zh_provider'],
            'zh_model': edit_selected_pair['zh_model'],
            'slice_duration': edit_selected_pair['slice_duration']
        }

    with col3:
        # Display configured NLP engine (read-only)
        st.info(f"NLP: {nlp_engine}")

    with col4:
        if st.button("Save", type="primary", key='btn_save'):
            changes = st.session_state.get('changes', [])
            if changes:
                with st.spinner("Saving changes..."):
                    # Get provider from file info
                    en_provider = file_info['en_provider'] if file_info is not None else 'crisperwhisper'
                    success = save_edit_word(interpret_file, changes, en_provider)
                    if success:
                        clear_session_data()
                        st.success("Changes saved successfully!")
                        time.sleep(0.5)
                        st.rerun()
            else:
                st.warning("No changes to save")

    # Store file info in session state
    if file_info is not None:
        st.session_state.edit_file_info = {
            'en_provider': file_info['en_provider'],
            'en_model': file_info['en_model'],
            'zh_provider': file_info['zh_provider'],
            'zh_model': file_info['zh_model'],
            'slice_duration': file_info['slice_duration'],
            'nlp_engine': nlp_engine
        }
        edit_model = file_info['en_model'] if file_info['en_model'] else file_info['zh_model']
    else:
        edit_model = None

    # Clear stored data if file or model changes
    current_edit_key = f"{interpret_file}_{edit_model}"
    if 'last_edit_key' in st.session_state and st.session_state.last_edit_key != current_edit_key:
        clear_session_data()
    st.session_state.last_edit_key = current_edit_key

    # Get provider from file info
    file_info_dict = st.session_state.get('edit_file_info', {})
    primary_provider = file_info_dict.get('en_provider', 'crisperwhisper')

    # Show NLP comparison if requested
    if st.session_state.get('show_nlp_comparison', False):
        show_nlp_engine_comparison(interpret_file, primary_provider)

    if not interpret_file:
        st.warning("Please select a file")
        return

    # File management section
    with st.expander("File Management", expanded=False):
        col_del1, col_del2 = st.columns([3, 1])
        with col_del1:
            # Show file stats for selected model
            stats = EVSDataUtils.get_file_stats(interpret_file, primary_provider, edit_model)
            display_file = get_file_display_name(interpret_file)
            st.write(f"**File:** {display_file} | **Model:** {edit_model}")
            st.write(f"**Words:** {stats['words_count']} | **Segments:** {stats['segments_count']} | **Languages:** {', '.join(stats['languages'])}")

        with col_del2:
            # Delete button with confirmation
            if st.button("Delete File", type="secondary", key='btn_delete_file'):
                st.session_state.confirm_delete = True

        # Confirmation dialog
        if st.session_state.get('confirm_delete', False):
            st.warning(f"Are you sure you want to delete '{display_file}'? This action cannot be undone.")
            col_confirm1, col_confirm2, col_confirm3 = st.columns([1, 1, 2])
            with col_confirm1:
                if st.button("Yes, Delete", type="primary", key='btn_confirm_delete'):
                    with st.spinner("Deleting..."):
                        success = EVSDataUtils.delete_file(interpret_file, primary_provider, edit_model)
                        if success:
                            st.success(f"File '{get_file_display_name(interpret_file)}' deleted successfully!")
                            st.session_state.confirm_delete = False
                            clear_session_data()
                            st.cache_data.clear()
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Failed to delete file")
                            st.session_state.confirm_delete = False
            with col_confirm2:
                if st.button("Cancel", key='btn_cancel_delete'):
                    st.session_state.confirm_delete = False
                    st.rerun()

    # Display instruction
    st.info(f"Language Pair: English -> Chinese | Double click the table cell to edit the transcription")

    if interpret_file:
        # Get slice_duration from file info, ensure it's a valid number
        slice_duration = file_info_dict.get('slice_duration', 30)
        # Safety conversion in case of bytes or invalid type from database
        if isinstance(slice_duration, bytes):
            try:
                import struct
                slice_duration = struct.unpack('<q', slice_duration)[0] if len(slice_duration) == 8 else 30
            except:
                slice_duration = 30
        slice_duration = float(slice_duration) if slice_duration else 30
        # Get both providers for correct language-specific updates
        en_provider = file_info_dict.get('en_provider', 'crisperwhisper')
        zh_provider = file_info_dict.get('zh_provider', 'funasr')
        display_edit_transcription(interpret_file, slice_duration, en_provider, zh_provider, source_lang, target_lang, edit_model)

def display_edit_transcription(interpret_file: str, slice_duration: int, en_provider: str, zh_provider: str, source_lang: str = 'en', target_lang: str = 'zh', model: str = None):
    """Display transcription editing table, fetching data from asr_results_words and using sliced audio files"""
    # Check if this is initial load (show progress bar only on first load)
    edit_render_key = f"edit_rendered_{interpret_file}_{model}"
    is_initial_render = edit_render_key not in st.session_state

    # Show progress bar during initial load
    if is_initial_render:
        loading_progress = st.progress(0)
        loading_text = st.empty()
        loading_text.text("Loading transcription data...")
        loading_progress.progress(10)
    else:
        loading_progress = None
        loading_text = None

    # Auto-detect providers/models used for each language
    file_asr_info = EVSDataUtils.get_file_asr_info(interpret_file)

    if is_initial_render and loading_progress:
        loading_progress.progress(20)
        loading_text.text("Detecting ASR configuration...")

    # Get provider/model for each language (auto-detected or fallback to provided values)
    en_info = file_asr_info.get('en')
    zh_info = file_asr_info.get('zh')

    # Use auto-detected providers if available, otherwise use passed parameters
    detected_en_provider = en_info['provider'] if en_info else en_provider
    en_model = en_info['model'] if en_info else model
    detected_zh_provider = zh_info['provider'] if zh_info else zh_provider
    zh_model = zh_info['model'] if zh_info else model

    # Update provider variables for use in the rest of the function
    en_provider = detected_en_provider
    zh_provider = detected_zh_provider

    # Show info about detected ASR settings
    if en_info or zh_info:
        with st.expander("Detected ASR Configuration", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                if en_info:
                    st.info(f"English: {en_provider} / {en_model}")
                else:
                    st.warning("English: No data found")
            with col2:
                if zh_info:
                    st.info(f"Chinese: {zh_provider} / {zh_model}")
                else:
                    st.warning("Chinese: No data found")

    if is_initial_render and loading_progress:
        loading_progress.progress(30)
        loading_text.text("Fetching data from database...")

    # Get data from database with per-language provider/model
    data = EVSDataUtils.get_asr_data(
        interpret_file,
        en_provider=en_provider, en_model=en_model,
        zh_provider=zh_provider, zh_model=zh_model
    )

    # Check for empty data
    if data.empty:
        if loading_progress:
            loading_progress.empty()
            loading_text.empty()
        st.warning("No transcription data found for this file.")
        st.info("Please ensure the audio has been processed with ASR first.")
        return

    if is_initial_render and loading_progress:
        loading_progress.progress(50)
        loading_text.text("Processing data...")

    df = data.copy()

    split_columns = df['combined_word'].str.split('&&', expand=True)

    # ensure split columns do not exceed 4 columns
    if len(split_columns.columns) < 4:
        for i in range(len(split_columns.columns), 4):
            split_columns[i] = ""

    # only use the first 4 columns
    split_columns = split_columns.iloc[:, 0:4]
    split_columns.columns = ['edit_word', 'pair_type', 'original_word', 'annotate']

    # assign split columns to df
    df[['edit_word', 'pair_type', 'original_word', 'annotate']] = split_columns

    # format timestamp
    df['timestamp'] = pd.to_datetime(df['start_time'], unit='s')
    df['time_str'] = df['timestamp'].dt.strftime('%M:%S.%f').str[:-4]
    df['time_group'] = df['timestamp'].dt.floor(f'{slice_duration}s')

    if is_initial_render and loading_progress:
        loading_progress.progress(60)
        loading_text.text("Rendering tables...")

    # Render the tables
    total_groups = len(df.groupby('time_group'))
    for idx, (name, group) in enumerate(df.groupby('time_group')):
        time_group_key = name.strftime('%M:%S')
        # calculate group number from minutes and seconds
        minutes = int(time_group_key.split(':')[0])
        seconds = int(time_group_key.split(':')[1])
        group_number = int((minutes * 60 + seconds) / slice_duration)

        # Update progress during initial render
        if is_initial_render and loading_progress:
            progress_pct = 60 + int((idx / total_groups) * 40)
            loading_progress.progress(progress_pct)
            loading_text.text(f"Rendering time group {idx + 1} of {total_groups}...")

        # create audio path, using group number with provider/model subfolder
        file_stem = os.path.splitext(interpret_file)[0]
        model_folder = f"{en_model}_{zh_model}" if en_model and zh_model else None
        if model_folder:
            audio_path = os.path.join(SLICE_AUDIO_PATH, file_stem, en_provider, model_folder, f'{group_number}.mp3')
        else:
            audio_path = os.path.join(SLICE_AUDIO_PATH, file_stem, f'{group_number}.mp3')
        col1, col2 = st.columns([2, 3])
        with col1:
            st.header(f"Time: {time_group_key}")

        with col2:
            if os.path.exists(audio_path):
                st.audio(audio_path, start_time=0)


        # get all unique timestamps in this group
        timestamps = sorted(group['time_str'].unique())

        # create labels based on language pair
        source_label = "English (Source)" if source_lang == 'en' else "Chinese (Source)"
        target_label = "Chinese (Target)" if target_lang == 'zh' else "English (Target)"

        # create a data dictionary containing two rows
        data_dict = {
            'lang': [source_label, target_label],
        }

        # add timestamp columns
        for ts in timestamps:
            data_dict[ts] = ['', '']  # Empty values for source and target

        # fill source language words (first row)
        source_words = group[group['lang'] == source_lang]
        styles_dict = {}
        for ts in timestamps:
            source_row = source_words[source_words['time_str'] == ts]
            if not source_row.empty:
                data_dict[ts][0] = source_row.iloc[0]['edit_word']
                if not pd.isna(source_row.iloc[0]['pair_type']) and source_row.iloc[0]['pair_type'] == 'S':
                    styles_dict[f'{ts}_0'] = GREEN
                elif not pd.isna(source_row.iloc[0]['annotate']) and source_row.iloc[0]['annotate'] == 'T':
                    styles_dict[f'{ts}_0'] = YELLOW

        # fill target language words (second row)
        target_words = group[group['lang'] == target_lang]
        for ts in timestamps:
            target_row = target_words[target_words['time_str'] == ts]
            if not target_row.empty:
                data_dict[ts][1] = target_row.iloc[0]['edit_word']
                if not pd.isna(target_row.iloc[0]['pair_type']) and target_row.iloc[0]['pair_type'] == 'E':
                    styles_dict[f'{ts}_1'] = PINK
                elif not pd.isna(target_row.iloc[0]['annotate']) and target_row.iloc[0]['annotate'] == 'T':
                    styles_dict[f'{ts}_1'] = YELLOW

        # create DataFrame
        df_duration = pd.DataFrame(data_dict)

        # style for edit mode
        def make_style(df):
            styles = pd.DataFrame('', index=df.index, columns=df.columns)
            for ts in timestamps:
                # source row style
                if f'{ts}_0' in styles_dict:
                    styles.loc[0, ts] = styles_dict[f'{ts}_0']
                # target row style
                if f'{ts}_1' in styles_dict:
                    styles.loc[1, ts] = styles_dict[f'{ts}_1']
            return styles

        styled_df = df_duration.style.apply(make_style, axis=None)

        # use data editor to edit the styled dataframe
        edited_df = st.data_editor(
            styled_df,
            key=f"edit_table_{interpret_file}_{model}_{time_group_key}",
            disabled=['lang'],
            hide_index=True,
            column_config={
                "lang": st.column_config.TextColumn(
                    "Language",
                    width="small",
                ),
                **{
                    ts: st.column_config.TextColumn(
                        ts,
                        width="small",
                    ) for ts in timestamps
                }
            }
        )

        # check changes
        if not edited_df.equals(df_duration):
            for ts in timestamps:
                # check source language changes (English)
                if edited_df.iloc[0][ts] != df_duration.iloc[0][ts] and edited_df.iloc[0][ts] != '':
                    source_data = group[(group['lang'] == source_lang) & (group['time_str'] == ts)]
                    if not source_data.empty:
                        source_data = source_data.iloc[0]
                        if 'changes' not in st.session_state:
                            st.session_state.changes = []
                        # Use correct provider based on language
                        source_provider = en_provider if source_lang == 'en' else zh_provider
                        st.session_state.changes.append({
                            'timestamp': ts,
                            'lang': source_lang,
                            'segment_id': source_data['segment_id'],
                            'word_seq_no': source_data['word_seq_no'],
                            'new_value': edited_df.iloc[0][ts],
                            'original_value': df_duration.iloc[0][ts],
                            'time_group': time_group_key,
                            'asr_provider': source_provider
                        })

                # check target language changes (Chinese)
                if edited_df.iloc[1][ts] != df_duration.iloc[1][ts] and edited_df.iloc[1][ts] != '':
                    target_data = group[(group['lang'] == target_lang) & (group['time_str'] == ts)]
                    if not target_data.empty:
                        target_data = target_data.iloc[0]
                        if 'changes' not in st.session_state:
                            st.session_state.changes = []
                        # Use correct provider based on language
                        target_provider = zh_provider if target_lang == 'zh' else en_provider
                        st.session_state.changes.append({
                            'timestamp': ts,
                            'lang': target_lang,
                            'segment_id': target_data['segment_id'],
                            'word_seq_no': target_data['word_seq_no'],
                            'new_value': edited_df.iloc[1][ts],
                            'original_value': df_duration.iloc[1][ts],
                            'time_group': time_group_key,
                            'asr_provider': target_provider
                        })

    # Clear progress bar and mark render complete
    if is_initial_render and loading_progress:
        loading_progress.empty()
        loading_text.empty()
        st.session_state[edit_render_key] = True


def save_pair_button_handler(interpret_file: str, asr_provider: str = None,
                             source_provider: str = None, target_provider: str = None):
    """Save selected word pairs to database.

    Uses segment_id and word_seq_no for reliable matching.
    Supports different providers for source and target languages.

    Args:
        interpret_file: The interpretation file name
        asr_provider: Default ASR provider (used as fallback)
        source_provider: ASR provider for source language (English)
        target_provider: ASR provider for target language (Chinese)
    """
    # Get the current file
    if not interpret_file:
        st.error("No file selected. Please select a file first.")
        return False

    # Check if we have selections
    if not hasattr(st.session_state, 'en_selections') or not hasattr(st.session_state, 'zh_selections'):
        st.warning("No selections found. Please select words first.")
        return False

    en_selections = list(st.session_state.en_selections.values())
    zh_selections = list(st.session_state.zh_selections.values())

    # Verify we have selections
    if not en_selections or not zh_selections:
        st.warning("Please select both English and Chinese words.")
        return False

    # Sort by time
    en_selections = sorted(en_selections, key=lambda x: float(x['time']))
    zh_selections = sorted(zh_selections, key=lambda x: float(x['time']))

    # Create pairs using segment_id and word_seq_no
    pair_count = min(len(en_selections), len(zh_selections))
    pairs = []

    for i in range(pair_count):
        en = en_selections[i]
        zh = zh_selections[i]

        pairs.append({
            'source_segment_id': en.get('segment_id'),
            'source_word_seq_no': en.get('word_seq_no'),
            'target_segment_id': zh.get('segment_id'),
            'target_word_seq_no': zh.get('word_seq_no'),
        })

    # Display what will be saved
    st.write(f"Saving {len(pairs)} pairs...")
    for i, (en, zh) in enumerate(zip(en_selections[:pair_count], zh_selections[:pair_count])):
        evs = float(zh['time']) - float(en['time'])
        st.write(f"Pair #{i+1}: EN='{en['word']}' ({en['time']:.3f}s) -> ZH='{zh['word']}' ({zh['time']:.3f}s) EVS={evs:.3f}s")

    # Save to database using segment_id and word_seq_no
    # Use provided providers or fall back to default
    src_provider = source_provider or asr_provider
    tgt_provider = target_provider or asr_provider

    success = EVSDataUtils.update_evs_by_ids(
        file_name=interpret_file,
        asr_provider=asr_provider or src_provider,
        pairs=pairs,
        source_provider=src_provider,
        target_provider=tgt_provider
    )

    if success:
        st.success(f"Successfully saved {len(pairs)} pairs!")
        # Keep selections after save - don't clear them
        # This maintains the selected state in the UI
        return True
    else:
        st.error("Failed to save pairs to database")
        return False


def _llm_pairs_to_selections(llm_pairs: list, en_data, zh_data) -> tuple:
    """Convert LLM pair results to en_selections/zh_selections format.

    Bridges the gap: LLM pairs have segment_id + start_time but not word_seq_no.
    Looks up word_seq_no from the DataFrames.
    """
    en_selections = {}
    zh_selections = {}

    for pair in llm_pairs:
        # Look up EN word_seq_no
        en_match = en_data[
            (en_data['segment_id'] == pair['en_segment']) &
            (abs(en_data['start_time'] - pair['en_time']) < 0.05)
        ]
        if en_match.empty:
            en_match = en_data[
                (en_data['segment_id'] == pair['en_segment']) &
                (en_data['edit_word'] == pair['en_word'])
            ]
        if en_match.empty:
            continue
        en_row = en_match.iloc[0]
        en_seg = int(en_row['segment_id'])
        en_wsn = int(en_row['word_seq_no'])
        en_key = f"en_{en_seg}_{en_wsn}"
        en_selections[en_key] = {
            'time': float(en_row['start_time']),
            'word': str(en_row['edit_word']),
            'segment_id': en_seg,
            'word_seq_no': en_wsn,
        }

        # Look up ZH word_seq_no
        zh_match = zh_data[
            (zh_data['segment_id'] == pair['zh_segment']) &
            (abs(zh_data['start_time'] - pair['zh_time']) < 0.05)
        ]
        if zh_match.empty:
            zh_match = zh_data[
                (zh_data['segment_id'] == pair['zh_segment']) &
                (zh_data['edit_word'] == pair['zh_word'])
            ]
        if zh_match.empty:
            continue
        zh_row = zh_match.iloc[0]
        zh_seg = int(zh_row['segment_id'])
        zh_wsn = int(zh_row['word_seq_no'])
        zh_key = f"zh_{zh_seg}_{zh_wsn}"
        zh_selections[zh_key] = {
            'time': float(zh_row['start_time']),
            'word': str(zh_row['edit_word']),
            'segment_id': zh_seg,
            'word_seq_no': zh_wsn,
        }

    return en_selections, zh_selections


def process_selected_word(word, ts, selected_list, progress_placeholder, lang_name):
    # Helper function to process a selected word and add it to the appropriate list
    if pd.notna(word) and str(word).strip():
        # Convert timestamp to seconds if it's in MM:SS.ms format
        if ':' in ts:
            try:
                minutes, seconds = ts.split(':')
                seconds_value = float(minutes) * 60 + float(seconds)
                ts_seconds = f"{seconds_value:.3f}"
            except ValueError:
                # If conversion fails, use the original timestamp
                ts_seconds = ts
        else:
            # Already in seconds format
            ts_seconds = ts

        selected_list.append({
            'time': ts_seconds,
            'word': str(word).strip()
        })
        progress_placeholder.success(f"✓ {lang_name} selected: {ts} ({ts_seconds}s) - {word}")


def create_word_pairs(selected_en, selected_zh, progress_placeholder):
    # Helper function to create pairs from selected English and Chinese words
    selected_pairs = {}
    pair_seq = 1

    if not selected_en or not selected_zh:
        progress_placeholder.warning("Cannot create pairs: Missing either English or Chinese selections")
        return {}

    # Sort selections by time
    sorted_en = sorted(selected_en, key=lambda x: float(x['time']))
    sorted_zh = sorted(selected_zh, key=lambda x: float(x['time']))

    progress_placeholder.write(f"Sorting: {len(sorted_en)} English words, {len(sorted_zh)} Chinese words")

    # Create pairs - use the minimum number from either list
    min_len = min(len(sorted_en), len(sorted_zh))

    for i in range(min_len):
        en = sorted_en[i]
        zh = sorted_zh[i]

        try:
            # Calculate EVS (ear-voice span)
            en_time = float(en['time'])
            zh_time = float(zh['time'])
            evs = zh_time - en_time  # Calculate EVS

            # Create the pair
            selected_pairs[pair_seq] = {
                'en_time': en['time'],
                'en': en['word'],
                'zh_time': zh['time'],
                'zh': zh['word'],
                'evs': evs
            }

            progress_placeholder.write(f"Created pair #{pair_seq}: EN={en['time']}s '{en['word']}', ZH={zh['time']}s '{zh['word']}', EVS={evs:.3f}s")
            pair_seq += 1

        except Exception as e:
            progress_placeholder.error(f"Error creating pair: {str(e)}, en={en}, zh={zh}")

    progress_placeholder.success(f"Created {len(selected_pairs)} pairs")
    return selected_pairs


def get_evs_data(interpret_file: str, slice_duration: int, en_provider: str, zh_provider: str, model: str = None):
    """Get EVS data from database with per-language provider support.

    Args:
        interpret_file: The interpretation file name
        slice_duration: Duration of audio slices in seconds
        en_provider: ASR provider for English
        zh_provider: ASR provider for Chinese
        model: Model name (optional, used for caching key)
    """
    # Generate cache key that includes all relevant parameters
    cache_key = f"table_data_{interpret_file}_{en_provider}_{zh_provider}_{model}"

    # Load data if needed or if cache key changed
    if cache_key not in st.session_state:
        # Get data from database with per-language provider
        data = EVSDataUtils.get_asr_data(
            interpret_file,
            en_provider=en_provider,
            zh_provider=zh_provider
        )

        if data.empty:
            st.warning(f"No data found for file {get_file_display_name(interpret_file)}")
            return pd.DataFrame()

        df = data.copy()

        # Process the data
        split_data = df['combined_word'].str.split('&&', expand=True)
        required_columns = 4
        actual_columns = split_data.shape[1]

        # Add missing columns if needed
        if actual_columns < required_columns:
            for i in range(actual_columns, required_columns):
                split_data[i] = None

        # Assign the parsed columns
        df['edit_word'] = split_data[0]
        df['pair_type'] = split_data[1]
        df['original_word'] = split_data[2]
        df['annotate'] = split_data[3]

        # Format times - use short format (e.g., "1.5" instead of "00:01.50")
        df['timestamp'] = pd.to_datetime(df['start_time'], unit='s')
        # Short format: total seconds with 1 decimal (e.g., 61.5 for 1:01.5)
        df['time_str'] = df['start_time'].apply(lambda x: f"{x:.1f}")
        df['time_group'] = df['timestamp'].dt.floor(f'{slice_duration}s')
        st.session_state[cache_key] = df
        st.session_state.table_data = df

    df = st.session_state[cache_key]

    # Load existing EVS pairs from database (simple cache like evs_client_simple)
    if 'existing_evs_pairs' not in st.session_state:
        st.session_state.existing_evs_pairs = EVSDataUtils.get_asr_pair_evs(interpret_file)

    # Initialize or update selections based on database pairs
    if 'en_selections' not in st.session_state:
        st.session_state.en_selections = {}

    if 'zh_selections' not in st.session_state:
        st.session_state.zh_selections = {}

    # Pre-select checkboxes based on existing pairs in the database
    existing_pairs = st.session_state.existing_evs_pairs

    # Create dictionaries to quickly look up which words are in pairs
    en_paired_words = {}
    zh_paired_words = {}

    if not existing_pairs.empty:
        for _, row in existing_pairs.iterrows():
            # Source (English) words in pairs
            en_segment_id = row['en_segment_id']
            en_word_seq_no = row['en_word_seq_no']
            en_key = f"en_{en_segment_id}_{en_word_seq_no}"

            # Add to paired words dictionary
            en_paired_words[en_key] = {
                'time': row['en_start_time'],
                'word': row['en_edit_word'],
                'segment_id': en_segment_id,
                'word_seq_no': en_word_seq_no
            }

            # Target (Chinese) words in pairs
            zh_segment_id = row['zh_segment_id']
            zh_word_seq_no = row['zh_word_seq_no']
            zh_key = f"zh_{zh_segment_id}_{zh_word_seq_no}"

            # Add to paired words dictionary
            zh_paired_words[zh_key] = {
                'time': row['zh_start_time'],
                'word': row['zh_edit_word'],
                'segment_id': zh_segment_id,
                'word_seq_no': zh_word_seq_no
            }

    # Initialize session state with paired words from database
    # This ensures checkboxes are pre-selected based on saved pairs
    if not hasattr(st.session_state, 'selections_initialized') or not st.session_state.selections_initialized:
        st.session_state.en_selections = en_paired_words.copy()
        st.session_state.zh_selections = zh_paired_words.copy()
        st.session_state.selections_initialized = True

    return df


def render_annotate_evs_tab():
    """Render the EVS annotate tab with improved functionality.

    Features:
    - Auto-detection of ASR providers per language (EN/ZH)
    - Model pair selection UI
    - Compact button-based EVS annotation view
    - Display settings persistence
    - Loading overlay to prevent double-clicks
    """
    # Fixed source/target: EN (Source) -> ZH (Target)
    source_lang = 'en'
    target_lang = 'zh'

    # Generate session ID for tracking
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]

    # Loading overlay CSS/JS
    st.markdown("""
    <style>
        .loading-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5);
            z-index: 9999;
            justify-content: center;
            align-items: center;
        }
        .loading-overlay.active {
            display: flex;
        }
        .loading-spinner {
            width: 50px;
            height: 50px;
            border: 5px solid #f3f3f3;
            border-top: 5px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
    <div id="loading-overlay" class="loading-overlay">
        <div class="loading-spinner"></div>
    </div>
    """, unsafe_allow_html=True)

    # Show loading overlay when processing (controlled by session state)
    if st.session_state.get('evs_processing', False):
        st.markdown("""
        <script>
            document.getElementById('loading-overlay').classList.add('active');
        </script>
        """, unsafe_allow_html=True)
        st.session_state.evs_processing = False

    # Get all files with their transcription info (refresh each render)
    files_df = EVSDataUtils.get_all_files_with_transcriptions()

    if files_df.empty:
        st.warning("No transcription files found. Please upload and transcribe audio files first.")
        return

    # Row 1: File selection, model pair, buttons
    col1, col2, col_auto, col5 = st.columns([1, 2, 1, 0.3])

    with col1:
        # File selection dropdown - no default selection
        file_options = [""] + files_df['file_name'].tolist()
        interpret_file = st.selectbox(
            "Select File",
            options=file_options,
            index=0,
            format_func=lambda x: "-- Select a file --" if x == "" else get_file_display_name(x),
            key='evs_analysis_file'
        )
        # Convert empty string to None for easier handling
        if interpret_file == "":
            interpret_file = None

    # Get available model pairs for selected file (cached per file, use refresh button to clear)
    evs_model_pairs = []
    evs_selected_pair = None
    evs_file_info = None

    with col2:
        if interpret_file:
            # Cache model pairs per file (queries take ~230ms)
            model_pairs_key = f"evs_model_pairs_{interpret_file}"
            if model_pairs_key not in st.session_state:
                st.session_state[model_pairs_key] = EVSDataUtils.get_file_model_pairs(interpret_file)
            evs_model_pairs = st.session_state[model_pairs_key]

            if evs_model_pairs:
                # Always show dropdown for model selection (even if only 1 pair)
                pair_options = [p['display_name'] for p in evs_model_pairs]
                selected_display = st.selectbox(
                    "Select Model Pair",
                    options=pair_options,
                    index=0,
                    key='evs_model_pair_select'
                )
                evs_selected_pair = next((p for p in evs_model_pairs if p['display_name'] == selected_display), evs_model_pairs[0])
            else:
                st.warning("No model pairs found. Click 🔄 to refresh.")

    # Convert to file_info dict for compatibility
    if evs_selected_pair:
        evs_file_info = {
            'en_provider': evs_selected_pair['en_provider'],
            'en_model': evs_selected_pair['en_model'],
            'zh_provider': evs_selected_pair['zh_provider'],
            'zh_model': evs_selected_pair['zh_model'],
            'slice_duration': evs_selected_pair['slice_duration']
        }

    # Get providers from file info
    en_provider = evs_file_info['en_provider'] if evs_file_info else 'crisperwhisper'
    zh_provider = evs_file_info['zh_provider'] if evs_file_info else 'funasr'
    evs_model = evs_file_info['en_model'] if evs_file_info and evs_file_info['en_model'] else (evs_file_info['zh_model'] if evs_file_info else None)

    with col_auto:
        auto_align_clicked = st.button(
            "🤖 Auto Align",
            key="auto_align_button",
            help="Use LLM to automatically pair EN-ZH words. Results appear as selections for review.",
            use_container_width=True,
        )

    with col5:
        if st.button('🔄', key="refresh_evs_files", help="Refresh file list and model pairs"):
            # Clear model pairs cache for all files
            keys_to_delete = [k for k in st.session_state.keys() if k.startswith('evs_model_pairs_')]
            for k in keys_to_delete:
                del st.session_state[k]
            if 'existing_evs_pairs' in st.session_state:
                del st.session_state.existing_evs_pairs
            if 'table_data' in st.session_state:
                del st.session_state.table_data
            st.session_state.selections_initialized = False
            st.rerun()

    # Handle Auto Align button click
    if auto_align_clicked and interpret_file:
        with st.spinner("LLM is analyzing word pairs..."):
            try:
                asr_data = EVSDataUtils.get_asr_data(
                    interpret_file,
                    en_provider=en_provider,
                    zh_provider=zh_provider,
                )
                en_data = asr_data[asr_data['lang'] == 'en']
                zh_data = asr_data[asr_data['lang'] == 'zh']

                if en_data.empty or zh_data.empty:
                    st.error("Need both EN and ZH word data for auto alignment.")
                else:
                    llm_pairs = create_llm_based_pairs(en_data, zh_data)
                    if llm_pairs:
                        en_sel, zh_sel = _llm_pairs_to_selections(llm_pairs, en_data, zh_data)
                        st.session_state.en_selections = en_sel
                        st.session_state.zh_selections = zh_sel
                        st.session_state.selections_initialized = True
                        st.success(f"LLM found {len(en_sel)} pairs. Review below, then click **Save Pairs**.")
                        st.rerun()
                    else:
                        st.warning("LLM could not find valid pairs. Try manual annotation.")
            except Exception as e:
                st.error(f"Auto alignment failed: {e}")
                logger.exception("Auto align failed")

    if not interpret_file:
        st.warning("Please select a file")
        return

    # Store file info in session state
    st.session_state.evs_file_info = {
        'en_provider': en_provider,
        'zh_provider': zh_provider,
        'model': evs_model,
        'slice_duration': evs_file_info['slice_duration'] if evs_file_info else 30
    }

    # Track file/model changes to clear cached data
    current_evs_file_key = f"{interpret_file}_{evs_model}"
    if 'current_evs_file_key' not in st.session_state:
        st.session_state.current_evs_file_key = current_evs_file_key
    elif st.session_state.current_evs_file_key != current_evs_file_key:
        st.session_state.current_evs_file_key = current_evs_file_key
        st.session_state.en_selections = {}
        st.session_state.zh_selections = {}
        st.session_state.selections_initialized = False
        # Clear cached data when file changes (match evs_client_simple)
        if 'table_data' in st.session_state:
            del st.session_state.table_data
        if 'existing_evs_pairs' in st.session_state:
            del st.session_state.existing_evs_pairs

    # Add table loading instruction
    st.info("Select English and Chinese words in the tables below, then click 'Save Pairs' to create pairs")

    # Initialize selected_pairs in session state if not present
    if 'selected_pairs' not in st.session_state:
        st.session_state.selected_pairs = {}

    slice_duration = evs_file_info['slice_duration'] if evs_file_info else 30
    en_model = evs_file_info['en_model'] if evs_file_info else None
    zh_model = evs_file_info['zh_model'] if evs_file_info else None
    display_evs_annotate_table(interpret_file, slice_duration, en_provider, zh_provider, source_lang, target_lang, evs_model, en_model, zh_model)

def display_evs_annotate_table(interpret_file: str, slice_duration: int, en_provider: str, zh_provider: str, source_lang: str = 'en', target_lang: str = 'zh', model: str = None, en_model: str = None, zh_model: str = None):
    """Display EVS annotate table with compact button-based annotation view.

    Always shows both languages (EN and ZH), with dynamic labels based on source/target selection.
    en_provider and zh_provider allow different ASR providers per language.
    """
    # Style selector and configuration - load from database
    if 'evs_display_config_loaded' not in st.session_state:
        try:
            db_config = EVSDataUtils.get_asr_config('evs_display_settings')
            if db_config and 'config' in db_config:
                saved = db_config['config']
                if isinstance(saved, str):
                    saved = json.loads(saved)
                st.session_state.evs_display_style = saved.get('style', "Compact (Buttons)")
                st.session_state.evs_seconds_per_row = saved.get('seconds_per_row', 25)
                st.session_state.evs_word_size = saved.get('word_size', 14)
            else:
                st.session_state.evs_display_style = "Compact (Buttons)"
                st.session_state.evs_seconds_per_row = 25
                st.session_state.evs_word_size = 14
        except Exception as e:
            logger.warning(f"Failed to load EVS display config: {str(e)}")
            st.session_state.evs_display_style = "Compact (Buttons)"
            st.session_state.evs_seconds_per_row = 25
            st.session_state.evs_word_size = 14
        st.session_state.evs_display_config_loaded = True

    config_col1, config_col2, save_col = st.columns([1, 1, 1])

    with config_col1:
        seconds_per_row = st.number_input(
            "Sec/Row",
            min_value=1,
            max_value=30,
            value=st.session_state.evs_seconds_per_row,
            step=1,
            key="evs_seconds_per_row_input"
        )
        st.session_state.evs_seconds_per_row = seconds_per_row

    with config_col2:
        word_size = st.number_input(
            "Font Size",
            min_value=10,
            max_value=24,
            value=st.session_state.evs_word_size,
            step=2,
            key="evs_word_size_input"
        )
        st.session_state.evs_word_size = word_size

    with save_col:
        st.write("")  # Spacer for alignment
        if st.button("Save", key="save_evs_display_config"):
            config_to_save = {
                'style': "Compact (Buttons)",
                'seconds_per_row': st.session_state.evs_seconds_per_row,
                'word_size': st.session_state.evs_word_size
            }
            if EVSDataUtils.save_asr_config('evs_display_settings', config_to_save):
                st.success("Saved!")
            else:
                st.error("Failed")

    # Check if this is the initial render (not a rerun from checkbox selection)
    table_render_key = f"table_rendered_{interpret_file}_{model}"
    is_initial_render = table_render_key not in st.session_state

    # Only show progress bar on initial render, not on selection reruns
    progress_bar = None
    progress_text = None
    if is_initial_render:
        progress_bar = st.progress(0)
        progress_text = st.empty()
        progress_text.text("Loading data from database...")

    # Load data (progress bar shown before this)
    df = get_evs_data(interpret_file, slice_duration, en_provider, zh_provider, model)

    if df.empty:
        if progress_bar:
            progress_bar.empty()
            progress_text.empty()
        st.warning("No data found for this file.")
        return

    if is_initial_render and progress_bar:
        progress_bar.progress(30)
        progress_text.text("Processing data...")

    # Color palette for EVS pairs - distinct colors for easy visual identification
    PAIR_COLORS = [
        '#FF6B6B',  # Red
        '#4ECDC4',  # Teal
        '#45B7D1',  # Sky Blue
        '#96CEB4',  # Sage Green
        '#FFEAA7',  # Yellow
        '#DDA0DD',  # Plum
        '#98D8C8',  # Mint
        '#F7DC6F',  # Gold
        '#BB8FCE',  # Purple
        '#85C1E9',  # Light Blue
        '#F8B500',  # Orange
        '#82E0AA',  # Light Green
        '#F1948A',  # Salmon
        '#AED6F1',  # Powder Blue
        '#D7BDE2',  # Lavender
    ]

    def get_pair_color(pair_seq):
        """Get color for a pair sequence number"""
        if pair_seq is None:
            return None
        return PAIR_COLORS[(pair_seq - 1) % len(PAIR_COLORS)]

    # Apply custom font size and compact spacing CSS (once, before loop)
    word_size = st.session_state.get('evs_word_size', 14)
    st.markdown(f"""
        <style>
            /* Font size for buttons */
            [data-testid="stVerticalBlock"] button p {{
                font-size: {word_size}px !important;
                line-height: 1.2 !important;
            }}
            /* Reduce column gaps */
            [data-testid="stHorizontalBlock"] {{
                gap: 0.2rem !important;
            }}
            /* Reduce button padding and margins */
            [data-testid="stVerticalBlock"] button {{
                padding: 0.15rem 0.3rem !important;
                margin: 0 !important;
                min-height: 0 !important;
            }}
            /* Reduce vertical spacing between elements */
            [data-testid="stVerticalBlock"] > div {{
                margin-bottom: 0 !important;
                padding-bottom: 0 !important;
            }}
            /* Compact caption/timestamp styling */
            [data-testid="stCaptionContainer"] {{
                margin: 0 !important;
                padding: 0 !important;
            }}
            [data-testid="stCaptionContainer"] p {{
                margin: 0 !important;
                padding: 0 !important;
                font-size: 11px !important;
                line-height: 1 !important;
            }}
            /* Remove gaps in column containers */
            [data-testid="stColumn"] > div {{
                gap: 0 !important;
            }}
            [data-testid="stColumn"] {{
                padding: 0 !important;
            }}
            /* Paired word styling */
            .paired-word {{
                padding: 4px 8px;
                border-radius: 4px;
                margin: 2px 0;
                display: inline-block;
                font-size: {word_size}px;
                cursor: pointer;
                border: 1px solid #ccc;
                text-align: center;
                width: 100%;
                box-sizing: border-box;
            }}
            .paired-word:hover {{
                opacity: 0.8;
            }}
            .pair-indicator {{
                font-size: 10px;
                font-weight: bold;
                margin-left: 4px;
            }}
        </style>
        """, unsafe_allow_html=True)

    # Cache groupby result to avoid computing twice
    grouped = list(df.groupby('time_group'))
    total_groups = len(grouped)

    # For Compact mode: collect all groups' word data, then render component once
    compact_groups_data = []

    for idx, (name, group) in enumerate(grouped):
        # Only update progress on initial render (from 30% to 100%)
        if progress_bar is not None:
            progress_pct = 30 + int((idx / total_groups) * 70)
            progress_bar.progress(progress_pct)
            progress_text.text(f"Rendering time group {idx + 1} of {total_groups}...")

        time_group_key = name.strftime('%M:%S')

        # Calculate group number
        minutes = int(time_group_key.split(':')[0])
        seconds = int(time_group_key.split(':')[1])
        group_number = int((minutes * 60 + seconds) / slice_duration)

        # Audio player - build path with provider/model subfolders
        file_stem = os.path.splitext(interpret_file)[0]
        model_folder = f"{en_model}_{zh_model}" if en_model and zh_model else None
        if model_folder:
            audio_path = os.path.join(
                SLICE_AUDIO_PATH, file_stem, en_provider, model_folder,
                f'{group_number}.mp3'
            )
        else:
            audio_path = os.path.join(
                SLICE_AUDIO_PATH, file_stem,
                f'{group_number}.mp3'
            )

        # Get all unique timestamps for this group - sort using time_str_to_seconds for MM:SS.xx format
        all_timestamps = sorted(group['time_str'].unique(), key=time_str_to_seconds)

        # Build word data for this time group
        en_words = []
        zh_words = []
        en_metadata = {}
        zh_metadata = {}

        for ts in all_timestamps:
            # Process English words
            en_data = group[(group['lang'] == 'en') & (group['time_str'] == ts)]
            if not en_data.empty and pd.notna(en_data.iloc[0]['edit_word']) and str(en_data.iloc[0]['edit_word']).strip():
                # Get pair info if available
                pair_type = en_data.iloc[0].get('pair_type', None)
                pair_seq = en_data.iloc[0].get('pair_seq', None) if 'pair_seq' in en_data.columns else None
                word_info = {
                    'ts': ts,
                    'word': str(en_data.iloc[0]['edit_word']),
                    'segment_id': int(en_data.iloc[0]['segment_id']),
                    'word_seq_no': int(en_data.iloc[0]['word_seq_no']),
                    'start_time': float(en_data.iloc[0]['start_time']),
                    'pair_type': pair_type if pd.notna(pair_type) else None,
                    'pair_seq': int(pair_seq) if pd.notna(pair_seq) else None
                }
                en_words.append(word_info)
                en_metadata[ts] = word_info

            # Process Chinese words
            zh_data = group[(group['lang'] == 'zh') & (group['time_str'] == ts)]
            if not zh_data.empty and pd.notna(zh_data.iloc[0]['edit_word']) and str(zh_data.iloc[0]['edit_word']).strip():
                # Get pair info if available
                pair_type = zh_data.iloc[0].get('pair_type', None)
                pair_seq = zh_data.iloc[0].get('pair_seq', None) if 'pair_seq' in zh_data.columns else None
                word_info = {
                    'ts': ts,
                    'word': str(zh_data.iloc[0]['edit_word']),
                    'segment_id': int(zh_data.iloc[0]['segment_id']),
                    'word_seq_no': int(zh_data.iloc[0]['word_seq_no']),
                    'start_time': float(zh_data.iloc[0]['start_time']),
                    'pair_type': pair_type if pd.notna(pair_type) else None,
                    'pair_seq': int(pair_seq) if pd.notna(pair_seq) else None
                }
                zh_words.append(word_info)
                zh_metadata[ts] = word_info

        # Collect word data for custom component
        combined_words = []
        for ts in all_timestamps:
            en_info = en_metadata.get(ts)
            zh_info = zh_metadata.get(ts)
            if en_info or zh_info:
                combined_words.append({
                    'ts': ts,
                    'ts_float': float(ts),
                    'en': en_info,
                    'zh': zh_info
                })
        # Encode audio as base64 data URI for the component
        audio_data_uri = None
        if os.path.exists(audio_path):
            try:
                with open(audio_path, 'rb') as af:
                    audio_b64 = base64.b64encode(af.read()).decode('ascii')
                audio_data_uri = f"data:audio/mpeg;base64,{audio_b64}"
            except Exception:
                pass
        compact_groups_data.append({
            'time_group_key': time_group_key,
            'words': combined_words,
            'audio_src': audio_data_uri,
        })

    # === Render custom component (after data collection) ===
    if compact_groups_data:
        seconds_per_row = st.session_state.get('evs_seconds_per_row', 25)
        word_size = st.session_state.get('evs_word_size', 14)

        result = evs_annotator(
            groups=compact_groups_data,
            seconds_per_row=seconds_per_row,
            font_size=word_size,
            en_selections=dict(st.session_state.get('en_selections', {})),
            zh_selections=dict(st.session_state.get('zh_selections', {})),
            key=f"evs_annotator_{interpret_file}_{model}",
        )

        # Handle actions from component
        if result and result.get('action') == 'save_pairs':
            st.session_state.en_selections = result.get('en_selections', {})
            st.session_state.zh_selections = result.get('zh_selections', {})
            success = save_pair_button_handler(
                interpret_file,
                asr_provider=en_provider,
                source_provider=en_provider,
                target_provider=zh_provider
            )
            if success:
                st.success("Pairs saved!")
                if 'existing_evs_pairs' in st.session_state:
                    del st.session_state.existing_evs_pairs
                if 'table_data' in st.session_state:
                    del st.session_state.table_data
                st.session_state.selections_initialized = False
                st.session_state.en_selections = {}
                st.session_state.zh_selections = {}
                time.sleep(0.5)
                st.session_state.evs_processing = True
                st.rerun()

        if result and result.get('action') == 'clear_all':
            if EVSDataUtils.reset_evs(interpret_file):
                st.success("All pairs deleted from database.")
                if 'existing_evs_pairs' in st.session_state:
                    del st.session_state.existing_evs_pairs
                if 'table_data' in st.session_state:
                    del st.session_state.table_data
                st.session_state.selections_initialized = False
                st.session_state.en_selections = {}
                st.session_state.zh_selections = {}
                time.sleep(0.5)
                st.session_state.evs_processing = True
                st.rerun()


    # Clean up progress bar only if it was shown
    if progress_bar is not None:
        progress_bar.empty()
        progress_text.empty()

    # Mark table as rendered to skip progress bar on subsequent reruns
    st.session_state[table_render_key] = True

def update_pairs_from_selections():
    # Update pairs based on current selections
    selected_en = []
    selected_zh = []

    # Collect selections from all tables
    for key in st.session_state:
        if key.startswith('editor_table_'):
            # check if it is a dataframe
            if not isinstance(st.session_state.get(key), pd.DataFrame):
                logging.warning(f"session {st.session_state.get('session_id', 'unknown')}: table {key} is not a dataframe")
                continue

            df = st.session_state[key]
            for col in df.columns:
                if col.endswith('_select'):
                    ts = col[:-7]  # Get timestamp from column name
                    if df.iloc[0][col]:  # English selection
                        selected_en.append({
                            'time': ts,
                            'word': df.iloc[0][f"{ts}_word"]
                        })
                    if df.iloc[1][col]:  # Chinese selection
                        selected_zh.append({
                            'time': ts,
                            'word': df.iloc[1][f"{ts}_word"]
                        })

    # Create pairs based on selected words
    selected_pairs = {}
    pair_seq = 1

    # Create pairs matching the pattern in the existing pairs table
    if len(selected_en) == len(selected_zh):
        # Sort selections by time
        sorted_en = sorted(selected_en, key=lambda x: x['time'])
        sorted_zh = sorted(selected_zh, key=lambda x: x['time'])

        # Create pairs in order
        for en, zh in zip(sorted_en, sorted_zh):
            evs = calculate_evs(en['time'], zh['time'])
            selected_pairs[pair_seq] = {
                'en_time': en['time'],
                'en': en['word'],
                'zh_time': zh['time'],
                'zh': zh['word'],
                'evs': evs
            }
            pair_seq += 1

    # Update session state
    st.session_state.selected_pairs = selected_pairs

    """Load pairs from current selections in the table and refresh EVS pairs display"""
    # Display pairs using existing pairs from session state
    if st.session_state.selected_pairs:
        pairs_df = pd.DataFrame([
            {
                'Pair': pair_seq,
                'en_time': pair_data['en_time'],
                'en': pair_data['en'],
                'zh_time': pair_data['zh_time'],
                'zh': pair_data['zh'],
                'evs': pair_data['evs']
            }
            for pair_seq, pair_data in st.session_state.selected_pairs.items()
        ]).sort_values('Pair')

        # add title
        st.subheader("Existing & New Pairs")
        st.dataframe(
            pairs_df,
            hide_index=True,
            column_config={
                'Pair': st.column_config.NumberColumn('Pair'),
                'en_time': st.column_config.TextColumn('en_time', width='small'),
                'en': st.column_config.TextColumn('en', width='small'),
                'zh_time': st.column_config.TextColumn('zh_time', width='small'),
                'zh': st.column_config.TextColumn('zh', width='small'),
                'EVS': st.column_config.NumberColumn('evs', width='small', format="%.3f")
            }
        )

def time_str_to_seconds(time_str):
    """Convert time string to seconds. Handles multiple formats:
    - "MM:SS.mmm" (e.g., "5:13.200")
    - "SS.mmm" (e.g., "13.200")
    - Plain seconds as string (e.g., "513.2")
    """
    if time_str is None:
        return 0

    time_str = str(time_str).strip().lstrip(':')

    try:
        # If contains ":", parse as MM:SS.mmm format
        if ':' in time_str:
            minutes, rest = time_str.split(':')
            if '.' in rest:
                seconds, milliseconds = rest.split('.')
                return (int(minutes) * 60) + float(seconds) + (float(milliseconds) / 1000)
            else:
                return (int(minutes) * 60) + float(rest)
        else:
            # Already in seconds (e.g., "513.2")
            return float(time_str)
    except (ValueError, AttributeError) as e:
        logger.debug(f"Time string parsing failed for '{time_str}': {e}")
        return 0

def calculate_evs(start_time, end_time):
    """Calculate EVS value"""
    try:
        # ensure two times are floats
        start = float(start_time)
        end = float(end_time)
        return end - start
    except (ValueError, TypeError) as e:
        logging.error(f"Calcluate EVS error: {str(e)}, start_time={start_time}, end_time={end_time}")
        return 0.0

def slice_audio(input_file, duration):
    audio = AudioSegment.from_file(input_file)
    slice_duation = duration * 1000  # in milliseconds

    output_prefix = os.path.splitext(input_file.name)[0]

    container = st.empty()

    for i in range(0, len(audio), slice_duation):
        chunk = audio[i:i + slice_duation]
        output_path = f"{EVS_RESOURCES_PATH}/slice_audio_files/{output_prefix}"
        if not os.path.exists(output_path):
            container.write(f"Creating directory {output_path}")
            os.makedirs(output_path)
        container.empty()
        container.write(f"Exporting chunk {i//slice_duation} to {output_path}/{i//slice_duation}.mp3")
        chunk.export(f"{output_path}/{i//slice_duation}.mp3", format="mp3")

    container.empty()
    container.write("Audio sliced successfully")

    return output_path

def render_analysis_concordance_tab():
    # Render the corpus analysis tab with concordance and word list subtabs

    # Get all files with their transcription info (supports different ASR per language)
    concordance_files_df = EVSDataUtils.get_all_files_with_transcriptions()

    # Early return if no files available
    if concordance_files_df.empty:
        st.warning("No transcription files found. Please upload and transcribe audio files first.")
        return

    col1, col2 = st.columns([1, 8])
    with col1:
        # Step 1: File selection (with empty option for "select a file")
        file_options = [""] + concordance_files_df['file_name'].tolist()
        concordance_file = st.selectbox(
            "File",
            options=file_options,
            index=0,
            format_func=lambda x: "-- Select a file --" if x == "" else get_file_display_name(x),
            key='concordance_file'
        )
        # Convert empty string to None
        if concordance_file == "":
            concordance_file = None

        # Step 2: Model pair selection (only shown after file is selected)
        concordance_selected_pair = None
        concordance_en_provider = None
        concordance_zh_provider = None
        concordance_model_pairs = []

        if concordance_file:
            concordance_model_pairs = EVSDataUtils.get_file_model_pairs(concordance_file)
            if concordance_model_pairs:
                if len(concordance_model_pairs) == 1:
                    concordance_selected_pair = concordance_model_pairs[0]
                    st.caption(f"Model: {concordance_selected_pair['display_name']}")
                else:
                    pair_options = [p['display_name'] for p in concordance_model_pairs]
                    selected_display = st.selectbox(
                        "Model Pair",
                        options=pair_options,
                        index=0,
                        key='concordance_model_pair'
                    )
                    concordance_selected_pair = next((p for p in concordance_model_pairs if p['display_name'] == selected_display), concordance_model_pairs[0])

                # Set providers from selected pair
                if concordance_selected_pair:
                    concordance_en_provider = concordance_selected_pair['en_provider']
                    concordance_zh_provider = concordance_selected_pair['zh_provider']
            else:
                st.warning("No model pairs found for this file.")

        # Step 3: Language selection
        st.write("Language")
        concordance_language = st.selectbox(
            "Language Selection",
            options=['All', 'en', 'zh'],
            index=0,
            key='concordance_language',
            label_visibility="collapsed"
        )

        # Determine which ASR provider to use based on language selection
        if concordance_selected_pair:
            if concordance_language == 'All':
                # For "All" language, pass 'All' so DB functions search across all providers
                concordance_asr_provider = 'All'
            elif concordance_language == 'en':
                concordance_asr_provider = concordance_en_provider
            else:  # zh
                concordance_asr_provider = concordance_zh_provider
        else:
            concordance_asr_provider = None

        concordance_search_term = st.text_input("Search Term", key="concordance_search_term")

    with col2:
        # Create subtabs
        concordance_tab, plot_tab, wordlist_tab, keyword_tab, collocates_tab, clusters_tab, file_view_tab = st.tabs([
            "Concordance",
            "Concordance Plot",
            "Word List",
            "Keyword List",
            "Collocation",
            "Word Clusters",
            "File View"
        ])

        # Concordance subtab
        with concordance_tab:
            st.subheader("Concordance")
            # Search Term field at top with Start button
            col1, col2 = st.columns([1, 4])
            with col1:
                context_size = st.number_input("Context Size", min_value=1, max_value=20, value=5, label_visibility="visible", key="concordance_context_size")

            # Trigger search on either Enter key or Start button
            if st.button("Start", type="primary", key="concordance_start"):
                if not concordance_search_term:
                    st.warning("Please enter a search term and click 'Start' to generate the plot.")
                    return

                with st.spinner("Searching..."):
                    try:
                        # Modified query to get full sentences for context
                        df = EVSDataUtils.search_interpreted_words(concordance_search_term, concordance_file, concordance_language, concordance_asr_provider)

                        if df.empty:
                            st.info("No matches found.")
                            return

                        # 使用ConcordanceUtils生成concordance行
                        concordance_df = ConcordanceUtils.generate_concordance_lines(df, concordance_search_term, context_size)

                        # Apply file name privacy protection to File column
                        if not concordance_df.empty and 'File' in concordance_df.columns:
                            concordance_df['File'] = concordance_df['File'].apply(get_file_display_name)

                        st.write(f"Found {len(concordance_df)} matches")

                        # Display the dataframe at the bottom
                        st.dataframe(
                            concordance_df,
                            column_config={
                                'File': st.column_config.TextColumn('File', width='medium'),
                                'Left Context': st.column_config.TextColumn('Left Context', width='large'),
                                'Hit': st.column_config.TextColumn('Hit', width='medium'),
                                'Right Context': st.column_config.TextColumn('Right Context', width='large'),
                                'Language': st.column_config.TextColumn('Lang', width='small'),
                                'Time': st.column_config.TextColumn('Time', width='medium')
                            },
                            hide_index=True,
                            width='stretch'
                        )

                    except Exception as e:
                        st.error(f"Error executing search: {str(e)}")
                        logger.error(f"Search error: {str(e)}", exc_info=True)
            else:
                st.info("Please enter a search term and click 'Start' to generate the plot.")

        with plot_tab:
            st.subheader("Concordance Plot")

            col1, col2 = st.columns([1, 4])
            with col1:
                context_size = st.number_input("Context Size", min_value=1, max_value=20, value=5, label_visibility="visible", key="plot_context_size")


            if st.button("Start", type="primary", key="plot_start"):
                if not concordance_search_term:
                    st.warning("Please enter a search term and click 'Start' to generate the plot.")
                    return

                with st.spinner("Generating plot..."):
                    try:
                        df = EVSDataUtils.get_search_words(concordance_search_term, concordance_file, concordance_language, concordance_asr_provider)

                        if df.empty:
                            st.info("No matches found.")
                            return

                        # use ConcordanceUtils to generate concordance plot data
                        df, results = ConcordanceUtils.generate_concordance_plot(df, concordance_search_term)

                        # create results dataframe
                        results_df = pd.DataFrame(results)

                        # Apply file name privacy protection to FileName column
                        if not results_df.empty and 'FileName' in results_df.columns:
                            results_df['FileName'] = results_df['FileName'].apply(get_file_display_name)

                        # Display summary
                        st.markdown(f"**Total Hits:** {df['Freq'].sum()}    **Total Files With Hits:** {len(df)}")

                        # Display results table
                        st.dataframe(
                            results_df,
                            column_config={
                                'Row': st.column_config.NumberColumn('Row', width=60),
                                'FileID': st.column_config.TextColumn('FileID', width=70),
                                'FileName': st.column_config.TextColumn('FileName', width=150),
                                'FileTokens': st.column_config.NumberColumn('FileTokens', width=100),
                                'Freq': st.column_config.NumberColumn('Freq', width=70),
                                'NormFreq': st.column_config.NumberColumn('NormFreq', format="%.3f", width=100),
                                'Plot': st.column_config.TextColumn('Plot', width=400, help="Hit distribution plot")
                            },
                            hide_index=True,
                            width='stretch'
                        )

                    except Exception as e:
                        st.error(f"Error generating plot: {str(e)}")
                        logger.error(f"Plot error: {str(e)}", exc_info=True)
            else:
                st.info("Please enter a search term and click 'Start' to generate the plot.")

        with wordlist_tab:
            st.subheader("Word List")

            if st.button("Generate Word List", type="primary", key='btn_wordlist'):
                st.session_state.session_state.wordlist_data = None

                try:
                    st.session_state.session_state.wordlist_data = EVSDataUtils.get_word_frequency(concordance_file, concordance_language, concordance_asr_provider)

                    if st.session_state.session_state.wordlist_data.empty:
                        st.info("No data found.")
                        return

                except Exception as e:
                    st.error(f"Error generating word list: {str(e)}")
                    logger.error(f"Word list error: {str(e)}", exc_info=True)

                # display word list if there is data
                if st.session_state.session_state.wordlist_data is not None and not st.session_state.session_state.wordlist_data.empty:
                    df = st.session_state.session_state.wordlist_data
                    # Add rank column
                    df['Rank'] = range(1, len(st.session_state.session_state.wordlist_data) + 1)

                    # Reorder columns
                    df = df[['Rank', 'edit_word', 'lang', 'frequency']]

                    # Display results
                    st.write(f"Total Types: {len(df)}")
                    st.write(f"Total Tokens: {df['frequency'].sum()}")

                    st.dataframe(
                        df,
                        column_config={
                            'Rank': st.column_config.NumberColumn('Rank', width='small'),
                            'edit_word': st.column_config.TextColumn('Word', width='medium'),
                            'lang': st.column_config.TextColumn('Lang', width='small'),
                            'frequency': st.column_config.NumberColumn('Freq', width='small')
                        },
                        hide_index=True
                    )
                else:
                    # Optionally, display a message if no data is available yet or after clicking the button
                    if 'wordlist_data' in st.session_state.session_state and st.session_state.session_state.wordlist_data is None:
                        st.info("Click 'Generate Word List' to see the results.")
                    elif 'wordlist_data' in st.session_state.session_state and st.session_state.session_state.wordlist_data is not None and st.session_state.session_state.wordlist_data.empty:
                        # This case is already handled by the check above, but added for clarity
                        st.info("No data found for the selected criteria.")
                    else:
                        # Initial state before button press
                        st.info("Click 'Generate Word List' to see the results.")

        with keyword_tab:
            st.subheader("Keyword List")

            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                target_file = create_file_selectbox_with_all(
                    concordance_files_df,
                    label="Target File",
                    key='target_file'
                )

            with col2:
                reference_file = create_file_selectbox_with_all(
                    concordance_files_df,
                    label="Reference File",
                    key='reference_file'
                )

            with col3:
                min_freq = st.number_input("Min. Frequency", min_value=1, value=1, key="keyword_min_freq")

            if st.button("Generate Keyword List", type="primary", key="keyword_gen_btn"):
                if target_file == reference_file:
                    st.warning("Target and reference files must be different")
                    return
                else:
                    try:
                        target_corpus, reference_corpus = EVSDataUtils.get_reference_and_target_corpus(target_file, reference_file, concordance_language, concordance_asr_provider)

                        if target_corpus.empty or reference_corpus.empty:
                            st.warning("No data found in one or both corpora")
                            return

                        # Calculate keyword list
                        keywords = ConcordanceUtils.calculate_keyword_list(
                            target_corpus,
                            reference_corpus,
                            min_freq=min_freq
                        )

                        # Display results
                        st.write(f"Target corpus total: {target_corpus['frequency'].sum():,} words")
                        st.write(f"Reference corpus total: {reference_corpus['frequency'].sum():,} words")

                        # Format results for display
                        display_df = keywords[[
                            'edit_word',
                            'lang',
                            'frequency_target',
                            'freq_pmw_target',
                            'frequency_ref',
                            'freq_pmw_ref',
                            'log_likelihood',
                            'keyness'
                        ]].copy()

                        display_df.columns = [
                            'Word',
                            'Lang',
                            'Freq (Target)',
                            'Freq/M (Target)',
                            'Freq (Ref)',
                            'Freq/M (Ref)',
                            'Keyness',
                            'Type'
                        ]

                        st.dataframe(
                            display_df,
                            column_config={
                                'Word': st.column_config.TextColumn('Word', width='medium'),
                                'Lang': st.column_config.TextColumn('Lang', width='small'),
                                'Freq (Target)': st.column_config.NumberColumn('Freq\n(Target)', width='small'),
                                'Freq/M (Target)': st.column_config.NumberColumn('Freq/M\n(Target)', format="%.2f", width='small'),
                                'Freq (Ref)': st.column_config.NumberColumn('Freq\n(Ref)', width='small'),
                                'Freq/M (Ref)': st.column_config.NumberColumn('Freq/M\n(Ref)', format="%.2f", width='small'),
                                'Keyness': st.column_config.NumberColumn('Keyness', format="%.2f", width='small'),
                                'Type': st.column_config.TextColumn('Type', width='small'),
                            },
                            hide_index=True
                        )

                    except Exception as e:
                        st.error(f"Error generating keyword list: {str(e)}")

        with collocates_tab:
            st.subheader("Collocation")

            col1, col2 = st.columns([1, 1])

            with col1:
                window_size = st.number_input("Window Size", min_value=1, max_value=10, value=5, key="collocates_window_size")
            with col2:
                min_freq = st.number_input("Min. Frequency", min_value=1, value=2, key="collocates_min_freq")

            start_search = st.button("Calculate", type="primary", key="collocates_start_btn")

            # Calculate collocation when search is triggered
            if start_search:

                if not concordance_search_term:
                    st.warning("Please enter a search term and click 'Start' to generate the plot.")
                    return

                try:
                    df = EVSDataUtils.get_lang_file_words(concordance_file, concordance_language, concordance_asr_provider)

                    if df.empty:
                        st.info("No data found.")
                        return

                            # Calculate collocation using ConcordanceUtils
                    collocates, raw_collocates = ConcordanceUtils.calculate_collocates(
                        df, concordance_search_term,
                        window_size=window_size,
                        min_freq=min_freq
                    )

                    if collocates.empty:
                        st.info("No collocations found.")
                        return

                    # Display results
                    st.dataframe(
                        collocates,
                        column_config={
                            'collocate': st.column_config.TextColumn('Collocation', width='medium'),
                            'frequency': st.column_config.NumberColumn('Freq', width='small'),
                            'mean_position': st.column_config.NumberColumn('Mean Pos', format="%.2f", width='small'),
                            'std_position': st.column_config.NumberColumn('SD Pos', format="%.2f", width='small'),
                            'n_files': st.column_config.NumberColumn('Files', width='small'),
                            'n_sentences': st.column_config.NumberColumn('Sentences', width='small')
                        },
                        hide_index=True
                    )

                    # Display visualization if enough data
                    if not raw_collocates.empty:
                        st.subheader("Position Distribution")

                        # Create frequency by position plot
                        pos_freq = raw_collocates.groupby('position')['collocate'].count().reset_index()
                        pos_freq.columns = ['Position', 'Frequency']

                        fig = go.Figure(data=[
                            go.Bar(
                                x=pos_freq['Position'],
                                y=pos_freq['Frequency'],
                                text=pos_freq['Frequency'],
                                textposition='auto',
                            )
                        ])

                        fig.update_layout(
                            title='Collocation Positions',
                            xaxis_title='Position Relative to Search Term',
                            yaxis_title='Frequency',
                            bargap=0.2,
                            showlegend=False
                        )

                        st.plotly_chart(fig, width='stretch')

                except Exception as e:
                    st.error(f"Error calculating collocation: {str(e)}")
                    logger.error(f"Collocation error: {str(e)}", exc_info=True)

            else:
                st.info("Please enter a search term and click 'Start' to generate the plot.")
        # clusters tab
        with clusters_tab:
            st.subheader("Word Clusters")

            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

            with col1:
                cluster_size = st.slider(
                    "Cluster Size (N-gram)",
                    min_value=2,
                    max_value=5,
                    value=2,
                    key="clusters_size_slider"
                )
            with col2:
                min_freq = st.number_input(
                    "Min. Frequency",
                    min_value=1,
                    value=2,
                    key="clusters_min_freq"
                )
            with col3:
                sort_by = st.selectbox(
                    "Sort by",
                    options=['Frequency', 'Length', 'Alphabetical'],
                    key='clusters_sort_by'
                )
            with col4: # show the top n options
                top_n = st.number_input("Top N", min_value=1, value=20)

            if st.button("Calculate", type="primary", key="clusters_calculate_btn"):
                with st.spinner("Cluster..."):
                    try:
                        df = EVSDataUtils.get_cluster_data(concordance_file, concordance_language, concordance_asr_provider)

                        if df.empty:
                            st.info("No data found.")
                            return

                            # Calculate clusters using ConcordanceUtils
                        clusters = ConcordanceUtils.calculate_clusters(
                            df,
                            cluster_size=cluster_size,
                            min_freq=min_freq
                        )

                        if clusters.empty:
                            st.info("No clusters found.")
                            return

                        # Sort results
                        if sort_by == 'Frequency':
                            clusters = clusters.sort_values('frequency', ascending=False)
                        elif sort_by == 'Length':
                            clusters['length'] = clusters['cluster'].astype(str).str.len()
                            clusters = clusters.sort_values(['length', 'frequency'], ascending=[False, False])
                            clusters = clusters.drop('length', axis=1)
                        else:  # Alphabetical
                            clusters = clusters.sort_values('cluster')

                        # Display summary statistics
                        total_clusters = len(clusters)
                        total_occurrences = clusters['frequency'].sum()

                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.metric("Total Unique Clusters", total_clusters)
                        with col2:
                            st.metric("Total Occurrences", total_occurrences)

                        # Display detailed results
                        st.dataframe(
                            clusters,
                            column_config={
                                'cluster': st.column_config.TextColumn('Cluster', width='large'),
                                'lang': st.column_config.TextColumn('Lang', width='small'),
                                'frequency': st.column_config.NumberColumn('Freq', width='small'),
                                'n_files': st.column_config.NumberColumn('Files', width='small'),
                                'n_sentences': st.column_config.NumberColumn('Sentences', width='small')
                            },
                            hide_index=True
                        )

                        # Display visualization if enough data
                        if len(clusters) > 0:
                            st.subheader("Top Clusters")

                            # Get top 20 clusters
                            top_clusters = clusters.head(top_n)

                            fig = go.Figure(data=[
                                go.Bar(
                                    x=top_clusters['frequency'],
                                    y=top_clusters['cluster'],
                                    orientation='h',
                                    text=top_clusters['frequency'],
                                    textposition='outside',
                                    textangle=0,  # Rotate text labels
                                    hovertemplate='<b>%{y}</b><br>' +
                                                'Frequency: %{x}<br>' +
                                                '<extra></extra>'
                                )
                            ])

                            fig.update_layout(
                                title='Most Frequent Clusters',
                                xaxis_title='Frequency',
                                yaxis_title='Cluster',
                                height=600,
                                yaxis={'categoryorder':'total ascending'},
                                # Add padding to ensure rotated labels are visible
                                margin=dict(r=100),
                                uniformtext=dict(mode='hide', minsize=8)
                            )

                            st.plotly_chart(fig, width='stretch')

                    except Exception as e:
                        st.error(f"Error calculating clusters: {str(e)}")
                        logger.error(f"Clusters error: {str(e)}", exc_info=True)

        # File view tab
        with file_view_tab:
            st.subheader("File View")
            if not concordance_file or concordance_file == 'All':
                st.warning("Please select a specific file to view content.")
                return

            if concordance_file and concordance_selected_pair:
                # Get content using the correct provider for each language
                en_provider = concordance_selected_pair.get('en_provider', 'crisperwhisper')
                zh_provider = concordance_selected_pair.get('zh_provider', 'funasr')

                # Fetch data for both languages using their respective providers
                df_en = EVSDataUtils.get_file_content(concordance_file, en_provider)
                df_zh = EVSDataUtils.get_file_content(concordance_file, zh_provider) if zh_provider != en_provider else pd.DataFrame()

                # Combine the dataframes
                if not df_zh.empty:
                    df_content = pd.concat([df_en, df_zh], ignore_index=True)
                else:
                    df_content = df_en

                if not df_content.empty:
                    # Create tabs for different views
                    view_tabs = st.tabs(['English', 'Chinese', 'Parallel'])

                    # Format text for each language
                    def format_text_by_language(df, lang):
                        lang_df = df[df['lang'] == lang].sort_values(['segment_id', 'word_seq_no'])
                        sentences = []
                        for _, sentence_group in lang_df.groupby('segment_id'):
                            words = sentence_group['edit_word'].fillna('').astype(str)  # Convert to string
                            if lang == 'en':
                                sentence = ' '.join(words)
                            else:
                                sentence = ''.join(words)
                            if sentence.strip():
                                sentences.append(sentence)
                        return '\n'.join(sentences)

                    en_text = format_text_by_language(df_content, 'en')
                    zh_text = format_text_by_language(df_content, 'zh')

                    with view_tabs[0]:  # English
                        st.text_area("English Text", en_text, height=500)
                        st.write(f"Total sentences: {len(en_text.splitlines())}")

                    with view_tabs[1]:  # Chinese
                        st.text_area("Chinese Text", zh_text, height=500)
                        st.write(f"Total sentences: {len(zh_text.splitlines())}")

                    with view_tabs[2]:  # Parallel View
                        col1, col2 = st.columns(2)
                        with col1:
                            st.text_area("English", en_text, height=500)
                        with col2:
                            st.text_area("Chinese", zh_text, height=500)
                else:
                    st.warning("No data found for this file")

def main():
    """Main application entry point"""

            # Add global CSS for audio player width control
    st.markdown("""
    <style>
    /* Simple audio player width control */
    audio {
        max-width: 400px;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize database manager (only once per session)
    if 'db_manager' not in st.session_state:
        try:
            from db_manager import DBManager
            st.session_state.db_manager = DBManager.get_instance()
            logger.info("Database manager initialized successfully")
        except Exception as e:
            logger.error(f"Database manager initialization failed: {str(e)}")
            st.error(f"Database initialization error: {str(e)}")

    # Initialize email queue if not already initialized
    try:
        from email_queue import initialize_email_queue, get_email_queue
        if 'email_queue_initialized' not in st.session_state:
            initialize_email_queue(AUTH_CONFIG['SMTP'])
            st.session_state.email_queue_initialized = True
            logger.info("Email queue initialized successfully from main()")
    except Exception as e:
        logger.error(f"Email queue initialization failed: {str(e)}")
        st.error(f"Email queue initialization error: {str(e)}")

    # Initialize session state if needed
    initialize_session_state()

    # Initialize user database and email queue
    # try:
    #     UserUtils.init_db()
    #     logger.info("用户数据库初始化成功")
    # except Exception as e:
    #     logger.error(f"用户数据库初始化失败: {str(e)}")
    #     st.error(f"数据库初始化错误: {str(e)}")

    # Check if user is authenticated
    if not check_authentication():
        render_login_page()
        return

        # Set app title and show header
    st.title("EVS Navigation")

    # Note: File alias info moved to Admin Panel
    # show_file_alias_info()  # Commented out - functionality moved to Admin Panel

    # Add user privacy controls for non-admin users
    if not st.session_state.get('is_admin', False):
        from privacy_settings import render_user_privacy_controls
        render_user_privacy_controls()

    # Create tabs based on user role
    if st.session_state.get('is_admin', False):
        tabs = st.tabs([
            "Transcribe Audio",
            "File List",
            "Edit Transcription",
            "Annotate EVS",
            "Analyse Concordance",
            "Speech Duration",
            "SI Analysis",
            "Download Audio",
            "Admin Panel"
        ])
    else:
        tabs = st.tabs([
            "Transcribe Audio",
            "File List",
            "Edit Transcription",
            "Annotate EVS",
            "Analysis Concordance",
            "Speech Duration",
            "SI Analysis",
            "Download Audio"
        ])

    with tabs[0]:
        # Language-specific ASR configuration imports
        from config.asr_language_config import (
            get_model_options_for_ui,
            get_chinese_asr_recommendations,
            check_funasr_available,
            check_crisperwhisper_available,
            get_available_providers,
            FUNASR_MODELS,
            CRISPERWHISPER_MODELS
        )

        # Check FunASR availability
        funasr_available = check_funasr_available()
        # Check CrisperWhisper availability
        crisperwhisper_available = check_crisperwhisper_available()

        # Show GPU and cache status
        from utils.asr_utils import get_gpu_info, get_cache_status
        gpu_info = get_gpu_info()
        cache_status = get_cache_status()

        col_gpu, col_cache = st.columns(2)
        with col_gpu:
            if gpu_info['available']:
                st.success(f"🚀 GPU: **{gpu_info['name']}** ({gpu_info['memory_free']:.1f}GB free)")
            else:
                st.warning("⚠️ No GPU - using CPU (slower)")
        with col_cache:
            if cache_status['count'] > 0:
                # Format cached model names for display
                model_names = [m.replace('funasr_', 'FunASR:').replace('crisperwhisper', 'CrisperWhisper')
                              for m in cache_status['cached_models']]
                st.info(f"📦 Cached: {', '.join(model_names)}")
            else:
                st.caption("📦 No models cached yet")

        # Load language ASR defaults from database first, then session state, then defaults
        lang_asr_defaults = None

        # Try loading from database
        try:
            db_result = EVSDataUtils.get_asr_config('language_defaults')
            if db_result and 'config' in db_result:
                lang_asr_defaults = db_result['config']
                # Sync to session state for consistency
                st.session_state['lang_asr_config'] = lang_asr_defaults
        except Exception as e:
            logger.debug(f"No language defaults in database: {e}")

        # Fallback to session state or defaults
        if not lang_asr_defaults:
            lang_asr_defaults = st.session_state.get('lang_asr_config', {
                'en': {'provider': 'crisperwhisper', 'model': 'crisperwhisper'},
                'zh': {'provider': 'funasr', 'model': 'paraformer-zh'}
            })

        # Apply auto-detected languages before widgets are created (one-time)
        if st.session_state.get('apply_auto_languages', False):
            detected_langs = st.session_state.get('auto_detected_languages', {})
            # Map detected language codes to display names
            lang_code_to_name = {v: k for k, v in LANGUAGES.items()}  # {'en': 'English', 'zh': 'Chinese'}

            ch1_lang_code = detected_langs.get('channel1')
            ch2_lang_code = detected_langs.get('channel2')

            if ch1_lang_code in lang_code_to_name:
                st.session_state['common_lang1'] = lang_code_to_name[ch1_lang_code]
            if ch2_lang_code in lang_code_to_name:
                st.session_state['common_lang2'] = lang_code_to_name[ch2_lang_code]

            st.session_state['apply_auto_languages'] = False

        # Channel Language Settings
        lang_options = list(LANGUAGES.keys())  # ['English', 'Chinese']

        # Initialize session state with defaults if not set (avoids index/session_state conflict)
        if 'common_lang1' not in st.session_state:
            st.session_state['common_lang1'] = 'English'
        if 'common_lang2' not in st.session_state:
            st.session_state['common_lang2'] = 'Chinese'

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            lang1 = st.selectbox(
                "Channel 1 Language",
                lang_options,
                key="common_lang1"
            )
        with col2:
            lang2 = st.selectbox(
                "Channel 2 Language",
                lang_options,
                key="common_lang2"
            )
        with col3:
            slice_duration = st.number_input(
                "Duration (seconds)",
                value=60,
                min_value=30,
                max_value=180,
                step=30,
                key="slice_duration"
            )

        # Channel preview status is now shown automatically after upload (removed manual message)

        # Display language detection result message (shown after rerun)
        if 'detection_message' in st.session_state:
            msg = st.session_state['detection_message']
            st.success(f"🔍 Language detected: Channel 1 = **{msg['ch1_lang'].upper()}** ({msg['ch1_conf']:.0%}), Channel 2 = **{msg['ch2_lang'].upper()}** ({msg['ch2_conf']:.0%})")
            st.info("✅ Language selections updated automatically based on detection")
            # Clear message after displaying
            del st.session_state['detection_message']

        # Show current channel language settings
        st.info(f"ℹ️ Current settings: Channel 1 → {lang1}, Channel 2 → {lang2}")

        # ASR Configuration by Language
        st.markdown("### 🎯 ASR Configuration by Language")
        st.info("Configure ASR provider and model for each language separately")

        # Always use language-specific ASR (remove the toggle)
        model_folder = ""

        col_en, col_zh = st.columns(2)

        # English ASR Configuration
        with col_en:
            st.markdown("#### 🇬🇧 English ASR")

            # English provider: CrisperWhisper (verbatim ASR)
            en_provider = 'crisperwhisper'
            en_model = 'crisperwhisper'
            st.info("**CrisperWhisper**: Verbatim English ASR with precise word-level timestamps, filler/stutter/false-start detection. ~3GB VRAM")

        # Chinese ASR Configuration
        with col_zh:
            st.markdown("#### 🇨🇳 Chinese ASR")

            # Chinese provider: FunASR
            zh_provider = "funasr"
            st.info("**FunASR**: Alibaba's Paraformer model for high-accuracy Chinese ASR with punctuation and timestamps.")
            zh_models = ["paraformer-zh", "SenseVoiceSmall"]
            zh_default_model = lang_asr_defaults.get('zh', {}).get('model', 'paraformer-zh')
            zh_model_idx = zh_models.index(zh_default_model) if zh_default_model in zh_models else 0

            zh_model = st.selectbox(
                "FunASR Model",
                zh_models,
                index=zh_model_idx,
                key="asr_zh_model_funasr",
                help="paraformer-zh provides the best Chinese ASR accuracy"
            )

        # Show current configuration summary
        st.markdown("---")
        st.markdown("**📋 Current ASR Configuration:**")

        # Get NLP engine from admin config
        nlp_config = EVSDataUtils.get_asr_config('nlp')
        if nlp_config and 'config' in nlp_config:
            import json
            nlp_cfg = nlp_config['config']
            if isinstance(nlp_cfg, str):
                try:
                    nlp_cfg = json.loads(nlp_cfg)
                except:
                    nlp_cfg = {}
            configured_nlp_engine = nlp_cfg.get('engine', 'jieba')
        else:
            configured_nlp_engine = 'jieba'

        config_col1, config_col2, config_col3 = st.columns([2, 2, 1])
        with config_col1:
            st.success(f"🇬🇧 English: {en_provider} / {en_model}")
        with config_col2:
            st.success(f"🇨🇳 Chinese: {zh_provider} / {zh_model}")
        with config_col3:
            st.info(f"NLP: {configured_nlp_engine}")

        # Show FunASR installation hint if not available
        if not funasr_available:
            with st.expander("💡 Improve Chinese ASR - Install FunASR"):
                st.markdown(get_chinese_asr_recommendations())
                st.code("pip install funasr modelscope torch torchaudio", language="bash")

        # Set model_folder for file organization
        model_folder = f"{en_model}_{zh_model}"

        # Set flag for language-specific ASR (always true now)
        # Store in a way that doesn't conflict with widgets
        use_language_specific = True

        # Set asr_provider for other parts of the code
        asr_provider = "CrisperWhisper"
        selected_asr_model = en_model

        # Upload mode selection
        upload_mode = st.radio(
            "Upload Mode",
            ["Stereo file (auto-separate)", "Pre-separated mono files"],
            key="upload_mode",
            horizontal=True,
            help="Choose stereo for a single file with both channels, or pre-separated for already split mono files"
        )

        # Variables to track uploaded files
        uploaded_file = None
        en_uploaded_file = None
        zh_uploaded_file = None

        if upload_mode == "Stereo file (auto-separate)":
            uploaded_file = st.file_uploader(
                "Upload Stereo Audio File",
                type=['wav', 'mp3', 'm4a'],
                key="asr_audio_uploader"
            )
        else:
            # Pre-separated mode - dual file uploaders
            st.info("Upload two separate mono audio files - one for each language channel")
            col_en, col_zh = st.columns(2)
            with col_en:
                en_uploaded_file = st.file_uploader(
                    "English Channel Audio",
                    type=['wav', 'mp3', 'm4a'],
                    key="en_audio_uploader"
                )
                if en_uploaded_file:
                    st.audio(en_uploaded_file)
            with col_zh:
                zh_uploaded_file = st.file_uploader(
                    "Chinese Channel Audio",
                    type=['wav', 'mp3', 'm4a'],
                    key="zh_audio_uploader"
                )
                if zh_uploaded_file:
                    st.audio(zh_uploaded_file)

        # Helper function to save uploaded file to temp path
        def save_uploaded_to_temp(uploaded_file, suffix='.mp3'):
            """Save uploaded file to a temporary file and return the path"""
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                return temp_file.name

        # Helper function to safely get filename with proper encoding
        def get_safe_filename(uploaded_file):
            """Get filename with proper UTF-8 encoding for Chinese characters"""
            import hashlib
            from datetime import datetime

            try:
                filename = uploaded_file.name
                if not filename:
                    raise ValueError("Empty filename")

                # Get the base filename
                filename = os.path.basename(filename)

                # Try multiple encoding fixes
                fixed_filename = None

                # Method 1: If it's bytes, decode as UTF-8
                if isinstance(filename, bytes):
                    try:
                        fixed_filename = filename.decode('utf-8')
                    except UnicodeDecodeError:
                        try:
                            fixed_filename = filename.decode('gbk')  # Common Chinese encoding on Windows
                        except UnicodeDecodeError:
                            pass

                # Method 2: Try to fix mojibake (UTF-8 interpreted as Latin-1)
                if fixed_filename is None and isinstance(filename, str):
                    try:
                        # Check if it looks like mojibake (has invalid high bytes)
                        fixed_filename = filename.encode('latin-1').decode('utf-8')
                    except (UnicodeDecodeError, UnicodeEncodeError):
                        pass

                # Method 3: Try GBK -> UTF-8 conversion (Windows Chinese)
                if fixed_filename is None and isinstance(filename, str):
                    try:
                        fixed_filename = filename.encode('gbk').decode('gbk')
                    except (UnicodeDecodeError, UnicodeEncodeError):
                        pass

                # Method 4: Keep original if it's valid UTF-8
                if fixed_filename is None:
                    try:
                        filename.encode('utf-8')
                        fixed_filename = filename
                    except UnicodeEncodeError:
                        pass

                # If we got a valid filename, return it
                if fixed_filename:
                    return fixed_filename

                # Fallback: use original
                return filename

            except Exception as e:
                logger.warning(f"Failed to decode filename, using fallback: {e}")
                # Fallback: generate a safe filename with timestamp
                ext = '.mp3'
                try:
                    ext = os.path.splitext(uploaded_file.name)[-1] or '.mp3'
                except:
                    pass
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                try:
                    hash_suffix = hashlib.md5(str(uploaded_file.name).encode('utf-8', errors='ignore')).hexdigest()[:6]
                except:
                    hash_suffix = hashlib.md5(str(timestamp).encode()).hexdigest()[:6]
                return f"audio_{timestamp}_{hash_suffix}{ext}"

        if uploaded_file:
            # Create file alias for privacy protection immediately when file is uploaded
            base_filename = get_safe_filename(uploaded_file)
            alias_manager = st.session_state.file_alias_manager
            file_alias = alias_manager.get_or_create_alias(base_filename)

            # Display alias instead of original filename
            st.success(f"📁 File uploaded: **{file_alias}**")

            # Generate unique file ID for tracking auto-preview
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"

            # Auto-preview: Automatically separate channels when a new file is uploaded
            if st.session_state.get('auto_preview_file_id') != file_id:
                # Clean up old preview files if switching to a different file
                if 'channel_preview' in st.session_state:
                    try:
                        old_preview = st.session_state['channel_preview']
                        if 'temp_dir' in old_preview and os.path.exists(old_preview['temp_dir']):
                            import shutil
                            shutil.rmtree(old_preview['temp_dir'])
                            logger.info(f"Cleaned up old preview directory: {old_preview['temp_dir']}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up old preview files: {str(e)}")
                    del st.session_state['channel_preview']

                # Clear language detection state for new file
                for key in ['auto_detected_languages', 'apply_auto_languages', 'detection_message']:
                    if key in st.session_state:
                        del st.session_state[key]

                # Automatically separate audio channels
                with st.spinner("Automatically separating audio channels..."):
                    # Create temporary audio file for channel separation
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                        temp_file.write(uploaded_file.getbuffer())
                        temp_audio_file = temp_file.name

                    try:
                        # Separate audio channels
                        channel1_path, channel2_path, temp_dir, is_mono = ASRUtils.separate_audio_channels(temp_audio_file)

                        if is_mono:
                            st.warning("⚠️ Mono audio detected - single channel mode will be used.")
                            st.session_state['channel_preview'] = {'is_mono': True}
                        else:
                            st.session_state['channel_preview'] = {
                                'is_mono': False,
                                'channel1_path': str(channel1_path),
                                'channel2_path': str(channel2_path),
                                'temp_dir': str(temp_dir)
                            }

                        # Mark this file as processed
                        st.session_state['auto_preview_file_id'] = file_id

                    except Exception as e:
                        st.error(f"Channel separation failed: {str(e)}")
                        logger.error(f"Auto channel separation error: {str(e)}", exc_info=True)

                    finally:
                        # Clean up temporary audio file
                        try:
                            if os.path.exists(temp_audio_file):
                                os.unlink(temp_audio_file)
                        except OSError as e:
                            logger.debug(f"Could not delete temp audio file: {e}")

                # Auto-detect language for each channel (runs after separation completes)
                if 'channel_preview' in st.session_state:
                    preview_info = st.session_state['channel_preview']
                    if not preview_info.get('is_mono', True) and 'language_detection' not in preview_info:
                        with st.spinner("Detecting languages in each channel..."):
                            try:
                                detection_result = ASRUtils.detect_channel_languages(
                                    preview_info['channel1_path'],
                                    preview_info['channel2_path'],
                                    model_name="tiny"  # Use tiny model for fast detection
                                )

                                # Store detection results
                                st.session_state['channel_preview']['language_detection'] = detection_result

                                if detection_result['detection_successful']:
                                    ch1_lang = detection_result['channel1']['language']
                                    ch1_conf = detection_result['channel1']['confidence']
                                    ch2_lang = detection_result['channel2']['language']
                                    ch2_conf = detection_result['channel2']['confidence']

                                    # Store detected languages to auto-update the dropdowns
                                    st.session_state['auto_detected_languages'] = {
                                        'channel1': ch1_lang,
                                        'channel2': ch2_lang
                                    }
                                    st.session_state['apply_auto_languages'] = True

                                    # Store detection message for display after rerun
                                    st.session_state['detection_message'] = {
                                        'ch1_lang': ch1_lang,
                                        'ch1_conf': ch1_conf,
                                        'ch2_lang': ch2_lang,
                                        'ch2_conf': ch2_conf
                                    }

                                    # Rerun to apply the auto-detected languages to the dropdowns
                                    st.rerun()
                                else:
                                    st.warning("⚠️ Could not auto-detect languages. Please listen and verify manually.")

                            except Exception as e:
                                logger.error(f"Language detection error: {str(e)}", exc_info=True)
                                st.warning(f"⚠️ Language detection failed: {str(e)}. Please verify manually.")

            # Add CSS to hide the original filename in the file uploader
            st.markdown("""
            <style>
            div[data-testid="stFileUploader"] > div > div > div > div {
                display: none !important;
            }
            div[data-testid="stFileUploader"] > div > div > div > button {
                display: none !important;
            }
            </style>
            """, unsafe_allow_html=True)

            output_path = ""

            # Persistent channel preview display (auto-generated on upload)
            if 'channel_preview' in st.session_state:
                preview_info = st.session_state['channel_preview']
                if not preview_info.get('is_mono', True):
                    st.markdown("### 🎧 Channel Preview")
                    col1, col2 = st.columns(2)

                    # Get language detection results if available
                    detection = preview_info.get('language_detection', {})
                    ch1_detected = detection.get('channel1', {}).get('language')
                    ch1_conf = detection.get('channel1', {}).get('confidence', 0)
                    ch2_detected = detection.get('channel2', {}).get('language')
                    ch2_conf = detection.get('channel2', {}).get('confidence', 0)

                    with col1:
                        st.markdown("**Channel 1 (Left)**")
                        channel1_path = preview_info.get('channel1_path')
                        if channel1_path and os.path.exists(channel1_path):
                            st.audio(str(channel1_path))
                            if ch1_detected:
                                st.caption(f"🔍 Detected: {ch1_detected.upper()} ({ch1_conf:.0%})")
                            st.info(f"🎯 ASR: {lang1}")
                        else:
                            st.error("Channel 1 audio file not found")

                    with col2:
                        st.markdown("**Channel 2 (Right)**")
                        channel2_path = preview_info.get('channel2_path')
                        if channel2_path and os.path.exists(channel2_path):
                            st.audio(str(channel2_path))
                            if ch2_detected:
                                st.caption(f"🔍 Detected: {ch2_detected.upper()} ({ch2_conf:.0%})")
                            st.info(f"🎯 ASR: {lang2}")
                        else:
                            st.error("Channel 2 audio file not found")

                    # Original stereo audio in collapsed expander
                    with st.expander("🔊 Original Stereo Audio"):
                        st.audio(uploaded_file)

                    # Language assignment guide in collapsed expander
                    with st.expander("📋 Language Assignment Guide"):
                        st.markdown("""
Language is auto-detected and the dropdowns above are updated automatically.
If the detection is incorrect, manually change the Channel Language dropdowns above.
                        """)
                else:
                    # Mono audio - show original
                    st.audio(uploaded_file)
            else:
                # Fallback if no preview (shouldn't happen normally)
                st.audio(uploaded_file)

        # Handle pre-separated file uploads
        if en_uploaded_file and zh_uploaded_file:
            alias_manager = st.session_state.file_alias_manager

            # Create aliases for both files (with proper encoding)
            en_base_filename = get_safe_filename(en_uploaded_file)
            zh_base_filename = get_safe_filename(zh_uploaded_file)
            en_file_alias = alias_manager.get_or_create_alias(en_base_filename)
            zh_file_alias = alias_manager.get_or_create_alias(zh_base_filename)

            st.success(f"📁 English file: **{en_file_alias}** | Chinese file: **{zh_file_alias}**")

        # Determine if we can process (either stereo file or both pre-separated files)
        can_process = uploaded_file or (en_uploaded_file and zh_uploaded_file)

        # Check for existing file + model combination and show overwrite warning
        overwrite_confirmed = True  # Default to True if no existing data
        if can_process and asr_provider == "CrisperWhisper":
            # Get the file name to check
            check_filename = None
            if upload_mode == "Pre-separated mono files" and en_uploaded_file:
                check_filename = get_safe_filename(en_uploaded_file)
            elif uploaded_file:
                check_filename = get_safe_filename(uploaded_file)

            if check_filename:
                # Check if file + model already exists
                existing = EVSDataUtils.check_file_model_exists(
                    check_filename,
                    selected_asr_model,
                    ASR_PROVIDERS[asr_provider]
                )
                if existing:
                    st.warning(f"⚠️ Data already exists for file '{check_filename}' with model '{selected_asr_model}'. Processing will overwrite existing data.")
                    overwrite_confirmed = st.checkbox(
                        "I understand and want to overwrite existing data",
                        key="overwrite_confirm"
                    )

        if can_process and overwrite_confirmed and st.button("Process Audio", key="asr_transcribe_button"):
            with st.spinner("Starting transcription..."):
                container = st.empty()
                # Get duration from session state
                slice_duration = st.session_state.get('duration', 30)  # Default to 30 if not set
                is_pre_separated = upload_mode == "Pre-separated mono files"

                # Initialize variables
                channel1_path = None
                channel2_path = None
                is_mono = False
                en_base_filename = None
                zh_base_filename = None

                if is_pre_separated:
                    # Pre-separated mode: use uploaded files directly
                    st.info("📂 Using pre-separated audio files (skipping channel separation)")

                    # Save uploaded files to temp paths
                    channel1_path = save_uploaded_to_temp(en_uploaded_file)  # English channel
                    channel2_path = save_uploaded_to_temp(zh_uploaded_file)  # Chinese channel
                    is_mono = False  # Treat as two separate channels

                    # Get filenames for each file individually (with proper encoding)
                    en_base_filename = get_safe_filename(en_uploaded_file)
                    zh_base_filename = get_safe_filename(zh_uploaded_file)

                    # Create aliases for privacy
                    alias_manager = st.session_state.file_alias_manager
                    en_file_alias = alias_manager.get_or_create_alias(en_base_filename)
                    zh_file_alias = alias_manager.get_or_create_alias(zh_base_filename)
                    logger.info(f"Pre-separated mode: EN='{en_file_alias}', ZH='{zh_file_alias}'")

                    # For pre-separated, we don't need to slice as each file is already separate
                    # Set output_path to a dummy value to pass the check
                    output_path = "pre-separated"
                    audio_file = channel1_path  # Use EN file as reference

                else:
                    # Stereo mode: existing logic
                    # Check if need to slice audio
                    if slice_duration > 0:
                        output_path = ASRUtils.slice_audio_file(uploaded_file, slice_duration, ASR_PROVIDERS[asr_provider], model_folder)

                    # Check conversion result
                    if not output_path:
                        st.error("Failed to convert audio file. Please check the format and try again.")
                        return

                    # Ensure uploaded_file exists and has name attribute
                    if not uploaded_file or not hasattr(uploaded_file, 'name'):
                        st.error("Invalid upload file or file name not available.")
                        return

                    # to be compatible with subsequent processing, create a directory containing the converted MP3 file
                    sub_dir = os.path.splitext(uploaded_file.name)[0]
                    target_path = f"{SLICE_AUDIO_PATH}/{sub_dir}"
                    os.makedirs(target_path, exist_ok=True)

                    # Fix: Create audio file from uploaded data instead of copying directory
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                        temp_file.write(uploaded_file.getbuffer())
                        audio_file = temp_file.name

                    # Save original audio for playback
                    original_audio_path = os.path.join(target_path, "original.mp3")
                    if not os.path.exists(original_audio_path):
                        uploaded_file.seek(0)
                        with open(original_audio_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                        logger.info(f"Saved original audio to {original_audio_path}")

                    target_path = f"{SLICE_AUDIO_PATH}/{os.path.splitext(uploaded_file.name)[0]}"

                    container.write(f"Transcribing {audio_file}...")

                    # Check if channels were already separated during preview
                    if ('channel_preview' in st.session_state and
                        not st.session_state['channel_preview'].get('is_mono', True) and
                        'channel1_path' in st.session_state['channel_preview'] and
                        'channel2_path' in st.session_state['channel_preview']):

                        # Reuse previously separated channels
                        channel1_path = st.session_state['channel_preview']['channel1_path']
                        channel2_path = st.session_state['channel_preview']['channel2_path']
                        is_mono = False
                        st.info("🔄 Reusing channel files separated during preview")
                    else:
                        # Separate audio channels
                        channel1_path, channel2_path, _, is_mono = ASRUtils.separate_audio_channels(audio_file)

                # Log audio channel information
                if is_mono:
                    st.info("Mono audio detected, will only process the first channel")
                elif not is_pre_separated:
                    st.info("Stereo audio detected, will process both channels separately")

                with st.spinner("Transcribing audio..."):
                    # Determine language assignment and filenames based on mode
                    if is_pre_separated:
                        # Pre-separated mode: fixed assignment (EN file → English, ZH file → Chinese)
                        actual_lang1 = "English"
                        actual_lang2 = "Chinese"
                        actual_channel1_lang = "en"  # English channel
                        actual_channel2_lang = "zh"  # Chinese channel
                        # Use English filename as base for both languages (for EVS annotation)
                        base_filename = en_base_filename
                        channel1_filename = base_filename
                        channel2_filename = base_filename  # Same as EN for EVS to show both languages
                        alias_manager = st.session_state.file_alias_manager
                        file_alias = alias_manager.get_or_create_alias(base_filename)
                        st.info("📂 Pre-separated: English file → EN channel, Chinese file → ZH channel")
                    else:
                        # Stereo mode: get filename and check swap
                        base_filename = get_safe_filename(uploaded_file)
                        channel1_filename = base_filename
                        channel2_filename = base_filename
                        alias_manager = st.session_state.file_alias_manager
                        file_alias = alias_manager.get_or_create_alias(base_filename)
                        logger.info(f"Using alias '{file_alias}' for uploaded file '{base_filename}'")

                        # Use language settings directly from dropdowns (auto-detected)
                        actual_lang1 = lang1
                        actual_lang2 = lang2
                        actual_channel1_lang = LANGUAGES[lang1]
                        actual_channel2_lang = LANGUAGES[lang2]
                        st.info(f"🎯 Channel 1 → {lang1}, Channel 2 → {lang2}")

                    # Process first channel (English for pre-separated)
                    if channel1_path:
                        words_data_1 = pd.DataFrame()
                        segments_data_1 = pd.DataFrame()

                        with st.spinner(f"Transcribing {actual_lang1} channel..."):
                            # Get ASR settings based on channel language
                            if actual_channel1_lang == 'en':
                                transcribe_provider = 'crisperwhisper'
                                transcribe_model = en_model
                                st.info(f"🇬🇧 Using CrisperWhisper for English")
                            else:  # Chinese
                                transcribe_provider = 'funasr'
                                transcribe_model = st.session_state.get('asr_zh_model_funasr', zh_model)
                                st.info(f"🇨🇳 Using FunASR/{transcribe_model} for Chinese")

                            words_data_1, segments_data_1 = ASRUtils.transcribe_audio(
                                channel1_path, transcribe_provider, transcribe_model,
                                actual_channel1_lang, channel1_filename, slice_duration, audio_file, 1
                            )

                            # Check transcription results
                            if words_data_1 is not None and len(words_data_1) > 0:
                                save_asr_result_to_database(words_data_1, segments_data_1)
                                st.success(f"Channel 1 ({actual_lang1}) - Results saved to database - {file_alias}")
                            else:
                                error_detail = ASRUtils._last_error or "No transcription results returned"
                                st.error(f"Failed to transcribe channel 1 - {file_alias}: {error_detail}")

                    # Process second channel if available (Chinese for pre-separated)
                    if channel2_path:
                        words_data_2 = pd.DataFrame()
                        segments_data_2 = pd.DataFrame()
                        with st.spinner(f"Transcribing {actual_lang2} channel..."):
                            # Get ASR settings based on channel language
                            if actual_channel2_lang == 'zh':
                                transcribe_provider = 'funasr'
                                transcribe_model = st.session_state.get('asr_zh_model_funasr', zh_model)
                                st.info(f"🇨🇳 Using FunASR/{transcribe_model} for Chinese")
                            else:  # English
                                transcribe_provider = 'crisperwhisper'
                                transcribe_model = en_model
                                st.info(f"🇬🇧 Using CrisperWhisper for English")

                            words_data_2, segments_data_2 = ASRUtils.transcribe_audio(
                                channel2_path, transcribe_provider, transcribe_model,
                                actual_channel2_lang, channel2_filename, slice_duration, audio_file, 2
                            )

                            # Check transcription results
                            if words_data_2 is not None and len(words_data_2) > 0:
                                save_asr_result_to_database(words_data_2, segments_data_2)
                                st.success(f"Channel 2 ({actual_lang2}) - Results saved to database - {file_alias}")
                            else:
                                error_detail = ASRUtils._last_error or "No transcription results returned"
                                st.error(f"Failed to transcribe channel 2 - {file_alias}: {error_detail}")
                    else:
                        if not is_pre_separated:
                            st.error("The audio file must have two channels.")

                    # Clean up temporary audio file
                    try:
                        if os.path.exists(audio_file):
                            os.unlink(audio_file)
                            logger.info(f"Cleaned up temporary file: {audio_file}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up temporary file {audio_file}: {str(e)}")

                    # Clean up channel preview files if they exist
                    if 'channel_preview' in st.session_state:
                        try:
                            preview_info = st.session_state['channel_preview']
                            if 'temp_dir' in preview_info and os.path.exists(preview_info['temp_dir']):
                                import shutil
                                shutil.rmtree(preview_info['temp_dir'])
                                logger.info(f"Cleaned up preview directory: {preview_info['temp_dir']}")
                            # Clear the preview session state
                            del st.session_state['channel_preview']
                        except Exception as e:
                            logger.warning(f"Failed to clean up preview files: {str(e)}")

                    # Clear auto-preview file tracking so next upload triggers new preview
                    if 'auto_preview_file_id' in st.session_state:
                        del st.session_state['auto_preview_file_id']

                    # Clear language detection session state
                    for key in ['auto_detected_languages', 'apply_auto_languages', 'detection_message']:
                        if key in st.session_state:
                            del st.session_state[key]

                    # Process Chinese text with configured NLP engine for word segmentation
                    # Get NLP engine from admin config
                    nlp_config = EVSDataUtils.get_asr_config('nlp')
                    if nlp_config and 'config' in nlp_config:
                        nlp_cfg = nlp_config['config']
                        if isinstance(nlp_cfg, str):
                            try:
                                nlp_cfg = json.loads(nlp_cfg)
                            except (json.JSONDecodeError, TypeError, ValueError):
                                nlp_cfg = {}
                        selected_nlp_engine = nlp_cfg.get('engine', 'jieba')
                    else:
                        selected_nlp_engine = 'jieba'

                    # Determine the actual Chinese ASR provider from database
                    zh_provider_for_nlp = 'funasr'  # default fallback
                    file_asr_info = EVSDataUtils.get_file_asr_info(base_filename)
                    zh_info = file_asr_info.get('zh')
                    if zh_info:
                        zh_provider_for_nlp = zh_info['provider']

                    with st.spinner(f"Processing Chinese with {selected_nlp_engine}..."):
                        try:
                            nlp_success = process_chinese_nlp_unified(base_filename, zh_provider_for_nlp, selected_nlp_engine)
                            if nlp_success:
                                st.success(f"✅ Chinese NLP processing complete ({selected_nlp_engine})")
                            else:
                                st.warning(f"⚠️ Chinese NLP processing skipped (no Chinese data or {selected_nlp_engine} failed)")
                        except Exception as e:
                            logger.warning(f"Chinese NLP processing failed: {str(e)}")
                            st.warning(f"⚠️ Chinese NLP processing failed: {str(e)}")

                    st.success("✅ Processing complete! Files are now available in other tabs.")

                    container.empty()
                    container.write("Transcription completed successfully")

        st.divider()

    with tabs[1]:  # File List tab
        render_file_list_tab()
    with tabs[2]:
        render_edit_transcription_tab()
    with tabs[3]:
        render_annotate_evs_tab()
    with tabs[4]:
        render_analysis_concordance_tab()

    with tabs[5]:  # Speech Duration tab
        st.subheader("Speech and Pause Duration Analysis")

        # Get all files with their transcription info (supports different ASR per language)
        duration_files_df = EVSDataUtils.get_all_files_with_transcriptions()

        if duration_files_df.empty:
            st.warning("No transcription files found.")
            duration_file = None
        else:
            # File and model selection
            sel_col1, sel_col2 = st.columns([1, 1])
            with sel_col1:
                file_options = [""] + duration_files_df['file_name'].tolist()
                duration_file = st.selectbox(
                    "File",
                    options=file_options,
                    index=0,
                    format_func=lambda x: "-- Select a file --" if x == "" else get_file_display_name(x),
                    key='duration_file'
                )
                if duration_file == "":
                    duration_file = None

            duration_selected_pair = None
            if duration_file:
                duration_model_pairs = EVSDataUtils.get_file_model_pairs(duration_file)
                if duration_model_pairs:
                    with sel_col2:
                        if len(duration_model_pairs) == 1:
                            duration_selected_pair = duration_model_pairs[0]
                            st.caption(f"Model: {duration_selected_pair['display_name']}")
                        else:
                            pair_options = [p['display_name'] for p in duration_model_pairs]
                            selected_display = st.selectbox(
                                "Model Pair",
                                options=pair_options,
                                index=0,
                                key='duration_model_pair'
                            )
                            duration_selected_pair = next((p for p in duration_model_pairs if p['display_name'] == selected_display), duration_model_pairs[0])

        if duration_file and duration_selected_pair:
            display_name = get_file_display_name(duration_file)
            en_provider = duration_selected_pair['en_provider']
            zh_provider = duration_selected_pair['zh_provider']

            # X-axis option
            opt_col1, opt_col2 = st.columns([1, 1])
            with opt_col1:
                x_axis_mode = st.radio(
                    "X-axis",
                    options=["Duration", "Segment"],
                    index=0,
                    horizontal=True,
                    key='duration_x_axis',
                    label_visibility="collapsed"
                )
            with opt_col2:
                if x_axis_mode == "Duration":
                    bucket_size = st.number_input(
                        "Interval (sec)",
                        min_value=5, max_value=300, value=30, step=5,
                        key='duration_bucket_size'
                    )

            # Load data for both languages
            en_df_raw = EVSDataUtils.get_asr_data(duration_file, en_provider)
            zh_df_raw = EVSDataUtils.get_asr_data(duration_file, zh_provider)
            en_df = en_df_raw[en_df_raw['lang'] == 'en'].copy() if not en_df_raw.empty else pd.DataFrame()
            zh_df = zh_df_raw[zh_df_raw['lang'] == 'zh'].copy() if not zh_df_raw.empty else pd.DataFrame()

            # Color palette for stacked word bars
            WORD_COLORS = [
                '#4285f4', '#34a853', '#fbbc04', '#ea4335', '#46bdc6',
                '#7baaf7', '#57bb8a', '#f7cb4d', '#e06666', '#76d7c4',
                '#a4c2f4', '#93c47d', '#ffd966', '#e69138', '#6fa8dc',
                '#8e7cc3', '#c27ba0', '#6d9eeb', '#b6d7a8', '#ffe599',
            ]

            def fmt_time(t):
                return f"{int(t//60)}:{t%60:05.2f}"

            def make_duration_chart(df, lang_label, color_pause):
                """Create a stacked speech/pause duration chart for one language.
                Each ASR word is a stacked bar segment; pause stacked on top."""
                df = df.sort_values('start_time').reset_index(drop=True)
                df['pause_duration'] = df['start_time'].shift(-1) - (df['start_time'] + df['duration'])
                df['word_text'] = df['combined_word'].str.split('&&').str[0]

                fig = go.Figure()

                if x_axis_mode == "Segment":
                    # Group by segment_id, stack each word within segment
                    segments = df.groupby('segment_id', sort=True)
                    seg_labels = []
                    seg_pause = []

                    # Find max words per segment for trace count
                    max_words = segments.size().max()

                    # Build one trace per word position (0th word, 1st word, ...)
                    for word_idx in range(max_words):
                        y_vals = []
                        texts = []
                        x_vals = []
                        for seg_id, seg_df in segments:
                            seg_df = seg_df.sort_values('start_time').reset_index(drop=True)
                            if word_idx < len(seg_df):
                                row = seg_df.iloc[word_idx]
                                y_vals.append(row['duration'])
                                texts.append(f"{row['word_text']}<br>{fmt_time(row['start_time'])}")
                            else:
                                y_vals.append(0)
                                texts.append('')
                            x_vals.append(str(seg_id))

                        color = WORD_COLORS[word_idx % len(WORD_COLORS)]
                        fig.add_trace(go.Bar(
                            name=f'Word {word_idx+1}',
                            x=x_vals, y=y_vals,
                            marker_color=color,
                            text=texts,
                            textposition='inside', insidetextanchor='middle',
                            textfont_size=9, constraintext='inside',
                            showlegend=(word_idx == 0),
                        ))

                    # Pause trace: use last word's pause per segment
                    for seg_id, seg_df in segments:
                        seg_df = seg_df.sort_values('start_time').reset_index(drop=True)
                        last_pause = seg_df.iloc[-1]['pause_duration']
                        seg_labels.append(str(seg_id))
                        seg_pause.append(max(0, last_pause) if pd.notna(last_pause) else 0)

                    fig.add_trace(go.Bar(
                        name='Pause', x=seg_labels, y=seg_pause,
                        marker_color=color_pause,
                        text=[f"{p:.2f}s" if p > 0.01 else '' for p in seg_pause],
                        textposition='inside', insidetextanchor='middle',
                        textfont_size=9, constraintext='inside',
                    ))
                    fig.update_layout(
                        title=f'{lang_label} - {display_name}',
                        xaxis_title='Segment ID', yaxis_title='Duration (seconds)',
                        barmode='stack', height=500, showlegend=True,
                        xaxis_type='category',
                    )

                else:
                    # Duration mode: group by time buckets, stack words within each
                    max_time = df['start_time'].max() + df['duration'].iloc[-1] if len(df) > 0 else 0
                    bucket_edges = list(range(0, int(max_time) + bucket_size + 1, bucket_size))

                    # Assign each word to a bucket
                    df['bucket'] = pd.cut(
                        df['start_time'], bins=bucket_edges, right=False,
                        labels=[f"{fmt_time(bucket_edges[i])}-{fmt_time(bucket_edges[i+1])}"
                                for i in range(len(bucket_edges)-1)]
                    )

                    buckets = df.groupby('bucket', sort=True, observed=True)
                    bucket_labels_list = [str(b) for b in buckets.groups.keys()]
                    max_words = buckets.size().max() if len(buckets) > 0 else 0

                    # Stack word traces
                    for word_idx in range(max_words):
                        y_vals = []
                        texts = []
                        x_vals = []
                        for b_label, b_df in buckets:
                            b_df = b_df.sort_values('start_time').reset_index(drop=True)
                            if word_idx < len(b_df):
                                row = b_df.iloc[word_idx]
                                y_vals.append(row['duration'])
                                texts.append(f"{row['word_text']}<br>{fmt_time(row['start_time'])}")
                            else:
                                y_vals.append(0)
                                texts.append('')
                            x_vals.append(str(b_label))

                        color = WORD_COLORS[word_idx % len(WORD_COLORS)]
                        fig.add_trace(go.Bar(
                            name=f'Word {word_idx+1}',
                            x=x_vals, y=y_vals,
                            marker_color=color,
                            text=texts,
                            textposition='inside', insidetextanchor='middle',
                            textfont_size=9, constraintext='inside',
                            showlegend=(word_idx == 0),
                        ))

                    # Pause: sum of pauses within each bucket
                    bucket_pause = []
                    for b_label, b_df in buckets:
                        p = b_df['pause_duration'].dropna().clip(lower=0).sum()
                        bucket_pause.append(p)

                    fig.add_trace(go.Bar(
                        name='Pause', x=bucket_labels_list, y=bucket_pause,
                        marker_color=color_pause,
                        text=[f"{p:.2f}s" if p > 0.01 else '' for p in bucket_pause],
                        textposition='inside', insidetextanchor='middle',
                        textfont_size=9, constraintext='inside',
                    ))
                    fig.update_layout(
                        title=f'{lang_label} - {display_name}',
                        xaxis_title='Time Interval', yaxis_title='Duration (seconds)',
                        barmode='stack', height=500, showlegend=True,
                        xaxis_type='category',
                    )

                return fig, df

            # English chart
            if not en_df.empty:
                st.markdown("#### English")
                en_fig, en_df = make_duration_chart(en_df, 'English Speech & Pause', '#ea4335')
                st.plotly_chart(en_fig, use_container_width=True)
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Total Speech", f"{en_df['duration'].sum():.2f}s")
                with c2:
                    st.metric("Avg Speech", f"{en_df['duration'].mean():.2f}s")
                with c3:
                    st.metric("Total Pause", f"{en_df['pause_duration'].sum():.2f}s")
                with c4:
                    st.metric("Avg Pause", f"{en_df['pause_duration'].mean():.2f}s")
            else:
                st.info("No English data found.")

            # Chinese chart
            if not zh_df.empty:
                st.markdown("#### Chinese")
                zh_fig, zh_df = make_duration_chart(zh_df, 'Chinese Speech & Pause', '#f4b400')
                st.plotly_chart(zh_fig, use_container_width=True)
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Total Speech", f"{zh_df['duration'].sum():.2f}s")
                with c2:
                    st.metric("Avg Speech", f"{zh_df['duration'].mean():.2f}s")
                with c3:
                    st.metric("Total Pause", f"{zh_df['pause_duration'].sum():.2f}s")
                with c4:
                    st.metric("Avg Pause", f"{zh_df['pause_duration'].mean():.2f}s")
            else:
                st.info("No Chinese data found.")

            # Combined detail table
            if not en_df.empty or not zh_df.empty:
                st.subheader("Detailed Duration Data")
                for lang_label, lang_df in [('English', en_df), ('Chinese', zh_df)]:
                    if lang_df.empty:
                        continue
                    st.markdown(f"**{lang_label}**")
                    detail_df = lang_df[['segment_id', 'start_time', 'combined_word', 'duration', 'pause_duration']].copy()
                    detail_df['combined_word'] = detail_df['combined_word'].str.split('&&').str[0]
                    detail_df['start_time'] = detail_df['start_time'].apply(lambda t: f"{int(t//60)}:{t%60:05.2f}")
                    detail_df.columns = ['Segment ID', 'Timestamp', 'Text', 'Speech Duration', 'Pause Duration']
                    st.dataframe(
                        detail_df,
                        column_config={
                            'Segment ID': st.column_config.NumberColumn('Segment ID', width='small'),
                            'Timestamp': st.column_config.TextColumn('Timestamp', width='small'),
                            'Text': st.column_config.TextColumn('Text', width='medium'),
                            'Speech Duration': st.column_config.NumberColumn('Speech (s)', format="%.2f", width='small'),
                            'Pause Duration': st.column_config.NumberColumn('Pause (s)', format="%.2f", width='small')
                        },
                        hide_index=True
                    )

        elif duration_file and not duration_selected_pair:
            st.warning("No model pair found for the selected file.")
        else:
            st.info("Please select a file to analyze speech and pause durations.")

    with tabs[6]:  # SI Analysis tab
        render_si_analysis_tab()

    with tabs[7]:  # Download Audio tab
        render_download_audio_tab()

    # Add Admin Panel tab for admin users
    if st.session_state.get('is_admin', False):
        with tabs[8]:  # Admin Panel tab
            render_admin_panel()  # 调用现有的管理员面板渲染函数

def render_si_analysis_tab():
    """Render the SI analysis tab with enhanced diagnostics and advanced interpretation analysis"""
    st.subheader("🎯 Simultaneous Interpretation Analysis")

    # Data diagnostics section
    with st.expander("📊 Data Diagnostics", expanded=False):
        if st.button("🔍 Data Structure Diagnostics", help="Check ASR data structure and completeness"):
            try:
                # Get available files
                available_files = EVSDataUtils.get_interpret_files("crisperwhisper")
                if available_files.empty:
                    available_files = EVSDataUtils.get_interpret_files("funasr")
                if not available_files.empty:
                    sample_file = available_files.iloc[0]['file_name']

                    # Get sample data for diagnostics
                    asr_data = EVSDataUtils.get_asr_data(sample_file, "crisperwhisper")
                    if asr_data.empty:
                        asr_data = EVSDataUtils.get_asr_data(sample_file, "funasr")

                    if not asr_data.empty:
                        st.success(f"✅ Data Structure Check - File: {sample_file}")

                        # Check columns
                        st.write("**Data Column Structure:**")
                        cols = list(asr_data.columns)
                        st.write(f"- Total {len(cols)} columns: {', '.join(cols)}")

                        # Check key columns
                        required_cols = ['start_time', 'end_time', 'duration', 'lang', 'edit_word']
                        missing_cols = [col for col in required_cols if col not in cols]
                        available_cols = [col for col in required_cols if col in cols]

                        if available_cols:
                            st.write("**✅ Available Key Columns:**")
                            for col in available_cols:
                                sample_val = asr_data[col].iloc[0] if not asr_data[col].empty else "N/A"
                                st.write(f"  - {col}: {sample_val}")

                        if missing_cols:
                            st.write("**⚠️ Missing Columns:**")
                            for col in missing_cols:
                                st.write(f"  - {col}")
                                if col == 'end_time' and 'duration' in cols:
                                    st.info("💡 End time can be calculated from start_time + duration")

                        # Check data quality
                        st.write("**Data Quality Check:**")
                        en_count = len(asr_data[asr_data['lang'] == 'en'])
                        zh_count = len(asr_data[asr_data['lang'] == 'zh'])
                        st.write(f"- English Vocabulary: {en_count}")
                        st.write(f"- Chinese Vocabulary: {zh_count}")

                        if en_count > 0 and zh_count > 0:
                            st.success("✅ Contains bilingual data, suitable for EVS analysis")
                        else:
                            st.warning("⚠️ Missing bilingual data, cannot perform EVS pairing")

                        # Check time ranges
                        min_time = asr_data['start_time'].min()
                        max_time = asr_data['start_time'].max()
                        st.write(f"- Time Range: {min_time:.2f}s - {max_time:.2f}s ({(max_time-min_time)/60:.1f} minutes)")

                    else:
                        st.error("❌ Unable to get ASR data for diagnosis")
                else:
                    st.warning("⚠️ No available files found for diagnosis")

            except Exception as e:
                st.error(f"❌ Error during diagnosis: {str(e)}")
                logger.error(f"Diagnostic error: {str(e)}")

    # Analysis features description
    st.markdown("""
    **Complete Analysis Features:**
    - 📊 **Basic SI Metrics**: EVS analysis, quality metrics, language distribution, confidence scores
    - 🎯 **Translation Accuracy Assessment**: LLM-powered semantic accuracy evaluation
    - 🔍 **Terminology Consistency Check**: Professional term usage analysis
    - 📈 **Discourse Analysis**: Cohesion and coherence evaluation
    - 🌐 **Cultural Adaptation Analysis**: Cultural context appropriateness
    - ⚡ **Advanced Performance Metrics**: Enhanced EVS and fluency indicators
    - 💡 **Improvement Recommendations**: Personalized suggestions for better performance
    """)

    # Original analysis interface
    available_files = EVSDataUtils.get_interpret_files("crisperwhisper")
    if available_files.empty:
        available_files = EVSDataUtils.get_interpret_files("funasr")

    # Create layout with selectors
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        si_asr_provider = st.selectbox(
            "ASR Provider",
            list(ASR_PROVIDERS.keys()),
            index=0,
            key='si_asr_provider'
        )

    with col2:
        # Get available files for the selected ASR provider
        files_df = EVSDataUtils.get_interpret_files(ASR_PROVIDERS[si_asr_provider])
        si_analysis_file = create_file_selectbox(
            files_df,
            label="File",
            key='si_analysis_file'
        )

    with col3:
        # Add a clear cache button
        if st.button("🧹 Clear Cache", type="secondary", key="clear_cache_button",
                     help="Clear all cached analysis results"):
            # Clear all analysis-related session state
            cache_keys = ['si_analysis_results', 'llm_improved_evs', 'show_abnormal_pairs', 'advanced_si_results',
                         'si_analysis_file_name', 'si_analysis_provider']
            for key in cache_keys:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("✅ 缓存已清除！请重新运行分析。")
            st.rerun()

        if st.button("🚀 Start Complete Analysis", type="primary", key="si_analysis_start",
                     help="Perform comprehensive SI analysis including basic metrics and advanced interpretation assessment"):
            if not si_analysis_file:
                st.warning("Please select a file to analyze")
                return

            # Clear any existing analysis results with more thorough cleanup
            cache_keys = ['si_analysis_results', 'llm_improved_evs', 'show_abnormal_pairs', 'advanced_si_results',
                         'si_analysis_file_name', 'si_analysis_provider']
            for key in cache_keys:
                if key in st.session_state:
                    del st.session_state[key]

            with st.spinner("Performing comprehensive SI analysis..."):
                try:
                    # Get ASR data for analysis
                    asr_data = EVSDataUtils.get_asr_data(si_analysis_file, ASR_PROVIDERS[si_asr_provider])

                    if asr_data.empty:
                        st.warning("No data found for the selected file")
                        return

                    # Phase 1: Perform basic SI analysis
                    st.text("Phase 1: Basic SI Analysis...")
                    analysis_results = perform_si_analysis(asr_data, si_analysis_file, ASR_PROVIDERS[si_asr_provider])

                    # Store basic results in session state
                    st.session_state.si_analysis_results = analysis_results
                    st.session_state.si_analysis_file_name = si_analysis_file
                    st.session_state.si_analysis_provider = ASR_PROVIDERS[si_asr_provider]

                    # Phase 2: Perform advanced analysis
                    st.text("Phase 2: Advanced Interpretation Analysis...")
                    advanced_results = perform_advanced_si_analysis(asr_data, si_analysis_file, ASR_PROVIDERS[si_asr_provider])

                    # Store advanced results in session state
                    st.session_state.advanced_si_results = advanced_results

                    # Debug: Log the structure of analysis results
                    logger.info(f"Basic analysis results structure: {list(analysis_results.keys())}")
                    logger.info(f"Advanced analysis results structure: {list(advanced_results.keys()) if isinstance(advanced_results, dict) else type(advanced_results)}")

                    st.success("✅ Complete analysis finished successfully! Both basic and advanced results are displayed below.")

                    # Force rerun to show results immediately
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    logger.error(f"Complete SI Analysis error: {str(e)}", exc_info=True)

    # Display analysis results if available
    if 'si_analysis_results' in st.session_state and st.session_state.si_analysis_results:
        display_si_analysis_results()

    # Display advanced analysis results if available
    if 'advanced_si_results' in st.session_state:
        if st.session_state.advanced_si_results:
            st.markdown("---")
            st.subheader("🔬 Advanced Analysis Results")
            # Debug information
            if 'error' in st.session_state.advanced_si_results:
                st.error(f"Advanced analysis encountered an error: {st.session_state.advanced_si_results['error']}")
            else:
                display_advanced_analysis_results(st.session_state.advanced_si_results)
        else:
            st.warning("Advanced analysis results are empty. Please try running the analysis again.")

def perform_si_analysis(asr_data, file_name, asr_provider):
    """Perform simultaneous interpretation analysis on ASR data"""
    try:
        # No need to parse combined_word - fields already exist directly in database
        # Just ensure required fields exist
        required_fields = ['edit_word', 'pair_type', 'original_word', 'annotate']
        for field in required_fields:
            if field not in asr_data.columns:
                asr_data[field] = None

        # Separate English and Chinese data
        en_data = asr_data[asr_data['lang'] == 'en'].copy()
        zh_data = asr_data[asr_data['lang'] == 'zh'].copy()

        # Calculate basic statistics
        analysis_results = {
            'file_name': file_name,
            'asr_provider': asr_provider,
            'total_segments': len(asr_data['segment_id'].unique()),
            'total_words': len(asr_data),
            'en_words': len(en_data),
            'zh_words': len(zh_data),
            'analysis_timestamp': datetime.now().isoformat(),
            'detailed_analysis': {}
        }

        # Initialize detailed analysis with default values
        analysis_results['detailed_analysis'] = {
            'evs': {'error': 'Not analyzed'},
            'quality': {'error': 'Not analyzed'},
            'language': {'error': 'Not analyzed'},
            'confidence': {'error': 'Not analyzed'}
        }

        # Time synchronization analysis (EVS)
        try:
            evs_analysis = analyze_time_synchronization(en_data, zh_data)
            analysis_results['detailed_analysis']['evs'] = evs_analysis
            logger.info(f"EVS analysis completed: {len(evs_analysis.get('evs_values', []))} pairs found")
        except Exception as e:
            logger.error(f"EVS analysis failed: {str(e)}")
            analysis_results['detailed_analysis']['evs'] = {'error': f"EVS analysis failed: {str(e)}"}

        # Quality analysis
        try:
            quality_analysis = analyze_interpretation_quality(asr_data, en_data, zh_data)
            analysis_results['detailed_analysis']['quality'] = quality_analysis
            logger.info("Quality analysis completed")
        except Exception as e:
            logger.error(f"Quality analysis failed: {str(e)}")
            analysis_results['detailed_analysis']['quality'] = {'error': f"Quality analysis failed: {str(e)}"}

        # Language distribution analysis
        try:
            lang_analysis = analyze_language_distribution(asr_data)
            analysis_results['detailed_analysis']['language'] = lang_analysis
            logger.info("Language analysis completed")
        except Exception as e:
            logger.error(f"Language analysis failed: {str(e)}")
            analysis_results['detailed_analysis']['language'] = {'error': f"Language analysis failed: {str(e)}"}

        # Confidence analysis
        try:
            confidence_analysis = analyze_confidence_scores(asr_data)
            analysis_results['detailed_analysis']['confidence'] = confidence_analysis
            logger.info("Confidence analysis completed")
        except Exception as e:
            logger.error(f"Confidence analysis failed: {str(e)}")
            analysis_results['detailed_analysis']['confidence'] = {'error': f"Confidence analysis failed: {str(e)}"}

        return analysis_results

    except Exception as e:
        logger.error(f"Error in SI analysis: {str(e)}", exc_info=True)

        # Return a basic structure even on failure
        return {
            'file_name': file_name,
            'asr_provider': asr_provider,
            'total_segments': 0,
            'total_words': 0,
            'en_words': 0,
            'zh_words': 0,
            'analysis_timestamp': datetime.now().isoformat(),
            'detailed_analysis': {
                'evs': {'error': f"Analysis failed: {str(e)}"},
                'quality': {'error': f"Analysis failed: {str(e)}"},
                'language': {'error': f"Analysis failed: {str(e)}"},
                'confidence': {'error': f"Analysis failed: {str(e)}"}
            },
            'error': str(e)
        }

def analyze_time_synchronization(en_data, zh_data):
    """Analyze time synchronization between English and Chinese"""
    try:
        evs_values = []
        pair_details = []  # Store detailed pair information
        analysis_method = 'no_pairs_found'

        # Check if we have both English and Chinese data
        if en_data.empty:
            logger.warning("No English data found for EVS analysis")
            return {
                'mean_evs': 0,
                'median_evs': 0,
                'std_evs': 0,
                'min_evs': 0,
                'max_evs': 0,
                'total_pairs': 0,
                'analysis_method': 'no_english_data',
                'evs_values': [],
                'pair_details': []
            }

        if zh_data.empty:
            logger.warning("No Chinese data found for EVS analysis")
            return {
                'mean_evs': 0,
                'median_evs': 0,
                'std_evs': 0,
                'min_evs': 0,
                'max_evs': 0,
                'total_pairs': 0,
                'analysis_method': 'no_chinese_data',
                'evs_values': [],
                'pair_details': []
            }

        # Method 1: Try to use manually annotated pairs first
        en_paired = en_data[en_data['pair_type'] == 'S']
        zh_paired = zh_data[zh_data['pair_type'] == 'E']

        if not en_paired.empty and not zh_paired.empty:
            logger.info(f"Found manual pairs: {len(en_paired)} English, {len(zh_paired)} Chinese")
            # Calculate EVS for manually paired words
            for _, en_word in en_paired.iterrows():
                try:
                    # Find corresponding Chinese word in the same segment or nearby
                    zh_candidates = zh_paired[
                        (zh_paired['segment_id'] >= en_word['segment_id'] - 1) &
                        (zh_paired['segment_id'] <= en_word['segment_id'] + 1)
                    ]

                    if not zh_candidates.empty:
                        # Take the closest Chinese word by time
                        zh_word = zh_candidates.iloc[
                            (zh_candidates['start_time'] - en_word['start_time']).abs().argmin()
                        ]

                        evs = zh_word['start_time'] - en_word['start_time']
                        evs_values.append(evs)

                        # Store pair details
                        pair_details.append({
                            'en_word': str(en_word['edit_word']) if pd.notna(en_word['edit_word']) else '',
                            'en_time': float(en_word['start_time']),
                            'en_segment': int(en_word['segment_id']),
                            'zh_word': str(zh_word['edit_word']) if pd.notna(zh_word['edit_word']) else '',
                            'zh_time': float(zh_word['start_time']),
                            'zh_segment': int(zh_word['segment_id']),
                            'evs': float(evs),
                            'method': 'manual_pairs'
                        })

                except Exception as e:
                    logger.warning(f"Error processing manual pair: {str(e)}")
                    continue

            if evs_values:
                analysis_method = 'manual_pairs'

        # Method 2: Use LLM for intelligent semantic pairing (full-text processing)
        if len(evs_values) == 0:
            logger.info("No manual pairs found, trying LLM-based semantic pairing with full text")
            logger.info(f"Input data: EN={len(en_data)} words, ZH={len(zh_data)} words")

            # Test LLM connectivity first
            from llm_config import get_active_llm_config
            llm_config = get_active_llm_config()
            logger.info(f"Using LLM provider: {llm_config.get('llm_provider', 'unknown')}")
            logger.info(f"LLM model: {llm_config.get('llm_model', 'unknown')}")

            try:
                llm_pairs = create_llm_based_pairs(en_data, zh_data)
                logger.info(f"LLM pairing returned: {len(llm_pairs)} pairs")

                if llm_pairs:
                    for i, pair in enumerate(llm_pairs):
                        evs_values.append(pair['evs'])
                        pair_details.append(pair)
                        logger.debug(f"EVS pair {i+1}: {pair['en_word']}({pair['en_time']:.3f}s) -> {pair['zh_word']}({pair['zh_time']:.3f}s) = {pair['evs']:.3f}s")
                    analysis_method = 'llm_semantic'
                    logger.info(f"LLM created {len(llm_pairs)} semantic pairs")
                else:
                    logger.warning("LLM pairing returned no pairs")
                    # Add diagnostic information
                    logger.warning(f"Possible issues:")
                    logger.warning(f"1. LLM server not responding")
                    logger.warning(f"2. LLM response parsing failed")
                    logger.warning(f"3. All pairs filtered out due to low confidence")
                    logger.warning(f"4. JSON parsing issues")
            except Exception as e:
                logger.error(f"LLM pairing failed: {str(e)}", exc_info=True)

        # Calculate statistics if we have EVS values
        if evs_values:
            logger.info(f"EVS analysis completed with {len(evs_values)} pairs using method: {analysis_method}")
            return {
                'mean_evs': float(np.mean(evs_values)),
                'median_evs': float(np.median(evs_values)),
                'std_evs': float(np.std(evs_values)),
                'min_evs': float(np.min(evs_values)),
                'max_evs': float(np.max(evs_values)),
                'total_pairs': len(evs_values),
                'analysis_method': analysis_method,
                'evs_values': [float(x) for x in evs_values],
                'pair_details': pair_details  # Add detailed pair information
            }
        else:
            logger.warning("No EVS pairs could be created")
            return {
                'mean_evs': 0,
                'median_evs': 0,
                'std_evs': 0,
                'min_evs': 0,
                'max_evs': 0,
                'total_pairs': 0,
                'analysis_method': 'no_pairs_found',
                'evs_values': [],
                'pair_details': []
            }

    except Exception as e:
        logger.error(f"Error in EVS analysis: {str(e)}", exc_info=True)
        return {
            'mean_evs': 0,
            'median_evs': 0,
            'std_evs': 0,
            'min_evs': 0,
            'max_evs': 0,
            'total_pairs': 0,
            'analysis_method': 'error',
            'evs_values': [],
            'pair_details': [],
            'error': str(e)
        }

def analyze_interpretation_quality(asr_data, en_data, zh_data):
    """Analyze interpretation quality"""
    try:
        # Calculate word rate (words per minute)
        if not asr_data.empty:
            # Check if end_time column exists, if not use start_time + duration or create it
            if 'end_time' not in asr_data.columns:
                logger.warning("end_time column not found, attempting to calculate from start_time and duration")
                if 'duration' in asr_data.columns:
                    asr_data = asr_data.copy()
                    asr_data['end_time'] = asr_data['start_time'] + asr_data['duration']
                    logger.info("Successfully calculated end_time from start_time + duration")
                else:
                    logger.warning("Neither end_time nor duration columns found, using start_time as approximation")
                    asr_data = asr_data.copy()
                    asr_data['end_time'] = asr_data['start_time'] + 1.0  # Default 1 second duration

            # Now calculate total duration
            total_duration = asr_data['end_time'].max() - asr_data['start_time'].min()
            if total_duration > 0:
                en_wpm = len(en_data) / (total_duration / 60) if total_duration > 0 else 0
                zh_wpm = len(zh_data) / (total_duration / 60) if total_duration > 0 else 0
            else:
                en_wpm = zh_wpm = 0
        else:
            en_wpm = zh_wpm = 0

        # Calculate coverage (percentage of segments with both languages)
        segments_with_both = 0
        all_segments = asr_data['segment_id'].unique()

        for segment in all_segments:
            try:
                segment_data = asr_data[asr_data['segment_id'] == segment]
                has_en = 'en' in segment_data['lang'].values
                has_zh = 'zh' in segment_data['lang'].values
                if has_en and has_zh:
                    segments_with_both += 1
            except Exception as e:
                logger.warning(f"Error processing segment {segment}: {str(e)}")
                continue

        coverage_rate = (segments_with_both / len(all_segments)) * 100 if len(all_segments) > 0 else 0

        return {
            'english_wpm': float(en_wpm),
            'chinese_wpm': float(zh_wpm),
            'coverage_rate': float(coverage_rate),
            'total_segments': int(len(all_segments)),
            'bilingual_segments': int(segments_with_both)
        }

    except Exception as e:
        logger.error(f"Error in quality analysis: {str(e)}", exc_info=True)
        return {
            'english_wpm': 0.0,
            'chinese_wpm': 0.0,
            'coverage_rate': 0.0,
            'total_segments': 0,
            'bilingual_segments': 0,
            'error': str(e)
        }

def analyze_language_distribution(asr_data):
    """Analyze language distribution"""
    try:
        if asr_data.empty:
            return {
                'english_count': 0,
                'chinese_count': 0,
                'english_percentage': 0.0,
                'chinese_percentage': 0.0,
                'total_words': 0
            }

        lang_counts = asr_data['lang'].value_counts()
        total_words = len(asr_data)

        english_count = int(lang_counts.get('en', 0))
        chinese_count = int(lang_counts.get('zh', 0))

        return {
            'english_count': english_count,
            'chinese_count': chinese_count,
            'english_percentage': float((english_count / total_words) * 100) if total_words > 0 else 0.0,
            'chinese_percentage': float((chinese_count / total_words) * 100) if total_words > 0 else 0.0,
            'total_words': int(total_words)
        }

    except Exception as e:
        logger.error(f"Error in language analysis: {str(e)}", exc_info=True)
        return {
            'english_count': 0,
            'chinese_count': 0,
            'english_percentage': 0.0,
            'chinese_percentage': 0.0,
            'total_words': 0,
            'error': str(e)
        }

def analyze_confidence_scores(asr_data):
    """Analyze confidence scores"""
    try:
        # Check if confidence column exists
        if 'confidence' not in asr_data.columns:
            logger.info("No confidence column found in data")
            return {
                'overall': {
                    'mean_confidence': 0.0,
                    'median_confidence': 0.0,
                    'std_confidence': 0.0,
                    'min_confidence': 0.0,
                    'max_confidence': 0.0
                },
                'by_language': {
                    'en': {'mean': 0.0, 'count': 0},
                    'zh': {'mean': 0.0, 'count': 0}
                },
                'total_with_confidence': 0,
                'error': 'Confidence column not available in data'
            }

        # Filter out null confidence values
        confidence_data = asr_data[pd.notna(asr_data['confidence'])]

        if not confidence_data.empty:
            overall_stats = {
                'mean_confidence': float(confidence_data['confidence'].mean()),
                'median_confidence': float(confidence_data['confidence'].median()),
                'std_confidence': float(confidence_data['confidence'].std()),
                'min_confidence': float(confidence_data['confidence'].min()),
                'max_confidence': float(confidence_data['confidence'].max())
            }

            # Language-specific confidence
            lang_confidence = {}
            for lang in ['en', 'zh']:
                try:
                    lang_data = confidence_data[confidence_data['lang'] == lang]
                    if not lang_data.empty:
                        lang_confidence[lang] = {
                            'mean': float(lang_data['confidence'].mean()),
                            'count': int(len(lang_data))
                        }
                    else:
                        lang_confidence[lang] = {'mean': 0.0, 'count': 0}
                except Exception as e:
                    logger.warning(f"Error processing {lang} confidence: {str(e)}")
                    lang_confidence[lang] = {'mean': 0.0, 'count': 0}

            return {
                'overall': overall_stats,
                'by_language': lang_confidence,
                'total_with_confidence': int(len(confidence_data))
            }
        else:
            return {
                'overall': {
                    'mean_confidence': 0.0,
                    'median_confidence': 0.0,
                    'std_confidence': 0.0,
                    'min_confidence': 0.0,
                    'max_confidence': 0.0
                },
                'by_language': {
                    'en': {'mean': 0.0, 'count': 0},
                    'zh': {'mean': 0.0, 'count': 0}
                },
                'total_with_confidence': 0
            }

    except Exception as e:
        logger.error(f"Error in confidence analysis: {str(e)}", exc_info=True)
        return {
            'overall': {
                'mean_confidence': 0.0,
                'median_confidence': 0.0,
                'std_confidence': 0.0,
                'min_confidence': 0.0,
                'max_confidence': 0.0
            },
            'by_language': {
                'en': {'mean': 0.0, 'count': 0},
                'zh': {'mean': 0.0, 'count': 0}
            },
            'total_with_confidence': 0,
            'error': str(e)
        }

def display_si_analysis_results():
    """Display SI analysis results"""
    try:
        # Validate results exist and have correct structure
        if 'si_analysis_results' not in st.session_state:
            st.error("No analysis results found in session state")
            return

        results = st.session_state.si_analysis_results

        # Debug: Validate results structure
        if not isinstance(results, dict):
            st.error(f"Analysis results is not a dictionary: {type(results)}")
            return

        # Check required keys
        required_keys = ['file_name', 'asr_provider', 'analysis_timestamp', 'detailed_analysis']
        missing_keys = [key for key in required_keys if key not in results]
        if missing_keys:
            st.error(f"Missing required keys in analysis results: {missing_keys}")
            st.write("Available keys:", list(results.keys()))
            return

        # Validate detailed_analysis structure
        detailed_analysis = results.get('detailed_analysis', {})
        if not isinstance(detailed_analysis, dict):
            st.error(f"detailed_analysis is not a dictionary: {type(detailed_analysis)}")
            return

        # Log detailed analysis structure for debugging
        logger.info(f"Detailed analysis keys: {list(detailed_analysis.keys())}")
        for key, value in detailed_analysis.items():
            if isinstance(value, dict):
                logger.info(f"  {key}: {list(value.keys())}")
            else:
                logger.info(f"  {key}: {type(value)}")

        # Display header information
        display_name = get_file_display_name(results['file_name'])
        st.subheader(f"Analysis Results: {display_name}")
        st.write(f"**ASR Provider:** {results['asr_provider']}")
        # Handle analysis_timestamp - it might be a string or datetime object
        analysis_time = results['analysis_timestamp']
        if isinstance(analysis_time, str):
            # If it's already a string, display it directly
            st.write(f"**Analysis Time:** {analysis_time}")
        else:
            # If it's a datetime object, format it
            try:
                st.write(f"**Analysis Time:** {analysis_time.strftime('%Y-%m-%d %H:%M:%S')}")
            except AttributeError:
                # Fallback if it's neither string nor datetime
                st.write(f"**Analysis Time:** {str(analysis_time)}")

        # Create tabs for different analysis types
        analysis_tabs = st.tabs([
            "Overview",
            "Time Synchronization",
            "Quality Metrics",
            "Language Distribution",
            "Confidence Analysis"
        ])

        # Overview tab
        with analysis_tabs[0]:
            st.subheader("Analysis Overview")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Segments", results.get('total_segments', 0))
            with col2:
                st.metric("Total Words", results.get('total_words', 0))
            with col3:
                st.metric("English Words", results.get('en_words', 0))
            with col4:
                st.metric("Chinese Words", results.get('zh_words', 0))

        # Time Synchronization tab
        with analysis_tabs[1]:
            st.subheader("Time Synchronization Analysis (EVS)")

            # Multiple checks for EVS data
            try:
                # Check if detailed analysis exists and has EVS data
                if 'detailed_analysis' not in results:
                    st.error("No detailed analysis found in results")
                    return

                if 'evs' not in results['detailed_analysis']:
                    st.error("No EVS analysis found in detailed analysis")
                    st.write("Available analysis types:", list(results['detailed_analysis'].keys()))
                    return

                evs_data = results['detailed_analysis']['evs']

                # Validate EVS data structure
                if not isinstance(evs_data, dict):
                    st.error(f"EVS data is not a dictionary: {type(evs_data)}")
                    return

                logger.info(f"EVS data structure: {list(evs_data.keys())}")

                if 'error' in evs_data:
                    st.error(f"EVS Analysis Error: {evs_data.get('error', 'Unknown error')}")
                    return

                # Rest of EVS analysis display...
                # Display analysis method
                method_info = {
                    'manual_pairs': '基于手动标注的配对词分析 - 最高精度',
                    'llm_semantic': 'LLM智能语义全文配对分析 - 自动识别语义对应关系，处理完整转录文本',
                    'no_pairs_found': '未找到可分析的配对 - 建议配置LLM或手动标注'
                }

                analysis_method = evs_data.get('analysis_method', 'unknown')
                method_desc = method_info.get(analysis_method, f'未知方法: {analysis_method}')

                if analysis_method == 'llm_semantic':
                    st.info(f"**分析方法:** {method_desc}")
                    st.success("🤖 **LLM全文分析特点:**\n- 智能动态批次大小（最大30K tokens）\n- 优先全文处理避免配对丢失\n- 智能识别语义对应关系\n- 考虑上下文连贯性和专业术语\n- 自动适配不同LLM提供商\n- 自动过滤不合理的EVS值")
                else:
                    st.info(f"**分析方法:** {method_desc}")

                if evs_data.get('total_pairs', 0) > 0:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean EVS", f"{evs_data.get('mean_evs', 0):.3f}s")
                        st.metric("Median EVS", f"{evs_data.get('median_evs', 0):.3f}s")
                    with col2:
                        st.metric("Std Deviation", f"{evs_data.get('std_evs', 0):.3f}s")
                        st.metric("Min EVS", f"{evs_data.get('min_evs', 0):.3f}s")
                    with col3:
                        st.metric("Max EVS", f"{evs_data.get('max_evs', 0):.3f}s")
                        st.metric("Total Pairs", evs_data.get('total_pairs', 0))

                    # EVS quality assessment
                    mean_evs = abs(evs_data.get('mean_evs', 0))  # Use absolute value for assessment
                    if mean_evs <= 2.0:
                        evs_quality = "Excellent"
                        quality_color = "green"
                    elif mean_evs <= 3.0:
                        evs_quality = "Good"
                        quality_color = "blue"
                    elif mean_evs <= 5.0:
                        evs_quality = "Acceptable"
                        quality_color = "orange"
                    else:
                        evs_quality = "Needs Improvement"
                        quality_color = "red"

                    st.markdown(f"**EVS Quality Assessment:** :{quality_color}[{evs_quality}]")

                    # Add EVS distribution visualization
                    if evs_data.get('evs_values') and len(evs_data['evs_values']) > 1:
                        st.subheader("EVS Distribution")

                        # Create histogram
                        fig = go.Figure(data=[
                            go.Histogram(
                                x=evs_data['evs_values'],
                                nbinsx=min(20, len(evs_data['evs_values'])),
                                name='EVS Distribution',
                                marker_color='skyblue',
                                opacity=0.7
                            )
                        ])

                        # Add mean line
                        fig.add_vline(
                            x=evs_data['mean_evs'],
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Mean: {evs_data['mean_evs']:.3f}s"
                        )

                        # Add median line
                        fig.add_vline(
                            x=evs_data['median_evs'],
                            line_dash="dot",
                            line_color="green",
                            annotation_text=f"Median: {evs_data['median_evs']:.3f}s"
                        )

                        fig.update_layout(
                            title='EVS Value Distribution',
                            xaxis_title='EVS (seconds)',
                            yaxis_title='Frequency',
                            showlegend=False,
                            height=400
                        )

                        st.plotly_chart(fig, width='stretch')

                        # Add interpretation guide
                        st.info("""
                        **EVS Value Interpretation:**
                        - **Positive**: Chinese starts after English (normal SI mode)
                        - **Negative**: Chinese starts before English (possible prediction or overlap)
                        - **Near 0**: Almost simultaneous start
                        """)

                    # Additional insights based on analysis method
                    if analysis_method == 'llm_semantic':
                        st.success("✅ Using LLM semantic analysis for full-text matching. This is the most accurate analysis method, able to identify semantic correspondences and consider context continuity.")
                    elif analysis_method == 'manual_pairs':
                        st.success("✅ Using manual pair analysis for precise analysis.")

                    # Show pair details
                    if evs_data.get('pair_details'):
                        st.subheader("Pair Vocabulary Details")

                        # Convert pair details to DataFrame for display
                        try:
                            # Validate pair_details structure first
                            if not evs_data['pair_details']:
                                st.warning("Pair details are empty")
                                return

                            # Debug: Log the structure of pair_details
                            logger.info(f"Processing {len(evs_data['pair_details'])} pair details")
                            sample_pair = evs_data['pair_details'][0] if evs_data['pair_details'] else {}
                            logger.info(f"Sample pair structure: {list(sample_pair.keys()) if sample_pair else 'No pairs'}")

                            pairs_df = pd.DataFrame(evs_data['pair_details'])

                            # Validate that the DataFrame has required columns
                            required_columns = ['en_word', 'en_time', 'zh_word', 'zh_time', 'evs', 'method']
                            missing_columns = [col for col in required_columns if col not in pairs_df.columns]

                            if missing_columns:
                                st.error(f"Pair data is missing required columns: {missing_columns}")
                                st.write("Available columns:", list(pairs_df.columns))
                                return

                            # Ensure 'evs' column exists and is numeric
                            if 'evs' not in pairs_df.columns:
                                st.error("Pair data is missing 'evs' column")
                                st.write("Available columns:", list(pairs_df.columns))
                                return

                            # Convert EVS to numeric, handling any non-numeric values
                            pairs_df['evs'] = pd.to_numeric(pairs_df['evs'], errors='coerce')

                            # Remove rows with NaN EVS values
                            nan_count = pairs_df['evs'].isna().sum()
                            if nan_count > 0:
                                st.warning(f"Found {nan_count} invalid EVS values, ignored")
                                pairs_df = pairs_df.dropna(subset=['evs'])

                            if pairs_df.empty:
                                st.warning("No valid pair data after processing")
                                return

                            # Show LLM confidence if available
                            if 'confidence' in pairs_df.columns and analysis_method == 'llm_semantic':
                                avg_confidence = pairs_df['confidence'].mean()
                                st.metric("LLM Pair Average Confidence", f"{avg_confidence:.2f}")

                            # Check for abnormal EVS values
                            abnormal_pairs = pairs_df[abs(pairs_df['evs']) > 10]  # EVS > 10 seconds considered abnormal

                            if not abnormal_pairs.empty:
                                st.error(f"⚠️ Found {len(abnormal_pairs)} abnormal EVS values (>10 seconds), which may indicate incorrect pairings.")

                                # Add LLM improvement option
                                col1, col2 = st.columns([1, 1])
                                with col1:
                                    if st.button("Use LLM to Improve Analysis", type="primary", key="llm_improve_evs"):
                                        with st.spinner("Using LLM to analyze and improve pairings..."):
                                            try:
                                                improved_results = improve_evs_analysis_with_llm(
                                                    evs_data['pair_details'],
                                                    st.session_state.si_analysis_file_name
                                                )
                                                if improved_results:
                                                    st.session_state.llm_improved_evs = improved_results
                                                    st.success("LLM analysis completed! See the improved suggestions below.")
                                            except Exception as e:
                                                st.error(f"LLM analysis failed: {str(e)}")

                                with col2:
                                    if st.button("Show Abnormal Pairs", key="show_abnormal_pairs"):
                                        st.session_state.show_abnormal_pairs = True

                            # Display pair details table
                            display_df = pairs_df.copy()

                            # Safely format columns
                            try:
                                display_df['evs_formatted'] = display_df['evs'].apply(lambda x: f"{x:.3f}s" if pd.notna(x) else "N/A")
                                display_df['en_time_formatted'] = display_df['en_time'].apply(lambda x: f"{x:.3f}s" if pd.notna(x) else "N/A")
                                display_df['zh_time_formatted'] = display_df['zh_time'].apply(lambda x: f"{x:.3f}s" if pd.notna(x) else "N/A")
                            except Exception as format_error:
                                st.error(f"格式化数据时出错: {str(format_error)}")
                                logger.error(f"Formatting error: {str(format_error)}", exc_info=True)
                                return

                            # Color code abnormal EVS values
                            def highlight_abnormal_evs(row):
                                try:
                                    evs_val = row['evs']
                                    if pd.isna(evs_val):
                                        return ['background-color: #f0f0f0'] * len(row)
                                    elif abs(evs_val) > 10:
                                        return ['background-color: #ffcccc'] * len(row)
                                    elif abs(evs_val) > 5:
                                        return ['background-color: #fff2cc'] * len(row)
                                    else:
                                        return [''] * len(row)
                                except Exception as e:
                                    logger.warning(f"Error in highlight function: {str(e)}")
                                    return [''] * len(row)

                            # Select columns that exist in the DataFrame
                            display_columns = []
                            column_mapping = {
                                'en_word': 'en_word',
                                'en_time_formatted': 'en_time_formatted',
                                'zh_word': 'zh_word',
                                'zh_time_formatted': 'zh_time_formatted',
                                'evs_formatted': 'evs_formatted',
                                'method': 'method'
                            }

                            # Add confidence column if available
                            if 'confidence' in pairs_df.columns:
                                column_mapping['confidence'] = 'confidence'

                            for col in column_mapping:
                                if col in display_df.columns:
                                    display_columns.append(col)
                                else:
                                    st.warning(f"列 '{col}' 不存在于数据中")

                            if display_columns:
                                styled_df = display_df[display_columns].style.apply(highlight_abnormal_evs, axis=1)

                                column_config = {
                                    'en_word': st.column_config.TextColumn('English Word', width='medium'),
                                    'en_time_formatted': st.column_config.TextColumn('EN Time', width='small'),
                                    'zh_word': st.column_config.TextColumn('Chinese Word', width='medium'),
                                    'zh_time_formatted': st.column_config.TextColumn('ZH Time', width='small'),
                                    'evs_formatted': st.column_config.TextColumn('EVS', width='small'),
                                    'method': st.column_config.TextColumn('Method', width='small')
                                }

                                # Add confidence column config if available
                                if 'confidence' in display_columns:
                                    column_config['confidence'] = st.column_config.NumberColumn('Confidence', format="%.2f", width='small')

                                st.dataframe(
                                    styled_df,
                                    column_config=column_config,
                                    hide_index=True,
                                    width='stretch'
                                )
                            else:
                                st.error("No columns to display")

                            # Show abnormal pairs if requested
                            if st.session_state.get('show_abnormal_pairs', False):
                                st.subheader("Abnormal Pair Details")
                                if not abnormal_pairs.empty:
                                    abnormal_display_columns = ['en_word', 'en_time', 'zh_word', 'zh_time', 'evs']
                                    available_abnormal_columns = [col for col in abnormal_display_columns if col in abnormal_pairs.columns]

                                    if available_abnormal_columns:
                                        st.dataframe(
                                            abnormal_pairs[available_abnormal_columns],
                                            column_config={
                                                'en_word': 'English Word',
                                                'en_time': st.column_config.NumberColumn('EN Time (s)', format="%.3f"),
                                                'zh_word': 'Chinese Word',
                                                'zh_time': st.column_config.NumberColumn('ZH Time (s)', format="%.3f"),
                                                'evs': st.column_config.NumberColumn('EVS (s)', format="%.3f")
                                            },
                                            hide_index=True
                                        )
                                    else:
                                        st.error("Abnormal pair data format is incorrect")
                                else:
                                    st.info("No abnormal pairs found.")

                        except Exception as e:
                            st.error(f"Error processing pair details: {str(e)}")
                            logger.error(f"Error processing pair details: {str(e)}", exc_info=True)

                            # Show debugging information
                            if st.checkbox("Show Debug Information", key="show_pair_debug"):
                                st.write("**Pair Details Structure:**")
                                if 'pair_details' in evs_data:
                                    st.write(f"Number of pairs: {len(evs_data['pair_details'])}")
                                    if evs_data['pair_details']:
                                        st.write("Sample pair:")
                                        st.json(evs_data['pair_details'][0])
                                else:
                                    st.write("No pair_details found")

                    # Display LLM analysis results if available
                    if 'llm_improved_evs' in st.session_state and st.session_state.llm_improved_evs:
                        st.subheader("LLM Analysis and Improvement Suggestions")
                        llm_results = st.session_state.llm_improved_evs

                        if 'analysis' in llm_results:
                            st.write("**Analysis Report:**")
                            st.write(llm_results['analysis'])

                        if 'recommendations' in llm_results:
                            st.write("**Improvement Suggestions:**")
                            for i, rec in enumerate(llm_results['recommendations'], 1):
                                st.write(f"{i}. {rec}")

                        if 'filtered_pairs' in llm_results:
                            st.write("**LLM Recommended Valid Pairs:**")
                            filtered_df = pd.DataFrame(llm_results['filtered_pairs'])
                            if not filtered_df.empty:
                                st.dataframe(
                                    filtered_df,
                                    column_config={
                                        'en_word': 'English Word',
                                        'zh_word': 'Chinese Word',
                                        'evs': st.column_config.NumberColumn('EVS (s)', format="%.3f"),
                                        'confidence': st.column_config.NumberColumn('Confidence', format="%.2f")
                                    },
                                    hide_index=True
                                )
                else:
                    st.warning("No valid English-Chinese pairs found. Please check:")
                    st.write("1. Whether English and Chinese content is included in the data")
                    st.write("2. Whether vocabulary pairs have been annotated in the 'Annotate EVS' tab")
                    st.write("3. Whether the segment_id and timestamps in the data are correct")

            except Exception as evs_error:
                st.error(f"Error in EVS analysis section: {str(evs_error)}")
                logger.error(f"Error in EVS analysis section: {str(evs_error)}", exc_info=True)

                # Show EVS data structure for debugging
                if st.checkbox("Show EVS Debug Information", key="show_evs_debug"):
                    st.write("**EVS Data Structure:**")
                    if 'detailed_analysis' in results and 'evs' in results['detailed_analysis']:
                        st.json(results['detailed_analysis']['evs'])
                    else:
                        st.write("No EVS data found")
                        st.write("Available detailed analysis:", list(results.get('detailed_analysis', {}).keys()))

        # Quality Metrics tab
        with analysis_tabs[2]:
            st.subheader("Interpretation Quality Metrics")

            # Add introduction and explanation
            with st.expander("📖 About Quality Metrics", expanded=False):
                st.markdown("""
                **Quality Metrics** provides comprehensive analysis of simultaneous interpretation performance:

                **🎯 Core Indicators:**
                - **WPM (Words Per Minute)**: Measures speech rate for both source and target languages
                - **Segment Analysis**: Evaluates completeness of interpretation across audio segments
                - **Coverage Rate**: Percentage of segments containing both source and target languages

                **📊 Performance Evaluation:**
                - **Fluency Assessment**: Compares English-Chinese speech rate differences
                - **Completeness Analysis**: Identifies potential gaps or omissions in interpretation
                - **Workload Monitoring**: High WPM may indicate intensive interpretation periods
                - **Quality Benchmarking**: Provides objective performance metrics for interpreters

                **💡 Practical Applications:**
                - Performance benchmarking for interpreter evaluation
                - Training reference with quantitative indicators
                - Real-time quality monitoring during interpretation
                - Identifying areas for improvement in speed and coverage
                """)

            # Check if quality analysis exists
            if 'detailed_analysis' in results and 'quality' in results['detailed_analysis']:
                quality_data = results['detailed_analysis']['quality']

                if 'error' not in quality_data:
                    # Core metrics display
                    st.subheader("📈 Speech Rate Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        english_wpm = quality_data.get('english_wpm', 0)
                        st.metric("English WPM", f"{english_wpm:.1f}",
                                help="Words per minute in English (source language)")

                        # WPM interpretation
                        if english_wpm > 0:
                            if english_wpm < 120:
                                st.info("🟢 **Normal pace** - Comfortable speaking rate")
                            elif english_wpm < 150:
                                st.warning("🟡 **Moderate pace** - Standard professional rate")
                            else:
                                st.error("🔴 **Fast pace** - High-intensity speaking rate")

                    with col2:
                        chinese_wpm = quality_data.get('chinese_wpm', 0)
                        st.metric("Chinese WPM", f"{chinese_wpm:.1f}",
                                help="Words per minute in Chinese (target language)")

                        # Speed comparison analysis
                        if english_wpm > 0 and chinese_wpm > 0:
                            speed_ratio = (chinese_wpm / english_wpm) * 100
                            if speed_ratio >= 90:
                                st.success(f"🟢 **Speed Rate Balance: Excellent balance** ({speed_ratio:.1f}% ratio)")
                            elif speed_ratio >= 75:
                                st.info(f"🟡 **Speed Rate Balance: Good balance** ({speed_ratio:.1f}% ratio)")
                            else:
                                st.warning(f"🟠 **Speed Rate Balance: Speed gap detected** ({speed_ratio:.1f}% ratio)")

                    st.subheader("🎯 Segment Coverage Analysis")
                    col3, col4 = st.columns(2)
                    with col3:
                        st.metric("Total Segments", quality_data.get('total_segments', 0),
                                help="Total number of audio segments analyzed")
                    with col4:
                        st.metric("Bilingual Segments", quality_data.get('bilingual_segments', 0),
                                help="Segments containing both English and Chinese")

                    # Coverage rate with interpretation
                    coverage_rate = quality_data.get('coverage_rate', 0)
                    st.metric("Coverage Rate", f"{coverage_rate:.1f}%",
                            help="Percentage of segments with successful interpretation")

                    # Coverage interpretation
                    if coverage_rate >= 80:
                        st.success("🟢 **Excellent coverage** - High interpretation completeness")
                    elif coverage_rate >= 60:
                        st.info("🟡 **Good coverage** - Acceptable interpretation level")
                    elif coverage_rate >= 40:
                        st.warning("🟠 **Moderate coverage** - Room for improvement")
                    else:
                        st.error("🔴 **Low coverage** - Significant gaps in interpretation")

                    # Create visualization for coverage
                    total_segments = quality_data.get('total_segments', 0)
                    bilingual_segments = quality_data.get('bilingual_segments', 0)

                    if total_segments > 0:
                        st.subheader("📊 Segment Coverage Visualization")

                        fig = go.Figure(data=[
                            go.Bar(
                                x=['Bilingual Segments', 'Monolingual Segments'],
                                y=[bilingual_segments, total_segments - bilingual_segments],
                                marker_color=['#2E8B57', '#FF8C00'],  # Sea Green and Dark Orange
                                text=[f"{bilingual_segments}<br>({(bilingual_segments/total_segments*100):.1f}%)",
                                     f"{total_segments - bilingual_segments}<br>({((total_segments - bilingual_segments)/total_segments*100):.1f}%)"],
                                textposition='auto',
                                textfont=dict(color='white', size=12)
                            )
                        ])

                        fig.update_layout(
                            title='Segment Coverage Analysis',
                            yaxis_title='Number of Segments',
                            showlegend=False,
                            height=400
                        )

                        st.plotly_chart(fig, width='stretch')

                        # Analysis summary
                        st.subheader("📋 Quality Assessment Summary")
                        col5, col6 = st.columns(2)
                        with col5:
                            st.markdown("**🎯 Performance Indicators:**")
                            st.write(f"• Speech rate balance: {((chinese_wpm / english_wpm) * 100):.1f}%" if english_wpm > 0 else "• Speech rate balance: N/A")
                            st.write(f"• Interpretation coverage: {coverage_rate:.1f}%")
                            st.write(f"• Total processing time: ~{(total_segments * 30):.0f} seconds" if total_segments > 0 else "• Total processing time: N/A")

                        with col6:
                            st.markdown("**💡 Recommendations:**")
                            if coverage_rate < 60:
                                st.write("• Focus on improving interpretation completeness")
                            if abs(english_wpm - chinese_wpm) > 20:
                                st.write("• Work on maintaining consistent speech pace")
                            if english_wpm > 150:
                                st.write("• Consider slowing down source speech rate")
                            if coverage_rate >= 80 and abs(english_wpm - chinese_wpm) <= 15:
                                st.write("• 🎉 Excellent interpretation performance!")

                else:
                    st.error(f"Quality Analysis Error: {quality_data.get('error', 'Unknown error')}")
            else:
                st.error("Quality analysis data is not available.")

        # Language Distribution tab
        with analysis_tabs[3]:
            st.subheader("Language Distribution")

            # Check if language analysis exists
            if 'detailed_analysis' in results and 'language' in results['detailed_analysis']:
                lang_data = results['detailed_analysis']['language']

                if 'error' not in lang_data:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("English Words", f"{lang_data.get('english_count', 0)} ({lang_data.get('english_percentage', 0):.1f}%)")
                    with col2:
                        st.metric("Chinese Words", f"{lang_data.get('chinese_count', 0)} ({lang_data.get('chinese_percentage', 0):.1f}%)")

                    # Create pie chart for language distribution
                    english_count = lang_data.get('english_count', 0)
                    chinese_count = lang_data.get('chinese_count', 0)

                    if english_count > 0 or chinese_count > 0:
                        fig = go.Figure(data=[go.Pie(
                            labels=['English', 'Chinese'],
                            values=[english_count, chinese_count],
                            hole=.3
                        )])

                        fig.update_layout(title="Language Distribution")
                        st.plotly_chart(fig, width='stretch')
                else:
                    st.error(f"Language Analysis Error: {lang_data.get('error', 'Unknown error')}")
            else:
                st.error("Language distribution analysis data is not available.")

        # Confidence Analysis tab
        with analysis_tabs[4]:
            st.subheader("Confidence Score Analysis")

            # Check if confidence analysis exists
            if 'detailed_analysis' in results and 'confidence' in results['detailed_analysis']:
                conf_data = results['detailed_analysis']['confidence']

                if 'error' not in conf_data:
                    # Overall confidence metrics
                    overall = conf_data.get('overall', {})
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean Confidence", f"{overall.get('mean_confidence', 0):.3f}")
                        st.metric("Min Confidence", f"{overall.get('min_confidence', 0):.3f}")
                    with col2:
                        st.metric("Median Confidence", f"{overall.get('median_confidence', 0):.3f}")
                        st.metric("Max Confidence", f"{overall.get('max_confidence', 0):.3f}")
                    with col3:
                        st.metric("Std Deviation", f"{overall.get('std_confidence', 0):.3f}")
                        st.metric("Words with Confidence", conf_data.get('total_with_confidence', 0))

                    # Language-specific confidence
                    st.subheader("Confidence by Language")
                    by_lang = conf_data.get('by_language', {})

                    col1, col2 = st.columns(2)
                    with col1:
                        en_data = by_lang.get('en', {})
                        st.metric("English Confidence", f"{en_data.get('mean', 0):.3f}")
                        st.metric("English Words", en_data.get('count', 0))
                    with col2:
                        zh_data = by_lang.get('zh', {})
                        st.metric("Chinese Confidence", f"{zh_data.get('mean', 0):.3f}")
                        st.metric("Chinese Words", zh_data.get('count', 0))

                    # Create bar chart for language comparison
                    en_mean = by_lang.get('en', {}).get('mean', 0)
                    zh_mean = by_lang.get('zh', {}).get('mean', 0)

                    if en_mean > 0 or zh_mean > 0:
                        fig = go.Figure(data=[
                            go.Bar(
                                x=['English', 'Chinese'],
                                y=[en_mean, zh_mean],
                                marker_color=['blue', 'red'],
                                text=[f"{en_mean:.3f}", f"{zh_mean:.3f}"],
                                textposition='auto'
                            )
                        ])

                        fig.update_layout(
                            title='Average Confidence by Language',
                            yaxis_title='Confidence Score',
                            yaxis=dict(range=[0, 1]),
                            showlegend=False
                        )

                        st.plotly_chart(fig, width='stretch')
                else:
                    st.error(f"Confidence Analysis Error: {conf_data.get('error', 'Unknown error')}")
            else:
                st.error("Confidence score analysis data is not available.")

        # Save analysis results button
        if st.button("Save Analysis Results", type="primary", key="save_si_analysis"):
            try:
                # Save to database (implement this function in EVSDataUtils)
                success = save_si_analysis_results(results)
                if success:
                    st.success("Analysis results saved successfully!")
                else:
                    st.error("Failed to save analysis results")
            except Exception as e:
                st.error(f"Error saving results: {str(e)}")

    except Exception as e:
        st.error(f"Error displaying analysis results: {str(e)}")
        logger.error(f"Error displaying SI analysis results: {str(e)}", exc_info=True)

        # Show debug information for troubleshooting
        if st.checkbox("Show Debug Information", key="show_debug_info"):
            st.write("**Current Result Structure:**")
            if 'si_analysis_results' in st.session_state:
                st.json(st.session_state.si_analysis_results)

def save_si_analysis_results(results):
    """Save SI analysis results to database with file_name and asr_provider as unique key"""
    try:
        # Get alias for the file name for privacy protection
        original_file_name = results['file_name']
        alias_manager = st.session_state.file_alias_manager
        alias_file_name = alias_manager.get_or_create_alias(original_file_name)

        logger.info(f"Attempting to save SI analysis results for {original_file_name} (alias: {alias_file_name}) with ASR provider {results['asr_provider']}")

        # Check if analysis results already exist for this file and ASR provider (using alias)
        existing_results = EVSDataUtils.get_si_analysis_results(
            file_name=alias_file_name,
            asr_provider=results['asr_provider'],
            analysis_type='complete_analysis',
            limit=1
        )

        # If results exist, delete them first to avoid duplicates
        if not existing_results.empty:
            logger.info(f"Found existing analysis results, deleting before saving new ones")
            for _, row in existing_results.iterrows():
                EVSDataUtils.delete_si_analysis_result(row['id'])

        # Prepare analysis data for database storage
        # Create a copy of results with alias file name for privacy protection
        results_with_alias = results.copy()
        results_with_alias['file_name'] = alias_file_name

        analysis_data = {
            'total_segments': results.get('total_segments', 0),
            'total_words': results.get('total_words', 0),
            'en_words': results.get('en_words', 0),
            'zh_words': results.get('zh_words', 0),
            'analysis_timestamp': results.get('analysis_timestamp'),
            'detailed_analysis': results.get('detailed_analysis', {}),
            'analysis_results': results_with_alias  # Store complete results with alias file name
        }

        # Ensure total_segments is correctly passed from the updated results
        if 'total_segments' in results:
            analysis_data['total_segments'] = results['total_segments']

        # Focus on Quality Metrics - these are verified and reliable
        quality_data = results.get('detailed_analysis', {}).get('quality', {})
        if 'error' not in quality_data:
            # Extract speech rate analysis (verified metrics)
            speech_rate = quality_data.get('speech_rate', {})
            segment_coverage = quality_data.get('segment_coverage', {})

            analysis_data.update({
                # Speech Rate Metrics
                'overall_score': speech_rate.get('en_wpm', 0),  # English WPM
                'accuracy_score': speech_rate.get('zh_wpm', 0),  # Chinese WPM
                'fluency_score': speech_rate.get('speed_ratio', 0),  # Speed ratio
                'completeness_score': segment_coverage.get('coverage_rate', 0),  # Coverage rate

                # Quality Level based on coverage rate
                'quality_level': determine_coverage_quality_level(segment_coverage.get('coverage_rate', 0)),

                # Additional verified metrics
                'total_segments': segment_coverage.get('total_segments', analysis_data.get('total_segments', 0)),
                'bilingual_segments': segment_coverage.get('bilingual_segments', 0),
                'coverage_rate': segment_coverage.get('coverage_rate', 0),  # Ensure coverage_rate is saved
                'processing_time_ms': quality_data.get('processing_time_ms', 0),

                # Speech rate details
                'en_wpm': speech_rate.get('en_wpm', 0),
                'zh_wpm': speech_rate.get('zh_wpm', 0),
                'speed_ratio': speech_rate.get('speed_ratio', 0),
                'pace_assessment': speech_rate.get('pace_assessment', 'Unknown'),
                'balance_assessment': speech_rate.get('balance_assessment', 'Unknown')
            })

        # Only include confidence metrics if they exist and are reliable
        confidence_data = results.get('detailed_analysis', {}).get('confidence', {})
        if 'error' not in confidence_data and confidence_data.get('total_with_confidence', 0) > 0:
            overall_conf = confidence_data.get('overall', {})
            analysis_data.update({
                'confidence_mean': overall_conf.get('mean_confidence', 0),
                'confidence_std': overall_conf.get('std_confidence', 0),
                'confidence_words_count': confidence_data.get('total_with_confidence', 0)
            })

        # Skip EVS metrics for now as they need further verification
        # evs_data = results.get('detailed_analysis', {}).get('evs', {})
        # Note: EVS analysis will be included once validation is complete

        # Save main analysis result using alias file name
        analysis_id = EVSDataUtils.save_si_analysis_result(
            file_name=alias_file_name,
            asr_provider=results['asr_provider'],
            analysis_type='complete_analysis',
            analysis_data=analysis_data,
            created_by=st.session_state.get('user_email', 'system')
        )

        if analysis_id > 0:
            logger.info(f"Successfully saved SI analysis results with ID: {analysis_id}")

            # Save additional detailed data if available
            save_additional_analysis_details(analysis_id, results)

            return True
        else:
            logger.error("Failed to save SI analysis results to database")
            return False

    except Exception as e:
        logger.error(f"Error saving SI analysis results: {str(e)}", exc_info=True)
        return False

def determine_evs_quality_level(mean_evs):
    """Determine quality level based on mean EVS value (deprecated - for future use)"""
    if mean_evs is None:
        return "Unknown"

    abs_evs = abs(mean_evs)
    if abs_evs <= 2.0:
        return "Excellent"
    elif abs_evs <= 3.0:
        return "Good"
    elif abs_evs <= 5.0:
        return "Acceptable"
    else:
        return "Needs Improvement"

def determine_coverage_quality_level(coverage_rate):
    """Determine quality level based on segment coverage rate"""
    if coverage_rate is None:
        return "Unknown"

    # Convert to percentage if it's a decimal
    if coverage_rate <= 1.0:
        coverage_rate = coverage_rate * 100

    if coverage_rate >= 90.0:
        return "Excellent"
    elif coverage_rate >= 75.0:
        return "Good"
    elif coverage_rate >= 60.0:
        return "Acceptable"
    else:
        return "Needs Improvement"

def save_additional_analysis_details(analysis_id, results):
    """Save additional analysis details focusing on verified quality metrics"""
    try:
        # Save quality metrics details - these are verified and reliable
        quality_data = results.get('detailed_analysis', {}).get('quality', {})
        if quality_data and 'error' not in quality_data:
            quality_details = []

            # Save speech rate analysis details
            speech_rate = quality_data.get('speech_rate', {})
            if speech_rate:
                quality_details.append({
                    'type': 'speech_rate_analysis',
                    'severity': 'info',
                    'segment_index': 0,
                    'timestamp': 0,
                    'source': f"English WPM: {speech_rate.get('en_wpm', 0)}",
                    'target': f"Chinese WPM: {speech_rate.get('zh_wpm', 0)}",
                    'description': f"Speed Ratio: {speech_rate.get('speed_ratio', 0):.2f}, Pace: {speech_rate.get('pace_assessment', 'Unknown')}",
                    'suggestion': f"Balance: {speech_rate.get('balance_assessment', 'Unknown')}"
                })

            # Save segment coverage details
            segment_coverage = quality_data.get('segment_coverage', {})
            if segment_coverage:
                coverage_rate = segment_coverage.get('coverage_rate', 0)
                if coverage_rate <= 1.0:
                    coverage_rate = coverage_rate * 100

                quality_details.append({
                    'type': 'segment_coverage_analysis',
                    'severity': 'info',
                    'segment_index': 0,
                    'timestamp': 0,
                    'source': f"Total Segments: {segment_coverage.get('total_segments', 0)}",
                    'target': f"Bilingual Segments: {segment_coverage.get('bilingual_segments', 0)}",
                    'description': f"Coverage Rate: {coverage_rate:.1f}%",
                    'suggestion': f"Coverage Assessment: {segment_coverage.get('coverage_assessment', 'Unknown')}"
                })

            # Save any quality issues if they exist
            if 'issues' in quality_data and quality_data['issues']:
                for issue in quality_data['issues']:
                    quality_details.append({
                        'type': 'quality_issue',
                        'severity': issue.get('severity', 'medium'),
                        'segment_index': issue.get('segment', 0),
                        'timestamp': issue.get('timestamp', 0),
                        'source': issue.get('source', ''),
                        'target': issue.get('target', ''),
                        'description': issue.get('description', ''),
                        'suggestion': issue.get('suggestion', '')
                    })

            if quality_details:
                EVSDataUtils.save_si_error_details(analysis_id, quality_details)
                logger.info(f"Saved {len(quality_details)} quality analysis details")

        # Save confidence analysis details if reliable
        confidence_data = results.get('detailed_analysis', {}).get('confidence', {})
        if confidence_data and 'error' not in confidence_data and confidence_data.get('total_with_confidence', 0) > 0:
            confidence_details = []

            overall_conf = confidence_data.get('overall', {})
            by_lang = confidence_data.get('by_language', {})

            confidence_details.append({
                'type': 'confidence_analysis',
                'severity': 'info',
                'segment_index': 0,
                'timestamp': 0,
                'source': f"Overall Mean: {overall_conf.get('mean_confidence', 0):.3f}",
                'target': f"Words with Confidence: {confidence_data.get('total_with_confidence', 0)}",
                'description': f"Std Dev: {overall_conf.get('std_confidence', 0):.3f}",
                'suggestion': f"EN: {by_lang.get('en', {}).get('mean', 0):.3f}, ZH: {by_lang.get('zh', {}).get('mean', 0):.3f}"
            })

            if confidence_details:
                EVSDataUtils.save_si_error_details(analysis_id, confidence_details)
                logger.info(f"Saved {len(confidence_details)} confidence analysis details")

        # Skip EVS details for now - will be added once validation is complete
        # logger.info("EVS analysis details skipped - awaiting validation")

    except Exception as e:
        logger.error(f"Error saving additional analysis details: {str(e)}", exc_info=True)

def improve_evs_analysis_with_llm(pair_details, file_name):
    """Use LLM to analyze and improve EVS pair quality"""
    try:
        # Load LLM configuration
        from llm_config import get_active_llm_config

        llm_config = get_active_llm_config()
        if not llm_config:
            raise Exception("LLM configuration not found")

        # Prepare data for LLM analysis
        analysis_prompt = f"""
As a quality analysis expert for simultaneous interpretation, please analyze the EVS data for the following English-Chinese pairs and provide improvement suggestions.

File Name: {file_name}
Total Pairs: {len(pair_details)}

Pair Details:
"""

        # Add sample pairs for analysis (limit to avoid token overflow)
        sample_size = min(20, len(pair_details))
        for i, pair in enumerate(pair_details[:sample_size]):
            analysis_prompt += f"""
Pair {i+1}:
- English: "{pair['en_word']}" (Time: {pair['en_time']:.3f}s, Segment: {pair['en_segment']})
- Chinese: "{pair['zh_word']}" (Time: {pair['zh_time']:.3f}s, Segment: {pair['zh_segment']})
- EVS: {pair['evs']:.3f}s
- Pairing Method: {pair['method']}
"""

        analysis_prompt += f"""

Please analyze the following issues:
1. The reasonableness of EVS values (normal SI EVS usually ranges from 0-5 seconds)
2. The semantic relevance of English-Chinese vocabulary pairs
3. The logical timestamp
4. Possible reasons for abnormal values

Please provide:
1. Overall analysis report
2. Specific improvement suggestions
3. Recommended high-quality pairs to retain (if any)

Please reply in Chinese, in JSON format:
{{
    "analysis": "Analysis report text",
    "recommendations": ["Suggestion 1", "Suggestion 2", "Suggestion 3"],
    "quality_issues": ["Issue 1", "Issue 2"],
    "filtered_pairs": [
        {{"en_word": "word", "zh_word": "word", "evs": 1.5, "confidence": 0.9, "reason": "reason"}}
    ]
}}
"""

        # Call LLM
        from clients import get_llm_client
        llm_client = get_llm_client()

        logger.info(f"Using LLM for EVS analysis improvement: {llm_config['llm_provider']}")

        try:
            response = llm_client.generate(analysis_prompt)
            logger.info(f"LLM analysis response received, length: {len(response) if response else 0}")

            if not response or response.startswith("Error") or "Error" in response:
                logger.error(f"LLM analysis failed: {response}")
                st.error(f"LLM server error: {response}")
                raise Exception(f"LLM response error: {response}")

        except ConnectionError as ce:
            logger.error(f"LLM connection failed: {str(ce)}")
            st.error(str(ce))
            raise
        except Exception as llm_error:
            logger.error(f"LLM analysis call failed: {str(llm_error)}")
            raise Exception(f"LLM service unavailable: {str(llm_error)}")

        # Parse LLM response
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                return result
            else:
                # If no JSON found, return raw analysis
                return {
                    "analysis": response,
                    "recommendations": ["LLM response format parsing failed, please check the original analysis"],
                    "quality_issues": [],
                    "filtered_pairs": []
                }

        except json.JSONDecodeError:
            return {
                "analysis": response,
                "recommendations": ["JSON parsing failed, but LLM provided text analysis"],
                "quality_issues": [],
                "filtered_pairs": []
            }

    except Exception as e:
        logger.error(f"LLM EVS analysis failed: {str(e)}")
        return {
            "analysis": f"LLM analysis failed: {str(e)}",
            "recommendations": ["Please check LLM configuration", "Manually check abnormal pairs"],
            "quality_issues": ["LLM service unavailable"],
            "filtered_pairs": []
        }

def call_ollama_llm(prompt, config):
    """Call Ollama LLM API"""
    import requests

    url = f"{config['llm_base_url']}/api/generate"
    data = {
        "model": config['llm_model'],
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": config['llm_temperature'],
            "num_predict": config['llm_max_tokens']
        }
    }

    response = requests.post(url, json=data, timeout=config['llm_request_timeout'])
    response.raise_for_status()

    result = response.json()
    return result.get('response', '')

def call_openai_llm(prompt, config):
    """Call OpenAI API"""
    import openai

    client = openai.OpenAI(
        api_key=config.get('llm_api_key', ''),
        base_url=config.get('llm_base_url', 'https://api.openai.com/v1')
    )

    response = client.chat.completions.create(
        model=config['llm_model'],
        messages=[
            {"role": "system", "content": "You are a professional simultaneous interpretation quality analysis expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=config['llm_temperature'],
        max_tokens=config['llm_max_tokens']
    )

    return response.choices[0].message.content

def create_llm_based_pairs(en_data, zh_data):
    """Create EVS pairs using LLM semantic analysis"""
    try:
        from llm_config import get_active_llm_config
        from clients import get_llm_client

        # Get LLM configuration
        llm_config = get_active_llm_config()
        if not llm_config:
            logger.warning("LLM configuration not available for semantic pairing")
            return []

        logger.info(f"LLM Config: provider={llm_config.get('llm_provider')}, model={llm_config.get('llm_model')}")

        # Use unified LLM client
        llm_client = get_llm_client()

        # Prepare data for LLM analysis
        # Sort by time and process full data
        en_sorted = en_data.sort_values('start_time')
        zh_sorted = zh_data.sort_values('start_time')

        # Process all words (remove limitation)
        logger.info(f"Processing full text: {len(en_sorted)} English words, {len(zh_sorted)} Chinese words")

        # Create context strings for all words
        en_context = []
        zh_context = []

        for _, word in en_sorted.iterrows():
            en_context.append({
                'word': str(word['edit_word']) if pd.notna(word['edit_word']) else '',
                'time': float(word['start_time']),
                'segment': int(word['segment_id'])
            })

        for _, word in zh_sorted.iterrows():
            zh_context.append({
                'word': str(word['edit_word']) if pd.notna(word['edit_word']) else '',
                'time': float(word['start_time']),
                'segment': int(word['segment_id'])
            })

        # Dynamic batch sizing based on LLM provider and context window
        def get_optimal_batch_size(provider, total_words):
            """Calculate optimal batch size based on LLM capabilities"""
            context_limits = {
                'openai': 25000,      # GPT-4 has large context window
                'ollama': 8000,       # Local models may have smaller context
                'claude': 30000,      # Claude has very large context window
                'qwen': 15000,        # Qwen has good context size
                'default': 10000
            }

            # Estimate tokens: roughly 1.3 tokens per word for mixed EN/ZH
            estimated_tokens_per_word = 1.5
            max_words_per_batch = context_limits.get(provider.lower(), context_limits['default']) // estimated_tokens_per_word

            # Try to process all in one batch if possible, otherwise use large batches
            if total_words <= max_words_per_batch:
                return total_words  # Process all at once
            else:
                # Use large batches but not too many
                return max(max_words_per_batch // 2, 500)  # At least 500 words per batch

        total_words = min(len(en_context), len(zh_context))
        batch_size = get_optimal_batch_size(llm_config.get('llm_provider', 'default'), total_words)

        logger.info(f"Using batch size: {batch_size} words for provider: {llm_config.get('llm_provider', 'unknown')}")

        # Process in optimized batches
        all_pairs = []
        num_batches = (min(len(en_context), len(zh_context)) + batch_size - 1) // batch_size
        logger.info(f"Will process {num_batches} batch(es)")

        for batch_start in range(0, min(len(en_context), len(zh_context)), batch_size):
            try:
                batch_end = min(batch_start + batch_size, min(len(en_context), len(zh_context)))
                en_batch = en_context[batch_start:batch_end]
                zh_batch = zh_context[batch_start:batch_end]

                batch_num = batch_start // batch_size + 1
                logger.info(f"Processing batch {batch_num}/{num_batches}: {len(en_batch)} EN words, {len(zh_batch)} ZH words")

                # Create LLM prompt for semantic pairing
                prompt = f"""
As a simultaneous interpretation expert, please analyze the following English and Chinese vocabulary, find the semantic related pairs.

English vocabulary (in chronological order):
"""

                for i, en_word in enumerate(en_batch):
                    prompt += f"{batch_start + i + 1}. \"{en_word['word']}\" (时间: {en_word['time']:.3f}s, 片段: {en_word['segment']})\n"

                prompt += f"""
Chinese vocabulary (in chronological order):
"""

                for i, zh_word in enumerate(zh_batch):
                    prompt += f"{batch_start + i + 1}. \"{zh_word['word']}\" (时间: {zh_word['time']:.3f}s, 片段: {zh_word['segment']})\n"

                # Dynamic max pairs based on batch size
                max_pairs_for_batch = min(len(en_batch), len(zh_batch), max(50, len(en_batch) // 3))

                prompt += f"""

Please find the semantic related English-Chinese pairs, consider:
1. The semantic correspondence of vocabulary
2. The reasonableness of time (Chinese usually starts after English within 0-8 seconds)
3. The coherence of context
4. The correspondence of professional terms
5. The characteristics of simultaneous interpretation (Chinese may be a translation or simplified expression of English)

**Important Requirements:**
- Must use the exact timestamps provided in the list above, do not modify or estimate time
- en_time must be the exact timestamp from the English vocabulary list
- zh_time must be the exact timestamp from the Chinese vocabulary list
- Each pairing timestamp must be copied exactly from the list above

Please return the pairing results in JSON format:
{{
    "pairs": [
        {{
            "en_word": "English word",
            "en_time": the exact timestamp copied from the English list,
            "en_segment": segment number,
            "zh_word": "中文词",
            "zh_time": the exact timestamp copied from the Chinese list,
            "zh_segment": segment number,
            "confidence": 0.0-1.0,
            "reason": "reason for pairing"
        }}
    ]
}}

**Pairing Example (please follow this format):**
If the English list has "Good" (Time: 2.040s, Segment: 0)
and the Chinese list has "好的" (Time: 3.580s, Segment: 1)
then the correct pairing format is:
{{
    "en_word": "Good",
    "en_time": 2.040,
    "en_segment": 0,
    "zh_word": "好的",
    "zh_time": 3.580,
    "zh_segment": 1,
    "confidence": 0.8,
    "reason": "semantic correspondence"
}}

Only return pairs with high confidence (>0.6), and the maximum number of pairs for this batch is {max_pairs_for_batch}.
Prioritize pairs with the most semantic relevance and the most reasonable time.
Note: en_time and zh_time must be different, because this is simultaneous interpretation, there will be a time difference.
"""

                # Call LLM using unified client
                logger.info(f"Calling LLM for semantic pairing batch {batch_num}...")
                logger.debug(f"Prompt length: {len(prompt)} characters")

                try:
                    response = llm_client.generate(prompt)
                    logger.info(f"LLM response received for batch {batch_num}, length: {len(response) if response else 0} characters")

                    if response:
                        logger.debug(f"Response preview: {response[:300]}...")

                        # Check for error messages in response
                        if response.startswith("Error") or "Error" in response:
                            logger.error(f"LLM returned error response: {response}")
                            st.error(f"LLM server error: {response}")
                            return []
                    else:
                        logger.error("LLM returned empty response!")
                        st.error("LLM returned empty response. Please check that the LLM server is running.")
                        return []

                    # Parse LLM response
                    batch_pairs = parse_llm_pairing_response(response)
                    all_pairs.extend(batch_pairs)

                    logger.info(f"Batch {batch_num}: Found {len(batch_pairs)} pairs")

                except ConnectionError as ce:
                    logger.error(f"LLM connection failed: {str(ce)}")
                    st.error(str(ce))
                    return []
                except Exception as llm_error:
                    logger.error(f"LLM call failed for batch {batch_num}: {str(llm_error)}", exc_info=True)
                    st.error(f"LLM call failed: {str(llm_error)}")
                    return []

            except Exception as e:
                logger.warning(f"Error processing batch {batch_start//batch_size + 1}: {str(e)}")
                continue

        # Convert to EVS format
        evs_pairs = []

        # Create lookup dictionaries for validation and correction
        en_word_times = {}
        zh_word_times = {}

        for _, word in en_data.iterrows():
            word_str = str(word['edit_word']) if pd.notna(word['edit_word']) else ''
            if word_str not in en_word_times:
                en_word_times[word_str] = []
            en_word_times[word_str].append(float(word['start_time']))

        for _, word in zh_data.iterrows():
            word_str = str(word['edit_word']) if pd.notna(word['edit_word']) else ''
            if word_str not in zh_word_times:
                zh_word_times[word_str] = []
            zh_word_times[word_str].append(float(word['start_time']))

        en_times = set(float(word['start_time']) for _, word in en_data.iterrows())
        zh_times = set(float(word['start_time']) for _, word in zh_data.iterrows())

        logger.info(f"Available EN times: {len(en_times)}, ZH times: {len(zh_times)}")
        logger.info(f"Total LLM pairs before processing: {len(all_pairs)}")

        for i, pair in enumerate(all_pairs):
            try:
                en_word = pair['en_word']
                zh_word = pair['zh_word']
                en_time = float(pair['en_time'])
                zh_time = float(pair['zh_time'])

                # Try to correct invalid timestamps with more flexible matching
                corrected_en_time = en_time
                corrected_zh_time = zh_time

                # Validate and correct EN timestamp with tolerance
                if en_time not in en_times:
                    if en_word in en_word_times and en_word_times[en_word]:
                        # Find closest time within tolerance (±0.5s)
                        closest_time = min(en_word_times[en_word], key=lambda t: abs(t - en_time))
                        if abs(closest_time - en_time) <= 0.5:
                            corrected_en_time = closest_time
                            logger.info(f"Pair {i+1}: Corrected EN time {en_time} -> {corrected_en_time} for word '{en_word}'")
                        else:
                            # Use the first occurrence if no close match
                            corrected_en_time = en_word_times[en_word][0]
                            logger.info(f"Pair {i+1}: Used first occurrence EN time {en_time} -> {corrected_en_time} for word '{en_word}'")
                    else:
                        # Try to find similar word or use closest time
                        closest_en_time = min(en_times, key=lambda t: abs(t - en_time))
                        if abs(closest_en_time - en_time) <= 1.0:  # 1 second tolerance
                            corrected_en_time = closest_en_time
                            logger.info(f"Pair {i+1}: Used closest EN time {en_time} -> {corrected_en_time} for word '{en_word}'")
                        else:
                            logger.warning(f"Pair {i+1}: EN word '{en_word}' not found, skipping")
                            continue

                # Validate and correct ZH timestamp with more flexible matching
                if zh_time not in zh_times:
                    if zh_word in zh_word_times and zh_word_times[zh_word]:
                        # Find the best matching time (prefer times within reasonable EVS range)
                        expected_zh_time_range = (corrected_en_time - 1, corrected_en_time + 15)  # Expanded range
                        best_zh_time = None

                        # First try to find within reasonable EVS range
                        for candidate_time in zh_word_times[zh_word]:
                            if expected_zh_time_range[0] <= candidate_time <= expected_zh_time_range[1]:
                                if best_zh_time is None or abs(candidate_time - zh_time) < abs(best_zh_time - zh_time):
                                    best_zh_time = candidate_time

                        if best_zh_time is not None:
                            corrected_zh_time = best_zh_time
                            logger.info(f"Pair {i+1}: Corrected ZH time {zh_time} -> {corrected_zh_time} for word '{zh_word}'")
                        else:
                            # Use closest time within tolerance
                            closest_time = min(zh_word_times[zh_word], key=lambda t: abs(t - zh_time))
                            if abs(closest_time - zh_time) <= 0.5:
                                corrected_zh_time = closest_time
                                logger.info(f"Pair {i+1}: Used closest ZH time {zh_time} -> {corrected_zh_time} for word '{zh_word}'")
                            else:
                                corrected_zh_time = zh_word_times[zh_word][0]
                                logger.info(f"Pair {i+1}: Used first occurrence ZH time {zh_time} -> {corrected_zh_time} for word '{zh_word}'")
                    else:
                        # Try to find closest time
                        closest_zh_time = min(zh_times, key=lambda t: abs(t - zh_time))
                        if abs(closest_zh_time - zh_time) <= 1.0:  # 1 second tolerance
                            corrected_zh_time = closest_zh_time
                            logger.info(f"Pair {i+1}: Used closest ZH time {zh_time} -> {corrected_zh_time} for word '{zh_word}'")
                        else:
                            logger.warning(f"Pair {i+1}: ZH word '{zh_word}' not found, skipping")
                            continue

                evs = corrected_zh_time - corrected_en_time

                # Check for suspicious zero EVS
                if abs(evs) < 0.001:
                    logger.warning(f"Pair {i+1}: Suspicious zero EVS - {en_word}({corrected_en_time}) -> {zh_word}({corrected_zh_time})")
                    continue

                # Relaxed EVS filtering - accept wider range for interpretation
                if -5.0 <= evs <= 20.0:  # Much more relaxed range
                    evs_pairs.append({
                        'en_word': en_word,
                        'en_time': float(corrected_en_time),
                        'en_segment': int(pair['en_segment']),
                        'zh_word': zh_word,
                        'zh_time': float(corrected_zh_time),
                        'zh_segment': int(pair['zh_segment']),
                        'evs': float(evs),
                        'method': 'llm_semantic',
                        'confidence': float(pair.get('confidence', 0.8)),
                        'reason': pair.get('reason', 'LLM semantic pairing')
                    })
                    logger.info(f"Pair {i+1}: {en_word}({corrected_en_time:.3f}s) -> {zh_word}({corrected_zh_time:.3f}s) = {evs:.3f}s")
                else:
                    logger.warning(f"Pair {i+1}: EVS {evs:.3f}s outside reasonable range (-5s to +20s)")

            except Exception as e:
                logger.warning(f"Error processing LLM pair {i+1}: {str(e)}")
                continue

        logger.info(f"LLM created {len(evs_pairs)} valid semantic pairs from {len(all_pairs)} total pairs")
        return evs_pairs

    except Exception as e:
        logger.error(f"Error in LLM-based pairing: {str(e)}", exc_info=True)
        return []

def parse_llm_pairing_response(response):
    """Parse LLM response for pairing data"""
    try:
        logger.info(f"Parsing LLM response, length: {len(response)}")
        logger.debug(f"LLM response content: {response[:500]}...")  # Log first 500 chars

        # Clean and prepare response for JSON extraction
        cleaned_response = response.strip()

        # Multiple JSON extraction strategies
        json_candidates = []

        # Strategy 1: Look for complete JSON objects
        json_pattern = re.compile(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\}))*\}', re.DOTALL)
        json_matches = json_pattern.findall(cleaned_response)
        json_candidates.extend(json_matches)

        # Strategy 2: Look for "pairs" array specifically
        pairs_pattern = re.compile(r'"pairs"\s*:\s*\[.*?\]', re.DOTALL | re.IGNORECASE)
        pairs_match = pairs_pattern.search(cleaned_response)
        if pairs_match:
            # Try to build a valid JSON object around the pairs array
            pairs_content = pairs_match.group()
            json_candidates.append(f'{{{pairs_content}}}')

        # Strategy 3: Extract from markdown code blocks
        code_block_pattern = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.DOTALL | re.IGNORECASE)
        code_matches = code_block_pattern.findall(cleaned_response)
        json_candidates.extend(code_matches)

        logger.info(f"Found {len(json_candidates)} JSON candidates")

        # Try parsing each candidate
        for i, json_candidate in enumerate(json_candidates):
            try:
                logger.debug(f"Trying JSON candidate {i+1}: {json_candidate[:200]}...")

                # Clean the candidate
                clean_candidate = json_candidate.strip()

                # Parse JSON
                result = json.loads(clean_candidate)

                if isinstance(result, dict) and 'pairs' in result and isinstance(result['pairs'], list):
                    # Validate and filter pairs
                    valid_pairs = []
                    total_pairs = len(result['pairs'])
                    logger.info(f"Found {total_pairs} pairs in JSON candidate {i+1}")

                    for j, pair in enumerate(result['pairs']):
                        if not isinstance(pair, dict):
                            logger.warning(f"Pair {j+1} is not a dictionary: {pair}")
                            continue

                        # Check required keys
                        required_keys = ['en_word', 'en_time', 'zh_word', 'zh_time']
                        if all(key in pair for key in required_keys):
                            # Validate data types
                            try:
                                en_word = str(pair['en_word']).strip()
                                zh_word = str(pair['zh_word']).strip()
                                en_time = float(pair['en_time'])
                                zh_time = float(pair['zh_time'])
                                confidence = float(pair.get('confidence', 0.7))

                                # Skip empty words
                                if not en_word or not zh_word:
                                    logger.warning(f"Pair {j+1} has empty words: en='{en_word}', zh='{zh_word}'")
                                    continue

                                # Update the pair with validated values
                                pair['en_word'] = en_word
                                pair['zh_word'] = zh_word
                                pair['en_time'] = en_time
                                pair['zh_time'] = zh_time
                                pair['confidence'] = confidence

                                # Check confidence threshold
                                if confidence > 0.6:
                                    valid_pairs.append(pair)
                                    logger.debug(f"Pair {j+1}: {en_word} ({en_time}) -> {zh_word} ({zh_time}) confidence: {confidence}")
                                else:
                                    logger.debug(f"Filtered out pair {j+1} due to low confidence: {confidence}")

                            except (ValueError, TypeError) as ve:
                                logger.warning(f"Pair {j+1} has invalid data types: {str(ve)}")
                                continue
                        else:
                            missing_keys = [key for key in required_keys if key not in pair]
                            logger.warning(f"Pair {j+1} missing required keys: {missing_keys}")

                    logger.info(f"Accepted {len(valid_pairs)} pairs out of {total_pairs} from JSON candidate {i+1}")
                    if valid_pairs:  # Return the first successful parse with valid pairs
                        return valid_pairs

                else:
                    logger.warning(f"JSON candidate {i+1} doesn't have valid pairs structure")

            except json.JSONDecodeError as jde:
                logger.debug(f"JSON candidate {i+1} parse failed: {str(jde)}")
                continue
            except Exception as e:
                logger.warning(f"Error processing JSON candidate {i+1}: {str(e)}")
                continue

        # If all JSON candidates failed, try simple extraction
        logger.warning("All JSON parsing failed, attempting simple extraction")
        return extract_simple_pairs(response)

    except Exception as e:
        logger.error(f"Error parsing LLM response: {str(e)}", exc_info=True)
        return []

def extract_simple_pairs(response):
    """Extract simple pairs from non-JSON response using pattern matching"""
    try:
        pairs = []
        lines = response.split('\n')

        logger.info(f"Attempting simple pair extraction from {len(lines)} lines")

        # Pattern 1: Look for explicit pair format
        # Example: "Good" (2.040s) -> "好的" (3.580s)
        pair_pattern = re.compile(
            r'"([^"]+)"\s*\(([0-9]+\.?[0-9]*)s?\)\s*[-→>]+\s*"([^"]+)"\s*\(([0-9]+\.?[0-9]*)s?\)',
            re.IGNORECASE
        )

        # Pattern 2: Simple word-to-word mapping
        # Example: word1 -> word2
        simple_pattern = re.compile(
            r'([a-zA-Z0-9\u4e00-\u9fff]+)\s*[-→>]+\s*([a-zA-Z0-9\u4e00-\u9fff]+)',
            re.IGNORECASE
        )

        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Try pattern 1 (with times)
            match = pair_pattern.search(line)
            if match:
                en_word, en_time, zh_word, zh_time = match.groups()
                try:
                    pairs.append({
                        'en_word': en_word.strip(),
                        'en_time': float(en_time),
                        'en_segment': 0,  # Default segment
                        'zh_word': zh_word.strip(),
                        'zh_time': float(zh_time),
                        'zh_segment': 0,  # Default segment
                        'confidence': 0.7,  # Default confidence
                        'reason': 'Simple pattern extraction with times'
                    })
                    logger.debug(f"Extracted pair from line {line_num}: {en_word} -> {zh_word}")
                except ValueError as ve:
                    logger.warning(f"Invalid time values in line {line_num}: {str(ve)}")
                continue

            # Try pattern 2 (simple mapping, need to infer times)
            match = simple_pattern.search(line)
            if match:
                word1, word2 = match.groups()
                # Basic heuristic: if word1 contains only ASCII, it's probably English
                if re.match(r'^[a-zA-Z0-9.,!?\'"-]+$', word1.strip()):
                    en_word, zh_word = word1.strip(), word2.strip()
                else:
                    zh_word, en_word = word1.strip(), word2.strip()

                # Can't extract times from simple patterns, would need to match with original data
                logger.debug(f"Found simple mapping in line {line_num}: {en_word} -> {zh_word} (times need inference)")

        logger.info(f"Simple extraction found {len(pairs)} pairs")
        return pairs

    except Exception as e:
        logger.warning(f"Simple pair extraction failed: {str(e)}")
        return []

def perform_advanced_si_analysis(asr_data, file_name, asr_provider):
    """Perform advanced simultaneous interpretation analysis using LLM and sophisticated metrics"""
    try:
        from datetime import datetime
        import re
        from collections import Counter

        # No need to parse combined_word - fields already exist directly in database
        # Just ensure required fields exist
        required_fields = ['edit_word', 'pair_type', 'original_word', 'annotate']
        for field in required_fields:
            if field not in asr_data.columns:
                asr_data[field] = None

        # Separate languages
        en_data = asr_data[asr_data['lang'] == 'en'].copy()
        zh_data = asr_data[asr_data['lang'] == 'zh'].copy()

        # Initialize results structure
        advanced_results = {
            'file_name': file_name,
            'asr_provider': asr_provider,
            'analysis_timestamp': datetime.now().isoformat(),
            'translation_accuracy': {},
            'terminology_analysis': {},
            'discourse_analysis': {},
            'cultural_analysis': {},
            'performance_metrics': {},
            'recommendations': []
        }

        # 1. Translation Accuracy Assessment
        logger.info("Performing translation accuracy assessment...")
        accuracy_results = analyze_translation_accuracy(en_data, zh_data, file_name, asr_provider)
        advanced_results['translation_accuracy'] = accuracy_results

        # 2. Terminology Consistency Analysis
        logger.info("Performing terminology consistency analysis...")
        terminology_results = analyze_terminology_consistency(en_data, zh_data)
        advanced_results['terminology_analysis'] = terminology_results

        # 3. Discourse Analysis
        logger.info("Performing discourse analysis...")
        discourse_results = analyze_discourse_structure(en_data, zh_data)
        advanced_results['discourse_analysis'] = discourse_results

        # 4. Cultural Adaptation Analysis
        logger.info("Performing cultural adaptation analysis...")
        cultural_results = analyze_cultural_adaptation(en_data, zh_data)
        advanced_results['cultural_analysis'] = cultural_results

        # 5. Advanced Performance Metrics
        logger.info("Calculating advanced performance metrics...")
        performance_results = calculate_advanced_performance_metrics(asr_data, en_data, zh_data)
        advanced_results['performance_metrics'] = performance_results

        # 6. Generate Recommendations
        logger.info("Generating improvement recommendations...")
        recommendations = generate_improvement_recommendations(advanced_results)
        advanced_results['recommendations'] = recommendations

        return advanced_results

    except Exception as e:
        logger.error(f"Advanced analysis failed: {str(e)}", exc_info=True)
        return {
            'file_name': file_name,
            'asr_provider': asr_provider,
            'analysis_timestamp': datetime.now().isoformat(),
            'translation_accuracy': {'error': f"Analysis failed: {str(e)}"},
            'terminology_analysis': {'error': f"Analysis failed: {str(e)}"},
            'discourse_analysis': {'error': f"Analysis failed: {str(e)}"},
            'cultural_analysis': {'error': f"Analysis failed: {str(e)}"},
            'performance_metrics': {'error': f"Analysis failed: {str(e)}"},
            'recommendations': [],
            'error': f"Advanced analysis failed: {str(e)}"
        }

def analyze_translation_accuracy(en_data, zh_data, file_name=None, asr_provider=None):
    """Analyze translation accuracy using time-based mapping instead of segment_id matching"""
    try:
        from utils.analysis_utils import analyze_translation_accuracy_with_time_mapping

        # Get file_name and asr_provider from parameters or try to extract from session state
        if file_name is None or asr_provider is None:
            import streamlit as st
            if hasattr(st, 'session_state'):
                file_name = file_name or getattr(st.session_state, 'selected_file', 'unknown')
                asr_provider = asr_provider or getattr(st.session_state, 'selected_asr_provider', 'unknown')
            else:
                file_name = file_name or 'unknown'
                asr_provider = asr_provider or 'unknown'

        logger.info(f"Starting time-based analysis for file: {file_name}, provider: {asr_provider}")

        # DEBUG: Test if the exact parameters work
        from save_asr_results import get_db_connection
        import pandas as pd
        with get_db_connection() as conn:
            test_query = """
            SELECT COUNT(*) as segment_count FROM asr_results_segments
            WHERE file_name = ? AND asr_provider = ?
            """
            test_result = pd.read_sql_query(test_query, conn, params=[file_name, asr_provider])
            logger.info(f"Segments found for these parameters: {test_result.iloc[0]['segment_count']}")
            if test_result.iloc[0]['segment_count'] == 0:
                logger.error(f"No segments found! This explains the failure.")

        # Test import first
        try:
            logger.info("Testing import of time mapping function...")
            test_result = analyze_translation_accuracy_with_time_mapping.__name__
            logger.info(f"Import successful: {test_result}")
        except Exception as import_error:
            logger.error(f"Import test failed: {str(import_error)}")
            raise import_error

        # Use the new time-based mapping analysis
        translation_df, summary_stats = analyze_translation_accuracy_with_time_mapping(file_name, asr_provider)

        logger.info(f"Time-based analysis returned {len(translation_df)} pairs")

        if translation_df.empty:
            # Fallback to old method if new method fails
            logger.warning("Time-based analysis returned empty results, falling back to old method")
            return analyze_translation_accuracy_fallback(en_data, zh_data)

        logger.info(f"Time-based analysis successful! Processing {len(translation_df)} pairs")

        # Convert new format to old format for compatibility
        translation_pairs = []
        try:
            for _, row in translation_df.iterrows():
                translation_pairs.append({
                    'segment_id': row['segment_id'],
                    'english': row['english_text'],  # Match the actual column name from time mapping
                    'chinese': row['chinese_text'],  # Match the actual column name from time mapping
                    'en_word_count': row['en_words'],
                    'zh_word_count': row['zh_words'],
                    'length_ratio': row['length_ratio'],
                    'time_alignment': row['time_alignment']
                })
        except Exception as conversion_error:
            logger.error(f"Error converting time-based results: {str(conversion_error)}")
            logger.error(f"DataFrame columns: {list(translation_df.columns)}")
            logger.error(f"Sample row: {translation_df.iloc[0].to_dict() if not translation_df.empty else 'Empty'}")
            raise conversion_error

        # Calculate metrics in old format
        total_en_segments = len(set([pair['segment_id'].split('-')[0] for pair in translation_pairs]))
        total_zh_segments = len(set([pair['segment_id'].split('-')[1] for pair in translation_pairs]))
        coverage_rate = (len(translation_pairs) / max(total_en_segments, total_zh_segments)) * 100 if max(total_en_segments, total_zh_segments) > 0 else 0

        content_categories = {
            'segment_coverage': coverage_rate / 100,
            'length_consistency': 1 - abs(summary_stats['avg_length_ratio'] - 0.7) if summary_stats['avg_length_ratio'] > 0 else 0,
            'time_alignment': summary_stats['avg_time_alignment'],
            'translation_completeness': summary_stats['total_pairs'] / max(total_en_segments, total_zh_segments) if max(total_en_segments, total_zh_segments) > 0 else 0
        }

        semantic_accuracy = sum(content_categories.values()) / len(content_categories)

        return {
            'total_segments_analyzed': summary_stats['total_pairs'],
            'coverage_rate': coverage_rate,
            'semantic_accuracy': semantic_accuracy,
            'content_preservation': content_categories,
            'translation_pairs': translation_pairs,
            'avg_length_ratio': summary_stats['avg_length_ratio'],
            'avg_time_alignment': summary_stats['avg_time_alignment'],
            'accuracy_grade': 'Excellent' if semantic_accuracy > 0.8 else 'Good' if semantic_accuracy > 0.6 else 'Fair'
        }

    except Exception as e:
        logger.error(f"Time-based translation accuracy analysis failed: {str(e)}, falling back to old method")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return analyze_translation_accuracy_fallback(en_data, zh_data)

def analyze_translation_accuracy_fallback(en_data, zh_data):
    """Fallback method using segment_id matching (original method)"""
    try:
        # Calculate basic accuracy metrics
        total_en_segments = len(en_data['segment_id'].unique()) if not en_data.empty else 0
        total_zh_segments = len(zh_data['segment_id'].unique()) if not zh_data.empty else 0

        # Find paired segments (segments that have both English and Chinese content)
        paired_segments = []
        if not en_data.empty and not zh_data.empty:
            en_segments = set(en_data['segment_id'].unique())
            zh_segments = set(zh_data['segment_id'].unique())
            paired_segments = list(en_segments.intersection(zh_segments))

        # Calculate coverage rate (percentage of segments with both languages)
        coverage_rate = len(paired_segments) / max(total_en_segments, total_zh_segments) * 100 if max(total_en_segments, total_zh_segments) > 0 else 0

        # Analyze translation pairs in detail
        translation_pairs = []
        word_length_ratios = []
        time_alignment_scores = []

        for segment_id in paired_segments:  # Analyze all paired segments
            en_segment = en_data[en_data['segment_id'] == segment_id]
            zh_segment = zh_data[zh_data['segment_id'] == segment_id]

            en_text = ' '.join(en_segment['edit_word'].fillna('').astype(str))
            zh_text = ''.join(zh_segment['edit_word'].fillna('').astype(str))

            if en_text.strip() and zh_text.strip():
                # Calculate word length ratio (Chinese words / English words)
                en_word_count = len(en_segment)
                zh_word_count = len(zh_segment)
                length_ratio = zh_word_count / en_word_count if en_word_count > 0 else 0
                word_length_ratios.append(length_ratio)

                # Calculate time alignment quality
                en_start = en_segment['start_time'].min()
                en_end = en_segment['start_time'].max()
                zh_start = zh_segment['start_time'].min()
                zh_end = zh_segment['start_time'].max()

                # Time overlap percentage
                overlap_start = max(en_start, zh_start)
                overlap_end = min(en_end, zh_end)
                overlap = max(0, overlap_end - overlap_start)
                total_span = max(en_end, zh_end) - min(en_start, zh_start)
                time_alignment = overlap / total_span if total_span > 0 else 0
                time_alignment_scores.append(time_alignment)

                translation_pairs.append({
                    'segment_id': segment_id,
                    'english': en_text,
                    'chinese': zh_text,
                    'en_word_count': en_word_count,
                    'zh_word_count': zh_word_count,
                    'length_ratio': length_ratio,
                    'time_alignment': time_alignment
                })

        # Calculate overall metrics
        avg_length_ratio = sum(word_length_ratios) / len(word_length_ratios) if word_length_ratios else 0
        avg_time_alignment = sum(time_alignment_scores) / len(time_alignment_scores) if time_alignment_scores else 0

        # Content preservation analysis based on real metrics
        content_categories = {
            'segment_coverage': coverage_rate / 100,  # 0-1 scale
            'length_consistency': 1 - abs(avg_length_ratio - 0.7) if avg_length_ratio > 0 else 0,  # Ideal ratio around 0.7 for ZH/EN
            'time_alignment': avg_time_alignment,
            'translation_completeness': len(paired_segments) / max(total_en_segments, total_zh_segments) if max(total_en_segments, total_zh_segments) > 0 else 0
        }

        # Overall accuracy score based on real metrics
        semantic_accuracy = sum(content_categories.values()) / len(content_categories)

        return {
            'total_segments_analyzed': len(paired_segments),
            'coverage_rate': coverage_rate,
            'semantic_accuracy': semantic_accuracy,
            'content_preservation': content_categories,
            'translation_pairs': translation_pairs,
            'avg_length_ratio': avg_length_ratio,
            'avg_time_alignment': avg_time_alignment,
            'accuracy_grade': 'Excellent' if semantic_accuracy > 0.8 else 'Good' if semantic_accuracy > 0.6 else 'Fair'
        }

    except Exception as e:
        logger.error(f"Translation accuracy analysis failed: {str(e)}")
        return {'error': f"Translation accuracy analysis failed: {str(e)}"}

def analyze_terminology_consistency(en_data, zh_data):
    """Analyze consistency of professional terminology usage"""
    try:
        from collections import Counter
        import re

        # Extract potential technical terms (capitalized words, specific patterns)
        en_terms = []
        zh_terms = []

        if not en_data.empty:
            en_text = ' '.join(en_data['edit_word'].fillna('').astype(str))
            # Find capitalized terms, acronyms, and technical patterns
            en_terms = re.findall(r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b|[A-Z]{2,}', en_text)

        if not zh_data.empty:
            zh_text = ''.join(zh_data['edit_word'].fillna('').astype(str))
            # Find potential Chinese technical terms (simplified approach)
            zh_terms = re.findall(r'[\u4e00-\u9fff]{2,}', zh_text)

        # Analyze term frequency and consistency
        en_term_freq = Counter(en_terms)
        zh_term_freq = Counter(zh_terms)

        # Calculate consistency metrics
        most_common_en = en_term_freq.most_common(10)
        most_common_zh = zh_term_freq.most_common(10)

        # Calculate terminology consistency score based on real data
        # Score based on term repetition and distribution
        unique_en_terms = len(en_term_freq)
        unique_zh_terms = len(zh_term_freq)
        total_en_occurrences = sum(en_term_freq.values())
        total_zh_occurrences = sum(zh_term_freq.values())

        # Consistency score based on term distribution uniformity
        en_consistency = 1 - (len(most_common_en) / unique_en_terms) if unique_en_terms > 0 else 0
        zh_consistency = 1 - (len(most_common_zh) / unique_zh_terms) if unique_zh_terms > 0 else 0
        consistency_score = (en_consistency + zh_consistency) / 2

        # Domain analysis
        domain_categories = {
            'business': len([term for term, _ in most_common_en if term.lower() in ['business', 'company', 'market', 'revenue']]),
            'technology': len([term for term, _ in most_common_en if term.lower() in ['system', 'technology', 'data', 'software']]),
            'medical': len([term for term, _ in most_common_en if term.lower() in ['patient', 'treatment', 'medical', 'health']]),
            'legal': len([term for term, _ in most_common_en if term.lower() in ['law', 'legal', 'court', 'contract']])
        }

        dominant_domain = max(domain_categories, key=domain_categories.get) if any(domain_categories.values()) else 'general'

        return {
            'english_terms': most_common_en,
            'chinese_terms': most_common_zh,
            'consistency_score': consistency_score,
            'dominant_domain': dominant_domain,
            'domain_distribution': domain_categories,
            'total_unique_terms': len(en_term_freq) + len(zh_term_freq),
            'terminology_grade': 'Excellent' if consistency_score > 0.9 else 'Good' if consistency_score > 0.8 else 'Fair'
        }

    except Exception as e:
        logger.error(f"Terminology consistency analysis failed: {str(e)}")
        return {'error': f"Terminology consistency analysis failed: {str(e)}"}

def analyze_discourse_structure(en_data, zh_data):
    """Analyze discourse coherence and structure preservation"""
    try:
        # Calculate discourse metrics
        en_seg_set = set(en_data['segment_id'].unique()) if not en_data.empty else set()
        zh_seg_set = set(zh_data['segment_id'].unique()) if not zh_data.empty else set()
        paired_segments = list(en_seg_set.intersection(zh_seg_set))
        en_segments = len(en_seg_set)
        zh_segments = len(zh_seg_set)

        # Coherence analysis based on real data metrics
        # Calculate segment continuity and consistency
        segment_gaps_en = []
        segment_gaps_zh = []

        if not en_data.empty:
            en_segments_sorted = sorted(en_data['segment_id'].unique())
            for i in range(1, len(en_segments_sorted)):
                gap = en_segments_sorted[i] - en_segments_sorted[i-1]
                segment_gaps_en.append(gap)

        if not zh_data.empty:
            zh_segments_sorted = sorted(zh_data['segment_id'].unique())
            for i in range(1, len(zh_segments_sorted)):
                gap = zh_segments_sorted[i] - zh_segments_sorted[i-1]
                segment_gaps_zh.append(gap)

        # Calculate consistency metrics based on real patterns
        avg_gap_en = sum(segment_gaps_en) / len(segment_gaps_en) if segment_gaps_en else 1
        avg_gap_zh = sum(segment_gaps_zh) / len(segment_gaps_zh) if segment_gaps_zh else 1

        coherence_metrics = {
            'logical_flow': min(1.0, 2.0 / avg_gap_en) if avg_gap_en > 0 else 0,  # Better with smaller gaps
            'temporal_consistency': 1 - abs(avg_gap_en - avg_gap_zh) / max(avg_gap_en, avg_gap_zh) if max(avg_gap_en, avg_gap_zh) > 0 else 0,
            'thematic_unity': len(paired_segments) / max(en_segments, zh_segments) if max(en_segments, zh_segments) > 0 else 0,
            'transition_quality': min(1.0, 2.0 / avg_gap_zh) if avg_gap_zh > 0 else 0
        }

        # Segment structure analysis
        avg_en_words_per_segment = len(en_data) / en_segments if en_segments > 0 else 0
        avg_zh_words_per_segment = len(zh_data) / zh_segments if zh_segments > 0 else 0

        structure_balance = abs(avg_en_words_per_segment - avg_zh_words_per_segment) / max(avg_en_words_per_segment, avg_zh_words_per_segment) if max(avg_en_words_per_segment, avg_zh_words_per_segment) > 0 else 1
        structure_score = max(0, 1 - structure_balance)

        # Information density analysis
        info_density = {
            'english_density': avg_en_words_per_segment,
            'chinese_density': avg_zh_words_per_segment,
            'density_ratio': avg_zh_words_per_segment / avg_en_words_per_segment if avg_en_words_per_segment > 0 else 0,
            'balance_score': structure_score
        }

        overall_coherence = sum(coherence_metrics.values()) / len(coherence_metrics)

        return {
            'coherence_metrics': coherence_metrics,
            'overall_coherence': overall_coherence,
            'information_density': info_density,
            'segment_structure': {
                'english_segments': en_segments,
                'chinese_segments': zh_segments,
                'avg_en_words': avg_en_words_per_segment,
                'avg_zh_words': avg_zh_words_per_segment
            },
            'discourse_grade': 'Excellent' if overall_coherence > 0.9 else 'Good' if overall_coherence > 0.8 else 'Fair'
        }

    except Exception as e:
        logger.error(f"Discourse analysis failed: {str(e)}")
        return {'error': f"Discourse analysis failed: {str(e)}"}

def analyze_cultural_adaptation(en_data, zh_data):
    """Analyze cultural adaptation and localization quality"""
    try:
        # Cultural markers analysis based on real text content
        import re

        cultural_elements = {
            'numbers_dates': 0,
            'currency_units': 0,
            'cultural_references': 0,
            'honorifics': 0
        }

        # Analyze English text for cultural elements
        if not en_data.empty:
            en_text = ' '.join(en_data['edit_word'].fillna('').astype(str))
            cultural_elements['numbers_dates'] += len(re.findall(r'\d+', en_text))
            cultural_elements['currency_units'] += len(re.findall(r'\$|\€|\£|dollar|euro|pound|yuan|rmb', en_text, re.IGNORECASE))
            cultural_elements['cultural_references'] += len(re.findall(r'mr\.|mrs\.|dr\.|sir|madam', en_text, re.IGNORECASE))

        # Analyze Chinese text for cultural elements
        if not zh_data.empty:
            zh_text = ''.join(zh_data['edit_word'].fillna('').astype(str))
            cultural_elements['honorifics'] += len(re.findall(r'先生|女士|博士|教授|经理|总裁', zh_text))
            cultural_elements['currency_units'] += len(re.findall(r'元|块|毛|分|美元|欧元|英镑', zh_text))

        # Calculate cultural adaptation metrics based on element presence
        total_elements = sum(cultural_elements.values())

        cultural_metrics = {
            'cultural_sensitivity': min(1.0, cultural_elements['honorifics'] / 10) if cultural_elements['honorifics'] > 0 else 0.5,
            'localization_quality': min(1.0, cultural_elements['currency_units'] / 5) if cultural_elements['currency_units'] > 0 else 0.7,
            'idiom_adaptation': min(1.0, total_elements / 20) if total_elements > 0 else 0.6,
            'context_appropriateness': min(1.0, cultural_elements['cultural_references'] / 5) if cultural_elements['cultural_references'] > 0 else 0.8
        }



        adaptation_score = sum(cultural_metrics.values()) / len(cultural_metrics)

        return {
            'cultural_metrics': cultural_metrics,
            'cultural_elements': cultural_elements,
            'adaptation_score': adaptation_score,
            'localization_recommendations': [
                'Consider cultural context in number formatting',
                'Adapt idiomatic expressions appropriately',
                'Maintain cultural sensitivity in tone'
            ],
            'cultural_grade': 'Excellent' if adaptation_score > 0.9 else 'Good' if adaptation_score > 0.8 else 'Fair'
        }

    except Exception as e:
        logger.error(f"Cultural adaptation analysis failed: {str(e)}")
        return {'error': f"Cultural adaptation analysis failed: {str(e)}"}

def calculate_advanced_performance_metrics(asr_data, en_data, zh_data):
    """Calculate advanced performance metrics beyond basic EVS"""
    try:
        # Advanced timing metrics
        if not en_data.empty and not zh_data.empty:
            en_duration = en_data['duration'].sum() if 'duration' in en_data.columns else 0
            zh_duration = zh_data['duration'].sum() if 'duration' in zh_data.columns else 0

            # Processing efficiency
            processing_efficiency = min(zh_duration / en_duration, 1.0) if en_duration > 0 else 0

            # Cognitive load indicators based on real data
            segments = asr_data['segment_id'].unique()
            word_densities = []
            segment_durations = []

            for seg_id in segments:
                seg_data = asr_data[asr_data['segment_id'] == seg_id]
                word_count = len(seg_data)
                if 'start_time' in seg_data.columns and len(seg_data) > 1:
                    duration = seg_data['start_time'].max() - seg_data['start_time'].min()
                    if duration > 0:
                        word_densities.append(word_count / duration)
                        segment_durations.append(duration)

            if word_densities:
                avg_density = sum(word_densities) / len(word_densities)
                density_variance = sum((d - avg_density) ** 2 for d in word_densities) / len(word_densities)

                cognitive_load = {
                    'word_density_variance': min(1.0, density_variance / avg_density) if avg_density > 0 else 0,
                    'pause_pattern_irregularity': min(1.0, len(set(segment_durations)) / len(segment_durations)) if segment_durations else 0,
                    'speed_fluctuation': min(1.0, (max(word_densities) - min(word_densities)) / avg_density) if avg_density > 0 else 0,
                    'complexity_handling': max(0, 1 - density_variance / avg_density) if avg_density > 0 else 0.5
                }
            else:
                cognitive_load = {
                    'word_density_variance': 0,
                    'pause_pattern_irregularity': 0,
                    'speed_fluctuation': 0,
                    'complexity_handling': 0.5
                }

            # Quality consistency over time based on real segment data
            time_segments = min(5, len(segments))  # Divide into up to 5 time segments
            consistency_scores = []

            for i, seg_id in enumerate(sorted(segments)[:time_segments]):
                seg_data = asr_data[asr_data['segment_id'] == seg_id]
                seg_en = seg_data[seg_data['lang'] == 'en']
                seg_zh = seg_data[seg_data['lang'] == 'zh']

                # Calculate segment quality score
                has_both_langs = len(seg_en) > 0 and len(seg_zh) > 0
                word_balance = min(len(seg_en), len(seg_zh)) / max(len(seg_en), len(seg_zh), 1)
                segment_score = (0.7 if has_both_langs else 0.3) + (0.3 * word_balance)
                consistency_scores.append(segment_score)
            quality_variance = np.std(consistency_scores) if len(consistency_scores) > 1 else 0

            # Interpreter fatigue indicators
            fatigue_indicators = {
                'early_performance': np.mean(consistency_scores[:2]) if len(consistency_scores) >= 2 else 0,
                'late_performance': np.mean(consistency_scores[-2:]) if len(consistency_scores) >= 2 else 0,
                'performance_decline': max(0, np.mean(consistency_scores[:2]) - np.mean(consistency_scores[-2:])) if len(consistency_scores) >= 4 else 0
            }

        else:
            processing_efficiency = 0
            cognitive_load = {'error': 'Insufficient data'}
            quality_variance = 0
            fatigue_indicators = {'error': 'Insufficient data'}

        return {
            'processing_efficiency': processing_efficiency,
            'cognitive_load_indicators': cognitive_load,
            'quality_consistency': {
                'variance': quality_variance,
                'consistency_score': 1 - min(quality_variance, 1)
            },
            'fatigue_analysis': fatigue_indicators,
            'overall_performance_score': processing_efficiency * 0.4 + (1 - quality_variance) * 0.3 + (1 - min(quality_variance, 1)) * 0.3
        }

    except Exception as e:
        logger.error(f"Advanced performance metrics calculation failed: {str(e)}")
        return {'error': f"Advanced performance metrics calculation failed: {str(e)}"}

def generate_improvement_recommendations(analysis_results):
    """Generate personalized improvement recommendations based on analysis results"""
    recommendations = []

    try:
        # Translation accuracy recommendations
        if 'translation_accuracy' in analysis_results:
            accuracy = analysis_results['translation_accuracy']
            if accuracy.get('semantic_accuracy', 0) < 0.8:
                recommendations.append({
                    'category': 'Translation Accuracy',
                    'priority': 'High',
                    'recommendation': 'Focus on semantic accuracy - consider slowing down to ensure meaning preservation',
                    'specific_actions': ['Practice active listening', 'Use note-taking techniques', 'Work on glossary building']
                })

        # Terminology recommendations
        if 'terminology_analysis' in analysis_results:
            terminology = analysis_results['terminology_analysis']
            if terminology.get('consistency_score', 0) < 0.8:
                recommendations.append({
                    'category': 'Terminology',
                    'priority': 'Medium',
                    'recommendation': 'Improve terminology consistency through specialized vocabulary building',
                    'specific_actions': ['Create domain-specific glossaries', 'Practice technical terminology', 'Use consistent translations for key terms']
                })

        # Performance recommendations
        if 'performance_metrics' in analysis_results:
            performance = analysis_results['performance_metrics']
            if performance.get('processing_efficiency', 0) < 0.7:
                recommendations.append({
                    'category': 'Performance',
                    'priority': 'High',
                    'recommendation': 'Work on processing speed and efficiency',
                    'specific_actions': ['Practice shadowing exercises', 'Improve working memory', 'Develop anticipation skills']
                })

        # Cultural adaptation recommendations
        if 'cultural_analysis' in analysis_results:
            cultural = analysis_results['cultural_analysis']
            if cultural.get('adaptation_score', 0) < 0.8:
                recommendations.append({
                    'category': 'Cultural Adaptation',
                    'priority': 'Medium',
                    'recommendation': 'Enhance cultural sensitivity and localization skills',
                    'specific_actions': ['Study cultural differences', 'Practice cultural adaptation', 'Learn localization techniques']
                })

        # If no specific issues found, provide general recommendations
        if not recommendations:
            recommendations.append({
                'category': 'General',
                'priority': 'Low',
                'recommendation': 'Maintain excellent performance and continue professional development',
                'specific_actions': ['Regular practice sessions', 'Stay updated with domain knowledge', 'Peer review and feedback']
            })

        return recommendations

    except Exception as e:
        logger.error(f"Recommendation generation failed: {str(e)}")
        return [{'category': 'Error', 'priority': 'High', 'recommendation': f'Error generating recommendations: {str(e)}'}]

def display_advanced_analysis_results(results):
    """Display advanced analysis results in an organized format"""
    if 'error' in results:
        st.error(f"Advanced analysis error: {results['error']}")
        return

    # Create tabs for different analysis aspects
    adv_tabs = st.tabs([
        "🎯 Translation Accuracy",
        "🔍 Terminology Analysis",
        "📊 Discourse Analysis",
        "🌐 Cultural Adaptation",
        "⚡ Performance Metrics",
        "💡 Recommendations"
    ])

    # Translation Accuracy Tab
    with adv_tabs[0]:
        st.subheader("Translation Accuracy Assessment")

        accuracy_data = results.get('translation_accuracy', {})
        if 'error' not in accuracy_data:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Semantic Accuracy", f"{accuracy_data.get('semantic_accuracy', 0):.1%}")
            with col2:
                st.metric("Coverage Rate", f"{accuracy_data.get('coverage_rate', 0):.1f}%")
            with col3:
                grade = accuracy_data.get('accuracy_grade', 'Unknown')
                st.metric("Accuracy Grade", grade)

            # Content preservation breakdown
            if 'content_preservation' in accuracy_data:
                st.subheader("Content Preservation Analysis")
                preservation = accuracy_data['content_preservation']

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Factual Accuracy", f"{preservation.get('factual_accuracy', 0):.1%}")
                    st.metric("Technical Terms", f"{preservation.get('technical_terms', 0):.1%}")
                with col2:
                    st.metric("Emotional Tone", f"{preservation.get('emotional_tone', 0):.1%}")
                    st.metric("Cultural Context", f"{preservation.get('cultural_context', 0):.1%}")

                        # Complete translation analysis table
            if 'translation_pairs' in accuracy_data and accuracy_data['translation_pairs']:
                st.subheader("Complete Translation Analysis")

                # DEBUG: Show data source information
                st.info(f"🔍 数据来源调试: 分析时间戳 {results.get('analysis_timestamp', 'Unknown')}")

                # Check first pair to verify data format
                first_pair = accuracy_data['translation_pairs'][0]
                first_segment_id = first_pair.get('segment_id')

                # DEBUG: Show actual segment_id content
                st.code(f"第一个segment_id: {repr(first_segment_id)} (类型: {type(first_segment_id)})")

                if isinstance(first_segment_id, str) and '-' in str(first_segment_id):
                    st.success("✅ 使用新的时间映射数据 (EN#-ZH# 格式)")
                else:
                    st.warning("⚠️ 仍在使用旧的segment_id数据")
                    if isinstance(first_segment_id, str):
                        st.info(f"字符串格式但没有'-': {first_segment_id}")
                    else:
                        st.info(f"非字符串类型: {type(first_segment_id)}")

                    # Additional debug info
                    st.error("🔍 新的时间映射算法失败，查看服务器日志获取详细错误信息")

                # Create DataFrame for all translation pairs
                import pandas as pd

                pairs_data = []
                for pair in accuracy_data['translation_pairs']:
                    pairs_data.append({
                        'Segment ID': pair.get('segment_id', 0),
                        'English Text': pair.get('english', '')[:100] + ('...' if len(pair.get('english', '')) > 100 else ''),
                        'Chinese Text': pair.get('chinese', '')[:50] + ('...' if len(pair.get('chinese', '')) > 50 else ''),
                        'EN Words': pair.get('en_word_count', 0),
                        'ZH Words': pair.get('zh_word_count', 0),
                        'Length Ratio': f"{pair.get('length_ratio', 0):.2f}",
                        'Time Alignment': f"{pair.get('time_alignment', 0):.1%}",
                        'Quality Score': f"{(pair.get('time_alignment', 0) + min(pair.get('length_ratio', 0), 1.0)) / 2:.1%}"
                    })

                if pairs_data:
                    df_pairs = pd.DataFrame(pairs_data)

                    # Add filtering and sorting options
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        min_quality = st.slider("Minimum Quality Score", 0.0, 1.0, 0.0, 0.1)
                    with col2:
                        sort_by = st.selectbox("Sort by", ['Segment ID', 'Quality Score', 'Length Ratio', 'Time Alignment'])
                    with col3:
                        sort_order = st.selectbox("Order", ['Ascending', 'Descending'])

                    # Apply filters and sorting
                    df_filtered = df_pairs.copy()

                    # Convert quality score back to float for filtering
                    df_filtered['Quality_Numeric'] = [float(q.strip('%'))/100 for q in df_filtered['Quality Score']]
                    df_filtered = df_filtered[df_filtered['Quality_Numeric'] >= min_quality]

                    # Sort the dataframe
                    sort_column_map = {
                        'Segment ID': 'Segment ID',
                        'Quality Score': 'Quality_Numeric',
                        'Length Ratio': 'Length Ratio',
                        'Time Alignment': 'Time Alignment'
                    }

                    if sort_by in sort_column_map:
                        ascending = sort_order == 'Ascending'
                        if sort_by == 'Length Ratio':
                            df_filtered['Length_Numeric'] = df_filtered['Length Ratio'].astype(float)
                            df_filtered = df_filtered.sort_values('Length_Numeric', ascending=ascending)
                        elif sort_by == 'Time Alignment':
                            df_filtered['Alignment_Numeric'] = [float(a.strip('%'))/100 for a in df_filtered['Time Alignment']]
                            df_filtered = df_filtered.sort_values('Alignment_Numeric', ascending=ascending)
                        else:
                            df_filtered = df_filtered.sort_values(sort_column_map[sort_by], ascending=ascending)

                    # Remove helper columns before display
                    display_columns = ['Segment ID', 'English Text', 'Chinese Text', 'EN Words', 'ZH Words', 'Length Ratio', 'Time Alignment', 'Quality Score']
                    df_display = df_filtered[display_columns]

                    st.write(f"Showing {len(df_display)} of {len(df_pairs)} translation pairs")

                    # Display the table with styling
                    st.dataframe(
                        df_display,
                        width='stretch',
                        hide_index=True,
                        column_config={
                            'Segment ID': st.column_config.TextColumn('Segment', width="small"),
                            'English Text': st.column_config.TextColumn('English', width="large"),
                            'Chinese Text': st.column_config.TextColumn('Chinese', width="large"),
                            'EN Words': st.column_config.NumberColumn('EN Words', format="%d"),
                            'ZH Words': st.column_config.NumberColumn('ZH Words', format="%d"),
                            'Length Ratio': st.column_config.TextColumn('Ratio'),
                            'Time Alignment': st.column_config.TextColumn('Alignment'),
                            'Quality Score': st.column_config.TextColumn('Quality')
                        }
                    )

                    # Summary statistics
                    st.subheader("Translation Analysis Summary")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        avg_en_words = sum(pair.get('en_word_count', 0) for pair in accuracy_data['translation_pairs']) / len(accuracy_data['translation_pairs'])
                        st.metric("Avg EN Words/Segment", f"{avg_en_words:.1f}")

                    with col2:
                        avg_zh_words = sum(pair.get('zh_word_count', 0) for pair in accuracy_data['translation_pairs']) / len(accuracy_data['translation_pairs'])
                        st.metric("Avg ZH Words/Segment", f"{avg_zh_words:.1f}")

                    with col3:
                        avg_ratio = sum(pair.get('length_ratio', 0) for pair in accuracy_data['translation_pairs']) / len(accuracy_data['translation_pairs'])
                        st.metric("Avg Length Ratio", f"{avg_ratio:.2f}")

                    with col4:
                        avg_alignment = sum(pair.get('time_alignment', 0) for pair in accuracy_data['translation_pairs']) / len(accuracy_data['translation_pairs'])
                        st.metric("Avg Time Alignment", f"{avg_alignment:.1%}")

                    # Export functionality
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📊 Download Translation Analysis (CSV)", type="secondary"):
                            csv = df_pairs.to_csv(index=False)
                            st.download_button(
                                label="💾 Download CSV File",
                                data=csv,
                                file_name=f"translation_analysis_{results.get('file_name', 'data')}.csv",
                                mime="text/csv"
                            )

                    with col2:
                        # Quality distribution chart
                        if st.button("📈 Show Quality Distribution", type="secondary"):
                            quality_scores = [(pair.get('time_alignment', 0) + min(pair.get('length_ratio', 0), 1.0)) / 2
                                            for pair in accuracy_data['translation_pairs']]

                            import plotly.graph_objects as go
                            fig = go.Figure(data=[go.Histogram(x=quality_scores, nbinsx=10)])
                            fig.update_layout(
                                title="Translation Quality Score Distribution",
                                xaxis_title="Quality Score",
                                yaxis_title="Number of Segments"
                            )
                            st.plotly_chart(fig, width='stretch')
                else:
                    st.info("No translation pairs found for analysis")
        else:
            st.error(accuracy_data['error'])

    # Terminology Analysis Tab
    with adv_tabs[1]:
        st.subheader("Terminology Consistency Analysis")

        terminology_data = results.get('terminology_analysis', {})
        if 'error' not in terminology_data:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Consistency Score", f"{terminology_data.get('consistency_score', 0):.1%}")
            with col2:
                st.metric("Dominant Domain", terminology_data.get('dominant_domain', 'Unknown'))
            with col3:
                st.metric("Unique Terms", terminology_data.get('total_unique_terms', 0))

            # Domain distribution
            if 'domain_distribution' in terminology_data:
                st.subheader("Domain Distribution")
                domain_dist = terminology_data['domain_distribution']

                # Create bar chart for domain distribution
                import plotly.graph_objects as go
                fig = go.Figure(data=[
                    go.Bar(x=list(domain_dist.keys()), y=list(domain_dist.values()))
                ])
                fig.update_layout(title="Domain-Specific Term Usage", xaxis_title="Domain", yaxis_title="Term Count")
                st.plotly_chart(fig, width='stretch')

            # Most common terms
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("English Terms")
                en_terms = terminology_data.get('english_terms', [])
                for term, freq in en_terms[:5]:
                    st.write(f"• {term} ({freq})")

            with col2:
                st.subheader("Chinese Terms")
                zh_terms = terminology_data.get('chinese_terms', [])
                for term, freq in zh_terms[:5]:
                    st.write(f"• {term} ({freq})")
        else:
            st.error(terminology_data['error'])

    # Discourse Analysis Tab
    with adv_tabs[2]:
        st.subheader("Discourse Structure Analysis")

        discourse_data = results.get('discourse_analysis', {})
        if 'error' not in discourse_data:
            st.metric("Overall Coherence", f"{discourse_data.get('overall_coherence', 0):.1%}")

            # Coherence metrics
            if 'coherence_metrics' in discourse_data:
                st.subheader("Coherence Breakdown")
                coherence = discourse_data['coherence_metrics']

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Logical Flow", f"{coherence.get('logical_flow', 0):.1%}")
                    st.metric("Thematic Unity", f"{coherence.get('thematic_unity', 0):.1%}")
                with col2:
                    st.metric("Temporal Consistency", f"{coherence.get('temporal_consistency', 0):.1%}")
                    st.metric("Transition Quality", f"{coherence.get('transition_quality', 0):.1%}")

            # Information density
            if 'information_density' in discourse_data:
                st.subheader("Information Density Analysis")
                density = discourse_data['information_density']

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("English Density", f"{density.get('english_density', 0):.1f} words/segment")
                with col2:
                    st.metric("Chinese Density", f"{density.get('chinese_density', 0):.1f} words/segment")
                with col3:
                    st.metric("Balance Score", f"{density.get('balance_score', 0):.1%}")
        else:
            st.error(discourse_data['error'])

    # Cultural Adaptation Tab
    with adv_tabs[3]:
        st.subheader("Cultural Adaptation Analysis")

        cultural_data = results.get('cultural_analysis', {})
        if 'error' not in cultural_data:
            st.metric("Cultural Adaptation Score", f"{cultural_data.get('adaptation_score', 0):.1%}")

            # Cultural metrics breakdown
            if 'cultural_metrics' in cultural_data:
                st.subheader("Cultural Metrics")
                metrics = cultural_data['cultural_metrics']

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Cultural Sensitivity", f"{metrics.get('cultural_sensitivity', 0):.1%}")
                    st.metric("Idiom Adaptation", f"{metrics.get('idiom_adaptation', 0):.1%}")
                with col2:
                    st.metric("Localization Quality", f"{metrics.get('localization_quality', 0):.1%}")
                    st.metric("Context Appropriateness", f"{metrics.get('context_appropriateness', 0):.1%}")

            # Recommendations
            if 'localization_recommendations' in cultural_data:
                st.subheader("Localization Recommendations")
                for rec in cultural_data['localization_recommendations']:
                    st.write(f"• {rec}")
        else:
            st.error(cultural_data['error'])

    # Performance Metrics Tab
    with adv_tabs[4]:
        st.subheader("Advanced Performance Metrics")

        performance_data = results.get('performance_metrics', {})
        if 'error' not in performance_data:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Processing Efficiency", f"{performance_data.get('processing_efficiency', 0):.1%}")
            with col2:
                if 'quality_consistency' in performance_data:
                    consistency = performance_data['quality_consistency']
                    st.metric("Quality Consistency", f"{consistency.get('consistency_score', 0):.1%}")
            with col3:
                st.metric("Overall Performance", f"{performance_data.get('overall_performance_score', 0):.1%}")

            # Cognitive load indicators
            if 'cognitive_load_indicators' in performance_data and 'error' not in performance_data['cognitive_load_indicators']:
                st.subheader("Cognitive Load Analysis")
                cognitive = performance_data['cognitive_load_indicators']

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Word Density Variance", f"{cognitive.get('word_density_variance', 0):.2f}")
                    st.metric("Speed Fluctuation", f"{cognitive.get('speed_fluctuation', 0):.2f}")
                with col2:
                    st.metric("Pause Pattern Irregularity", f"{cognitive.get('pause_pattern_irregularity', 0):.2f}")
                    st.metric("Complexity Handling", f"{cognitive.get('complexity_handling', 0):.1%}")

            # Fatigue analysis
            if 'fatigue_analysis' in performance_data and 'error' not in performance_data['fatigue_analysis']:
                st.subheader("Interpreter Fatigue Analysis")
                fatigue = performance_data['fatigue_analysis']

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Early Performance", f"{fatigue.get('early_performance', 0):.1%}")
                with col2:
                    st.metric("Late Performance", f"{fatigue.get('late_performance', 0):.1%}")
                with col3:
                    decline = fatigue.get('performance_decline', 0)
                    st.metric("Performance Decline", f"{decline:.1%}",
                             delta=f"{-decline:.1%}" if decline > 0 else None)
        else:
            st.error(performance_data['error'])

    # Recommendations Tab
    with adv_tabs[5]:
        st.subheader("Improvement Recommendations")

        recommendations = results.get('recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations):
                priority_color = {
                    'High': '🔴',
                    'Medium': '🟡',
                    'Low': '🟢'
                }.get(rec.get('priority', 'Medium'), '⚪')

                # Use markdown containers instead of expanders
                st.markdown(f"### {priority_color} {rec.get('category', 'General')} - {rec.get('priority', 'Medium')} Priority")
                st.write(f"**Recommendation:** {rec.get('recommendation', 'No recommendation provided')}")

                if 'specific_actions' in rec and rec['specific_actions']:
                    st.write("**Specific Actions:**")
                    for action in rec['specific_actions']:
                        st.write(f"• {action}")
                st.markdown("---")
        else:
            st.info("No specific recommendations generated")

    # Export advanced results
    st.markdown("---")
    if st.button("📊 Export Advanced Analysis Report", type="secondary"):
        try:
            import json
            from datetime import datetime

            # Prepare export data
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'file_name': results.get('file_name', 'unknown'),
                'analysis_results': results
            }

            json_str = json.dumps(export_data, indent=2, ensure_ascii=False, default=str)

            st.download_button(
                label="💾 Download Report (JSON)",
                data=json_str,
                file_name=f"advanced_si_analysis_{results.get('file_name', 'report')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            st.success("Advanced analysis report ready for download!")

        except Exception as e:
            st.error(f"Failed to generate report: {str(e)}")

def show_nlp_results_review(file_name: str, asr_provider: str):
    """
    Show comprehensive NLP results review and comparison

    Args:
        file_name: File name
        asr_provider: ASR provider
    """
    try:
        st.subheader("🔍 中文NLP结果检视")

        # Get NLP results from database
        with EVSDataUtils.get_db_connection() as conn:
            query = """
            SELECT edit_word, nlp_word, nlp_pos, nlp_confidence, nlp_engine,
                   nlp_engine_info, nlp_processed_at, nlp_comparison
            FROM asr_results_words
            WHERE file_name = ? AND asr_provider = ? AND lang = 'zh'
                AND nlp_word IS NOT NULL
            ORDER BY segment_id, word_seq_no
            LIMIT 100
            """

            cursor = conn.cursor()
            cursor.execute(query, [file_name, asr_provider])
            results = cursor.fetchall()

            if not results:
                st.warning("未找到NLP处理结果。请先运行中文NLP分词。")
                return

            # Convert to DataFrame for easier handling
            df = pd.DataFrame(results, columns=[
                'original_word', 'nlp_word', 'nlp_pos', 'nlp_confidence',
                'nlp_engine', 'nlp_engine_info', 'nlp_processed_at', 'nlp_comparison'
            ])

            # Show summary statistics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("处理词数", len(df))

            with col2:
                engine_used = df['nlp_engine'].iloc[0] if not df.empty else "未知"
                st.metric("使用引擎", engine_used)

            with col3:
                avg_confidence = df['nlp_confidence'].mean() if not df.empty else 0
                st.metric("平均置信度", f"{avg_confidence:.3f}")

            with col4:
                processed_time = df['nlp_processed_at'].iloc[0] if not df.empty else "未知"
                if processed_time != "未知":
                    processed_time = processed_time[:19]  # Remove microseconds
                st.metric("处理时间", processed_time)

            # Show detailed results table
            st.subheader("📋 详细结果对比")

            # Create comparison DataFrame
            comparison_df = df[['original_word', 'nlp_word', 'nlp_pos', 'nlp_confidence']].copy()
            comparison_df.columns = ['原始词', 'NLP分词', '词性', '置信度']

            # Add improvement indicator
            comparison_df['改进状态'] = comparison_df.apply(
                lambda row: "✅ 改进" if row['原始词'] != row['NLP分词'] else "⚪ 相同", axis=1
            )

            # Show interactive table
            st.dataframe(
                comparison_df,
                width='stretch',
                height=400
            )

            # Show engine comparison option
            st.subheader("⚖️ 引擎对比")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("🔄 对比jieba vs HanLP", key='btn_compare_engines'):
                    st.session_state['show_nlp_comparison'] = True
                    st.rerun()

            with col2:
                if st.button("📊 生成测试报告", key='btn_generate_report'):
                    generate_nlp_test_report(file_name, asr_provider, df)

            # Show word frequency analysis
            st.subheader("📈 词频分析")

            from collections import Counter

            # Original words frequency
            original_freq = Counter(df['original_word'].tolist())
            nlp_freq = Counter(df['nlp_word'].tolist())

            col1, col2 = st.columns(2)

            with col1:
                st.write("**原始词频 (前10)**")
                original_top = pd.DataFrame(original_freq.most_common(10), columns=['词汇', '频次'])
                st.dataframe(original_top, width='stretch')

            with col2:
                st.write("**NLP词频 (前10)**")
                nlp_top = pd.DataFrame(nlp_freq.most_common(10), columns=['词汇', '频次'])
                st.dataframe(nlp_top, width='stretch')

            # Show POS distribution
            st.subheader("🏷️ 词性分布")

            pos_freq = Counter(df['nlp_pos'].tolist())
            pos_df = pd.DataFrame(list(pos_freq.items()), columns=['词性', '数量'])
            pos_df = pos_df.sort_values('数量', ascending=False)

            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(pos_df, width='stretch')

            with col2:
                st.bar_chart(pos_df.set_index('词性'))

    except Exception as e:
        logger.error(f"Error showing NLP results review: {str(e)}")
        st.error(f"显示NLP结果时出错: {str(e)}")

def show_nlp_engine_comparison(file_name: str, asr_provider: str):
    """
    Show comparison between jieba and HanLP engines

    Args:
        file_name: File name
        asr_provider: ASR provider
    """
    try:
        st.subheader("⚖️ NLP引擎对比分析")

        # Get sample text for comparison
        with EVSDataUtils.get_db_connection() as conn:
            query = """
            SELECT GROUP_CONCAT(edit_word, ' ') as full_text
            FROM asr_results_words
            WHERE file_name = ? AND asr_provider = ? AND lang = 'zh'
            LIMIT 50
            """

            cursor = conn.cursor()
            cursor.execute(query, [file_name, asr_provider])
            result = cursor.fetchone()

            if not result or not result[0]:
                st.warning("未找到中文文本进行对比")
                st.session_state['show_nlp_comparison'] = False
                return

            sample_text = result[0]

        st.write(f"**测试文本 (前200字符):** {sample_text[:200]}...")

        # Compare engines
        with st.spinner("正在对比两个NLP引擎..."):
            try:
                from chinese_nlp_unified import ChineseNLPUnified, NLPEngine

                # Create processors
                jieba_processor = ChineseNLPUnified(NLPEngine.JIEBA)
                hanlp_processor = ChineseNLPUnified(NLPEngine.HANLP)

                # Process with both engines
                jieba_words = jieba_processor.segment_text(sample_text)
                jieba_pos = jieba_processor.pos_tag(sample_text)

                hanlp_words = hanlp_processor.segment_text(sample_text)
                hanlp_pos = hanlp_processor.pos_tag(sample_text)

                # Show comparison results
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🔤 jieba 结果")
                    st.metric("分词数量", len(jieba_words))
                    st.write("**分词结果 (前20个):**")
                    st.write(" | ".join(jieba_words[:20]))

                    st.write("**词性标注 (前10个):**")
                    jieba_pos_df = pd.DataFrame(jieba_pos[:10], columns=['词汇', '词性'])
                    st.dataframe(jieba_pos_df, width='stretch')

                with col2:
                    st.subheader("🧠 HanLP 结果")
                    st.metric("分词数量", len(hanlp_words))
                    st.write("**分词结果 (前20个):**")
                    st.write(" | ".join(hanlp_words[:20]))

                    st.write("**词性标注 (前10个):**")
                    hanlp_pos_df = pd.DataFrame(hanlp_pos[:10], columns=['词汇', '词性'])
                    st.dataframe(hanlp_pos_df, width='stretch')

                # Show difference analysis
                st.subheader("📊 差异分析")

                col1, col2, col3 = st.columns(3)

                with col1:
                    word_diff = abs(len(jieba_words) - len(hanlp_words))
                    st.metric("分词数量差异", word_diff)

                with col2:
                    # Calculate overlap
                    jieba_set = set(jieba_words)
                    hanlp_set = set(hanlp_words)
                    overlap = len(jieba_set.intersection(hanlp_set))
                    total_unique = len(jieba_set.union(hanlp_set))
                    overlap_ratio = overlap / total_unique if total_unique > 0 else 0
                    st.metric("词汇重叠率", f"{overlap_ratio:.2%}")

                with col3:
                    # Performance recommendation
                    if len(jieba_words) < len(hanlp_words):
                        st.metric("推荐引擎", "jieba (更简洁)")
                    elif len(hanlp_words) < len(jieba_words):
                        st.metric("推荐引擎", "HanLP (更细致)")
                    else:
                        st.metric("推荐引擎", "相当 (可选任一)")

                # Show unique words
                st.subheader("🔍 独有词汇")

                col1, col2 = st.columns(2)

                with col1:
                    jieba_unique = jieba_set - hanlp_set
                    st.write(f"**jieba独有词汇 ({len(jieba_unique)}个):**")
                    if jieba_unique:
                        st.write(" | ".join(list(jieba_unique)[:10]))
                    else:
                        st.write("无独有词汇")

                with col2:
                    hanlp_unique = hanlp_set - jieba_set
                    st.write(f"**HanLP独有词汇 ({len(hanlp_unique)}个):**")
                    if hanlp_unique:
                        st.write(" | ".join(list(hanlp_unique)[:10]))
                    else:
                        st.write("无独有词汇")

            except Exception as e:
                st.error(f"引擎对比失败: {str(e)}")

        # Close comparison view
        if st.button("关闭对比", key='btn_close_comparison'):
            st.session_state['show_nlp_comparison'] = False
            st.rerun()

    except Exception as e:
        logger.error(f"Error in NLP engine comparison: {str(e)}")
        st.error(f"引擎对比时出错: {str(e)}")
        st.session_state['show_nlp_comparison'] = False

def generate_nlp_test_report(file_name: str, asr_provider: str, results_df: pd.DataFrame):
    """
    Generate a comprehensive NLP test report

    Args:
        file_name: File name
        asr_provider: ASR provider
        results_df: Results DataFrame
    """
    try:
        st.subheader("📋 NLP测试报告")

        # Generate report content
        report_content = f"""
# 中文NLP处理测试报告

## 基本信息
- **文件名**: {file_name}
- **ASR提供商**: {asr_provider}
- **处理时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **处理词数**: {len(results_df)}

## 引擎信息
- **使用引擎**: {results_df['nlp_engine'].iloc[0] if not results_df.empty else '未知'}
- **平均置信度**: {results_df['nlp_confidence'].mean():.3f}

## 处理统计
- **改进词数**: {len(results_df[results_df['original_word'] != results_df['nlp_word']])}
- **改进比例**: {len(results_df[results_df['original_word'] != results_df['nlp_word']]) / len(results_df) * 100:.1f}%

## 词性分布
"""

        # Add POS distribution
        from collections import Counter
        pos_freq = Counter(results_df['nlp_pos'].tolist())
        for pos, count in pos_freq.most_common(10):
            report_content += f"- **{pos}**: {count}个\n"

        report_content += f"""

## 高频词汇 (前10)
"""

        # Add word frequency
        word_freq = Counter(results_df['nlp_word'].tolist())
        for word, count in word_freq.most_common(10):
            report_content += f"- **{word}**: {count}次\n"

        report_content += f"""

## 改进示例
"""

        # Add improvement examples
        improved = results_df[results_df['original_word'] != results_df['nlp_word']].head(5)
        for _, row in improved.iterrows():
            report_content += f"- {row['original_word']} → {row['nlp_word']} ({row['nlp_pos']})\n"

        # Show report
        st.markdown(report_content)

        # Download button
        st.download_button(
            label="📥 下载报告",
            data=report_content,
            file_name=f"nlp_report_{file_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            key='btn_download_report'
        )

    except Exception as e:
        logger.error(f"Error generating NLP test report: {str(e)}")
        st.error(f"生成报告时出错: {str(e)}")

if __name__ == "__main__":
    main()