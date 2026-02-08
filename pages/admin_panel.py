"""
Admin Panel Module - ASR Configuration and Display Settings
"""

import streamlit as st
import json
import time
import logging

from db_utils import EVSDataUtils
from config.display_config import get_file_aliasing_enabled, set_file_aliasing_enabled

logger = logging.getLogger(__name__)


def render_admin_panel():
    """Render admin panel with ASR Configuration"""
    st.subheader("ASR Configuration")

    # Define available ASR providers
    ASR_PROVIDER_INFO = {
        "crisperwhisper": {
            "name": "CrisperWhisper",
            "description": "Verbatim English ASR with precise word-level timestamps, filler/stutter detection",
            "install_command": "pip install transformers torch torchaudio",
            "check_import": "transformers",
            "models": ["crisperwhisper"],
            "default": "crisperwhisper"
        },
        "funasr": {
            "name": "FunASR (Alibaba)",
            "description": "Best accuracy for Chinese speech recognition",
            "install_command": "pip install funasr modelscope torch torchaudio",
            "check_import": "funasr",
            "models": ["paraformer-zh", "paraformer-en", "SenseVoiceSmall"],
            "default": "paraformer-zh"
        }
    }

    def check_provider_installed(provider_key):
        """Check if a provider is installed"""
        info = ASR_PROVIDER_INFO.get(provider_key, {})
        check_import = info.get("check_import", "")
        if not check_import:
            return False
        try:
            __import__(check_import)
            return True
        except ImportError:
            return False

    try:
        # Load current ASR configuration
        asr_config = EVSDataUtils.get_asr_config()
        if not asr_config:
            asr_config = {}

        # Get installed providers
        installed_providers = [k for k in ASR_PROVIDER_INFO.keys() if check_provider_installed(k)]

        # Load current language defaults
        db_lang_config = None
        try:
            db_result = EVSDataUtils.get_asr_config('language_defaults')
            if db_result and 'config' in db_result:
                db_lang_config = db_result['config']
        except:
            pass

        lang_asr_config = db_lang_config or {
            'en': {'provider': 'crisperwhisper', 'model': 'crisperwhisper'},
            'zh': {'provider': 'funasr', 'model': 'paraformer-zh'}
        }

        # === Language Defaults Section ===
        st.markdown("### Language Defaults")

        if not installed_providers:
            st.warning("No ASR providers installed.")
        else:
            col_en, col_zh = st.columns(2)

            with col_en:
                st.markdown("#### English")
                en_current_provider = lang_asr_config.get('en', {}).get('provider', 'crisperwhisper')
                en_provider_idx = installed_providers.index(en_current_provider) if en_current_provider in installed_providers else 0

                en_provider = st.selectbox(
                    "Provider",
                    options=installed_providers,
                    index=en_provider_idx,
                    key="lang_en_provider",
                    format_func=lambda x: ASR_PROVIDER_INFO[x]['name']
                )

                en_models = ASR_PROVIDER_INFO[en_provider]['models']
                en_current_model = lang_asr_config.get('en', {}).get('model', 'small')
                en_model_idx = en_models.index(en_current_model) if en_current_model in en_models else 0

                en_model = st.selectbox(
                    "Model",
                    options=en_models,
                    index=en_model_idx,
                    key="lang_en_model"
                )

            with col_zh:
                st.markdown("#### Chinese")
                zh_current_provider = lang_asr_config.get('zh', {}).get('provider', 'funasr')
                zh_provider_idx = installed_providers.index(zh_current_provider) if zh_current_provider in installed_providers else 0

                zh_provider = st.selectbox(
                    "Provider",
                    options=installed_providers,
                    index=zh_provider_idx,
                    key="lang_zh_provider",
                    format_func=lambda x: ASR_PROVIDER_INFO[x]['name']
                )

                zh_models = ASR_PROVIDER_INFO[zh_provider]['models']
                zh_current_model = lang_asr_config.get('zh', {}).get('model', 'paraformer-zh')
                zh_model_idx = zh_models.index(zh_current_model) if zh_current_model in zh_models else 0

                zh_model = st.selectbox(
                    "Model",
                    options=zh_models,
                    index=zh_model_idx,
                    key="lang_zh_model"
                )

            if st.button("Save Language Defaults", type="primary"):
                new_lang_config = {
                    'enabled': True,
                    'en': {'provider': en_provider, 'model': en_model},
                    'zh': {'provider': zh_provider, 'model': zh_model}
                }
                if EVSDataUtils.save_asr_config('language_defaults', new_lang_config):
                    st.success("Saved!")
                else:
                    st.error("Failed to save")

        st.markdown("---")

        # === NLP Engine Section ===
        st.markdown("### Chinese NLP Engine")

        nlp_config = asr_config.get('nlp', {})
        if isinstance(nlp_config, str):
            try:
                nlp_config = json.loads(nlp_config)
            except:
                nlp_config = {}

        current_nlp_engine = nlp_config.get('engine', 'jieba')

        col1, col2 = st.columns([1, 2])
        with col1:
            nlp_engine = st.selectbox(
                "NLP Engine",
                options=["jieba", "hanlp"],
                index=0 if current_nlp_engine == "jieba" else 1,
                key="nlp_engine"
            )
        with col2:
            if nlp_engine == "jieba":
                st.caption("jieba: Lightweight, fast")
            else:
                st.caption("HanLP: Deep learning, more accurate")

        if st.button("Save NLP Engine"):
            if EVSDataUtils.save_asr_config('nlp', {'engine': nlp_engine}):
                st.success(f"Set to: {nlp_engine}")
            else:
                st.error("Failed to save")

        st.markdown("---")

        # === Display Settings Section ===
        st.markdown("### Display Settings")

        # File Aliasing Toggle
        current_aliasing = get_file_aliasing_enabled()

        file_aliasing = st.checkbox(
            "Enable File Aliasing (Privacy Mode)",
            value=current_aliasing,
            key="file_aliasing_toggle",
            help="When enabled, files are displayed as 'file_01', 'file_02', etc. instead of actual file names"
        )

        if file_aliasing != current_aliasing:
            set_file_aliasing_enabled(file_aliasing)
            if file_aliasing:
                st.success("File aliasing enabled - files will show as file_01, file_02, etc.")
            else:
                st.info("File aliasing disabled - showing original file names")
            st.rerun()

        if file_aliasing:
            st.caption("Files displayed as: file_01, file_02, file_03...")
        else:
            st.caption("Files displayed with original names")

        st.markdown("---")

        # === Provider Status Section ===
        st.markdown("### Provider Status")

        for provider_key, provider_info in ASR_PROVIDER_INFO.items():
            is_installed = check_provider_installed(provider_key)
            status = "Installed" if is_installed else "Not Installed"
            st.markdown(f"**{provider_info['name']}**: {status}")
            if not is_installed:
                st.code(provider_info['install_command'], language="bash")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        st.error(f"Error: {str(e)}")
