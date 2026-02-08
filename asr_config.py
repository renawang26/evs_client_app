"""
ASR (Automatic Speech Recognition) configuration file.
This file contains configuration settings for various ASR providers.
"""

# ASR Providers Dictionary - maps provider name to internal identifier
ASR_PROVIDERS = {
    "CrisperWhisper": "crisperwhisper",
    "FunASR": "funasr",
    "Google": "google",
    "Tencent": "tencent",
    "IBM": "ibm"
}

# Languages Dictionary - maps display name to language code
LANGUAGES = {
    "English": "en",
    "Chinese": "zh"
}

# ASR Model Configuration
ASR_CONFIG = {
    # CrisperWhisper configuration (English verbatim ASR)
    "crisperwhisper": {
        "available_models": ["crisperwhisper"],
        "default_model": "crisperwhisper",
        "language_codes": {
            "en": "en"
        }
    },

    # FunASR configuration (Chinese ASR)
    "funasr": {
        "available_models": ["paraformer-zh", "paraformer-en", "SenseVoiceSmall"],
        "default_model": "paraformer-zh",
        "language_codes": {
            "zh": "zh",
            "en": "en"
        }
    },

    # Google Speech-to-Text configuration
    "google": {
        "language_codes": {
            "en": "en-US",
            "zh": "zh-CN"
        },
        "sample_rate": 16000,
        "encoding": "LINEAR16",
        "credentials_file": "./keys/google_credentials.json"
    },

    # Tencent Cloud ASR configuration
    "tencent": {
        "available_models": [
            "zh-CN-yunxiaochen",
            "zh-CN-yunxiaochen-medical",
            "zh-CN-yunxiaochen-medical-2",
            "zh-CN-yunxiaochen-medical-3",
            "zh-CN-yunxiaochen-medical-4"
        ],
        "default_model": "zh-CN-yunxiaochen",
        "engine_type": {
            "en": "16k_en",
            "zh": "16k_zh"
        },
        "api_config_file": "./keys/tencent_cloud_config.json"
    },

    # IBM Watson ASR configuration
    "ibm": {
        "available_models": [
            "en-US_BroadbandModel",
            "en-US_NarrowbandModel",
            "zh-CN_BroadbandModel",
            "zh-CN_NarrowbandModel"
        ],
        "default_model": "en-US_BroadbandModel",
        "language_codes": {
            "en": "en-US",
            "zh": "zh-CN"
        }
    }
}

# Default slice duration (in seconds)
DEFAULT_SLICE_DURATION = 60

# Resource paths
EVS_RESOURCES_PATH = "./evs_resources"
SLICE_AUDIO_PATH = f"{EVS_RESOURCES_PATH}/slice_audio_files"