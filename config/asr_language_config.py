"""
Language-specific ASR Configuration
支持按语言选择不同的ASR模型

This module provides:
1. Language-specific ASR model recommendations
2. Functions to get appropriate ASR model for each language
3. Configuration for dual-ASR processing (CrisperWhisper for EN, FunASR for ZH)
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ASRModelInfo:
    """ASR model information"""
    provider: str           # e.g., 'crisperwhisper', 'funasr'
    model_name: str         # e.g., 'crisperwhisper', 'paraformer-zh'
    display_name: str       # e.g., 'CrisperWhisper'
    languages: List[str]    # Supported languages: ['en'], ['zh'], ['en', 'zh']
    description: str        # Model description
    is_recommended: bool    # Whether recommended for the language


# =============================================================================
# ASR Model Definitions
# =============================================================================

# CrisperWhisper (nyrahealth) - Verbatim English ASR with precise word timestamps
CRISPERWHISPER_MODELS = {
    "crisperwhisper": ASRModelInfo(
        provider="crisperwhisper",
        model_name="crisperwhisper",
        display_name="CrisperWhisper",
        languages=["en"],
        description="Verbatim English ASR with crisp word-level timestamps, filler/stutter/false-start detection. ~3GB VRAM",
        is_recommended=True
    ),
}

# FunASR Models (Alibaba) - Excellent for Chinese
FUNASR_MODELS = {
    "paraformer-zh": ASRModelInfo(
        provider="funasr",
        model_name="paraformer-zh",
        display_name="Paraformer Chinese",
        languages=["zh"],
        description="Best Chinese ASR. Non-autoregressive, fast & accurate",
        is_recommended=True
    ),
    "paraformer-en": ASRModelInfo(
        provider="funasr",
        model_name="paraformer-en",
        display_name="Paraformer English",
        languages=["en"],
        description="Paraformer for English. Fast & accurate",
        is_recommended=False
    ),
    "sensevoice-small": ASRModelInfo(
        provider="funasr",
        model_name="SenseVoiceSmall",
        display_name="SenseVoice Small",
        languages=["en", "zh"],
        description="Multilingual ASR with emotion detection. Fast",
        is_recommended=False
    ),
}


# =============================================================================
# Language-specific Recommendations
# =============================================================================

# Recommended ASR models by language
LANGUAGE_ASR_RECOMMENDATIONS = {
    "en": {
        "recommended": [
            ("crisperwhisper", "crisperwhisper", "Verbatim transcription with precise word timestamps, filler/stutter detection"),
        ],
        "alternatives": [
            ("funasr", "paraformer-en", "Fast non-autoregressive model"),
        ],
        "default_provider": "crisperwhisper",
        "default_model": "crisperwhisper",
    },
    "zh": {
        "recommended": [
            ("funasr", "paraformer-zh", "BEST for Chinese - highly recommended"),
        ],
        "alternatives": [
            ("funasr", "sensevoice-small", "Multilingual with emotion detection"),
        ],
        "default_provider": "funasr",
        "default_model": "paraformer-zh",
        "notes": """
Chinese ASR Recommendations:
1. FunASR Paraformer (RECOMMENDED)
   - Install: pip install funasr modelscope torch torchaudio
   - Best accuracy for Chinese speech recognition
   - Non-autoregressive model = faster inference
   - Supports punctuation and timestamps

2. SenseVoice
   - Also from FunASR ecosystem
   - Supports emotion detection
   - Good for multimodal analysis
"""
    }
}


# =============================================================================
# Default Configuration
# =============================================================================

# Whether to use different ASR models for different languages
ENABLE_LANGUAGE_SPECIFIC_ASR = True

# Default ASR settings per language
DEFAULT_LANGUAGE_ASR = {
    "en": {
        "provider": "crisperwhisper",
        "model": "crisperwhisper"
    },
    "zh": {
        "provider": "funasr",
        "model": "paraformer-zh"
    }
}


# =============================================================================
# Utility Functions
# =============================================================================

def get_available_models_for_language(lang: str) -> List[ASRModelInfo]:
    """
    Get all available ASR models for a specific language

    Args:
        lang: Language code ('en' or 'zh')

    Returns:
        List of ASRModelInfo objects supporting the language
    """
    models = []

    # Add CrisperWhisper models
    for model in CRISPERWHISPER_MODELS.values():
        if lang in model.languages:
            models.append(model)

    # Add FunASR models
    for model in FUNASR_MODELS.values():
        if lang in model.languages:
            models.append(model)

    return models


def get_recommended_model_for_language(lang: str) -> Tuple[str, str]:
    """
    Get the recommended ASR provider and model for a language

    Args:
        lang: Language code ('en' or 'zh')

    Returns:
        Tuple of (provider, model_name)
    """
    if lang in LANGUAGE_ASR_RECOMMENDATIONS:
        rec = LANGUAGE_ASR_RECOMMENDATIONS[lang]
        return (rec["default_provider"], rec["default_model"])

    # Fallback: CrisperWhisper for EN, FunASR for ZH
    if lang == "zh":
        return ("funasr", "paraformer-zh")
    return ("crisperwhisper", "crisperwhisper")


def get_model_options_for_ui(lang: str, provider: str = None) -> List[Dict]:
    """
    Get model options formatted for Streamlit UI

    Args:
        lang: Language code ('en' or 'zh')
        provider: Optional provider filter

    Returns:
        List of dicts with 'value', 'label', 'description' keys
    """
    models = get_available_models_for_language(lang)

    if provider:
        models = [m for m in models if m.provider == provider]

    options = []
    for model in models:
        rec_tag = " [RECOMMENDED]" if model.is_recommended else ""
        options.append({
            "value": f"{model.provider}:{model.model_name}",
            "label": f"{model.display_name}{rec_tag}",
            "description": model.description,
            "provider": model.provider,
            "model_name": model.model_name,
            "is_recommended": model.is_recommended
        })

    # Sort with recommended first
    options.sort(key=lambda x: (not x["is_recommended"], x["label"]))

    return options


def get_language_asr_config(lang: str) -> Dict:
    """
    Get the current ASR configuration for a language

    Args:
        lang: Language code ('en' or 'zh')

    Returns:
        Dict with 'provider' and 'model' keys
    """
    return DEFAULT_LANGUAGE_ASR.get(lang, {"provider": "crisperwhisper", "model": "crisperwhisper"})


def set_language_asr_config(lang: str, provider: str, model: str):
    """
    Set the ASR configuration for a language (in-memory)

    Args:
        lang: Language code ('en' or 'zh')
        provider: ASR provider name
        model: Model name
    """
    DEFAULT_LANGUAGE_ASR[lang] = {
        "provider": provider,
        "model": model
    }


def get_chinese_asr_recommendations() -> str:
    """
    Get formatted Chinese ASR recommendations text

    Returns:
        Markdown-formatted recommendations text
    """
    return """
## Chinese ASR Model Recommendations

### 1. FunASR Paraformer (HIGHLY RECOMMENDED)
- **Best accuracy** for Chinese speech recognition
- Non-autoregressive model = **faster inference**
- Excellent punctuation and timestamp support
- Installation:
  ```bash
  pip install funasr modelscope torch torchaudio
  ```
- Model: `paraformer-zh`

### 2. SenseVoice
- Multilingual ASR from Alibaba FunASR
- Supports emotion detection and event recognition
- Good for multimodal analysis
- Model: `SenseVoiceSmall`

### Quick Comparison

| Model | Chinese Accuracy | Speed | VRAM | Notes |
|-------|-----------------|-------|------|-------|
| Paraformer-ZH | ★★★★★ | ★★★★☆ | ~2GB | Best for Chinese |
| SenseVoice | ★★★★☆ | ★★★★☆ | ~2GB | With emotion detection |

### Recommendation Summary
- For **best Chinese accuracy**: Use FunASR Paraformer-ZH
- For **multimodal analysis**: Consider SenseVoice
"""


def check_crisperwhisper_available() -> bool:
    """
    Check if CrisperWhisper (HuggingFace transformers) is installed and available

    Returns:
        True if transformers pipeline is available, False otherwise
    """
    try:
        from transformers import pipeline
        return True
    except ImportError:
        return False


def check_funasr_available() -> bool:
    """
    Check if FunASR is installed and available

    Returns:
        True if FunASR is available, False otherwise
    """
    try:
        import funasr
        return True
    except ImportError:
        return False


def get_available_providers() -> List[str]:
    """
    Get list of available ASR providers based on installed packages

    Returns:
        List of available provider names
    """
    providers = []

    if check_crisperwhisper_available():
        providers.append("crisperwhisper")

    if check_funasr_available():
        providers.append("funasr")

    return providers
