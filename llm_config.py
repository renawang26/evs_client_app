"""
LLM Configuration Management for SI Analysis
"""

import json
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Default LLM configuration
DEFAULT_LLM_CONFIG = {
    "active_llm_provider": "ollama",
    "llm_configs": {
        "ollama": {
            "llm_provider": "ollama",
            "llm_model": "gemma3:27b",
            "llm_temperature": 0.2,
            "llm_max_tokens": 40960,
            "llm_base_url": "http://localhost:11434",
            "llm_request_timeout": 300
        },
        "openai": {
            "llm_provider": "openai",
            "llm_model": "gpt-3.5-turbo",
            "llm_temperature": 0.2,
            "llm_max_tokens": 4096,
            "llm_base_url": "https://api.openai.com/v1",
            "llm_request_timeout": 300,
            "llm_api_key": ""
        }
    }
}

# Default analysis rules
DEFAULT_ANALYSIS_RULES = {
    "quality_metrics": {
        "accuracy_weight": 0.4,
        "fluency_weight": 0.3,
        "completeness_weight": 0.3,
        "scoring_thresholds": {
            "excellent": 0.9,
            "good": 0.7,
            "acceptable": 0.5,
            "poor": 0.3
        }
    },
    "timing_analysis": {
        "delay_thresholds": {
            "excellent": 1.0,
            "good": 2.0,
            "acceptable": 3.0,
            "needs_improvement": 5.0
        },
        "speaking_rate": {
            "english_normal": [150, 200],  # words per minute
            "chinese_normal": [180, 220]   # characters per minute
        },
        "processing_time_base": 0.5
    },
    "analysis_settings": {
        "min_segment_length": 3,
        "max_processing_time": 300,
        "confidence_threshold": 0.7,
        "enable_llm_enhancement": True,
        "llm_models": {
            "analysis": "gemma3:27b",
            "correction": "gemma3:27b"
        }
    },
    "error_patterns": [
        {
            "id": "tense_error_001",
            "type": "tense_error",
            "pattern": r"\b(will|shall)\s+have\s+\w+ed\b",
            "description": "Future perfect tense usage",
            "severity": "medium",
            "enabled": True
        },
        {
            "id": "terminology_error_001",
            "type": "terminology_error",
            "pattern": r"\b(economy|economic|economical)\b",
            "description": "Economic terminology confusion",
            "severity": "high",
            "enabled": True
        }
    ],
    "cultural_rules": {
        "idiom_mappings": {
            "break a leg": "祝好运",
            "piece of cake": "小菜一碟",
            "hit the nail on the head": "一针见血"
        },
        "terminology_dictionary": {
            "business": {
                "shareholder": "股东",
                "stakeholder": "利益相关者",
                "ROI": "投资回报率"
            },
            "technology": {
                "artificial intelligence": "人工智能",
                "machine learning": "机器学习",
                "blockchain": "区块链"
            }
        }
    }
}

CONFIG_FILE = Path("config/llm_config.json")
ANALYSIS_RULES_FILE = Path("config/analysis_rules.json")

def ensure_config_dir():
    """Ensure config directory exists"""
    CONFIG_FILE.parent.mkdir(exist_ok=True)
    ANALYSIS_RULES_FILE.parent.mkdir(exist_ok=True)

def load_llm_config():
    """Load LLM configuration from file"""
    try:
        ensure_config_dir()
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Create default config file
            save_llm_config(DEFAULT_LLM_CONFIG)
            return DEFAULT_LLM_CONFIG
    except Exception as e:
        logger.error(f"Failed to load LLM config: {str(e)}")
        return DEFAULT_LLM_CONFIG

def save_llm_config(config):
    """Save LLM configuration to file"""
    try:
        ensure_config_dir()
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save LLM config: {str(e)}")
        return False

def update_llm_config(config):
    """Update LLM configuration"""
    return save_llm_config(config)

def get_active_llm_config():
    """Get configuration for the active LLM provider"""
    try:
        config = load_llm_config()
        active_provider = config.get("active_llm_provider", "ollama")

        if active_provider in config.get("llm_configs", {}):
            return config["llm_configs"][active_provider]
        else:
            logger.warning(f"Active provider {active_provider} not found in config")
            return DEFAULT_LLM_CONFIG["llm_configs"]["ollama"]

    except Exception as e:
        logger.error(f"Failed to get active LLM config: {str(e)}")
        return DEFAULT_LLM_CONFIG["llm_configs"]["ollama"]

def load_analysis_rules():
    """Load analysis rules from file"""
    try:
        ensure_config_dir()
        if ANALYSIS_RULES_FILE.exists():
            with open(ANALYSIS_RULES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Create default rules file
            save_analysis_rules(DEFAULT_ANALYSIS_RULES)
            return DEFAULT_ANALYSIS_RULES
    except Exception as e:
        logger.error(f"Failed to load analysis rules: {str(e)}")
        return DEFAULT_ANALYSIS_RULES

def save_analysis_rules(rules):
    """Save analysis rules to file"""
    try:
        ensure_config_dir()
        with open(ANALYSIS_RULES_FILE, 'w', encoding='utf-8') as f:
            json.dump(rules, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save analysis rules: {str(e)}")
        return False