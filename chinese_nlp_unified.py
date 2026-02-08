#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Chinese NLP Processor for ASR Post-processing
统一中文NLP处理器，支持jieba和HanLP两种选项

This module provides Chinese text segmentation and POS tagging functionality
to improve ASR results while preserving original data.
"""

import pandas as pd
import logging
import re
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
import sqlite3
from pathlib import Path
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

class NLPEngine(Enum):
    """NLP Engine options"""
    JIEBA = "jieba"
    HANLP = "hanlp"

class ChineseNLPUnified:
    """Unified Chinese NLP processor supporting both jieba and HanLP"""

    def __init__(self, engine: NLPEngine = NLPEngine.JIEBA):
        """
        Initialize the Chinese NLP processor

        Args:
            engine: NLP engine to use (jieba or hanlp)
        """
        self.engine = engine
        self.tokenizer = None
        self.pos_tagger = None
        self.pipeline = None
        self.initialization_error = None  # Store initialization error

        self.setup_nlp_engine()

    def setup_nlp_engine(self):
        """Setup the selected NLP engine"""
        try:
            if self.engine == NLPEngine.JIEBA:
                self._setup_jieba()
            elif self.engine == NLPEngine.HANLP:
                self._setup_hanlp()
            else:
                raise ValueError(f"Unsupported NLP engine: {self.engine}")

            logger.info(f"Chinese NLP processor initialized with {self.engine.value}")

        except Exception as e:
            error_msg = f"Error setting up {self.engine.value}: {str(e)}"
            logger.error(error_msg)
            self.initialization_error = error_msg

            # For HanLP failures, don't fallback - let the caller handle the error
            if self.engine == NLPEngine.HANLP:
                raise Exception(f"HanLP initialization failed: {str(e)}")

            # For jieba failures, re-raise
            if self.engine == NLPEngine.JIEBA:
                raise Exception(f"Jieba initialization failed: {str(e)}")
            else:
                raise

    def _setup_jieba(self):
        """Setup jieba with custom dictionaries"""
        try:
            import jieba
            import jieba.posseg as pseg

            # Store references
            self.jieba = jieba
            self.pseg = pseg

            # Enable parallel processing only on POSIX systems (Linux/macOS)
            import os
            if os.name == 'posix':
                jieba.enable_parallel(4)
                logger.info("Enabled jieba parallel processing")
            else:
                logger.info("Skipping parallel processing on Windows system")

            # Load custom dictionary if exists
            custom_dict_path = Path("./config/chinese_custom_dict.txt")
            if custom_dict_path.exists():
                jieba.load_userdict(str(custom_dict_path))
                logger.info("Loaded custom Chinese dictionary")

            # Add common interpretation terms
            interpretation_terms = [
                ("同声传译", 10, "n"),
                ("口译员", 10, "n"),
                ("翻译质量", 10, "n"),
                ("语音识别", 10, "n"),
                ("早上好", 10, "i"),
                ("下午好", 10, "i"),
                ("晚上好", 10, "i"),
                ("谢谢大家", 10, "i"),
                ("非常感谢", 10, "i"),
                ("各位朋友", 10, "n"),
                ("女士们先生们", 10, "n")
            ]

            for word, freq, pos in interpretation_terms:
                jieba.add_word(word, freq, pos)

            logger.info("Jieba setup completed successfully")

        except ImportError:
            logger.error("jieba not installed. Please install: pip install jieba")
            raise
        except Exception as e:
            logger.error(f"Error setting up jieba: {str(e)}")
            raise

    def _setup_hanlp(self):
        """Setup HanLP with pre-trained models"""
        try:
            # Set environment variable for protobuf compatibility
            import os
            os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

            import hanlp

            # Store reference
            self.hanlp = hanlp

            logger.info("Initializing HanLP models...")

            # Load tokenizer (分词器)
            self.tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
            logger.info("Loaded HanLP tokenizer")

            # Load POS tagger (词性标注器)
            self.pos_tagger = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)
            logger.info("Loaded HanLP POS tagger")

            # Create a pipeline for efficiency
            self.pipeline = hanlp.pipeline() \
                .append(self.tokenizer, output_key='tok') \
                .append(self.pos_tagger, input_key='tok', output_key='pos')
            logger.info("Created HanLP pipeline")

            logger.info("HanLP setup completed successfully")

        except ImportError:
            logger.error("HanLP not installed. Please install: pip install hanlp")
            raise
        except Exception as e:
            logger.error(f"Error setting up HanLP: {str(e)}")
            # Fallback to basic tokenizer if full pipeline fails
            try:
                logger.warning("Falling back to basic HanLP tokenizer")
                self.tokenizer = self.hanlp.load(self.hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
                self.pos_tagger = None
                self.pipeline = None
                logger.info("Basic HanLP tokenizer initialized")
            except Exception as fallback_error:
                logger.error(f"Failed to initialize even basic HanLP: {str(fallback_error)}")
                logger.error("=================================ERROR LOG ENDS=================================")
                raise

    def segment_text(self, text: str) -> List[str]:
        """
        Segment Chinese text into words

        Args:
            text: Input Chinese text

        Returns:
            List of segmented words
        """
        if not text or not text.strip():
            return []

        try:
            # Clean text
            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                return []

            if self.engine == NLPEngine.JIEBA:
                return self._segment_with_jieba(cleaned_text)
            elif self.engine == NLPEngine.HANLP:
                return self._segment_with_hanlp(cleaned_text)
            else:
                raise ValueError(f"Unsupported engine: {self.engine}")

        except Exception as e:
            logger.error(f"Error segmenting text with {self.engine.value}: {str(e)}")
            return [text]  # Return original text as fallback

    def _segment_with_jieba(self, text: str) -> List[str]:
        """Segment text using jieba"""
        words = list(self.jieba.cut(text, cut_all=False, HMM=True))

        # Filter out empty words and single punctuation
        filtered_words = [
            word.strip() for word in words
            if word.strip() and not (len(word.strip()) == 1 and not word.strip().isalnum())
        ]

        return filtered_words

    def _segment_with_hanlp(self, text: str) -> List[str]:
        """Segment text using HanLP"""
        if self.pipeline:
            # Use pipeline for better performance
            result = self.pipeline(text)
            words = result['tok']
        else:
            # Use basic tokenizer
            words = self.tokenizer(text)

        # Filter out empty words and single punctuation
        filtered_words = [
            word.strip() for word in words
            if word.strip() and not (len(word.strip()) == 1 and not word.strip().isalnum())
        ]

        return filtered_words

    def pos_tag(self, text: str) -> List[Tuple[str, str]]:
        """
        Perform POS tagging on Chinese text

        Args:
            text: Input Chinese text

        Returns:
            List of (word, pos_tag) tuples
        """
        if not text or not text.strip():
            return []

        try:
            # Clean text
            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                return []

            if self.engine == NLPEngine.JIEBA:
                return self._pos_tag_with_jieba(cleaned_text)
            elif self.engine == NLPEngine.HANLP:
                return self._pos_tag_with_hanlp(cleaned_text)
            else:
                raise ValueError(f"Unsupported engine: {self.engine}")

        except Exception as e:
            logger.error(f"Error POS tagging with {self.engine.value}: {str(e)}")
            return [(text, 'x')]  # Return original text with unknown POS

    def _pos_tag_with_jieba(self, text: str) -> List[Tuple[str, str]]:
        """POS tagging using jieba"""
        words_pos = self.pseg.lcut(text)
        result = [(word.strip(), flag) for word, flag in words_pos if word.strip()]
        return result

    def _pos_tag_with_hanlp(self, text: str) -> List[Tuple[str, str]]:
        """POS tagging using HanLP"""
        if self.pipeline:
            # Use pipeline for integrated processing
            result = self.pipeline(text)
            tokens = result['tok']
            pos_tags = result['pos']

            # Ensure we have matching tokens and POS tags
            if len(tokens) == len(pos_tags):
                return list(zip(tokens, pos_tags))
            else:
                logger.warning("Token and POS tag count mismatch, using basic approach")

        # Fallback: use separate tokenizer and POS tagger
        if self.pos_tagger:
            tokens = self.tokenizer(text)
            pos_tags = self.pos_tagger(tokens)
            return list(zip(tokens, pos_tags))
        else:
            # No POS tagger available, segment only
            tokens = self.segment_text(text)
            return [(word, "unk") for word in tokens]

    def clean_text(self, text: str) -> str:
        """
        Clean text for better segmentation

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Remove special characters that might interfere with segmentation
        # Keep Chinese characters, English letters, numbers, and common punctuation
        text = re.sub(r'[^\u4e00-\u9fff\u3400-\u4dbf\w\s，。！？；：""''（）【】《》、]', '', text)

        return text

    def get_engine_info(self) -> Dict[str, str]:
        """Get information about the current NLP engine"""
        info = {
            'engine': str(self.engine.value) if isinstance(self.engine, NLPEngine) else str(self.engine),
            'status': 'initialized'
        }

        if self.initialization_error:
            info['status'] = 'error'
            info['error'] = self.initialization_error

        if self.engine == NLPEngine.JIEBA:
            info['description'] = 'jieba - Fast Chinese text segmentation'
            info['features'] = 'Segmentation, POS tagging, Custom dictionary'
        elif self.engine == NLPEngine.HANLP:
            info['description'] = 'HanLP - Multi-task NLP toolkit'
            info['features'] = 'Advanced segmentation, POS tagging, NER, Parsing'
            if self.tokenizer:
                info['tokenizer'] = 'Loaded'
            if self.pos_tagger:
                info['pos_tagger'] = 'Loaded'
            if self.pipeline:
                info['pipeline'] = 'Loaded'
        return info

    def is_initialized(self) -> bool:
        """Check if the NLP processor is properly initialized"""
        return self.initialization_error is None

    def get_initialization_error(self) -> Optional[str]:
        """Get the initialization error message if any"""
        return self.initialization_error

    def compare_engines(self, text: str) -> Dict[str, Dict]:
        """
        Compare results from both engines for the same text

        Args:
            text: Input text to compare

        Returns:
            Dictionary with comparison results
        """
        results = {}

        # Test jieba
        try:
            jieba_processor = ChineseNLPUnified(NLPEngine.JIEBA)
            jieba_words = jieba_processor.segment_text(text)
            jieba_pos = jieba_processor.pos_tag(text)

            results['jieba'] = {
                'words': jieba_words,
                'word_count': len(jieba_words),
                'pos_tags': jieba_pos,
                'status': 'success'
            }
        except Exception as e:
            results['jieba'] = {
                'status': 'error',
                'error': str(e)
            }

        # Test HanLP
        try:
            hanlp_processor = ChineseNLPUnified(NLPEngine.HANLP)
            hanlp_words = hanlp_processor.segment_text(text)
            hanlp_pos = hanlp_processor.pos_tag(text)

            results['hanlp'] = {
                'words': hanlp_words,
                'word_count': len(hanlp_words),
                'pos_tags': hanlp_pos,
                'status': 'success'
            }
        except Exception as e:
            results['hanlp'] = {
                'status': 'error',
                'error': str(e)
            }

        # Add comparison summary
        if results.get('jieba', {}).get('status') == 'success' and \
           results.get('hanlp', {}).get('status') == 'success':
            results['comparison'] = {
                'original_text': text,
                'jieba_word_count': results['jieba']['word_count'],
                'hanlp_word_count': results['hanlp']['word_count'],
                'difference': abs(results['jieba']['word_count'] - results['hanlp']['word_count']),
                'processed_at': datetime.now().isoformat()
            }

        return results

def create_nlp_processor(engine_name: str) -> ChineseNLPUnified:
    """
    Factory function to create NLP processor

    Args:
        engine_name: 'jieba' or 'hanlp'

    Returns:
        ChineseNLPUnified instance
    """
    if engine_name.lower() == 'jieba':
        return ChineseNLPUnified(NLPEngine.JIEBA)
    elif engine_name.lower() == 'hanlp':
        return ChineseNLPUnified(NLPEngine.HANLP)
    else:
        raise ValueError(f"Unsupported engine: {engine_name}. Use 'jieba' or 'hanlp'")