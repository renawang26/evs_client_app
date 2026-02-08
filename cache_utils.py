"""
缓存工具模块
提供数据库连接缓存和常用数据缓存功能
"""

import sqlite3
import streamlit as st
import pandas as pd
import logging
from pathlib import Path
from typing import List, Optional
from functools import wraps

logger = logging.getLogger(__name__)

# Database path constant
REPOSITORY_ROOT = "./data"
DB_PATH = Path(REPOSITORY_ROOT) / 'evs_repository.db'


@st.cache_resource
def get_cached_db_connection():
    """
    获取缓存的数据库连接
    使用 st.cache_resource 确保连接在整个会话期间复用
    """
    try:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        logger.info("Created cached database connection")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Failed to create cached database connection: {e}")
        raise


def get_db_connection_context():
    """
    获取数据库连接的上下文管理器版本
    用于需要事务控制的操作
    """
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_interpret_files(asr_provider: str) -> pd.DataFrame:
    """
    获取缓存的口译文件列表

    Args:
        asr_provider: ASR提供商名称

    Returns:
        DataFrame: 口译文件列表
    """
    try:
        conn = get_cached_db_connection()
        query = """
        SELECT DISTINCT(file_name), slice_duration
        FROM asr_results_segments
        WHERE asr_provider = ?
        """
        return pd.read_sql_query(query, conn, params=[asr_provider])
    except Exception as e:
        logger.error(f"Error getting cached interpret files: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_processed_files() -> List[str]:
    """
    获取缓存的已处理文件列表

    Returns:
        List[str]: 已处理的文件名列表
    """
    try:
        conn = get_cached_db_connection()
        query = "SELECT DISTINCT file_name FROM asr_results_words ORDER BY file_name"
        result = pd.read_sql_query(query, conn)
        return result['file_name'].tolist()
    except Exception as e:
        logger.error(f"Error getting cached processed files: {e}")
        return []


@st.cache_data(ttl=60)  # Cache for 1 minute
def get_cached_asr_data(interpret_file: str, asr_provider: str) -> pd.DataFrame:
    """
    获取缓存的ASR数据

    Args:
        interpret_file: 口译文件名
        asr_provider: ASR提供商

    Returns:
        DataFrame: ASR数据
    """
    try:
        conn = get_cached_db_connection()
        query = """
        SELECT * FROM (
            SELECT id as interpret_id, file_name, lang,
                segment_id, word_seq_no,
                COALESCE(edit_word, '') || '&&' ||
                COALESCE(pair_type, '') || '&&' ||
                COALESCE(word, '') || '&&' ||
                COALESCE(annotate, '') AS combined_word,
                (end_time - start_time) AS duration, start_time, end_time,
                slice_duration, confidence, word, edit_word, pair_type, annotate
            FROM asr_results_words
            WHERE file_name = ? AND lang = 'en' AND asr_provider = ?
            ORDER BY start_time ASC
        )
        UNION ALL
        SELECT * FROM (
            SELECT id as interpret_id, file_name, lang,
                segment_id, word_seq_no,
                COALESCE(edit_word, '') || '&&' ||
                COALESCE(pair_type, '') || '&&' ||
                COALESCE(word, '') || '&&' ||
                COALESCE(annotate, '') AS combined_word,
                (end_time - start_time) AS duration, start_time, end_time,
                slice_duration, confidence, word, edit_word, pair_type, annotate
            FROM asr_results_words
            WHERE file_name = ? AND lang = 'zh' AND asr_provider = ?
            ORDER BY start_time ASC
        )
        ORDER BY lang, start_time ASC;
        """
        return pd.read_sql_query(query, conn, params=[interpret_file, asr_provider, interpret_file, asr_provider])
    except Exception as e:
        logger.error(f"Error getting cached ASR data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_word_frequency(interpret_file: str, lang: str, asr_provider: str) -> pd.DataFrame:
    """
    获取缓存的词频数据

    Args:
        interpret_file: 口译文件名 ('All' 表示所有文件)
        lang: 语言代码 ('All' 表示所有语言)
        asr_provider: ASR提供商 ('All' 表示所有提供商)

    Returns:
        DataFrame: 词频数据
    """
    try:
        conn = get_cached_db_connection()

        base_query = """
            SELECT edit_word, lang, COUNT(*) as frequency
            FROM asr_results_words
            WHERE edit_word IS NOT NULL
        """

        params = []
        conditions = []

        if interpret_file != 'All':
            conditions.append("file_name = ?")
            params.append(interpret_file)

        if lang != 'All':
            conditions.append("lang = ?")
            params.append(lang)

        if asr_provider != 'All':
            conditions.append("asr_provider = ?")
            params.append(asr_provider)

        final_query = base_query
        if conditions:
            final_query = f"{base_query} AND {' AND '.join(conditions)}"

        final_query += " GROUP BY edit_word, lang ORDER BY frequency DESC"

        return pd.read_sql_query(final_query, conn, params=params)
    except Exception as e:
        logger.error(f"Error getting cached word frequency: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_asr_config(provider: Optional[str] = None):
    """
    获取缓存的ASR配置

    Args:
        provider: ASR提供商名称，None表示获取所有配置

    Returns:
        配置数据字典
    """
    import json
    try:
        conn = get_cached_db_connection()
        cursor = conn.cursor()

        if provider:
            cursor.execute("SELECT provider, config FROM asr_config WHERE provider = ?", (provider,))
            row = cursor.fetchone()
            if row:
                return {
                    "provider": row[0],
                    "config": json.loads(row[1])
                }
            return None
        else:
            cursor.execute("SELECT provider, config FROM asr_config")
            rows = cursor.fetchall()
            result = {}
            for row in rows:
                result[row[0]] = json.loads(row[1])
            return result
    except Exception as e:
        logger.error(f"Error getting cached ASR config: {e}")
        return {} if provider is None else None


def clear_cache(cache_type: str = 'all'):
    """
    清除指定类型的缓存

    Args:
        cache_type: 缓存类型 ('all', 'files', 'asr_data', 'word_freq', 'config')
    """
    if cache_type == 'all':
        st.cache_data.clear()
        logger.info("Cleared all data caches")
    elif cache_type == 'files':
        get_cached_interpret_files.clear()
        get_cached_processed_files.clear()
        logger.info("Cleared file caches")
    elif cache_type == 'asr_data':
        get_cached_asr_data.clear()
        logger.info("Cleared ASR data cache")
    elif cache_type == 'word_freq':
        get_cached_word_frequency.clear()
        logger.info("Cleared word frequency cache")
    elif cache_type == 'config':
        get_cached_asr_config.clear()
        logger.info("Cleared config cache")


def invalidate_cache_on_write(func):
    """
    装饰器：在写操作后自动清除相关缓存
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        # 写操作成功后清除相关缓存
        if result:
            clear_cache('all')
        return result
    return wrapper
