#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced analysis utilities for simultaneous interpretation data analysis
"""

import pandas as pd
import logging
from save_asr_results import get_db_connection

# Setup logging
logger = logging.getLogger(__name__)

def create_time_based_segment_mapping(segments_df_en, segments_df_zh, max_time_gap=5.0):
    """
    Create time-based mapping between English and Chinese segments instead of relying on segment_id

    Args:
        segments_df_en: English segments DataFrame
        segments_df_zh: Chinese segments DataFrame
        max_time_gap: Maximum acceptable time gap for mapping (seconds)

    Returns:
        list: List of tuples (en_segment, zh_segment, time_alignment_score)
    """
    if segments_df_en.empty or segments_df_zh.empty:
        return []

    mapped_pairs = []
    used_zh_segments = set()

    for _, en_seg in segments_df_en.iterrows():
        en_start = en_seg['start_time']
        en_end = en_seg['end_time']
        en_mid = (en_start + en_end) / 2

        best_match = None
        best_score = float('inf')

        # Find the best matching Chinese segment based on time overlap
        for _, zh_seg in segments_df_zh.iterrows():
            if zh_seg['segment_id'] in used_zh_segments:
                continue

            zh_start = zh_seg['start_time']
            zh_end = zh_seg['end_time']
            zh_mid = (zh_start + zh_end) / 2

            # Calculate time alignment score (lower is better)
            time_gap = abs(en_mid - zh_mid)

            # Check for time overlap or proximity
            overlap_start = max(en_start, zh_start)
            overlap_end = min(en_end, zh_end)
            overlap_duration = max(0, overlap_end - overlap_start)

            # Calculate alignment score considering both overlap and proximity
            if overlap_duration > 0:
                # Prefer overlapping segments
                alignment_score = time_gap
            else:
                # For non-overlapping segments, penalize by distance
                alignment_score = time_gap + max_time_gap

            # Only consider segments within acceptable time window
            if alignment_score < best_score and time_gap <= max_time_gap:
                best_score = alignment_score
                best_match = zh_seg

        # Add the best match if found
        if best_match is not None:
            used_zh_segments.add(best_match['segment_id'])

            # Calculate final time alignment score (0-1, higher is better)
            time_alignment = max(0, 1 - (best_score / max_time_gap))

            mapped_pairs.append((en_seg, best_match, time_alignment))

    return mapped_pairs

def analyze_translation_accuracy_with_time_mapping(file_name, asr_provider):
    """
    Analyze translation accuracy using time-based segment mapping
    """
    try:
        logger.info(f"Time mapping function called with: file_name='{file_name}', asr_provider='{asr_provider}'")

        # Get segments data
        with get_db_connection() as conn:
            segments_query = """
            SELECT * FROM asr_results_segments
            WHERE file_name = ? AND asr_provider = ?
            ORDER BY lang, start_time
            """
            segments_df = pd.read_sql_query(segments_query, conn, params=[file_name, asr_provider])

        logger.info(f"Retrieved {len(segments_df)} segments from database")

        if segments_df.empty:
            logger.warning(f"No segments found for file_name='{file_name}', asr_provider='{asr_provider}'")
            return pd.DataFrame(), {}

        # Separate English and Chinese segments
        segments_df_en = segments_df[segments_df['lang'] == 'en'].copy()
        segments_df_zh = segments_df[segments_df['lang'] == 'zh'].copy()

        if segments_df_en.empty or segments_df_zh.empty:
            return pd.DataFrame(), {}

        # Create time-based mapping
        mapped_pairs = create_time_based_segment_mapping(segments_df_en, segments_df_zh)

        if not mapped_pairs:
            return pd.DataFrame(), {}

        # Analyze each mapped pair
        analysis_results = []

        for en_seg, zh_seg, time_alignment in mapped_pairs:
            # Get word count using actual text data
            en_text = str(en_seg.get('edit_text', en_seg.get('text', '')))
            zh_text = str(zh_seg.get('edit_text', zh_seg.get('text', '')))

            en_words = len(en_text.split()) if en_text else 0
            zh_words = len(zh_text) if zh_text else 0  # Chinese character count

            # Calculate length ratio
            length_ratio = zh_words / en_words if en_words > 0 else 0

            # Calculate quality score based on time alignment and reasonable length ratio
            # Good length ratio for EN-ZH is typically 0.4-0.8
            length_score = 1.0 - abs(0.6 - min(length_ratio, 1.0))
            quality_score = (time_alignment * 0.6 + length_score * 0.4)

            analysis_results.append({
                'segment_id': f"EN{en_seg['segment_id']}-ZH{zh_seg['segment_id']}",
                'en_segment_id': en_seg['segment_id'],
                'zh_segment_id': zh_seg['segment_id'],
                'english_text': en_text,
                'chinese_text': zh_text,
                'en_start_time': en_seg['start_time'],
                'en_end_time': en_seg['end_time'],
                'zh_start_time': zh_seg['start_time'],
                'zh_end_time': zh_seg['end_time'],
                'en_words': en_words,
                'zh_words': zh_words,
                'length_ratio': length_ratio,
                'time_alignment': time_alignment,
                'quality_score': quality_score
            })

        # Create DataFrame
        results_df = pd.DataFrame(analysis_results)

        # Calculate summary statistics
        summary_stats = {
            'total_pairs': len(results_df),
            'avg_time_alignment': results_df['time_alignment'].mean(),
            'avg_length_ratio': results_df['length_ratio'].mean(),
            'avg_quality_score': results_df['quality_score'].mean(),
            'avg_en_words': results_df['en_words'].mean(),
            'avg_zh_words': results_df['zh_words'].mean(),
            'well_aligned_pairs': len(results_df[results_df['time_alignment'] > 0.7]),
            'good_quality_pairs': len(results_df[results_df['quality_score'] > 0.6])
        }

        return results_df, summary_stats

    except Exception as e:
        logger.error(f"Error in time-based translation analysis: {str(e)}")
        return pd.DataFrame(), {}