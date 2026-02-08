import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)

class ConcordanceUtils:
    @staticmethod
    def calculate_collocates(df: pd.DataFrame, search_term: str, window_size: int = 5, min_freq: int = 1):
        """Calculate collocates for a search term within a specified window size"""
        collocates = []
        search_term = str(search_term)

        # Group by sentence to maintain sentence boundaries
        for _, group in df.groupby(['file_name', 'segment_id']):
            words = group.sort_values('word_seq_no')
            term_positions = words[words['edit_word'].astype(str).str.contains(search_term, case=False, na=False)].index

            for pos in term_positions:
                start_idx = max(0, words.index.get_loc(pos) - window_size)
                end_idx = min(len(words), words.index.get_loc(pos) + window_size + 1)
                window_words = words.iloc[start_idx:end_idx]

                for _, word_row in window_words.iterrows():
                    if not pd.isna(word_row['edit_word']):
                        word_str = str(word_row['edit_word'])
                        if word_str.lower() != search_term.lower():
                            position = words.index.get_loc(pos) - words.index.get_loc(word_row.name)
                            collocates.append({
                                'collocate': word_str,
                                'position': position,
                                'file': word_row['file_name'],
                                'sentence': word_row['segment_id']
                            })

        if collocates:
            coll_df = pd.DataFrame(collocates)
            stats = coll_df.groupby('collocate').agg({
                'position': ['count', 'mean', 'std'],
                'file': 'nunique',
                'sentence': 'nunique'
            }).reset_index()

            stats.columns = ['collocate', 'frequency', 'mean_position', 'std_position', 'n_files', 'n_sentences']
            stats = stats[stats['frequency'] >= min_freq]
            stats = stats.sort_values('frequency', ascending=False)

            return stats, coll_df

        return pd.DataFrame(), pd.DataFrame()

    @staticmethod
    def calculate_clusters(df: pd.DataFrame, cluster_size: int = 2, min_freq: int = 2):
        """Calculate word clusters (n-grams) from text"""
        clusters = []

        for _, group in df.groupby(['file_name', 'segment_id', 'lang']):
            words = group.sort_values('word_seq_no')['edit_word'].apply(
                lambda x: str(x) if pd.notnull(x) else ''
            ).tolist()

            for i in range(len(words) - cluster_size + 1):
                cluster = words[i:i + cluster_size]
                if all(word.strip() for word in cluster):
                    cluster_text = ' '.join(cluster) if group.iloc[0]['lang'] == 'en' else ''.join(cluster)
                    clusters.append({
                        'cluster': cluster_text,
                        'file': group.iloc[0]['file_name'],
                        'sentence': group.iloc[0]['segment_id'],
                        'lang': group.iloc[0]['lang']
                    })

        if clusters:
            clusters_df = pd.DataFrame(clusters)
            stats = clusters_df.groupby(['cluster', 'lang']).agg({
                'file': ['count', 'nunique'],
                'sentence': 'nunique'
            }).reset_index()

            stats.columns = ['cluster', 'lang', 'frequency', 'n_files', 'n_sentences']
            stats = stats[stats['frequency'] >= min_freq]
            stats = stats.sort_values('frequency', ascending=False)

            return stats

        return pd.DataFrame()

    @staticmethod
    def calculate_keyword_list(target_corpus: pd.DataFrame, reference_corpus: pd.DataFrame,
                             min_freq: int = 1, min_range: int = 1):
        """
        Calculate keyword list using log-likelihood statistic

        Args:
            target_corpus: DataFrame with word frequencies from target corpus
            reference_corpus: DataFrame with word frequencies from reference corpus
            min_freq: Minimum frequency threshold
            min_range: Minimum number of texts the word must appear in
        """
        # Calculate total words in each corpus
        target_total = target_corpus['frequency'].sum()
        ref_total = reference_corpus['frequency'].sum()

        # Merge the two frequency lists
        merged = pd.merge(
            target_corpus,
            reference_corpus,
            on=['edit_word', 'lang'],
            how='outer',
            suffixes=('_target', '_ref')
        ).fillna(0)

        # Calculate expected frequencies
        merged['expected_target'] = merged['frequency_ref'] * (target_total / ref_total)
        merged['expected_ref'] = merged['frequency_target'] * (ref_total / target_total)

        # Calculate log-likelihood statistic
        def log_likelihood(O1, O2, E1, E2):
            def safe_log(x):
                return np.log(x) if x > 0 else 0

            if O1 + O2 == 0:
                return 0

            return 2 * (
                O1 * safe_log(O1/E1 if E1 > 0 else 0) +
                O2 * safe_log(O2/E2 if E2 > 0 else 0)
            )

        # Apply log-likelihood calculation
        merged['log_likelihood'] = merged.apply(
            lambda x: log_likelihood(
                x['frequency_target'],
                x['frequency_ref'],
                x['expected_target'],
                x['expected_ref']
            ),
            axis=1
        )

        # Calculate relative frequencies per million words
        merged['freq_pmw_target'] = (merged['frequency_target'] / target_total) * 1000000
        merged['freq_pmw_ref'] = (merged['frequency_ref'] / ref_total) * 1000000

        # Apply frequency threshold
        merged = merged[
            (merged['frequency_target'] >= min_freq) |
            (merged['frequency_ref'] >= min_freq)
        ]

        # Sort by log-likelihood score
        merged = merged.sort_values('log_likelihood', ascending=False)

        # Add keyness information
        merged['keyness'] = np.where(
            merged['freq_pmw_target'] > merged['freq_pmw_ref'],
            'Positive',
            'Negative'
        )

        return merged

    @staticmethod
    def generate_concordance_lines(df: pd.DataFrame, search_term: str, context_size: int = 5) -> pd.DataFrame:
        """
        Generate concordance lines from a DataFrame of words

        Args:
            df: DataFrame containing words with their context
            search_term: The term to search for
            context_size: Number of words to include in context

        Returns:
            DataFrame with concordance lines
        """
        concordance_data = []

        # Group by sentence
        for (file, sent_no), group in df.groupby(['file_name', 'segment_id']):
            words = group.sort_values('word_seq_no')
            words['edit_word'] = words['edit_word'].astype(str)

            for idx, row in words.iterrows():
                if str(search_term).lower() in str(row['edit_word']).lower():
                    word_list = words['edit_word'].tolist()
                    current_idx = words.index[words.index == idx][0]
                    word_idx = words.index.get_loc(current_idx)

                    left_context = ' '.join(word_list[max(0, word_idx - context_size):word_idx])
                    right_context = ' '.join(word_list[word_idx + 1:word_idx + context_size + 1])

                    concordance_data.append({
                        'File': file,
                        'Left Context': left_context,
                        'Hit': row['edit_word'],
                        'Right Context': right_context,
                        'Language': row['lang'],
                        'Time': row['start_time']
                    })

        return pd.DataFrame(concordance_data)

    @staticmethod
    def generate_concordance_plot(df: pd.DataFrame, search_term: str) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Generate concordance plot data

        Args:
            df: DataFrame containing words
            search_term: The term to search for

        Returns:
            Tuple containing (plot_data_df, results_list)
        """
        if df.empty:
            return pd.DataFrame(), []

        # Filter out files with no hits
        df = df[df['Freq'] > 0]

        # Process the data
        results = []
        for idx, row in df.iterrows():
            # Create plot string
            plot = ""
            if row['positions']:
                positions = [float(p) for p in row['positions'].split(',')]
                max_pos = float(row['max_pos'])
                plot_width = 50  # Number of characters in plot
                for i in range(plot_width):
                    pos_found = False
                    for pos in positions:
                        rel_pos = (pos / max_pos) * plot_width
                        if abs(i - rel_pos) < 0.5:
                            plot += "|"
                            pos_found = True
                            break
                    if not pos_found:
                        plot += " "

            # Create result row
            results.append({
                'Row': idx + 1,
                'FileID': f"{idx + 1:02d}",
                'FileName': row['FileName'],
                'FileTokens': row['FileTokens'],
                'Freq': row['Freq'],
                'NormFreq': row['NormFreq'],
                'Plot': plot
            })

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        return df, results
