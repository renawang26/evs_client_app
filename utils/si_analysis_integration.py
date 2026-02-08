"""
同声传译分析工具集成模块

该模块负责将 si-analysis-tools.py 中的各种分析工具集成到主应用程序中，
提供统一的接口来调用各种分析功能并将结果保存到数据库中。
"""

import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import sys
import os

# 添加 tools 目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tools'))

try:
    from tools.si_analysis_tools import (
        ExtractFileContentTool,
        AnalyzeTranslationQualityTool,
        DetectTranslationErrorsTool,
        AnalyzeTimingSyncTool,
        CheckCulturalAdaptationTool,
        SuggestImprovementsTool,
        GenerateCorrectedVersionTool
    )
except ImportError as e:
    print(f"Warning: Could not import si-analysis-tools: {e}")
    # Define placeholder classes for development
    class ExtractFileContentTool:
        def __call__(self, *args, **kwargs): return {}
    class AnalyzeTranslationQualityTool:
        def __call__(self, *args, **kwargs): return {}
    class DetectTranslationErrorsTool:
        def __call__(self, *args, **kwargs): return {}
    class AnalyzeTimingSyncTool:
        def __call__(self, *args, **kwargs): return {}
    class CheckCulturalAdaptationTool:
        def __call__(self, *args, **kwargs): return {}
    class SuggestImprovementsTool:
        def __call__(self, *args, **kwargs): return {}
    class GenerateCorrectedVersionTool:
        def __call__(self, *args, **kwargs): return {}

from db_utils import EVSDataUtils
from llm_config import load_analysis_rules, get_active_llm_config
import logging

logger = logging.getLogger(__name__)

class SIAnalysisManager:
    """
    Simultaneous Interpretation Analysis Manager

    Responsible for coordinating the use of various analysis tools and saving the results to the database
    """

    def __init__(self):
        self.analysis_config = load_analysis_rules()
        self.llm_config = get_active_llm_config()

        # Initialize analysis tools
        try:
            self.extract_tool = ExtractFileContentTool()
            self.quality_tool = AnalyzeTranslationQualityTool()
            self.error_tool = DetectTranslationErrorsTool()
            self.timing_tool = AnalyzeTimingSyncTool()
            self.cultural_tool = CheckCulturalAdaptationTool()
            self.suggestion_tool = SuggestImprovementsTool()
            self.correction_tool = GenerateCorrectedVersionTool()
        except Exception as e:
            logger.error(f"Failed to initialize analysis tools: {e}")
            raise

    def analyze_file_content(self, file_path: str, file_name: str, asr_provider: str) -> Dict[str, Any]:
        """
        Analyze file content and extract ASR results

        Args:
            file_path: file path
            file_name: file name
            asr_provider: ASR provider

        Returns:
            Dict: extracted content data
        """
        try:
            logger.info(f"Extracting file content: {file_name}")

            # Use the extraction tool to get the file content
            content = self.extract_tool(file_path)

            if content:
                logger.info(f"Successfully extracted file content, containing {len(content.get('segments', []))} segments")
                return content
            else:
                logger.warning(f"File content extraction failed: {file_name}")
                return {}

        except Exception as e:
            logger.error(f"Error extracting file content: {str(e)}")
            return {}

    def analyze_translation_quality(self, segments: List[Dict], file_name: str, asr_provider: str,
                                  created_by: str = None) -> Optional[int]:
        """
        Analyze translation quality

        Args:
            segments: transcription segments list
            file_name: file name
            asr_provider: ASR provider
            created_by: creator

        Returns:
            Optional[int]: analysis result ID, return None if failed
        """
        try:
            start_time = time.time()
            logger.info(f"Starting translation quality analysis: {file_name}")

            # Use the quality analysis tool
            quality_results = self.quality_tool(segments)

            processing_time = int((time.time() - start_time) * 1000)

            # Prepare data to be saved to the database
            analysis_data = {
                'overall_score': quality_results.get('overall_score'),
                'accuracy_score': quality_results.get('accuracy_score'),
                'fluency_score': quality_results.get('fluency_score'),
                'completeness_score': quality_results.get('completeness_score'),
                'quality_level': quality_results.get('quality_level'),
                'total_segments': len(segments),
                'processing_time_ms': processing_time,
                'llm_model': self.llm_config.get('llm_model'),
                'analysis_config': self.analysis_config.get('quality_metrics'),
                'analysis_results': quality_results
            }

            # Save to database
            analysis_id = EVSDataUtils.save_si_analysis_result(
                file_name=file_name,
                asr_provider=asr_provider,
                analysis_type='quality',
                analysis_data=analysis_data,
                created_by=created_by
            )

            if analysis_id > 0:
                logger.info(f"Translation quality analysis completed, result ID: {analysis_id}")
                return analysis_id
            else:
                logger.error("Failed to save translation quality analysis results")
                return None

        except Exception as e:
            logger.error(f"Error during translation quality analysis: {str(e)}")
            return None

    def detect_translation_errors(self, segments: List[Dict], file_name: str, asr_provider: str,
                                 created_by: str = None) -> Optional[int]:
        """
        Detect translation errors

        Args:
            segments: transcription segments list
            file_name: file name
            asr_provider: ASR provider
            created_by: creator

        Returns:
            Optional[int]: analysis result ID, return None if failed
        """
        try:
            start_time = time.time()
            logger.info(f"Starting error detection analysis: {file_name}")

            # Use the error detection tool
            errors = self.error_tool(segments)

            processing_time = int((time.time() - start_time) * 1000)

            # 统计错误信息
            error_stats = {}
            total_errors = len(errors) if errors else 0

            if errors:
                for error in errors:
                    error_type = error.get('type', 'unknown')
                    if error_type in error_stats:
                        error_stats[error_type] += 1
                    else:
                        error_stats[error_type] = 1

            # 计算错误密度（错误数/总片段数）
            error_density = total_errors / len(segments) if segments else 0

            # 准备保存到数据库的数据
            analysis_data = {
                'total_errors': total_errors,
                'error_statistics': error_stats,
                'error_density': error_density,
                'total_segments': len(segments),
                'processing_time_ms': processing_time,
                'llm_model': self.llm_config.get('llm_model'),
                'analysis_config': self.analysis_config.get('error_patterns'),
                'analysis_results': errors
            }

            # 保存到数据库
            analysis_id = EVSDataUtils.save_si_analysis_result(
                file_name=file_name,
                asr_provider=asr_provider,
                analysis_type='errors',
                analysis_data=analysis_data,
                created_by=created_by
            )

            if analysis_id > 0:
                # Save error details
                if errors:
                    EVSDataUtils.save_si_error_details(analysis_id, errors)

                logger.info(f"Error detection analysis completed, found {total_errors} errors, result ID: {analysis_id}")
                return analysis_id
            else:
                logger.error("Failed to save error detection analysis results")
                return None

        except Exception as e:
            logger.error(f"Error during error detection analysis: {str(e)}")
            return None

    def analyze_timing_sync(self, segments: List[Dict], file_name: str, asr_provider: str,
                           created_by: str = None) -> Optional[int]:
        """
        Analyze timing synchronization

        Args:
            segments: transcription segments list
            file_name: file name
            asr_provider: ASR provider
            created_by: creator

        Returns:
            Optional[int]: analysis result ID, return None if failed
        """
        try:
            start_time = time.time()
            logger.info(f"Starting timing synchronization analysis: {file_name}")

            # Use the timing synchronization analysis tool
            timing_results = self.timing_tool(segments)

            processing_time = int((time.time() - start_time) * 1000)

            # Prepare data to be saved to the database
            analysis_data = {
                'average_delay': timing_results.get('average_delay'),
                'max_delay': timing_results.get('max_delay'),
                'sync_quality': timing_results.get('sync_quality'),
                'sync_issue_count': timing_results.get('sync_issue_count', 0),
                'total_segments': len(segments),
                'processing_time_ms': processing_time,
                'llm_model': self.llm_config.get('llm_model'),
                'analysis_config': self.analysis_config.get('timing_analysis'),
                'analysis_results': timing_results
            }

            # Save to database
            analysis_id = EVSDataUtils.save_si_analysis_result(
                file_name=file_name,
                asr_provider=asr_provider,
                analysis_type='timing',
                analysis_data=analysis_data,
                created_by=created_by
            )

            if analysis_id > 0:
                # Save timing synchronization issue details
                timing_issues = timing_results.get('timing_issues', [])
                if timing_issues:
                    EVSDataUtils.save_si_timing_issues(analysis_id, timing_issues)

                logger.info(f"Timing synchronization analysis completed, result ID: {analysis_id}")
                return analysis_id
            else:
                logger.error("Failed to save timing synchronization analysis results")
                return None

        except Exception as e:
            logger.error(f"Error during timing synchronization analysis: {str(e)}")
            return None

    def check_cultural_adaptation(self, segments: List[Dict], file_name: str, asr_provider: str,
                                 created_by: str = None) -> Optional[int]:
        """
        Check cultural adaptation

        Args:
            segments: transcription segments list
            file_name: file name
            asr_provider: ASR provider
            created_by: creator

        Returns:
            Optional[int]: analysis result ID, return None if failed
        """
        try:
            start_time = time.time()
            logger.info(f"Starting cultural adaptation check: {file_name}")

            # Use the cultural adaptation check tool
            cultural_results = self.cultural_tool(segments)

            processing_time = int((time.time() - start_time) * 1000)

            # Prepare data to be saved to the database
            analysis_data = {
                'adaptation_score': cultural_results.get('adaptation_score'),
                'cultural_issue_count': cultural_results.get('cultural_issue_count', 0),
                'adaptation_level': cultural_results.get('adaptation_level'),
                'total_segments': len(segments),
                'processing_time_ms': processing_time,
                'llm_model': self.llm_config.get('llm_model'),
                'analysis_config': self.analysis_config.get('cultural_rules'),
                'analysis_results': cultural_results
            }

            # Save to database
            analysis_id = EVSDataUtils.save_si_analysis_result(
                file_name=file_name,
                asr_provider=asr_provider,
                analysis_type='cultural',
                analysis_data=analysis_data,
                created_by=created_by
            )

            if analysis_id > 0:
                # Save cultural issue details
                cultural_issues = cultural_results.get('cultural_issues', [])
                if cultural_issues:
                    EVSDataUtils.save_si_cultural_issues(analysis_id, cultural_issues)

                logger.info(f"Cultural adaptation check completed, result ID: {analysis_id}")
                return analysis_id
            else:
                logger.error("Failed to save cultural adaptation check results")
                return None

        except Exception as e:
            logger.error(f"Error during cultural adaptation check: {str(e)}")
            return None

    def generate_improvement_suggestions(self, analysis_results: Dict[str, Any], file_name: str,
                                       asr_provider: str, created_by: str = None) -> Optional[int]:
        """
        Generate improvement suggestions

        Args:
            analysis_results: previous analysis results
            file_name: file name
            asr_provider: ASR provider
            created_by: creator

        Returns:
            Optional[int]: analysis result ID, return None if failed
        """
        try:
            start_time = time.time()
            logger.info(f"Starting to generate improvement suggestions: {file_name}")

            # Use the improvement suggestion tool
            suggestions = self.suggestion_tool(analysis_results)

            processing_time = int((time.time() - start_time) * 1000)

            # Prepare data to be saved to the database
            analysis_data = {
                'total_segments': analysis_results.get('total_segments', 0),
                'processing_time_ms': processing_time,
                'llm_model': self.llm_config.get('llm_model'),
                'analysis_config': self.analysis_config,
                'analysis_results': suggestions
            }

            # Save to database
            analysis_id = EVSDataUtils.save_si_analysis_result(
                file_name=file_name,
                asr_provider=asr_provider,
                analysis_type='suggestions',
                analysis_data=analysis_data,
                created_by=created_by
            )

            if analysis_id > 0:
                logger.info(f"Improvement suggestions generated, result ID: {analysis_id}")
                return analysis_id
            else:
                logger.error("Failed to save improvement suggestions results")
                return None

        except Exception as e:
            logger.error(f"Error during improvement suggestions generation: {str(e)}")
            return None

    def run_complete_analysis(self, file_path: str, file_name: str, asr_provider: str,
                             created_by: str = None) -> Dict[str, Any]:
        """
        Run the complete simultaneous interpretation analysis process

        Args:
            file_path: file path
            file_name: file name
            asr_provider: ASR provider
            created_by: creator

        Returns:
            Dict: dictionary containing all analysis result IDs
        """
        try:
            logger.info(f"Starting complete analysis process: {file_name}")

            # 1. Extract file content
            content = self.analyze_file_content(file_path, file_name, asr_provider)
            if not content or not content.get('segments'):
                return {'error': 'File content extraction failed'}

            segments = content['segments']
            results = {}

            # 2. Translation quality analysis
            quality_id = self.analyze_translation_quality(segments, file_name, asr_provider, created_by)
            if quality_id:
                results['quality_analysis_id'] = quality_id

            # 3. Error detection
            error_id = self.detect_translation_errors(segments, file_name, asr_provider, created_by)
            if error_id:
                results['error_analysis_id'] = error_id

            # 4. Timing synchronization analysis
            timing_id = self.analyze_timing_sync(segments, file_name, asr_provider, created_by)
            if timing_id:
                results['timing_analysis_id'] = timing_id

            # 5. Cultural adaptation check
            cultural_id = self.check_cultural_adaptation(segments, file_name, asr_provider, created_by)
            if cultural_id:
                results['cultural_analysis_id'] = cultural_id

            # 6. Generate improvement suggestions (based on previous analysis results)
            if results:
                analysis_summary = {
                    'quality_analysis_id': quality_id,
                    'error_analysis_id': error_id,
                    'timing_analysis_id': timing_id,
                    'cultural_analysis_id': cultural_id,
                    'total_segments': len(segments)
                }

                suggestion_id = self.generate_improvement_suggestions(
                    analysis_summary, file_name, asr_provider, created_by
                )
                if suggestion_id:
                    results['suggestion_analysis_id'] = suggestion_id

            logger.info(f"Complete analysis process completed: {file_name}, generated {len(results)} analysis results")
            return results

        except Exception as e:
            logger.error(f"Error during complete analysis process: {str(e)}")
            return {'error': str(e)}

# Create a global analysis manager instance
try:
    si_analysis_manager = SIAnalysisManager()
except Exception as e:
    logger.warning(f"Failed to initialize SI Analysis Manager: {e}")
    si_analysis_manager = None