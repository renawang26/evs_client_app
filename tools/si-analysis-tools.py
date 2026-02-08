"""
Simultaneous Interpretation ASR Result Analysis Tools

基于Google ADK框架的同声传译质量分析工具集
"""

import os
import re
import json
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from google.adk.tools import BaseTool

from .tool_registry import register_tool
from llm_config import load_analysis_rules
from config.config import ANALYSIS_RULES_PATH


@register_tool
class ExtractFileContentTool(BaseTool):
    """提取ASR结果文件内容的工具"""
    
    def __init__(self):
        super().__init__(
            name="extract_file_content",
            description="Extract content from uploaded ASR result files"
        )
    
    def __call__(self, file_path: str) -> Dict[str, Any]:
        """
        提取文件内容
        
        Args:
            file_path: ASR结果文件路径
            
        Returns:
            包含文件内容和元数据的字典
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析ASR格式（支持JSON、SRT、VTT等格式）
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.json':
                data = json.loads(content)
                segments = self._parse_json_asr(data)
            elif file_ext in ['.srt', '.vtt']:
                segments = self._parse_subtitle_format(content, file_ext)
            else:
                segments = self._parse_plain_text(content)
                
            return {
                "success": True,
                "file_path": file_path,
                "format": file_ext,
                "segments": segments,
                "total_segments": len(segments)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _parse_json_asr(self, data: Dict) -> List[Dict]:
        """解析JSON格式的ASR结果"""
        segments = []
        if "segments" in data:
            for seg in data["segments"]:
                segments.append({
                    "start_time": seg.get("start", 0),
                    "end_time": seg.get("end", 0),
                    "source_text": seg.get("source", ""),
                    "target_text": seg.get("translation", ""),
                    "confidence": seg.get("confidence", 1.0)
                })
        return segments
    
    def _parse_subtitle_format(self, content: str, format: str) -> List[Dict]:
        """解析字幕格式（SRT/VTT）"""
        # 简化实现，实际应使用专门的字幕解析库
        segments = []
        # 解析逻辑...
        return segments
    
    def _parse_plain_text(self, content: str) -> List[Dict]:
        """解析纯文本格式"""
        segments = []
        lines = content.strip().split('\n')
        for i, line in enumerate(lines):
            if line.strip():
                segments.append({
                    "start_time": i * 3,  # 假设每句3秒
                    "end_time": (i + 1) * 3,
                    "source_text": "",
                    "target_text": line.strip(),
                    "confidence": 1.0
                })
        return segments


@register_tool
class AnalyzeTranslationQualityTool(BaseTool):
    """分析翻译质量的工具"""
    
    def __init__(self, rules_config_path=ANALYSIS_RULES_PATH):
        super().__init__(
            name="analyze_translation_quality",
            description="Analyze translation accuracy, fluency, and completeness"
        )
        # 加载质量评估规则
        self.rules_config_path = rules_config_path
        rules_config = load_analysis_rules(rules_config_path)
        self.quality_metrics = rules_config.get("quality_metrics", {})
        
    def __call__(self, segments: List[Dict]) -> Dict[str, Any]:
        """
        分析翻译质量
        
        Args:
            segments: ASR片段列表
            
        Returns:
            质量评估结果
        """
        accuracy_scores = []
        fluency_scores = []
        completeness_scores = []
        
        for segment in segments:
            source = segment.get("source_text", "")
            target = segment.get("target_text", "")
            
            # 准确性评分
            accuracy = self._calculate_accuracy(source, target)
            accuracy_scores.append(accuracy)
            
            # 流畅性评分
            fluency = self._calculate_fluency(target)
            fluency_scores.append(fluency)
            
            # 完整性评分
            completeness = self._calculate_completeness(source, target)
            completeness_scores.append(completeness)
        
        # 计算平均分
        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
        avg_fluency = sum(fluency_scores) / len(fluency_scores) if fluency_scores else 0
        avg_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0
        
        # 总体质量分数
        overall_score = (avg_accuracy * 0.4 + avg_fluency * 0.3 + avg_completeness * 0.3)
        
        return {
            "overall_score": round(overall_score, 2),
            "accuracy_score": round(avg_accuracy, 2),
            "fluency_score": round(avg_fluency, 2),
            "completeness_score": round(avg_completeness, 2),
            "quality_level": self._get_quality_level(overall_score),
            "segment_count": len(segments)
        }
    
    def _calculate_accuracy(self, source: str, target: str) -> float:
        """计算准确性分数"""
        # 这里应该使用更复杂的NLP技术，如语义相似度
        # 简化实现：基于关键词匹配
        if not source or not target:
            return 0.7  # 默认分数
        
        # 检查数字、专有名词等的准确性
        numbers_in_source = re.findall(r'\d+', source)
        numbers_in_target = re.findall(r'\d+', target)
        
        if numbers_in_source:
            matched = sum(1 for n in numbers_in_source if n in numbers_in_target)
            accuracy = matched / len(numbers_in_source)
        else:
            accuracy = 0.8  # 默认基础分
            
        return min(accuracy, 1.0)
    
    def _calculate_fluency(self, target: str) -> float:
        """计算流畅性分数"""
        if not target:
            return 0.0
            
        # 检查常见的不流畅标记
        disfluency_markers = ['呃', '啊', '那个', '...', '--', '，，', '。。']
        disfluency_count = sum(1 for marker in disfluency_markers if marker in target)
        
        # 检查句子结构完整性
        has_punctuation = any(p in target for p in ['。', '！', '？', '.', '!', '?'])
        
        # 计算分数
        fluency = 1.0
        fluency -= disfluency_count * 0.1
        if not has_punctuation and len(target) > 20:
            fluency -= 0.2
            
        return max(fluency, 0.0)
    
    def _calculate_completeness(self, source: str, target: str) -> float:
        """计算完整性分数"""
        if not source:
            return 0.8  # 如果没有源文本，给默认分
            
        # 简化实现：基于长度比较
        source_len = len(source.split())
        target_len = len(target)
        
        if source_len == 0:
            return 0.8
            
        # 期望的长度比例（中英文长度比例约为1:1.5）
        expected_ratio = 1.5
        actual_ratio = target_len / source_len if source_len > 0 else 0
        
        # 计算偏差
        deviation = abs(actual_ratio - expected_ratio) / expected_ratio
        completeness = 1.0 - min(deviation, 1.0)
        
        return completeness
    
    def _get_quality_level(self, score: float) -> str:
        """根据分数获取质量等级"""
        if score >= 0.9:
            return "优秀"
        elif score >= 0.8:
            return "良好"
        elif score >= 0.7:
            return "合格"
        elif score >= 0.6:
            return "需改进"
        else:
            return "不合格"


@register_tool
class DetectTranslationErrorsTool(BaseTool):
    """检测翻译错误的工具"""
    
    def __init__(self, rules_config_path=ANALYSIS_RULES_PATH):
        super().__init__(
            name="detect_translation_errors",
            description="Detect common interpretation errors"
        )
        # 加载错误检测规则
        rules_config = load_analysis_rules(rules_config_path)
        self.error_patterns = rules_config.get("error_patterns", [])
    
    def __call__(self, segments: List[Dict]) -> Dict[str, Any]:
        """
        检测翻译错误
        
        Args:
            segments: ASR片段列表
            
        Returns:
            错误检测结果
        """
        errors = []
        error_stats = {
            "omission": 0,
            "mistranslation": 0,
            "tense_error": 0,
            "terminology_error": 0,
            "cultural_error": 0,
            "grammar_error": 0
        }
        
        for i, segment in enumerate(segments):
            segment_errors = self._detect_segment_errors(segment, i)
            errors.extend(segment_errors)
            
            # 统计错误类型
            for error in segment_errors:
                error_type = error["type"]
                if error_type in error_stats:
                    error_stats[error_type] += 1
        
        return {
            "total_errors": len(errors),
            "error_list": errors,
            "error_statistics": error_stats,
            "error_density": len(errors) / len(segments) if segments else 0
        }
    
    def _detect_segment_errors(self, segment: Dict, index: int) -> List[Dict]:
        """检测单个片段的错误"""
        errors = []
        source = segment.get("source_text", "")
        target = segment.get("target_text", "")
        start_time = segment.get("start_time", 0)
        
        # 检测漏译
        if source and not target:
            errors.append({
                "type": "omission",
                "severity": "high",
                "timestamp": start_time,
                "segment_index": index,
                "description": "整句漏译",
                "source": source,
                "target": target
            })
        
        # 检测时态错误（简化实现）
        tense_markers = {
            "past": ["了", "过", "曾经"],
            "future": ["将", "会", "即将"],
            "present": ["正在", "在", "着"]
        }
        
        # 检测术语错误
        if self.error_patterns:
            for pattern in self.error_patterns:
                if pattern.get("enabled", True):
                    if re.search(pattern["pattern"], target):
                        errors.append({
                            "type": pattern["type"],
                            "severity": pattern.get("severity", "medium"),
                            "timestamp": start_time,
                            "segment_index": index,
                            "description": pattern["description"],
                            "source": source,
                            "target": target
                        })
        
        return errors


@register_tool
class AnalyzeTimingSyncTool(BaseTool):
    """分析时间同步性的工具"""
    
    def __init__(self):
        super().__init__(
            name="analyze_timing_sync",
            description="Check temporal synchronization between translation and source"
        )
    
    def __call__(self, segments: List[Dict]) -> Dict[str, Any]:
        """
        分析时间同步性
        
        Args:
            segments: ASR片段列表
            
        Returns:
            时间同步分析结果
        """
        delays = []
        sync_issues = []
        
        for i, segment in enumerate(segments):
            start_time = segment.get("start_time", 0)
            end_time = segment.get("end_time", 0)
            duration = end_time - start_time
            
            # 计算翻译延迟（假设基于文本长度）
            source_len = len(segment.get("source_text", ""))
            target_len = len(segment.get("target_text", ""))
            
            # 估算延迟
            estimated_delay = self._estimate_delay(source_len, target_len, duration)
            delays.append(estimated_delay)
            
            # 检测同步问题
            if estimated_delay > 3.0:  # 延迟超过3秒
                sync_issues.append({
                    "segment_index": i,
                    "timestamp": start_time,
                    "delay": estimated_delay,
                    "severity": "high" if estimated_delay > 5.0 else "medium",
                    "description": f"翻译延迟过大：{estimated_delay:.1f}秒"
                })
        
        avg_delay = sum(delays) / len(delays) if delays else 0
        
        return {
            "average_delay": round(avg_delay, 2),
            "max_delay": round(max(delays), 2) if delays else 0,
            "sync_issues": sync_issues,
            "sync_quality": self._get_sync_quality(avg_delay),
            "issue_count": len(sync_issues)
        }
    
    def _estimate_delay(self, source_len: int, target_len: int, duration: float) -> float:
        """估算翻译延迟"""
        # 简化模型：基于文本长度和持续时间
        if source_len == 0:
            return 0.0
            
        # 假设说话速度为每秒3-4个字
        speaking_rate = 3.5
        source_speaking_time = source_len / speaking_rate
        
        # 翻译处理时间
        processing_time = 0.5 + (source_len * 0.01)  # 基础延迟 + 长度相关延迟
        
        return min(processing_time, duration)
    
    def _get_sync_quality(self, avg_delay: float) -> str:
        """评估同步质量"""
        if avg_delay <= 1.0:
            return "优秀"
        elif avg_delay <= 2.0:
            return "良好"
        elif avg_delay <= 3.0:
            return "可接受"
        else:
            return "需改进"


@register_tool
class CheckCulturalAdaptationTool(BaseTool):
    """检查文化适应性的工具"""
    
    def __init__(self, rules_config_path=ANALYSIS_RULES_PATH):
        super().__init__(
            name="check_cultural_adaptation",
            description="Evaluate cultural adaptability and localization quality"
        )
        # 加载文化适应性规则
        rules_config = load_analysis_rules(rules_config_path)
        self.cultural_rules = rules_config.get("cultural_rules", {})
    
    def __call__(self, segments: List[Dict]) -> Dict[str, Any]:
        """
        检查文化适应性
        
        Args:
            segments: ASR片段列表
            
        Returns:
            文化适应性评估结果
        """
        cultural_issues = []
        adaptation_score = 1.0
        
        for i, segment in enumerate(segments):
            target = segment.get("target_text", "")
            issues = self._check_cultural_issues(target, i, segment.get("start_time", 0))
            cultural_issues.extend(issues)
        
        # 计算适应性分数
        if segments:
            issue_rate = len(cultural_issues) / len(segments)
            adaptation_score = max(1.0 - issue_rate * 0.5, 0.0)
        
        return {
            "adaptation_score": round(adaptation_score, 2),
            "cultural_issues": cultural_issues,
            "issue_count": len(cultural_issues),
            "adaptation_level": self._get_adaptation_level(adaptation_score)
        }
    
    def _check_cultural_issues(self, text: str, index: int, timestamp: float) -> List[Dict]:
        """检查文化相关问题"""
        issues = []
        
        # 检查习语和俗语的本地化
        idiom_patterns = [
            ("rain cats and dogs", "下猫下狗", "倾盆大雨"),
            ("piece of cake", "一块蛋糕", "小菜一碟"),
            ("break a leg", "摔断腿", "祝好运")
        ]
        
        for original, literal, proper in idiom_patterns:
            if literal in text:
                issues.append({
                    "type": "idiom_literal",
                    "severity": "medium",
                    "segment_index": index,
                    "timestamp": timestamp,
                    "description": f"习语直译：'{literal}'应译为'{proper}'",
                    "suggestion": proper
                })
        
        # 检查敬语和礼貌用语
        politeness_issues = self._check_politeness(text)
        if politeness_issues:
            issues.extend([{
                "type": "politeness",
                "severity": "low",
                "segment_index": index,
                "timestamp": timestamp,
                "description": issue,
                "suggestion": ""
            } for issue in politeness_issues])
        
        return issues
    
    def _check_politeness(self, text: str) -> List[str]:
        """检查敬语使用"""
        issues = []
        
        # 简化实现：检查基本的敬语使用
        if "你" in text and any(formal in text for formal in ["先生", "女士", "教授", "博士"]):
            if "您" not in text:
                issues.append("正式场合建议使用'您'而非'你'")
        
        return issues
    
    def _get_adaptation_level(self, score: float) -> str:
        """获取文化适应性等级"""
        if score >= 0.9:
            return "优秀"
        elif score >= 0.8:
            return "良好"
        elif score >= 0.7:
            return "合格"
        else:
            return "需改进"


@register_tool
class SuggestImprovementsTool(BaseTool):
    """提供改进建议的工具"""
    
    def __init__(self):
        super().__init__(
            name="suggest_improvements",
            description="Provide specific improvement recommendations"
        )
    
    def __call__(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于分析结果提供改进建议
        
        Args:
            analysis_results: 各项分析的汇总结果
            
        Returns:
            改进建议
        """
        suggestions = []
        priority_levels = {"high": [], "medium": [], "low": []}
        
        # 基于质量分析提供建议
        quality = analysis_results.get("quality_analysis", {})
        if quality.get("accuracy_score", 1.0) < 0.8:
            suggestion = {
                "category": "准确性",
                "priority": "high",
                "description": "翻译准确性需要提升",
                "recommendations": [
                    "加强专业术语词汇表的使用",
                    "提高数字和专有名词的识别准确率",
                    "建立领域特定的翻译记忆库"
                ]
            }
            suggestions.append(suggestion)
            priority_levels["high"].append(suggestion)
        
        # 基于错误检测提供建议
        errors = analysis_results.get("error_detection", {})
        error_stats = errors.get("error_statistics", {})
        
        if error_stats.get("omission", 0) > 5:
            suggestion = {
                "category": "完整性",
                "priority": "high",
                "description": "存在较多漏译现象",
                "recommendations": [
                    "优化语音识别的分段算法",
                    "增加翻译缓冲时间",
                    "提高对快速语音的处理能力"
                ]
            }
            suggestions.append(suggestion)
            priority_levels["high"].append(suggestion)
        
        # 基于时间同步分析提供建议
        timing = analysis_results.get("timing_analysis", {})
        if timing.get("average_delay", 0) > 3.0:
            suggestion = {
                "category": "实时性",
                "priority": "medium",
                "description": "翻译延迟较大",
                "recommendations": [
                    "优化翻译模型的推理速度",
                    "考虑使用增量式翻译策略",
                    "平衡翻译质量与速度的权衡"
                ]
            }
            suggestions.append(suggestion)
            priority_levels["medium"].append(suggestion)
        
        # 基于文化适应性提供建议
        cultural = analysis_results.get("cultural_analysis", {})
        if cultural.get("adaptation_score", 1.0) < 0.8:
            suggestion = {
                "category": "本地化",
                "priority": "low",
                "description": "文化适应性有待提高",
                "recommendations": [
                    "建立常用习语和俗语的对照表",
                    "加强文化背景知识的训练",
                    "根据目标受众调整语言风格"
                ]
            }
            suggestions.append(suggestion)
            priority_levels["low"].append(suggestion)
        
        return {
            "total_suggestions": len(suggestions),
            "suggestions": suggestions,
            "priority_summary": {
                "high": len(priority_levels["high"]),
                "medium": len(priority_levels["medium"]),
                "low": len(priority_levels["low"])
            },
            "action_items": self._generate_action_items(priority_levels)
        }
    
    def _generate_action_items(self, priority_levels: Dict) -> List[str]:
        """生成具体的行动项"""
        action_items = []
        
        # 高优先级行动项
        for suggestion in priority_levels["high"]:
            action_items.append(f"[高优先级] {suggestion['category']}: {suggestion['recommendations'][0]}")
        
        # 中优先级行动项
        for suggestion in priority_levels["medium"]:
            if len(action_items) < 5:  # 限制行动项数量
                action_items.append(f"[中优先级] {suggestion['category']}: {suggestion['recommendations'][0]}")
        
        return action_items


@register_tool
class GenerateCorrectedVersionTool(BaseTool):
    """生成修正版本的工具"""
    
    def __init__(self):
        super().__init__(
            name="generate_corrected_version",
            description="Generate corrected translation version"
        )
    
    def __call__(self, segments: List[Dict], errors: List[Dict]) -> Dict[str, Any]:
        """
        生成修正后的翻译版本
        
        Args:
            segments: 原始ASR片段
            errors: 检测到的错误列表
            
        Returns:
            修正后的版本
        """
        corrected_segments = []
        corrections_made = 0
        
        # 创建错误索引映射
        error_map = {}
        for error in errors:
            idx = error.get("segment_index", -1)
            if idx not in error_map:
                error_map[idx] = []
            error_map[idx].append(error)
        
        # 逐段修正
        for i, segment in enumerate(segments):
            corrected_segment = segment.copy()
            
            if i in error_map:
                # 应用修正
                original_text = segment.get("target_text", "")
                corrected_text = self._apply_corrections(original_text, error_map[i])
                
                if corrected_text != original_text:
                    corrected_segment["target_text"] = corrected_text
                    corrected_segment["is_corrected"] = True
                    corrected_segment["corrections"] = error_map[i]
                    corrections_made += 1
            
            corrected_segments.append(corrected_segment)
        
        return {
            "corrected_segments": corrected_segments,
            "corrections_made": corrections_made,
            "correction_rate": corrections_made / len(segments) if segments else 0,
            "summary": self._generate_correction_summary(corrections_made, len(segments))
        }
    
    def _apply_corrections(self, text: str, errors: List[Dict]) -> str:
        """应用修正到文本"""
        corrected = text
        
        for error in errors:
            error_type = error.get("type", "")
            
            if error_type == "terminology_error":
                # 术语修正
                if "suggestion" in error:
                    # 简单的替换逻辑，实际应更智能
                    corrected = corrected.replace(error.get("target", ""), error["suggestion"])
            
            elif error_type == "idiom_literal":
                # 习语修正
                if "suggestion" in error:
                    corrected = corrected.replace(error.get("description", "").split("'")[1], error["suggestion"])
            
            elif error_type == "grammar_error":
                # 语法修正（需要更复杂的NLP处理）
                pass
        
        return corrected
    
    def _generate_correction_summary(self, corrections_made: int, total_segments: int) -> str:
        """生成修正摘要"""
        if corrections_made == 0:
            return "未发现需要修正的内容"
        
        rate = corrections_made / total_segments * 100
        return f"共修正了{corrections_made}处错误，占总片段数的{rate:.1f}%"


# 配置文件示例结构 (analysis_rules.json)
EXAMPLE_RULES_CONFIG = {
    "quality_metrics": {
        "accuracy_weight": 0.4,
        "fluency_weight": 0.3,
        "completeness_weight": 0.3
    },
    "error_patterns": [
        {
            "id": "tense_error_001",
            "type": "tense_error",
            "pattern": r"will.*了|将要.*了",
            "description": "时态矛盾：将来时与过去时混用",
            "severity": "medium",
            "enabled": True
        },
        {
            "id": "terminology_001",
            "type": "terminology_error",
            "pattern": r"machine learning",
            "description": "术语翻译不当",
            "severity": "low",
            "suggestion": "机器学习",
            "enabled": True
        }
    ],
    "cultural_rules": {
        "idiom_mappings": {
            "rain cats and dogs": "倾盆大雨",
            "piece of cake": "小菜一碟",
            "break a leg": "祝好运"
        },
        "formality_rules": {
            "formal_titles": ["先生", "女士", "教授", "博士"],
            "require_honorific": True
        }
    }
}