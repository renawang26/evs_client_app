import sqlite3
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import threading

# 尝试导入统一数据库管理器
try:
    from db_manager import get_db_connection
    USE_UNIFIED_DB = True
except ImportError:
    from db_utils import EVSDataUtils
    get_db_connection = EVSDataUtils.get_db_connection
    USE_UNIFIED_DB = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"Email metrics using unified DB: {USE_UNIFIED_DB}")

class EmailMetrics:
    """
    Email metrics collector for monitoring email queue performance.
    Records metrics such as success rate, failure reasons, and timing.
    """

    def __init__(self, db_connection=None):
        """
        Initialize the email metrics collector.

        Args:
            db_connection: SQLite database connection (optional)
        """
        self.db_conn = db_connection or get_db_connection()
        self._initialize_db()

        # Lock for thread safety
        self.lock = threading.Lock()

        # In-memory cache for recent metrics
        self.recent_metrics = []
        self.max_cache_size = 100

        logger.info("Email metrics collector initialized")

    def _initialize_db(self):
        """Initialize database tables for storing email metrics"""
        try:
            cursor = self.db_conn.cursor()

            # Create email_metrics table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS email_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email_id TEXT,
                    email_type TEXT,
                    recipient TEXT,
                    queued_at TIMESTAMP,
                    sent_at TIMESTAMP NULL,
                    status TEXT,
                    error_message TEXT NULL,
                    priority INTEGER,
                    attempt_count INTEGER,
                    queue_time_ms INTEGER,
                    metadata TEXT
                )
            ''')

            # Create email_metrics_summary table for aggregated data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS email_metrics_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    hour INTEGER,
                    total_sent INTEGER,
                    total_failed INTEGER,
                    total_queued INTEGER,
                    avg_queue_time_ms REAL,
                    p95_queue_time_ms REAL,
                    common_errors TEXT
                )
            ''')

            self.db_conn.commit()
            logger.info("Email metrics tables initialized")
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {str(e)}")

    def record_queued(self, email_id: str, email_type: str, recipient: str,
                     priority: int, metadata: Dict[str, Any]) -> None:
        """
        Record when an email is queued.

        Args:
            email_id: Unique identifier for the email
            email_type: Type of email (verification, notification, etc.)
            recipient: Recipient email address
            priority: Priority level
            metadata: Additional metadata about the email
        """
        try:
            with self.lock:
                cursor = self.db_conn.cursor()

                # Convert metadata to JSON string
                metadata_json = json.dumps(metadata)

                # Record queue event
                cursor.execute('''
                    INSERT INTO email_metrics
                    (email_id, email_type, recipient, queued_at, status, priority, attempt_count, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    email_id,
                    email_type,
                    recipient,
                    datetime.now().isoformat(),
                    'queued',
                    priority,
                    0,
                    metadata_json
                ))

                self.db_conn.commit()
                logger.debug(f"Recorded queue event for email {email_id}")
        except Exception as e:
            logger.error(f"Error recording queued email: {str(e)}")

    def record_sent(self, email_id: str, attempt_count: int = 1) -> None:
        """
        Record when an email is successfully sent.

        Args:
            email_id: Unique identifier for the email
            attempt_count: Number of attempts before success
        """
        try:
            with self.lock:
                cursor = self.db_conn.cursor()
                sent_time = datetime.now()

                # Get queue time for calculating delay
                cursor.execute(
                    "SELECT queued_at FROM email_metrics WHERE email_id = ?",
                    (email_id,)
                )
                result = cursor.fetchone()

                if result:
                    queued_at = datetime.fromisoformat(result[0])
                    queue_time_ms = int((sent_time - queued_at).total_seconds() * 1000)

                    # Update record with sent status
                    cursor.execute('''
                        UPDATE email_metrics
                        SET status = ?, sent_at = ?, attempt_count = ?, queue_time_ms = ?
                        WHERE email_id = ?
                    ''', (
                        'sent',
                        sent_time.isoformat(),
                        attempt_count,
                        queue_time_ms,
                        email_id
                    ))

                    self.db_conn.commit()

                    # Cache for recent metrics
                    if len(self.recent_metrics) >= self.max_cache_size:
                        self.recent_metrics.pop(0)

                    self.recent_metrics.append({
                        'email_id': email_id,
                        'status': 'sent',
                        'queue_time_ms': queue_time_ms,
                        'attempts': attempt_count,
                        'timestamp': sent_time.isoformat()
                    })

                    logger.debug(f"Recorded sent event for email {email_id}")
                else:
                    logger.warning(f"Cannot find queued email with ID {email_id}")
        except Exception as e:
            logger.error(f"Error recording sent email: {str(e)}")

    def record_failed(self, email_id: str, error_message: str, attempt_count: int = 1) -> None:
        """
        Record when an email fails to send.

        Args:
            email_id: Unique identifier for the email
            error_message: Error message explaining the failure
            attempt_count: Number of attempts so far
        """
        try:
            with self.lock:
                cursor = self.db_conn.cursor()
                failure_time = datetime.now()

                # Update record with failure status
                cursor.execute('''
                    UPDATE email_metrics
                    SET status = ?, error_message = ?, attempt_count = ?
                    WHERE email_id = ?
                ''', (
                    'failed' if attempt_count >= 3 else 'retrying',
                    error_message,
                    attempt_count,
                    email_id
                ))

                self.db_conn.commit()

                # Add to recent metrics cache
                if len(self.recent_metrics) >= self.max_cache_size:
                    self.recent_metrics.pop(0)

                self.recent_metrics.append({
                    'email_id': email_id,
                    'status': 'failed' if attempt_count >= 3 else 'retrying',
                    'error': error_message,
                    'attempts': attempt_count,
                    'timestamp': failure_time.isoformat()
                })

                logger.debug(f"Recorded failure event for email {email_id}")
        except Exception as e:
            logger.error(f"Error recording failed email: {str(e)}")

    def get_success_rate(self, time_period: str = 'day') -> Dict[str, Any]:
        """
        Calculate email success rate for a given time period.

        Args:
            time_period: 'hour', 'day', 'week', or 'month'

        Returns:
            Dictionary with success rate statistics
        """
        try:
            cursor = self.db_conn.cursor()

            # Calculate time range based on period
            now = datetime.now()
            if time_period == 'hour':
                start_time = now - timedelta(hours=1)
            elif time_period == 'day':
                start_time = now - timedelta(days=1)
            elif time_period == 'week':
                start_time = now - timedelta(weeks=1)
            elif time_period == 'month':
                start_time = now - timedelta(days=30)
            else:
                start_time = now - timedelta(days=1)  # Default to day

            # Query for all emails in time period
            cursor.execute('''
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'sent' THEN 1 ELSE 0 END) as sent,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    SUM(CASE WHEN status = 'queued' OR status = 'retrying' THEN 1 ELSE 0 END) as pending,
                    AVG(CASE WHEN status = 'sent' THEN queue_time_ms ELSE NULL END) as avg_time
                FROM email_metrics
                WHERE queued_at >= ?
            ''', (start_time.isoformat(),))

            result = cursor.fetchone()

            if result:
                total, sent, failed, pending, avg_time = result

                # Calculate success rate
                success_rate = (sent / total) * 100 if total > 0 else 0

                return {
                    'period': time_period,
                    'total_emails': total,
                    'sent_emails': sent,
                    'failed_emails': failed,
                    'pending_emails': pending,
                    'success_rate': success_rate,
                    'avg_queue_time_ms': avg_time or 0
                }

            return {
                'period': time_period,
                'total_emails': 0,
                'sent_emails': 0,
                'failed_emails': 0,
                'pending_emails': 0,
                'success_rate': 0,
                'avg_queue_time_ms': 0
            }

        except Exception as e:
            logger.error(f"Error calculating success rate: {str(e)}")
            return {
                'period': time_period,
                'error': str(e)
            }

    def get_failure_reasons(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get most common failure reasons for emails.

        Args:
            limit: Maximum number of reasons to return

        Returns:
            List of dictionaries with error messages and counts
        """
        try:
            cursor = self.db_conn.cursor()

            cursor.execute('''
                SELECT error_message, COUNT(*) as count
                FROM email_metrics
                WHERE status = 'failed'
                GROUP BY error_message
                ORDER BY count DESC
                LIMIT ?
            ''', (limit,))

            results = cursor.fetchall()

            return [
                {'reason': reason, 'count': count}
                for reason, count in results
            ]

        except Exception as e:
            logger.error(f"Error getting failure reasons: {str(e)}")
            return []

    def get_queue_time_histogram(self) -> Dict[str, Any]:
        """
        Get histogram data for email queue times.

        Returns:
            Dictionary with histogram data
        """
        try:
            cursor = self.db_conn.cursor()

            cursor.execute('''
                SELECT queue_time_ms
                FROM email_metrics
                WHERE status = 'sent' AND queue_time_ms IS NOT NULL
                ORDER BY queue_time_ms
            ''')

            results = cursor.fetchall()
            queue_times = [result[0] for result in results]

            if not queue_times:
                return {
                    'error': "No data available",
                    'data': []
                }

            # Calculate stats
            avg_time = sum(queue_times) / len(queue_times)
            queue_times.sort()
            median_time = queue_times[len(queue_times) // 2]

            # Calculate percentiles
            p95_index = int(len(queue_times) * 0.95)
            p95_time = queue_times[p95_index] if p95_index < len(queue_times) else queue_times[-1]

            p99_index = int(len(queue_times) * 0.99)
            p99_time = queue_times[p99_index] if p99_index < len(queue_times) else queue_times[-1]

            # Generate histogram data (10 bins)
            if len(queue_times) >= 10:
                max_time = queue_times[-1]
                min_time = queue_times[0]
                bin_size = max(1, (max_time - min_time) // 10)

                histogram = {}
                for time in queue_times:
                    bin_key = (time // bin_size) * bin_size
                    histogram[bin_key] = histogram.get(bin_key, 0) + 1

                histogram_data = [
                    {'bin': bin_key, 'count': count}
                    for bin_key, count in sorted(histogram.items())
                ]
            else:
                histogram_data = [
                    {'bin': time, 'count': 1}
                    for time in queue_times
                ]

            return {
                'avg_time': avg_time,
                'median_time': median_time,
                'p95_time': p95_time,
                'p99_time': p99_time,
                'min_time': queue_times[0],
                'max_time': queue_times[-1],
                'total_emails': len(queue_times),
                'histogram': histogram_data
            }

        except Exception as e:
            logger.error(f"Error generating queue time histogram: {str(e)}")
            return {
                'error': str(e),
                'data': []
            }

    def generate_daily_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive daily report of email metrics.

        Returns:
            Dictionary with daily report data
        """
        try:
            # Get data for the last 24 hours
            yesterday = datetime.now() - timedelta(days=1)

            cursor = self.db_conn.cursor()

            # Get hourly statistics
            cursor.execute('''
                SELECT
                    strftime('%H', queued_at) as hour,
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'sent' THEN 1 ELSE 0 END) as sent,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    AVG(CASE WHEN status = 'sent' THEN queue_time_ms ELSE NULL END) as avg_time
                FROM email_metrics
                WHERE queued_at >= ?
                GROUP BY hour
                ORDER BY hour
            ''', (yesterday.isoformat(),))

            hourly_data = cursor.fetchall()

            # Get email type breakdown
            cursor.execute('''
                SELECT
                    email_type,
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'sent' THEN 1 ELSE 0 END) as sent,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
                FROM email_metrics
                WHERE queued_at >= ?
                GROUP BY email_type
                ORDER BY total DESC
            ''', (yesterday.isoformat(),))

            type_data = cursor.fetchall()

            # Get overall statistics
            daily_stats = self.get_success_rate('day')
            failure_reasons = self.get_failure_reasons(5)
            queue_times = self.get_queue_time_histogram()

            # Format data for report
            hourly_stats = [
                {
                    'hour': hour,
                    'total': total,
                    'sent': sent,
                    'failed': failed,
                    'avg_time_ms': avg_time or 0
                }
                for hour, total, sent, failed, avg_time in hourly_data
            ]

            email_types = [
                {
                    'type': email_type,
                    'total': total,
                    'sent': sent,
                    'failed': failed,
                    'success_rate': (sent / total) * 100 if total > 0 else 0
                }
                for email_type, total, sent, failed in type_data
            ]

            return {
                'report_date': datetime.now().strftime('%Y-%m-%d'),
                'period_start': yesterday.isoformat(),
                'period_end': datetime.now().isoformat(),
                'overall': daily_stats,
                'hourly_stats': hourly_stats,
                'email_types': email_types,
                'failure_reasons': failure_reasons,
                'queue_times': queue_times
            }

        except Exception as e:
            logger.error(f"Error generating daily report: {str(e)}")
            return {
                'error': str(e),
                'report_date': datetime.now().strftime('%Y-%m-%d')
            }

    def generate_chart(self, chart_type: str = 'success_rate') -> Optional[str]:
        """
        Generate a chart visualization as a base64 encoded image.

        Args:
            chart_type: Type of chart ('success_rate', 'queue_time', 'volume')

        Returns:
            Base64 encoded PNG image or None if error
        """
        try:
            plt.figure(figsize=(10, 6))

            if chart_type == 'success_rate':
                # Get last 7 days of data
                cursor = self.db_conn.cursor()
                cursor.execute('''
                    SELECT
                        strftime('%Y-%m-%d', queued_at) as date,
                        COUNT(*) as total,
                        SUM(CASE WHEN status = 'sent' THEN 1 ELSE 0 END) as sent
                    FROM email_metrics
                    WHERE queued_at >= date('now', '-7 days')
                    GROUP BY date
                    ORDER BY date
                ''')

                results = cursor.fetchall()

                dates = []
                rates = []

                for date, total, sent in results:
                    dates.append(date)
                    rates.append((sent / total) * 100 if total > 0 else 0)

                plt.bar(dates, rates, color='#4CAF50')
                plt.xlabel('Date')
                plt.ylabel('Success Rate (%)')
                plt.title('Email Success Rate (Last 7 Days)')
                plt.ylim(0, 100)
                plt.xticks(rotation=45)
                plt.tight_layout()

            elif chart_type == 'queue_time':
                # Get queue time data
                queue_data = self.get_queue_time_histogram()

                if 'error' in queue_data:
                    return None

                bins = [item['bin'] for item in queue_data['histogram']]
                counts = [item['count'] for item in queue_data['histogram']]

                plt.bar(bins, counts, color='#2196F3')
                plt.axvline(x=queue_data['avg_time'], color='r', linestyle='--', label=f"Avg: {queue_data['avg_time']:.0f}ms")
                plt.axvline(x=queue_data['p95_time'], color='g', linestyle='--', label=f"95th: {queue_data['p95_time']:.0f}ms")

                plt.xlabel('Queue Time (ms)')
                plt.ylabel('Number of Emails')
                plt.title('Email Queue Time Distribution')
                plt.legend()
                plt.tight_layout()

            elif chart_type == 'volume':
                # Get volume by hour
                cursor = self.db_conn.cursor()
                cursor.execute('''
                    SELECT
                        strftime('%H', queued_at) as hour,
                        COUNT(*) as total
                    FROM email_metrics
                    WHERE queued_at >= date('now', '-1 day')
                    GROUP BY hour
                    ORDER BY hour
                ''')

                results = cursor.fetchall()

                hours = []
                counts = []

                for hour, count in results:
                    hours.append(hour)
                    counts.append(count)

                plt.plot(hours, counts, marker='o', linestyle='-', color='#FF9800')
                plt.xlabel('Hour of Day')
                plt.ylabel('Number of Emails')
                plt.title('Email Volume by Hour (Last 24 Hours)')
                plt.xticks(rotation=45)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()

            # Save plot to a bytes buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)

            # Convert to base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            plt.close()

            return image_base64

        except Exception as e:
            logger.error(f"Error generating chart: {str(e)}")
            return None

    def close(self):
        """Close database connection"""
        if self.db_conn:
            self.db_conn.close()
            logger.info("Email metrics database connection closed")


# Global email metrics instance
email_metrics = None

def get_email_metrics() -> EmailMetrics:
    """
    Get the global email metrics instance.

    Returns:
        EmailMetrics: Global email metrics instance

    Raises:
        RuntimeError: If email metrics not initialized
    """
    global email_metrics

    if email_metrics is None:
        raise RuntimeError("Email metrics not initialized. Call initialize_email_metrics first.")

    return email_metrics

def initialize_email_metrics() -> None:
    """Initialize global email metrics instance"""
    global email_metrics

    if email_metrics is not None:
        logger.warning("Email metrics already initialized")
        return

    email_metrics = EmailMetrics()
    logger.info("Global email metrics initialized")

    return email_metrics