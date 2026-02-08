import threading
import queue
import time
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import uuid
import json
import sqlite3

# 尝试导入统一数据库管理器
try:
    from db_manager import get_db_connection
    USE_UNIFIED_DB = True
except ImportError:
    from db_utils import EVSDataUtils
    get_db_connection = EVSDataUtils.get_db_connection
    USE_UNIFIED_DB = False

logger = logging.getLogger(__name__)
logger.info(f"Email queue using unified DB: {USE_UNIFIED_DB}")

class EmailQueue:
    """
    Email queue system to prevent server overload during registration and other email-sending operations.
    Implements a worker thread model with configurable sending rate limits.
    """

    def __init__(self,
                 smtp_server: str,
                 smtp_port: int,
                 smtp_username: str,
                 smtp_password: str,
                 from_email: str,
                 max_emails_per_minute: int = 20,
                 max_retry_attempts: int = 3,
                 retry_delay: int = 300):
        """
        Initialize the email queue system.

        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            smtp_username: SMTP username for authentication
            smtp_password: SMTP password for authentication
            from_email: The sender email address
            max_emails_per_minute: Maximum number of emails to send per minute
            max_retry_attempts: Maximum number of retry attempts for failed emails
            retry_delay: Delay between retry attempts in seconds
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username
        self.smtp_password = smtp_password
        self.from_email = from_email

        self.max_emails_per_minute = max_emails_per_minute
        self.max_retry_attempts = max_retry_attempts
        self.retry_delay = retry_delay

        # Email queues
        self.email_queue = queue.Queue()  # Main queue for pending emails
        self.retry_queue = queue.Queue()  # Queue for failed emails that will be retried

        # Email counter for rate limiting
        self.email_count = 0
        self.last_reset_time = time.time()

        # Statistics
        self.stats = {
            "emails_queued": 0,
            "emails_sent": 0,
            "emails_failed": 0,
            "current_queue_size": 0
        }

        # 初始化数据库表
        self._init_db_table()

        # Start worker thread
        self.active = True
        self.worker_thread = threading.Thread(target=self._process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()

        # Initialize metrics if available
        self.metrics_enabled = False
        try:
            from email_metrics import get_email_metrics, initialize_email_metrics
            # Initialize metrics if not already initialized
            try:
                self.metrics = get_email_metrics()
            except RuntimeError:
                self.metrics = initialize_email_metrics()

            self.metrics_enabled = True
            logger.info("Email metrics enabled for queue")
        except (ImportError, Exception) as e:
            logger.warning(f"Email metrics not available: {str(e)}")

        logger.info("Email queue system initialized")

    def _init_db_table(self):
        """初始化数据库表用于存储邮件队列"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # 创建邮件队列表（如果不存在）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS email_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email_id TEXT UNIQUE NOT NULL,
                    recipient TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    body TEXT NOT NULL,
                    is_html INTEGER DEFAULT 1,
                    priority INTEGER DEFAULT 3,
                    status TEXT DEFAULT 'queued',
                    attempt_count INTEGER DEFAULT 0,
                    queued_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    sent_at TIMESTAMP,
                    metadata TEXT
                )
            """)
            conn.commit()
            logger.info("邮件队列表初始化完成")

            # 加载现有的未发送邮件
            cursor.execute("""
                SELECT * FROM email_queue
                WHERE status = 'queued' OR status = 'retrying'
                ORDER BY priority, queued_at
            """)

            pending_emails = cursor.fetchall()
            if pending_emails:
                for email in pending_emails:
                    # 转换为字典
                    email_data = {
                        "id": email["email_id"],
                        "to": email["recipient"],
                        "subject": email["subject"],
                        "body": email["body"],
                        "is_html": bool(email["is_html"]),
                        "priority": email["priority"],
                        "attempt": email["attempt_count"],
                        "queued_at": email["queued_at"],
                        "metadata": json.loads(email["metadata"] or "{}")
                    }

                    # 添加到队列
                    self.email_queue.put(email_data)
                    self.stats["emails_queued"] += 1

                logger.info(f"从数据库加载了 {len(pending_emails)} 个待发送邮件")

        except Exception as e:
            logger.error(f"初始化邮件队列表失败: {str(e)}")

    def _save_email_to_db(self, email_data):
        """将邮件保存到数据库"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # 转换元数据为JSON
            metadata_json = json.dumps(email_data.get("metadata", {}))

            # 插入或更新邮件记录
            cursor.execute("""
                INSERT OR REPLACE INTO email_queue (
                    email_id, recipient, subject, body, is_html,
                    priority, status, attempt_count, queued_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                email_data["id"],
                email_data["to"],
                email_data["subject"],
                email_data["body"],
                1 if email_data["is_html"] else 0,
                email_data["priority"],
                "queued",
                email_data["attempt"],
                email_data["queued_at"],
                metadata_json
            ))

            conn.commit()
            logger.debug(f"邮件 {email_data['id']} 已保存到数据库")

        except Exception as e:
            logger.error(f"保存邮件到数据库失败: {str(e)}")

    def _update_email_status(self, email_id, status, attempt_count=None, error_message=None):
        """更新数据库中的邮件状态"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            update_fields = ["status"]
            params = [status]

            if status == "sent":
                update_fields.append("sent_at")
                params.append(datetime.now().isoformat())

            if attempt_count is not None:
                update_fields.append("attempt_count")
                params.append(attempt_count)

            if error_message is not None:
                update_fields.append("error_message")
                params.append(error_message)

            # 构建更新语句
            set_clause = ", ".join([f"{field} = ?" for field in update_fields])
            query = f"UPDATE email_queue SET {set_clause} WHERE email_id = ?"

            # 添加email_id参数
            params.append(email_id)

            cursor.execute(query, params)
            conn.commit()

            logger.debug(f"邮件 {email_id} 状态更新为 {status}")

        except Exception as e:
            logger.error(f"更新邮件状态失败: {str(e)}")

    def enqueue_email(self, to_email: str, subject: str, body: str,
                     is_html: bool = True, priority: int = 1,
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add an email to the sending queue.

        Args:
            to_email: Recipient email address
            subject: Email subject line
            body: Email body content
            is_html: Whether the email body is HTML content
            priority: Priority level (1-5, 1 being highest)
            metadata: Additional metadata for tracking/reference

        Returns:
            bool: True if successfully queued, False otherwise
        """
        try:
            # Generate unique email ID for tracking
            email_id = str(uuid.uuid4())

            email_data = {
                "id": email_id,
                "to": to_email,
                "subject": subject,
                "body": body,
                "is_html": is_html,
                "priority": priority,
                "metadata": metadata or {},
                "attempt": 0,
                "queued_at": datetime.now().isoformat()
            }

            # 保存到数据库
            self._save_email_to_db(email_data)

            # Determine email type from metadata for metrics
            email_type = "unknown"
            if metadata and "type" in metadata:
                email_type = metadata["type"]

            # Record metrics if enabled
            if self.metrics_enabled:
                try:
                    self.metrics.record_queued(
                        email_id=email_id,
                        email_type=email_type,
                        recipient=to_email,
                        priority=priority,
                        metadata=metadata or {}
                    )
                except Exception as e:
                    logger.error(f"Failed to record email metrics: {str(e)}")

            self.email_queue.put(email_data)
            self.stats["emails_queued"] += 1
            self.stats["current_queue_size"] = self.email_queue.qsize()

            logger.info(f"Email to {to_email} queued successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to queue email: {str(e)}")
            return False

    def _process_queue(self):
        """Worker thread that processes the email queue"""
        while self.active:
            try:
                # Rate limiting check
                current_time = time.time()
                if current_time - self.last_reset_time >= 60:
                    # Reset counter every minute
                    self.email_count = 0
                    self.last_reset_time = current_time

                # Process retry queue first with priority
                if not self.retry_queue.empty() and self.email_count < self.max_emails_per_minute:
                    email_data = self.retry_queue.get(block=False)
                    self._send_email(email_data)
                    self.retry_queue.task_done()
                # Then process main queue
                elif not self.email_queue.empty() and self.email_count < self.max_emails_per_minute:
                    email_data = self.email_queue.get(block=False)
                    self._send_email(email_data)
                    self.email_queue.task_done()
                else:
                    # Sleep to prevent high CPU usage
                    time.sleep(0.1)

                # Update queue size statistic
                self.stats["current_queue_size"] = self.email_queue.qsize()

            except queue.Empty:
                # No emails in queue, sleep briefly
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error in email queue processing: {str(e)}")
                time.sleep(1)  # Sleep to prevent high CPU usage on persistent errors

    def _send_email(self, email_data: Dict[str, Any]) -> None:
        """
        Send an email using SMTP.

        Args:
            email_data: Dictionary containing email details
        """
        # Check if we've reached rate limit
        if self.email_count >= self.max_emails_per_minute:
            # Put back in queue and return
            self.email_queue.put(email_data)
            logger.warning("Rate limit reached, email re-queued")
            return

        email_id = email_data.get("id", str(uuid.uuid4()))
        to_email = email_data["to"]
        subject = email_data["subject"]
        body = email_data["body"]
        is_html = email_data["is_html"]
        attempt = email_data["attempt"]

        try:
            # Create message
            message = MIMEMultipart()
            message["From"] = self.from_email
            message["To"] = to_email
            message["Subject"] = subject

            # Add email ID for tracking
            message["X-Email-ID"] = email_id

            # Attach body with appropriate content type
            content_type = "html" if is_html else "plain"
            message.attach(MIMEText(body, content_type))

            # Connect to SMTP server and send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(message)

            # Update stats and counter
            self.stats["emails_sent"] += 1
            self.email_count += 1

            # 更新数据库状态
            self._update_email_status(email_id, "sent", attempt + 1)

            # Record metrics if enabled
            if self.metrics_enabled:
                try:
                    self.metrics.record_sent(
                        email_id=email_id,
                        attempt_count=attempt + 1
                    )
                except Exception as e:
                    logger.error(f"Failed to record sent email metrics: {str(e)}")

            logger.info(f"Email sent successfully to {to_email}")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to send email to {to_email}: {error_msg}")
            self.stats["emails_failed"] += 1

            # 更新数据库状态
            status = "retrying" if email_data["attempt"] < self.max_retry_attempts else "failed"
            self._update_email_status(email_id, status, attempt + 1, error_msg)

            # Record failure metrics if enabled
            if self.metrics_enabled:
                try:
                    self.metrics.record_failed(
                        email_id=email_id,
                        error_message=error_msg,
                        attempt_count=attempt + 1
                    )
                except Exception as metrics_error:
                    logger.error(f"Failed to record failure metrics: {str(metrics_error)}")

            # Increment attempt counter
            email_data["attempt"] += 1

            # Retry if under max attempts
            if email_data["attempt"] < self.max_retry_attempts:
                logger.info(f"Scheduling retry for email to {to_email} (attempt {email_data['attempt']})")
                # Add to retry queue with delay
                retry_thread = threading.Thread(
                    target=self._delayed_retry,
                    args=(email_data,)
                )
                retry_thread.daemon = True
                retry_thread.start()
            else:
                logger.warning(f"Maximum retry attempts reached for email to {to_email}")

    def _delayed_retry(self, email_data: Dict[str, Any]) -> None:
        """
        Wait for specified delay and then add email back to retry queue.

        Args:
            email_data: Email data dictionary
        """
        time.sleep(self.retry_delay)
        self.retry_queue.put(email_data)
        logger.info(f"Email to {email_data['to']} added to retry queue")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics about the email queue.

        Returns:
            Dictionary with queue statistics
        """
        stats = self.stats.copy()
        stats["current_queue_size"] = self.email_queue.qsize()
        stats["current_retry_queue_size"] = self.retry_queue.qsize()

        # Add metrics data if available
        if self.metrics_enabled:
            try:
                # Get success rate for last hour
                hour_stats = self.metrics.get_success_rate('hour')
                if 'error' not in hour_stats:
                    stats["hourly_success_rate"] = hour_stats.get('success_rate', 0)
                    stats["hourly_avg_queue_time_ms"] = hour_stats.get('avg_queue_time_ms', 0)

                # Get top failure reasons
                failure_reasons = self.metrics.get_failure_reasons(3)
                if failure_reasons:
                    stats["top_failure_reasons"] = failure_reasons
            except Exception as e:
                logger.error(f"Failed to get metrics data: {str(e)}")

        # 添加数据库统计信息
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # 获取各种状态的邮件数量
            cursor.execute("""
                SELECT status, COUNT(*) as count
                FROM email_queue
                GROUP BY status
            """)

            db_status = {}
            for row in cursor.fetchall():
                db_status[row[0]] = row[1]

            stats["db_status"] = db_status

            # 获取今日发送数量
            today = datetime.now().strftime('%Y-%m-%d')
            cursor.execute("""
                SELECT COUNT(*)
                FROM email_queue
                WHERE status = 'sent'
                AND DATE(sent_at) = ?
            """, (today,))

            stats["sent_today"] = cursor.fetchone()[0]

        except Exception as e:
            logger.error(f"Failed to get database statistics: {str(e)}")

        return stats

    def get_pending_emails(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取挂起的邮件列表"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT *
                FROM email_queue
                WHERE status IN ('queued', 'retrying')
                ORDER BY priority, queued_at
                LIMIT ?
            """, (limit,))

            results = []
            for row in cursor.fetchall():
                # 创建字典并格式化日期
                email = dict(row)
                if email.get('metadata'):
                    try:
                        email['metadata'] = json.loads(email['metadata'])
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.debug(f"Failed to parse email metadata: {e}")
                        email['metadata'] = {}
                results.append(email)

            return results

        except Exception as e:
            logger.error(f"获取挂起邮件失败: {str(e)}")
            return []

    def get_email_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取邮件发送历史"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT *
                FROM email_queue
                ORDER BY
                    CASE
                        WHEN status = 'sent' THEN sent_at
                        ELSE queued_at
                    END DESC
                LIMIT ?
            """, (limit,))

            results = []
            for row in cursor.fetchall():
                # 创建字典并格式化日期
                email = dict(row)
                if email.get('metadata'):
                    try:
                        email['metadata'] = json.loads(email['metadata'])
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.debug(f"Failed to parse email metadata: {e}")
                        email['metadata'] = {}
                results.append(email)

            return results

        except Exception as e:
            logger.error(f"获取邮件历史失败: {str(e)}")
            return []

    def shutdown(self) -> None:
        """Shutdown the queue worker thread gracefully"""
        self.active = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        logger.info("Email queue system shutdown")


# Global email queue instance
email_queue = None

def initialize_email_queue(config: Dict[str, Any]) -> None:
    """
    Initialize the global email queue with configuration.

    Args:
        config: Dictionary containing SMTP configuration
    """
    global email_queue

    if email_queue is not None:
        logger.warning("Email queue already initialized")
        return

    # Initialize email metrics first if available
    try:
        from email_metrics import initialize_email_metrics
        initialize_email_metrics()
        logger.info("Email metrics initialized before queue")
    except (ImportError, Exception) as e:
        logger.warning(f"Could not initialize email metrics: {str(e)}")

    email_queue = EmailQueue(
        smtp_server=config["SERVER"],
        smtp_port=config["PORT"],
        smtp_username=config["USERNAME"],
        smtp_password=config["PASSWORD"],
        from_email=config["FROM_EMAIL"],
        max_emails_per_minute=config.get("MAX_EMAILS_PER_MINUTE", 20),
        max_retry_attempts=config.get("MAX_RETRY_ATTEMPTS", 3),
        retry_delay=config.get("RETRY_DELAY", 300)
    )

    logger.info("Global email queue initialized")

def get_email_queue() -> EmailQueue:
    """
    Get the global email queue instance.

    Returns:
        EmailQueue: The global email queue instance

    Raises:
        RuntimeError: If the email queue is not initialized
    """
    global email_queue

    if email_queue is None:
        raise RuntimeError("Email queue not initialized. Call initialize_email_queue first.")

    return email_queue