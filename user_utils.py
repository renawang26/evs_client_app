import sqlite3
import re
import hashlib
import logging
from pathlib import Path
import os
import smtplib
import uuid
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from config import AUTH_CONFIG, BASE_URL
import pandas as pd

# 尝试导入统一数据库管理器
try:
    from db_manager import get_db_connection
    USE_UNIFIED_DB = True
except ImportError:
    from db_utils import EVSDataUtils
    get_db_connection = EVSDataUtils.get_db_connection
    USE_UNIFIED_DB = False

from email_queue import initialize_email_queue, get_email_queue

logger = logging.getLogger(__name__)
logger.info(f"User utils using unified DB: {USE_UNIFIED_DB}")

class UserUtils:
    # Load configuration
    SMTP_SERVER = AUTH_CONFIG['SMTP']['SERVER']
    SMTP_PORT = AUTH_CONFIG['SMTP']['PORT']
    SMTP_USERNAME = AUTH_CONFIG['SMTP']['USERNAME']
    SMTP_PASSWORD = AUTH_CONFIG['SMTP']['PASSWORD']
    SMTP_FROM_EMAIL = AUTH_CONFIG['SMTP']['FROM_EMAIL']
    BASE_URL = BASE_URL
    # Email queue settings
    EMAIL_QUEUE_ENABLED = AUTH_CONFIG['SMTP'].get('QUEUE_ENABLED', True)

    @staticmethod
    def get_db_connection():
        """获取数据库连接"""
        try:
            # 使用统一的数据库连接
            conn = get_db_connection()

            # 验证用户表是否存在并包含所需列
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(users)")
            columns = {row[1] for row in cursor.fetchall()}

            # 检查users表是否存在
            if not columns:
                logger.warning("Users表不存在，将通过init_db()创建")
                return conn

            # 定义期望的列
            expected_columns = {
                'id', 'email', 'password_hash', 'verification_token',
                'verified', 'token_expiry', 'created_at', 'last_login'
            }

            # 检查缺少的列
            missing_columns = expected_columns - columns

            # 如果有缺少的列，添加它们
            if missing_columns:
                logger.warning(f"添加users表缺少的列: {missing_columns}")
                for column in missing_columns:
                    if column == 'verified':
                        cursor.execute("ALTER TABLE users ADD COLUMN verified INTEGER DEFAULT 0")
                    elif column == 'token_expiry':
                        cursor.execute("ALTER TABLE users ADD COLUMN token_expiry TIMESTAMP")
                    elif column == 'verification_token':
                        cursor.execute("ALTER TABLE users ADD COLUMN verification_token TEXT")
                    elif column == 'created_at':
                        cursor.execute("ALTER TABLE users ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
                    elif column == 'last_login':
                        cursor.execute("ALTER TABLE users ADD COLUMN last_login TIMESTAMP")

                conn.commit()
                logger.info("用户表结构更新完成")

            return conn
        except sqlite3.Error as e:
            logger.error(f"数据库连接错误: {str(e)}")
            raise

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()

    @staticmethod
    def validate_email_format(email: str) -> bool:
        """Validate email format"""
        if not AUTH_CONFIG['EMAIL_VALIDATION']['ENABLE']:
            return True
        return re.match(AUTH_CONFIG['EMAIL_VALIDATION']['PATTERN'], email) is not None

    @staticmethod
    def email_exists(email: str) -> bool:
        """Check if email already exists in the database"""
        try:
            with UserUtils.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM users WHERE email = ?", (email,))
                count = cursor.fetchone()[0]
                return count > 0
        except sqlite3.Error as e:
            logger.error(f"Error checking email existence: {str(e)}")
            return False

    @staticmethod
    def record_login_attempt(email: str, status: str, ip_address: str = None, user_agent: str = None) -> None:
        """Record login attempt in login_history table"""
        try:
            with UserUtils.get_db_connection() as conn:
                cursor = conn.cursor()

                # Get user ID from email
                cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
                result = cursor.fetchone()

                if result:
                    user_id = result[0]
                    # Insert login record
                    cursor.execute(
                        """INSERT INTO login_history
                        (user_id, login_time, ip_address, user_agent, login_status)
                        VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?)""",
                        (user_id, ip_address, user_agent, status)
                    )
                    conn.commit()
                    logger.info(f"Login attempt recorded: {email}, status: {status}")
                else:
                    logger.warning(f"Failed to record login: User not found {email}")
        except sqlite3.Error as e:
            logger.error(f"Error recording login attempt: {str(e)}")

    @staticmethod
    def send_verification_email(email: str, verification_token: str) -> bool:
        """Send verification email to user using email queue if enabled"""
        try:
            # Create verification link
            verification_link = f"{UserUtils.BASE_URL}?verify={verification_token}&email={email}"

            # Email body in HTML format
            body = f"""
            <html>
            <body>
                <h2>Welcome to EVS Navigation!</h2>
                <p>Thank you for registering. Please click the link below to verify your email address:</p>
                <p><a href="{verification_link}">Verify Email</a></p>
                <p>Or copy and paste this URL into your browser:</p>
                <p>{verification_link}</p>
                <p>This link will expire in 24 hours.</p>
                <p>If you did not register for an account, please ignore this email.</p>
            </body>
            </html>
            """

            subject = "Verify Your Email for EVS Navigation"

            # Check if email queue is enabled
            if UserUtils.EMAIL_QUEUE_ENABLED:
                try:
                    # Use the email queue system
                    email_queue = get_email_queue()
                    # Add metadata for tracking
                    metadata = {
                        "type": "verification",
                        "token": verification_token
                    }

                    # Queue the email with high priority (1)
                    success = email_queue.enqueue_email(
                        to_email=email,
                        subject=subject,
                        body=body,
                        is_html=True,
                        priority=1,
                        metadata=metadata
                    )

                    if success:
                        logger.info(f"Verification email to {email} queued successfully")
                        return True
                    else:
                        logger.error(f"Failed to queue verification email to {email}")
                        return False
                except Exception as e:
                    logger.error(f"Email queue error: {str(e)}. Falling back to direct sending.")
                    # Fall back to direct sending if queue fails

            # Direct sending (if queue is disabled or if queue fails)
            message = MIMEMultipart()
            message['From'] = UserUtils.SMTP_FROM_EMAIL
            message['To'] = email
            message['Subject'] = subject
            message.attach(MIMEText(body, 'html'))

            # Connect to SMTP server and send email directly
            with smtplib.SMTP(UserUtils.SMTP_SERVER, UserUtils.SMTP_PORT) as server:
                server.starttls()
                server.login(UserUtils.SMTP_USERNAME, UserUtils.SMTP_PASSWORD)
                server.send_message(message)

            logger.info(f"Verification email sent directly to {email}")
            return True

        except Exception as e:
            logger.error(f"Error sending verification email: {str(e)}")
            return False

    @staticmethod
    def init_db():
        """Initialize database with user type support and email queue"""
        with UserUtils.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    email TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    user_type TEXT DEFAULT 'user',
                    verified BOOLEAN DEFAULT FALSE,
                    verification_token TEXT,
                    token_expiry TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            """)
            conn.commit()

        # Initialize email queue if enabled
        if UserUtils.EMAIL_QUEUE_ENABLED:
            try:
                # Initialize the global email queue with SMTP configuration
                initialize_email_queue(AUTH_CONFIG['SMTP'])
                logger.info("Email queue initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize email queue: {str(e)}")
                logger.warning("Email queue disabled, falling back to direct sending")
                UserUtils.EMAIL_QUEUE_ENABLED = False

    @staticmethod
    def register_user(email, password, user_type='user'):
        """Register a new user with verification"""
        try:
            if not UserUtils.validate_email_format(email):
                logger.warning(f"Email validation failed: {email}")
                return False, "Invalid email format"

            if UserUtils.email_exists(email):
                logger.warning(f"Email already exists: {email}")
                return False, "Email already exists"

            # Generate verification token and set expiry
            verification_token = str(uuid.uuid4())
            token_expiry = datetime.now() + timedelta(hours=24)  # 24 hour expiry

            # Hash password
            password_hash = UserUtils.hash_password(password)

            with UserUtils.get_db_connection() as conn:
                cursor = conn.cursor()
                # Insert new user
                cursor.execute("""
                    INSERT INTO users
                    (email, password_hash, user_type, verified, verification_token, token_expiry)
                    VALUES (?, ?, ?, 0, ?, ?)
                """, (email, password_hash, user_type, verification_token, token_expiry))
                conn.commit()

            # Send verification email with rate limiting via queue
            if UserUtils.send_verification_email(email, verification_token):
                logger.info(f"User registered successfully: {email}")
                return True, "Registration successful. Please check your email to verify your account."
            else:
                logger.error(f"Failed to send verification email to {email}")
                return False, "Registration successful but failed to send verification email. Please contact support."

        except Exception as e:
            logger.error(f"Error registering user: {str(e)}")
            return False, f"Registration failed: {str(e)}"

    @staticmethod
    def validate_admin(email, password):
        """Validate admin credentials"""
        try:
            with UserUtils.get_db_connection() as conn:
                cursor = conn.cursor()
                password_hash = hashlib.sha256(password.encode()).hexdigest()

                cursor.execute("""
                    SELECT verified, user_type FROM users
                    WHERE email = ? AND password_hash = ?
                """, (email, password_hash))

                result = cursor.fetchone()
                if result and result[0] and result[1] == 'admin':
                    return True
            return False
        except Exception as e:
            logger.error(f"Error validating admin: {str(e)}")
            return False

    @staticmethod
    def get_all_users():
        """Get all users for admin panel"""
        try:
            with UserUtils.get_db_connection() as conn:
                query = """
                    SELECT email, user_type, verified, created_at, last_login
                    FROM users
                    ORDER BY created_at DESC
                """
                return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Error getting users: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def verify_email(email: str, verification_token: str) -> bool:
        """Verify user email using token"""
        try:
            with UserUtils.get_db_connection() as conn:
                cursor = conn.cursor()

                # Get user with matching email and token
                cursor.execute(
                    """SELECT token_expiry FROM users
                    WHERE email = ? AND verification_token = ? AND verified = 0""",
                    (email, verification_token)
                )
                result = cursor.fetchone()

                # If no matching user found
                if not result:
                    logger.warning(f"Invalid verification attempt: {email}")
                    return False

                # Check if token is expired
                token_expiry = datetime.strptime(result[0], '%Y-%m-%d %H:%M:%S')
                if token_expiry < datetime.now():
                    logger.warning(f"Expired verification token: {email}")
                    return False

                # Mark user as verified
                cursor.execute(
                    """UPDATE users SET
                    verified = 1,
                    verification_token = NULL
                    WHERE email = ?""",
                    (email,)
                )
                conn.commit()

                logger.info(f"User verified successfully: {email}")
                return True

        except (sqlite3.Error, ValueError) as e:
            logger.error(f"Email verification failed: {str(e)}")
            return False

    @staticmethod
    def validate_user(email: str, password: str, ip_address: str = None, user_agent: str = None) -> bool:
        """Validate user credentials and check verification status if enabled"""
        try:
            with UserUtils.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, password_hash, verified FROM users WHERE email = ?",
                    (email,)
                )
                user = cursor.fetchone()

                if not user:
                    UserUtils.record_login_attempt(email, "FAILED_USER_NOT_FOUND", ip_address, user_agent)
                    logger.warning(f"Login attempt failed: User does not exist {email}")
                    return False

                # Check verification status only if email verification is enabled
                if AUTH_CONFIG['ENABLE_EMAIL_VERIFICATION'] and user[2] == 0:
                    UserUtils.record_login_attempt(email, "FAILED_NOT_VERIFIED", ip_address, user_agent)
                    logger.warning(f"Login attempt failed: Email not verified {email}")
                    return False

                # Validate password
                password_hash = UserUtils.hash_password(password)
                if password_hash == user[1]:
                    cursor.execute(
                        "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE email = ?",
                        (email,)
                    )
                    conn.commit()
                    UserUtils.record_login_attempt(email, "SUCCESS", ip_address, user_agent)
                    logger.info(f"User login successful: {email}")
                    return True
                else:
                    UserUtils.record_login_attempt(email, "FAILED_WRONG_PASSWORD", ip_address, user_agent)
                    logger.warning(f"Login attempt failed: Incorrect password {email}")
                    return False

        except sqlite3.Error as e:
            try:
                UserUtils.record_login_attempt(email, f"ERROR: {str(e)[:50]}", ip_address, user_agent)
            except sqlite3.Error as record_error:
                logger.debug(f"Failed to record login attempt: {record_error}")
            logger.error(f"User validation failed: {str(e)}")
            return False

    @staticmethod
    def resend_verification_email(email: str) -> bool:
        """Resend verification email to user with new token using queue system"""
        try:
            with UserUtils.get_db_connection() as conn:
                cursor = conn.cursor()

                # Check if email exists and is not verified
                cursor.execute("SELECT verified FROM users WHERE email = ?", (email,))
                result = cursor.fetchone()

                if not result:
                    logger.warning(f"Email not found for verification resend: {email}")
                    return False

                if result[0] == 1:
                    logger.info(f"Email already verified: {email}")
                    return False

                # Generate new verification token and update expiry
                verification_token = str(uuid.uuid4())
                token_expiry = datetime.now() + timedelta(hours=24)  # 24 hour expiry

                # Update user record with new token
                cursor.execute("""
                    UPDATE users
                    SET verification_token = ?, token_expiry = ?
                    WHERE email = ?
                """, (verification_token, token_expiry, email))
                conn.commit()

                # Send verification email using queue system
                if UserUtils.send_verification_email(email, verification_token):
                    logger.info(f"Verification email resent to {email}")
                    return True
                else:
                    logger.error(f"Failed to resend verification email to {email}")
                    return False

        except Exception as e:
            logger.error(f"Error resending verification email: {str(e)}")
            return False

    @staticmethod
    def get_login_history(email: str = None, limit: int = 100) -> list:
        """Get login history for a specific user or all users"""
        try:
            with UserUtils.get_db_connection() as conn:
                cursor = conn.cursor()

                if email:
                    # Get login history for specific user
                    cursor.execute("""
                        SELECT
                            u.email,
                            h.login_time,
                            h.ip_address,
                            h.user_agent,
                            h.login_status
                        FROM login_history h
                        JOIN users u ON h.user_id = u.id
                        WHERE u.email = ?
                        ORDER BY h.login_time DESC
                        LIMIT ?
                    """, (email, limit))
                else:
                    # Get login history for all users
                    cursor.execute("""
                        SELECT
                            u.email,
                            h.login_time,
                            h.ip_address,
                            h.user_agent,
                            h.login_status
                        FROM login_history h
                        JOIN users u ON h.user_id = u.id
                        ORDER BY h.login_time DESC
                        LIMIT ?
                    """, (limit,))

                return cursor.fetchall()

        except sqlite3.Error as e:
            logger.error(f"Error retrieving login history: {str(e)}")
            return []

    @staticmethod
    def is_admin_user(email: str) -> bool:
        """检查用户是否是管理员

        Args:
            email: 用户邮箱

        Returns:
            bool: 是否是管理员
        """
        try:
            with UserUtils.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT user_type FROM users WHERE email = ?",
                    (email,)
                )
                result = cursor.fetchone()
                return result is not None and result[0] == 'admin'
        except Exception as e:
            logger.error(f"检查管理员权限失败: {str(e)}")
            return False