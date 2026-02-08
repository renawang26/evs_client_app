import sqlite3
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class DBManager:
    """统一的数据库管理类，负责所有模块的数据库连接和表初始化"""
    REPOSITORY_ROOT = "./data"
    _instance = None
    _db_connection = None
    _db_path = os.path.join(REPOSITORY_ROOT, "evs_repository.db")
    _sql_schema_file = "create_evs_tables.sql"

    @classmethod
    def get_instance(cls):
        """单例模式获取数据库管理实例"""
        if cls._instance is None:
            cls._instance = DBManager()
        return cls._instance

    def __init__(self):
        """初始化数据库管理器"""
        if DBManager._instance is not None:
            raise RuntimeError("请使用 get_instance() 获取 DBManager 实例")

        # 确保数据库文件所在目录存在
        db_dir = Path(self._db_path).parent
        if not db_dir.exists():
            os.makedirs(db_dir)

        # 初始化数据库连接
        self._initialize_connection()

        # 初始化所有表结构
       # self._initialize_tables_from_sql()

    def _initialize_connection(self):
        """初始化数据库连接"""
        try:
            self._db_connection = sqlite3.connect(self._db_path, check_same_thread=False)
            self._db_connection.row_factory = sqlite3.Row
            logger.info(f"数据库连接已建立: {self._db_path}")
        except sqlite3.Error as e:
            logger.error(f"数据库连接失败: {str(e)}")
            raise

    def _initialize_tables_from_sql(self):
        """从SQL文件初始化所有模块的数据库表"""
        try:
            # 检查SQL文件是否存在
            sql_path = Path(self._sql_schema_file)
            if not sql_path.exists():
                logger.error(f"SQL模式文件不存在: {sql_path}")
                # 如果SQL文件不存在，则使用硬编码的表定义
                self._initialize_tables_hardcoded()
                return

            # 读取SQL文件内容
            with open(sql_path, 'r') as f:
                sql_content = f.read()

            # 拆分SQL语句并执行
            cursor = self._db_connection.cursor()

            # 按分号拆分语句
            sql_statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]

            success_count = 0
            for i, stmt in enumerate(sql_statements):
                try:
                    cursor.execute(stmt)
                    success_count += 1
                except sqlite3.Error as e:
                    logger.warning(f"执行第{i+1}个SQL语句时出现错误: {e}")
                    logger.debug(f"出错的SQL: {stmt[:100]}...")

            self._db_connection.commit()
            logger.info(f"从SQL文件成功执行了 {success_count}/{len(sql_statements)} 个建表语句")

        except Exception as e:
            logger.error(f"从SQL文件初始化表失败: {str(e)}")
            # 回退到硬编码的表定义
            logger.info("尝试使用硬编码的表定义...")
            self._initialize_tables_hardcoded()

    def _initialize_tables_hardcoded(self):
        """使用硬编码的SQL语句初始化表（备用方案）"""
        try:
            cursor = self._db_connection.cursor()

            # 用户表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    verification_token TEXT,
                    verified INTEGER DEFAULT 0,
                    token_expiry TIMESTAMP,
                    user_type TEXT
                )
            """)

            # 登录历史表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS login_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    user_agent TEXT,
                    login_status TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)

            # 邮件指标表
            cursor.execute("""
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
            """)

            # 邮件指标摘要表
            cursor.execute("""
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
            """)

            self._db_connection.commit()
            logger.info("使用硬编码SQL成功初始化所有必要的表")

        except sqlite3.Error as e:
            logger.error(f"使用硬编码SQL初始化表失败: {str(e)}")
            raise

    def execute_query(self, query, params=None):
        """执行SQL查询并返回结果"""
        try:
            cursor = self._db_connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor
        except sqlite3.Error as e:
            logger.error(f"执行查询失败: {str(e)}")
            raise

    def get_connection(self):
        """获取数据库连接"""
        if self._db_connection is None:
            self._initialize_connection()
        return self._db_connection

    def close(self):
        """关闭数据库连接"""
        if self._db_connection:
            self._db_connection.close()
            self._db_connection = None
            logger.info("数据库连接已关闭")

# 提供获取数据库连接的全局函数
def get_db_connection():
    """获取统一的数据库连接"""
    return DBManager.get_instance().get_connection()