import sqlite3
import pandas as pd
from pathlib import Path

def analyze_database(db_path):
    """分析SQLite数据库的结构和内容"""
    try:
        # 连接到数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 获取所有表名
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        print("\n=== 数据库表结构分析 ===")
        print(f"数据库路径: {db_path}")
        print(f"总表数: {len(tables)}")

        # 分析每个表
        for table in tables:
            table_name = table[0]
            print(f"\n--- 表名: {table_name} ---")

            # 获取表结构
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()

            print("列信息:")
            for col in columns:
                print(f"  - {col[1]} ({col[2]})")

            # 获取行数
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            row_count = cursor.fetchone()[0]
            print(f"总行数: {row_count}")

            # 显示前5行数据
            if row_count > 0:
                print("\n前5行数据:")
                df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5", conn)
                print(df)

            print("-" * 50)

        conn.close()

    except Exception as e:
        print(f"分析数据库时出错: {str(e)}")

if __name__ == "__main__":
    # 数据库路径
    db_path = Path("./evs_repository.db")

    if not db_path.exists():
        print(f"错误: 数据库文件 {db_path} 不存在")
    else:
        analyze_database(db_path)