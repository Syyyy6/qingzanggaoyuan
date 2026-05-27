import pandas as pd
from neo4j import GraphDatabase

# ================= 配置区域 =================

NEO4J_URI = "bolt://localhost:7687"

NEO4J_USER = "neo4j"

NEO4J_PASSWORD = "2006315147hsy"

CSV_FILE = "rules.csv"

# ============================================


class Neo4jConnection:

    def __init__(self, uri, user, pwd):

        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, pwd)
        )

    def close(self):

        if self.driver:
            self.driver.close()

    # ==========================================
    # 创建唯一约束
    # ==========================================
    def create_constraint(self):

        with self.driver.session() as session:

            try:

                # Unit 唯一
                session.run("""
                CREATE CONSTRAINT unit_name_unique
                IF NOT EXISTS
                FOR (u:Unit)
                REQUIRE u.name IS UNIQUE
                """)

                # Project 唯一
                session.run("""
                CREATE CONSTRAINT project_name_unique
                IF NOT EXISTS
                FOR (p:Project)
                REQUIRE p.name IS UNIQUE
                """)

                print(">>> 数据库约束检查完成")

            except Exception as e:

                print(f"[错误] 创建约束失败: {e}")

    # ==========================================
    # 插入数据
    # ==========================================
    def insert_data(self, df):

        with self.driver.session() as session:

            for index, row in df.iterrows():

                try:

                    # ==========================================
                    # 读取 Excel / CSV 字段
                    # ==========================================

                    unit_name = str(
                        row['unit']
                    ).strip()

                    project_name = str(
                        row['project']
                    ).strip()

                    action_val = str(
                        row['action']
                    ).strip()

                    source_val = str(
                        row['source']
                    ).strip()

                    detail_val = str(
                        row['detail']
                    ).strip()

                    # ==========================================
                    # Cypher
                    # ==========================================

                    query = """
                    MERGE (u:Unit {
                        name: $unit_name
                    })

                    MERGE (p:Project {
                        name: $project_name
                    })

                    MERGE (u)-[r:HAS_RULE]->(p)

                    SET
                        r.action = $action_val,
                        r.source = $source_val,
                        r.detail = $detail_val
                    """

                    session.run(
                        query,

                        unit_name=unit_name,

                        project_name=project_name,

                        action_val=action_val,

                        source_val=source_val,

                        detail_val=detail_val
                    )

                    print(
                        f"[导入成功] "
                        f"{unit_name} -> "
                        f"{project_name}"
                    )

                except Exception as e:

                    print(
                        f"[错误] 第 {index} 行导入失败: {e}"
                    )

        print("\n>>> 全部数据导入完成！")



# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":

    print(f">>> 正在读取文件: {CSV_FILE}")

    try:

        # ==========================================
        # 读取 CSV
        # ==========================================

        df = pd.read_csv(
            CSV_FILE,
            encoding='utf-8'
        )

        print(
            f">>> 成功读取 "
            f"{len(df)} 行数据"
        )

        # ==========================================
        # 初始化 Neo4j
        # ==========================================

        conn = Neo4jConnection(
            NEO4J_URI,
            NEO4J_USER,
            NEO4J_PASSWORD
        )

        # ==========================================
        # 创建约束
        # ==========================================

        conn.create_constraint()

        # ==========================================
        # 导入数据
        # ==========================================

        conn.insert_data(df)

        # ==========================================
        # 关闭连接
        # ==========================================

        conn.close()

        print("\n>>> Neo4j 图谱构建完成")

    except FileNotFoundError:

        print(
            f"[错误] 找不到文件: {CSV_FILE}"
        )

    except Exception as e:

        print(f"[错误] 程序运行失败: {e}")