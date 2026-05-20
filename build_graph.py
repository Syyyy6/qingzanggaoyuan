import pandas as pd
from neo4j import GraphDatabase
import re

# ================= 配置区域 =================
# 1. 检查这里：你的 Neo4j 密码是多少？
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "2006315147hsy"  # <--- ⚠️ 重点：请填入你的实际密码

# 2. CSV 文件路径
CSV_FILE = "rules.csv"
# ============================================

class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self.driver = GraphDatabase.driver(uri, auth=(user, pwd))

    def close(self):
        if self.driver:
            self.driver.close()

    def create_constraint(self):
        """创建唯一性约束，防止数据重复"""
        with self.driver.session() as session:
            try:
                # 为 Unit 节点创建唯一约束
                session.run("CREATE CONSTRAINT unit_name_unique IF NOT EXISTS FOR (u:Unit) REQUIRE u.name IS UNIQUE")
                # 为 Project 节点创建唯一约束
                session.run("CREATE CONSTRAINT project_name_unique IF NOT EXISTS FOR (p:Project) REQUIRE p.name IS UNIQUE")
                print(">>> 数据库约束检查完成 (Constraints checked).")
            except Exception as e:
                print(f"Error creating constraints: {e}")

    def insert_data(self, df):
        """
        批量插入数据（修改版：将 Action 作为一个整体属性存储）
        """
        with self.driver.session() as session:
            for index, row in df.iterrows():
                unit_name = str(row['Unit']).strip()
                project_name = str(row['Project']).strip()
                action_raw = str(row['Action']).strip()

                # --- 核心修改点 ---
                # 1. 不再解析拆分 Action，直接使用原始字符串
                # 2. 统一使用通用的关系类型 (如 HAS_RULE)，或者你可以根据需要保留之前的逻辑
                #    这里为了简化，我们统一使用 HAS_RULE 关系，重点在于属性 action
                
                relation_type = "HAS_RULE" 

                # 3. 构建 Cypher 查询
                # 注意：我们将属性直接存为 r.action
                query = f"""
                MERGE (u:Unit {{name: $unit_name}})
                MERGE (p:Project {{name: $project_name}})
                MERGE (u)-[r:{relation_type}]->(p)
                SET r.action = $action_val
                """

                try:
                    session.run(query, 
                                unit_name=unit_name, 
                                project_name=project_name, 
                                action_val=action_raw) # 传入完整的字符串
                except Exception as e:
                    print(f"Error inserting row {index}: {e}")

        print(">>> 数据导入完成！")

if __name__ == "__main__":
    print(f"正在读取 {CSV_FILE} ...")

    try:
        # 读取 CSV，确保使用 utf-8 编码
        df = pd.read_csv(CSV_FILE)
        print(f"成功读取 {len(df)} 行数据。")

        # 初始化连接
        conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

        # 1. 创建约束
        conn.create_constraint()

        # 2. 插入数据
        conn.insert_data(df)

        # 3. 关闭连接
        conn.close()

    except FileNotFoundError:
        print(f"错误：找不到文件 {CSV_FILE}，请确保它和脚本在同一目录下。")
    except Exception as e:
        print(f"发生未知错误: {e}")