import pandas as pd
from neo4j import GraphDatabase

# ================= 配置区域 =================
# 1. 检查这里：你的 Neo4j 密码是多少？
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "......"  # <--- ⚠️ 重点：如果不确定，去浏览器 http://localhost:7474 试一下

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
        """批量插入数据"""
        with self.driver.session() as session:
            for index, row in df.iterrows():
                unit_name = row['Unit']
                project_name = row['Project']
                action = row['Action']

                # 根据 Action 决定关系类型
                # 如果是"禁止"，建立 FORBIDS 关系；如果是"允许"，建立 ALLOWS 关系
                # 也可以统一建立 HAS_RULE 关系，把动作作为属性，这里演示建立不同关系
                relation_type = "FORBIDS" if "禁止" in action else "ALLOWS"

                # Cypher 查询语句
                # MERGE 确保节点不重复创建，然后建立关系
                query = f"""
                MERGE (u:Unit {{name: $unit_name}})
                MERGE (p:Project {{name: $project_name}})
                MERGE (u)-[r:{relation_type}]->(p)
                """

                try:
                    session.run(query, unit_name=unit_name, project_name=project_name)
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