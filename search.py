import sys
import io
import json
from datetime import datetime, timedelta
from collections import Counter

# 强制标准输出使用 UTF-8 编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from neo4j import GraphDatabase

# 1. 引入生成器类
from answer_generator import AnswerGenerator

# ==========================================
# 新增：引入术语校验器
# ==========================================
try:
    from term_validator import TermValidator
except ImportError:
    print("[错误] 未找到 term_validator.py，请确保该文件存在")
    # 定义一个空类防止报错，实际运行会失败
    class TermValidator:
        def __init__(self): pass
        def check(self, q): return True, ""

# ==========================================
# 引入实体对齐类
# ==========================================
try:
    from shitiduiqi import EntityAligner
except ImportError:
    print("[错误] 未找到 shitiduiqi.py")
    class EntityAligner:
        def __init__(self): pass
        def expand_query(self, q): return q

class HybridSearchEngine:
    def __init__(self):
        print(">>> 正在初始化混合检索引擎...")
        
        # --- A. 初始化术语校验器 (新增) ---
        self.validator = TermValidator()
        
        # --- B. 初始化实体对齐器 ---
        self.aligner = EntityAligner()
        
        # --- C. 加载 FAISS 向量库 ---
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
            self.vector_db = FAISS.load_local("vector_db_laws", self.embeddings, allow_dangerous_deserialization=True)
            print("[成功] FAISS 向量库加载完成")
        except Exception as e:
            print(f"[错误] 向量库加载失败: {e}")
            self.vector_db = None

        # --- D. 连接 Neo4j 图谱 ---
        try:
            self.driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "......"))
            self.driver.verify_connectivity()
            print("[成功] Neo4j 连接成功")
        except Exception as e:
            print(f"[错误] Neo4j 连接失败: {e}")
            self.driver = None

    # ==========================================
    # 路径 A: 图谱查询 (硬规则)
    # ==========================================
    def search_knowledge_graph(self, unit_name, project_name):
        if not self.driver:
            return []

        print(f"[图谱查询] 正在查询: 单元='{unit_name[:20]}...', 项目='{project_name[:20]}...'")
        
        cypher = """
        MATCH (u:Unit)-[r]-(p:Project) 
        WHERE u.name CONTAINS $unit AND p.name CONTAINS $project
        RETURN u.name as unit_name, type(r) as relation_type, p.name as project_name
        """
        
        with self.driver.session() as session:
            try:
                result = session.run(cypher, unit=unit_name, project=project_name)
                records = list(result)
                
                formatted_results = []
                for rec in records:
                    rule = f"[图谱硬规则] {rec['unit_name']} {rec['relation_type']} {rec['project_name']}"
                    formatted_results.append(rule)
                
                if formatted_results:
                    print(f"[图谱查询] 命中 {len(formatted_results)} 条硬规则!")
                    return formatted_results
                else:
                    print("[图谱查询] 未找到直接关联规则。")
                    return []
            except Exception as e:
                print(f"[错误] 图谱查询出错: {e}")
                return []

    # ==========================================
    # 路径 B: 向量检索 (软法规 + 信号标签)
    # ==========================================
    def search_vector_db(self, query, top_k=5):
        if not self.vector_db:
            return [], []

        print(f"[向量检索] 正在检索 Top-{top_k} 相似条款...")
        docs = self.vector_db.similarity_search(query, k=top_k)
        
        results = []
        signal_tags = []
        
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source_file', '未知法律')
            date_str = doc.metadata.get('date', '1900-01-01')
            article = doc.metadata.get('article', '')
            
            strength = doc.metadata.get('signal_strength', '未知')
            polarity = doc.metadata.get('polarity', '中性')
            
            signal_tags.append({
                "date": date_str,
                "strength": strength,
                "polarity": polarity,
                "source": source
            })

            content = doc.page_content.strip()
            content = content.replace('\n', ' ').replace('\u3000', ' ')
            
            formatted_text = f"[{date_str}] 《{source}》{article}：{content}"
            results.append(formatted_text)
            
        return results, signal_tags

    # ==========================================
    # 独立模块：政策趋势分析 (政策雷达)
    # ==========================================
    def analyze_policy_trend(self, signal_tags):
        if not signal_tags:
            return "暂无相关政策趋势数据。"

        print("\n>>> [政策雷达] 正在分析监管趋势...")

        threshold_year = 2023 
        
        recent_count = 0
        old_count = 0
        recent_tight = 0
        recent_support = 0

        for tag in signal_tags:
            date_str = tag['date']
            year = int(date_str[:4]) if date_str and date_str[0].isdigit() else 1900
            
            if year >= threshold_year:
                recent_count += 1
                if tag['polarity'] == '收紧/禁止':
                    recent_tight += 1
                elif tag['polarity'] == '鼓励/支持':
                    recent_support += 1
            else:
                old_count += 1

        trend_report = ""
        
        if recent_count > old_count:
            if recent_tight > recent_support:
                trend_report = f"【监管剧烈收紧】近3年相关法规数量激增（{recent_count}条），且高频出现‘禁止/严禁’词汇。建议：立即停止相关规划，规避合规风险。"
            elif recent_support > recent_tight:
                trend_report = f"【政策红利期】近3年相关政策密集出台（{recent_count}条），且多为‘鼓励/支持’导向。建议：抓住窗口期，尽快申报试点项目。"
            else:
                trend_report = f"【监管高频调整】近3年法规更新频繁（{recent_count}条），方向不一。建议：密切关注最新动态，保持合规弹性。"
        else:
            trend_report = "【监管常态化】近期相关法规更新平稳。建议：按现行标准合规操作即可。"
            
        return trend_report

    # ==========================================
    # 主流程：校验 + 检索 + 生成
    # ==========================================
    def run(self, user_query):
        print(f"\n--- [系统] 收到用户提问: {user_query} ---")

        # [新增] 0. 术语严格校验
        is_valid, message = self.validator.check(user_query)
        
        if not is_valid:
            # 如果校验不通过，直接打印提示，结束流程
            print("\n" + "="*40)
            print("系统提示")
            print("="*40)
            print(message)
            return # 直接退出函数，不再执行后续检索

        # 1. 预处理：使用扩展查询
        # 只有校验通过才会执行到这里
        search_query = self.aligner.expand_query(user_query)
        
        # 打印一下扩展后的效果，方便调试
        if search_query != user_query:
            print(f"[查询扩展] 原始: {user_query} -> 扩展: {search_query}")

        # 2. 并行检索
        # 把扩展后的 query 传给图谱（精确匹配标准词）
        kg_results = self.search_knowledge_graph(search_query, search_query)
        
        # 把扩展后的 query 传给向量（语义匹配原话+标准词）
        vector_results, signal_tags = self.search_vector_db(search_query)

        # 3. 组装数据
        kg_str = "\n".join(kg_results) if kg_results else "未找到相关图谱规则。"
        vector_str = "\n".join(vector_results) if vector_results else "未找到相关参考法条。"

        # 4. 调用趋势分析模块
        trend_advice = self.analyze_policy_trend(signal_tags)
        print(f"\n[政策雷达] 分析完成: {trend_advice}")

        # 5. 调用生成器 (发送给阿里云)
        # 注意：我们把趋势报告也塞进 rag_context 里
        full_rag_context = vector_str + "\n\n" + f"[政策趋势信号]: {trend_advice}"

        print("\n>>> 正在将检索结果发送给阿里云大模型...")
        generator = AnswerGenerator()

        final_answer = generator.generate(
            user_question=user_query,
            kg_result=kg_str,
            rag_context=full_rag_context
        )

        # 6. 输出
        print("\n" + "="*40)
        print("AI 最终回答")
        print("="*40)
        print(final_answer)

# ==========================================
# 测试运行
# ==========================================
if __name__ == "__main__":
    engine = HybridSearchEngine()
    # 你可以尝试修改这里的测试语，看看校验器的效果
    # 测试1 (标准词): "在三江源能不能采矿？" -> 应该通过
    # 测试2 (非标准词): "我想在三江源搞个风车转转" -> 假设字典没风车，应该拦截
    engine.run("在三江源能不能采矿？")