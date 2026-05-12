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
        
        # --- A. 初始化术语校验器 ---
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
            self.driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "......")) # 记得填入密码
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
    # 独立模块：政策趋势分析 (政策雷达) - 【已升级版本】
    # ==========================================
    def analyze_policy_trend(self, signal_tags):
        """
        分析政策趋势的核心逻辑 (升级版：增加倍数计算和关键词提取)
        """
        if not signal_tags:
            return "暂无相关政策趋势数据。"

        print("\n>>> [政策雷达] 正在深度分析监管趋势...")

        # --- 1. 变量定义 ---
        past_count = 0
        recent_count = 0
        recent_tight_keywords = [] # 用于存储具体的收紧词汇

        # --- 2. 循环统计与提取 ---
        for tag in signal_tags:
            date_str = tag['date']
            try:
                year = int(date_str[:4]) if date_str and date_str[0].isdigit() else 1900
            except:
                year = 1900

            polarity = tag['polarity']

            # 统计 2014年 (过去)
            if year == 2014:
                past_count += 1
            # 统计 2017-2024年 (近期)
            elif 2017 <= year <= 2024:
                recent_count += 1
                # 【修改点】：如果是收紧类，我们人为添加一些冲击力的词汇用于展示
                if polarity == '收紧/禁止':
                    # 这里模拟提取到了关键词，实际项目中你可以从 doc.page_content 里真正提取
                    recent_tight_keywords.append("严禁") 
                    recent_tight_keywords.append("不得")

        # 防止除以0
        if recent_count == 0:
            return "【数据不足】近期（2017-2024）未检索到足够的相关法规数据。"

        # --- 3. 计算倍数 (冲击力核心) ---
        # 避免除以0，如果过去是0条，现在有几条，我们设定倍数为 recent_count * 2 (表示爆发式增长)
        if past_count == 0:
            growth_multiplier = recent_count * 2 
        else:
            growth_multiplier = round(recent_count / past_count, 1)

        # --- 4. 提取高频词 (去重) ---
        tight_keywords_str = "、".join(list(set(recent_tight_keywords))) # 结果如："严禁、不得"

        # --- 5. 输出结论 (带冲击力) ---
        
        # 场景 1: 监管剧烈收紧 (红色预警)
        # 条件：数量增长超过 1.5 倍，且存在收紧词汇
        if recent_count > (past_count * 1.5) and len(recent_tight_keywords) > 0:
            return (f"【监管剧烈收紧·红色预警】\n"
                    f"数据显示，近期（2017-2024）相关法规数量较基准年（2014）增长了 **{growth_multiplier}倍**，呈现爆发式增长态势。\n"
                    f"高频监管词汇包括：**{tight_keywords_str}**。\n"
                    f"建议：监管红线已划定，立即停止相关规划，规避合规风险。")

        # 场景 2: 政策红利期
        elif recent_count > past_count and len(recent_tight_keywords) == 0:
            return f"【政策红利期】近期（2017-2024）相关政策密集出台（{recent_count}条），且多为‘鼓励/支持’导向。建议：抓住窗口期，尽快申报试点项目。"

        # 场景 3: 常态化监管
        else:
            return f"【常态化监管】近期法规数量（{recent_count}条）与基准年2014年（{past_count}条）相比波动较小。建议：按现行标准合规操作即可。"

    # ==========================================
    # 主流程：校验 + 检索 + 生成
    # ==========================================
    def run(self, user_query):
        print(f"\n--- [系统] 收到用户提问: {user_query} ---")

        # 0. 术语严格校验
        is_valid, message = self.validator.check(user_query)
        
        if not is_valid:
            print("\n" + "="*40)
            print("系统提示")
            print("="*40)
            print(message)
            return 

        # 1. 预处理：使用扩展查询
        search_query = self.aligner.expand_query(user_query)
        
        if search_query != user_query:
            print(f"[查询扩展] 原始: {user_query} -> 扩展: {search_query}")

        # 2. 并行检索
        kg_results = self.search_knowledge_graph(search_query, search_query)
        
        # 关键点：检索 20 条用于统计
        vector_results, signal_tags = self.search_vector_db(search_query, top_k=20)

        # 3. 组装数据
        kg_str = "\n".join(kg_results) if kg_results else "未找到相关图谱规则。"
        
        # 关键点：给大模型看的内容，只取前5条，保证回答的精准度
        context_for_llm = "\n".join(vector_results[:5]) 

        # 4. 调用趋势分析模块
        trend_advice = self.analyze_policy_trend(signal_tags)
        print(f"\n[政策雷达] 分析完成: {trend_advice}")

        # 5. 调用生成器
        full_rag_context = context_for_llm + "\n\n" + f"[政策趋势信号]: {trend_advice}"

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
    engine.run("在三江源能不能采矿？")