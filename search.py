import sys
import io
import json
import re
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
# 【新增】引入政策约束感知重排序模块
# ==========================================
from rerank import PolicyReranker

# ==========================================
# 引入实体对齐类 (现在它全权负责三级漏斗防御)
# ==========================================
try:
    from shitiduiqi import EntityAligner
except ImportError:
    print("[错误] 未找到 shitiduiqi.py")
    # 提供一个假的类防止报错
    class EntityAligner:
        def __init__(self): pass
        def expand_query(self, q): return q, True, ""  # 注意：这里为了兼容，只返回3个值

class HybridSearchEngine:
    def __init__(self):
        print(">>> 正在初始化混合检索引擎...")
        
        # --- A. 初始化实体对齐器 (三级漏斗防御的核心) ---
        self.aligner = EntityAligner()
        
        # ==========================================
        # 【新增】初始化 rerank 模块
        # ==========================================
        self.reranker = PolicyReranker()
        
        # --- B. 加载 FAISS 向量库 ---
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
            self.vector_db = FAISS.load_local("vector_db_laws", self.embeddings, allow_dangerous_deserialization=True)
            print("[成功] FAISS 向量库加载完成")
        except Exception as e:
            print(f"[错误] 向量库加载失败: {e}")
            self.vector_db = None
            
        # --- C. 连接 Neo4j 图谱 ---
        try:
            # 记得填入你的真实密码
            self.driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "2006315147hsy"))
            self.driver.verify_connectivity()
            print("[成功] Neo4j 连接成功")
        except Exception as e:
            print(f"[错误] Neo4j 连接失败: {e}")
            self.driver = None

    # ==========================================
    # 路径 A: 图谱查询 (硬规则)
    # ==========================================
    def search_knowledge_graph(self, unit_name, project_name):
        # 如果传进来的是空或者默认值，直接返回空（防止图谱查全表）
        if not unit_name or not project_name or unit_name == "通用条款" or project_name == "通用条款":
            print("[图谱查询] 检测到通用条款或空值，跳过图谱查询。")
            return []
            
        if not self.driver:
            return []
        
        print(f"[图谱查询] 正在查询: 单元(地域)='{unit_name}', 项目='{project_name}'")
        
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
        
        # ==========================================
        # 【修改1】
        # 先召回更多候选法规
        # ==========================================
        docs = self.vector_db.similarity_search(query, k=20)

        # ==========================================
        # 【新增】
        # 政策约束感知重排序（核心创新）
        # ==========================================
        docs = self.reranker.rerank(
            query=query,
            docs=docs,
            embeddings_model=self.embeddings,
            top_k=top_k
        )

        results = []
        signal_tags = []
        
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', '未知法律')
            date_str = doc.metadata.get('date', '1900-01-01')
            article = doc.metadata.get('article', '')
            
            # 【修改点1】提取新增的两个核心标签
            strength = doc.metadata.get('signal_strength', '未知')
            polarity = doc.metadata.get('polarity', '中性')
            clause_type = doc.metadata.get('clause_type', '具体条款')
            file_type = doc.metadata.get('file_type', 'normal')
            
            signal_tags.append({
                "date": date_str,
                "strength": strength,
                "polarity": polarity,
                "source": source,
                "clause_type": clause_type,
                "file_type": file_type
            })
            
            content = doc.page_content.strip()
            content = content.replace('\n', ' ').replace('\u3000', ' ')
            formatted_text = f"[{date_str}] 《{source}》{article}：{content}"
            results.append(formatted_text)
            
        return results, signal_tags

    # ==========================================
    # 【修改点2】独立模块：政策趋势分析 (完全重写)
    # 实现：分级匹配（总趋势）+ 时间窗口扫描（分趋势）
    # ==========================================
    def analyze_policy_trend(self, user_question):
        if not self.vector_db:
            return "暂无相关政策趋势数据。"
            
        print("\n>>> [政策雷达] 正在深度分析监管趋势...")
        
        # --- 第一步：分级匹配（总趋势定性） ---
        kg_result = ""
        
        # 1. 精准检索：在红头文件中找具体条款
        filter_specific = {"file_type": "red_head", "clause_type": "具体条款"}
        specific_docs = self.vector_db.similarity_search(user_question, k=3, filter=filter_specific)
        
        if specific_docs:
            print(">>> 命中红头文件中的具体管控条款！")
            kg_result = "\n\n".join([doc.page_content for doc in specific_docs])
        else:
            # 2. 兜底检索：如果没找到具体条款，去捞总则原则
            print(">>> 未找到具体管控条款，触发兜底逻辑，正在检索总则/原则...")
            filter_principle = {"file_type": "red_head", "clause_type": "总则/原则"}
            principle_docs = self.vector_db.similarity_search(user_question, k=2, filter=filter_principle)
            
            if principle_docs:
                kg_result = "【根本原则兜底】法律虽未直接提及该具体事项，但根据《青藏高原生态保护法》确立的以下根本原则：\n"
                kg_result += "\n".join([doc.page_content for doc in principle_docs])
                kg_result += "\n任何活动都必须以不破坏当地生态系统为前提，面临较高的合规性审查风险。"
            else:
                kg_result = "未在核心法律库中找到相关依据。"
        
        # --- 第二步：时间窗口扫描（分趋势定量） ---
        # 扩大样本量，检索前20条相关法规用于统计
        all_relevant_docs = self.vector_db.similarity_search(user_question, k=20)

        # ==========================================
        # 【新增】
        # 对趋势分析结果也进行 rerank
        # 保证趋势分析依据更可靠
        # ==========================================
        all_relevant_docs = self.reranker.rerank(
            query=user_question,
            docs=all_relevant_docs,
            embeddings_model=self.embeddings,
            top_k=20
        )

        past_count = 0 # 过去（2014年及以前）
        recent_count = 0 # 近期（2017年-2024年）
        tight_keywords = ["严禁", "禁止", "关停", "取缔", "红线", "不得", "限制"]
        tight_recent_count = 0 # 近期文件中包含收紧词汇的数量
        
        for doc in all_relevant_docs:
            date_str = doc.metadata.get('date', '')
            # 提取年份（适配中文日期格式，如“2023年4月26日”）
            year_match = re.search(r"(\d{4})年", date_str)
            if not year_match:
                continue
            year = int(year_match.group(1))
            text = doc.page_content
            
            # 时间窗口划分与统计
            if year <= 2014:
                past_count += 1
            elif 2017 <= year <= 2024:
                recent_count += 1
                # 统计近期文件中的“收紧”词汇占比
                if any(kw in text for kw in tight_keywords):
                    tight_recent_count += 1
        
        # 生成趋势结论
        trend_conclusion = ""
        if recent_count > past_count * 1.5: # 近期数量明显多于过去
            if recent_count > 0 and tight_recent_count / recent_count > 0.4:
                trend_conclusion = f"【分趋势】监管剧烈收紧（红色预警）。近期相关法规数量（{recent_count}条）远超过去（{past_count}条），且包含大量“禁止”、“红线”等强监管词汇。"
            else:
                trend_conclusion = f"【分趋势】政策活跃期。近期相关法规数量（{recent_count}条）显著增加，表明该领域正处于政策密集落地阶段。"
        elif past_count > 0 and recent_count == 0:
            trend_conclusion = f"【分趋势】常态化监管。相关法规主要集中在过去（{past_count}条），近期暂无新增重大变动。"
        else:
            trend_conclusion = f"【分趋势】平稳过渡。近期与过去的法规数量基本持平（近期{recent_count}条 vs 过去{past_count}条）。"
        
        # 将总趋势和分趋势合并，返回给主流程
        final_trend_advice = f"【总趋势依据】\n{kg_result}\n\n{trend_conclusion}"
        return final_trend_advice

    # ==========================================
    # 主流程：实体对齐(三级漏斗) + 检索 + 生成
    # ==========================================
    def run(self, user_query):
        print(f"\n--- [系统] 收到用户提问: {user_query} ---")
        
        # 【三级漏斗防御体系】全权交由 EntityAligner 处理
        # 【核心逻辑变更】现在 expand_query 只返回 3 个值，但 expanded_query 字符串里包含了 | 分隔的标准术语
        expanded_query, is_aligned, align_msg = self.aligner.expand_query(user_query)

        # 如果实体对齐阶段失败（向量没匹配到，字典也没有），直接拦截并提示
        if not is_aligned:
            print("\n" + "="*40)
            print("系统提示")
            print("="*40)
            print(align_msg)
            return

        # --- 【核心改进】解析拼接的 Query ---
        # 初始化标准术语变量
        std_location = ""
        std_project = ""
        
        # 检查 expanded_query 中是否包含 "|" 分隔符
        if "|" in expanded_query:
            # 分割字符串，最多分割成2部分（防止正文里也有|）
            parts = expanded_query.split("|", 2)
            # 第一部分是真正的查询语句（可能包含扩展词）
            expanded_query = parts[0].strip()
            
            # 如果有第二部分，说明有标准地域
            if len(parts) > 1 and parts[1].strip() != "通用条款":
                std_location = parts[1].strip()
            
            # 如果有第三部分，说明有标准项目
            if len(parts) > 2 and parts[2].strip() != "通用条款":
                std_project = parts[2].strip()
            
            # 【调试用】打印提取结果
            print(f"[实体解析] 提取标准地域: '{std_location}', 标准项目: '{std_project}'")

        # 如果对齐成功，打印扩展后的查询语句（只打印用户看的部分）
        if expanded_query != user_query:
            print(f"[查询扩展] 原始: {user_query} -> 扩展: {expanded_query}")

        # --- 【精准查询】路径 A: 图谱查询 ---
        # 只有当 std_location 和 std_project 都不为空时，才进行精准图谱查询
        if std_location and std_project:
            kg_results = self.search_knowledge_graph(std_location, std_project)
        else:
            # 如果没有提取到精准实体，给图谱传入空值或原始Query（取决于你的Cypher逻辑，这里建议传空避免查全表）
            print("[图谱查询] 未提取到精准实体对，跳过硬规则匹配。")
            kg_results = []

        # --- 【模糊查询】路径 B: 向量检索 ---
        # 向量检索使用经过实体对齐扩展后的 Query（包含同义词扩展），效果更好
        vector_results, signal_tags = self.search_vector_db(expanded_query, top_k=20)
        
        # 3. 组装数据
        kg_str = "\n".join(kg_results) if kg_results else "未找到相关图谱规则。"
        
        # 关键点：给大模型看的内容，只取前5条，保证回答的精准度
        context_for_llm = "\n".join(vector_results[:5])
        
        # 4. 调用趋势分析模块（传入用户提问，实现分级匹配与时间窗口统计）
        trend_advice = self.analyze_policy_trend(user_query)
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