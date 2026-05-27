import sys
import io
import re

# 强制标准输出使用 UTF-8 编码
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer,
    encoding='utf-8'
)

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from neo4j import GraphDatabase

# ==========================================
# 引入生成器
# ==========================================
from answer_generator import AnswerGenerator

# ==========================================
# 引入 rerank
# ==========================================
from rerank import PolicyReranker

# ==========================================
# 引入实体对齐
# ==========================================
try:

    from shitiduiqi import EntityAligner

except ImportError:

    print("[错误] 未找到 shitiduiqi.py")

    class EntityAligner:

        def __init__(self):
            pass

        def expand_query(self, q):
            return q, True, ""


class HybridSearchEngine:

    def __init__(self):

        print(">>> 正在初始化混合检索引擎...")

        # ==========================================
        # A. 实体对齐
        # ==========================================
        self.aligner = EntityAligner()

        # ==========================================
        # B. rerank
        # ==========================================
        self.reranker = PolicyReranker()

        # ==========================================
        # C. 向量库
        # ==========================================
        try:

            self.embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-zh-v1.5"
            )

            self.vector_db = FAISS.load_local(
                "vector_db_laws",
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            print("[成功] FAISS 向量库加载完成")

        except Exception as e:

            print(f"[错误] 向量库加载失败: {e}")

            self.vector_db = None

        # ==========================================
        # D. Neo4j
        # ==========================================
        try:

            self.driver = GraphDatabase.driver(
                "bolt://localhost:7687",
                auth=("neo4j", "2006315147hsy")
            )

            self.driver.verify_connectivity()

            print("[成功] Neo4j 连接成功")

        except Exception as e:

            print(f"[错误] Neo4j 连接失败: {e}")

            self.driver = None

    # ==========================================
    # 图谱查询
    # ==========================================
    def search_knowledge_graph(
            self,
            unit_name,
            project_name
    ):

        if (
            not unit_name
            or not project_name
            or unit_name == "通用条款"
            or project_name == "通用条款"
        ):

            print("[图谱查询] 检测到空值，跳过查询")

            return []

        if not self.driver:
            return []

        print(
            f"[图谱查询] "
            f"地域='{unit_name}' "
            f"项目='{project_name}'"
        )

        cypher = """
        MATCH (u:Unit)-[r:HAS_RULE]->(p:Project)

        WHERE
            u.name CONTAINS $unit
            AND
            p.name CONTAINS $project

        RETURN
            u.name AS unit_name,
            p.name AS project_name,
            r.action AS action,
            r.source AS source,
            r.detail AS detail
        """

        with self.driver.session() as session:

            try:

                result = session.run(
                    cypher,
                    unit=unit_name,
                    project=project_name
                )

                records = list(result)

                formatted_results = []

                for rec in records:

                    unit = rec["unit_name"]
                    project = rec["project_name"]

                    action = rec["action"]
                    source = rec["source"]
                    detail = rec["detail"]

                    rule = {
                        "unit": unit,
                        "project": project,
                        "action": action,
                        "source": source,
                        "detail": detail
                    }

                    formatted_results.append(rule)

                if formatted_results:

                    print(
                        f"[图谱查询] "
                        f"命中 {len(formatted_results)} 条规则"
                    )

                    return formatted_results

                else:

                    print("[图谱查询] 未命中规则")

                    return []

            except Exception as e:

                print(f"[错误] 图谱查询失败: {e}")

                return []

    # ==========================================
    # 图谱规则优先拦截
    # ==========================================
    def graph_rule_priority_judge(self, kg_results):

        """
        核心思想：

        图谱规则是“硬规则”

        一旦出现：
        禁止 > 限制 > 允许

        直接形成主结论
        """

        if not kg_results:
            return None

        actions = []

        for item in kg_results:

            action = str(
                item.get("action", "")
            )

            actions.append(action)

        # ==========================================
        # 禁止优先
        # ==========================================
        prohibit_keywords = [
            "禁止",
            "严禁",
            "不得"
        ]

        for action in actions:

            if any(
                kw in action
                for kw in prohibit_keywords
            ):

                print(
                    "[图谱规则优先] "
                    "命中禁止性规则"
                )

                return {
                    "final_conclusion": "🔴禁止",
                    "risk_level": "high"
                }

        # ==========================================
        # 限制其次
        # ==========================================
        restrict_keywords = [
            "限制",
            "审批",
            "备案"
        ]

        for action in actions:

            if any(
                kw in action
                for kw in restrict_keywords
            ):

                print(
                    "[图谱规则优先] "
                    "命中限制性规则"
                )

                return {
                    "final_conclusion": "🟡限制（需审批）",
                    "risk_level": "medium"
                }

        # ==========================================
        # 否则默认允许
        # ==========================================
        print(
            "[图谱规则优先] "
            "命中允许性规则"
        )

        return {
            "final_conclusion": "🟢允许",
            "risk_level": "low"
        }

    # ==========================================
    # 从图谱规则中提取最终裁决
    # ==========================================
    def extract_final_decision(self, kg_results):
        """
        优先级：
        禁止 > 限制 > 允许
        """
        if not kg_results:
            return None

        # --- 修改开始 ---
        # 判断 kg_results 里的元素是不是字典，如果是，就把里面的值提取出来拼成字符串
        if isinstance(kg_results[0], dict):
            # 把每个字典里的所有文本值(value)提取出来，用空格隔开，再换行拼接
            all_text = "\n".join([" ".join(str(v) for v in item.values()) for item in kg_results])
        else:
            # 如果本来就是字符串列表，就按原来的方式拼接
            all_text = "\n".join(kg_results)
        # --- 修改结束 ---

        # 禁止优先
        if (
            "禁止" in all_text
            or "严禁" in all_text
            or "不得" in all_text
        ):
            return "🔴禁止"

        # 限制
        if (
            "限制" in all_text
            or "审批" in all_text
        ):
            return "🟡限制（需审批）"

        # 允许
        if (
            "允许" in all_text
            or "可开展" in all_text
        ):
            return "🟢允许"

        return None

    # ==========================================
    # 向量检索
    # ==========================================
    def search_vector_db(self, query, top_k=5):

        if not self.vector_db:
            return [], []

        print(
            f"[向量检索] "
            f"正在检索 Top-{top_k}"
        )

        # ==========================================
        # 召回更多候选
        # ==========================================
        docs = self.vector_db.similarity_search(
            query,
            k=20
        )

        # ==========================================
        # rerank
        # ==========================================
        docs = self.reranker.rerank(
            query=query,
            docs=docs,
            embeddings_model=self.embeddings,
            top_k=top_k
        )

        results = []

        signal_tags = []

        for doc in docs:

            source = doc.metadata.get(
                'source',
                '未知法律'
            )

            date_str = doc.metadata.get(
                'date',
                '未知日期'
            )

            article = doc.metadata.get(
                'article',
                ''
            )

            strength = doc.metadata.get(
                'signal_strength',
                '未知'
            )

            polarity = doc.metadata.get(
                'polarity',
                '中性'
            )

            clause_type = doc.metadata.get(
                'clause_type',
                '具体条款'
            )

            file_type = doc.metadata.get(
                'file_type',
                'normal'
            )

            signal_tags.append({

                "date": date_str,
                "strength": strength,
                "polarity": polarity,
                "source": source,
                "clause_type": clause_type,
                "file_type": file_type
            })

            content = doc.page_content.strip()

            content = content.replace(
                '\n',
                ' '
            ).replace(
                '\u3000',
                ' '
            )

            formatted_text = (
                f"[{date_str}] "
                f"《{source}》"
                f"{article}："
                f"{content}"
            )

            results.append(formatted_text)

        return results, signal_tags

    # ==========================================
    # 政策趋势分析
    # ==========================================
    def analyze_policy_trend(self, user_question):

        if not self.vector_db:
            return "暂无相关政策趋势数据。"

        print("\n>>> [政策雷达] 正在分析监管趋势...")

        # ==========================================
        # 红头文件优先
        # ==========================================
        kg_result = ""

        filter_specific = {

            "file_type": "red_head",
            "clause_type": "具体条款"
        }

        specific_docs = self.vector_db.similarity_search(
            user_question,
            k=3,
            filter=filter_specific
        )

        if specific_docs:

            print(">>> 命中红头文件具体条款")

            kg_result = "\n\n".join([

                doc.page_content

                for doc in specific_docs
            ])

        else:

            print(">>> 启动原则性兜底")

            filter_principle = {

                "file_type": "red_head",
                "clause_type": "总则/原则"
            }

            principle_docs = self.vector_db.similarity_search(
                user_question,
                k=2,
                filter=filter_principle
            )

            if principle_docs:

                kg_result = (
                    "【根本原则兜底】\n"
                )

                kg_result += "\n".join([

                    doc.page_content

                    for doc in principle_docs
                ])

            else:

                kg_result = "暂无趋势依据"

        # ==========================================
        # 时间窗口扫描
        # ==========================================
        all_docs = self.vector_db.similarity_search(
            user_question,
            k=20
        )

        all_docs = self.reranker.rerank(
            query=user_question,
            docs=all_docs,
            embeddings_model=self.embeddings,
            top_k=20
        )

        past_count = 0
        recent_count = 0

        tight_keywords = [
            "严禁",
            "禁止",
            "关停",
            "取缔",
            "红线",
            "不得",
            "限制"
        ]

        tight_recent_count = 0

        for doc in all_docs:

            date_str = doc.metadata.get(
                'date',
                ''
            )

            year_match = re.search(
                r"(\d{4})年",
                date_str
            )

            if not year_match:
                continue

            year = int(year_match.group(1))

            text = doc.page_content

            if year <= 2014:

                past_count += 1

            elif 2017 <= year <= 2024:

                recent_count += 1

                if any(
                    kw in text
                    for kw in tight_keywords
                ):

                    tight_recent_count += 1

        # ==========================================
        # 趋势生成
        # ==========================================
        if recent_count > past_count * 1.5:

            if (
                recent_count > 0
                and
                tight_recent_count / recent_count > 0.4
            ):

                trend_conclusion = (
                    "【分趋势】监管明显收紧。"
                )

            else:

                trend_conclusion = (
                    "【分趋势】政策活跃期。"
                )

        elif past_count > 0 and recent_count == 0:

            trend_conclusion = (
                "【分趋势】常态化监管。"
            )

        else:

            trend_conclusion = (
                "【分趋势】平稳过渡。"
            )

        final_trend_advice = (
            f"【总趋势依据】\n"
            f"{kg_result}\n\n"
            f"{trend_conclusion}"
        )

        return final_trend_advice

    # ==========================================
    # 格式化图谱规则
    # ==========================================
    def format_kg_rules(self, kg_results):

        if not kg_results:
            return "未找到相关图谱规则。"

        formatted_rules = []

        for item in kg_results:

            text = f"""
[图谱硬规则]
地域：{item['unit']}
项目：{item['project']}
管控结论：{item['action']}
依据来源：{item['source']}
详细要求：{item['detail']}
            """.strip()

            formatted_rules.append(text)

        return "\n\n".join(formatted_rules)

    # ==========================================
    # 主流程
    # ==========================================
    def run(self, user_query):

        print(
            f"\n--- [系统] 收到用户提问: "
            f"{user_query} ---"
        )

        # ==========================================
        # 实体对齐
        # ==========================================
        expanded_query, is_aligned, align_msg = (
            self.aligner.expand_query(user_query)
        )

        if not is_aligned:

            print("\n" + "=" * 40)

            print("系统提示")

            print("=" * 40)

            print(align_msg)

            return

        # ==========================================
        # 解析实体
        # ==========================================
        std_location = ""
        std_project = ""

        if "|" in expanded_query:

            parts = expanded_query.split("|", 2)

            expanded_query = parts[0].strip()

            if (
                len(parts) > 1
                and
                parts[1].strip() != "通用条款"
            ):

                std_location = parts[1].strip()

            if (
                len(parts) > 2
                and
                parts[2].strip() != "通用条款"
            ):

                std_project = parts[2].strip()

            print(
                f"[实体解析] "
                f"地域='{std_location}' "
                f"项目='{std_project}'"
            )

        # ==========================================
        # 图谱查询
        # ==========================================
        if std_location and std_project:

            kg_results = self.search_knowledge_graph(
                std_location,
                std_project
            )

        else:

            print(
                "[图谱查询] "
                "未提取到精准实体"
            )

            kg_results = []
        if not kg_results:
            print("[系统拦截] 图谱未命中，终止流程。")
            
            # 构造最终回答
            final_answer = (
            f"【合规查询结果】\n"
            f"基于 {std_location} 的相关规定，\n"
            f"未检索到针对 '{std_project}' 项目的专项法律条款。\n"
            f"建议咨询当地主管部门确认。"
            )
            
            # 直接打印并返回，不再执行向量检索
            print("\n" + "=" * 40)
            print("AI 最终回答")
            print("=" * 40)
            print(final_answer)
            return final_answer # 直接返回，函数结束
        # ==========================================
        # 图谱规则优先判定（核心新增）
        # ==========================================
        graph_priority_result = (
            self.graph_rule_priority_judge(
                kg_results
            )
        )
        # ==========================================
        # 图谱硬裁决
        # ==========================================
        forced_decision = self.extract_final_decision(
            kg_results
        )

        print(
            f"[图谱硬裁决] 最终结论: "
            f"{forced_decision}"
        )
        # ==========================================
        # 向量检索
        # ==========================================
        vector_results, signal_tags = (
            self.search_vector_db(
                expanded_query,
                top_k=20
            )
        )

        # ==========================================
        # 图谱规则格式化
        # ==========================================
        kg_str = self.format_kg_rules(
            kg_results
        )

        # ==========================================
        # 只取前5条法规
        # ==========================================
        context_for_llm = "\n".join(
            vector_results[:5]
        )

        # ==========================================
        # 趋势分析
        # ==========================================
        trend_advice = self.analyze_policy_trend(
            user_query
        )

        print(
            f"\n[政策雷达] "
            f"分析完成: {trend_advice}"
        )

        # ==========================================
        # 拼接 RAG Context
        # ==========================================
        full_rag_context = (
            context_for_llm
            + "\n\n"
            + f"[政策趋势信号]: {trend_advice}"
        )

        print(
            "\n>>> 正在发送给大模型..."
        )

        # ==========================================
        # 调用生成器
        # ==========================================
        generator = AnswerGenerator()

        final_answer = generator.generate(
        user_question=user_query,
        kg_result=kg_str,
        rag_context=full_rag_context,
        forced_decision=forced_decision
        )

        # ==========================================
        # 输出
        # ==========================================
        print("\n" + "=" * 40)

        print("AI 最终回答")

        print("=" * 40)

        print(final_answer)


# ==========================================
# 测试
# ==========================================
if __name__ == "__main__":

    engine = HybridSearchEngine()

    engine.run(
        "在格尔木胡杨林能不能进行测绘工程？"
    )