import re
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class PolicyReranker:

    def __init__(self):

        # ==========================================
        # 一、政策强度权重
        # ==========================================
        self.signal_strength_weight = {
            "强信号": 1.0,
            "弱信号": 0.5
        }

        # ==========================================
        # 二、政策倾向权重
        # ==========================================
        self.polarity_weight = {
            "收紧/禁止": 1.0,
            "中性": 0.6,
            "鼓励/支持": 0.3
        }

        # ==========================================
        # 三、条款层级权重
        # ==========================================
        self.clause_type_weight = {
            "具体条款": 1.0,
            "总则/原则": 0.5
        }

        # ==========================================
        # 四、法律效力权重
        # ==========================================
        self.file_type_weight = {
            "red_head": 1.0,
            "normal": 0.6
        }

        # ==========================================
        # 五、准入动作权重（新增）
        # 禁止 > 限制 > 允许
        # ==========================================
        self.action_level_weight = {
            "禁止": 1.0,
            "限制": 0.7,
            "允许": 0.4
        }

    # ==========================================
    # 提取年份
    # ==========================================
    def extract_year(self, date_str):

        if not date_str:
            return 2000

        match = re.search(r"(\d{4})年", date_str)

        if match:
            return int(match.group(1))

        return 2000

    # ==========================================
    # 时间权重
    # ==========================================
    def calculate_time_weight(self, year):

        current_year = datetime.now().year

        diff = current_year - year

        if diff <= 1:
            return 1.0
        elif diff <= 3:
            return 0.9
        elif diff <= 5:
            return 0.8
        elif diff <= 10:
            return 0.6
        else:
            return 0.4

    # ==========================================
    # 语义相似度
    # ==========================================
    def calculate_semantic_similarity(
            self,
            query_vector,
            doc_vector
    ):

        similarity = cosine_similarity(
            [query_vector],
            [doc_vector]
        )[0][0]

        return float(similarity)

    # ==========================================
    # 地域匹配
    # ==========================================
    def calculate_location_score(
            self,
            query,
            location
    ):

        if not location:
            return 0.0

        if location in query:
            return 1.0

        return 0.0

    # ==========================================
    # 项目匹配
    # ==========================================
    def calculate_project_score(
            self,
            query,
            project
    ):

        if not project:
            return 0.0

        if project in query:
            return 1.0

        return 0.0

    # ==========================================
    # 最终评分
    # ==========================================
    def calculate_final_score(
            self,
            semantic_similarity,
            signal_strength,
            polarity,
            clause_type,
            file_type,
            time_weight,
            location_score,
            project_score,
            action_level
    ):

        # ==========================================
        # 超参数
        # ==========================================

        alpha = 0.30  # 语义
        beta = 0.15   # 政策强度
        gamma = 0.10  # 条款层级
        delta = 0.10  # 法律效力
        epsilon = 0.10 # 时间权重

        zeta = 0.15   # 地域匹配
        eta = 0.15    # 项目匹配

        theta = 0.15  # 禁止优先

        # ==========================================
        # 各项得分
        # ==========================================

        signal_score = self.signal_strength_weight.get(
            signal_strength,
            0.5
        )

        polarity_score = self.polarity_weight.get(
            polarity,
            0.5
        )

        clause_score = self.clause_type_weight.get(
            clause_type,
            0.5
        )

        file_score = self.file_type_weight.get(
            file_type,
            0.5
        )

        action_score = self.action_level_weight.get(
            action_level,
            0.5
        )

        # ==========================================
        # policy weight
        # ==========================================

        policy_weight = (
            signal_score * 0.4 +
            polarity_score * 0.6
        )

        # ==========================================
        # final score
        # ==========================================

        final_score = (
            alpha * semantic_similarity +
            beta * policy_weight +
            gamma * clause_score +
            delta * file_score +
            epsilon * time_weight +
            zeta * location_score +
            eta * project_score +
            theta * action_score
        )

        return round(final_score, 4)

    # ==========================================
    # rerank 主函数
    # ==========================================
    def rerank(
            self,
            query,
            docs,
            embeddings_model,
            top_k=5
    ):

        print("\n>>> [PolicyReranker] 开始重排序...")

        if not docs:
            return []

        query_vector = embeddings_model.embed_query(query)

        reranked_results = []

        for doc in docs:

            try:

                # ==========================================
                # 文档向量
                # ==========================================

                doc_vector = embeddings_model.embed_query(
                    doc.page_content
                )

                semantic_similarity = (
                    self.calculate_semantic_similarity(
                        query_vector,
                        doc_vector
                    )
                )

                metadata = doc.metadata

                signal_strength = metadata.get(
                    "signal_strength",
                    "弱信号"
                )

                polarity = metadata.get(
                    "polarity",
                    "中性"
                )

                clause_type = metadata.get(
                    "clause_type",
                    "具体条款"
                )

                file_type = metadata.get(
                    "file_type",
                    "normal"
                )

                date_str = metadata.get(
                    "date",
                    ""
                )

                location = metadata.get(
                    "location",
                    ""
                )

                project = metadata.get(
                    "project",
                    ""
                )

                action_level = metadata.get(
                    "action_level",
                    "限制"
                )

                # ==========================================
                # 时间权重
                # ==========================================

                year = self.extract_year(date_str)

                time_weight = (
                    self.calculate_time_weight(year)
                )

                # ==========================================
                # 地域/项目匹配
                # ==========================================

                location_score = (
                    self.calculate_location_score(
                        query,
                        location
                    )
                )

                project_score = (
                    self.calculate_project_score(
                        query,
                        project
                    )
                )

                # ==========================================
                # 最终分数
                # ==========================================

                final_score = (
                    self.calculate_final_score(
                        semantic_similarity,
                        signal_strength,
                        polarity,
                        clause_type,
                        file_type,
                        time_weight,
                        location_score,
                        project_score,
                        action_level
                    )
                )

                reranked_results.append({
                    "doc": doc,
                    "score": final_score
                })

            except Exception as e:

                print(f"[重排序错误] {e}")

        # ==========================================
        # 排序
        # ==========================================

        reranked_results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        # ==========================================
        # 返回 TopK
        # ==========================================

        final_docs = [
            item["doc"]
            for item in reranked_results[:top_k]
        ]

        return final_docs