# ==========================================
# 文件名：policy_reranker.py
# 功能：
# “政策约束感知重排序模块”
#
# 核心思想：
# 不仅看 embedding similarity
# 还融合：
# 1. 政策强度（signal_strength）
# 2. 政策倾向（polarity）
# 3. 条款层级（clause_type）
# 4. 文件法律效力（file_type）
# 5. 时间权重（新法优先）
#
# 最终实现：
# “法律感知检索”
# ==========================================

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
        # 核心：
        # 禁止 > 限制 > 中性 > 鼓励
        # ==========================================
        self.polarity_weight = {
            "收紧/禁止": 1.0,
            "中性": 0.6,
            "鼓励/支持": 0.3
        }

        # ==========================================
        # 三、条款层级权重
        # 核心：
        # 具体条款 > 总则原则
        # ==========================================
        self.clause_type_weight = {
            "具体条款": 1.0,
            "总则/原则": 0.5
        }

        # ==========================================
        # 四、法律效力权重
        # 核心：
        # 红头文件 > 普通文件
        # ==========================================
        self.file_type_weight = {
            "red_head": 1.0,
            "normal": 0.6
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
    # 时间权重（新法优于旧法）
    # ==========================================
    def calculate_time_weight(self, year):

        current_year = datetime.now().year

        diff = current_year - year

        # 越新权重越高
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
    # 计算语义相似度
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
    # 核心：
    # 计算法规总分
    # ==========================================
    def calculate_final_score(
            self,
            semantic_similarity,
            signal_strength,
            polarity,
            clause_type,
            file_type,
            time_weight
    ):

        # ==========================================
        # 超参数（论文里可调）
        # ==========================================

        alpha = 0.45   # 语义相似度
        beta = 0.20    # 政策强度
        gamma = 0.15   # 条款层级
        delta = 0.10   # 法律效力
        epsilon = 0.10 # 时间权重

        # ==========================================
        # 各项权重
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

        # ==========================================
        # PolicyWeight
        # 融合：
        # signal_strength + polarity
        # ==========================================

        policy_weight = (
            signal_score * 0.4 +
            polarity_score * 0.6
        )

        # ==========================================
        # 最终总分
        # ==========================================

        final_score = (
            alpha * semantic_similarity +
            beta * policy_weight +
            gamma * clause_score +
            delta * file_score +
            epsilon * time_weight
        )

        return round(final_score, 4)

    # ==========================================
    # 主函数：
    # 对 FAISS 召回结果重排序
    # ==========================================
    def rerank(
            self,
            query,
            docs,
            embeddings_model,
            top_k=5
    ):

        print("\n>>> [PolicyReranker] 开始政策约束感知重排序...")

        if not docs:
            return []

        # ==========================================
        # 1. query 向量化
        # ==========================================

        query_vector = embeddings_model.embed_query(query)

        reranked_results = []

        # ==========================================
        # 2. 遍历所有召回法规
        # ==========================================

        for doc in docs:

            try:

                # ==========================================
                # 文档向量
                # ==========================================

                doc_vector = embeddings_model.embed_query(
                    doc.page_content
                )

                # ==========================================
                # 语义相似度
                # ==========================================

                semantic_similarity = self.calculate_semantic_similarity(
                    query_vector,
                    doc_vector
                )

                # ==========================================
                # metadata 提取
                # ==========================================

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

                # ==========================================
                # 时间权重
                # ==========================================

                year = self.extract_year(date_str)

                time_weight = self.calculate_time_weight(year)

                # ==========================================
                # 最终评分
                # ==========================================

                final_score = self.calculate_final_score(
                    semantic_similarity=semantic_similarity,
                    signal_strength=signal_strength,
                    polarity=polarity,
                    clause_type=clause_type,
                    file_type=file_type,
                    time_weight=time_weight
                )

                # ==========================================
                # 保存结果
                # ==========================================

                reranked_results.append({
                    "doc": doc,
                    "score": final_score,
                    "semantic_similarity": semantic_similarity,
                    "signal_strength": signal_strength,
                    "polarity": polarity,
                    "clause_type": clause_type,
                    "file_type": file_type,
                    "time_weight": time_weight
                })

            except Exception as e:

                print(f"[重排序错误] {e}")

        # ==========================================
        # 3. 按总分降序排序
        # ==========================================

        reranked_results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        # ==========================================
        # 4. 打印 Top 结果（调试）
        # ==========================================

        print("\n>>> [Rerank Top Results]")

        for i, item in enumerate(reranked_results[:top_k]):

            doc = item["doc"]

            print(f"\nTOP-{i+1}")
            print(f"总分: {item['score']}")
            print(f"语义相似度: {round(item['semantic_similarity'], 4)}")
            print(f"政策强度: {item['signal_strength']}")
            print(f"政策倾向: {item['polarity']}")
            print(f"条款层级: {item['clause_type']}")
            print(f"文件级别: {item['file_type']}")
            print(f"时间权重: {item['time_weight']}")
            print(f"内容: {doc.page_content[:120]}")

        # ==========================================
        # 5. 返回 rerank 后的 docs
        # ==========================================

        final_docs = [
            item["doc"]
            for item in reranked_results[:top_k]
        ]

        return final_docs