import json
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz


class EntityAligner:
    def __init__(self, dict_path='synonyms.json'):

        # ==========================================
        # 加载同义词字典
        # ==========================================
        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                self.synonyms = json.load(f)

            self.standard_terms = list(self.synonyms.keys())

            print(f"[实体对齐] 成功加载同义词库，共 {len(self.standard_terms)} 个标准术语。")

        except FileNotFoundError:
            self.synonyms = {}
            self.standard_terms = []
            print("[警告] 未找到 synonyms.json")

        # ==========================================
        # 初始化向量模型
        # ==========================================
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-zh-v1.5"
            )

            # 提前缓存标准术语向量
            if self.standard_terms:
                self.term_vectors = self.embeddings.embed_documents(
                    self.standard_terms
                )
            else:
                self.term_vectors = []

            print("[实体对齐] 向量模型加载成功")

        except Exception as e:
            print(f"[警告] 向量模型加载失败: {e}")
            self.embeddings = None
            self.term_vectors = []

    # ==========================================
    # 新增：向量语义实体识别
    # ==========================================
    def semantic_match(self, text, threshold=0.72):
        """
        使用向量语义匹配实体
        返回：
            [(标准术语, type, 相似度)]
        """

        if not self.embeddings or not self.term_vectors:
            return []

        results = []

        try:
            # 用户短语向量化
            query_vector = self.embeddings.embed_query(text)

            # 计算与所有标准术语的相似度
            similarities = cosine_similarity(
                [query_vector],
                self.term_vectors
            )[0]

            # 找最高相似度
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]

            # 超过阈值才认定成功
            if best_score >= threshold:

                matched_term = self.standard_terms[best_idx]

                term_info = self.synonyms.get(matched_term, {})
                term_type = term_info.get("type", "unknown")

                results.append(
                    (matched_term, term_type, float(best_score))
                )

        except Exception as e:
            print(f"[语义识别错误] {e}")

        return results

    # ==========================================
    # 核心函数
    # ==========================================
    def expand_query(self, query):

        if not self.synonyms:
            return f"{query}|{query}|{query}", True, ""

        extracted_location = ""
        extracted_project = ""

        # ==========================================
        # 第一层：向量语义识别（主逻辑）
        # ==========================================
        print("[第一层] 正在进行向量语义实体识别...")

        query_len = len(query)

        for i in range(query_len):

            for j in range(i + 2, min(i + 11, query_len + 1)):

                sub_text = query[i:j]

                semantic_results = self.semantic_match(sub_text)

                for matched_term, term_type, score in semantic_results:

                    if term_type == "location" and not extracted_location:

                        extracted_location = matched_term

                        print(
                            f"[向量识别] 地域: {matched_term} "
                            f"(触发词: {sub_text}, 相似度: {score:.3f})"
                        )

                    elif term_type == "project" and not extracted_project:

                        extracted_project = matched_term

                        print(
                            f"[向量识别] 项目: {matched_term} "
                            f"(触发词: {sub_text}, 相似度: {score:.3f})"
                        )

                # 两个实体都找到后提前结束
                if extracted_location and extracted_project:
                    break

            if extracted_location and extracted_project:
                break

        # ==========================================
        # 第二层：字典 alias 匹配（兜底）
        # ==========================================
        if not extracted_location or not extracted_project:

            print("[第二层] 向量识别不完整，启动字典兜底匹配...")

            for i in range(query_len):

                for j in range(i + 2, min(i + 11, query_len + 1)):

                    sub_text = query[i:j]

                    for standard_term, info in self.synonyms.items():

                        if sub_text in info.get('aliases', []):

                            term_type = info.get('type')

                            if term_type == 'location' and not extracted_location:

                                extracted_location = standard_term

                                print(
                                    f"[字典识别] 地域: {standard_term} "
                                    f"(触发词: {sub_text})"
                                )

                            elif term_type == 'project' and not extracted_project:

                                extracted_project = standard_term

                                print(
                                    f"[字典识别] 项目: {standard_term} "
                                    f"(触发词: {sub_text})"
                                )

                    if extracted_location and extracted_project:
                        break

                if extracted_location and extracted_project:
                    break

        # ==========================================
        # 第三层：成功返回
        # ==========================================
        if extracted_location and extracted_project:

            expanded_query = (
                f"{query}|{extracted_location}|{extracted_project}"
            )

            return expanded_query, True, ""

        # ==========================================
        # 第四层：失败拦截
        # ==========================================
        print("[识别失败] 未识别到完整实体")

        matches = process.extract(
            query,
            self.standard_terms,
            limit=3,
            scorer=fuzz.WRatio
        )

        recommendations = [m[0] for m in matches]

        msg = f"⚠️ 术语识别失败\n\n"
        msg += f"未能识别标准地域与项目术语。\n"
        msg += f"您可能想问：**{', '.join(recommendations)}**？\n"
        msg += f"请尝试包含明确地域（如：三江源）和项目（如：采矿）。"

        return f"{query}|{query}|{query}", False, msg

    # ==========================================
    # 保留旧 normalize
    # ==========================================
    def normalize(self, query):

        standard_query = query

        for standard, info in self.synonyms.items():

            for alias in info.get('aliases', []):

                if alias in query:
                    standard_query = standard_query.replace(
                        alias,
                        standard
                    )

        return standard_query


# ==========================================
# 本地测试
# ==========================================
if __name__ == "__main__":

    aligner = EntityAligner()

    print("\n--- 测试1：正常识别 ---")

    q1 = "我想在三江源搞个矿场"

    result, is_valid, msg = aligner.expand_query(q1)

    print(f"\n输入: {q1}")

    if is_valid:

        print(f"输出: {result}")

        parts = result.split('|')

        print(
            f"地域: {parts[1]} | 项目: {parts[2]}"
        )

    else:
        print(msg)

    print("\n--- 测试2：失败拦截 ---")

    q2 = "我想去那边蹦迪"

    result, is_valid, msg = aligner.expand_query(q2)

    if is_valid:
        print(result)
    else:
        print(msg)