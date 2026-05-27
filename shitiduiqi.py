import json
import numpy as np

from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

from rapidfuzz import process, fuzz


class EntityAligner:

    def __init__(
            self,
            location_dict_path='location.json',
            project_dict_path='project.json'
    ):

        # ==========================================
        # 加载地域词典
        # ==========================================
        try:

            with open(
                location_dict_path,
                'r',
                encoding='utf-8'
            ) as f:

                self.location_dict = json.load(f)

            print(
                f"[实体对齐] 成功加载地域词典："
                f"{len(self.location_dict)} 个地域实体"
            )

        except Exception as e:

            print(f"[警告] 地域词典加载失败: {e}")

            self.location_dict = {}

        # ==========================================
        # 加载项目词典
        # ==========================================
        try:

            with open(
                project_dict_path,
                'r',
                encoding='utf-8'
            ) as f:

                self.project_dict = json.load(f)

            print(
                f"[实体对齐] 成功加载项目词典："
                f"{len(self.project_dict)} 个项目实体"
            )

        except Exception as e:

            print(f"[警告] 项目词典加载失败: {e}")

            self.project_dict = {}

        # ==========================================
        # 地域标准术语
        # ==========================================
        self.location_terms = list(
            self.location_dict.keys()
        )

        # ==========================================
        # 项目标标准术语
        # ==========================================
        self.project_terms = list(
            self.project_dict.keys()
        )

        # ==========================================
        # 初始化 embedding 模型
        # ==========================================
        try:

            self.embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-zh-v1.5"
            )

            print("[实体对齐] embedding 模型加载成功")

        except Exception as e:

            print(f"[警告] embedding 模型加载失败: {e}")

            self.embeddings = None

        # ==========================================
        # 分类型向量
        # ==========================================
        self.location_vectors = []
        self.project_vectors = []

        # ==========================================
        # 地域向量
        # ==========================================
        if self.embeddings and self.location_terms:

            self.location_vectors = (
                self.embeddings.embed_documents(
                    self.location_terms
                )
            )

            print(
                f"[实体对齐] 已生成 "
                f"{len(self.location_vectors)} 个地域向量"
            )

        # ==========================================
        # 项目向量
        # ==========================================
        if self.embeddings and self.project_terms:

            self.project_vectors = (
                self.embeddings.embed_documents(
                    self.project_terms
                )
            )

            print(
                f"[实体对齐] 已生成 "
                f"{len(self.project_vectors)} 个项目向量"
            )

        # ==========================================
        # 不同阈值（核心改进）
        # ==========================================
        self.location_threshold = 0.72
        self.project_threshold = 0.66

    # ==========================================
    # alias 最长匹配
    # ==========================================
    def exact_match_from_dict(
            self,
            query,
            entity_dict,
            entity_type
    ):

        """
        核心改进：

        1. alias 最长优先
        2. alias 命中后锁定
        3. 标准术语也参与匹配
        """

        best_term = ""
        best_len = 0

        for standard_term, info in entity_dict.items():

            aliases = info.get("aliases", [])

            # 标准术语也加入匹配
            all_candidates = aliases + [standard_term]

            for alias in all_candidates:

                if alias in query:

                    alias_len = len(alias)

                    # 最长优先
                    if alias_len > best_len:

                        best_term = standard_term

                        best_len = alias_len

                        print(
                            f"[精确匹配-{entity_type}] "
                            f"{standard_term} "
                            f"(命中: {alias})"
                        )

        return best_term

    # ==========================================
    # location embedding 匹配
    # ==========================================
    def semantic_match_location(
            self,
            text
    ):

        if (
                not self.embeddings
                or not self.location_vectors
        ):
            return None

        try:

            query_vector = self.embeddings.embed_query(
                text
            )

            similarities = cosine_similarity(
                [query_vector],
                self.location_vectors
            )[0]

            # ==========================================
            # Top-2
            # ==========================================
            top2_idx = np.argsort(
                similarities
            )[-2:][::-1]

            best_idx = top2_idx[0]

            best_score = similarities[best_idx]

            second_score = (
                similarities[top2_idx[1]]
                if len(top2_idx) > 1
                else 0
            )

            # ==========================================
            # second-best fallback
            # ==========================================
            score_gap = best_score - second_score

            if (
                    best_score >= self.location_threshold
                    and score_gap > 0.05
            ):

                matched_term = (
                    self.location_terms[best_idx]
                )

                return (
                    matched_term,
                    float(best_score)
                )

        except Exception as e:

            print(f"[地域语义识别错误] {e}")

        return None

    # ==========================================
    # project embedding 匹配
    # ==========================================
    def semantic_match_project(
            self,
            text
    ):

        if (
                not self.embeddings
                or not self.project_vectors
        ):
            return None

        try:

            query_vector = self.embeddings.embed_query(
                text
            )

            similarities = cosine_similarity(
                [query_vector],
                self.project_vectors
            )[0]

            # ==========================================
            # Top-2
            # ==========================================
            top2_idx = np.argsort(
                similarities
            )[-2:][::-1]

            best_idx = top2_idx[0]

            best_score = similarities[best_idx]

            second_score = (
                similarities[top2_idx[1]]
                if len(top2_idx) > 1
                else 0
            )

            # ==========================================
            # second-best fallback
            # ==========================================
            score_gap = best_score - second_score

            if (
                    best_score >= self.project_threshold
                    and score_gap > 0.03
            ):

                matched_term = (
                    self.project_terms[best_idx]
                )

                return (
                    matched_term,
                    float(best_score)
                )

        except Exception as e:

            print(f"[项目语义识别错误] {e}")

        return None

    # ==========================================
    # expand_query
    # ==========================================
    def expand_query(self, query):

        # ==========================================
        # 无词典保护
        # ==========================================
        if (
                not self.location_dict
                or not self.project_dict
        ):

            return (
                f"{query}|{query}|{query}",
                True,
                ""
            )

        extracted_location = ""
        extracted_project = ""

        # ==========================================
        # 第一层：
        # alias 精确匹配（最高优先级）
        # ==========================================
        print("\n[第一层] alias 精确匹配")

        # ==========================================
        # 地域
        # ==========================================
        extracted_location = (
            self.exact_match_from_dict(
                query=query,
                entity_dict=self.location_dict,
                entity_type="地域"
            )
        )

        # ==========================================
        # 项目
        # ==========================================
        extracted_project = (
            self.exact_match_from_dict(
                query=query,
                entity_dict=self.project_dict,
                entity_type="项目"
            )
        )

        # ==========================================
        # alias 命中后锁定
        # 不再做 embedding 覆盖
        # ==========================================
        location_locked = bool(extracted_location)
        project_locked = bool(extracted_project)

        # ==========================================
        # 第二层：
        # embedding 补充识别
        # ==========================================
        if not location_locked or not project_locked:

            print("\n[第二层] embedding 语义识别")

            query_len = len(query)

            for i in range(query_len):

                for j in range(
                        i + 2,
                        min(i + 12, query_len + 1)
                ):

                    sub_text = query[i:j]

                    # ==========================================
                    # location
                    # ==========================================
                    if not location_locked:

                        location_result = (
                            self.semantic_match_location(
                                sub_text
                            )
                        )

                        if location_result:

                            matched_term, score = (
                                location_result
                            )

                            extracted_location = (
                                matched_term
                            )

                            location_locked = True

                            print(
                                f"[向量识别-地域] "
                                f"{matched_term} "
                                f"(触发词: {sub_text}, "
                                f"score={score:.3f})"
                            )

                    # ==========================================
                    # project
                    # ==========================================
                    if not project_locked:

                        project_result = (
                            self.semantic_match_project(
                                sub_text
                            )
                        )

                        if project_result:

                            matched_term, score = (
                                project_result
                            )

                            extracted_project = (
                                matched_term
                            )

                            project_locked = True

                            print(
                                f"[向量识别-项目] "
                                f"{matched_term} "
                                f"(触发词: {sub_text}, "
                                f"score={score:.3f})"
                            )

                    # ==========================================
                    # 全找到提前退出
                    # ==========================================
                    if (
                            location_locked
                            and project_locked
                    ):
                        break

                if (
                        location_locked
                        and project_locked
                ):
                    break

        # ==========================================
        # 第三层：
        # 成功返回
        # ==========================================
        if (
                extracted_location
                and extracted_project
        ):

            expanded_query = (
                f"{query}|"
                f"{extracted_location}|"
                f"{extracted_project}"
            )

            return (
                expanded_query,
                True,
                ""
            )

        # ==========================================
        # 第四层：
        # fallback 推荐
        # ==========================================
        print("\n[识别失败] 未识别完整实体")

        all_terms = (
                self.location_terms
                + self.project_terms
        )

        matches = process.extract(
            query,
            all_terms,
            limit=5,
            scorer=fuzz.WRatio
        )

        recommendations = [
            m[0]
            for m in matches
        ]

        msg = "术语识别失败\n\n"

        if not extracted_location:

            msg += "未识别到地域实体。\n"

        if not extracted_project:

            msg += "未识别到项目实体。\n"

        msg += (
            "\n您可能想问：\n"
            f"{', '.join(recommendations)}\n"
        )

        msg += (
            "\n请尝试使用：\n"
            "明确地域 + 明确行为\n"
            "例如：\n"
            "“在三江源能不能采矿”"
        )

        return (
            f"{query}|{query}|{query}",
            False,
            msg
        )

    # ==========================================
    # normalize
    # ==========================================
    def normalize(self, query):

        standard_query = query

        # ==========================================
        # 地域归一化
        # ==========================================
        for standard, info in self.location_dict.items():

            aliases = info.get("aliases", [])

            for alias in aliases:

                if alias in standard_query:

                    standard_query = (
                        standard_query.replace(
                            alias,
                            standard
                        )
                    )

        # ==========================================
        # 项目归一化
        # ==========================================
        for standard, info in self.project_dict.items():

            aliases = info.get("aliases", [])

            for alias in aliases:

                if alias in standard_query:

                    standard_query = (
                        standard_query.replace(
                            alias,
                            standard
                        )
                    )

        return standard_query


# ==========================================
# 本地测试
# ==========================================
if __name__ == "__main__":

    aligner = EntityAligner()

    print("\n========================")
    print("测试1")
    print("========================")

    q1 = "在格尔木胡杨林能不能采矿"

    result, is_valid, msg = (
        aligner.expand_query(q1)
    )

    print(f"\n输入: {q1}")

    if is_valid:

        print(f"输出: {result}")

        parts = result.split('|')

        print(
            f"地域: {parts[1]}"
        )

        print(
            f"项目: {parts[2]}"
        )

    else:

        print(msg)

    print("\n========================")
    print("测试2")
    print("========================")

    q2 = "三江源能不能搞开发"

    result, is_valid, msg = (
        aligner.expand_query(q2)
    )

    print(f"\n输入: {q2}")

    if is_valid:

        print(f"输出: {result}")

    else:

        print(msg)