import json

class EntityAligner:
    def __init__(self, dict_path='synonyms.json'):
        # 尝试加载字典
        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                # 此时加载的格式为：{"标准词": ["别名1", "别名2"]}
                self.synonyms = json.load(f)
                print(f"[实体对齐] 成功加载同义词库，共 {len(self.synonyms)} 组标准术语。")
        except FileNotFoundError:
            self.synonyms = {}
            print("⚠️ 未找到 synonyms.json，使用空字典。")

    def expand_query(self, query):
        """
        核心逻辑：查询扩展
        策略：保留用户原话（给向量检索用），追加标准词（给图谱/关键词用）。
        """
        if not self.synonyms:
            return query

        standard_terms_to_add = []
        
        # 1. 遍历字典：standard 是标准词，aliases 是别名列表
        for standard, aliases in self.synonyms.items():
            # 检查这个标准词对应的每一个别名
            for alias in aliases:
                if alias in query:
                    # 如果命中别名，且该标准词还没被加入列表，则加入
                    if standard not in standard_terms_to_add:
                        standard_terms_to_add.append(standard)
                        print(f"🔍 识别到术语：'{alias}' -> 映射为 '{standard}'")

        # 2. 如果没有命中任何标准词，直接返回原话（靠向量模型兜底）
        if not standard_terms_to_add:
            return query
            
        # 3. 拼接：原话 + 标准词
        # 使用空格分隔，有助于向量模型区分词汇边界
        expanded_query = f"{query} {' '.join(standard_terms_to_add)}"
        return expanded_query

    def normalize(self, query):
        """
        【旧方法】暴力替换（仅用于必须精确匹配的场景，一般不推荐在主流程使用）
        """
        standard_query = query
        for standard, aliases in self.synonyms.items():
            for alias in aliases:
                if alias in query:
                    standard_query = standard_query.replace(alias, standard)
        return standard_query

# ==========================================
# 本地测试代码
# ==========================================
if __name__ == "__main__":
    # 假设你的 synonyms.json 就在同级目录下
    aligner = EntityAligner()

    print("\n--- 测试 1：命中高频词 ---")
    q1 = "我想在玉树搞风电"
    print(f"输入: {q1}")
    print(f"输出: {aligner.expand_query(q1)}")
    # 预期：我想在玉树搞风电 光伏/风电/水电

    print("\n--- 测试 2：命中多个词 ---")
    q2 = "我想去采矿还要盖楼"
    print(f"输入: {q2}")
    print(f"输出: {aligner.expand_query(q2)}")
    # 预期：我想去采矿还要盖楼 战略性矿产 房地产开发

    print("\n--- 测试 3：生僻口语（无命中）---")
    q3 = "我想在这里搞个风车转转"
    print(f"输入: {q3}")
    print(f"输出: {aligner.expand_query(q3)}")
    # 预期：我想在这里搞个风车转转 (保持不变，交给向量模型)