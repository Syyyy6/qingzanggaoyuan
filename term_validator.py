import json
from rapidfuzz import process, fuzz  # 引入模糊匹配库

class TermValidator:
    def __init__(self, dict_path="synonyms.json"):
        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            # 提取所有标准词（字典的键）
            self.standard_terms = list(self.data.keys())
            print(f"[校验器] 已加载 {len(self.standard_terms)} 个标准术语。")
        except FileNotFoundError:
            print(f"[错误] 未找到字典文件 {dict_path}")
            self.standard_terms = []

    def check(self, user_query):
        """
        检查用户输入，并推荐最可能的 3 个标准术语。
        返回: (是否通过, 提示信息)
        """
        if not self.standard_terms:
            return True, "" 

        # --- 1. 基础检查：是否包含标准词或别名 ---
        found_any = False
        
        # 检查是否包含标准词
        for term in self.standard_terms:
            if term in user_query:
                found_any = True
                break
        
        # 检查是否包含别名
        if not found_any:
            for aliases in self.data.values():
                for alias in aliases:
                    if alias in user_query:
                        found_any = True
                        break
                if found_any: break

        # 如果找到了，直接通过
        if found_any:
            return True, "术语校验通过"

        # --- 2. 智能推荐：如果没找到，猜测用户想说什么 ---
        # 使用 RapidFuzz 提取与 user_query 最相似的 3 个标准术语
        # limit=3 表示只取前 3 名
        # scorer=fuzz.WRatio 表示使用加权比率算法（适合中文短语）
        matches = process.extract(user_query, self.standard_terms, limit=3, scorer=fuzz.WRatio)
        
        # matches 的格式是 [('标准词A', 分数, 索引), ('标准词B', 分数, 索引)...]
        # 我们只需要提取出词本身
        recommendations = [match[0] for match in matches]
        
        # 获取最高相似度分数，用于判断是否“完全不沾边”
        top_score = matches[0][1] if matches else 0

        # --- 3. 生成提示语 ---
        # 如果最高相似度极低（比如低于 20 分），说明用户输入的可能是乱码或完全无关的词
        if top_score < 20:
             msg = f"⚠️ **术语识别失败**\n\n未能在字典中找到与“{user_query}”相关的术语。\n请尝试使用更专业的词汇。"
        else:
            # 正常情况：展示最相似的 3 个词
            msg = f"⚠️ **术语识别失败**\n\n"
            msg += f"未能在您的提问中识别到标准术语。\n"
            msg += f"系统猜测您可能想问关于：**{', '.join(recommendations)}**？\n"
            msg += f"请使用上述标准术语重新提问。"

        return False, msg