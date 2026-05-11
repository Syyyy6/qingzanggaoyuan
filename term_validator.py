# term_validator.py
import json
import random

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
        检查用户输入。
        返回: (是否通过, 提示信息)
        """
        if not self.standard_terms:
            return True, "" # 如果字典为空，直接放行

        # 简单的检查逻辑：
        # 1. 尝试用现有的 aligner 逻辑看能不能匹配到标准词
        # 这里为了演示，我们做一个简单的反向检查：
        # 如果用户输入的 query 里包含的词，完全无法映射到任何一个标准词，则拦截。
        
        # 这里为了简化，我们只检查 query 是否包含标准词，或者标准词的别名
        # 实际项目中建议引入 jieba 分词来提高精度
        
        found_any = False
        
        # 检查是否包含标准词
        for term in self.standard_terms:
            if term in user_query:
                found_any = True
                break
        
        # 检查是否包含别名 (需要遍历字典的值)
        if not found_any:
            for aliases in self.data.values():
                for alias in aliases:
                    if alias in user_query:
                        found_any = True
                        break
                if found_any: break

        if found_any:
            return True, "术语校验通过"
        else:
            # 没找到任何已知术语，给出建议
            # 随机选3个标准词作为例子
            examples = random.sample(self.standard_terms, min(3, len(self.standard_terms)))
            msg = f"⚠️ **术语识别失败**\n\n"
            msg += f"未能在您的提问中识别到标准术语（如：{', '.join(examples)} 等）。\n"
            msg += f"为了确保回答的准确性，**请您使用标准术语重新提问**。\n"
            return False, msg