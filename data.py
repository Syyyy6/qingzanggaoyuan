import os
import re
import json

# 1. 配置路径
input_folder = 'laws_data'  # 你的txt文件所在的文件夹
output_file = 'laws.jsonl'    # 最终生成的向量库源文件

# 2. 定义“信号打标”函数（核心修改：增加条款类型识别）
def add_signal_tags(text, source_name):
    """
    根据文本内容和法律名称，打上信号标签
    返回：(信号强度, 情感极性, 条款类型)
    """
    # --- 标签1：信号强度 ---
    strong_keywords = ["法", "纲要", "规划", "决定", "条例"]
    weak_keywords = ["通知", "办法", "细则", "批复", "函"]
    
    signal_strength = "弱信号"
    if any(kw in source_name for kw in strong_keywords):
        signal_strength = "强信号"
    elif any(kw in source_name for kw in weak_keywords):
        signal_strength = "弱信号"

    # --- 标签2：情感极性 ---
    tight_keywords = ["严禁", "禁止", "关停", "取缔", "红线", "一票否决", "不得", "限制", "淘汰"]
    support_keywords = ["鼓励", "支持", "补贴", "试点", "大力发展", "推广", "奖励", "优先"]
    
    tight_count = sum(text.count(kw) for kw in tight_keywords)
    support_count = sum(text.count(kw) for kw in support_keywords)
    
    polarity = "中性"
    if tight_count > support_count:
        polarity = "收紧/禁止"
    elif support_count > tight_count:
        polarity = "鼓励/支持"

    # --- 标签3：条款类型（新增逻辑，用于分级匹配） ---
    # 识别是否为总则或基本原则（用于兜底话术）
    principle_keywords = ["总则", "基本原则", "保护第一", "自然恢复为主", "统筹规划"]
    clause_type = "具体条款" # 默认为具体条款
    # 如果文本中包含总则关键词，或者条款序号在前十条（通常总则在前），且包含原则性词汇
    if any(kw in text for kw in principle_keywords) or (re.search(r"第[一二三四五六七八九十]+条", text) and len(text) < 100):
        # 进一步简单判断，如果包含“应当”、“坚持”、“原则”等词，大概率是原则性条款
        if any(kw in text for kw in ["坚持", "应当", "原则", "基础"]):
            clause_type = "总则/原则"

    return signal_strength, polarity, clause_type

# 3. 定义切分函数
def split_text_to_chunks(text, source_name, date_str, is_red_head_file):
    """
    输入：全文，法律名，日期，是否为红头文件
    输出：切分后的数据列表（带标签）
    """
    # 正则：按“第x条”切分
    pattern = r"(第[一二三四五六七八九十百零]+条.*?)(?=第[一二三四五六七八九十百零]+条|$)"
    raw_chunks = re.findall(pattern, text, re.DOTALL)

    processed_chunks = []
    for chunk in raw_chunks:
        clean_text = chunk.strip()
        if len(clean_text) > 5:
            # 调用打标函数，接收新增的条款类型
            strength, polarity, clause_type = add_signal_tags(clean_text, source_name)
            
            item = {
                "text": clean_text,
                "source": source_name,
                "date": date_str,
                "signal_strength": strength, 
                "polarity": polarity,
                "clause_type": clause_type, # 新增字段
                # 如果是《青藏高原生态保护法》，打上红头文件标签，方便检索时优先提取
                "file_type": "red_head" if is_red_head_file else "normal" 
            }
            processed_chunks.append(item)
    return processed_chunks

# 4. 主循环：遍历文件夹
all_data = []

# 定义红头文件的核心名称，用于精准识别
red_head_files = ["中华人民共和国青藏高原生态保护法"]

for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(input_folder, filename)

        # --- 文件名解析逻辑 ---
        name_without_ext = filename.replace(".txt", "")
        
        # 提取日期
        date_match = re.search(r"（(.*?)）", name_without_ext)
        if date_match:
            date_str = date_match.group(1)
            law_name = re.sub(r"（.*?）", "", name_without_ext)
        else:
            date_str = "Unknown"
            law_name = name_without_ext

        # 去除编号
        if "_" in law_name:
            law_name = law_name.split("_", 1)[1]

        print(f"正在处理：{law_name} (日期: {date_str})...")

        # 判断当前文件是否属于我们指定的“红头文件”
        is_red_head = any(rhf in law_name for rhf in red_head_files)

        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 调用切分函数，传入是否为红头文件的标识
        chunks = split_text_to_chunks(content, law_name, date_str, is_red_head)
        all_data.extend(chunks)

# 5. 写入最终文件
print(f"处理完成！共提取到 {len(all_data)} 条数据。正在写入 {output_file}...")

with open(output_file, 'w', encoding='utf-8') as f:
    for item in all_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("任务结束！")