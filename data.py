import os
import re
import json

# 1. 配置路径
input_folder = 'laws_data'  # 你的txt文件所在的文件夹
output_file = 'laws.jsonl'    # 最终生成的向量库源文件

# 2. 定义“信号打标”函数
def add_signal_tags(text, source_name):
    """
    根据文本内容和法律名称，打上信号标签
    返回：(信号强度, 情感极性)
    """
    # --- 标签1：信号强度 ---
    # 强信号关键词（代表国家大方向、顶层设计）
    strong_keywords = ["法", "纲要", "规划", "决定", "条例"]
    # 弱信号关键词（代表执行层面、具体操作）
    weak_keywords = ["通知", "办法", "细则", "批复", "函"]
    
    signal_strength = "弱信号" # 默认为弱
    # 只要标题里包含强信号词，就定为强信号
    if any(kw in source_name for kw in strong_keywords):
        signal_strength = "强信号"
    # 如果标题里只有弱信号词，且没有强信号词，定为弱
    elif any(kw in source_name for kw in weak_keywords):
        signal_strength = "弱信号"

    # --- 标签2：情感极性 ---
    # 收紧/禁止类关键词
    tight_keywords = ["严禁", "禁止", "关停", "取缔", "红线", "一票否决", "不得", "限制", "淘汰"]
    # 鼓励/支持类关键词
    support_keywords = ["鼓励", "支持", "补贴", "试点", "大力发展", "推广", "奖励", "优先"]
    
    # 统计关键词出现的次数
    tight_count = sum(text.count(kw) for kw in tight_keywords)
    support_count = sum(text.count(kw) for kw in support_keywords)
    
    # 判定极性
    polarity = "中性"
    if tight_count > support_count:
        polarity = "收紧/禁止"
    elif support_count > tight_count:
        polarity = "鼓励/支持"
    # 如果数量一样，保持中性（或者你可以设定默认偏向）

    return signal_strength, polarity

# 3. 定义切分函数
def split_text_to_chunks(text, source_name, date_str):
    """
    输入：全文，法律名，日期
    输出：切分后的数据列表（带标签）
    """
    # 正则：按“第x条”切分
    pattern = r"(第[一二三四五六七八九十百零]+条.*?)(?=第[一二三四五六七八九十百零]+条|$)"
    raw_chunks = re.findall(pattern, text, re.DOTALL)

    processed_chunks = []
    for chunk in raw_chunks:
        clean_text = chunk.strip()
        if len(clean_text) > 5:
            # 调用打标函数
            strength, polarity = add_signal_tags(clean_text, source_name)
            
            item = {
                "text": clean_text,
                "source": source_name,
                "date": date_str,
                # --- 新增的标签字段 ---
                "signal_strength": strength, 
                "polarity": polarity
            }
            processed_chunks.append(item)
    return processed_chunks

# 4. 主循环：遍历文件夹
all_data = []

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

        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 调用切分函数（此时会自动打上标签）
        chunks = split_text_to_chunks(content, law_name, date_str)
        all_data.extend(chunks)

# 5. 写入最终文件
print(f"处理完成！共提取到 {len(all_data)} 条数据。正在写入 {output_file}...")

with open(output_file, 'w', encoding='utf-8') as f:
    for item in all_data:
        # ensure_ascii=False 保证中文标签正常显示，不变成 \uXXXX
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("任务结束！")