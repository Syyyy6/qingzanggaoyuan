import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==========================================
# 第一步：配置元数据映射规则
# ==========================================

FILE_CONFIG = {
    "01_青藏高原生态保护法.txt": {"level": 1, "date": "2023-04-26", "type": "国家法律"},
    "中华人民共和国自然保护区条例.txt": {"level": 2, "date": "2026-06-01", "type": "行政法规"}, 
    "青海湖流域生态环境保护条例.txt": {"level": 3, "date": "2022-01-01", "type": "地方特殊法"},
    "青海省生态环境保护条例.txt": {"level": 3, "date": "2022-05-01", "type": "地方法规"},
    "甘肃省环境保护条例.txt": {"level": 3, "date": "2019-09-26", "type": "地方法规"},
    "四川省环境保护条例.txt": {"level": 3, "date": "2017-09-22", "type": "地方法规"},
    "西藏自治区环境保护条例.txt": {"level": 3, "date": "2018-09-20", "type": "地方法规"},
    "云南省生态环境保护条例.txt": {"level": 3, "date": "2024-09-01", "type": "地方法规"},
}

# ==========================================
# 新增：信号打标逻辑
# ==========================================

def get_signal_tags(text_content, source_filename):
    """
    根据文本内容和文件名，计算信号标签
    """
    # 1. 信号强度：主要看文件名（source_filename）
    strong_keywords = ["法", "纲要", "规划", "决定", "条例"]
    weak_keywords = ["通知", "办法", "细则", "批复", "函"]
    
    signal_strength = "弱信号"
    if any(kw in source_filename for kw in strong_keywords):
        signal_strength = "强信号"
    elif any(kw in source_filename for kw in weak_keywords):
        signal_strength = "弱信号"

    # 2. 情感极性：主要看具体条款内容（text_content）
    tight_keywords = ["严禁", "禁止", "关停", "取缔", "红线", "一票否决", "不得", "限制", "淘汰"]
    support_keywords = ["鼓励", "支持", "补贴", "试点", "大力发展", "推广", "奖励", "优先"]
    
    tight_count = sum(text_content.count(kw) for kw in tight_keywords)
    support_count = sum(text_content.count(kw) for kw in support_keywords)
    
    polarity = "中性"
    if tight_count > support_count:
        polarity = "收紧/禁止"
    elif support_count > tight_count:
        polarity = "鼓励/支持"
        
    return signal_strength, polarity

def extract_metadata(filename):
    """根据文件名返回预设的元数据"""
    config = FILE_CONFIG.get(filename)
    if config:
        return config
    
    # 兜底逻辑
    date_match = re.search(r'\d{4}[-年]\d{1,2}[-月]\d{1,2}', filename)
    date_str = date_match.group(0) if date_match else "2000-01-01"
    return {"level": 9, "date": date_str, "type": "未知法规"}

# ==========================================
# 第二步：加载与精准切分
# ==========================================

def build_store():
    docs = []
    
    # 针对法律文本的切分策略
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n第",    
            "\n\n",    
            "。",      
            "；",      
            "\n",      
            " "        
        ],
        chunk_size=800,      
        chunk_overlap=100,   
        length_function=len,
    )

    data_dir = "laws_data" 
    print(f"正在读取 {data_dir} 目录下的法律文本...")
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)
            
            # 1. 加载文本
            loader = TextLoader(file_path, encoding='utf-8')
            raw_docs = loader.load()
            
            # 2. 获取基础元数据 (日期、层级等)
            base_meta = extract_metadata(filename)
            
            # 3. 切分文本
            split_docs = text_splitter.split_documents(raw_docs)
            
            for doc in split_docs:
                # --- 核心修改点开始 ---
                
                # A. 注入基础元数据
                doc.metadata.update(base_meta)
                doc.metadata['source_file'] = filename 
                
                # B. 计算并注入信号标签
                # 注意：这里传入 doc.page_content，因为极性是看具体条款内容的
                strength, polarity = get_signal_tags(doc.page_content, filename)
                doc.metadata['signal_strength'] = strength
                doc.metadata['polarity'] = polarity
                
                # --- 核心修改点结束 ---
                
            docs.extend(split_docs)
            print(f"已处理: {filename} -> 切分为 {len(split_docs)} 个片段")

    print(f"总共切分出 {len(docs)} 个文本块。")

    # ==========================================
    # 第三步：向量化与保存
    # ==========================================
    print("正在生成向量 (使用 BGE 模型)...")
    
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    
    # 创建向量库
    db = FAISS.from_documents(docs, embeddings)
    
    # 保存到本地
    db.save_local("vector_db_laws")
    print("向量库已保存至 'vector_db_laws' 文件夹！")

if __name__ == "__main__":
    build_store()