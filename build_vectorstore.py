import json
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ==========================================
# 第一步：配置
# ==========================================
JSONL_FILE = 'laws.jsonl'       # 你的预制数据文件
VECTOR_DB_PATH = 'vector_db_laws' # 输出的向量库文件夹

# ==========================================
# 第二步：加载数据与构建文档
# ==========================================
def build_store_from_jsonl():
    if not os.path.exists(JSONL_FILE):
        print(f"错误：找不到文件 '{JSONL_FILE}'，请先运行数据预处理脚本。")
        return

    docs = []
    print(f"正在读取 {JSONL_FILE} ...")

    try:
        with open(JSONL_FILE, 'r', encoding='utf-8') as f:
            # 逐行读取 JSONL
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    
                    # 将 JSON 数据转换为 LangChain 的 Document 对象
                    # page_content 是文本内容，metadata 是标签（信号、极性等）
                    doc = Document(
                        page_content=data['text'],
                        metadata={
                            "source": data['source'],
                            "date": data['date'],
                            "signal_strength": data['signal_strength'], # 核心标签：强/弱信号
                            "polarity": data['polarity']                # 核心标签：收紧/鼓励
                        }
                    )
                    docs.append(doc)
        
        print(f"成功加载 {len(docs)} 条带标签的数据片段。")

        # ==========================================
        # 第三步：向量化与保存
        # ==========================================
        print("正在生成向量 (使用 BGE 模型)...")
        
        # 使用 BGE 模型，对中文法律语义支持较好
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
        
        # 创建向量库
        # from_documents 会自动计算向量并建立索引
        db = FAISS.from_documents(docs, embeddings)
        
        # 保存到本地
        db.save_local(VECTOR_DB_PATH)
        print(f" 向量库已成功保存至 '{VECTOR_DB_PATH}' 文件夹！")
        print("   现在你可以用这个库进行带标签的检索了。")

    except Exception as e:
        print(f" 处理出错: {e}")

if __name__ == "__main__":
    build_store_from_jsonl()