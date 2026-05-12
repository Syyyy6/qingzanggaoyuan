import os
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

class AnswerGenerator:
    def __init__(self):
        # ==========================================
        # 【关键修改】初始化云端大模型 (替换原来的 Ollama)
        # ==========================================
        
        # 1. 设置 API 地址 (小米专属地址)
        # 注意：如果 search.py 里加了 os.environ，这里其实可以省略，但写在这里更保险
        api_base_url = "https://api.mimo.com/v1" # <--- 请替换为小米邮件里的真实地址
        
        # 2. 设置 API Key (小米专属 Key)
        # 建议通过环境变量设置，或者直接填在这里
        api_key = os.getenv("XIAOMI_API_KEY", "你的小米API_KEY") # <--- 请替换为真实 Key
        
        # 3. 初始化模型
        # 注意：model_name 需要根据小米平台支持的模型填写，例如 "mimo-7b" 或 "gpt-4o"
        self.llm = ChatOpenAI(
            model_name="mimo-7b",      # <--- 确认小米支持的模型名称
            temperature=0.3,           # 保持严谨
            api_key=api_key,
            base_url=api_base_url,     # 核心修改：指向小米
            timeout=120
        )
        print(f">>> 已初始化云端模型 (Base URL: {api_base_url})")

    def generate(self, user_question, kg_result, rag_context):
        """
        使用云端 API 生成回答
        """
        
        # ==========================================
        # 1. 构建提示词 (保持之前的逻辑不变)
        # ==========================================
        prompt_template = """
### 角色设定
你是一名精通中国环境法与青藏高原生态保护政策的资深法律顾问。你的回答必须严谨、准确，并且能够结合最新的政策趋势为用户提供决策支持。

### 输入数据
请基于以下检索到的信息回答用户问题：

1.  **用户提问**：
    {user_question}

2.  **图谱硬规则（最高优先级）**：
    {kg_result}

3.  **参考法条与政策趋势**：
    {rag_context}

### 思维链（推理规则）
请严格按照以下步骤进行思考和推理：

1.  **事实认定与冲突解决**：
    *   **图谱优先**：首先检查【图谱硬规则】。如果图谱中有明确的“禁止”或“允许”关系，这代表最新的监管红线，必须无条件采纳。
    *   **新法优于旧法**：如果图谱无结果，需对比【参考法条】的发布日期。若发现新旧法条冲突，必须依据“新法优于旧法”原则，采信最新发布的法律。
    *   **特别法优于一般法**：优先采信专门针对“青藏高原”、“三江源”的法律法规。

2.  **趋势研判**：
    *   仔细阅读【参考法条与政策趋势】末尾的 `[政策趋势信号]`。
    *   如果信号显示“监管剧烈收紧”，即使当前处于灰色地带，也要给出强烈的风险预警。

3.  **结论生成**：
    *   结论必须非黑即白（允许/禁止/限制），严禁模棱两可。

### 输出规范
请直接输出最终回答，不要输出推理过程。格式如下：

### 1. 最终结论
**[此处填入图标：🟢允许 / 🟡限制（需审批） / 🔴禁止]** [一句话简述结论]

### 2. 法律依据
*   **核心依据**：[引用最权威的法律名称及条款]

### 3. 政策趋势与建议
[结合趋势信号给出建议]
"""

        final_prompt = prompt_template.format(
            user_question=user_question,
            kg_result=kg_result,
            rag_context=rag_context
        )

        # ==========================================
        # 2. 调用云端 API
        # ==========================================
        print(f"\n>>> 正在呼叫云端大模型...")
        
        try:
            # 使用 LangChain 的 invoke 方法发送消息
            response = self.llm.invoke([HumanMessage(content=final_prompt)])
            
            # 返回内容
            return response.content
            
        except Exception as e:
            return f"[错误] 调用云端模型失败: {str(e)}。请检查 API Key 和网络连接。"