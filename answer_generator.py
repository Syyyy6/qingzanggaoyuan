import requests
import json

class AnswerGenerator:
    def __init__(self):
        # Ollama 默认运行在本地 11434 端口
        self.url = "http://localhost:11434/api/generate"
        # 这里填你刚才下载的模型名字，比如 'qwen2.5:7b' 或 'llama3'
        self.model_name = "qwen2.5:7b" 

    def generate(self, user_question, kg_result, rag_context):
        """
        使用本地 Ollama 模型生成回答
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
        # 2. 调用本地 Ollama 模型
        # ==========================================
        print(f"\n>>> 正在呼叫本地模型 {self.model_name}...")
        
        payload = {
            "model": self.model_name,
            "prompt": final_prompt,
            "stream": False,  # 关闭流式输出，直接获取完整结果
            "options": {
                "temperature": 0.3  # 低温度（0-0.5），让法律回答更稳定、严谨
            }
        }

        try:
            response = requests.post(
                self.url, 
                json=payload,
                timeout=120  # 本地模型推理可能需要几秒到几十秒，设置长一点的超时时间
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                return f"[错误] 调用本地模型失败: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"[错误] 连接失败: {str(e)}。请确保 Ollama 服务已启动 (ollama serve)"