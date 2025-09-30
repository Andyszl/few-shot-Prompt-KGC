import pandas as pd
from fewshot.fewshot_selector import get_fewshot_examples

def build_fewshot_prompt(head, relation, data, num_examples=3):
    """
    构建 Few-Shot 提示字符串：
    - head, relation: 查询
    - data: DataFrame 或 list of dict，用于挑示例
    - num_examples: 示例数量
    返回完整的 prompt 文本。
    """
    exs = get_fewshot_examples(relation, data, num_examples)
    prompt = ""
    for ex in exs:
        prompt += f"示例: 头实体: {ex['head']}；关系: {ex['relation']}；尾实体: {ex['tail']}\n"
    prompt += f"现在，头实体: {head}；关系: {relation}；尾实体: "
    return prompt
