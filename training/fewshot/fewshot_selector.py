import pandas as pd
import random

def get_fewshot_examples(relation, data, num_examples=3):
    """
    从 data 中抽取 num_examples 条特定 relation 的示例。
    支持：
      - pandas.DataFrame （需含 'relation' 列）
      - list of dict （每个 dict 含 'relation' 键）
    返回 list of dict，每个 {'head','relation','tail'}。
    """
    examples = []
    if isinstance(data, pd.DataFrame):
        sub = data[data['relation'] == relation]
        if len(sub) == 0:
            return []
        sub = sub.sample(n=min(num_examples,len(sub)), random_state=42)
        examples = sub[['head','relation','tail']].to_dict('records')
    else:
        # list of dict
        cand = [d for d in data if d.get('relation') == relation]
        if len(cand) == 0:
            return []
        examples = random.sample(cand, k=min(num_examples,len(cand)))
    return examples
