import random
import torch
from torch.utils.data import Dataset

class PromptDataset(Dataset):
    def __init__(self, df, rel2triples, tokenizer, k, max_length=512):
        self.records = df.to_dict('records')
        self.rel2triples = rel2triples
        self.tokenizer = tokenizer
        self.k = k
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        h, r, t = rec['head'], rec['relation'], rec['tail']
        # dynamic few-shot sampling
        examples = random.sample(self.rel2triples[r], min(self.k, len(self.rel2triples[r])))
        examples = [ex for ex in examples if not (ex[0] == h and ex[2] == t)]
        # build prompt
        prompt_lines = [f"示例：{hh} 的 {r} 是 {tt}。" for hh, _, tt in examples]
        prompt = "".join(prompt_lines) + f"请问：{h} 的 {r} 是"
        # encode prompt and answer
        ans_text = t + self.tokenizer.eos_token
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        ans_ids = self.tokenizer.encode(ans_text, add_special_tokens=False)
        # concatenate and truncate
        input_ids_list = (prompt_ids + ans_ids)[:self.max_length]
        # mask prompt portion
        labels_list = [-100] * len(prompt_ids) + ans_ids
        labels_list = labels_list[:self.max_length]
        # pad sequences to max_length
        pad_len = self.max_length - len(input_ids_list)
        input_ids_list += [self.tokenizer.pad_token_id] * pad_len
        labels_list += [-100] * pad_len
        attention_mask = [1] * (self.max_length - pad_len) + [0] * pad_len
        # convert to tensors
        return {
            'input_ids': torch.tensor(input_ids_list, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels_list, dtype=torch.long)
             }


def calculate_metrics(ranks):
    """
    ranks: list of int or None
    返回 (hits@1, mrr)
    """
    total = len(ranks)
    hits1 = sum(1 for r in ranks if r == 1) / total
    # MRR: None->0
    mrr = sum((1/r if r else 0) for r in ranks) / total
    return hits1, mrr
