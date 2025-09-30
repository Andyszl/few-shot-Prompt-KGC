import warnings
# 忽略 transformers 生成配置的用户警告
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.generation.configuration_utils")
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

import torch
from tqdm import tqdm
import pandas as pd
from transformers import PreTrainedModel, PreTrainedTokenizer


def parse_model_output(seqs: torch.Tensor, tokenizer: PreTrainedTokenizer, prompt_len: int):
    """
    将 model.generate 返回的多条序列 seqs 转成一个候选 tail 列表。
    
    参数:
        seqs: torch.Tensor of shape (num_return_sequences, seq_len)，来自 model.generate
        tokenizer: 对应的分词器
        prompt_len: prompt 部分的 token 数，用于截断
    
    返回:
        List[str]: 按生成顺序排列的候选 tail
    """
    cands = []
    for seq in seqs:
        new_tokens = seq[prompt_len:].cpu()
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        first_line = raw.splitlines()[0].strip() if raw else ""
        cands.append(first_line)
    return cands


def evaluate_model(model: PreTrainedModel,
                   tokenizer: PreTrainedTokenizer,
                   test_data,
                   train_data,
                   fewshot_k: int = 3,
                   top_k: int = 10,
                   max_new_tokens: int = 10):
    """
    在 test_data 上评估模型，返回 (hits1, hits3, hits10, mrr)。

    - model, tokenizer: 已加载并 eval() 的 PyTorch 模型与分词器
    - test_data: pandas.DataFrame 或 list of dict，需含 'head','relation','tail'
    - train_data: pandas.DataFrame 或 list of dict，用于抽 Few-Shot 示例
    - fewshot_k: Few-Shot 示例数
    - top_k: beam search 候选数
    - max_new_tokens: 每个序列生成的新 tokens 上限
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # 统一 DataFrame
    if not isinstance(test_data, pd.DataFrame):
        test_data = pd.DataFrame(test_data)
    if not isinstance(train_data, pd.DataFrame):
        train_data = pd.DataFrame(train_data)

    hits1_count = 0
    hits3_count = 0
    hits10_count = 0
    reciprocal_ranks = []

    for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Evaluating"):
        head       = row['head']
        relation   = row['relation']
        true_tail  = row['tail']

        # Few-Shot 示例
        fs = train_data[train_data['relation'] == relation].head(fewshot_k)
        examples = [
            (ex.get('head', ex['head']), relation, ex.get('tail', ex['tail']))
            for _, ex in fs.iterrows()
        ]

        # 构建 prompt
        prompt_lines = [f"{h} - {r} -> {t}" for h, r, t in examples]
        prompt = "\n".join(prompt_lines) + f"\n{head} - {relation} ->"

        # 编码 & 生成
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True).to(device)
        prompt_len = inputs.input_ids.shape[1]
        outs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=top_k,
            num_return_sequences=top_k,
            do_sample=False
        )

        # 解码候选
        cands = parse_model_output(outs, tokenizer, prompt_len)

        # 计算排名
        if cands and cands[0] == true_tail:
            hits1_count += 1
        if true_tail in cands[:3]:
            hits3_count += 1
        if true_tail in cands[:10]:
            hits10_count += 1

        if true_tail in cands:
            rank = cands.index(true_tail) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    total = len(test_data)
    hits1 = hits1_count / total
    hits3 = hits3_count / total
    hits10 = hits10_count / total
    mrr   = sum(reciprocal_ranks) / total

    return hits1, hits3, hits10, mrr


def evaluate_batch_model(model: PreTrainedModel,
                         tokenizer: PreTrainedTokenizer,
                         test_data,
                         train_data,
                         fewshot_k: int = 3,
                         top_k: int = 10,
                         max_new_tokens: int = 10,
                         batch_size: int = 8):
    """
    批量评估模型，返回 (hits1, hits3, hits10, mrr)。
    参数同 evaluate_model，额外 batch_size 控制批次大小。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # 确保 tokenizer 有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # 统一 DataFrame
    if not isinstance(test_data, pd.DataFrame):
        test_data = pd.DataFrame(test_data)
    if not isinstance(train_data, pd.DataFrame):
        train_data = pd.DataFrame(train_data)

    # 构建 prompts 与真值
    prompts, true_tails, raw_lens = [], [], []
    for _, row in test_data.iterrows():
        h, r, t_true = row['head'], row['relation'], row['tail']
        fs = train_data[train_data['relation'] == r].head(fewshot_k)
        examples = [(ex['head'], r, ex['tail']) for _, ex in fs.iterrows()]
        lines = [f"{hh} - {rr} -> {tt}" for hh, rr, tt in examples]
        prompt = "\n".join(lines) + f"\n{h} - {r} ->"
        prompts.append(prompt)
        true_tails.append(t_true)
        raw_lens.append(len(tokenizer(prompt, truncation=True)['input_ids']))

    hits1_count = 0
    hits3_count = 0
    hits10_count = 0
    reciprocal_ranks = []
    total = len(prompts)

    # 分批处理
    for i in tqdm(range(0, total, batch_size), desc="Batched Evaluating"):
        batch_prompts    = prompts[i : i + batch_size]
        batch_true_tails = true_tails[i : i + batch_size]
        batch_raw_lens   = raw_lens[i : i + batch_size]

        # 批量编码 & 生成
        inputs = tokenizer(batch_prompts,
                           return_tensors='pt',
                           truncation=True,
                           padding=True).to(device)
        outs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=top_k,
            num_return_sequences=top_k,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False
        )  # (batch_size*top_k, seq_len)

        # 拆分每个样本的 top_k 输出
        outs_batches = outs.split(top_k, dim=0)

        # 逐样本统计
        for idx, seqs in enumerate(outs_batches):
            prompt_len = batch_raw_lens[idx]
            cands = parse_model_output(seqs, tokenizer, prompt_len)
            true_tail = batch_true_tails[idx]

            if cands and cands[0] == true_tail:
                hits1_count += 1
            if true_tail in cands[:3]:
                hits3_count += 1
            if true_tail in cands[:10]:
                hits10_count += 1

            if true_tail in cands:
                rank = cands.index(true_tail) + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

    hits1 = hits1_count / total
    hits3 = hits3_count / total
    hits10 = hits10_count / total
    mrr   = sum(reciprocal_ranks) / total

    return hits1, hits3, hits10, mrr
