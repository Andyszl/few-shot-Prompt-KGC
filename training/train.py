import os
import random
import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from prompts.prompt_builder import build_fewshot_prompt
from fewshot.fewshot_selector import get_fewshot_examples
from infer import evaluate_model


def train_model(model: PreTrainedModel,
                tokenizer: PreTrainedTokenizer,
                train_data,
                valid_data=None,
                epochs=3,
                batch_size=16,
                learning_rate=5e-5,
                output_dir="outputs",
                num_examples=3,
                top_k=10):
    """
    微调模型并保存验证集上最优权重。

    train_data, valid_data: pandas.DataFrame 或 list of dict，
      DataFrame需含 ['head','head','relation','tail','tail'] 列。
    output_dir: 保存最佳模型的目录，会在 {output_dir}/best_model 中生成模型。
    num_examples: Few-Shot 示例数
    top_k: 验证时 beam search 大小
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # 将 DataFrame 转为 list of dict，优先使用 head/tail
    def to_records(data):
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            records = []
            for _, row in data.iterrows():
                records.append({
                    'head': row.get('head', row['head']),
                    'relation': row['relation'],
                    'tail': row.get('tail', row['tail'])
                })
            return records
        return data

    train_records = to_records(train_data)
    valid_records = to_records(valid_data) if valid_data is not None else None

    best_mrr = -1.0
    best_dir = os.path.join(output_dir, "best_model")
    os.makedirs(best_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batch_count = 0

        random.shuffle(train_records)
        for i in tqdm(range(0, len(train_records), batch_size), desc=f"Epoch {epoch}"):
            batch = train_records[i:i+batch_size]
            optimizer.zero_grad()

            batch_loss = 0.0
            valid_count = 0
            # 对每个样本分别计算 loss 并累加
            for sample in batch:
                head = sample['head']
                relation = sample['relation']
                tail = sample['tail']
                # 构建 Few-Shot 提示
                prompt = build_fewshot_prompt(head, relation, train_records, num_examples)
                full_text = prompt + tail
                # 编码
                encoding = tokenizer(full_text, return_tensors='pt', truncation=True).to(device)
                input_ids = encoding.input_ids
                attention_mask = encoding.attention_mask
                # 获取 prompt 长度
                prompt_len = tokenizer(prompt, return_tensors='pt').input_ids.shape[1]
                # 构造 labels，屏蔽提示部分
                labels = input_ids.clone()
                labels[:, :prompt_len] = -100
                # 前向计算
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                if torch.isnan(loss):
                    continue
                batch_loss += loss
                valid_count += 1

            if valid_count == 0:
                continue
            # 平均并反向
            batch_loss = batch_loss / valid_count
            batch_loss.backward()
            optimizer.step()
            batch_loss_item = batch_loss.item()

            total_loss += batch_loss_item
            batch_count += 1
            # 清理显存
            torch.cuda.empty_cache()

        avg_loss = total_loss / max(batch_count, 1)
        print(f"[Epoch {epoch}] 平均 Loss: {avg_loss:.4f}")

        # 验证并保存最佳模型
        if valid_records is not None:
            model.eval()
            hits1, hits3, hits10, mrr = evaluate_model(
                model, tokenizer,
                valid_data, train_data,
                num_examples=num_examples,
                max_new_tokens=top_k
            )
            print(f"Hits@1: {hits1:.4f}, Hits@3: {hits3:.4f}, Hits@10: {hits10:.4f}, MRR: {mrr:.4f}")
            if mrr > best_mrr:
                best_mrr = mrr
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                print(f"→ 保存最佳模型，MRR: {best_mrr:.4f}")
            model.train()

    print("训练完成，最佳模型保存在:", best_dir)
