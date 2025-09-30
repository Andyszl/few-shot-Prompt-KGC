import os
import argparse
import logging
import sys
import pandas as pd
import torch
import swanlab
from swanlab.integration.transformers import SwanLabCallback
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, TaskType
from infer import evaluate_model  

# PromptDataset 保持不变
class PromptDataset(Dataset):
    def __init__(self, records, tokenizer, fewshot_k=3, max_input_length=512, max_target_length=128):
        self.records = records
        self.tokenizer = tokenizer
        self.fewshot_k = fewshot_k
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        head, relation, tail = rec['head'], rec['relation'], rec['tail']
        # few-shot 示例
        cands = [ex for ex in self.records if ex['relation']==relation and ex != rec]
        examples = cands[:self.fewshot_k]
        prompt = ""
        for ex in examples:
            prompt += f"示例: 头实体: {ex['head']}；关系: {ex['relation']}；尾实体: {ex['tail']}\n"
        prompt += f"现在，头实体: {head}；关系: {relation}；尾实体: "
        target = " " + tail

        # 编码
        enc_prompt = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_input_length,
            return_attention_mask=True
        )
        with self.tokenizer.as_target_tokenizer():
            enc_target = self.tokenizer(
                target,
                truncation=True,
                max_length=self.max_target_length
            )
        input_ids = enc_prompt["input_ids"] + enc_target["input_ids"]
        attention_mask = enc_prompt["attention_mask"] + [1] * len(enc_target["input_ids"])
        labels = [-100] * len(enc_prompt["input_ids"]) + enc_target["input_ids"]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

# 自定义 Trainer，实现每 eval_steps 调用 evaluate_model
class KGTrainer(Trainer):
    def __init__(self, train_data, valid_data, fewshot_k, top_k, max_new_tokens, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_data = train_data
        self.valid_data = valid_data
        self.fewshot_k = fewshot_k
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens

    def evaluate(self, eval_dataset=None, **kwargs):
        # 首先跑原生 eval（计算 loss 等）
        base_metrics = super().evaluate(eval_dataset, **kwargs)
        # 再跑 Hits/MRR
        hits1, hits3, hits10, mrr = evaluate_model(
            model=self.model,
            tokenizer=self.tokenizer,
            test_data=self.valid_data,
            train_data=self.train_data,
            fewshot_k=self.fewshot_k,
            top_k=self.top_k,
            max_new_tokens=self.max_new_tokens
        )
        info = f"Eval Metrics — Hits@1: {hits1:.4f}, Hits@3: {hits3:.4f}, Hits@10: {hits10:.4f}, MRR: {mrr:.4f}"
        print(info)
        logging.getLogger(__name__).info(info)
        base_metrics["eval_hits1"] = hits1
        base_metrics["eval_hits3"] = hits3
        base_metrics["eval_hits10"] = hits10
        base_metrics["eval_mrr"] = mrr
        return base_metrics

def data_collator(batch):
    pad_id = batch[0]["input_ids"][0].new_tensor([0]).new_full((), batch[0]["input_ids"].size(0), dtype=torch.long)
    # 其实直接按之前实现即可
    input_ids = [ex["input_ids"] for ex in batch]
    attention_masks = [ex["attention_mask"] for ex in batch]
    labels = [ex["labels"] for ex in batch]
    max_len = max(x.size(0) for x in input_ids)
    padded_input_ids = torch.stack([
        torch.cat([seq, seq.new_full((max_len-seq.size(0),), pad_id)], dim=0)
        for seq in input_ids
    ])
    padded_attention = torch.stack([
        torch.cat([seq, seq.new_full((max_len-seq.size(0),), 0)], dim=0)
        for seq in attention_masks
    ])
    padded_labels = torch.stack([
        torch.cat([seq, seq.new_full((max_len-seq.size(0),), -100)], dim=0)
        for seq in labels
    ])
    return {"input_ids": padded_input_ids,
            "attention_mask": padded_attention,
            "labels": padded_labels}

def main():
    parser = argparse.ArgumentParser(description="LoRA 微调 Qwen3-0.6B (知识图谱补全)")
    parser.add_argument("--model_name", type=str, default="./models/Qwen3-0.6B", help="预训练模型名称或路径")
    parser.add_argument("--data_dir", type=str, default="./data/OpenBG500/", help="数据目录，包含 train.csv 和 test.csv")
    parser.add_argument("--output_dir", type=str, default="./outputs/peft_d0627/", help="保存模型和日志的目录")
    parser.add_argument("--num_epochs", type=int, default=100, help="训练 epoch 数")
    parser.add_argument("--batch_size", type=int, default=8, help="训练 batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--fewshot_k", type=int, default=3, help="few-shot 示例数")
    parser.add_argument("--top_k", type=int, default=10, help="beam 搜索大小")
    parser.add_argument("--max_new_tokens", type=int, default=10, help="生成时每条序列最大新 tokens 数量")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        handlers=[
                            logging.FileHandler(os.path.join(args.output_dir, "train.log")),
                            logging.StreamHandler(sys.stdout)
                        ])

    # 1. 读取数据
    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    valid_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    train_records = train_df.to_dict(orient="records")
    valid_records = valid_df.to_dict(orient="records")

    # 2. 加载 tokenizer & 模型 + LoRA
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_config)
    logging.info("Model and LoRA loaded.")

    # 3. 构建 Dataset
    train_dataset = PromptDataset(train_records, tokenizer, fewshot_k=args.fewshot_k)
    valid_dataset = PromptDataset(valid_records, tokenizer, fewshot_k=args.fewshot_k)

    # 4. Trainer 配置
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type='cosine',
        # warmup_ratio=0.01,
        weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        save_total_limit = 3,
        greater_is_better=False,
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to="none"
    )
    # 登录 wandb
    swanlab.login(api_key="W6lCirsyPCHlASdYtFE1F", save=True)
    swanlab_callback = SwanLabCallback(project="kgc0923", experiment_name="lora0923")
    
    
    trainer = KGTrainer(
        train_data=train_df,
        valid_data=valid_df,
        fewshot_k=args.fewshot_k,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3),swanlab_callback],
        data_collator=data_collator
    )

    # 5. 训练
    trainer.train()
    logging.info("Training completed.")

    # 6. 保存 LoRA Adapter
    peft_dir = os.path.join(args.output_dir, "best_adapter")
    model.save_pretrained(peft_dir)
    logging.info(f"Best adapter saved to {peft_dir}")

if __name__ == "__main__":
    main()
