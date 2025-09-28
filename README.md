# few-shot Prompt KGC
Few-shot Prompt for Knowledge Graph Completion

> 在少样本（K-shot）场景下，利用提示工程（Prompting）进行LoRA微调，对知识图谱进行缺失三元组补全与推理。

**Last updated:** 2025-09-23

## ✨ Highlights
- **Few-shot 提示**：用极少示例驱动关系/实体推断。
- **可选 Graph-RAG 检索增强**：从图或文本侧检索证据，拼接到提示中。
- **对比传统 KGE**：与 `torchKGE/` 下嵌入式方法做指标对照。
- **Notebook 上手**：提供“数据抽样 / 模型测试 / 测试代码”3 个示例笔记本。

## 📦 Repository Structure
```text
few-shot-Prompt-KGC/
├─ graphRAG/                    # 检索与上下文整理（Graph-RAG 风格，可选）
├─ knowledge_graph_completion/  # Few-shot 任务/提示相关代码
├─ torchKGE/                    # 传统嵌入式 KGE 基线
├─ outputs/                     # 结果与可视化输出
├─ 数据抽样.ipynb
├─ 模型测试.ipynb
└─ 测试代码.ipynb
````

> 以上结构依据仓库当前可见内容，如有新增/重命名以实际为准。
> Source: repo root file list. ([GitHub][1])

## 🛠 Environment

* Python 3.9+（建议 3.10）
* 见 `requirements.txt`（可按需删减）
* 可使用 GPU（PyTorch CUDA）

Quick start:

```bash
conda create -n promptkgc python=3.10 -y
conda activate promptkgc
pip install -r requirements.txt
```

## ⚙️ Configuration

在项目根目录新建/修改 `config.yaml`（示例）：

```yaml
seed: 42
device: "cuda:0"     # 或 "cpu"

task: "tail-prediction"   # 支持: tail-prediction / head-prediction / relation-prediction
k_shots: 5
candidate_pool_size: 100  # 闭集候选上限

data:
  root: "./data"
  dataset: "your_dataset"  # 你的数据集目录名
  fewshot_dir: "./data/your_dataset/fewshot"

retrieval:
  enable: true
  topk: 8
  encoder: "sentence-transformers/all-MiniLM-L6-v2"
  index: "faiss"

llm:
  provider: "openai"      # 或 "dashscope"/"azure"/"qianfan" 等
  model: "gpt-4o"         # 替换为你真实使用的模型
  temperature: 0.0
  max_tokens: 512
  api_key: "${OPENAI_API_KEY}"   # 从环境变量读取

prompt:
  template: "./knowledge_graph_completion/prompts/fewshot_tail.txt"
  place_examples: "before_query"  # few-shot 示例放置策略

paths:
  out_dir: "./outputs"
```

## 📚 Data Layout

将数据放在 `./data/{your_dataset}/`：

```
data/
  your_dataset/
    train.txt       # 每行: head \t relation \t tail
    valid.txt
    test.txt
    entities.txt
    relations.txt
```

用 `数据抽样.ipynb` 生成 few-shot 切分（支持集/查询集），输出到 `data/your_dataset/fewshot/`，供 few-shot 提示推理与评测使用。([GitHub][1])

## 🚀 How to Run

### 开源数据验证 A：Notebook（推荐快速上手）

1. 打开 `数据抽样.ipynb`：对OpenBG500开源数据进行抽样，选择低频的数据共9894条数据，拆分为训练集和测试集，保存在/knowledge_graph_completion/data/OpenBG500下面；
   - 训练集：train.csv，共8707条数据
   - 测试集：test.csv，共1187条数据
2. 打开 `模型测试.ipynb`：使用抽样数据进行few-shot 有效性验证，并进行LoRA训练，使用样本数据跑通整体流程；


### 开源数据验证 B：Python 脚本（全量数据训练及验证）

> peft_train.py

```bash
# Few-shot + LoRA 训练
nohup python peft_train.py
```
>训练后使用最优评估结果来进行评估和测试；本项目最优的模型为./outputs/checkpoint-3450，为了充分训练，一开始训练的epoch为1000，最新代码已经启用了早停机制；
>评估结果，./outputs/train.log


### 开源数据验证 C：torchkge基线模型验证（全量数据训练及验证）
对抽样数据在基线模型，transE、DistMult、CompeExModel三个模型上进行对比测试
测试代码：./torchKGE/论文数据.ipynb


## 📈 Metrics

* **MRR**（Mean Reciprocal Rank）
* **Hits\@k**（k ∈ {1,3,10}）


建议：输出总体 + 按关系频次分桶（长尾对比）+ 案例可视化（含提示与检索证据）。



### requirements.txt

```txt
# Core
numpy>=1.24
pandas>=2.0
scipy>=1.11
scikit-learn>=1.3
tqdm>=4.66

# Deep learning
torch>=2.1
torchvision>=0.16
torchaudio>=2.1

# LLM / prompts (任选其一或多家，根据你实际调用改动)
openai>=1.30
httpx>=0.27
transformers>=4.42

# Retrieval / text encoders (如启用 Graph-RAG/向量索引)
sentence-transformers>=2.5
faiss-cpu>=1.7.4
networkx>=3.2

# Utils
pyyaml>=6.0.1
matplotlib>=3.8
ipykernel>=6.29
jupyter>=1.0
````

> 如果你的 `torchKGE/` 或 `knowledge_graph_completion/` 内已固定某些库版本，请把上述版本号改成与你代码一致的范围。

---

### config.yaml（示例）

```yaml
seed: 42
device: "cuda:0"     # 或 "cpu"

task: "tail-prediction"   # tail-prediction / head-prediction / relation-prediction
k_shots: 5
candidate_pool_size: 100

data:
  root: "./data"
  dataset: "your_dataset"
  fewshot_dir: "./data/your_dataset/fewshot"

retrieval:
  enable: true
  topk: 8
  encoder: "sentence-transformers/all-MiniLM-L6-v2"
  index: "faiss"

llm:
  provider: "openai"
  model: "gpt-4o"
  temperature: 0.0
  max_tokens: 512
  api_key: "${OPENAI_API_KEY}"   # 建议用环境变量注入

prompt:
  template: "./knowledge_graph_completion/prompts/fewshot_tail.txt"
  place_examples: "before_query"

paths:
  out_dir: "./outputs"
```
