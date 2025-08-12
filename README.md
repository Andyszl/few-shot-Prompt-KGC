# few-shot Prompt KGC
Few-shot Prompt for Knowledge Graph Completion

> 在少样本（K-shot）场景下，利用提示工程（Prompting）与可选的检索增强，对知识图谱进行缺失三元组补全与推理。

**Last updated:** 2025-08-12

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

### 方式 A：Notebook（推荐快速上手）

1. 打开 `数据抽样.ipynb`：按关系采样 K-shot 支持集与查询集；
2. 打开 `模型测试.ipynb`：配置 `config.yaml` 路径，运行 few-shot 推理与指标统计；
3. 打开 `测试代码.ipynb`：用于代码片段验证与可视化。

> 运行后在 `outputs/` 下生成日志、图表与评测 CSV。([GitHub][1])

### 方式 B：Python 脚本（如果你已提供入口）

> 若你后续添加了命令行脚本，可参考如下约定命令（示例）：

```bash
# Few-shot + 检索增强
python -m knowledge_graph_completion.run \
  --config config.yaml \
  --task tail-prediction \
  --use-retrieval \
  --out ./outputs/run_$(date +%F)

# 传统 KGE 基线
python -m torchKGE.train \
  --data ./data/your_dataset \
  --model transe --dim 400 --epochs 100 \
  --out ./outputs/transe_baseline
```

## 📈 Metrics

* **MRR**（Mean Reciprocal Rank）
* **Hits\@k**（k ∈ {1,3,10}）
* 可选：Accuracy（闭集候选时）

建议：输出总体 + 按关系频次分桶（长尾对比）+ 案例可视化（含提示与检索证据）。

## 🔁 Reproducibility Checklist

* 固定 `seed / k_shots / candidate_pool_size / 评测协议（是否过滤已知三元组）`
* 公布 LLM 名称/版本、`temperature / max_tokens`
* 贴出完整提示模板与示例采样策略
* 标注检索 encoder/index 具体版本与参数
* 版本化 `outputs/` 与 `config.yaml`

## 🤝 Contributing

欢迎 PR / Issue：

* 新的数据处理脚本与评测协议
* 更多提示模板与采样策略
* 新的检索器、消歧与证据路由方法
* KGE/LLM 混合推理改进

## 📄 License

本仓库如未附带 LICENSE 文件，默认保留所有权利；建议添加 `MIT` 或 `Apache-2.0` 许可证，便于协作与复现。

## 🙏 Acknowledgements

感谢社区在少样本 KGC、提示学习与图检索方向的公开工作与启发。

````

---

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
