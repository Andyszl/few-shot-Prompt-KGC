# few-shot Prompt KGC
Few-shot Prompt for Knowledge Graph Completion

> 在少样本（K-shot）场景下，利用提示工程（Prompting）进行LoRA微调，对知识图谱进行缺失三元组补全与推理。

**Last updated:** 2025-09-23

## ✨ 1、Highlights
- **Few-shot 提示**：用极少示例驱动关系/实体推断。
- **可选 Graph-RAG 检索增强**：从图或文本侧检索证据，拼接到提示中。
- **对比传统 KGE**：与 `torchKGE/` 下嵌入式方法做指标对照。
- **Notebook 上手**：提供“数据抽样 / 模型测试 / 测试代码”3 个示例笔记本。

## 📦 2、Repository Structure
```text
few-shot-Prompt-KGC/
├─ stgrapgRAG/                    # b本研究的目标数据（Graph-RAG，结果）
├─ training /                    # Few-shot 任务/提示相关代码
├─ torchKGE/                    # 传统嵌入式 KGE 基线
├─ 数据抽样.ipynb
├─ 模型测试.ipynb
└─ 测试代码.ipynb
````

> 以上结构依据仓库当前可见内容，如有新增/重命名以实际为准。
> Source: repo root file list. ([GitHub][1])

## 🛠 3、Environment

* Python 3.9+（建议 3.10）
* 见 `requirements.txt`（可按需删减）
* 可使用 GPU（PyTorch CUDA）

Quick start:

```bash
conda create -n promptkgc python=3.10 -y
conda activate promptkgc
pip install -r requirements.txt
```


## 📚 4、Data Layout

将数据放在 `./data/{your_dataset}/`：

```
data/
  your_dataset/
    train.csv       # 每行: head \t relation \t tail
    test.csv
```

用 `数据抽样.ipynb` 生成 few-shot 切分（支持集/查询集），输出到 `data/your_dataset/fewshot/`，供 few-shot 提示推理与评测使用。([GitHub][1])

## 🚀 5、How to Run
### 5.1 方法论证
使用开源数据集，在本方法和基线模型上进行对比，验证方法的有效性

#### 开源数据验证 A：Notebook（推荐快速上手）

1. 打开 `数据抽样.ipynb`：对OpenBG500开源数据进行抽样，选择低频的数据共9894条数据，拆分为训练集和测试集，保存在/training/data/OpenBG500下面；
   - 训练集：train.csv，共8707条数据
   - 测试集：test.csv，共1187条数据
2. 打开 `模型测试.ipynb`：使用抽样数据进行few-shot 有效性验证，并进行LoRA训练，使用样本数据跑通整体流程；


#### 开源数据验证 B：Python 脚本（全量数据训练及验证）

> peft_train.py

```bash
# Few-shot + LoRA 训练
nohup python peft_train.py
```
>训练后使用最优评估结果来进行评估和测试；本项目最优的模型为./training/outputs/checkpoint-3450，为了充分训练，一开始训练的epoch为1000，最新代码已经启用了早停机制；
>评估结果，./outputs/train.log


#### 开源数据验证 C：torchkge基线模型验证（全量数据训练及验证）
对抽样数据在基线模型，transE、DistMult、CompeExModel三个模型上进行对比测试
测试代码：./torchKGE/论文数据.ipynb

### 5.2 论文数据构建
论文数据使用《全国生态状况调查评估技术规范—生态系统服务功能评估》、《全国生态状况调查评估技术规范—生态系统质量评估》构建初始知识图谱

1. 使用graphRAG创建初始图谱，生成的文件见./stgrapgRAG/output/
2. 训练，将graphRAG创建的图谱构建三元组数据集，得到训练样本；根据训练的代码（peft_train.py）对数据进行训练；得到的训练结果见./training/outputs/peft_d0626，训练过程，./training/outputs/peft_d0626/train.log
3. 知识补全，将训练好的模型对初始谱图进行补全，代码./complete_isolated_nodes.py，得到补全的知识图谱文件，./new_edges.csv
4. 将原始图谱和补全内容合并得到新的图谱，代码参看：./graphrag/query.ipynb


## 📈 6、Metrics

* **MRR**（Mean Reciprocal Rank）
* **Hits\@k**（k ∈ {1,3,10}）


建议：输出总体 + 按关系频次分桶（长尾对比）+ 案例可视化（含提示与检索证据）。



备注：./training/models下面用到的两个模型文件，可以在modelscope上进行下载相应的目录下进行调试
