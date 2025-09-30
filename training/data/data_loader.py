import os
import pandas as pd

def load_fb15k237(data_dir):
    """
    从 KGraph/FB15k-237 格式加载数据，并将 Freebase MID 映射为可读名称。

    参数:
        data_dir (str): 存放 train.txt、valid.txt、test.txt 
                        以及 FB15k_mid2name.txt 的目录
    返回:
        train_df, valid_df, test_df (pd.DataFrame):
          - 原始列 ['head','relation','tail']
          - 映射后列 ['head_name','tail_name']
    """
    # 1. 读取三元组文件
    files = ["train.txt", "valid.txt", "test.txt"]
    dfs = []
    for fname in files:
        path = os.path.join(data_dir, fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"未找到 {path}")
        df = pd.read_csv(path, sep='\t', header=None,
                         names=['head','relation','tail'], dtype=str)
        dfs.append(df)
    train_df, valid_df, test_df = dfs

    # 2. 读取 MID→名称 映射文件
    map_file = os.path.join(data_dir, "FB15k_mid2name.txt")
    if not os.path.isfile(map_file):
        raise FileNotFoundError(f"未找到映射文件 {map_file}，请确认已从 KGraph/FB15k-237 下载 FB15k_mid2name.txt")  # :contentReference[oaicite:1]{index=1}

    # 3. 构建映射字典
    mid2name = {}
    with open(map_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t', 1)
            if len(parts) == 2:
                mid, name = parts
                mid2name[mid] = name

    # 4. 应用映射，添加可读名称列
    for df in (train_df, valid_df, test_df):
        df['head_name'] = df['head'].map(lambda m: mid2name.get(m, m))
        df['tail_name'] = df['tail'].map(lambda m: mid2name.get(m, m))

    return train_df, valid_df, test_df


def load_openbg500(data_dir: str):
    """
    读取 OpenBG500 数据集。
    
    参数:
        data_dir: 包含以下文件的目录
            ├── OpenBG500_train.tsv
            ├── OpenBG500_dev.tsv
            ├── OpenBG500_test.tsv
            ├── OpenBG500_entity2text.tsv
            └── OpenBG500_relation2text.tsv

    返回:
        train_df, valid_df, test_df, ent2text, rel2text
        - train_df, valid_df: DataFrame，含列
            ['head', 'relation', 'tail',
             'head_name', 'relation_name', 'tail_name']
        - test_df: DataFrame，含列
            ['head', 'relation',  # no tail in test file
             'head_name', 'relation_name']
        - ent2text: dict 从实体 ID 映射到中文描述
        - rel2text: dict 从关系 ID 映射到中文描述
    """
    # 1. 读实体与关系的文本映射
    ent2txt_path = os.path.join(data_dir, "OpenBG500_entity2text.tsv")
    rel2txt_path = os.path.join(data_dir, "OpenBG500_relation2text.tsv")
    ent2text_df = pd.read_csv(ent2txt_path, sep="\t", header=None, names=["entity", "text"])
    rel2text_df = pd.read_csv(rel2txt_path, sep="\t", header=None, names=["relation", "text"])
    # 转成 dict，遇到缺失就保留原 ID
    ent2text = dict(zip(ent2text_df["entity"], ent2text_df["text"]))
    rel2text = dict(zip(rel2text_df["relation"], rel2text_df["text"]))

    # 2. 定义一个小函数：ID -> 描述
    def map_ent(eid):
        return ent2text.get(eid, eid)
    def map_rel(rid):
        return rel2text.get(rid, rid)

    # 3. 读取三元组
    def _load_triplet_file(path, has_tail=True):
        if has_tail:
            df = pd.read_csv(path, sep="\t", header=None, names=["head","relation","tail"])
            # 添加中文描述列
            df["head_name"]     = df["head"].map(map_ent)
            df["relation_name"] = df["relation"].map(map_rel)
            df["tail_name"]     = df["tail"].map(map_ent)
        else:
            # test 文件只有 head, relation
            df = pd.read_csv(path, sep="\t", header=None, names=["head","relation"])
            df["head_name"]     = df["head"].map(map_ent)
            df["relation_name"] = df["relation"].map(map_rel)
        return df

    train_path = os.path.join(data_dir, "OpenBG500_train.tsv")
    valid_path = os.path.join(data_dir, "OpenBG500_dev.tsv")
    test_path  = os.path.join(data_dir, "OpenBG500_test.tsv")

    train_df = _load_triplet_file(train_path, has_tail=True)
    valid_df = _load_triplet_file(valid_path, has_tail=True)
    test_df  = _load_triplet_file(test_path, has_tail=False)

    return train_df, valid_df, test_df, ent2text, rel2text
