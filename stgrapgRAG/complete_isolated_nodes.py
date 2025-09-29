# %%
import pandas as pd
import networkx as nx
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# %%
# === Configuration ===
EMBED_MODEL     = "./models/paraphrase-multilingual-mpnet-base-v2/"
SIM_THRESHOLD   = 0.75
TOP_K           = 10
MAX_LINKS       = 2

BASE_MODEL_PATH = "./models/Qwen3-0.6B"
ADAPTER_PATH    = "./outputs/peft_d0626/best_adapter"
FEWSHOT_K       = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EDGES_PATH      = "./output/create_final_relationships.parquet"
NODES_PATH      = "./output/create_final_nodes.parquet"
COMM_PATH       = "./output/create_final_communities.parquet"

# %%
def load_data():
    edges_df = pd.read_parquet(EDGES_PATH)
    nodes_df = pd.read_parquet(NODES_PATH)
    comm_df  = pd.read_parquet(COMM_PATH)
    return edges_df, nodes_df, comm_df

# %%
def build_graph(edges_df):
    G = nx.Graph()
    nodes = set(edges_df["source"].astype(str)) | set(edges_df["target"].astype(str))
    G.add_nodes_from(nodes)
    for _, row in edges_df.iterrows():
        src, tgt, rel = map(str, (row["source"], row["target"], row["description"]))
        G.add_edge(src, tgt, relation=rel)
    return G

# %%
def get_orphan_nodes(nodes_df):
    return nodes_df.loc[nodes_df["community"] == -1, "title"].astype(str).tolist()

# %%
def compute_embeddings(all_nodes):
    embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    embs = embedder.encode(
        all_nodes,
        convert_to_tensor=True,
        normalize_embeddings=True
    )
    return dict(zip(all_nodes, embs))

# %%
def load_fewshot_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, is_trainable=False)
    return tokenizer, model.eval().to(DEVICE)

# %%
def build_examples_by_relation(edges_df):
    examples = {}
    for _, row in edges_df.iterrows():
        rel = str(row["description"])
        subj = str(row["source"])
        obj  = str(row["target"])
        examples.setdefault(rel, []).append((subj, rel, obj))
    return examples

# %%
def verify_relation(head, rel, tail, examples, tokenizer, model):
    prompt = "".join(
        f"示例: 头实体: {h}；关系: {r}；尾实体: {t}\n"
        for h, r, t in examples[:FEWSHOT_K]
    )
    prompt += f"现在，头实体: {head}；关系: {rel}；尾实体:"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outs = model.generate(
        **inputs,
        max_new_tokens=20,
        num_beams=1,
        early_stopping=True
    )
    gen = tokenizer.decode(
        outs[0][inputs["input_ids"].size(-1):],
        skip_special_tokens=True
    ).strip().split("\n")[0]
    return gen == tail

# %%
def complete_orphans(G, orphan_nodes, node2emb, all_nodes_list, examples_by_rel, tokenizer, model):
    new_edges = []
    for node in tqdm(orphan_nodes, desc="补全孤立节点"):
        if node not in node2emb:
            continue
        emb = node2emb[node]
        sims = [
            (other, util.cos_sim(emb, node2emb[other]).item())
            for other in all_nodes_list if other != node
        ]
        sims.sort(key=lambda x: x[1], reverse=True)

        # --- Debug prints ---
        print(f"\n>>> 处理节点: {node!r}")
        print("Top-K 相似度候选:")
        for cand, sim in sims[:TOP_K]:
            print(f"  {cand!r}: sim={sim:.3f}")

        added = 0
        for candidate, sim in sims[:TOP_K]:
            if sim < SIM_THRESHOLD:
                print(f"  [跳过] {candidate!r} sim={sim:.3f} < {SIM_THRESHOLD}")
                break
            print(f"  尝试: {node!r} —? {candidate!r} （sim={sim:.3f}）")
            for rel, examples in examples_by_rel.items():
                ok = verify_relation(node, rel, candidate, examples, tokenizer, model)
                print(f"    验证关系 {rel!r}: {'✅' if ok else '❌'}")
                if ok:
                    G.add_edge(node, candidate, relation=rel)
                    new_edges.append((node, candidate, rel, sim))
                    print(f"    ➕ 添加边: ({node!r}) -[{rel}]-> ({candidate!r})")
                    added += 1
                    break
            if added >= MAX_LINKS:
                print(f"  已达最大连接数 {MAX_LINKS}，停止继续尝试")
                break
    return new_edges

# %%
def main():
    edges_df, nodes_df, _ = load_data()
    G = build_graph(edges_df)

    orphan_nodes   = get_orphan_nodes(nodes_df)
    all_nodes_list = list(G.nodes())
    node2emb       = compute_embeddings(all_nodes_list)

    examples_by_rel = build_examples_by_relation(edges_df)
    tokenizer, model = load_fewshot_model()

    new_edges = complete_orphans(
        G, orphan_nodes, node2emb, all_nodes_list,
        examples_by_rel, tokenizer, model
    )

    # === Convert to DataFrame and save ===
    new_edges_df = pd.DataFrame(
        new_edges,
        columns=["source", "target", "relation", "similarity"]
    )
    new_edges_df.to_csv("new_edges.csv", index=False, encoding="utf-8-sig")
    print(f"\n✅ 新增补全边数: {len(new_edges)}")
    print("已保存补全结果到 new_edges.csv")

if __name__ == "__main__":
    main()
