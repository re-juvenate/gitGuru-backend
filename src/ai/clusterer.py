import hdbscan
import pandas as pd
from langchain_community.vectorstores import FAISS


def embed(docs, embed_model):
    embeds = embed_model.embed_documents(docs)
    return embeds


def cluster(texts, embedder, min_s=2, max_s=1000):
    vecs = embed(texts, embedder)
    hdb = hdbscan.HDBSCAN(
        min_samples=min_s, min_cluster_size=min_s, max_cluster_size=max_s, metric="l1"
    ).fit(vecs)
    df = pd.DataFrame(
        {
            "text": [text for text in texts],
            "cluster": hdb.labels_,
        }
    )
    len(df)
    df = df.query("cluster != -1")

    cluster_texts = []
    for c in df.cluster.unique():
        c_str = "\n".join(
            [
                f"{row['text']}\n"
                for row in df.query(f"cluster == {c}").to_dict(orient="records")
            ]
        )
        cluster_texts.append(c_str)
    return cluster_texts
