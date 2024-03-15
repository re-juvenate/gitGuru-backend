import hdbscan
import pandas as pd


def _embed(docs, embed_model):
    embeds = embed_model.embed_documents(docs)
    return embeds


def cluster(texts, embedder):
    vecs = _embed(texts, embedder)
    hdb = hdbscan.HDBSCAN(min_samples=2, min_cluster_size=2, metric="l2").fit(vecs)
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
