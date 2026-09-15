import numpy as np
from sentence_transformers import SentenceTransformer
from fastmcp import FastMCP

class SimpleVectorDB:
    def __init__(self):
        self.vectors = []
        self.metadata = []

    def add(self, vector: list[float], meta: dict):
        vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self.vectors.append(vec)
        self.metadata.append(meta)

    def search(self, query_vector: list[float], top_k: int = 3):
        if not self.vectors:
            return []
        q_vec = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        scores = np.dot(np.array(self.vectors), q_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(float(scores[i]), self.metadata[i]) for i in top_indices]

# 多言語対応の軽量埋め込みモデル
model = SentenceTransformer("intfloat/multilingual-e5-small")

# FastMCP インスタンスの生成
mcp = FastMCP("SimpleVectorDB")
db = SimpleVectorDB()

@mcp.tool()
def add_document(text: str, category: str = "general") -> str:
    """テキストをベクトル化してVectorDBに登録します。"""
    embedding = model.encode(f"passage: {text}").tolist()
    metadata = {"text": text, "category": category}
    db.add(embedding, metadata)
    return f"登録完了 (現在の総データ数: {len(db.vectors)} 件)"

@mcp.tool()
def search_similar(query: str, top_k: int = 3) -> str:
    """自然言語クエリでVectorDBから類似度の高いテキストを検索します。"""
    query_embedding = model.encode(f"query: {query}").tolist()
    results = db.search(query_embedding, top_k=top_k)

    if not results:
        return "データベースに該当するデータがありません。"

    output = []
    for rank, (score, meta) in enumerate(results, 1):
        output.append(
            f"[{rank}] 類似度スコア: {score:.4f}\n"
            f"カテゴリ: {meta.get('category')}\n"
            f"本文: {meta['text']}"
        )

    return "\n\n".join(output)

if __name__ == "__main__":
    # SSE (Server-Sent Events) モードで起動
    mcp.run(transport="sse", host="0.0.0.0", port=5001)
