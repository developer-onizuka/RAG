import re
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

    def clear(self):
        self.vectors = []
        self.metadata = []

# 多言語対応の軽量埋め込みモデル
model = SentenceTransformer("intfloat/multilingual-e5-small")

# FastMCP インスタンスの生成
mcp = FastMCP("SimpleVectorDB")
db = SimpleVectorDB()

@mcp.tool()
def add_document(
    text: str, 
    category: str = "general", 
    chunk_size: int = 200, 
    overlap: int = 50
) -> str:
    """テキストを固定文字数（デフォルト200文字、重複50文字）で分割してVectorDBに登録します。"""
    # 改行コードの正規化と不要なエスケープの除去
    normalized_text = text.replace('\\n', '\n').replace('\r\n', '\n').strip()

    chunks = []
    start = 0
    text_length = len(normalized_text)

    while start < text_length:
        end = start + chunk_size
        chunk = normalized_text[start:end].strip()
        
        if len(chunk) > 10:  # 極端に短い末端データは除外
            chunks.append(chunk)
            
        # 次のチャンクの開始位置（オーバーラップ分だけ戻す）
        start += (chunk_size - overlap)
        if chunk_size <= overlap: # 無限ループ防止
            break

    added_count = 0
    for chunk in chunks:
        embedding = model.encode(f"passage: {chunk}").tolist()
        metadata = {"text": chunk, "category": category}
        db.add(embedding, metadata)
        added_count += 1

    return f"登録完了: {added_count} 個のチャンク（サイズ:{chunk_size}/重複:{overlap}）に分割登録しました。(現在の総データ数: {len(db.vectors)} 件)"

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

@mcp.tool()
def reset_db() -> str:
    """VectorDBに登録されているすべてのデータを消去します。"""
    db.clear()
    return "データベースをリセットしました。"

if __name__ == "__main__":
    # SSE (Server-Sent Events) モードで起動
    mcp.run(transport="sse", host="0.0.0.0", port=5001)
