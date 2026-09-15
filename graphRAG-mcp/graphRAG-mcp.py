import asyncio
import http.server
import os
import socketserver
import threading
import time
import webbrowser
from fastmcp import FastMCP
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import networkx as nx
import pandas as pd
from pyvis.network import Network

ROOT_DIR = "./data"
INPUT_DIR = f"{ROOT_DIR}/input"
OUTPUT_DIR = f"{ROOT_DIR}/output"
GRAPH_FILE = f"{OUTPUT_DIR}/knowledge_graph.graphml"
HTML_FILE = f"{OUTPUT_DIR}/graph.html"
WEB_PORT = 8080

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://svc-ollama:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")

indexing_jobs = {}

mcp = FastMCP("GraphRAG Indexer & Visualizer MCP (LangChain + Ollama)")


def start_local_web_server():
    """outputディレクトリをルートとした簡易Webサーバーをバックグラウンドで起動"""

    class Handler(http.server.SimpleHTTPRequestHandler):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=OUTPUT_DIR, **kwargs)

    def run_server():
        socketserver.TCPServer.allow_reuse_address = True
        with socketserver.TCPServer(("", WEB_PORT), Handler) as httpd:
            httpd.serve_forever()

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()


def load_or_create_graph() -> nx.DiGraph:
    if os.path.exists(GRAPH_FILE):
        try:
            return nx.read_graphml(GRAPH_FILE)
        except Exception:
            return nx.DiGraph()
    return nx.DiGraph()


def save_graph_and_parquet(G: nx.DiGraph):
    nx.write_graphml(G, GRAPH_FILE)
    nodes_data = [{"id": n, **data} for n, data in G.nodes(data=True)]
    pd.DataFrame(nodes_data).to_parquet(f"{OUTPUT_DIR}/nodes.parquet", index=False)
    edges_data = [{"source": u, "target": v, **data} for u, v, data in G.edges(data=True)]
    pd.DataFrame(edges_data).to_parquet(f"{OUTPUT_DIR}/edges.parquet", index=False)


async def _background_indexing_task(job_id: str, domain: str, content: str, file_name: str):
    indexing_jobs[job_id] = {
        "status": "processing",
        "progress": "テキスト分割・準備中...",
        "start_time": time.time(),
    }
    try:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key="ollama",
            base_url=OLLAMA_BASE_URL,
            temperature=0,
            request_timeout=1800.0,
        )

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        docs = text_splitter.create_documents([content], metadatas=[{"domain": domain}])

        transformer = LLMGraphTransformer(
            llm=llm,
            allowed_nodes=["Person", "Organization", "Location", "Concept", "Technology", "Event"],
        )

        indexing_jobs[job_id]["progress"] = f"全 {len(docs)} チャンクのグラフ抽出処理中..."
        graph_docs = await asyncio.to_thread(transformer.convert_to_graph_documents, docs)

        G = load_or_create_graph()
        added_nodes = 0
        added_edges = 0

        for graph_doc in graph_docs:
            for node in graph_doc.nodes:
                G.add_node(node.id, type=node.type, domain=domain)
                added_nodes += 1
            for rel in graph_doc.relationships:
                G.add_edge(rel.source.id, rel.target.id, relation=rel.type, domain=domain)
                added_edges += 1

        save_graph_and_parquet(G)

        elapsed = int(time.time() - indexing_jobs[job_id]["start_time"])
        indexing_jobs[job_id] = {
            "status": "completed",
            "message": f"成功: ナレッジグラフ更新完了 (所要時間: {elapsed}秒)。追加ノード: {added_nodes}, 追加エッジ: {added_edges}",
            "file_name": file_name,
        }
    except Exception as e:
        indexing_jobs[job_id] = {"status": "failed", "error": str(e)}


@mcp.tool()
async def register_document_and_index(domain: str, content: str) -> str:
    """即座に受領レスポンスを返し、非同期バックグラウンドでナレッジグラフを構築します"""
    safe_domain = domain.replace("/", "_").replace(" ", "_")
    timestamp = int(time.time())
    file_name = f"{safe_domain}_{timestamp}.txt"
    file_path = os.path.join(INPUT_DIR, file_name)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        job_id = f"job_{timestamp}"
        asyncio.create_task(_background_indexing_task(job_id, domain, content, file_name))

        return (
            f"受付完了: ドメイン '{domain}' のテキストを '{file_name}' に保存し、非同期でインデックス作成を開始しました。\n"
            f"ジョブID: '{job_id}'\n"
            f"※ 処理状態は 'check_job_status' ツールで確認できます。"
        )
    except Exception as e:
        return f"エラー: 処理受付中に例外が発生しました: {str(e)}"


@mcp.tool()
async def check_job_status(job_id: str) -> str:
    """バックグラウンドで実行中のインデックス作成ジョブのステータスを確認します"""
    job = indexing_jobs.get(job_id)
    if not job:
        return f"エラー: 指定されたジョブID '{job_id}' が見つかりません。"

    status = job.get("status")
    if status == "processing":
        elapsed = int(time.time() - job["start_time"])
        return f"実行中: {job['progress']} (経過時間: {elapsed}秒)"
    elif status == "completed":
        return f"完了: {job['message']}"
    elif status == "failed":
        return f"失敗: {job['error']}"
    return "不明なステータスです。"


@mcp.tool()
async def list_jobs() -> str:
    """登録されているすべてのインデックス作成ジョブ一覧とJobIDを確認します"""
    if not indexing_jobs:
        return "現在管理されているジョブはありません。"

    lines = ["=== ジョブ一覧 ==="]
    for j_id, info in indexing_jobs.items():
        status = info.get("status", "不明")
        lines.append(f"・JobID: {j_id} | ステータス: {status}")
    return "\n".join(lines)


@mcp.tool()
async def generate_visualization_html(auto_open: bool = True) -> str:
    """現在のナレッジグラフからインタラクティブなHTML可視化ファイルを生成し、Webブラウザで自動表示します。

    Args:
        auto_open: Trueの場合、生成後にPCの既定ブラウザで自動的に可視化画面を開きます。
    """
    if not os.path.exists(GRAPH_FILE):
        return f"エラー: ナレッジグラフファイル '{GRAPH_FILE}' が見つかりません。先にドメイン登録を実行してください。"

    try:

        def _build_html():
            G = nx.read_graphml(GRAPH_FILE)
            net = Network(height="850px", width="100%", notebook=False, directed=True)
            net.from_nx(G)
            net.toggle_physics(True)
            net.write_html(HTML_FILE)
            return G.number_of_nodes(), G.number_of_edges()

        num_nodes, num_edges = await asyncio.to_thread(_build_html)

        url = f"http://localhost:{WEB_PORT}/graph.html"
        if auto_open:
            webbrowser.open(url)

        return (
            f"成功: 可視化HTMLファイルを生成し、Webサーバー経由で公開しました。\n"
            f"・アクセスURL: {url}\n"
            f"・ノード数: {num_nodes}, エッジ数: {num_edges}\n"
            f"※ 自動でブラウザが開かない場合は上記URLを開いてください。"
        )
    except Exception as e:
        return f"エラー: 可視化HTMLの生成中に例外が発生しました: {str(e)}"


@mcp.tool()
async def query_knowledge_graph(domain: str = None, prompt: str = "") -> str:
    """【デバッグ・開発者専用 / LLM使用禁止】
    このツールはMCPサーバー内部の検証用です。LLMクライアント（Claude等）はユーザーへの回答作成にこのツールを使用してはいけません。
    関係性データの取得には必ず 'get_graph_context' ツールを使用してください。

    Args:
        domain: 検索対象のドメイン。省略時は全ドメイン。
        prompt: 開発検証用のプロンプト
    """
    if not os.path.exists(GRAPH_FILE):
        return f"エラー: ナレッジグラフファイル '{GRAPH_FILE}' が見つかりません。"

    try:
        G = nx.read_graphml(GRAPH_FILE)

        if G.number_of_nodes() == 0:
            return "エラー: ナレッジグラフにノードが存在しません。"

        triples = []
        for u, v, data in G.edges(data=True):
            edge_domain = data.get("domain", "")

            if domain and edge_domain != domain:
                continue

            rel = data.get("relation", "関連")
            domain_info = f" [{edge_domain}]" if edge_domain else ""
            triples.append(f"・{u} --({rel})--> {v}{domain_info}")

        if not triples:
            target = f"ドメイン '{domain}'" if domain else "グラフ全体"
            return f"{target} に該当する関係性データが見つかりませんでした。"

        context_text = "\n".join(triples[:300])

        llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key="ollama",
            base_url=OLLAMA_BASE_URL,
            temperature=0.2,
            request_timeout=120.0,
        )

        domain_instruction = f"「{domain}」ドメインの" if domain else "全ドメインの"
        system_prompt = (
            f"あなたは与えられたナレッジグラフ（{domain_instruction}データ）を解釈するアシスタントです。\n"
            "以下の【ナレッジグラフデータ】に記録されている関係性のみを根拠として、質問に回答してください。\n"
            "グラフに含まれない情報については「該当する関係性が記述されていません」と明記してください。\n\n"
            f"【ナレッジグラフデータ】\n{context_text}"
        )

        messages = [
            ("system", system_prompt),
            ("user", prompt),
        ]

        response = await asyncio.to_thread(llm.invoke, messages)
        return response.content

    except Exception as e:
        return f"エラー: 質問処理中に例外が発生しました: {str(e)}"


@mcp.tool()
async def get_graph_context(domain: str = "", entity: str = "") -> str:
    """Claude DesktopなどのLLMクライアント向けに、ナレッジグラフから関連する関係性（コンテキスト）を抽出してテキストで返します。

    Args:
        domain: 検索対象のドメイン（例: "dragonball"）。空文字または "all" で全体。
        entity: 絞り込みたいエンティティ名やキーワード（例: "悟空"）。空文字で全関係性を出力。
    """
    if not os.path.exists(GRAPH_FILE):
        return f"エラー: ナレッジグラフファイル '{GRAPH_FILE}' が見つかりません。"

    try:
        G = nx.read_graphml(GRAPH_FILE)
        if G.number_of_nodes() == 0:
            return "エラー: ナレッジグラフにノードが存在しません。"

        target_domain = None if domain.strip().lower() in ["", "all", "none", "*"] else domain.strip()
        target_entity = entity.strip()

        matched_triples = []

        for u, v, data in G.edges(data=True):
            edge_domain = data.get("domain", "")

            # 1. ドメインフィルタリング
            if target_domain and edge_domain != target_domain:
                continue

            rel = data.get("relation", "関連")

            # 2. エンティティ/キーワードフィルタリング (送信元・送信先・関係性のいずれかに含まれるか)
            if target_entity:
                if (
                    (target_entity not in str(u))
                    and (target_entity not in str(v))
                    and (target_entity not in str(rel))
                ):
                    continue

            domain_info = f" [{edge_domain}]" if edge_domain else ""
            matched_triples.append(f"・{u} --({rel})--> {v}{domain_info}")

        if not matched_triples:
            cond = []
            if target_domain:
                cond.append(f"ドメイン: '{target_domain}'")
            if target_entity:
                cond.append(f"キーワード: '{target_entity}'")
            cond_str = ", ".join(cond) if cond else "指定条件"
            return f"該当する関係性データ（コンテキスト）が見つかりませんでした ({cond_str})。"

        # 抽出結果をテキスト構造化して返す（これをClaude Desktopが読んで回答を生成する）
        header = f"【ナレッジグラフ抽出コンテキスト (該当件数: {len(matched_triples)}件)】\n"
        return header + "\n".join(matched_triples[:300])

    except Exception as e:
        return f"エラー: コンテキスト抽出中に例外が発生しました: {str(e)}"


if __name__ == "__main__":
    start_local_web_server()
    mcp.run(transport="sse", host="0.0.0.0", port=5001)
