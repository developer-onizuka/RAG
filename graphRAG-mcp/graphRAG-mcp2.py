import asyncio
import http.server
import os
import socketserver
import threading
import time
import webbrowser
import requests
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

# GraphDB設定
GRAPHDB_BASE_URL = os.getenv("GRAPHDB_BASE_URL", "http://svc-graphdb:7200")
REPOSITORY_ID = os.getenv("GRAPHDB_REPO", "ontology-repo")

indexing_jobs = {}

mcp = FastMCP("Ontology-Driven GraphRAG & Visualizer MCP")


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


def fetch_graphdb_ontology_schema() -> list[str]:
    """GraphDBからオントロジーのクラスやプロパティの一覧をSPARQLで動的取得する"""
    query = """
    SELECT DISTINCT ?class WHERE {
      { ?s a ?class . }
      UNION
      { ?class a <http://www.w3.org/2000/01/rdf-schema#Class> . }
    } LIMIT 50
    """
    url = f"{GRAPHDB_BASE_URL}/repositories/{REPOSITORY_ID}"
    headers = {"Accept": "application/sparql-results+json"}
    
    try:
        response = requests.get(url, params={"query": query}, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            classes = []
            for binding in data.get("results", {}).get("bindings", []):
                class_uri = binding.get("class", {}).get("value", "")
                local_name = class_uri.split("#")[-1].split("/")[-1]
                if local_name and not local_name.startswith("Resource"):
                    classes.append(local_name)
            return list(set(classes))
    except Exception:
        pass
    
    return ["Person", "Organization", "Location", "Concept", "Technology", "Event"]


def enrich_nodes_with_graphdb(G: nx.DiGraph, extracted_entities: set[str]) -> int:
    """【対策】抽出されたテキストノードに対応するGraphDBのURIを検索し、
    別ノードを作る代わりに「ノードの属性（プロパティ）」として付与します。
    """
    if not extracted_entities:
        return 0

    url = f"{GRAPHDB_BASE_URL}/repositories/{REPOSITORY_ID}"
    headers = {"Accept": "application/sparql-results+json"}
    enriched_count = 0

    for entity in extracted_entities:
        if entity not in G.nodes:
            continue

        # エンティティ名に一致するラベルやURIをGraphDBから検索
        sparql_query = f"""
        SELECT ?s ?p ?o WHERE {{
          ?s ?p ?o .
          FILTER(CONTAINS(LCASE(STR(?o)), LCASE("{entity}")) || CONTAINS(LCASE(STR(?s)), LCASE("{entity}")))
        }} LIMIT 5
        """
        try:
            response = requests.get(url, params={"query": sparql_query}, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                bindings = data.get("results", {}).get("bindings", [])
                for b in bindings:
                    s_uri = b.get("s", {}).get("value", "")
                    if s_uri:
                        # ノードの属性としてGraphDBのURIや情報を保持させる
                        G.nodes[entity]["graphdb_uri"] = s_uri
                        G.nodes[entity]["origin"] = "text_enriched_with_graphdb"
                        enriched_count += 1
                        break  # 最初に見つかった代表的なURIを格納
        except Exception:
            pass

    return enriched_count


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
    indexing_jobs[job_id]["status"] = "processing"
    indexing_jobs[job_id]["progress"] = "GraphDBオントロジー取得・テキスト分割準備中..."
    
    try:
        allowed_nodes = await asyncio.to_thread(fetch_graphdb_ontology_schema)

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
            allowed_nodes=allowed_nodes,
        )

        indexing_jobs[job_id]["progress"] = f"全 {len(docs)} チャンクのグラフ抽出処理中 (オントロジー準拠)..."
        graph_docs = await asyncio.to_thread(transformer.convert_to_graph_documents, docs)

        G = load_or_create_graph()
        added_nodes = 0
        added_edges = 0

        extracted_entities = set()
        for graph_doc in graph_docs:
            for node in graph_doc.nodes:
                extracted_entities.add(node.id)
                G.add_node(node.id, type=node.type, domain=domain, origin="text_extraction")
                added_nodes += 1
            for rel in graph_doc.relationships:
                # Pyvisの引数競合を避けるため source ではなく origin を使用
                G.add_edge(rel.source.id, rel.target.id, relation=rel.type, domain=domain, origin="text_extraction")
                added_edges += 1

        # 【対策の適用】GraphDBのURIを別ノードにせず、テキスト抽出ノードの属性（プロパティ）に統合
        indexing_jobs[job_id]["progress"] = "GraphDB上のオントロジー情報をノード属性として統合中..."
        enriched_count = await asyncio.to_thread(enrich_nodes_with_graphdb, G, extracted_entities)

        save_graph_and_parquet(G)

        elapsed = int(time.time() - indexing_jobs[job_id]["start_time"])
        indexing_jobs[job_id]["status"] = "completed"
        indexing_jobs[job_id]["message"] = f"成功: ナレッジグラフ更新完了 (所要時間: {elapsed}秒)。ノード: {added_nodes}, エッジ: {added_edges} (GraphDB属性付与: {enriched_count}件)"
        indexing_jobs[job_id]["file_name"] = file_name

    except asyncio.CancelledError:
        indexing_jobs[job_id]["status"] = "cancelled"
        indexing_jobs[job_id]["error"] = "ユーザーによってジョブがキャンセルされました。"
    except Exception as e:
        indexing_jobs[job_id]["status"] = "failed"
        indexing_jobs[job_id]["error"] = str(e)


@mcp.tool()
async def register_document_and_index(domain: str, content: str) -> str:
    """GraphDBのオントロジーを参照しながらテキストを解析し、ナレッジグラフを構築・統合します"""
    safe_domain = domain.replace("/", "_").replace(" ", "_")
    timestamp = int(time.time())
    file_name = f"{safe_domain}_{timestamp}.txt"
    file_path = os.path.join(INPUT_DIR, file_name)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        job_id = f"job_{timestamp}"
        
        indexing_jobs[job_id] = {
            "status": "processing",
            "progress": "初期化中...",
            "start_time": time.time(),
            "task": None
        }

        task = asyncio.create_task(_background_indexing_task(job_id, domain, content, file_name))
        indexing_jobs[job_id]["task"] = task

        return (
            f"受付完了: ドメイン '{domain}' のテキストを '{file_name}' に保存し、GraphDB連携インデックス作成を開始しました。\n"
            f"ジョブID: '{job_id}'\n"
            f"※ 処理状態は 'check_job_status' ツール、キャンセルは 'cancel_job' ツールで実行できます。"
        )
    except Exception as e:
        return f"エラー: 処理受付中に例外が発生しました: {str(e)}"


@mcp.tool()
async def cancel_job(job_id: str) -> str:
    """実行中のインデックス作成ジョブを強制停止（キャンセル）します"""
    job = indexing_jobs.get(job_id)
    if not job:
        return f"エラー: 指定されたジョブID '{job_id}' が見つかりません。"

    status = job.get("status")
    if status != "processing":
        return f"エラー: このジョブは現在実行中ではありません（現在のステータス: {status}）。"

    task = job.get("task")
    if task and not task.done():
        task.cancel()
        job["status"] = "cancelled"
        job["error"] = "ユーザーによってキャンセルされました。"
        return f"成功: ジョブ '{job_id}' のキャンセルを要求しました。"
    
    return f"エラー: 対象のタスクが見つからないか、すでに終了しています。"


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
    elif status == "cancelled":
        return f"キャンセル済み: {job.get('error', 'ユーザーによりキャンセルされました')}"
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
    """現在のナレッジグラフからインタラクティブなHTML可視化ファイルを生成し、Webブラウザで自動表示します。"""
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
async def get_graph_context(domain: str = "", entity: str = "") -> str:
    """LLMクライアント向けに、ナレッジグラフから関連する関係性（コンテキスト）を抽出してテキストで返します。"""
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

            if target_domain and edge_domain != target_domain:
                continue

            rel = data.get("relation", "関連")
            source_origin = data.get("origin", "text")

            if target_entity:
                if (
                    (target_entity not in str(u))
                    and (target_entity not in str(v))
                    and (target_entity not in str(rel))
                ):
                    continue

            domain_info = f" [{edge_domain}]" if edge_domain else ""
            
            # ノード属性（URI）があればコンテキストにさりげなく付与して表現力を高める
            u_data = G.nodes.get(u, {})
            v_data = G.nodes.get(v, {})
            u_uri = f" (URI: {u_data['graphdb_uri']})" if "graphdb_uri" in u_data else ""
            v_uri = f" (URI: {v_data['graphdb_uri']})" if "graphdb_uri" in v_data else ""

            matched_triples.append(f"・{u}{u_uri} --({rel})--> {v}{v_uri}{domain_info}")

        if not matched_triples:
            cond = []
            if target_domain:
                cond.append(f"ドメイン: '{target_domain}'")
            if target_entity:
                cond.append(f"キーワード: '{target_entity}'")
            cond_str = ", ".join(cond) if cond else "指定条件"
            return f"該当する関係性データ（コンテキスト）が見つかりませんでした ({cond_str})。"

        header = f"【GraphDB属性統合 抽出コンテキスト (該当件数: {len(matched_triples)}件)】\n"
        return header + "\n".join(matched_triples[:300])

    except Exception as e:
        return f"エラー: コンテキスト抽出中に例外が発生しました: {str(e)}"


@mcp.tool()
async def query_knowledge_graph(domain: str = None, prompt: str = "") -> str:
    """【デバッグ・開発者専用 / LLM使用禁止】"""
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


if __name__ == "__main__":
    start_local_web_server()
    mcp.run(transport="sse", host="0.0.0.0", port=5001)
