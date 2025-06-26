"""
RAG API - 以 FastAPI 建立 `/query` 端點，供 OpenWebUI 或其他服務呼叫。
啟動後將載入（或嘗試載入）既有的 ChromaDB 向量索引。
若未找到索引將回傳 503 錯誤，請先執行相同目錄下的 `chroma_llamaindex_example.py`
或自行撰寫腳本建置索引。

### 主要端點
POST /query
Body:
{
  "query": "請問阿里山在哪裡？",
  "top_k": 3          # 選填，預設 3
}

Response:
{
  "answer": "..."
}
"""

from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import torch
import chromadb

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index import StorageContext, VectorStoreIndex

# ----------------- 基本設定 -----------------
app = FastAPI(title="RAG API with LlamaIndex & Chroma")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class SourceItem(BaseModel):
    filename: str
    snippet: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = []

# 環境變數 / 預設值
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "taiwan_demo"  # 與 build_index 中保持一致
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ----------------- 載入索引 -----------------
index: VectorStoreIndex | None = None

def load_index() -> VectorStoreIndex:
    """從已存在的 ChromaDB collection 載入索引"""
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        raise FileNotFoundError(f"ChromaDB 目錄 {CHROMA_DIR} 不存在或為空，請先建置索引！")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    vector_store = ChromaVectorStore(chroma_client=client, collection_name=COLLECTION_NAME)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME, device=DEVICE)

    # 從 vector_store 建立 VectorStoreIndex
    idx = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    return idx

@app.on_event("startup")
def _startup():
    global index
    try:
        index = load_index()
        logging.info("✅ Index 載入完成，裝置: %s", DEVICE)
    except Exception as e:
        logging.error("❌ 無法載入索引: %s", e)
        index = None

# ----------------- API 端點 -----------------
@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    if index is None:
        raise HTTPException(status_code=503, detail="Index 尚未就緒，請稍後或先建置索引")

    query_engine = index.as_query_engine(similarity_top_k=req.top_k)
    resp = query_engine.query(req.query)

    sources: list[SourceItem] = []
    for n in resp.source_nodes:
        sources.append(
            SourceItem(
                filename=n.metadata.get("filename", ""),
                snippet=n.node.text[:120].strip().replace("\n", " "),
                score=round(float(n.score or 0), 4),
            )
        )
    return QueryResponse(answer=resp.response, sources=sources)
