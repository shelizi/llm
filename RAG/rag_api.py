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
from llama_index.core import Settings  # 新增

# 全域停用 LLM，避免需要 OpenAI API KEY
Settings.llm = None
import torch
import chromadb
import os

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, VectorStoreIndex

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

# 模型路徑配置
RAG_MODELS_DIR = Path(__file__).parent / "models"
RAG_CACHE_DIR = RAG_MODELS_DIR / "cache"
RAG_EMBEDDING_DIR = RAG_MODELS_DIR / "embedding"

# 確保模型目錄存在
RAG_MODELS_DIR.mkdir(exist_ok=True)
RAG_CACHE_DIR.mkdir(exist_ok=True)
RAG_EMBEDDING_DIR.mkdir(exist_ok=True)

# 強制設置 HuggingFace 緩存路徑到專案目錄
os.environ['HF_HOME'] = str(RAG_CACHE_DIR)
os.environ['TRANSFORMERS_CACHE'] = str(RAG_CACHE_DIR)
os.environ['TORCH_HOME'] = str(RAG_CACHE_DIR)
os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(RAG_EMBEDDING_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ----------------- 載入索引 -----------------
index: VectorStoreIndex | None = None

def check_and_download_embedding_model(model_name: str = EMBED_MODEL_NAME, max_retries: int = 3) -> bool:
    """Checks and downloads the embedding model, retrying on failure."""
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(RAG_EMBEDDING_DIR)
    
    for attempt in range(max_retries):
        try:
            logging.info(f"Attempt {attempt + 1}/{max_retries}: Loading embedding model '{model_name}'.")
            # LlamaIndex's HuggingFaceEmbedding should use the cache path set by the env var.
            embed_model = HuggingFaceEmbedding(model_name=model_name, device=DEVICE, cache_folder=str(RAG_EMBEDDING_DIR))
            logging.info("✅ Embedding model loaded successfully.")
            return True
        except Exception as e:
            logging.warning(f"⚠️ Failed to load model on attempt {attempt + 1}: {e}")
            if "MetadataIncompleteBuffer" in str(e) or "SafetensorsError" in str(e) or "not found" in str(e).lower():
                 logging.info("Corrupted or incomplete model detected. Attempting to re-download.")
                 try:
                    from sentence_transformers import SentenceTransformer
                    # This will download the model to the cache directory
                    SentenceTransformer(model_name, cache_folder=str(RAG_EMBEDDING_DIR))
                    logging.info("✅ Model re-downloaded successfully.")
                    # After downloading, try to load it again in the next loop iteration.
                 except Exception as download_error:
                    logging.error(f"❌ Failed to re-download model: {download_error}")
            
            if attempt < max_retries - 1:
                logging.info("Retrying in 5 seconds...")
                import time
                time.sleep(5)
    
    logging.error(f"❌ Failed to load embedding model after {max_retries} attempts.")
    return False

def load_index() -> VectorStoreIndex:
    """從已存在的 ChromaDB collection 載入索引"""
    # 首先檢查 embedding 模型
    if not check_and_download_embedding_model():
        raise RuntimeError(f"無法載入或下載 embedding 模型: {EMBED_MODEL_NAME}")
    
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        raise FileNotFoundError(f"ChromaDB 目錄 {CHROMA_DIR} 不存在或為空，請先建置索引！")

    try:
        # 使用 PersistentClient 而非 HttpClient
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        logging.info(f"ChromaDB 客戶端初始化成功，使用路徑: {CHROMA_DIR}")
    except Exception as e:
        logging.error(f"ChromaDB 客戶端初始化失敗: {e}")
        # 嘗試使用備用方法初始化
        client = chromadb.Client()
        logging.info("使用備用方法初始化 ChromaDB 客戶端")
        
    # 創建或獲取集合
    try:
        # 先檢查集合是否存在
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
            logging.info(f"獲取現有集合: {COLLECTION_NAME}")
        except Exception:
            # 如果不存在則創建
            collection = client.create_collection(name=COLLECTION_NAME)
            logging.info(f"創建新集合: {COLLECTION_NAME}")
            
        # 使用集合創建 vector_store
        vector_store = ChromaVectorStore(
            chroma_collection=collection,
            collection_name=COLLECTION_NAME
        )
    except Exception as e:
        logging.error(f"創建向量存儲失敗: {e}")
        # 嘗試直接使用 client 創建
        vector_store = ChromaVectorStore(
            chroma_client=client,
            collection_name=COLLECTION_NAME
        )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME, device=DEVICE, cache_folder=str(RAG_EMBEDDING_DIR))

    # 從 vector_store 建立 VectorStoreIndex
    idx = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    return idx

@app.on_event("startup")
def _startup():
    global index
    try:
        logging.info("正在嘗試載入索引...")
        index = load_index()
        logging.info("✅ Index 載入完成，裝置: %s", DEVICE)
    except Exception as e:
        logging.warning("⚠️ 無法載入索引 (這是正常的，如果還沒建立索引): %s", e)
        index = None

# ----------------- API 端點 -----------------
@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    global index
    if index is None:
        # 嘗試即時載入索引，以避免使用者已建好索引但尚未重啟服務
        try:
            logging.info("⚠️ 查詢時 index 為 None，嘗試重新載入索引 ...")
            index = load_index()
            logging.info("✅ 重新載入索引成功！")
        except Exception as e:
            logging.warning("❌ 重新載入索引失敗: %s", e)
            raise HTTPException(status_code=503, detail="Index 尚未就緒，請稍後或先建置索引")

    # 改用純向量檢索，避免需要 LLM/OpenAI API KEY
    retriever = index.as_retriever(similarity_top_k=req.top_k)
    nodes = retriever.retrieve(req.query)

    if not nodes:
        raise HTTPException(status_code=404, detail="查無相關內容")

    # 以最高分節點內容做為簡易回答
    answer_text = nodes[0].node.text.strip()

    sources: list[SourceItem] = []
    for n in nodes:
        sources.append(
            SourceItem(
                filename=n.metadata.get("filename", ""),
                snippet=n.node.text[:120].strip().replace("\n", " "),
                score=round(float(n.score or 0), 4),
            )
        )

    return QueryResponse(answer=answer_text, sources=sources)
