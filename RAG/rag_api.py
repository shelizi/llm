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
  "top_k": 5,         # 選填，預設 5，範圍 1-20
  "filenames": []     # 選填，指定要查詢的檔案名稱列表，空列表表示查詢所有檔案
}

Response:
{
  "answer": "...",
  "sources": [...]
}

GET /indexed_files
Response:
{
  "files": ["file1.txt", "file2.pdf", ...]
}
"""

from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging
from typing import List
from huggingface_hub import HfApi
from llama_index.core import Settings  # 新增

# 全域停用 LLM，避免需要 OpenAI API KEY
Settings.llm = None
import torch
import chromadb
import os
import asyncio

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, VectorStoreIndex

# ----------------- 基本設定 -----------------
app = FastAPI(title="RAG API with LlamaIndex & Chroma")

class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20, description="返回的相似文檔數量，範圍1-20")
    filenames: list[str] = Field(default=[], description="指定要查詢的檔案名稱列表，空列表表示查詢所有檔案")

class SourceItem(BaseModel):
    filename: str
    snippet: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = []
    enhanced_answer: str = ""  # AI增強回答
    answer_source: str = "retrieval"  # "retrieval" 或 "ai_enhanced"

# 環境變數 / 預設值
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "taiwan_demo"  # 與 build_index 中保持一致
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"

# 預先檢查/下載的嵌入模型清單 (可於其他模組覆寫或傳入自定義列表)
DEFAULT_EMBEDDING_MODELS: List[str] = [
    EMBED_MODEL_NAME,
    "nomic-ai/nomic-embed-text-v2",
    "jinaai/jina-embeddings-v2-base-zh",
    "Linq-AI-Research/Linq-Embed-Mistral",
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 模型路徑配置
# 檢查是否在 Docker 容器內 (透過掛載的 /models 目錄)
if Path("/models").exists() and Path("/models").is_dir():
    # Docker 容器內，使用掛載的路徑
    RAG_MODELS_DIR = Path("/models")
    RAG_CACHE_DIR = RAG_MODELS_DIR / "cache"
    RAG_EMBEDDING_DIR = RAG_MODELS_DIR / "embedding"
else:
    # 本地開發環境，使用相對路徑
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

def suggest_similar_models(query: str, limit: int = 5) -> List[str]:
    """Search Hugging Face Hub for models with names similar to the query."""
    try:
        api = HfApi()
        # Search both by repo id and tags in full-text mode
        results = api.search_models(full_text_search=query, limit=limit)
        return [item.modelId for item in results]
    except Exception as e:
        logging.debug(f"🔍 Hugging Face search failed: {e}")
        return []


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

    # 額外步驟: 嘗試在 Hugging Face Hub 搜尋相似模型名稱並提供建議
    similar = suggest_similar_models(model_name)
    if similar:
        logging.info("🔍 找到可能的相似模型名稱，請確認是否拼寫錯誤或選擇下列其中之一：")
        for s in similar:
            logging.info(f"  • {s}")
    else:
        logging.info("🔍 未在 Hugging Face Hub 找到相似模型名稱，請再次確認輸入是否正確。")

    return False


def check_and_download_embedding_models(model_names: List[str] | None = None) -> tuple[list[str], list[str]]:
    """Bulk check/download for a list of embedding models.

    Returns (ok_models, failed_models) lists.
    """
    if model_names is None:
        model_names = DEFAULT_EMBEDDING_MODELS

    ok_models: list[str] = []
    failed_models: list[str] = []
    for name in model_names:
        logging.info(f"🔎 檢查模型 {name} ...")
        if check_and_download_embedding_model(name):
            ok_models.append(name)
        else:
            failed_models.append(name)

    return ok_models, failed_models

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
async def query_endpoint(req: QueryRequest):
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
    # 如果指定了文件名稱，則使用元數據過濾
    if req.filenames:
        from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
        
        # 創建文件名過濾器
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="filename",
                    value=req.filenames,
                    operator=FilterOperator.IN
                )
            ]
        )
        retriever = index.as_retriever(similarity_top_k=req.top_k, filters=filters)
    else:
        retriever = index.as_retriever(similarity_top_k=req.top_k)
    
    nodes = retriever.retrieve(req.query)

    if not nodes:
        raise HTTPException(status_code=404, detail="查無相關內容")

    # 以最高分節點內容做為簡易回答
    answer_text = nodes[0].node.text.strip()

    sources: list[SourceItem] = []
    context_text = ""
    for n in nodes:
        sources.append(
            SourceItem(
                filename=n.metadata.get("filename", ""),
                snippet=n.node.text[:120].strip().replace("\n", " "),
                score=round(float(n.score or 0), 4),
            )
        )
        # 收集上下文用於AI增強回答
        context_text += f"\n文檔：{n.metadata.get('filename', '未知')}\n內容：{n.node.text}\n"

    # 嘗試使用AI增強回答
    enhanced_answer = ""
    answer_source = "retrieval"
    
    try:
        from openai_client import openai_client
        ai_result = await openai_client.generate_answer(req.query, context_text)
        
        if ai_result["success"]:
            enhanced_answer = ai_result["answer"]
            answer_source = "ai_enhanced"
            logging.info("✅ AI增強回答生成成功")
        else:
            logging.warning(f"AI增強回答失敗: {ai_result.get('error', '未知錯誤')}")
    except Exception as e:
        logging.warning(f"AI增強回答異常: {e}")

    return QueryResponse(
        answer=answer_text, 
        sources=sources,
        enhanced_answer=enhanced_answer,
        answer_source=answer_source
    )

@app.get("/indexed_files")
def get_indexed_files():
    """獲取已索引的文件列表"""
    global index
    if index is None:
        return {"files": []}
    
    try:
        # 從向量存儲中獲取所有文檔的元數據
        vector_store = index.vector_store
        if hasattr(vector_store, '_collection'):
            # 對於 ChromaVectorStore，直接查詢集合
            collection = vector_store._collection
            result = collection.get(include=['metadatas'])
            
            # 提取唯一的文件名
            filenames = set()
            if result and 'metadatas' in result:
                for metadata in result['metadatas']:
                    if metadata and 'filename' in metadata:
                        filenames.add(metadata['filename'])
            
            return {"files": sorted(list(filenames))}
        else:
            # 備用方法：通過查詢獲取
            retriever = index.as_retriever(similarity_top_k=100)  # 獲取較多結果
            nodes = retriever.retrieve("*")  # 使用通配符查詢
            
            filenames = set()
            for node in nodes:
                if hasattr(node, 'metadata') and 'filename' in node.metadata:
                    filenames.add(node.metadata['filename'])
            
            return {"files": sorted(list(filenames))}
    except Exception as e:
        logging.error(f"獲取已索引文件列表失敗: {e}")
        return {"files": []}

@app.get("/health")
def health_check():
    """健康檢查端點，返回API狀態和配置信息"""
    try:
        from config_manager import ConfigManager
        config_manager = ConfigManager()
        api_status = config_manager.get_api_status()
    except Exception:
        api_status = {"configured": False}
        
    return {
        "status": "healthy",
        "index_loaded": index is not None,
        "device": DEVICE,
        "embed_model": EMBED_MODEL_NAME,
        "chroma_dir": str(CHROMA_DIR),
        "collection_name": COLLECTION_NAME,
        "default_top_k": 5,
        "max_top_k": 20,
        "api_configured": api_status.get("configured", False)
    }
