"""
作者: Cascade
範例: 將文件使用 LlamaIndex 搭配指定 Embedding Model
      `intfloat/multilingual-e5-large-instruct` 寫入 ChromaDB，並示範查詢。

使用說明：
1. 安裝套件 (建議 python >=3.9)
   pip install llama-index chromadb sentence-transformers accelerate
2. 執行範例
   python chroma_llamaindex_example.py
3. 執行後，程式會將三段示範文字寫入 ChromaDB，之後可在終端機輸入自然語言查詢內容
   (直接按 Enter 結束程式)。

程式原則：
- 所有註解皆使用台灣繁體中文
- 嚴謹但簡潔的 logging 方便追蹤流程
"""

import logging
from pathlib import Path
import os

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import torch

# 設定 logging 格式
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def check_and_download_embedding_model(model_name: str, embedding_dir: Path) -> bool:
    """檢查並下載 embedding 模型到專案目錄，如果需要的話"""
    try:
        logging.info(f"🔍 檢查 embedding 模型: {model_name}")
        logging.info(f"📁 模型將下載到: {embedding_dir}")
        
        # 首先嘗試直接載入模型來檢查是否可用
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # 嘗試創建 embedding 模型實例
            embed_model = HuggingFaceEmbedding(model_name=model_name, device=device)
            logging.info(f"✅ Embedding 模型 {model_name} 已可用")
            return True
            
        except Exception as e:
            logging.warning(f"⚠️ Embedding 模型載入失敗: {e}")
            logging.info(f"🔄 正在下載 embedding 模型到專案目錄: {model_name}")
            
            # 使用 sentence-transformers 手動下載模型到指定目錄
            try:
                from sentence_transformers import SentenceTransformer
                
                # 確保環境變數已設置
                os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(embedding_dir)
                
                # 下載模型到專案目錄
                logging.info(f"📥 開始下載模型到: {embedding_dir}")
                model = SentenceTransformer(model_name, cache_folder=str(embedding_dir))
                logging.info(f"✅ 成功下載 embedding 模型到專案目錄: {model_name}")
                
                # 再次測試 LlamaIndex 的 HuggingFaceEmbedding
                embed_model = HuggingFaceEmbedding(model_name=model_name, device=device)
                logging.info(f"✅ Embedding 模型 {model_name} 現在可以正常使用")
                return True
                
            except Exception as download_error:
                logging.error(f"❌ 下載 embedding 模型失敗: {download_error}")
                return False
                
    except ImportError as e:
        logging.error(f"❌ 缺少必要的套件: {e}")
        logging.info("請運行: pip install sentence-transformers llama-index-embeddings-huggingface")
        return False

def build_index() -> VectorStoreIndex:
    """建立向量索引並將文件寫入 ChromaDB"""

    # 0. 設置模型路徑配置
    base_dir = Path(__file__).parent
    models_dir = base_dir / "models"
    cache_dir = models_dir / "cache"
    embedding_dir = models_dir / "embedding"
    
    # 創建必要目錄
    models_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    embedding_dir.mkdir(exist_ok=True)
    
    # 強制設置 HuggingFace 緩存路徑到專案目錄
    os.environ['HF_HOME'] = str(cache_dir)
    os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
    os.environ['TORCH_HOME'] = str(cache_dir)
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(embedding_dir)
    logging.info("設置模型緩存路徑: %s", cache_dir)
    logging.info("設置 embedding 模型路徑: %s", embedding_dir)

    # 1. 建立 HuggingFace Embedding 模型
    embed_model_name = "intfloat/multilingual-e5-large-instruct"
    logging.info("載入 Embedding 模型: %s", embed_model_name)
    
    # 檢查並下載 embedding 模型
    if not check_and_download_embedding_model(embed_model_name, embedding_dir):
        raise RuntimeError(f"無法載入或下載 embedding 模型: {embed_model_name}")
    
    # 自動偵測 GPU，若無則回退至 CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("使用裝置: %s", device)
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name, device=device)

    # 2. 準備文件資料 (可自行替換為任意文本)
    texts = [
        "台灣是位於東亞的島嶼，首都為臺北市。",
        "高雄市是台灣的海港城市，以熱情的港都聞名。",
        "阿里山的日出與雲海是來台旅遊不可錯過的景點。",
    ]
    documents = [Document(text=t) for t in texts]

    # 3. 建立 ChromaDB Client 與 Vector Store (資料持久化至 ./RAG/chroma_db)
    persist_dir = Path(__file__).parent / "chroma_db"
    persist_dir.mkdir(exist_ok=True)
    
    try:
        # 使用 PersistentClient 而非 HttpClient
        client = chromadb.PersistentClient(path=str(persist_dir))
        logging.info(f"ChromaDB 客戶端初始化成功，使用路徑: {persist_dir}")
    except Exception as e:
        logging.error(f"ChromaDB 客戶端初始化失敗: {e}")
        # 嘗試使用備用方法初始化
        client = chromadb.Client()
        logging.info("使用備用方法初始化 ChromaDB 客戶端")
        
    # 創建或獲取集合
    collection_name = "taiwan_demo"
    try:
        # 先檢查集合是否存在
        try:
            collection = client.get_collection(name=collection_name)
            logging.info(f"獲取現有集合: {collection_name}")
        except Exception:
            # 如果不存在則創建
            collection = client.create_collection(name=collection_name)
            logging.info(f"創建新集合: {collection_name}")
            
        # 使用集合創建 vector_store
        vector_store = ChromaVectorStore(
            chroma_collection=collection,
            collection_name=collection_name
        )
    except Exception as e:
        logging.error(f"創建向量存儲失敗: {e}")
        # 嘗試直接使用 client 創建
        vector_store = ChromaVectorStore(
            chroma_client=client,
            collection_name=collection_name
        )

    # 4. 建立 StorageContext 並將文件寫入
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    logging.info("✅ 文件已成功寫入 ChromaDB (路徑: %s)", persist_dir)
    return index


def query_loop(index: VectorStoreIndex) -> None:
    """互動式查詢迴圈，可多次輸入自然語言查詢"""

    query_engine = index.as_query_engine(similarity_top_k=3)
    print("\n===== 互動式查詢開始，按 Enter 直接結束 =====")
    while True:
        query = input("請輸入查詢內容: ").strip()
        if not query:
            print("查詢結束，再見！")
            break

        logging.info("開始查詢: %s", query)
        response = query_engine.query(query)
        print("\n----- 查詢結果 -----")
        print(response.response)  # LlamaIndex Response 物件
        print("--------------------\n")


if __name__ == "__main__":
    # 若已有持久化資料，可選擇略過重建 (此處為示範直接重建)
    idx = build_index()
    query_loop(idx)
