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

import chromadb
from llama_index import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import torch

# 設定 logging 格式
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def build_index() -> VectorStoreIndex:
    """建立向量索引並將文件寫入 ChromaDB"""

    # 1. 建立 HuggingFace Embedding 模型
    embed_model_name = "intfloat/multilingual-e5-large-instruct"
    logging.info("載入 Embedding 模型: %s", embed_model_name)
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
    client = chromadb.PersistentClient(path=str(persist_dir))
    vector_store = ChromaVectorStore(chroma_client=client, collection_name="taiwan_demo")

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
