# ----------------------------------------------------------------------------
# RAG 執行環境 Dockerfile
# ----------------------------------------------------------------------------
# 功能：
# 1. 以含 CUDA 的官方 PyTorch 執行環境為基底，支援 GPU 加速。
# 2. 安裝 LlamaIndex + ChromaDB 相關依賴。
# 3. 預設將 Hugging Face 與 Torch 的模型快取路徑指向 /models，
#    方便在啟動 container 時透過 volume(-v) 掛載現有模型快取做共用。
# ----------------------------------------------------------------------------

# 以官方 PyTorch CUDA Runtime 為基底 (Ubuntu 22.04 / CUDA 12.1 / CuDNN 8)
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# 設定工作目錄
WORKDIR /app

# 建立 Python 虛擬環境 (選用) - 此處直接使用系統 Python

# ---------- 安裝 Python 相依套件 ----------
# 先複製 requirements.txt 以利用 Docker 快取層
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# ---------- 環境變數 ----------
# 將模型快取路徑指向 /models/cache，啟動 container 時以 -v <host_path>:/models 掛載即可共用
ENV HF_HOME=/models/cache \
    TRANSFORMERS_CACHE=/models/cache \
    TORCH_HOME=/models/cache

# 將專案程式碼複製進容器
COPY . /app

# 預設執行互動式範例 (可改成自身應用之 entrypoint)
EXPOSE 8000
# 預設啟動 FastAPI RAG 服務
CMD ["uvicorn", "rag_api:app", "--host", "0.0.0.0", "--port", "8000"]
